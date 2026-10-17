"""Anticipatory Prior, Location-aware BBox CoT and Latent Patch-to-Token Alignment."""
import torch
from torch import nn
from torch.nn import functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModel
from qwen_vl_utils import process_vision_info

from .data import sam_image
from .losses import mask_losses
from .prompts import segmentation_prompt


class PTAHead(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden):
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class GeoReason(nn.Module):
    def __init__(self, config, tokenizer, pretrained=None):
        super().__init__()
        self.settings = config
        if pretrained is None:
            pretrained = AutoModel.from_pretrained(
                config["pretrained"], revision=config["revision"], trust_remote_code=True,
                torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        self.mllm = pretrained.model
        self.segmentor = pretrained.grounding_encoder.sam2_model.float()
        self.mask_projector = pretrained.text_hidden_fcs.float()
        self.seg_id = tokenizer.convert_tokens_to_ids("[SEG]")
        self.segmentor.requires_grad_(False)
        self.segmentor.sam_mask_decoder.float().requires_grad_(True)
        self.mllm.config.tie_word_embeddings = False
        self.mllm.config.text_config.tie_word_embeddings = False
        if self.mllm.get_input_embeddings().weight is self.mllm.get_output_embeddings().weight:
            self.mllm.lm_head.weight = nn.Parameter(self.mllm.get_input_embeddings().weight.detach().clone())
        targets = [name for name, layer in self.mllm.named_modules()
                   if "language_model." in name and isinstance(layer, nn.Linear)]
        self.mllm = prepare_model_for_kbit_training(self.mllm, use_gradient_checkpointing=True,
                                                   gradient_checkpointing_kwargs={"use_reentrant": False})
        self.mllm = get_peft_model(self.mllm, LoraConfig(
            r=config["lora_rank"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
            target_modules=targets, modules_to_save=["embed_tokens", "lm_head"],
            task_type="CAUSAL_LM", bias="none"))
        text_config = self.mllm.config.text_config
        self.pta_head = PTAHead(text_config.hidden_size, text_config.intermediate_size) if config["pta"] else None

    def encode_masks(self, images):
        # High-resolution projections belong to the trainable SAM mask decoder.
        features = self.segmentor.forward_image(images)
        _, features, _, sizes = self.segmentor._prepare_backbone_features(features)
        features[-1] = features[-1] + self.segmentor.no_mem_embed
        return [x.permute(1, 2, 0).reshape(x.shape[1], x.shape[2], *size)
                for x, size in zip(features, sizes)]

    def decode_masks(self, features, embeddings, sample_indices):
        n = len(embeddings)
        device = embeddings.device
        sparse, dense = self.segmentor.sam_prompt_encoder(
            points=(torch.zeros(n, 1, 2, device=device), -torch.ones(n, 1, dtype=torch.int32, device=device)),
            boxes=None, masks=None)
        sparse = torch.cat([sparse, embeddings[:, None]], dim=1)
        masks, ious, _, _ = self.segmentor.sam_mask_decoder(
            image_embeddings=features[-1][sample_indices],
            image_pe=self.segmentor.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=True, repeat_image=False,
            high_res_features=[x[sample_indices] for x in features[:-1]])
        return masks[torch.arange(n, device=device), ious.argmax(1)].float()

    def forward(self, input_ids, attention_mask, labels, pixel_values, image_grid_thw,
                sam_pixels, masks, patches):
        output = self.mllm(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                           pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                           output_hidden_states=True, use_cache=False)
        hidden = output.hidden_states[-1]
        active = [i for i, mask in enumerate(masks) if mask is not None and (input_ids[i] == self.seg_id).any()]
        losses = {"lm": output.loss}
        if not active:
            return losses
        queries = [hidden[i, input_ids[i] == self.seg_id] for i in active]
        # Five mask slots per image, cycling through prompt-side and answer-side queries.
        mask_queries = torch.cat([q[torch.arange(5, device=q.device) % len(q)] for q in queries])
        features = self.encode_masks(torch.stack([sam_pixels[i] for i in active]))
        indices = torch.arange(len(active), device=hidden.device).repeat_interleave(5)
        logits = self.decode_masks(features, self.mask_projector(mask_queries), indices)
        targets = torch.cat([F.interpolate(masks[i][None, None].float(), logits.shape[-2:], mode="nearest")
                             .expand(5, -1, -1, -1)[:, 0] for i in active])
        losses["bce"], losses["dice"] = mask_losses(logits, targets)
        if self.pta_head is not None:
            with torch.no_grad():
                visual, _ = self.mllm.get_base_model().get_image_features(pixel_values, image_grid_thw)
            predicted, target = [], []
            for i, q in zip(active, queries):
                if len(patches[i]):
                    predicted.append(self.pta_head(q))
                    pooled = visual[i][patches[i]].mean(0).detach()
                    target.append(pooled[None].expand(len(q), -1))
            if predicted:
                losses["pta"] = 1 - F.cosine_similarity(torch.cat(predicted).float(), torch.cat(target).float()).mean()
        return losses

    @torch.inference_mode()
    def predict(self, image, question, task, processor, max_new_tokens=2048):
        self.eval()
        device = next(self.parameters()).device
        text = segmentation_prompt(question, task, self.settings["ap"], self.settings["bbox_cot"], evaluation=True)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=images, return_tensors="pt",
                           min_pixels=512 * 28 * 28, max_pixels=2048 * 28 * 28).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = self.mllm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                        output_hidden_states=True, return_dict_in_generate=True, use_cache=True)
            hidden = torch.cat([step[-1][0] for step in output.hidden_states])
            queries = hidden[output.sequences[0, :-1] == self.seg_id]
            prediction = processor.tokenizer.decode(output.sequences[0, inputs.input_ids.shape[1]:], skip_special_tokens=False)
            if not len(queries):
                return torch.zeros(image.height, image.width, dtype=torch.bool, device=device), prediction
            features = self.encode_masks(sam_image(image, torch.bfloat16)[None].to(device))
            embeddings = self.mask_projector(queries)
            logits = torch.cat([self.decode_masks(features, q[None], torch.zeros(1, dtype=torch.long, device=device))
                                for q in embeddings])
            logits = F.interpolate(logits[:, None], (1024, 1024), mode="bilinear", align_corners=False)
            logits = F.interpolate(logits, image.size[::-1], mode="bilinear", align_corners=False)[:, 0]
        return (logits > 0).any(0), prediction

    def trainable_state_dict(self):
        return {name: param.detach().cpu() for name, param in self.named_parameters() if param.requires_grad}

    def load_trainable_state_dict(self, state):
        expected = {name for name, param in self.named_parameters() if param.requires_grad}
        if expected != set(state):
            raise ValueError(f"Checkpoint parameters differ: missing={expected - set(state)}, extra={set(state) - expected}")
        self.load_state_dict(state, strict=False)

    def prepare_for_inference(self):
        self.mllm.gradient_checkpointing_disable()
        self.mllm = self.mllm.merge_and_unload()
        self.to(dtype=torch.bfloat16).eval()
        return self
