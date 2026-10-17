"""Image, conversation and mask inputs for GeoReason."""
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
from transformers import AutoProcessor

from .masks import bbox_from_mask, decode_mask, foreground_patch_indices
from .prompts import (ANSWERS, SEG_INSTRUCTIONS, SEG_QUESTIONS,
                      segmentation_answer, segmentation_prompt)


def load_processor(config):
    processor = AutoProcessor.from_pretrained(config["pretrained"], revision=config["revision"],
                                             trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    return processor


def read_manifest(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f]


def sam_image(image, dtype=torch.float32):
    pixels = torch.from_numpy(np.array(image.resize((1024, 1024)))).permute(2, 0, 1).to(dtype) / 255
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=dtype)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=dtype)[:, None, None]
    return (pixels - mean) / std


def encode_conversation(turns, tokenizer, image_tokens, max_length):
    image_text = "<|vision_start|>" + "<|image_pad|>" * image_tokens + "<|vision_end|>"
    ids, labels = [], []
    for user, assistant in zip(turns[::2], turns[1::2]):
        question = user["value"].replace("<image>", image_text)
        prefix = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        answer_ids = tokenizer.encode(assistant["value"] + "<|im_end|>", add_special_tokens=False)
        separator = tokenizer.encode("\n", add_special_tokens=False)
        ids += prefix_ids + answer_ids + separator
        labels += [-100] * len(prefix_ids) + answer_ids + [-100] * len(separator)
    return dict(input_ids=torch.tensor(ids[:max_length]), labels=torch.tensor(labels[:max_length]))


class TrainingDataset(Dataset):
    def __init__(self, config, processor):
        self.config = config
        self.processor = processor
        self.root = Path(config["data"]["root"])
        self.rows, self.indices = [], []
        for name, repeat in config["data"]["repeats"].items():
            if repeat == 0:
                continue
            rows = read_manifest(Path(config["data"]["manifests"]) / f"{name}_train.jsonl")
            self.indices += list(range(len(self.rows), len(self.rows) + len(rows))) * repeat
            self.rows += rows

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.rows[self.indices[index]]
        image = Image.open(self.root / row["image"]).convert("RGB")
        model_config = self.config["model"]
        visual = self.processor.image_processor(images=image, return_tensors="pt",
                                               min_pixels=512 * 28 * 28, max_pixels=2048 * 28 * 28)
        grid = visual["image_grid_thw"][0]
        merge = self.processor.image_processor.merge_size
        mask, patches = None, torch.empty(0, dtype=torch.long)
        if row["task"] == "qa":
            turns = row["conversations"]
        else:
            mask = decode_mask(row, self.root)
            if mask.shape != (image.height, image.width):
                mask = np.array(Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST))
            if row["task"] == "reasoning":
                question = random.choice(row["questions"]).strip() + " " + random.choice(SEG_INSTRUCTIONS)
                reasoning = random.choice(row["reasoning"]).strip() if row["reasoning"] else "There is no target object in the image."
            else:
                question = random.choice(SEG_QUESTIONS).format(class_name=random.choice(row["questions"]).lower())
                reasoning = ""
            question = segmentation_prompt(question, row["task"], model_config["ap"], model_config["bbox_cot"])
            terminal = "[SEG]." if row["task"] == "reasoning" and not row["reasoning"] else random.choice(ANSWERS)
            answer = segmentation_answer(reasoning, terminal, bbox_from_mask(mask),
                                         row["task"], model_config["bbox_cot"])
            turns = [{"from": "human", "value": "<image>\n" + question},
                     {"from": "gpt", "value": answer}]
            patches = torch.tensor(foreground_patch_indices(mask, grid, merge), dtype=torch.long)
            mask = torch.from_numpy(mask).float()
        result = encode_conversation(turns, self.processor.tokenizer, int(grid.prod()) // merge**2,
                                     self.config["train"]["max_length"])
        result.update(pixel_values=visual["pixel_values"], image_grid_thw=visual["image_grid_thw"],
                      sam_pixels=sam_image(image) if mask is not None else None,
                      mask=mask, patches=patches)
        return result


class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        ids = pad_sequence([r["input_ids"] for r in rows], batch_first=True, padding_value=self.pad_token_id)
        lengths = torch.tensor([len(r["input_ids"]) for r in rows])
        return dict(input_ids=ids,
                    attention_mask=torch.arange(ids.shape[1])[None] < lengths[:, None],
                    labels=pad_sequence([r["labels"] for r in rows], batch_first=True, padding_value=-100),
                    pixel_values=torch.cat([r["pixel_values"] for r in rows]),
                    image_grid_thw=torch.cat([r["image_grid_thw"] for r in rows]),
                    sam_pixels=[r["sam_pixels"] for r in rows], masks=[r["mask"] for r in rows],
                    patches=[r["patches"] for r in rows])


class EpochSampler(Sampler):
    def __init__(self, dataset, global_batch_size, seed):
        self.size = len(dataset)
        self.total = math.ceil(self.size / global_batch_size) * global_batch_size
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return self.total

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist()
        return iter((indices * math.ceil(self.total / self.size))[:self.total])


def evaluation_queries(rows, seed):
    rng = np.random.RandomState(seed)
    for row in rows:
        questions = [str(rng.choice(row["questions"]))] if row["task"] == "reasoning" else row["questions"]
        for question in questions:
            if row["task"] == "reasoning":
                question += " Provide the segmentation result for this."
            else:
                question = f"Please segment {question} in this image. "
            yield row, question
