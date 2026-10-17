import itertools
import io
import unittest

import numpy as np
import torch

from georeason.data import Collator, EpochSampler, encode_conversation, evaluation_queries
from georeason.evaluate import mask_statistics
from georeason.losses import mask_losses
from georeason.masks import bbox_from_mask, foreground_patch_indices
from georeason.model import GeoReason, PTAHead
from georeason.prepare import auxiliary
from georeason.prompts import segmentation_answer, segmentation_prompt
from georeason.train import lr_multiplier


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class CoreTests(unittest.TestCase):
    def test_component_prompts(self):
        for task, ap, bbox in itertools.product(["referring", "reasoning"], [False, True], [False, True]):
            for evaluation in [False, True]:
                prompt = segmentation_prompt("Segment the target.", task, ap, bbox, evaluation)
                self.assertEqual("[SEG]" in prompt, ap)
                self.assertEqual("bbox" in prompt, bbox)
            answer = segmentation_answer("A building", "It is [SEG].", '{"bbox_2d": [0,0,10,10]}', task, bbox)
            self.assertIn("[SEG]", answer)
            self.assertEqual("bbox_2d" in answer, bbox)

    def test_bbox_and_patch_mapping(self):
        mask = np.zeros((8, 8), np.uint8)
        mask[2:6, 2:6] = 1
        self.assertEqual(bbox_from_mask(mask), '{"bbox_2d": [250,250,625,625]}')
        self.assertEqual(foreground_patch_indices(mask, [1, 8, 8]).tolist(), [5, 6, 9, 10])

    def test_supervised_tokens(self):
        turns = [{"value": "<image>\nQuestion?"}, {"value": "Answer [SEG]."},
                 {"value": "Another question?"}, {"value": "Second answer."}]
        encoded = encode_conversation(turns, CharacterTokenizer(), 2, 4096)
        supervised = encoded["input_ids"][encoded["labels"] != -100].tolist()
        self.assertEqual(bytes(supervised).decode(), "Answer [SEG].<|im_end|>Second answer.<|im_end|>")
        self.assertEqual(bytes(encoded["input_ids"].tolist()).decode().count("<|image_pad|>"), 2)

    def test_auxiliary_holdout_filter(self):
        rows = [dict(id="a", image="00001.png", conversations=[dict(from_="human", value="Question")]),
                dict(id="b", image="00002.png", conversations=[dict(from_="human", value="Question")]),
                dict(id="c", image="00003.png", conversations=[dict(from_="human", value="[refer] target")])]
        for row in rows:
            row["conversations"][0]["from"] = row["conversations"][0].pop("from_")
        result = auxiliary(rows, {1}, "geochat", "images")
        self.assertEqual([x["id"] for x in result], ["b"])

    def test_segmentation_loss_gradient(self):
        torch.manual_seed(42)
        logits = torch.randn(5, 16, 16, requires_grad=True)
        targets = (torch.rand_like(logits) > 0.5).float()
        bce, dice = mask_losses(logits, targets, num_points=64)
        (bce + dice).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_pta_head(self):
        head = PTAHead(8, 16)
        hidden = torch.randn(2, 8, requires_grad=True)
        target = torch.randn(2, 8)
        loss = 1 - torch.nn.functional.cosine_similarity(head(hidden), target).mean()
        loss.backward()
        self.assertEqual(head(hidden).shape, hidden.shape)
        self.assertTrue(all(p.grad is not None for p in head.parameters()))

    def test_metrics(self):
        values = mask_statistics(np.array([0, 1, 1]), np.array([0, 0, 1]))
        np.testing.assert_allclose(values, [0.5, 1, 2, 1])
        self.assertEqual(mask_statistics(np.zeros(2), np.zeros(2))[0], 1)

    def test_evaluation_queries(self):
        rows = [dict(id="a", task="reasoning", questions=["a", "b"]),
                dict(id="b", task="referring", questions=["left", "right"])]
        self.assertEqual(list(evaluation_queries(rows, 42)), list(evaluation_queries(rows, 42)))
        self.assertEqual(len(list(evaluation_queries(rows, 42))), 3)

    def test_learning_rate(self):
        self.assertEqual(lr_multiplier(0, 100, 5), 1e-5)
        self.assertEqual(lr_multiplier(5, 100, 5), 1)
        self.assertEqual(lr_multiplier(100, 100, 5), 0)

    def test_epoch_padding(self):
        sampler = EpochSampler(list(range(19)), 8, 42)
        self.assertEqual(len(sampler), 24)
        self.assertEqual(set(sampler), set(range(19)))
        first = list(sampler)
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))

    def test_checkpoint_roundtrip(self):
        model = GeoReason.__new__(GeoReason)
        torch.nn.Module.__init__(model)
        model.pta_head = PTAHead(8, 16)
        model.mask_projector = torch.nn.Linear(8, 4)
        model.frozen = torch.nn.Linear(8, 8).requires_grad_(False)
        state = model.trainable_state_dict()
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        with torch.no_grad():
            model.pta_head.gate_proj.weight.zero_()
        model.load_trainable_state_dict(saved)
        self.assertTrue(torch.equal(model.pta_head.gate_proj.weight, saved["pta_head.gate_proj.weight"]))
        self.assertFalse(any(k.startswith("frozen.") for k in saved))


if __name__ == "__main__":
    unittest.main()
