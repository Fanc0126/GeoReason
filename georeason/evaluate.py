"""Greedy-decoding evaluation on EarthReason and RRSIS-D."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from accelerate import PartialState
from tqdm import tqdm

from .data import evaluation_queries, load_processor, read_manifest
from .masks import decode_mask
from .model import GeoReason


def mask_statistics(prediction, target):
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    return np.array([intersection / union if union else 1.0, intersection, union, 1], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=["earthreason", "rrsisd"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--manifests", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-masks", action="store_true")
    args = parser.parse_args()
    state = PartialState()
    if state.is_main_process:
        args.output.mkdir(parents=True, exist_ok=False)
    state.wait_for_everyone()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    config = checkpoint["config"]["model"]
    processor = load_processor(config)
    model = GeoReason(config, processor.tokenizer)
    model.load_trainable_state_dict(checkpoint["model"])
    model.to(state.device).prepare_for_inference()
    rows = read_manifest(args.manifests / f"{args.dataset}_{args.split}.jsonl")
    queries = list(evaluation_queries(rows, args.seed))
    statistics = np.zeros(4, dtype=np.float64)
    if args.save_masks:
        (args.output / "masks").mkdir(exist_ok=True)
    with (args.output / f"predictions_rank{state.process_index}.jsonl").open("x") as f:
        for index in tqdm(range(state.process_index, len(queries), state.num_processes), disable=not state.is_main_process):
            row, question = queries[index]
            image = Image.open(args.data_root / row["image"]).convert("RGB")
            prediction, response = model.predict(image, question, row["task"], processor)
            prediction = prediction.cpu().numpy()
            target = decode_mask(row, args.data_root)
            statistics += mask_statistics(prediction, target)
            f.write(json.dumps(dict(index=index, id=row["id"], question=question, response=response)) + "\n")
            if args.save_masks:
                Image.fromarray(prediction.astype(np.uint8) * 255).save(args.output / "masks" / f"{index:06d}.png")
    statistics = torch.tensor(statistics, device=state.device)
    if state.num_processes > 1:
        torch.distributed.all_reduce(statistics)
    if state.is_main_process:
        giou_sum, intersection, union, count = statistics.tolist()
        result = dict(dataset=args.dataset, split=args.split, samples=int(count),
                      gIoU=100 * giou_sum / count, cIoU=100 * intersection / (union + 1e-10))
        with (args.output / "metrics.json").open("x") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
