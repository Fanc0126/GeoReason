"""Distributed GeoReason fine-tuning."""
import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import Collator, EpochSampler, TrainingDataset, load_processor
from .model import GeoReason


def lr_multiplier(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return 1e-5 + (1 - 1e-5) * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/georeason.yaml")
    parser.add_argument("--output")
    parser.add_argument("--data-root")
    parser.add_argument("--manifests")
    for component in ["ap", "bbox-cot", "pta"]:
        parser.add_argument("--" + component, action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    for component in ["ap", "bbox_cot", "pta"]:
        if getattr(args, component) is not None:
            config["model"][component] = getattr(args, component)
    if args.output:
        config["train"]["output"] = args.output
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.manifests:
        config["data"]["manifests"] = args.manifests
    settings = config["train"]
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=settings["gradient_accumulation"],
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    set_seed(settings["seed"])
    output_dir = Path(settings["output"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=False)
        with (output_dir / "config.yaml").open("x") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    accelerator.wait_for_everyone()
    processor = load_processor(config["model"])
    model = GeoReason(config["model"], processor.tokenizer)
    dataset = TrainingDataset(config, processor)
    global_batch = settings["batch_size"] * settings["gradient_accumulation"] * accelerator.num_processes
    sampler = EpochSampler(dataset, global_batch, settings["seed"])
    loader = DataLoader(dataset, batch_size=settings["batch_size"], sampler=sampler, num_workers=settings["workers"],
                        collate_fn=Collator(processor.tokenizer.pad_token_id), pin_memory=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=settings["learning_rate"], betas=(0.9, 0.999), weight_decay=settings["weight_decay"])
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    total_steps = math.ceil(len(loader) / settings["gradient_accumulation"]) * settings["epochs"]
    warmup_steps = round(total_steps * settings["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_multiplier(s, total_steps, warmup_steps))
    step = 0
    progress = tqdm(total=total_steps, disable=not accelerator.is_main_process)

    def save(name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            payload = dict(config=config, step=step, model=accelerator.unwrap_model(model).trainable_state_dict())
            with (output_dir / name).open("xb") as f:
                torch.save(payload, f)

    for epoch in range(settings["epochs"]):
        model.train()
        loader.set_epoch(epoch)
        for batch in loader:
            with accelerator.accumulate(model):
                losses = model(**batch)
                loss = sum(losses.values())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                step += 1
                progress.update()
                if step % 10 == 0:
                    mean_loss = accelerator.reduce(loss.detach(), reduction="mean").item()
                    progress.set_postfix(loss=f"{mean_loss:.4f}")
                    if accelerator.is_main_process:
                        with (output_dir / "train.jsonl").open("a") as f:
                            f.write(json.dumps(dict(step=step, epoch=epoch + 1, loss=mean_loss,
                                                    lr=scheduler.get_last_lr()[0])) + "\n")
                if step % settings["save_steps"] == 0:
                    save(f"step_{step}.pt")
    save("final.pt")
    progress.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
