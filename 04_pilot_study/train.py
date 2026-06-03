"""Minimal GPU training for binary relation-presence classifiers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .config import (
    CHECKPOINT_DIR,
    ModelSpec,
    TRAIN_BATCH_SIZE,
    TRAIN_LR,
    TRAIN_MAX_STEPS,
    TRAIN_SEEDS,
    TRAIN_WARMUP_RATIO,
    MAX_SEQ_LENGTH,
)


class RelationDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(ex["label"], dtype=torch.long)
        return item


def _require_gpu() -> torch.device:
    if not torch.cuda.is_available():
        print("ERROR: GPU required. Submit via sbatch project_1/04_pilot_study.sbatch", file=sys.stderr)
        sys.exit(1)
    return torch.device("cuda")


def checkpoint_path(spec: ModelSpec, seed: int) -> Path:
    return CHECKPOINT_DIR / spec.model_id / f"seed_{seed}"


def train_model(
    spec: ModelSpec,
    examples: list[dict],
    seed: int,
    force: bool = False,
) -> Path:
    out_dir = checkpoint_path(spec, seed)
    if (out_dir / "config.json").exists() and not force:
        print(f"  Checkpoint exists: {out_dir}")
        return out_dir

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _require_gpu()

    print(f"\n=== Training {spec.short_name} seed={seed} ===")
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(spec.hf_name, num_labels=2)
    model.to(device)

    ds = RelationDataset(examples, tokenizer, MAX_SEQ_LENGTH)
    loader = DataLoader(ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)

    optim = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR)
    total_steps = TRAIN_MAX_STEPS
    warmup = int(total_steps * TRAIN_WARMUP_RATIO)
    sched = get_linear_schedule_with_warmup(optim, warmup, total_steps)

    model.train()
    step = 0
    losses: list[float] = []
    while step < total_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()
            losses.append(float(loss.item()))
            step += 1
            if step % 500 == 0:
                print(f"    step {step}/{total_steps} loss={np.mean(losses[-100:]):.4f}")
            if step >= total_steps:
                break

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta = {
        "model_id": spec.model_id,
        "hf_name": spec.hf_name,
        "seed": seed,
        "train_steps": total_steps,
        "final_loss": float(np.mean(losses[-50:])) if losses else None,
        "n_train_examples": len(examples),
    }
    (out_dir / "04_train_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Saved checkpoint -> {out_dir}")
    return out_dir


def train_all_models(
    examples: list[dict],
    force: bool = False,
    model_ids: list[str] | None = None,
) -> None:
    from .config import MODELS, MODEL_BY_ID

    specs = MODELS if model_ids is None else [MODEL_BY_ID[m] for m in model_ids]
    for spec in specs:
        for seed in TRAIN_SEEDS:
            train_model(spec, examples, seed, force=force)
