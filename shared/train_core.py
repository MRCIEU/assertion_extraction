"""Core training loop with optional per-epoch checkpoint saving."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .constants import CHECKPOINT_CRITERION, EARLY_STOPPING_PATIENCE, MAX_EPOCHS, MAX_SEQ_LENGTH, TRAIN_BATCH_SIZE
from .models import ModelSpec


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
            ex["text"], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(ex["label"], dtype=torch.long)
        return item


def require_gpu() -> torch.device:
    if not torch.cuda.is_available():
        print("ERROR: GPU required.", file=sys.stderr)
        sys.exit(1)
    return torch.device("cuda")


def evaluate_loader(model, loader, device) -> tuple[float, float]:
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            losses.append(float(out.loss.item()))
            preds.extend(out.logits.argmax(dim=-1).cpu().numpy().tolist())
            labels.extend(batch["labels"].cpu().numpy().tolist())
    return float(np.mean(losses)), float(f1_score(labels, preds, average="binary", zero_division=0))


@dataclass
class RecipeConfig:
    lr: float
    warmup_ratio: float
    warmup_label: str

    def strategy_tag(self) -> str:
        lr_s = f"{self.lr:.0e}".replace("+", "")
        warm = "warmup" if self.warmup_ratio > 0 else "nowarmup"
        return f"val_f1_lr{lr_s}_{warm}"


def _save_ckpt(model, tokenizer, out_dir: Path, *, fp16: bool = False) -> None:
    """Save weights only (no optimizer state). fp16 applies to epoch snapshots; best stays fp32."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if fp16:
        state = model.state_dict()
        model.half()
        model.save_pretrained(out_dir)
        model.float()
        model.load_state_dict(state)
    else:
        model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def _prune_epoch_checkpoints(epochs_dir: Path, keep: int) -> None:
    """Retain only the keep most recent epoch_* directories (never touches best/)."""
    if keep <= 0 or not epochs_dir.exists():
        return
    dirs = sorted(epochs_dir.glob("epoch_*"), key=lambda p: p.name)
    for old in dirs[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


def train_with_epoch_checkpoints(
    spec: ModelSpec,
    seed: int,
    train_examples: list[dict],
    val_examples: list[dict],
    run_root: Path,
    recipe: RecipeConfig,
    *,
    save_epoch_fp16: bool = True,
    max_epoch_checkpoints_to_keep: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Train one run; save every epoch under run_root/epochs/epoch_NN/ (optionally fp16).
    Best val_f1 checkpoint saved full-precision at run_root/best/. Val metrics in training_log.json.
    No benchmark F1 here (step 2 scores best only; folder 20 scores per-epoch on demand).
    """
    log_path = run_root / "training_log.json"
    if log_path.exists() and not force:
        return json.loads(log_path.read_text(encoding="utf-8"))

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = require_gpu()
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(spec.hf_name, num_labels=2).to(device)
    train_loader = DataLoader(
        RelationDataset(train_examples, tokenizer, MAX_SEQ_LENGTH), batch_size=TRAIN_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        RelationDataset(val_examples, tokenizer, MAX_SEQ_LENGTH), batch_size=TRAIN_BATCH_SIZE
    )
    total_steps = max(1, len(train_loader)) * MAX_EPOCHS
    optim = torch.optim.AdamW(model.parameters(), lr=recipe.lr)
    if recipe.warmup_ratio > 0:
        sched = get_linear_schedule_with_warmup(optim, int(total_steps * recipe.warmup_ratio), total_steps)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda _: 1.0)

    best_val_f1, best_epoch, wait = -1.0, 0, 0
    epoch_rows: list[dict] = []
    best_dir = run_root / "best"
    epochs_dir = run_root / "epochs"

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()
            losses.append(float(out.loss.item()))

        val_loss, val_f1 = evaluate_loader(model, val_loader, device)
        ep_dir = epochs_dir / f"epoch_{epoch:02d}"
        _save_ckpt(model, tokenizer, ep_dir, fp16=save_epoch_fp16)

        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "val_f1": val_f1,
                "checkpoint": str(ep_dir),
                "checkpoint_precision": "fp16" if save_epoch_fp16 else "fp32",
            }
        )

        if val_f1 > best_val_f1:
            best_val_f1, best_epoch, wait = val_f1, epoch, 0
            if best_dir.exists():
                shutil.rmtree(best_dir)
            _save_ckpt(model, tokenizer, best_dir, fp16=False)
        else:
            wait += 1
            if wait >= EARLY_STOPPING_PATIENCE:
                break

        if max_epoch_checkpoints_to_keep is not None:
            _prune_epoch_checkpoints(epochs_dir, max_epoch_checkpoints_to_keep)

    payload = {
        "model_id": spec.model_id,
        "seed": seed,
        "training_strategy": recipe.strategy_tag(),
        "checkpoint_criterion": CHECKPOINT_CRITERION,
        "train_lr": recipe.lr,
        "warmup_label": recipe.warmup_label,
        "best_epoch_val_f1": best_epoch,
        "best_val_f1": best_val_f1,
        "best_checkpoint": str(best_dir),
        "best_checkpoint_precision": "fp32",
        "epoch_checkpoint_fp16": save_epoch_fp16,
        "epoch_curve": epoch_rows,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def train_best_only(
    spec: ModelSpec,
    seed: int,
    train_examples: list[dict],
    val_examples: list[dict],
    best_dir: Path,
    recipe: RecipeConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    meta_path = best_dir / "train_metadata.json"
    if meta_path.exists() and not force:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = require_gpu()
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(spec.hf_name, num_labels=2).to(device)
    train_loader = DataLoader(
        RelationDataset(train_examples, tokenizer, MAX_SEQ_LENGTH), batch_size=TRAIN_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        RelationDataset(val_examples, tokenizer, MAX_SEQ_LENGTH), batch_size=TRAIN_BATCH_SIZE
    )
    total_steps = max(1, len(train_loader)) * MAX_EPOCHS
    optim = torch.optim.AdamW(model.parameters(), lr=recipe.lr)
    if recipe.warmup_ratio > 0:
        sched = get_linear_schedule_with_warmup(optim, int(total_steps * recipe.warmup_ratio), total_steps)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda _: 1.0)

    best_val_f1, best_epoch, wait = -1.0, 0, 0
    epoch_rows: list[dict] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()
        val_loss, val_f1 = evaluate_loader(model, val_loader, device)
        epoch_rows.append({"epoch": epoch, "val_loss": val_loss, "val_f1": val_f1})
        if val_f1 > best_val_f1:
            best_val_f1, best_epoch, wait = val_f1, epoch, 0
            _save_ckpt(model, tokenizer, best_dir, fp16=False)
        else:
            wait += 1
            if wait >= EARLY_STOPPING_PATIENCE:
                break

    payload = {
        "model_id": spec.model_id,
        "seed": seed,
        "best_epoch_val_f1": best_epoch,
        "best_val_f1": best_val_f1,
        "epoch_curve": epoch_rows,
    }
    best_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
