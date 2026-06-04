"""GPU training with validation early stopping; checkpoint = best validation F1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .config import (
    CHECKPOINT_CRITERION,
    CHECKPOINT_DIR,
    COMPLETE_MARKER,
    EARLY_STOPPING_PATIENCE,
    MAX_EPOCHS,
    MAX_SEQ_LENGTH,
    TRAINING_STRATEGY,
    TRAIN_BATCH_SIZE,
    TRAIN_LR,
    TRAIN_WARMUP_RATIO,
    ModelSpec,
    RESULTS_DIR,
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
        print("ERROR: GPU required. Submit via sbatch.", file=sys.stderr)
        sys.exit(1)
    return torch.device("cuda")


def checkpoint_path(spec: ModelSpec, seed: int) -> Path:
    return CHECKPOINT_DIR / spec.model_id / f"seed_{seed}"


def result_path(spec: ModelSpec, seed: int) -> Path:
    return RESULTS_DIR / spec.model_id / f"seed_{seed}" / COMPLETE_MARKER


def is_complete(spec: ModelSpec, seed: int) -> bool:
    marker = result_path(spec, seed)
    if not marker.exists():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("training_strategy") == TRAINING_STRATEGY


def has_valid_checkpoint(spec: ModelSpec, seed: int) -> bool:
    """True if a checkpoint exists for the current training strategy (eval may still be pending)."""
    out_dir = checkpoint_path(spec, seed)
    meta_path = out_dir / "10_train_metadata.json"
    if not out_dir.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return meta.get("training_strategy") == TRAINING_STRATEGY


def _evaluate_loader(model, loader, device) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            losses.append(float(out.loss.item()))
            pred = out.logits.argmax(dim=-1).cpu().numpy()
            preds.extend(pred.tolist())
            labels.extend(batch["labels"].cpu().numpy().tolist())
    f1 = float(f1_score(labels, preds, average="binary", zero_division=0))
    return float(np.mean(losses)), f1


def train_model(
    spec: ModelSpec,
    train_examples: list[dict],
    val_examples: list[dict],
    seed: int,
    force: bool = False,
) -> Path:
    out_dir = checkpoint_path(spec, seed)
    marker = result_path(spec, seed)

    if is_complete(spec, seed) and not force:
        print(f"  Complete (skip train): {spec.model_id} seed={seed}")
        return out_dir

    if has_valid_checkpoint(spec, seed) and not force:
        print(f"  Checkpoint exists (skip train): {spec.model_id} seed={seed}")
        return out_dir

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _require_gpu()

    print(
        f"\n=== Training {spec.short_name} seed={seed} "
        f"(lr={TRAIN_LR}, warmup={TRAIN_WARMUP_RATIO}, ckpt={CHECKPOINT_CRITERION}) ==="
    )
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(spec.hf_name, num_labels=2)
    model.to(device)

    train_ds = RelationDataset(train_examples, tokenizer, MAX_SEQ_LENGTH)
    val_ds = RelationDataset(val_examples, tokenizer, MAX_SEQ_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * MAX_EPOCHS
    optim = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR)
    if TRAIN_WARMUP_RATIO > 0:
        warmup = int(total_steps * TRAIN_WARMUP_RATIO)
        sched: torch.optim.lr_scheduler.LRScheduler = get_linear_schedule_with_warmup(
            optim, warmup, total_steps
        )
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda _: 1.0)

    best_val_f1 = -1.0
    best_val_loss_at_best_f1 = float("inf")
    best_epoch = 0
    wait = 0
    epoch_rows: list[dict] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()
            epoch_losses.append(float(loss.item()))

        val_loss, val_f1 = _evaluate_loader(model, val_loader, device)
        mean_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        epoch_rows.append(
            {"epoch": epoch, "train_loss": mean_train_loss, "val_loss": val_loss, "val_f1": val_f1}
        )
        print(
            f"    epoch {epoch}/{MAX_EPOCHS} train_loss={mean_train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss_at_best_f1 = val_loss
            best_epoch = epoch
            wait = 0
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
        else:
            wait += 1
            if wait >= EARLY_STOPPING_PATIENCE:
                print(f"    early stop at epoch {epoch} (best val_f1 epoch {best_epoch})")
                break

    meta = {
        "model_id": spec.model_id,
        "hf_name": spec.hf_name,
        "seed": seed,
        "training_strategy": TRAINING_STRATEGY,
        "checkpoint_criterion": CHECKPOINT_CRITERION,
        "train_lr": TRAIN_LR,
        "train_warmup_ratio": TRAIN_WARMUP_RATIO,
        "best_epoch_val_f1": best_epoch,
        "best_val_f1": best_val_f1,
        "val_loss_at_best_f1": best_val_loss_at_best_f1,
        "n_train_examples": len(train_examples),
        "n_val_examples": len(val_examples),
        "epoch_curve": epoch_rows,
    }
    (out_dir / "10_train_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Saved checkpoint -> {out_dir} (best val_f1 epoch {best_epoch}, F1={best_val_f1:.4f})")
    return out_dir
