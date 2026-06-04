"""Train DeBERTa on recipe grid points (reuses Round-1 data format and eval protocol)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .config import (
    CHECKPOINT_CRITERION,
    CHECKPOINT_DIR,
    EARLY_STOPPING_PATIENCE,
    MAX_EPOCHS,
    GridPoint,
)

# Round-1 training stack (same corpus, leak check, entity-marked inputs, benchmark protocol)
_r1 = __import__(
    "10_round1_benchmark_kb.config",
    fromlist=["INFER_BATCH_SIZE", "MAX_SEQ_LENGTH", "TRAIN_BATCH_SIZE", "MODEL_BY_ID"],
)
_r1_train = __import__(
    "10_round1_benchmark_kb.train",
    fromlist=["RelationDataset", "_evaluate_loader", "_require_gpu"],
)
_r1_bench = __import__(
    "10_round1_benchmark_kb.benchmark_eval",
    fromlist=["build_biored_test_examples"],
)

INFER_BATCH_SIZE = _r1.INFER_BATCH_SIZE
MAX_SEQ_LENGTH = _r1.MAX_SEQ_LENGTH
TRAIN_BATCH_SIZE = _r1.TRAIN_BATCH_SIZE
DEBERTA_SPEC = _r1.MODEL_BY_ID["deberta_base"]
RelationDataset = _r1_train.RelationDataset
_evaluate_loader = _r1_train._evaluate_loader
_require_gpu = _r1_train._require_gpu
build_biored_test_examples = _r1_bench.build_biored_test_examples


def checkpoint_dir(point: GridPoint, seed: int) -> Path:
    return CHECKPOINT_DIR / point.key / f"seed_{seed}"


def marker_path(point: GridPoint, seed: int) -> Path:
    from .config import RESULTS_DIR, COMPLETE_MARKER

    return RESULTS_DIR / point.key / f"seed_{seed}" / COMPLETE_MARKER


def is_complete(point: GridPoint, seed: int) -> bool:
    path = marker_path(point, seed)
    return path.exists()


def evaluate_benchmark_f1_from_ckpt(ckpt_dir: Path, test_examples: list[dict]) -> dict:
    """Self-measured BioRED test presence-F1 (same protocol as Round 1)."""
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint {ckpt_dir}")

    device = _require_gpu()
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    model.to(device)
    model.eval()

    ds = RelationDataset(test_examples, tokenizer, MAX_SEQ_LENGTH)
    loader = DataLoader(ds, batch_size=INFER_BATCH_SIZE, shuffle=False)

    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            pred = logits.argmax(dim=-1).cpu().numpy()
            preds.extend(pred.tolist())
            labels.extend(batch["labels"].cpu().numpy().tolist())

    f1 = float(f1_score(labels, preds, average="binary", zero_division=0))
    return {
        "benchmark_f1": f1,
        "n_test_examples": len(test_examples),
        "n_positives": int(sum(labels)),
    }


def train_grid_point(
    point: GridPoint,
    seed: int,
    train_examples: list[dict],
    val_examples: list[dict],
    force: bool = False,
) -> Path:
    out_dir = checkpoint_dir(point, seed)
    marker = marker_path(point, seed)

    if is_complete(point, seed) and not force:
        print(f"  skip (complete): {point.key} seed={seed}")
        return out_dir

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _require_gpu()

    print(
        f"\n=== DeBERTa recipe grid: {point.key} seed={seed} "
        f"lr={point.lr:.0e} warmup={point.warmup_label} ==="
    )
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_SPEC.hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(DEBERTA_SPEC.hf_name, num_labels=2)
    model.to(device)

    train_ds = RelationDataset(train_examples, tokenizer, MAX_SEQ_LENGTH)
    val_ds = RelationDataset(val_examples, tokenizer, MAX_SEQ_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * MAX_EPOCHS
    optim = torch.optim.AdamW(model.parameters(), lr=point.lr)
    if point.warmup_ratio > 0:
        warmup_steps = int(total_steps * point.warmup_ratio)
        sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda _: 1.0)

    best_val_f1 = -1.0
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
        "model_id": DEBERTA_SPEC.model_id,
        "run_key": point.key,
        "seed": seed,
        "train_lr": point.lr,
        "warmup_label": point.warmup_label,
        "train_warmup_ratio": point.warmup_ratio,
        "checkpoint_criterion": CHECKPOINT_CRITERION,
        "best_epoch_val_f1": best_epoch,
        "best_val_f1": best_val_f1,
        "epoch_curve": epoch_rows,
    }
    (out_dir / "11_train_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Saved checkpoint -> {out_dir} (best val_f1 epoch {best_epoch}, F1={best_val_f1:.4f})")
    return out_dir
