"""Sweep training: log full val curves; expose val_loss vs val_f1 checkpoint selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from importlib import import_module

_r1 = import_module("10_round1_benchmark_kb.train")
RelationDataset = _r1.RelationDataset
_require_gpu = _r1._require_gpu

from .config import (
    EARLY_STOPPING_PATIENCE,
    MAX_EPOCHS,
    MAX_SEQ_LENGTH,
    SWEEP_CKPT_DIR,
    SWEEP_CKPT_F1_DIR,
    SWEEP_RESULTS_DIR,
    TRAIN_BATCH_SIZE,
    SweepRun,
)


def _checkpoint_dir(run: SweepRun, by_f1: bool = False) -> Path:
    base = SWEEP_CKPT_F1_DIR if by_f1 else SWEEP_CKPT_DIR
    return base / run.run_id


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


def train_sweep_run(
    run: SweepRun,
    train_examples: list[dict],
    val_examples: list[dict],
    force: bool = False,
) -> dict[str, Any]:
    marker = run.result_path()
    if marker.exists() and not force:
        return json.loads(marker.read_text(encoding="utf-8"))

    spec = __import__("10_round1_benchmark_kb.config", fromlist=["MODEL_BY_ID"]).MODEL_BY_ID[run.model_id]

    torch.manual_seed(run.seed)
    np.random.seed(run.seed)
    device = _require_gpu()

    print(
        f"\n=== Sweep {run.run_id} | {spec.short_name} lr={run.lr} warmup={run.warmup_label} ==="
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
    warmup_steps = int(total_steps * run.warmup_ratio) if run.warmup_ratio > 0 else 0

    optim = torch.optim.AdamW(model.parameters(), lr=run.lr)
    if warmup_steps > 0:
        sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda _: 1.0)

    best_val_loss = float("inf")
    best_epoch_by_loss = 0
    best_val_f1 = -1.0
    best_epoch_by_f1 = 0
    wait = 0
    epoch_rows: list[dict[str, Any]] = []

    ckpt_loss_dir = _checkpoint_dir(run, by_f1=False)
    ckpt_f1_dir = _checkpoint_dir(run, by_f1=True)

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
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_f1,
            }
        )
        print(
            f"    epoch {epoch}/{MAX_EPOCHS} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch_by_loss = epoch
            wait = 0
            ckpt_loss_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_loss_dir)
            tokenizer.save_pretrained(ckpt_loss_dir)
        else:
            wait += 1

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch_by_f1 = epoch
            ckpt_f1_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_f1_dir)
            tokenizer.save_pretrained(ckpt_f1_dir)

        if wait >= EARLY_STOPPING_PATIENCE:
            print(
                f"    early stop at epoch {epoch} "
                f"(best val_loss epoch {best_epoch_by_loss}, best val_f1 epoch {best_epoch_by_f1})"
            )
            break

    selection_disagrees = best_epoch_by_loss != best_epoch_by_f1
    val_f1_at_loss_epoch = next(
        (r["val_f1"] for r in epoch_rows if r["epoch"] == best_epoch_by_loss), None
    )
    val_loss_at_f1_epoch = next(
        (r["val_loss"] for r in epoch_rows if r["epoch"] == best_epoch_by_f1), None
    )

    payload: dict[str, Any] = {
        "run_id": run.run_id,
        "model_id": run.model_id,
        "short_name": spec.short_name,
        "lr": run.lr,
        "warmup_label": run.warmup_label,
        "warmup_ratio": run.warmup_ratio,
        "seed": run.seed,
        "selection_criterion_used": "val_loss",
        "best_epoch_by_val_loss": best_epoch_by_loss,
        "best_val_loss": best_val_loss,
        "val_f1_at_loss_epoch": val_f1_at_loss_epoch,
        "best_epoch_by_val_f1": best_epoch_by_f1,
        "best_val_f1": best_val_f1,
        "val_loss_at_f1_epoch": val_loss_at_f1_epoch,
        "selection_disagrees": selection_disagrees,
        "epochs_run": len(epoch_rows),
        "epoch_curve": epoch_rows,
        "checkpoint_val_loss": str(ckpt_loss_dir),
        "checkpoint_val_f1": str(ckpt_f1_dir),
    }

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"  done: best_loss_epoch={best_epoch_by_loss} best_f1_epoch={best_epoch_by_f1} "
        f"disagree={selection_disagrees}"
    )
    return payload
