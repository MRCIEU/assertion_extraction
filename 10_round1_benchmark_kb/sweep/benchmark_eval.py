"""Benchmark F1 evaluation for sweep checkpoints (BioRED test only — no KB)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from importlib import import_module

_be = import_module("10_round1_benchmark_kb.benchmark_eval")
build_biored_test_examples = _be.build_biored_test_examples
RelationDataset = import_module("10_round1_benchmark_kb.train").RelationDataset
_require_gpu = import_module("10_round1_benchmark_kb.train")._require_gpu

from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score, precision_recall_fscore_support

from importlib import import_module as im

_r1 = im("10_round1_benchmark_kb.config")
INFER_BATCH_SIZE = _r1.INFER_BATCH_SIZE
MAX_SEQ_LENGTH = _r1.MAX_SEQ_LENGTH


def evaluate_checkpoint_benchmark_f1(ckpt_dir: Path, test_examples: list[dict] | None = None) -> dict[str, float]:
    if test_examples is None:
        test_examples = build_biored_test_examples()

    device = _require_gpu()
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    model.to(device)
    model.eval()

    ds = RelationDataset(test_examples, tokenizer, MAX_SEQ_LENGTH)
    loader = DataLoader(ds, batch_size=INFER_BATCH_SIZE, shuffle=False)

    preds: list[int] = []
    labels: list[int] = []
    import torch

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            pred = logits.argmax(dim=-1).cpu().numpy()
            preds.extend(pred.tolist())
            labels.extend(batch["labels"].cpu().numpy().tolist())

    f1 = float(f1_score(labels, preds, average="binary", zero_division=0))
    prec, rec, _, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "benchmark_f1": f1,
        "benchmark_precision": float(prec),
        "benchmark_recall": float(rec),
    }


def add_benchmark_scores(payload: dict[str, Any], test_examples: list[dict] | None = None) -> dict[str, Any]:
    loss_ckpt = Path(payload["checkpoint_val_loss"])
    f1_ckpt = Path(payload["checkpoint_val_f1"])

    loss_scores = evaluate_checkpoint_benchmark_f1(loss_ckpt, test_examples)
    payload["benchmark_f1_val_loss_ckpt"] = loss_scores["benchmark_f1"]
    payload["benchmark_precision_val_loss_ckpt"] = loss_scores["benchmark_precision"]
    payload["benchmark_recall_val_loss_ckpt"] = loss_scores["benchmark_recall"]

    if payload.get("selection_disagrees"):
        f1_scores = evaluate_checkpoint_benchmark_f1(f1_ckpt, test_examples)
        payload["benchmark_f1_val_f1_ckpt"] = f1_scores["benchmark_f1"]
        payload["benchmark_precision_val_f1_ckpt"] = f1_scores["benchmark_precision"]
        payload["benchmark_recall_val_f1_ckpt"] = f1_scores["benchmark_recall"]
    else:
        payload["benchmark_f1_val_f1_ckpt"] = loss_scores["benchmark_f1"]

    print(
        f"  benchmark F1 (val_loss ckpt): {payload['benchmark_f1_val_loss_ckpt']:.4f} "
        f"| val_f1 ckpt: {payload.get('benchmark_f1_val_f1_ckpt', payload['benchmark_f1_val_loss_ckpt']):.4f}"
    )
    return payload
