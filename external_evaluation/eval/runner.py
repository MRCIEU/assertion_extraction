# -*- coding: utf-8 -*-
"""Evaluation API: load HR checkpoints and score rows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from external_evaluation.eval.predict_checkpoint import (
    filter_rows_for_vocab,
    load_model_from_checkpoint,
    predict_labels,
)
from external_evaluation.metrics.classification_metrics import compute_tier1_metrics, error_taxonomy_counts

logger = logging.getLogger(__name__)


def load_model_from_best_pt(
    ckpt_path: Path,
    device: torch.device | None = None,
    *,
    state_dict_strict: bool = True,
) -> tuple[Any, Any, dict[str, int], list[str], dict[str, Any]]:
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_model_from_checkpoint(ckpt_path, dev, state_dict_strict=state_dict_strict)


def evaluate_rows_on_loaded_model(
    model: Any,
    tokenizer: Any,
    label2id: dict[str, int],
    labels: list[str],
    load_report: dict[str, Any],
    ckpt_path: Path,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    max_length: int = 384,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Score ``rows`` with an already-loaded model (no checkpoint I/O)."""
    kept, skipped = filter_rows_for_vocab(rows, label2id)
    if not kept:
        return {
            "status": "no_evaluable_rows",
            "skipped_oov_labels": skipped,
            "checkpoint": str(ckpt_path),
            "load_report": load_report,
        }

    y_true = [r["label"] for r in kept]
    preds, confs = predict_labels(
        model, tokenizer, label2id, kept, device=device, max_length=max_length, batch_size=batch_size
    )
    metrics = compute_tier1_metrics(y_true, preds, labels)
    hi = [i for i, c in enumerate(confs) if c >= 0.85]
    hcp = sum(1 for i in hi if preds[i] == y_true[i]) / len(hi) if hi else None
    et = error_taxonomy_counts(y_true, preds)

    return {
        "status": "ok",
        "checkpoint": str(ckpt_path),
        "load_report": load_report,
        "skipped_oov_labels": skipped,
        "support": len(y_true),
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
        "high_conf_precision": hcp,
        "mean_max_prob": float(sum(confs) / len(confs)) if confs else None,
        "error_taxonomy": et,
    }


def evaluate_checkpoint(
    ckpt_path: Path,
    rows: list[dict[str, Any]],
    *,
    max_length: int = 384,
    batch_size: int = 8,
    state_dict_strict: bool = True,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, tok, l2i, labels, load_report = load_model_from_checkpoint(
            ckpt_path, device, state_dict_strict=state_dict_strict
        )
    except Exception as e:
        logger.exception("Checkpoint load failed: %s", ckpt_path)
        return {"status": "load_failed", "error": str(e), "checkpoint": str(ckpt_path)}

    ev = evaluate_rows_on_loaded_model(
        model,
        tok,
        l2i,
        labels,
        load_report,
        ckpt_path,
        rows,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ev


def predict_pairs(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    label2id: dict[str, int],
    *,
    max_length: int = 384,
    batch_size: int = 8,
    device: torch.device,
) -> tuple[list[int], list[int], list[float]]:
    kept, _ = filter_rows_for_vocab(rows, label2id)
    preds, confs = predict_labels(
        model, tokenizer, label2id, kept, device=device, max_length=max_length, batch_size=batch_size
    )
    y_true = [label2id[r["label"]] for r in kept]
    y_pred = [label2id[p] for p in preds]
    return y_true, y_pred, confs
