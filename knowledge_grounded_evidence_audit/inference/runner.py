# -*- coding: utf-8 -*-
"""
Optional batch metrics when gold labels exist (external-eval compatible API).
All comments in English.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

from inference.predict_checkpoint import predict_labels


def evaluate_checkpoint(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
    return {"status": "not_implemented", "note": "unused_stub"}


def evaluate_rows_on_loaded_model(
    model: Any,
    tokenizer: Any,
    label2id: Dict[str, int],
    _labels_ordered: List[str],
    _lr_placeholder: Any,
    checkpoint_path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    device: torch.device,
    max_length: int = 384,
    batch_size: int = 8,
) -> Dict[str, Any]:
    if not rows:
        return {"status": "no_rows", "checkpoint": str(checkpoint_path)}

    preds, _confs = predict_labels(
        model,
        tokenizer,
        label2id,
        list(rows),
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    gold = [r.get("label") for r in rows]
    if all(g is not None and g != "" for g in gold):
        labels_all = sorted(label2id.keys(), key=lambda x: label2id[x])
        macro_f1 = f1_score(gold, preds, average="macro", labels=labels_all, zero_division=0)
        p, r, _, _ = precision_recall_fscore_support(
            gold, preds, labels=labels_all, average=None, zero_division=0
        )
        macro_p = float(sum(p) / max(len(p), 1))
        macro_r = float(sum(r) / max(len(r), 1))
        return {
            "status": "ok",
            "checkpoint": str(checkpoint_path),
            "macro_precision": round(macro_p, 6),
            "macro_recall": round(macro_r, 6),
            "macro_f1": round(float(macro_f1), 6),
            "n_examples": len(rows),
        }

    return {
        "status": "ok_no_gold",
        "checkpoint": str(checkpoint_path),
        "n_examples": len(rows),
        "note": "labels_absent_skipped_supervised_metrics",
    }
