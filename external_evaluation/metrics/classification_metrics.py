# -*- coding: utf-8 -*-
"""Sklearn-based metrics for multi-class relation classification."""

from __future__ import annotations

import json
from typing import Any

try:
    from sklearn.metrics import precision_recall_fscore_support
except ImportError as e:  # pragma: no cover
    raise ImportError("external_evaluation requires scikit-learn") from e


def compute_tier1_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    """Macro / micro / per-class support; labels order fixed."""
    # Restrict to labels present in y_true or y_pred for sklearn stability
    present = sorted(set(y_true) | set(y_pred))
    lab = [x for x in labels if x in present] or present

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=lab, average="macro", zero_division=0
    )
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=lab, average="micro", zero_division=0
    )
    p_w, r_w, f_w, sup = precision_recall_fscore_support(y_true, y_pred, labels=lab, zero_division=0)

    per_class = [
        {
            "label": lab[i],
            "precision": float(p_w[i]),
            "recall": float(r_w[i]),
            "f1": float(f_w[i]),
            "support": int(sup[i]),
        }
        for i in range(len(lab))
    ]
    return {
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f_macro),
        "micro_precision": float(p_micro),
        "micro_recall": float(r_micro),
        "micro_f1": float(f_micro),
        "support_total": len(y_true),
        "per_class_json": json.dumps(per_class),
    }


def error_taxonomy_counts(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Coarse error buckets for Layer D."""
    n = len(y_true)
    if n == 0:
        return {}
    fp_inf = sum(1 for t, p in zip(y_true, y_pred) if t == "__NEGATIVE__" and p != "__NEGATIVE__")
    fn_conc = sum(1 for t, p in zip(y_true, y_pred) if t != "__NEGATIVE__" and p == "__NEGATIVE__")
    pred_dist: dict[str, int] = {}
    for p in y_pred:
        pred_dist[p] = pred_dist.get(p, 0) + 1
    maj = max(pred_dist.values()) if pred_dist else 0
    majority_collapse = 1.0 if maj / n >= 0.92 and n >= 16 else 0.0
    rare_fail = sum(
        1
        for t, p in zip(y_true, y_pred)
        if t not in ("__NEGATIVE__", "ASSOCIATION_GENERAL") and p != t
    )
    return {
        "rate_false_positive_inflation": round(fp_inf / n, 4),
        "rate_false_negative_concentration": round(fn_conc / n, 4),
        "flag_majority_class_collapse": majority_collapse,
        "count_rare_family_mismatch": rare_fail,
    }
