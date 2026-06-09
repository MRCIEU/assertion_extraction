"""Load and summarize step-2 matrix completion markers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from shared.constants import DEGENERATE_BENCHMARK_F1_MAX, TRAIN_SEEDS
from shared.models import MODELS

from .config import MATRIX_DATA, matrix_result_path, require_chosen_recipe

DEBERTA_COLLAPSE_F1 = 0.05
DEBERTA_SUPPRESSION_GAP = 0.05
DEBERTA_MODEL_ID = "deberta_base"


def load_matrix_table() -> pd.DataFrame:
    recipe = require_chosen_recipe()
    rows: list[dict] = []
    for spec in MODELS:
        for seed in TRAIN_SEEDS:
            path = matrix_result_path(spec.model_id, seed)
            if not path.exists():
                rows.append(
                    {
                        "model_id": spec.model_id,
                        "short_name": spec.short_name,
                        "seed": seed,
                        "missing": True,
                    }
                )
                continue
            m = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model_id": spec.model_id,
                    "short_name": spec.short_name,
                    "seed": seed,
                    "missing": False,
                    "recipe_lr": float(m.get("recipe_lr", 0)),
                    "recipe_warmup_label": m.get("recipe_warmup_label", ""),
                    "training_strategy": m.get("training_strategy", ""),
                    "benchmark_f1": float(m.get("benchmark_f1", 0)),
                    "benchmark_precision": float(m.get("benchmark_precision", 0)),
                    "benchmark_recall": float(m.get("benchmark_recall", 0)),
                    "best_val_f1": float(m.get("best_val_f1", 0)),
                    "best_epoch_val_f1": int(m.get("best_epoch_val_f1", 0)),
                    "n_epochs_run": len(m.get("epoch_curve", [])),
                    "degenerate": bool(m.get("degenerate", False)),
                    "collapsed": float(m.get("benchmark_f1", 0)) <= DEGENERATE_BENCHMARK_F1_MAX,
                    "completed_at": m.get("completed_at", ""),
                }
            )
    df = pd.DataFrame(rows)
    df["expected_recipe_lr"] = recipe.lr
    df["recipe_match"] = np.isclose(df.get("recipe_lr", np.nan), recipe.lr, rtol=0, atol=1e-12)
    return df


def encoder_summary(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[~df["missing"].astype(bool)].copy()
    rows: list[dict] = []
    for spec in MODELS:
        sub = ok[ok["model_id"] == spec.model_id]
        if sub.empty:
            continue
        f1 = sub["benchmark_f1"].astype(float)
        rows.append(
            {
                "model_id": spec.model_id,
                "short_name": spec.short_name,
                "n_runs": len(sub),
                "benchmark_f1_mean": float(f1.mean()),
                "benchmark_f1_std": float(f1.std(ddof=0)),
                "benchmark_f1_min": float(f1.min()),
                "benchmark_f1_max": float(f1.max()),
                "n_collapsed": int(sub["collapsed"].sum()),
                "best_val_f1_mean": float(sub["best_val_f1"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("benchmark_f1_mean", ascending=False)


def deberta_gate_verdict(df: pd.DataFrame, enc_summary: pd.DataFrame) -> tuple[bool, list[str]]:
    deb = df[(df["model_id"] == DEBERTA_MODEL_ID) & ~df["missing"].astype(bool)]
    deberta_f1s = {int(r["seed"]): float(r["benchmark_f1"]) for _, r in deb.iterrows()}
    reasons: list[str] = []

    collapsed = [s for s, f1 in deberta_f1s.items() if f1 <= DEBERTA_COLLAPSE_F1]
    if collapsed:
        reasons.append(
            f"DeBERTa collapsed (benchmark F1 <= {DEBERTA_COLLAPSE_F1}) on seeds {collapsed}"
        )

    deb_mean = float(deb["benchmark_f1"].mean()) if not deb.empty else 0.0
    peers = enc_summary[enc_summary["model_id"] != DEBERTA_MODEL_ID]
    peer_mean = float(peers["benchmark_f1_mean"].mean()) if not peers.empty else deb_mean
    gap = peer_mean - deb_mean
    if gap > DEBERTA_SUPPRESSION_GAP and deb_mean < peer_mean:
        reasons.append(
            f"DeBERTa seed mean {deb_mean:.3f} is {gap:.3f} below peer-encoder mean "
            f"{peer_mean:.3f} (threshold {DEBERTA_SUPPRESSION_GAP})"
        )

    return (not reasons), reasons


def matrix_footprint_gib() -> float:
    if not MATRIX_DATA.exists():
        return 0.0
    total = sum(f.stat().st_size for f in MATRIX_DATA.rglob("*") if f.is_file())
    return total / (1024**3)
