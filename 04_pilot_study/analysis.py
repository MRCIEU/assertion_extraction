"""Benchmark vs KB-ranking decoupling analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import MODELS, MODEL_BY_ID


def build_benchmark_kb_table(
    ranking_df: pd.DataFrame,
    kb_metric: str = "mrr",
) -> pd.DataFrame:
    """Merge known benchmark F1 with KB ranking metric; assign ranks."""
    meta = pd.DataFrame(
        [
            {
                "model_id": m.model_id,
                "short_name": m.short_name,
                "benchmark_name": m.benchmark_name,
                "benchmark_f1": m.benchmark_f1,
                "benchmark_source": m.benchmark_source,
            }
            for m in MODELS
        ]
    )
    merged = meta.merge(ranking_df, on="model_id", how="inner")
    merged["benchmark_rank"] = merged["benchmark_f1"].rank(ascending=False, method="min").astype(int)
    merged["kb_rank"] = merged[kb_metric].rank(ascending=False, method="min").astype(int)
    merged["rank_delta"] = merged["benchmark_rank"] - merged["kb_rank"]
    return merged.sort_values("benchmark_rank")


def rank_flips(table: pd.DataFrame) -> pd.DataFrame:
    """Concrete examples where benchmark and KB order disagree."""
    flips = table[table["rank_delta"] != 0].copy()
    flips = flips.sort_values("rank_delta", key=abs, ascending=False)
    return flips[
        [
            "short_name",
            "benchmark_f1",
            "benchmark_rank",
            "mrr",
            "auc_pr",
            "kb_rank",
            "rank_delta",
        ]
    ]


def decoupling_summary(table: pd.DataFrame, kb_metric: str = "mrr") -> dict[str, Any]:
    rho, pval = spearmanr(table["benchmark_f1"], table[kb_metric])
    n_flip = int((table["rank_delta"] != 0).sum())
    decoupled = n_flip >= 2 or (rho is not np.nan and rho < 0.5)
    return {
        "spearman_rho": round(float(rho), 4) if not np.isnan(rho) else None,
        "spearman_pvalue": round(float(pval), 6) if not np.isnan(pval) else None,
        "n_rank_flips": n_flip,
        "decoupled": bool(decoupled),
        "kb_metric_used": kb_metric,
    }


def calibration_decoupling(
    bench_table: pd.DataFrame,
    calibration_df: pd.DataFrame,
) -> dict[str, Any]:
    merged = bench_table.merge(calibration_df, on="model_id", how="inner")
    merged["calibration_rank"] = merged["ece"].rank(ascending=True, method="min").astype(int)
    rho, pval = spearmanr(merged["benchmark_f1"], merged["ece"])
    # Lower ECE = better calibration; expect negative rho if benchmark predicts calibration
    return {
        "spearman_rho_benchmark_vs_ece": round(float(rho), 4) if not np.isnan(rho) else None,
        "spearman_pvalue": round(float(pval), 6) if not np.isnan(pval) else None,
        "ece_spread": round(float(merged["ece"].max() - merged["ece"].min()), 4),
        "decoupled": bool(rho is np.nan or abs(rho) < 0.7),
        "table": merged[
            ["short_name", "benchmark_f1", "benchmark_rank", "ece", "calibration_rank"]
        ].sort_values("benchmark_rank"),
    }
