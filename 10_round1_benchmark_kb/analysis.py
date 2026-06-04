"""Statistical analyses A–D with bootstrap confidence intervals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .config import BOOTSTRAP_N, MODEL_BY_ID, PAIR_TYPES


def _bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat_fn,
    n_boot: int = BOOTSTRAP_N,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float | None]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 3:
        r, _ = stat_fn(x, y)
        return {"estimate": float(r) if not np.isnan(r) else None, "ci_lo": None, "ci_hi": None, "n": n}

    stats: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bx, by = x[idx], y[idx]
        if len(np.unique(bx)) < 2 or len(np.unique(by)) < 2:
            continue
        r, _ = stat_fn(bx, by)
        if not np.isnan(r):
            stats.append(float(r))

    r_obs, p_obs = stat_fn(x, y)
    if not stats:
        return {
            "estimate": float(r_obs) if not np.isnan(r_obs) else None,
            "p_value": float(p_obs) if not np.isnan(p_obs) else None,
            "ci_lo": None,
            "ci_hi": None,
            "n": n,
        }

    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "estimate": float(r_obs) if not np.isnan(r_obs) else None,
        "p_value": float(p_obs) if not np.isnan(p_obs) else None,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": n,
    }


def encoder_summary(per_run_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seed-level results to encoder means with 95% CIs."""
    rows: list[dict] = []
    for model_id, sub in per_run_df.groupby("model_id"):
        spec = MODEL_BY_ID.get(model_id)
        row: dict[str, Any] = {
            "model_id": model_id,
            "short_name": spec.short_name if spec else model_id,
            "architecture": spec.architecture if spec else "",
            "n_seeds": len(sub),
        }
        for col in [
            "benchmark_f1",
            "kb_mrr_gene_drug",
            "kb_mrr_gene_disease",
            "kb_mrr_overall",
            "ece",
        ]:
            if col not in sub.columns:
                continue
            vals = sub[col].dropna().astype(float)
            if len(vals) == 0:
                continue
            mean = float(vals.mean())
            se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            ci = 1.96 * se
            row[f"{col}_mean"] = mean
            row[f"{col}_ci_lo"] = mean - ci
            row[f"{col}_ci_hi"] = mean + ci
        rows.append(row)
    return pd.DataFrame(rows)


def benchmark_kb_correlation(
    encoder_df: pd.DataFrame,
    kb_col: str,
    pair_type: str,
) -> dict[str, Any]:
    x = encoder_df["benchmark_f1_mean"].astype(float).values
    y = encoder_df[kb_col].astype(float).values
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    spearman = _bootstrap_ci(x, y, spearmanr)
    pearson = _bootstrap_ci(x, y, pearsonr)

    table = encoder_df[["short_name", "benchmark_f1_mean", kb_col]].copy()
    table["benchmark_rank"] = table["benchmark_f1_mean"].rank(ascending=False, method="min").astype(int)
    table["kb_rank"] = table[kb_col].rank(ascending=False, method="min").astype(int)
    table["rank_delta"] = table["benchmark_rank"] - table["kb_rank"]
    flips = table[table["rank_delta"] != 0].sort_values("rank_delta", key=abs, ascending=False)

    return {
        "pair_type": pair_type,
        "kb_metric": kb_col,
        "spearman": spearman,
        "pearson": pearson,
        "rank_flips": flips,
        "n_encoders": int(len(x)),
    }


def benchmark_ece_correlation(encoder_df: pd.DataFrame) -> dict[str, Any]:
    x = encoder_df["benchmark_f1_mean"].astype(float).values
    y = encoder_df["ece_mean"].astype(float).values
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    spearman = _bootstrap_ci(x, y, spearmanr)
    pearson = _bootstrap_ci(x, y, pearsonr)
    return {"spearman": spearman, "pearson": pearson, "n_encoders": int(len(x))}


def flag_degenerate_runs(per_run_df: pd.DataFrame) -> pd.DataFrame:
    """Runs with failed validation (val F1=0) or collapsed benchmark F1."""
    rows: list[dict] = []
    for _, r in per_run_df.iterrows():
        reasons: list[str] = []
        if float(r.get("best_val_f1", 1)) <= 0.0:
            reasons.append("val_f1_zero")
        if float(r.get("benchmark_f1", 1)) <= 0.0:
            reasons.append("benchmark_f1_zero")
        if reasons:
            rows.append(
                {
                    "model_id": r["model_id"],
                    "seed": int(r["seed"]),
                    "run_id": r.get("run_id"),
                    "best_val_f1": float(r.get("best_val_f1", np.nan)),
                    "benchmark_f1": float(r.get("benchmark_f1", np.nan)),
                    "kb_mrr_overall": float(r.get("kb_mrr_overall", np.nan)),
                    "flags": ",".join(reasons),
                }
            )
    return pd.DataFrame(rows)


def encoder_summary_robust(per_run_df: pd.DataFrame) -> pd.DataFrame:
    """Encoder means excluding degenerate runs (for sensitivity analysis)."""
    clean = per_run_df[
        (per_run_df["benchmark_f1"].astype(float) > 0)
        & (per_run_df["best_val_f1"].astype(float) > 0)
    ]
    return encoder_summary(clean)


def encoder_vs_seed_noise(per_run_df: pd.DataFrame) -> pd.DataFrame:
    """Compare between-encoder spread to within-encoder seed SD for key metrics."""
    rows: list[dict] = []
    for metric in [
        "benchmark_f1",
        "kb_mrr_gene_drug",
        "kb_mrr_gene_disease",
        "kb_mrr_overall",
        "ece",
    ]:
        if metric not in per_run_df.columns:
            continue
        within = per_run_df.groupby("model_id")[metric].std(ddof=1).mean()
        between = per_run_df.groupby("model_id")[metric].mean().std(ddof=1)
        rows.append(
            {
                "metric": metric,
                "mean_within_encoder_sd": float(within) if not np.isnan(within) else 0.0,
                "between_encoder_sd": float(between) if not np.isnan(between) else 0.0,
                "between_exceeds_within": bool(between > within) if not (np.isnan(between) or np.isnan(within)) else None,
            }
        )
    return pd.DataFrame(rows)


def easy_hard_encoder_summary(subset_df: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged MRR on easy/hard subsets vs distance ranker baseline."""
    rows: list[dict] = []
    for subset_key, label in [
        ("easy_co_sentence", "easy"),
        ("hard_cross_sentence", "hard"),
    ]:
        sub = subset_df[subset_df["subset"] == subset_key]
        if sub.empty:
            continue
        dr_row = sub[sub["model_id"] == "distance_ranker"]
        dr_mrr = float(dr_row["mrr"].iloc[0]) if not dr_row.empty else np.nan
        models = sub[sub["model_id"] != "distance_ranker"].groupby("model_id")["mrr"].mean()
        for model_id, mrr in models.items():
            spec = MODEL_BY_ID.get(model_id)
            rows.append(
                {
                    "subset": label,
                    "model_id": model_id,
                    "short_name": spec.short_name if spec else model_id,
                    "mrr_mean": float(mrr),
                    "distance_ranker_mrr": dr_mrr,
                    "beats_distance_ranker": bool(mrr > dr_mrr) if not np.isnan(dr_mrr) else None,
                }
            )
    return pd.DataFrame(rows)


def sensitivity_correlations(
    encoder_df: pd.DataFrame,
    per_run_df: pd.DataFrame,
) -> list[dict]:
    """Benchmark–KB correlations on all encoders vs excluding degenerate runs."""
    rows: list[dict] = []
    robust = encoder_summary_robust(per_run_df)
    for label, edf in [("all_encoders", encoder_df), ("exclude_degenerate_runs", robust)]:
        if len(edf) < 3:
            continue
        for pt, col in [
            ("gene-drug", "kb_mrr_gene_drug_mean"),
            ("gene-disease", "kb_mrr_gene_disease_mean"),
        ]:
            if col not in edf.columns:
                continue
            res = benchmark_kb_correlation(edf, col, pt)
            rows.append(
                {
                    "analysis_set": label,
                    "pair_type": pt,
                    "metric": "spearman",
                    "estimate": res["spearman"].get("estimate"),
                    "ci_lo": res["spearman"].get("ci_lo"),
                    "ci_hi": res["spearman"].get("ci_hi"),
                    "n_encoders": res["n_encoders"],
                }
            )
    return rows


def benchmark_f1_range_check(encoder_df: pd.DataFrame) -> dict[str, Any]:
    vals = encoder_df["benchmark_f1_mean"].astype(float)
    spread = float(vals.max() - vals.min())
    encoder_f1_values = list(
        zip(encoder_df["short_name"], vals)
    )
    encoder_f1_values.sort(key=lambda x: x[1], reverse=True)
    return {
        "min_f1": float(vals.min()),
        "max_f1": float(vals.max()),
        "mean_f1": float(vals.mean()),
        "median_f1": float(vals.median()),
        "std_f1": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "spread": spread,
        "n_encoders": len(vals),
        "encoder_f1_values": encoder_f1_values,
    }


def rank_flip_table(encoder_df: pd.DataFrame, kb_col: str) -> pd.DataFrame:
    t = encoder_df[["short_name", "benchmark_f1_mean", kb_col]].copy()
    t["benchmark_rank"] = t["benchmark_f1_mean"].rank(ascending=False, method="min").astype(int)
    t["kb_rank"] = t[kb_col].rank(ascending=False, method="min").astype(int)
    t["rank_delta"] = t["benchmark_rank"] - t["kb_rank"]
    return t[t["rank_delta"] != 0].sort_values("rank_delta", key=abs, ascending=False)
