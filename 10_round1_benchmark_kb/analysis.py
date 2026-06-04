"""Round 1 analyses: mean-level (weaker) and seed-level variance components (primary)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, t as student_t

from .config import BOOTSTRAP_N, MODEL_BY_ID, TRAIN_SEEDS

DEGENERATE_BENCHMARK_MAX = 1e-6
DEGENERATE_VAL_MAX = 1e-6

# DeBERTa training failures (Round 1): excluded from all primary metrics.
COLLAPSED_DEBERTA_SEEDS = frozenset({45, 49})


def is_collapsed_deberta(row: pd.Series) -> bool:
    return row["model_id"] == "deberta_base" and int(row["seed"]) in COLLAPSED_DEBERTA_SEEDS


def is_clean_run(row: pd.Series) -> bool:
    if is_collapsed_deberta(row):
        return False
    return float(row["benchmark_f1"]) > DEGENERATE_BENCHMARK_MAX and float(
        row.get("best_val_f1", 1)
    ) > DEGENERATE_VAL_MAX


def filter_clean_runs(per_run: pd.DataFrame) -> pd.DataFrame:
    mask = per_run.apply(is_clean_run, axis=1)
    return per_run.loc[mask].copy()


def filter_easy_hard_runs(easy_hard: pd.DataFrame, per_run_clean: pd.DataFrame) -> pd.DataFrame:
    """Drop collapsed DeBERTa seeds from easy/hard subset averages (distance ranker kept)."""
    clean_keys = set(zip(per_run_clean["model_id"].astype(str), per_run_clean["seed"].astype(int)))

    def keep(row: pd.Series) -> bool:
        if row["model_id"] == "distance_ranker":
            return True
        return (str(row["model_id"]), int(row["seed"])) in clean_keys

    return easy_hard.loc[easy_hard.apply(keep, axis=1)].copy()


def print_deberta_kb_audit(per_run_all: pd.DataFrame) -> None:
    """Stdout audit: per-seed KB MRR and clean means (blocking data check)."""
    sub = per_run_all[per_run_all["model_id"] == "deberta_base"].sort_values("seed")
    print("\n=== DeBERTa per-seed KB MRR (stored results, all 8 seeds) ===")
    for _, r in sub.iterrows():
        seed = int(r["seed"])
        clean = is_clean_run(r)
        flag = "PRIMARY" if clean else "EXCLUDED (collapsed)"
        print(
            f"  seed {seed:2d} [{flag}]  benchmark_f1={float(r['benchmark_f1']):.4f}  "
            f"kb_gene_drug={float(r['kb_mrr_gene_drug']):.4f}  "
            f"kb_gene_disease={float(r['kb_mrr_gene_disease']):.4f}  ece={float(r['ece']):.4f}"
        )
    clean = filter_clean_runs(sub)
    print("\n=== DeBERTa clean-seed means (seeds 42,43,44,46,47,48 only) ===")
    print(
        f"  kb_gene_drug={clean['kb_mrr_gene_drug'].mean():.4f}  "
        f"kb_gene_disease={clean['kb_mrr_gene_disease'].mean():.4f}  "
        f"benchmark_f1={clean['benchmark_f1'].mean():.4f}  ece={clean['ece'].mean():.4f}"
    )
    print(
        f"  benchmark_f1 if all 8 seeds averaged (sensitivity)={sub['benchmark_f1'].mean():.4f}"
    )
    collapsed = sub[sub.apply(is_collapsed_deberta, axis=1)]
    if not collapsed.empty:
        print(
            "  Collapsed seeds carry non-missing KB MRR (~0.54) but are excluded; "
            "they do not inflate clean means (~0.68)."
        )


def flag_degenerate_runs(per_run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, r in per_run_df.iterrows():
        reasons: list[str] = []
        if float(r.get("best_val_f1", 1)) <= DEGENERATE_VAL_MAX:
            reasons.append("val_f1_zero")
        if float(r.get("benchmark_f1", 1)) <= DEGENERATE_BENCHMARK_MAX:
            reasons.append("benchmark_f1_zero")
        if reasons:
            rows.append(
                {
                    "model_id": r["model_id"],
                    "seed": int(r["seed"]),
                    "run_id": r.get("run_id"),
                    "best_val_f1": float(r.get("best_val_f1", np.nan)),
                    "benchmark_f1": float(r.get("benchmark_f1", np.nan)),
                    "flags": ",".join(reasons),
                }
            )
    return pd.DataFrame(rows)


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


def encoder_summary_seed_bootstrap(
    per_run: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-encoder mean and 95% CI from bootstrap over seeds (primary uncertainty)."""
    rng = np.random.default_rng(seed)
    metrics = ["benchmark_f1", "kb_mrr_gene_drug", "kb_mrr_gene_disease", "ece"]
    rows: list[dict] = []

    for model_id, sub in per_run.groupby("model_id"):
        spec = MODEL_BY_ID.get(model_id)
        row: dict[str, Any] = {
            "model_id": model_id,
            "short_name": spec.short_name if spec else model_id,
            "architecture": spec.architecture if spec else "",
            "n_seeds": len(sub),
        }
        for col in metrics:
            if col not in sub.columns:
                continue
            vals = sub[col].astype(float).values
            if len(vals) == 0:
                continue
            mean = float(np.mean(vals))
            if len(vals) == 1:
                row[f"{col}_mean"] = mean
                row[f"{col}_ci_lo"] = mean
                row[f"{col}_ci_hi"] = mean
                continue
            boots = [float(np.mean(rng.choice(vals, size=len(vals), replace=True))) for _ in range(n_boot)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            row[f"{col}_mean"] = mean
            row[f"{col}_ci_lo"] = float(lo)
            row[f"{col}_ci_hi"] = float(hi)
            if len(vals) >= 2:
                se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                tcrit = float(student_t.ppf(0.975, len(vals) - 1))
                row[f"{col}_t_ci_lo"] = mean - tcrit * se
                row[f"{col}_t_ci_hi"] = mean + tcrit * se
        rows.append(row)
    return pd.DataFrame(rows)


def encoder_summary(per_run_df: pd.DataFrame) -> pd.DataFrame:
    """Legacy encoder mean with normal-approximation CI (kept for mean-level analysis)."""
    rows: list[dict] = []
    for model_id, sub in per_run_df.groupby("model_id"):
        spec = MODEL_BY_ID.get(model_id)
        row: dict[str, Any] = {
            "model_id": model_id,
            "short_name": spec.short_name if spec else model_id,
            "n_seeds": len(sub),
        }
        for col in ["benchmark_f1", "kb_mrr_gene_drug", "kb_mrr_gene_disease", "ece"]:
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


def variance_components_icc(per_run: pd.DataFrame, metric: str) -> dict[str, Any]:
    """
    Decompose total variance into between-encoder and within-encoder (seed) shares.
    ICC-style ratio: between / (between + within).
    """
    if metric not in per_run.columns:
        raise ValueError(f"Missing metric {metric}")

    groups = per_run.groupby("model_id")[metric]
    y = per_run[metric].astype(float).values
    grand = float(np.mean(y))
    ss_total = float(np.sum((y - grand) ** 2))
    if ss_total <= 0:
        return {
            "metric": metric,
            "icc": 0.0,
            "encoder_variance_share": 0.0,
            "seed_variance_share": 0.0,
            "between_encoder_sd": 0.0,
            "mean_within_encoder_sd": 0.0,
            "n_runs": len(y),
            "n_encoders": per_run["model_id"].nunique(),
        }

    ss_between = 0.0
    for _, g in groups:
        m = float(g.mean())
        ss_between += len(g) * (m - grand) ** 2
    ss_within = ss_total - ss_between

    encoder_share = ss_between / ss_total
    seed_share = ss_within / ss_total

    encoder_means = groups.mean()
    within_sds = groups.std(ddof=1)
    between_var = float(encoder_means.var(ddof=1)) if len(encoder_means) > 1 else 0.0
    within_var = float(within_sds.mean() ** 2) if len(within_sds) else 0.0
    denom = between_var + within_var
    icc = float(between_var / denom) if denom > 0 else 0.0

    return {
        "metric": metric,
        "icc": icc,
        "encoder_variance_share": float(encoder_share),
        "seed_variance_share": float(seed_share),
        "between_encoder_sd": float(encoder_means.std(ddof=1)) if len(encoder_means) > 1 else 0.0,
        "mean_within_encoder_sd": float(within_sds.mean()) if len(within_sds) else 0.0,
        "n_runs": int(len(y)),
        "n_encoders": int(per_run["model_id"].nunique()),
    }


def variance_components_table(per_run: pd.DataFrame) -> pd.DataFrame:
    metrics = ["benchmark_f1", "kb_mrr_gene_drug", "kb_mrr_gene_disease", "ece"]
    return pd.DataFrame([variance_components_icc(per_run, m) for m in metrics if m in per_run.columns])


def cluster_bootstrap_benchmark_kb(
    per_run: pd.DataFrame,
    kb_col: str,
    pair_type: str,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 42,
) -> dict[str, Any]:
    """Seed-level Spearman with cluster bootstrap over encoders."""
    rng = np.random.default_rng(seed)
    encoders = per_run["model_id"].unique()
    x = per_run["benchmark_f1"].astype(float).values
    y = per_run[kb_col].astype(float).values
    r_obs, _ = spearmanr(x, y)

    boots: list[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(encoders, size=len(encoders), replace=True)
        parts = [per_run[per_run["model_id"] == e] for e in chosen]
        bdf = pd.concat(parts, ignore_index=True)
        if bdf["benchmark_f1"].nunique() < 2 or bdf[kb_col].nunique() < 2:
            continue
        r, _ = spearmanr(bdf["benchmark_f1"], bdf[kb_col])
        if not np.isnan(r):
            boots.append(float(r))

    if not boots:
        return {
            "pair_type": pair_type,
            "method": "seed_level_cluster_bootstrap",
            "spearman": float(r_obs) if not np.isnan(r_obs) else None,
            "ci_lo": None,
            "ci_hi": None,
            "n_runs": len(per_run),
            "n_encoders": len(encoders),
        }

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "pair_type": pair_type,
        "method": "seed_level_cluster_bootstrap",
        "spearman": float(r_obs) if not np.isnan(r_obs) else None,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_runs": len(per_run),
        "n_encoders": len(encoders),
    }


def mean_level_correlations(
    encoder_df: pd.DataFrame,
    analysis_set: str,
) -> pd.DataFrame:
    """Nine encoder means: Spearman/Pearson vs KB MRR (weaker approach)."""
    rows: list[dict] = []
    for pt, col in [
        ("gene-drug", "kb_mrr_gene_drug_mean"),
        ("gene-disease", "kb_mrr_gene_disease_mean"),
    ]:
        if col not in encoder_df.columns:
            continue
        x = encoder_df["benchmark_f1_mean"].astype(float).values
        y = encoder_df[col].astype(float).values
        for metric_name, fn in [("spearman", spearmanr), ("pearson", pearsonr)]:
            res = _bootstrap_ci(x, y, fn)
            rows.append(
                {
                    "analysis_set": analysis_set,
                    "method": "encoder_mean_n9",
                    "pair_type": pt,
                    "metric": metric_name,
                    "estimate": res["estimate"],
                    "ci_lo": res["ci_lo"],
                    "ci_hi": res["ci_hi"],
                    "n": res["n"],
                }
            )
    return pd.DataFrame(rows)


def benchmark_ece_correlation(encoder_df: pd.DataFrame, analysis_set: str = "primary") -> pd.DataFrame:
    x = encoder_df["benchmark_f1_mean"].astype(float).values
    y = encoder_df["ece_mean"].astype(float).values
    rows = []
    for metric_name, fn in [("spearman", spearmanr), ("pearson", pearsonr)]:
        res = _bootstrap_ci(x, y, fn)
        rows.append(
            {
                "analysis_set": analysis_set,
                "method": "encoder_mean_n9",
                "pair_type": "calibration",
                "metric": metric_name,
                "estimate": res["estimate"],
                "ci_lo": res["ci_lo"],
                "ci_hi": res["ci_hi"],
                "n": res["n"],
            }
        )
    return pd.DataFrame(rows)


def collapsed_seed_sensitivity(
    per_run_all: pd.DataFrame,
    per_run_clean: pd.DataFrame,
) -> pd.DataFrame:
    """Compare mean-level and seed-level headline stats: primary vs including collapsed seeds."""
    rows: list[dict] = []

    enc_all = encoder_summary(per_run_all)
    enc_clean = encoder_summary(per_run_clean)
    vc_all = variance_components_table(per_run_all)
    vc_clean = variance_components_table(per_run_clean)

    for label, enc, vc, pr in [
        ("primary_exclude_collapsed", enc_clean, vc_clean, per_run_clean),
        ("sensitivity_include_collapsed", enc_all, vc_all, per_run_all),
    ]:
        for pt, col in [("gene-drug", "kb_mrr_gene_drug"), ("gene-disease", "kb_mrr_gene_disease")]:
            mean_corr = mean_level_correlations(enc, label)
            sp_row = mean_corr[(mean_corr["pair_type"] == pt) & (mean_corr["metric"] == "spearman")]
            seed_assoc = cluster_bootstrap_benchmark_kb(pr, col, pt)
            vc_row = vc[vc["metric"] == col]
            rows.append(
                {
                    "analysis_set": label,
                    "pair_type": pt,
                    "mean_level_spearman": sp_row["estimate"].iloc[0] if len(sp_row) else None,
                    "mean_level_ci_lo": sp_row["ci_lo"].iloc[0] if len(sp_row) else None,
                    "mean_level_ci_hi": sp_row["ci_hi"].iloc[0] if len(sp_row) else None,
                    "seed_level_spearman": seed_assoc["spearman"],
                    "seed_level_ci_lo": seed_assoc["ci_lo"],
                    "seed_level_ci_hi": seed_assoc["ci_hi"],
                    "encoder_variance_share": vc_row["encoder_variance_share"].iloc[0] if len(vc_row) else None,
                    "seed_variance_share": vc_row["seed_variance_share"].iloc[0] if len(vc_row) else None,
                    "icc": vc_row["icc"].iloc[0] if len(vc_row) else None,
                }
            )
        deberta = enc[enc["model_id"] == "deberta_base"]
        if not deberta.empty:
            rows.append(
                {
                    "analysis_set": label,
                    "pair_type": "deberta_benchmark_f1_mean",
                    "mean_level_spearman": None,
                    "mean_level_ci_lo": None,
                    "mean_level_ci_hi": None,
                    "seed_level_spearman": None,
                    "seed_level_ci_lo": None,
                    "seed_level_ci_hi": None,
                    "encoder_variance_share": float(deberta["benchmark_f1_mean"].iloc[0]),
                    "seed_variance_share": None,
                    "icc": float(deberta["n_seeds"].iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def benchmark_f1_range_check(encoder_df: pd.DataFrame) -> dict[str, Any]:
    vals = encoder_df["benchmark_f1_mean"].astype(float)
    spread = float(vals.max() - vals.min())
    encoder_f1_values = sorted(
        zip(encoder_df["short_name"], vals),
        key=lambda x: x[1],
        reverse=True,
    )
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


def seed_level_association_table(per_run: pd.DataFrame, analysis_set: str) -> pd.DataFrame:
    rows = []
    for pt, col in [("gene-drug", "kb_mrr_gene_drug"), ("gene-disease", "kb_mrr_gene_disease")]:
        if col not in per_run.columns:
            continue
        res = cluster_bootstrap_benchmark_kb(per_run, col, pt)
        rows.append({**res, "analysis_set": analysis_set})
    return pd.DataFrame(rows)
