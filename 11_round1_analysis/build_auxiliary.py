"""Build auxiliary CSVs: easy/hard ranking, distance confound, calibration, pool-size tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from shared.distance_analysis import (
    distance_ranker_subset_metrics,
    enrich_with_proximity,
    score_proximity_correlations,
    subset_ranking_metrics,
)
from shared.inference import load_scores_jsonl
from shared.metrics_calibration import calibration_baselines, calibration_for_scores
from shared.metrics_ranking import per_abstract_mrr
from shared.pool_loader import load_primary_candidates
from shared.pool_stats import save_abstract_pool_sizes

from .config import OUTPUT_DIR, POOL_SIZE_CSV, SCORES_DIR


def build_easy_hard_ranking(per_run: pd.DataFrame) -> pd.DataFrame:
    pool = enrich_with_proximity(load_primary_candidates())
    rows: list[dict] = []
    rows.extend(distance_ranker_subset_metrics(pool).to_dict("records"))

    for _, run in per_run.iterrows():
        score_path = SCORES_DIR / run["model_id"] / f"seed_{int(run['seed'])}.jsonl"
        if not score_path.exists():
            continue
        scores = load_scores_jsonl(score_path)
        sub = subset_ranking_metrics(scores, pool)
        rows.extend(sub.to_dict("records"))

    return pd.DataFrame(rows)


def build_distance_correlation(per_run: pd.DataFrame) -> pd.DataFrame:
    pool = enrich_with_proximity(load_primary_candidates())
    parts: list[pd.DataFrame] = []
    for _, run in per_run.iterrows():
        score_path = SCORES_DIR / run["model_id"] / f"seed_{int(run['seed'])}.jsonl"
        if not score_path.exists():
            continue
        scores = load_scores_jsonl(score_path)
        parts.append(score_proximity_correlations(scores, pool))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_calibration_tables(per_run: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = load_primary_candidates()
    baselines = calibration_baselines(pool)
    rows: list[dict] = []
    for _, run in per_run.iterrows():
        score_path = SCORES_DIR / run["model_id"] / f"seed_{int(run['seed'])}.jsonl"
        if not score_path.exists():
            continue
        scores = load_scores_jsonl(score_path)
        rows.append(calibration_for_scores(scores, run["run_id"]))
    enc = pd.DataFrame(rows)
    base_rows = [calibration_for_scores(baselines[k], k) for k in baselines]
    return enc, pd.DataFrame(base_rows)


def build_pool_size_robustness(per_run: pd.DataFrame) -> pd.DataFrame:
    """Per-run correlation between per-abstract MRR and candidate-pool size (Analysis E)."""
    pool = load_primary_candidates()
    save_abstract_pool_sizes(POOL_SIZE_CSV, pool)
    sizes = pool.groupby("pmid").size()

    rows: list[dict] = []
    for _, run in per_run.iterrows():
        score_path = SCORES_DIR / run["model_id"] / f"seed_{int(run['seed'])}.jsonl"
        if not score_path.exists():
            continue
        scores = load_scores_jsonl(score_path)
        for pair_type in ["gene-drug", "gene-disease"]:
            sub = scores[scores["pair_type"] == pair_type]
            mrr_by_pmid = per_abstract_mrr(sub)
            if mrr_by_pmid.empty:
                continue
            aligned = pd.DataFrame({"mrr": mrr_by_pmid})
            aligned["pool_size"] = aligned.index.map(lambda p: sizes.get(str(p), np.nan))
            aligned = aligned.dropna()
            if len(aligned) < 3:
                continue
            sp_r, sp_p = spearmanr(aligned["pool_size"], aligned["mrr"])
            pe_r, pe_p = pearsonr(aligned["pool_size"], aligned["mrr"])
            rows.append(
                {
                    "model_id": run["model_id"],
                    "seed": int(run["seed"]),
                    "run_id": run["run_id"],
                    "pair_type": pair_type,
                    "n_abstracts": int(len(aligned)),
                    "pool_size_mean": float(aligned["pool_size"].mean()),
                    "pool_size_std": float(aligned["pool_size"].std(ddof=0)),
                    "mrr_mean": float(aligned["mrr"].mean()),
                    "spearman_r": float(sp_r),
                    "spearman_p": float(sp_p),
                    "pearson_r": float(pe_r),
                    "pearson_p": float(pe_p),
                }
            )
    return pd.DataFrame(rows)


def build_all_auxiliary(per_run: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eh = build_easy_hard_ranking(per_run)
    eh.to_csv(OUTPUT_DIR / "11_easy_hard_ranking.csv", index=False)
    dist = build_distance_correlation(per_run)
    if not dist.empty:
        dist.to_csv(OUTPUT_DIR / "11_distance_score_correlation.csv", index=False)
    pool_sz = build_pool_size_robustness(per_run)
    if not pool_sz.empty:
        pool_sz.to_csv(OUTPUT_DIR / "11_pool_size_robustness.csv", index=False)
    ece, baselines = build_calibration_tables(per_run)
    ece.to_csv(OUTPUT_DIR / "11_calibration_ece.csv", index=False)
    baselines.to_csv(OUTPUT_DIR / "11_calibration_baselines.csv", index=False)
