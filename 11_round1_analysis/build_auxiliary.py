"""Build auxiliary CSVs: easy/hard ranking, distance confound, calibration tables."""

from __future__ import annotations

import pandas as pd

from shared.distance_analysis import (
    distance_ranker_subset_metrics,
    enrich_with_proximity,
    score_proximity_correlations,
    subset_ranking_metrics,
)
from shared.inference import load_scores_jsonl
from shared.metrics_calibration import calibration_baselines, calibration_for_scores
from shared.pool_loader import load_primary_candidates

from .config import OUTPUT_DIR, SCORES_DIR


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


def build_all_auxiliary(per_run: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eh = build_easy_hard_ranking(per_run)
    eh.to_csv(OUTPUT_DIR / "11_easy_hard_ranking.csv", index=False)
    dist = build_distance_correlation(per_run)
    if not dist.empty:
        dist.to_csv(OUTPUT_DIR / "11_distance_score_correlation.csv", index=False)
    ece, baselines = build_calibration_tables(per_run)
    ece.to_csv(OUTPUT_DIR / "11_calibration_ece.csv", index=False)
    baselines.to_csv(OUTPUT_DIR / "11_calibration_baselines.csv", index=False)
