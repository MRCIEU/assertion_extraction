"""Ranking metrics with correct tie handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from .constants import RECALL_K_VALUES, SAMPLING_SEED


def _rank_within_abstracts(df: pd.DataFrame, seed: int = SAMPLING_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for _, group in df.groupby("pmid", sort=False):
        g = group.copy()
        tiebreak = rng.random(len(g))
        avg_rank = rankdata(-g["score"].to_numpy(), method="average")
        g["_sort_key"] = avg_rank + tiebreak * 1e-6
        g = g.sort_values("_sort_key", ascending=True)
        g["rank"] = np.arange(1, len(g) + 1)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def compute_mrr(df: pd.DataFrame, seed: int = SAMPLING_SEED) -> float:
    ranked = _rank_within_abstracts(df, seed=seed)
    mrrs: list[float] = []
    for _, group in ranked.groupby("pmid", sort=False):
        if not group["label_civic_curated_positive"].any():
            continue
        pos = group[group["label_civic_curated_positive"]]
        best_rank = float(pos["rank"].min())
        mrrs.append(1.0 / best_rank)
    return float(np.mean(mrrs)) if mrrs else 0.0


def compute_recall_at_k(df: pd.DataFrame, k: int, seed: int = SAMPLING_SEED) -> float:
    ranked = _rank_within_abstracts(df, seed=seed)
    recalls: list[float] = []
    for _, group in ranked.groupby("pmid", sort=False):
        n_pos = int(group["label_civic_curated_positive"].sum())
        if n_pos == 0:
            continue
        top_k = group.nsmallest(k, "rank")
        hits = int(top_k["label_civic_curated_positive"].sum())
        recalls.append(hits / n_pos)
    return float(np.mean(recalls)) if recalls else 0.0


def compute_global_auc_pr(df: pd.DataFrame) -> float:
    y = df["label_civic_curated_positive"].astype(int).values
    if len(np.unique(y)) < 2:
        return float(y.mean())
    return float(average_precision_score(y, df["score"].values))


def ranking_metrics_for_scores(df: pd.DataFrame, label: str, seed: int = SAMPLING_SEED) -> dict:
    row: dict = {
        "model_or_baseline": label,
        "mrr": compute_mrr(df, seed=seed),
        "auc_pr": compute_global_auc_pr(df),
        "n_candidates": len(df),
        "n_abstracts": df["pmid"].nunique(),
        "positive_rate": float(df["label_civic_curated_positive"].mean()),
    }
    for k in RECALL_K_VALUES:
        row[f"recall_at_{k}"] = compute_recall_at_k(df, k, seed=seed)
    return row


def metrics_by_pair_type(scores_df: pd.DataFrame, label: str, seed: int = SAMPLING_SEED) -> pd.DataFrame:
    rows = []
    for pair_type, sub in scores_df.groupby("pair_type"):
        row = ranking_metrics_for_scores(sub, label, seed=seed)
        row["pair_type"] = pair_type
        rows.append(row)
    return pd.DataFrame(rows)
