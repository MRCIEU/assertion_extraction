"""Ranking metrics and trivial baselines (shared evaluation logic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from .config import RECALL_K_VALUES, SAMPLING_SEED


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


def analytic_random_mrr(template: pd.DataFrame) -> float:
    values: list[float] = []
    for _, group in template.groupby("pmid", sort=False):
        n_pos = int(group["label_civic_curated_positive"].sum())
        if n_pos == 0:
            continue
        n = len(group)
        values.append(sum(1.0 / r for r in range(1, n + 1)) / n)
    return float(np.mean(values)) if values else 0.0


def verify_ranking_implementation(template: pd.DataFrame, seed: int = SAMPLING_SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    rand = template.copy()
    rand["score"] = rng.random(len(rand))
    const = template.copy()
    const["score"] = 0.5

    mrr_random = compute_mrr(rand, seed=seed)
    mrr_constant = compute_mrr(const, seed=seed)
    mrr_analytic = analytic_random_mrr(template)

    ok = abs(mrr_constant - mrr_random) < 0.05 and mrr_constant < 0.5
    print("\n=== Ranking implementation verification ===")
    print(f"  random Mean Reciprocal Rank (MRR):   {mrr_random:.4f}")
    print(f"  constant-score MRR: {mrr_constant:.4f}")
    print(f"  analytic E[MRR]: {mrr_analytic:.4f}")
    print(f"  |constant - random| = {abs(mrr_constant - mrr_random):.4f}  PASS={ok}")
    if not ok:
        raise RuntimeError("Ranking tie-handling verification failed.")
    return {
        "mrr_random": mrr_random,
        "mrr_constant": mrr_constant,
        "mrr_analytic": mrr_analytic,
        "abs_constant_random_diff": abs(mrr_constant - mrr_random),
        "verification_pass": ok,
    }


def ranking_metrics_for_scores(df: pd.DataFrame, label: str, seed: int = SAMPLING_SEED) -> dict:
    row = {
        "baseline": label,
        "mrr": compute_mrr(df, seed=seed),
        "auc_pr": compute_global_auc_pr(df),
        "n_candidates": len(df),
        "n_abstracts": df["pmid"].nunique(),
    }
    for k in RECALL_K_VALUES:
        row[f"recall_at_{k}"] = compute_recall_at_k(df, k, seed=seed)
    return row


def trivial_baselines(template: pd.DataFrame, seed: int = SAMPLING_SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rand = template.copy()
    rand["score"] = rng.random(len(rand))
    const = template.copy()
    const["score"] = 0.5
    return {"random": rand, "constant": const}


def evaluate_baselines(template: pd.DataFrame, seed: int = SAMPLING_SEED) -> pd.DataFrame:
    rows = [ranking_metrics_for_scores(sub, name, seed=seed) for name, sub in trivial_baselines(template, seed).items()]
    return pd.DataFrame(rows)
