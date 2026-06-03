"""Distance-confound diagnostic (analysis D) and easy/hard subset ranking (analysis A)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, pointbiserialr, spearmanr

from .config import MODEL_BY_ID, RECALL_K_VALUES, SAMPLING_SEED
from .metrics_ranking import compute_mrr, compute_recall_at_k, ranking_metrics_for_scores
from .pool_loader import load_primary_candidates

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_dist = __import__(
    "03_candidate_pool.distance_ranker",
    fromlist=["proximity_score", "_sentence_index"],
)


def enrich_with_proximity(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        abstract = str(row.get("abstract") or "")
        hs = row.get("head_offset")
        ts = row.get("tail_offset")
        he = hs + len(str(row.get("head_entity", ""))) if hs is not None and pd.notna(hs) else None
        te = ts + len(str(row.get("tail_entity", ""))) if ts is not None and pd.notna(ts) else None
        hi = _dist._sentence_index(abstract, int(hs) if hs is not None and pd.notna(hs) else None, he)
        ti = _dist._sentence_index(abstract, int(ts) if ts is not None and pd.notna(ts) else None, te)
        if hi is None or ti is None:
            sent_dist = np.nan
            co_sentence = False
        else:
            sent_dist = float(abs(hi - ti))
            co_sentence = sent_dist == 0.0
        prox = float(
            _dist.proximity_score(
                abstract,
                int(hs) if hs is not None and pd.notna(hs) else None,
                he,
                int(ts) if ts is not None and pd.notna(ts) else None,
                te,
            )
        )
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "sentence_distance": sent_dist,
                "co_sentence": co_sentence,
                "proximity_score": prox,
            }
        )
    feat = pd.DataFrame(rows)
    out = candidates.merge(feat, on="candidate_id", how="left")
    out["subset"] = np.where(
        out["co_sentence"],
        "easy_co_sentence",
        np.where(out["sentence_distance"].notna() & (out["sentence_distance"] > 0), "hard_cross_sentence", "unknown"),
    )
    return out


def distance_ranker_scores(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    out["score"] = out["proximity_score"]
    out["run_id"] = "distance_ranker"
    out["model_id"] = "distance_ranker"
    out["seed"] = -1
    return out


def score_proximity_correlations(scores_df: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    merged = scores_df.merge(
        pool[["candidate_id", "proximity_score", "co_sentence", "sentence_distance"]],
        on="candidate_id",
        how="inner",
    )
    rows: list[dict] = []
    for run_id, sub in merged.groupby("run_id"):
        score = sub["score"].astype(float)
        prox = sub["proximity_score"].astype(float)
        valid = score.notna() & prox.notna()
        if valid.sum() < 10:
            continue
        s, p = score[valid], prox[valid]
        pr, pp = pearsonr(s, p)
        sr, sp = spearmanr(s, p)
        pb, pb_p = pointbiserialr(sub.loc[valid, "co_sentence"].astype(int), s)
        rows.append(
            {
                "run_id": run_id,
                "model_id": sub["model_id"].iloc[0],
                "seed": int(sub["seed"].iloc[0]),
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_r": float(sr),
                "spearman_p": float(sp),
                "pointbiserial_co_sentence": float(pb),
                "n": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows)


def subset_ranking_metrics(
    scores_df: pd.DataFrame,
    pool: pd.DataFrame,
    seed: int = SAMPLING_SEED,
) -> pd.DataFrame:
    merged = scores_df.merge(pool[["candidate_id", "subset"]], on="candidate_id", how="inner")
    rows: list[dict] = []
    for run_id, sub in merged.groupby("run_id"):
        for subset, ss in sub.groupby("subset"):
            if subset == "unknown":
                continue
            row = ranking_metrics_for_scores(ss, run_id, seed=seed)
            row["run_id"] = run_id
            row["model_id"] = ss["model_id"].iloc[0]
            row["seed"] = int(ss["seed"].iloc[0])
            row["subset"] = subset
            rows.append(row)
    return pd.DataFrame(rows)


def distance_ranker_subset_metrics(pool: pd.DataFrame, seed: int = SAMPLING_SEED) -> pd.DataFrame:
    dr = distance_ranker_scores(pool)
    rows: list[dict] = []
    for subset, sub in pool.groupby("subset"):
        if subset == "unknown":
            continue
        ss = dr.merge(sub[["candidate_id"]], on="candidate_id")
        row = ranking_metrics_for_scores(ss, "distance_ranker", seed=seed)
        row["run_id"] = "distance_ranker"
        row["model_id"] = "distance_ranker"
        row["seed"] = -1
        row["subset"] = subset
        rows.append(row)
    return pd.DataFrame(rows)
