"""Distance-confound diagnostic for step 04 (no re-training)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, pointbiserialr, spearmanr

from .config import MODEL_BY_ID, OUTPUT_DIR, RECALL_K_VALUES, SAMPLING_SEED
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
    """Add sentence distance, co-sentence flag, and proximity score per candidate."""
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


def _correlation_table(scores_df: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    merged = scores_df.merge(
        pool[["candidate_id", "proximity_score", "co_sentence", "sentence_distance"]],
        on="candidate_id",
        how="inner",
    )
    rows: list[dict[str, Any]] = []
    for model_id, sub in merged.groupby("model_id"):
        score = sub["score"].astype(float)
        prox = sub["proximity_score"].astype(float)
        valid = prox.notna() & score.notna()
        pr_r, pr_p = pearsonr(score[valid], prox[valid]) if valid.sum() > 2 else (np.nan, np.nan)
        sp_r, sp_p = spearmanr(score[valid], prox[valid]) if valid.sum() > 2 else (np.nan, np.nan)
        co = sub["co_sentence"].astype(int)
        pb_r, pb_p = pointbiserialr(co, score) if len(sub) > 2 else (np.nan, np.nan)
        rows.append(
            {
                "model_id": model_id,
                "model_name": MODEL_BY_ID[model_id].short_name if model_id in MODEL_BY_ID else model_id,
                "pearson_r_proximity": round(float(pr_r), 4),
                "pearson_p_proximity": round(float(pr_p), 4),
                "spearman_r_proximity": round(float(sp_r), 4),
                "spearman_p_proximity": round(float(sp_p), 4),
                "pointbiserial_r_co_sentence": round(float(pb_r), 4),
                "pointbiserial_p_co_sentence": round(float(pb_p), 4),
                "n_candidates": len(sub),
            }
        )
    return pd.DataFrame(rows)


def _subset_ranking_table(scores_df: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    merged = scores_df.merge(pool[["candidate_id", "subset"]], on="candidate_id", how="inner")
    rows: list[dict[str, Any]] = []
    for subset in ("easy_co_sentence", "hard_cross_sentence", "all"):
        if subset == "all":
            sub_pool = pool[pool["subset"] != "unknown"]
        else:
            sub_pool = pool[pool["subset"] == subset]
        if sub_pool.empty:
            continue
        pool_ids = set(sub_pool["candidate_id"])
        sub_scores = merged[merged["candidate_id"].isin(pool_ids)]
        n_cand = len(sub_pool)
        n_pmids = sub_pool["pmid"].nunique()

        dist_scored = sub_pool.copy()
        dist_scored["score"] = dist_scored["proximity_score"]
        dist_scored["label_civic_curated_positive"] = dist_scored["label_civic_curated_positive"].astype(bool)
        dist_row = ranking_metrics_for_scores(
            dist_scored[["candidate_id", "pmid", "pair_type", "label_civic_curated_positive", "score"]],
            "distance_ranker",
        )
        rows.append(
            {
                "subset": subset,
                "ranker": "distance_ranker",
                "n_candidates": n_cand,
                "n_pmids": n_pmids,
                **{k: dist_row[k] for k in ["mrr", "auc_pr"] + [f"recall_at_{k}" for k in RECALL_K_VALUES]},
            }
        )

        for model_id, model_sub in sub_scores.groupby("model_id"):
            eval_df = model_sub[
                ["candidate_id", "pmid", "pair_type", "label_civic_curated_positive", "score"]
            ].copy()
            eval_df["label_civic_curated_positive"] = eval_df["label_civic_curated_positive"].astype(bool)
            mrow = ranking_metrics_for_scores(eval_df, model_id)
            rows.append(
                {
                    "subset": subset,
                    "ranker": model_id,
                    "n_candidates": n_cand,
                    "n_pmids": n_pmids,
                    **{k: mrow[k] for k in ["mrr", "auc_pr"] + [f"recall_at_{k}" for k in RECALL_K_VALUES]},
                }
            )
    return pd.DataFrame(rows)


def _positive_distance_distribution(pool: pd.DataFrame) -> pd.DataFrame:
    pos = pool[pool["label_civic_curated_positive"]].copy()
    known = pos[pos["sentence_distance"].notna()]
    hist_rows: list[dict[str, Any]] = []
    for dist_val, count in known["sentence_distance"].astype(int).value_counts().sort_index().items():
        hist_rows.append(
            {
                "sentence_distance": int(dist_val),
                "n_positives": int(count),
                "fraction_of_known": round(count / len(known), 4) if len(known) else None,
            }
        )
    summary = pd.DataFrame(hist_rows)
    summary.attrs["n_positives_total"] = len(pos)
    summary.attrs["n_positives_known_distance"] = len(known)
    summary.attrs["fraction_co_sentence"] = round(float(known["co_sentence"].mean()), 4) if len(known) else None
    return summary


def _interpret_confound(
    corr_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    pos_dist: pd.DataFrame,
) -> dict[str, Any]:
    best_id = "pubmedbert_base"
    hard = subset_df[subset_df["subset"] == "hard_cross_sentence"]
    easy = subset_df[subset_df["subset"] == "easy_co_sentence"]
    best_hard = hard[hard["ranker"] == best_id]
    dist_hard = hard[hard["ranker"] == "distance_ranker"]
    best_easy = easy[easy["ranker"] == best_id]
    dist_easy = easy[easy["ranker"] == "distance_ranker"]

    best_corr = corr_df[corr_df["model_id"] == best_id].iloc[0] if len(corr_df) else None
    co_frac = pos_dist.attrs.get("fraction_co_sentence", 0.0)

    hard_beat_dist = (
        bool(len(best_hard) and len(dist_hard) and best_hard.iloc[0]["mrr"] > dist_hard.iloc[0]["mrr"])
    )
    easy_dominated = (
        bool(len(best_easy) and len(dist_easy) and dist_easy.iloc[0]["mrr"] > best_easy.iloc[0]["mrr"])
    )

    high_prox_corr = bool(best_corr is not None and best_corr["pearson_r_proximity"] > 0.5)
    positives_co_sentence_heavy = bool(co_frac is not None and co_frac > 0.6)

    if hard_beat_dist:
        favoured = "under-training"
        summary = (
            "On cross-sentence (hard) pairs, the best trained model outranks the distance ranker, "
            "while the distance ranker's overall lead likely reflects co-sentence pairs where proximity is a strong cue. "
            "Together with non-trivial score–distance correlation, this favours **under-training** over a purely distance-dominated task."
        )
    elif positives_co_sentence_heavy and easy_dominated:
        favoured = "distance-dominated"
        summary = (
            "CIViC-curated positives are mostly co-sentence, and the distance ranker leads on that easy subset; "
            "trained models do not surpass it even on cross-sentence pairs. "
            "Evidence favours a **distance-dominated** ranking task (data property) rather than recoverable signal from longer training alone."
        )
    elif high_prox_corr and not hard_beat_dist:
        favoured = "distance-dominated"
        summary = (
            "Model scores correlate strongly with entity proximity, and the trained model does not beat the distance ranker "
            "on hard cross-sentence pairs. This pattern points to a **distance-dominated** learnable signal at pilot scale."
        )
    else:
        favoured = "mixed"
        summary = (
            "Evidence is **mixed**: proximity explains part of model behaviour and pool structure, but the hard-subset comparison "
            "is inconclusive at this pilot scale. Longer training and/or task redesign both remain plausible next steps."
        )

    implication = (
        "If under-training is favoured, scale up optimisation before redesigning the evaluation. "
        "If distance-dominated, consider harder negatives, cross-sentence emphasis, or features that reduce proximity shortcutting."
    )

    return {
        "favoured_explanation": favoured,
        "verdict_summary": summary,
        "next_step_implication": implication,
        "fraction_positives_co_sentence": co_frac,
        "best_model_hard_mrr": float(best_hard.iloc[0]["mrr"]) if len(best_hard) else None,
        "distance_ranker_hard_mrr": float(dist_hard.iloc[0]["mrr"]) if len(dist_hard) else None,
        "best_model_easy_mrr": float(best_easy.iloc[0]["mrr"]) if len(best_easy) else None,
        "distance_ranker_easy_mrr": float(dist_easy.iloc[0]["mrr"]) if len(dist_easy) else None,
        "best_model_pearson_r_proximity": float(best_corr["pearson_r_proximity"]) if best_corr is not None else None,
        "n_easy_candidates": int(easy.iloc[0]["n_candidates"]) if len(easy) else 0,
        "n_hard_candidates": int(hard.iloc[0]["n_candidates"]) if len(hard) else 0,
    }


def run_distance_confound_diagnostic(scores_df: pd.DataFrame) -> dict[str, Any]:
    print("\n=== Distance-confound diagnostic ===")
    pool = enrich_with_proximity(load_primary_candidates())
    n_easy = int((pool["subset"] == "easy_co_sentence").sum())
    n_hard = int((pool["subset"] == "hard_cross_sentence").sum())
    n_unknown = int((pool["subset"] == "unknown").sum())
    print(f"  Pool subsets: easy (co-sentence)={n_easy}, hard (cross-sentence)={n_hard}, unknown={n_unknown}")

    corr_df = _correlation_table(scores_df, pool)
    subset_df = _subset_ranking_table(scores_df, pool)
    pos_dist = _positive_distance_distribution(pool)
    verdict = _interpret_confound(corr_df, subset_df, pos_dist)

    corr_df.to_csv(OUTPUT_DIR / "04_distance_score_correlation.csv", index=False)
    subset_df.to_csv(OUTPUT_DIR / "04_distance_subset_ranking.csv", index=False)
    pos_dist.to_csv(OUTPUT_DIR / "04_positive_distance_distribution.csv", index=False)

    pos_summary = pd.DataFrame(
        [
            {
                "n_positives_total": pos_dist.attrs.get("n_positives_total"),
                "n_positives_known_distance": pos_dist.attrs.get("n_positives_known_distance"),
                "fraction_co_sentence": pos_dist.attrs.get("fraction_co_sentence"),
            }
        ]
    )
    pos_summary.to_csv(OUTPUT_DIR / "04_positive_distance_summary.csv", index=False)

    print("\n  Score–proximity correlation (PubMedBERT):")
    pub = corr_df[corr_df["model_id"] == "pubmedbert_base"]
    if len(pub):
        r = pub.iloc[0]
        print(f"    Pearson r (proximity)={r['pearson_r_proximity']:.3f}, co-sentence r={r['pointbiserial_r_co_sentence']:.3f}")

    print(f"  CIViC positives co-sentence: {verdict['fraction_positives_co_sentence']:.1%}")
    print(
        f"  Hard subset MRR: PubMedBERT={verdict['best_model_hard_mrr']:.3f} vs "
        f"distance ranker={verdict['distance_ranker_hard_mrr']:.3f}"
    )
    print(f"  Easy subset MRR: PubMedBERT={verdict['best_model_easy_mrr']:.3f} vs "
          f"distance ranker={verdict['distance_ranker_easy_mrr']:.3f}")
    print(f"  Verdict favours: {verdict['favoured_explanation']}")

    return {
        "correlation": corr_df,
        "subset_ranking": subset_df,
        "positive_distance_histogram": pos_dist,
        "positive_distance_summary": pos_summary,
        "verdict": verdict,
        "pool_features": pool,
    }
