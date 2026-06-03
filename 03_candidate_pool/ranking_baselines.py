"""Ranking baselines on the real frozen candidate pool (step 03)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    EVIDENCE_JSON,
    FROZEN_POOL_JSON,
    OUTPUT_DIR,
    PRIMARY_PAIR_TYPES,
    RANKING_BASELINES_CSV,
    RANKING_VERIFICATION_JSON,
    SAMPLING_SEED,
)
from .distance_ranker import score_candidates


def _load_primary_candidates() -> pd.DataFrame:
    pool = json.loads(FROZEN_POOL_JSON.read_text(encoding="utf-8"))
    records = json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))
    abstracts = {
        str(item["source"]["citationId"]): item["source"].get("abstract", "")
        for item in records
        if item.get("source", {}).get("citationId")
    }
    rows: list[dict[str, Any]] = []
    for ab in pool.get("abstracts", []):
        pmid = str(ab["pmid"])
        for cand in ab.get("candidates", []):
            if cand.get("scope") != "primary":
                continue
            rows.append({**cand, "abstract": abstracts.get(pmid, "")})
    df = pd.DataFrame(rows)
    df["label_civic_curated_positive"] = df["is_civic_positive"].astype(bool)
    return df


def _import_ranking_metrics():
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return __import__(
        "02_evaluation_protocol.ranking_metrics",
        fromlist=[
            "evaluate_baselines",
            "ranking_metrics_for_scores",
            "verify_ranking_implementation",
            "analytic_random_mrr",
        ],
    )


def compute_ranking_baselines(candidates: pd.DataFrame | None = None) -> dict[str, Any]:
    rm = _import_ranking_metrics()
    if candidates is None:
        candidates = _load_primary_candidates()

    template = candidates[
        ["candidate_id", "pmid", "pair_type", "label_civic_curated_positive"]
    ].copy()
    template["score"] = 0.0

    print("\n=== Ranking baselines on real frozen pool ===")
    print(f"  Primary candidates: {len(candidates)}")
    print(f"  PMIDs: {candidates['pmid'].nunique()}")
    print(f"  Positive rate: {candidates['label_civic_curated_positive'].mean():.3f}")

    verification = rm.verify_ranking_implementation(template)
    baselines = rm.evaluate_baselines(template)

    abstracts = dict(zip(candidates["pmid"].astype(str), candidates["abstract"]))
    dist_df = score_candidates(candidates.to_dict(orient="records"), abstracts)
    dist_row = rm.ranking_metrics_for_scores(dist_df, "distance_ranker")
    dist_row["baseline"] = "distance_ranker"
    baselines = pd.concat([baselines, pd.DataFrame([dist_row])], ignore_index=True)

    baselines.to_csv(RANKING_BASELINES_CSV, index=False)
    print(baselines.to_string(index=False))

    summary = {
        "n_candidates": len(candidates),
        "n_pmids": int(candidates["pmid"].nunique()),
        "positive_rate": float(candidates["label_civic_curated_positive"].mean()),
        "ranking_verification": verification,
        "analytic_random_mrr": rm.analytic_random_mrr(template),
        "baselines": baselines.to_dict(orient="records"),
    }
    RANKING_VERIFICATION_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Saved -> {RANKING_BASELINES_CSV}")
    return summary
