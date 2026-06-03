"""Shallow distance ranker baseline for step 03."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .matching import split_sentences


def _sentence_index(text: str, start: int | None, end: int | None) -> int | None:
    if start is None or end is None:
        return None
    cursor = 0
    for idx, sentence in enumerate(split_sentences(text)):
        pos = text.find(sentence, cursor)
        if pos == -1:
            continue
        sent_end = pos + len(sentence)
        cursor = sent_end
        if start >= pos and end <= sent_end:
            return idx
    return None


def proximity_score(abstract: str, head_start: int | None, head_end: int | None, tail_start: int | None, tail_end: int | None) -> float:
    """Higher score = closer entities (inverse distance)."""
    hs = _sentence_index(abstract, head_start, head_end)
    ts = _sentence_index(abstract, tail_start, tail_end)
    if hs is None or ts is None:
        return 0.0
    dist = abs(hs - ts)
    if dist == 0:
        return 1.0
    return 1.0 / (1.0 + dist)


def score_candidates(candidates: list[dict[str, Any]], abstracts: dict[str, str]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        abstract = abstracts.get(str(c["pmid"]), "")
        score = proximity_score(
            abstract,
            c.get("head_offset"),
            c.get("head_offset") + len(str(c.get("head_entity", ""))) if c.get("head_offset") is not None else None,
            c.get("tail_offset"),
            c.get("tail_offset") + len(str(c.get("tail_entity", ""))) if c.get("tail_offset") is not None else None,
        )
        rows.append({**c, "score": score, "baseline": "distance_ranker", "label_civic_curated_positive": bool(c.get("is_civic_positive"))})
    return pd.DataFrame(rows)


def evaluate_distance_ranker(scores_df: pd.DataFrame) -> dict[str, float]:
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    rm = __import__("02_evaluation_protocol.ranking_metrics", fromlist=["ranking_metrics_for_scores"])
    eval_df = scores_df.rename(columns={"is_civic_positive": "label_civic_curated_positive"})
    eval_df["label_civic_curated_positive"] = eval_df["label_civic_curated_positive"].astype(bool)
    return rm.ranking_metrics_for_scores(eval_df, "distance_ranker")
