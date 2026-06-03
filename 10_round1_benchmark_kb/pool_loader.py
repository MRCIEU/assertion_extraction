"""Load frozen candidate pool and abstract texts."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .config import EVIDENCE_JSON, FROZEN_POOL_JSON, PRIMARY_SCOPE


def build_abstract_lookup() -> dict[str, str]:
    records = json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))
    lookup: dict[str, str] = {}
    for item in records:
        source = item.get("source") or {}
        pmid = str(source.get("citationId") or "")
        abstract = source.get("abstract") or ""
        if pmid and abstract:
            lookup[pmid] = abstract
    return lookup


def load_primary_candidates() -> pd.DataFrame:
    """Primary-scope candidates from frozen step-03 pool."""
    pool = json.loads(FROZEN_POOL_JSON.read_text(encoding="utf-8"))
    abstracts = build_abstract_lookup()
    rows: list[dict[str, Any]] = []

    for ab in pool.get("abstracts", []):
        pmid = str(ab["pmid"])
        abstract = abstracts.get(pmid, "")
        for cand in ab.get("candidates", []):
            if cand.get("scope") != PRIMARY_SCOPE:
                continue
            rows.append({**cand, "abstract": abstract, "abstract_length": len(abstract)})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No primary-scope candidates loaded from frozen pool.")

    df["label_civic_curated_positive"] = df["is_civic_positive"].astype(bool)
    return df
