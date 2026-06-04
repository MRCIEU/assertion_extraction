"""Load frozen candidate pool."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .constants import PRIMARY_SCOPE
from .paths import upstream_paths


def load_primary_candidates() -> pd.DataFrame:
    paths = upstream_paths()
    pool = json.loads(paths["frozen_pool_json"].read_text(encoding="utf-8"))
    evidence = json.loads(paths["evidence_json"].read_text(encoding="utf-8"))
    abstracts = {
        str(i.get("source", {}).get("citationId", "")): i.get("source", {}).get("abstract", "")
        for i in evidence
    }
    rows: list[dict[str, Any]] = []
    for ab in pool.get("abstracts", []):
        pmid = str(ab["pmid"])
        abstract = abstracts.get(pmid, "")
        for cand in ab.get("candidates", []):
            if cand.get("scope") == PRIMARY_SCOPE:
                rows.append({**cand, "abstract": abstract})
    df = pd.DataFrame(rows)
    df["label_civic_curated_positive"] = df["is_civic_positive"].astype(bool)
    return df
