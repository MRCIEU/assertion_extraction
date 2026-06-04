"""Upstream artifact paths shared across training and analysis stages."""

from __future__ import annotations

from pathlib import Path

from _paths import OUTPUT_ROOT


def upstream_paths(root: Path | None = None) -> dict[str, Path]:
    base = root or OUTPUT_ROOT
    return {
        "excluded_pmids_json": base / "outputs" / "01_corpus_relevance" / "excluded_pmids.json",
        "frozen_pool_json": base / "outputs" / "03_candidate_pool" / "frozen_pool.json",
        "evidence_json": base / "data" / "00_civic_feasibility" / "evidence_items.json",
    }
