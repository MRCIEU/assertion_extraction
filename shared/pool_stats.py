"""Shared pool statistics (step 03 pool); reusable by folders 11 and 20."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .pool_loader import load_primary_candidates


def abstract_pool_sizes(candidates: pd.DataFrame | None = None) -> pd.Series:
    """Number of primary-scope candidates per abstract (PMID)."""
    if candidates is None:
        candidates = load_primary_candidates()
    return candidates.groupby("pmid").size().rename("pool_size")


def save_abstract_pool_sizes(out_path: Path, candidates: pd.DataFrame | None = None) -> Path:
    """Persist pool-size table for downstream reuse (folder 11, folder 20)."""
    sizes = abstract_pool_sizes(candidates)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = sizes.reset_index()
    df.columns = ["pmid", "pool_size"]
    df.to_csv(out_path, index=False)
    return out_path


def load_abstract_pool_sizes(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    return df.set_index("pmid")["pool_size"]
