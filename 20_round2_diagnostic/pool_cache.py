"""Load or build enriched candidate pool once (expensive proximity pass)."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from importlib import import_module

from .config import ENRICHED_POOL_CACHE

_r1pool = import_module("10_round1_benchmark_kb.pool_loader")
_r1dist = import_module("10_round1_benchmark_kb.distance_analysis")


def load_enriched_pool(force_rebuild: bool = False) -> pd.DataFrame:
    if ENRICHED_POOL_CACHE.exists() and not force_rebuild:
        print(f"  Loaded enriched pool cache ({ENRICHED_POOL_CACHE.name})")
        return pd.read_parquet(ENRICHED_POOL_CACHE)

    print("  Building enriched pool (one-time proximity pass over ~19k candidates)...")
    t0 = time.perf_counter()
    pool = _r1pool.load_primary_candidates()
    enriched = _r1dist.enrich_with_proximity(pool)
    ENRICHED_POOL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(ENRICHED_POOL_CACHE, index=False)
    print(f"  Enriched pool built in {time.perf_counter() - t0:.1f}s -> {ENRICHED_POOL_CACHE}")
    return enriched
