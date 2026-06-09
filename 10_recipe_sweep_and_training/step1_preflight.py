"""Preflight checks before step-1 recipe sweep on clean offset-marked data."""

from __future__ import annotations

import json
from pathlib import Path

from shared.constants import LEAKED_PMIDS
from shared.paths import upstream_paths

from .config import TRAIN_CACHE_DIR


def verify_clean_train_cache(cache_dir: Path | None = None) -> dict:
    cache_dir = cache_dir or TRAIN_CACHE_DIR
    train_path = cache_dir / "train_examples_train.jsonl"
    val_path = cache_dir / "train_examples_val.jsonl"

    if not train_path.exists() or not val_path.exists():
        raise SystemExit(
            f"Missing train cache under {cache_dir}. Run step 05 marker quality gate first."
        )

    offset_n = 0
    total = 0
    pmids: set[str] = set()
    for path in (train_path, val_path):
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            if row.get("marker_method") == "offset":
                offset_n += 1
            pmid = str(row.get("pmid", ""))
            if pmid:
                pmids.add(pmid)

    leaked = pmids & LEAKED_PMIDS
    excluded_path = upstream_paths()["excluded_pmids_json"]
    excluded: set[str] = set()
    if excluded_path.exists():
        data = json.loads(excluded_path.read_text(encoding="utf-8"))
        excluded = {str(x["pmid"]) if isinstance(x, dict) else str(x) for x in data.get("excluded_pmids", [])}
    hit_excluded = pmids & excluded

    offset_rate = offset_n / total if total else 0.0
    ok = offset_rate >= 0.999 and not leaked and not hit_excluded

    summary = {
        "cache_dir": str(cache_dir),
        "n_examples": total,
        "offset_marker_rate": offset_rate,
        "leaked_pmids_in_cache": sorted(leaked),
        "excluded_pmids_in_cache": sorted(hit_excluded),
        "ok": ok,
    }

    print("\n=== Step-1 sweep preflight (clean train cache) ===")
    print(f"  Cache path: {cache_dir}")
    print(f"  Examples: {total} (offset marker_method: {offset_rate:.1%})")
    print(f"  Leaked PMIDs in cache: {len(leaked)}")
    print(f"  Excluded PMIDs in cache: {len(hit_excluded)}")
    print(f"  Preflight: {'PASS' if ok else 'FAIL'}")

    if not ok:
        raise SystemExit("Preflight failed: cache is not clean offset-marked training data.")

    return summary
