#!/usr/bin/env python3.11
"""Generate `runs_inventory.csv` — the authoritative per-run manifest for
all Phase A (and eventually Phase B) trained runs.

For every run directory under $PROJECT_1_DATA_ROOT/fine_tuning_experiments/runs/
that matches `PA_*_*_s*` or `PB_*_*_s*`, record:

    run_id, phase, encoder_key, schema_key, seed,
    run_dir, best_pt_path, best_pt_sha256, best_pt_size_bytes,
    best_pt_mtime_utc, eval_json_path, eval_json_sha256,
    eval_version, git_commit, config_sha256, generated_utc

Two uses:
  1. **Reproducibility contract**: any post-lock re-execution of H6 /
     aggregation / figures can point to the exact checkpoint bytes used,
     via the recorded SHA-256.  A mismatch means the artifact was rewritten.
  2. **Backup integrity**: before/after a scratch-retention event the
     inventory is re-run and diffed; any SHA change signals either a
     re-train or a data-corruption event.

Output file:
    fine_tuning_experiments/schema_exp/runs_inventory.csv

Invocation:
    python3.11 -m fine_tuning_experiments.schema_exp.build_runs_inventory
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get(
    "PROJECT_1_DATA_ROOT", "/lus/lfs1aip2/projects/b5ac/project_1",
))
RUNS_ROOT = DATA_ROOT / "fine_tuning_experiments" / "runs" / "schema_exp"
OUT_CSV = SCRIPT_DIR / "runs_inventory.csv"

_RUN_RE = re.compile(r"^(PA|PB)_([A-Z]+)_([A-Za-z]+)_s(\d+)$")

_CSV_FIELDS = [
    "run_id", "phase", "encoder_key", "schema_key", "seed",
    "run_dir", "best_pt_path", "best_pt_sha256", "best_pt_size_bytes",
    "best_pt_mtime_utc", "eval_json_path", "eval_json_sha256",
    "eval_version", "git_commit", "config_sha256",
    "excluded", "excluded_reason", "generated_utc",
]

# Default exclusion policy (see §7.8 + Appendix B item 7 (j)):
# seeds outside the 1..10 primary-analysis range are non-primary smoke runs
# and are flagged excluded=true so the inventory manifest stays accurate
# without being mixed into any H1..H7 aggregation.
_PRIMARY_SEEDS = set(range(1, 11))


def _default_exclusion(seed: int, phase: str) -> tuple[bool, str]:
    if seed not in _PRIMARY_SEEDS:
        return True, f"seed={seed} outside primary 1..10 range ({phase} non-primary smoke)"
    return False, ""


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _iter_run_dirs(runs_root: Path) -> list[Path]:
    if not runs_root.is_dir():
        return []
    return sorted([p for p in runs_root.iterdir() if p.is_dir() and _RUN_RE.match(p.name)])


def _row_for_run(run_dir: Path, now_iso: str) -> dict[str, Any] | None:
    m = _RUN_RE.match(run_dir.name)
    if not m:
        return None
    phase, enc, schema, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
    best_pt = run_dir / "checkpoints" / "best.pt"
    eval_json = run_dir / "eval" / "phase_a_eval.json"
    manifest = run_dir / "run_manifest.json"

    row: dict[str, Any] = {
        "run_id": run_dir.name, "phase": phase,
        "encoder_key": enc, "schema_key": schema, "seed": seed,
        "run_dir": str(run_dir),
        "best_pt_path": str(best_pt) if best_pt.exists() else "",
        "best_pt_sha256": _sha256(best_pt) if best_pt.exists() else "",
        "best_pt_size_bytes": (best_pt.stat().st_size if best_pt.exists() else ""),
        "best_pt_mtime_utc": (
            datetime.fromtimestamp(best_pt.stat().st_mtime, tz=timezone.utc).isoformat()
            if best_pt.exists() else ""
        ),
        "eval_json_path": str(eval_json) if eval_json.exists() else "",
        "eval_json_sha256": _sha256(eval_json) if eval_json.exists() else "",
        "eval_version": "",
        "git_commit": "",
        "config_sha256": "",
        "excluded": "false",
        "excluded_reason": "",
        "generated_utc": now_iso,
    }
    excl, excl_reason = _default_exclusion(seed, phase)
    if excl:
        row["excluded"] = "true"
        row["excluded_reason"] = excl_reason
    if eval_json.exists():
        try:
            d = json.loads(eval_json.read_text())
            row["eval_version"] = d.get("eval_version", "")
        except Exception:
            pass
    if manifest.exists():
        try:
            m_d = json.loads(manifest.read_text())
            row["git_commit"] = m_d.get("git_commit", "") or ""
            row["config_sha256"] = m_d.get("config_sha256", "") or ""
        except Exception:
            pass
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    now_iso = datetime.now(timezone.utc).isoformat()
    run_dirs = _iter_run_dirs(args.runs_root)
    if not run_dirs:
        print(f"WARN: no run dirs under {args.runs_root}")
    rows: list[dict[str, Any]] = []
    for rd in run_dirs:
        row = _row_for_run(rd, now_iso)
        if row is None:
            continue
        rows.append(row)
        if not args.quiet:
            sha_short = (row["best_pt_sha256"] or "")[:12]
            size_mb = (row["best_pt_size_bytes"] or 0)
            if isinstance(size_mb, int):
                size_mb = size_mb / (1 << 20)
            print(f"  {row['run_id']:<25} "
                  f"best={sha_short or 'MISSING':<12} "
                  f"size={size_mb:6.1f} MB  "
                  f"eval_v={row['eval_version'] or '<legacy>'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
