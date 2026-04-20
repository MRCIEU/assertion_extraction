#!/usr/bin/env python3.11
"""
Downstream transfer & utility analysis sweep for knowledge-grounded evidence audit.

Heavy phases (Tier-1 sweep) MUST run under Slurm GPU — see scripts/run_transfer_tier1_gpu.sbatch.
Metadata / goldlite reassessment can run without GPU.

Override local block: KG_AUDIT_ALLOW_LOCAL=1 (debug only).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.downstream_transfer.aggregate_transfer import run_aggregate
from analysis.downstream_transfer.aggregate_tier2 import run_tier2_aggregate
from analysis.downstream_transfer.compose_downstream_report import compose
from analysis.downstream_transfer.final_project_consolidation import run_final_consolidation
from analysis.downstream_transfer.goldlite_core import run_goldlite_reassessment
from analysis.downstream_transfer.metadata_build import run_metadata_phase
from analysis.downstream_transfer.paths import MANIFESTS, OUT_ROOT, REPORTS, ensure_dirs
from analysis.downstream_transfer.sweep_tier1_runner import run_tier1_sweep
from analysis.downstream_transfer.sweep_tier2_runner import run_tier2_sweep
from analysis.downstream_transfer.tier2_placeholder import write_tier2_placeholders


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_slurm_for_sweep() -> None:
    if os.environ.get("KG_AUDIT_ALLOW_LOCAL") == "1":
        return
    if os.environ.get("SLURM_JOB_ID"):
        return
    print(
        "[downstream_transfer] Refusing Tier-1 sweep outside Slurm. Submit:\n"
        "  sbatch project_1/knowledge_grounded_evidence_audit/scripts/run_transfer_tier1_gpu.sbatch\n"
        "Override (not for production): KG_AUDIT_ALLOW_LOCAL=1",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(2)


def _append_tier2_status(phase: str, status: str, note: str = "") -> None:
    p = MANIFESTS / "tier2_run_status_table.csv"
    row = {"phase": phase, "status": status, "timestamp_utc": _utc(), "notes": note}
    newf = not p.is_file() or p.stat().st_size == 0
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if newf:
            w.writeheader()
        w.writerow(row)


def _append_status(phase: str, status: str, note: str = "") -> None:
    p = MANIFESTS / "run_status_table.csv"
    row = {"phase": phase, "status": status, "timestamp_utc": _utc(), "notes": note}
    newf = not p.is_file()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if newf:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        default="metadata",
        help="metadata,goldlite,tier1_sweep,aggregate,tier2_placeholders,tier2_sweep,tier2_aggregate,final_consolidation,reports,all_sweep,all_tier2_final",
    )
    args = ap.parse_args()

    ensure_dirs()
    print(f"[downstream_transfer] OUT_ROOT={OUT_ROOT}", flush=True)

    ph = args.phase.strip()
    phases = []
    if ph == "all_sweep":
        phases = ["metadata", "goldlite", "tier1_sweep", "aggregate", "tier2_placeholders", "reports"]
    elif ph == "all_tier2_final":
        phases = ["tier2_sweep", "tier2_aggregate", "final_consolidation", "reports"]
    else:
        phases = [p.strip() for p in ph.split(",") if p.strip()]

    for phase in phases:
        try:
            if phase == "metadata":
                run_metadata_phase()
            elif phase == "goldlite":
                run_goldlite_reassessment()
            elif phase == "tier1_sweep":
                _require_slurm_for_sweep()
                run_tier1_sweep()
            elif phase == "aggregate":
                run_aggregate()
            elif phase == "tier2_placeholders":
                write_tier2_placeholders()
            elif phase == "tier2_sweep":
                _require_slurm_for_sweep()
                run_tier2_sweep()
                _append_tier2_status("tier2_sweep", "ok")
            elif phase == "tier2_aggregate":
                run_tier2_aggregate()
                _append_tier2_status("tier2_aggregate", "ok")
            elif phase == "final_consolidation":
                run_final_consolidation()
                _append_tier2_status("final_consolidation", "ok")
            elif phase == "reports":
                compose()
            else:
                raise ValueError(f"unknown phase {phase}")
            _append_status(phase, "ok")
            print(f"[downstream_transfer] phase {phase} ok", flush=True)
        except Exception as e:
            _append_status(phase, "error", str(e)[:400])
            print(f"[downstream_transfer] ERROR {phase}: {e}", flush=True)
            traceback.print_exc()

    (REPORTS / "sweep_execution_summary.md").write_text(
        f"# Sweep execution summary\n\nUpdated {_utc()}\n\nSee `manifests/run_status_table.csv`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
