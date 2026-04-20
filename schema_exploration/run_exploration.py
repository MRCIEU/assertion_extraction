#!/usr/bin/env python3.11
"""
Schema Exploration — Main Entry Point

Steps:
  1. Build SC1 and SC3 JSONL packages (remapping BioRED relations)
  2. Per-head support audit (SC0 / SC1 / SC3)
  3. Generate schema comparison framework documentation

Outputs → ~/projects/project_1/schema_exploration/

Usage:
  python3.11 run_exploration.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import OUT_ROOT, REPORTS, TABLES, DATA_OUT, ensure_dirs
from pipeline.remap_packages import run as run_remap
from evaluation.head_support_audit import run as run_head_audit
from evaluation.schema_comparison_framework import run as run_framework


def main() -> None:
    ensure_dirs()
    print("=" * 60)
    print("SCHEMA EXPLORATION — Full Pipeline")
    print(f"Output root: {OUT_ROOT}")
    print("=" * 60)
    print()

    print("STEP 1: Build SC1 and SC3 Packages")
    print("-" * 40)
    run_remap()
    print()

    print("STEP 2: Per-Head Support Audit")
    print("-" * 40)
    run_head_audit()
    print()

    print("STEP 3: Schema Comparison Framework")
    print("-" * 40)
    run_framework()
    print()

    # Final index
    index = {
        "subproject": "schema_exploration",
        "output_root": str(OUT_ROOT),
        "schema_candidates": ["SC0", "SC1", "SC3"],
        "primary_selection_metric": "KB_surface_mean = mean(1 - P(NEGATIVE)) over 165 CIViC targets",
        "selection_rule": "SC* = argmax KB_surface_mean, subject to all heads F1 > 0.05",
        "status": "Data packages generated. GPU training experiments pending.",
        "reports": [
            "reports/remapping_report.json",
            "reports/head_support_summary.json",
            "reports/schema_comparison_framework.json",
        ],
        "tables": [
            "reports/tables/head_support_audit.csv",
            "reports/tables/biored_test_gold_by_schema.csv",
            "reports/tables/schema_feasibility_summary.csv",
        ],
        "data_packages": {
            "SC0": "Uses existing *_trn.jsonl packages (no remapping needed)",
            "SC1": "*_L1.jsonl packages in training_data_generation/data/processed/",
            "SC3": "*_SC3.jsonl packages in training_data_generation/data/processed/",
        },
        "next_step": "Run GPU experiments: M003-SC0, M003-SC1, M003-SC3, B1-SC0, B1-SC1, B1-SC3",
    }
    (REPORTS / "exploration_index.json").write_text(json.dumps(index, indent=2))
    print("=" * 60)
    print("SCHEMA EXPLORATION COMPLETE")
    print(f"Index: {REPORTS / 'exploration_index.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
