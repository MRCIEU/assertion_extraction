#!/usr/bin/env python3.11
"""
Dataset Inventory — Main Entry Point

Runs all dataset inventory audits in sequence:
  1. Raw data availability check
  2. Corpus statistics from packaged JSONL
  3. Train-test leakage audit and fix

All outputs go to: ~/projects/project_1/dataset_inventory/

Usage:
  python3.11 run_inventory.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path

# Ensure submodule imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import OUT_ROOT, REPORTS, TABLES, ensure_dirs
from audit.raw_data_audit import run as run_raw_audit
from audit.corpus_statistics import run as run_corpus_stats
from audit.leakage_audit import run as run_leakage_audit


def main() -> None:
    ensure_dirs()
    print("=" * 60)
    print("DATASET INVENTORY — Full Audit Suite")
    print(f"Output root: {OUT_ROOT}")
    print("=" * 60)
    print()

    # Step 1
    print("STEP 1: Raw Data Availability")
    print("-" * 40)
    run_raw_audit()
    print()

    # Step 2
    print("STEP 2: Corpus Statistics")
    print("-" * 40)
    run_corpus_stats()
    print()

    # Step 3
    print("STEP 3: Leakage Audit & Fix")
    print("-" * 40)
    run_leakage_audit()
    print()

    # Final index
    index = {
        "subproject": "dataset_inventory",
        "output_root": str(OUT_ROOT),
        "reports": [
            "reports/raw_data_availability.json",
            "reports/corpus_statistics_summary.json",
            "reports/leakage_audit.json",
        ],
        "tables": [
            "reports/tables/raw_data_summary.csv",
            "reports/tables/corpus_relation_stats.csv",
            "reports/tables/corpus_entity_stats.csv",
            "reports/tables/corpus_split_counts.csv",
            "reports/tables/entity_pair_distribution.csv",
            "reports/tables/leakage_summary.csv",
        ],
    }
    (REPORTS / "inventory_index.json").write_text(json.dumps(index, indent=2))
    print("=" * 60)
    print("DATASET INVENTORY COMPLETE")
    print(f"Index: {REPORTS / 'inventory_index.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
