#!/usr/bin/env python3
"""Step 00 entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from importlib import import_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 00: CIViC feasibility")
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument("--skip-fetch", action="store_true")
    args = parser.parse_args()

    fetch = import_module("00_civic_feasibility.fetch")
    inventory = import_module("00_civic_feasibility.inventory")
    analyses = import_module("00_civic_feasibility.analyses")
    report = import_module("00_civic_feasibility.report")

    print(f"=== Step 00 start {__import__('datetime').datetime.now().isoformat()} ===")
    if not args.skip_fetch:
        fetch.fetch_all(force=args.force_fetch)
    inventory.build_inventory()
    analyses.run_all_analyses()
    report.generate_report()
    print("=== Step 00 complete ===")


if __name__ == "__main__":
    main()
