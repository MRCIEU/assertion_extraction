#!/usr/bin/env python3
"""Step 06 entry point: OncoKB parallel KB feasibility probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from importlib import import_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 06: OncoKB feasibility probe")
    parser.add_argument("--force-fetch", action="store_true", help="Re-fetch API data")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    preflight = import_module("06_oncokb_feasibility.preflight")
    probe = import_module("06_oncokb_feasibility.probe")
    report = import_module("06_oncokb_feasibility.report")

    print(f"=== Step 06 start {__import__('datetime').datetime.now().isoformat()} ===")
    if not args.skip_preflight:
        preflight.run_preflight()
    result = probe.run_probe(force_fetch=args.force_fetch)
    report.generate_report(result)
    report.write_readme(result)
    print("=== Step 06 complete ===")


if __name__ == "__main__":
    main()
