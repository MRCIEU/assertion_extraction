#!/usr/bin/env python3
"""Step 05: marker and span quality gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from importlib import import_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 05: marker quality gate")
    parser.add_argument(
        "--skip-cache-rebuild",
        action="store_true",
        help="Run checks only without rebuilding train caches",
    )
    args = parser.parse_args()

    cfg = import_module("05_marker_quality_gate.config")
    checks_mod = import_module("05_marker_quality_gate.checks")
    report_mod = import_module("05_marker_quality_gate.report")

    print("=== Step 05: marker and span quality gate ===")
    results = checks_mod.run_all_checks(rebuild_cache=not args.skip_cache_rebuild)

    checks_df = pd.DataFrame(results["checks"])
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checks_df.to_csv(cfg.QUALITY_CHECKS_CSV, index=False)
    cfg.QUALITY_RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    report_path = report_mod.write_report(results, checks_df)

    print("\n=== Quality gate checks ===")
    for _, r in checks_df.iterrows():
        status = "PASS" if r["passed"] else "FAIL"
        before = f" | before: {r['before']}" if pd.notna(r.get("before")) and r.get("before") else ""
        print(f"  [{status}] {r['name']}: {r['detail']}{before}")

    print(f"\nOverall: {'PASS' if results['overall_pass'] else 'FAIL'}")
    print(f"Training same-sentence (native): {results.get('training_same_sentence_rate'):.1%}")
    if results.get("civic_offset_rate") is not None:
        print(f"CIViC pool offset insertion: {results['civic_offset_rate']:.1%}")
    print(f"Report -> {report_path}")
    print("=== Step 05 complete ===")

    if not results["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
