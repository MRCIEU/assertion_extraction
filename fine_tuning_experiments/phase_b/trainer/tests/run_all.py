#!/usr/bin/env python3.11
"""Run all six §7.8 trainer unit tests in sequence; exit 0 only if all pass.

Usage:
    python3.11 -m fine_tuning_experiments.phase_b.trainer.tests.run_all

Or:
    bash fine_tuning_experiments/phase_b/trainer/tests/run_all.sh

This is the command the pre-lock harness and `babysit`-style CI can invoke.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TESTS = [
    "test_label_space",
    "test_schema_expected_label",
    "test_negative_sampler",
    "test_eval_fields",
    "test_checkpoint_roundtrip",
    "test_training_determinism",
]

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def main() -> int:
    failures: list[str] = []
    for t in TESTS:
        print(f"\n=== {t} ===")
        rc = subprocess.call(
            ["python3.11", "-m",
             f"fine_tuning_experiments.phase_b.trainer.tests.{t}"],
            cwd=str(PROJECT_ROOT),
        )
        if rc != 0:
            failures.append(f"{t} exited with {rc}")
    print("\n===============================")
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PASS: all {len(TESTS)} trainer unit tests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
