#!/usr/bin/env python3
"""Round 1 entry point: benchmark rank vs KB ranking and calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from importlib import import_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 1: benchmark vs KB (GPU)")
    parser.add_argument("--train-eval-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--force-train-data", action="store_true")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    exclusive = [args.train_eval_only, args.train_only, args.eval_only, args.analyze_only]
    if sum(exclusive) > 1:
        raise SystemExit(
            "Use at most one of --train-eval-only, --train-only, --eval-only, --analyze-only"
        )

    train = eval_models = analyze = True
    if args.train_eval_only:
        analyze = False
    elif args.train_only:
        eval_models = analyze = False
    elif args.eval_only:
        train = analyze = False
    elif args.analyze_only:
        train = eval_models = False

    print(f"=== Round 1 start {__import__('datetime').datetime.now().isoformat()} ===")
    build = import_module("10_round1_benchmark_kb.build_round1")
    if args.analyze_only:
        build.run_analysis()
    else:
        build.run_matrix(
            train=train,
            eval_models=eval_models,
            analyze=analyze,
            force_train=args.force_train,
            force_eval=args.force_eval,
            force_train_data=args.force_train_data,
            model_ids=args.models,
            seeds=args.seeds,
        )
    print("=== Round 1 complete ===")


if __name__ == "__main__":
    main()
