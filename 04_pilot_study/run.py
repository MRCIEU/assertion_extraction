#!/usr/bin/env python3
"""Step 04 pilot study entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from importlib import import_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 04: pilot study (GPU)")
    parser.add_argument("--verify-ranking-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-score", action="store_true")
    parser.add_argument("--force-train-data", action="store_true")
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    if args.verify_ranking_only:
        pl = import_module("04_pilot_study.pool_loader")
        rm = import_module("04_pilot_study.ranking_metrics")
        t = pl.load_primary_candidates()[["candidate_id", "pmid", "label_civic_curated_positive"]]
        rm.verify_ranking_implementation(t.rename(columns={"label_civic_curated_positive": "label_civic_positive"}))
        return

    exclusive = [args.train_only, args.score_only, args.analyze_only]
    if sum(exclusive) > 1:
        raise SystemExit("Use at most one of --train-only, --score-only, --analyze-only")

    train = score = analyze = True
    if args.train_only:
        score = analyze = False
    elif args.score_only:
        train = analyze = False
    elif args.analyze_only:
        train = score = False

    print(f"=== Step 04 start {__import__('datetime').datetime.now().isoformat()} ===")
    build = import_module("04_pilot_study.build_pilot")
    build.run_pilot(
        train=train,
        score=score,
        analyze=analyze,
        force_train=args.force_train,
        force_score=args.force_score,
        force_train_data=args.force_train_data,
        model_ids=args.models,
    )
    print("=== Step 04 complete ===")


if __name__ == "__main__":
    main()
