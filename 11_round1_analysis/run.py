#!/usr/bin/env python3
"""Round 1 analysis: consumes folder-10 matrix checkpoints (no training)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from importlib import import_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 1 analysis (CPU; optional GPU rescore)")
    parser.add_argument("--analyze-only", action="store_true", help="Run analysis from stored scores")
    parser.add_argument("--score-only", action="store_true", help="Score KB at best checkpoints only")
    parser.add_argument("--rescore", action="store_true", help="Force KB rescoring before analysis")
    parser.add_argument("--force-score", action="store_true", help="Overwrite existing score files")
    parser.add_argument("--model-id", action="append", default=None, help="Limit scoring to encoder(s)")
    parser.add_argument("--seed", type=int, action="append", default=None, help="Limit scoring to seed(s)")
    args = parser.parse_args()

    if args.score_only:
        import_module("11_round1_analysis.score_runs").score_all_runs(
            force=args.force_score,
            model_ids=args.model_id,
            seeds=args.seed,
        )
        return

    if args.analyze_only or args.rescore:
        import_module("11_round1_analysis.run_analysis").run_analysis(
            rescore=args.rescore, force_score=args.force_score
        )
        return

    raise SystemExit("Use --score-only, --analyze-only, or --analyze-only --rescore")


if __name__ == "__main__":
    main()
