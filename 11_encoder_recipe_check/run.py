#!/usr/bin/env python3
"""Encoder recipe check: DeBERTa training diagnostic (benchmark F1 only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from importlib import import_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Encoder recipe check (DeBERTa grid)")
    parser.add_argument("--train-only", action="store_true", help="Run 4-point grid (+ bad-seed guard)")
    parser.add_argument("--analyze-only", action="store_true", help="Finalise tables, figures, report")
    parser.add_argument("--train-fallback-only", action="store_true", help="Optional lr=5e-6 + warmup (manual)")
    parser.add_argument("--force", action="store_true", help="Re-run even if markers exist")
    args = parser.parse_args()

    n_exclusive = sum([args.train_only, args.analyze_only, args.train_fallback_only])
    if n_exclusive > 1:
        raise SystemExit("Use at most one of --train-only, --analyze-only, --train-fallback-only")

    print(f"=== Encoder recipe check {__import__('datetime').datetime.now().isoformat()} ===")

    if args.analyze_only:
        import_module("11_encoder_recipe_check.analyze").run_analysis()
    elif args.train_fallback_only:
        import_module("11_encoder_recipe_check.run_grid").run_fallback(force=args.force)
    else:
        import_module("11_encoder_recipe_check.run_grid").run_grid(train=True, force=args.force)

    print("=== Done ===")


if __name__ == "__main__":
    main()
