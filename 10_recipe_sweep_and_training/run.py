#!/usr/bin/env python3
"""Recipe sweep (step 1) and full-matrix training (step 2). Produces models only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from importlib import import_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Recipe sweep and full-matrix training")
    parser.add_argument("--sweep-only", action="store_true", help="Step 1: recipe sweep grid")
    parser.add_argument(
        "--sweep-advisory-only",
        action="store_true",
        help="Step 1: build advisory table from existing sweep results",
    )
    parser.add_argument(
        "--decide-recipe",
        action="store_true",
        help="Step 1: recipe decision aid from sweep CSVs (CPU; non-binding)",
    )
    parser.add_argument("--train-only", action="store_true", help="Step 2: full matrix training")
    parser.add_argument(
        "--preflight-sweep",
        action="store_true",
        help="Verify clean offset-marked train cache before sweep",
    )
    parser.add_argument(
        "--preflight-train",
        action="store_true",
        help="Verify step-2 preconditions before full-matrix training",
    )
    parser.add_argument(
        "--accept-matrix",
        action="store_true",
        help="Post-training acceptance gate (DeBERTa hard gate)",
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if markers exist")
    parser.add_argument("--models", nargs="+", default=None, help="Limit to model IDs")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Limit to seeds")
    args = parser.parse_args()

    n = sum(
        [
            args.sweep_only,
            args.sweep_advisory_only,
            args.decide_recipe,
            args.train_only,
            args.preflight_sweep,
            args.preflight_train,
            args.accept_matrix,
        ]
    )
    if n != 1:
        raise SystemExit(
            "Specify exactly one of --sweep-only, --sweep-advisory-only, --decide-recipe, "
            "--train-only, --preflight-sweep, --preflight-train, --accept-matrix"
        )

    print(f"=== Recipe sweep and training {__import__('datetime').datetime.now().isoformat()} ===", flush=True)

    if args.preflight_train:
        print("[run] mode=preflight-train", flush=True)
        ok = import_module("10_recipe_sweep_and_training.step2_preflight").run_step2_preflight()
        if not ok:
            raise SystemExit(1)
    elif args.accept_matrix:
        print("[run] mode=accept-matrix", flush=True)
        code = import_module("10_recipe_sweep_and_training.step2_acceptance").run_acceptance_gate()
        raise SystemExit(code)
    elif args.preflight_sweep:
        print("[run] mode=preflight-sweep", flush=True)
        import_module("10_recipe_sweep_and_training.step1_preflight").verify_clean_train_cache()
    elif args.sweep_only:
        print("[run] mode=sweep-only", flush=True)
        import_module("10_recipe_sweep_and_training.step1_sweep").run_sweep(
            force=args.force, model_ids=args.models
        )
    elif args.sweep_advisory_only:
        print("[run] mode=sweep-advisory-only", flush=True)
        import_module("10_recipe_sweep_and_training.step1_advisory").run_advisory()
    elif args.decide_recipe:
        print("[run] mode=decide-recipe", flush=True)
        import_module("10_recipe_sweep_and_training.step1_decide").run_decide_recipe()
    else:
        print("[run] mode=train-only (step 2)", flush=True)
        if args.models:
            print(f"[run] model filter: {args.models}", flush=True)
        if args.seeds:
            print(f"[run] seed filter: {args.seeds}", flush=True)
        import_module("10_recipe_sweep_and_training.step2_train").run_matrix_training(
            force=args.force, model_ids=args.models, seeds=args.seeds
        )

    print("=== Done ===", flush=True)


if __name__ == "__main__":
    main()
