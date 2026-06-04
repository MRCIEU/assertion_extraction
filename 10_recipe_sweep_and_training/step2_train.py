"""Step 2: full-matrix training (9 encoders x 8 seeds) with per-epoch checkpoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from shared.benchmark_eval import build_biored_test_examples, evaluate_checkpoint_benchmark_f1
from shared.constants import TRAIN_SEEDS
from shared.models import MODELS, MODEL_BY_ID
from shared.train_core import train_with_epoch_checkpoints
from shared.train_data import build_train_val_examples

from .config import (
    ESTIMATED_AVG_EPOCHS_PER_RUN,
    ESTIMATED_FP16_CHECKPOINT_MIB,
    ESTIMATED_FP32_CHECKPOINT_MIB,
    MATRIX_COMPLETE,
    MAX_EPOCH_CHECKPOINTS_TO_KEEP,
    SAVE_EPOCH_CHECKPOINTS_FP16,
    TRAIN_CACHE_DIR,
    matrix_result_path,
    matrix_run_root,
    require_chosen_recipe,
)


def _write_complete(model_id: str, seed: int, recipe, training_log: dict, bench: dict) -> Path:
    path = matrix_result_path(model_id, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": model_id,
        "seed": seed,
        "recipe_lr": recipe.lr,
        "recipe_warmup_label": recipe.warmup_label,
        "training_strategy": recipe.strategy_tag(),
        "best_epoch_val_f1": training_log["best_epoch_val_f1"],
        "best_val_f1": training_log["best_val_f1"],
        "best_checkpoint": training_log["best_checkpoint"],
        **bench,
        "epoch_curve": training_log["epoch_curve"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def is_matrix_complete(model_id: str, seed: int) -> bool:
    return matrix_result_path(model_id, seed).exists()


def print_checkpoint_footprint_estimate(n_models: int, n_seeds: int) -> None:
    """Rough storage budget before training starts (MiB and GiB)."""
    epochs = ESTIMATED_AVG_EPOCHS_PER_RUN
    epoch_mib = ESTIMATED_FP16_CHECKPOINT_MIB if SAVE_EPOCH_CHECKPOINTS_FP16 else ESTIMATED_FP32_CHECKPOINT_MIB
    per_run_mib = epochs * epoch_mib + ESTIMATED_FP32_CHECKPOINT_MIB
    total_mib = n_models * n_seeds * per_run_mib
    print(
        f"\n=== Estimated checkpoint footprint (rough) ===\n"
        f"  Runs: {n_models} encoders x {n_seeds} seeds = {n_models * n_seeds}\n"
        f"  Assumed ~{epochs} epochs/run; epoch ckpt ~{epoch_mib} MiB "
        f"({'fp16' if SAVE_EPOCH_CHECKPOINTS_FP16 else 'fp32'}); best ckpt ~{ESTIMATED_FP32_CHECKPOINT_MIB} MiB (fp32)\n"
        f"  Per run ~{per_run_mib} MiB; total ~{total_mib / 1024:.1f} GiB\n"
        f"  MAX_EPOCH_CHECKPOINTS_TO_KEEP={MAX_EPOCH_CHECKPOINTS_TO_KEEP!r} "
        f"(None = keep all epoch checkpoints)\n"
    )


def run_matrix_training(
    *,
    force: bool = False,
    model_ids: list[str] | None = None,
    seeds: list[int] | None = None,
) -> None:
    """Train 72 runs: per-epoch weights + val metrics; benchmark F1 at best checkpoint only."""
    recipe = require_chosen_recipe()
    specs = MODELS if model_ids is None else [MODEL_BY_ID[m] for m in model_ids]
    seed_list = TRAIN_SEEDS if seeds is None else seeds

    print(
        f"\n=== Step-2 matrix training ===\n"
        f"Recipe: lr={recipe.lr}, warmup={recipe.warmup_label}\n"
        f"Encoders: {len(specs)}, seeds: {seed_list}\n"
    )
    print_checkpoint_footprint_estimate(len(specs), len(seed_list))

    train_examples, val_examples = build_train_val_examples(TRAIN_CACHE_DIR)
    test_examples = build_biored_test_examples()

    for spec in specs:
        for seed in seed_list:
            if is_matrix_complete(spec.model_id, seed) and not force:
                print(f"  skip (complete): {spec.model_id} seed={seed}")
                continue

            run_root = matrix_run_root(spec.model_id, seed)
            log = train_with_epoch_checkpoints(
                spec,
                seed,
                train_examples,
                val_examples,
                run_root,
                recipe,
                save_epoch_fp16=SAVE_EPOCH_CHECKPOINTS_FP16,
                max_epoch_checkpoints_to_keep=MAX_EPOCH_CHECKPOINTS_TO_KEEP,
                force=force,
            )
            best_ckpt = Path(log["best_checkpoint"])
            bench = evaluate_checkpoint_benchmark_f1(best_ckpt, test_examples)
            _write_complete(spec.model_id, seed, recipe, log, bench)
            print(
                f"  COMPLETE {spec.model_id} seed={seed}: "
                f"benchmark_f1={bench['benchmark_f1']:.3f} best_epoch={log['best_epoch_val_f1']}"
            )

    print("\n=== Step-2 matrix training complete ===")
