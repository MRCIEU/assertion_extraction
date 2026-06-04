"""Step 1: formal recipe sweep (train/val + benchmark F1 only; no KB)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from shared.benchmark_eval import build_biored_test_examples, evaluate_checkpoint_benchmark_f1
from shared.constants import DEGENERATE_BENCHMARK_F1_MAX, DEGENERATE_VAL_F1_MAX, GUARD_SEEDS
from shared.models import MODEL_BY_ID
from shared.train_core import RecipeConfig, train_best_only
from shared.train_data import build_train_val_examples

from .config import (
    SWEEP_COMPLETE,
    SWEEP_RESULTS_DIR,
    TRAIN_CACHE_DIR,
    SweepPoint,
    all_sweep_points,
)


def is_degenerate(best_val_f1: float, benchmark_f1: float) -> bool:
    return best_val_f1 <= DEGENERATE_VAL_F1_MAX or benchmark_f1 <= DEGENERATE_BENCHMARK_F1_MAX


def _write_marker(point: SweepPoint, payload: dict) -> Path:
    path = point.result_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_sweep_point(
    point: SweepPoint,
    train_examples: list[dict],
    val_examples: list[dict],
    test_examples: list[dict],
    *,
    force: bool = False,
    bad_seed_guard: bool = False,
) -> dict:
    marker = point.result_path()
    if marker.exists() and not force:
        return json.loads(marker.read_text(encoding="utf-8"))

    spec = MODEL_BY_ID[point.model_id]
    recipe = RecipeConfig(point.lr, point.warmup_ratio, point.warmup_label)
    ckpt_dir = point.ckpt_dir()

    meta = train_best_only(
        spec,
        point.seed,
        train_examples,
        val_examples,
        ckpt_dir,
        recipe,
        force=force,
    )
    bench = evaluate_checkpoint_benchmark_f1(ckpt_dir, test_examples)

    payload = {
        "run_id": point.run_id,
        "model_id": point.model_id,
        "short_name": spec.short_name,
        "lr": point.lr,
        "warmup_label": point.warmup_label,
        "warmup_ratio": point.warmup_ratio,
        "seed": point.seed,
        "bad_seed_guard": bad_seed_guard,
        "best_epoch_val_f1": meta["best_epoch_val_f1"],
        "best_val_f1": meta["best_val_f1"],
        "benchmark_f1": bench["benchmark_f1"],
        "epoch_curve": meta["epoch_curve"],
        "degenerate": is_degenerate(meta["best_val_f1"], bench["benchmark_f1"]),
        "checkpoint": str(ckpt_dir),
    }
    _write_marker(point, payload)
    print(
        f"  SWEEP {point.run_id}: benchmark_f1={bench['benchmark_f1']:.3f} "
        f"best_epoch={meta['best_epoch_val_f1']} degenerate={payload['degenerate']}"
    )
    return payload


def run_sweep(
    *,
    force: bool = False,
    model_ids: list[str] | None = None,
) -> None:
    """Run the full sweep grid at seed 42; bad-seed guard on collapse."""
    train_examples, val_examples = build_train_val_examples(TRAIN_CACHE_DIR)
    test_examples = build_biored_test_examples()

    points = all_sweep_points()
    if model_ids:
        points = [p for p in points if p.model_id in model_ids]

    for point in points:
        payload = run_sweep_point(
            point, train_examples, val_examples, test_examples, force=force
        )
        if payload["degenerate"] and not payload.get("bad_seed_guard"):
            print(f"  Bad-seed guard triggered for {point.run_id}; re-running with seeds {GUARD_SEEDS}")
            guard_point = SweepPoint(
                model_id=point.model_id,
                lr=point.lr,
                warmup_label=point.warmup_label,
                warmup_ratio=point.warmup_ratio,
                seed=GUARD_SEEDS[0],
            )
            for gseed in GUARD_SEEDS:
                gp = SweepPoint(
                    model_id=point.model_id,
                    lr=point.lr,
                    warmup_label=point.warmup_label,
                    warmup_ratio=point.warmup_ratio,
                    seed=gseed,
                )
                run_sweep_point(
                    gp, train_examples, val_examples, test_examples, force=force, bad_seed_guard=True
                )

    print(f"\n=== Step-1 sweep complete ({len(points)} primary points) ===")
