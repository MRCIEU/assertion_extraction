"""Run DeBERTa recipe grid (+ bad-seed guard). Benchmark F1 only; no KB."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    DEGENERATE_BENCHMARK_F1_MAX,
    DEGENERATE_VAL_F1_MAX,
    GRID_POINTS,
    GUARD_SEEDS,
    PRIMARY_SEED,
    OUTPUT_DIR,
    RESULTS_DIR,
    COMPLETE_MARKER,
)
from .config import GridPoint
from .train_grid import (
    build_biored_test_examples,
    checkpoint_dir,
    evaluate_benchmark_f1_from_ckpt,
    is_complete,
    marker_path,
    train_grid_point,
)

_r1_data = __import__(
    "10_round1_benchmark_kb.train_data",
    fromlist=["build_train_val_examples"],
)
build_train_val_examples = _r1_data.build_train_val_examples


def is_degenerate(best_val_f1: float, benchmark_f1: float) -> bool:
    return best_val_f1 <= DEGENERATE_VAL_F1_MAX or benchmark_f1 <= DEGENERATE_BENCHMARK_F1_MAX


def _write_marker(point: GridPoint, seed: int, payload: dict) -> Path:
    path = marker_path(point, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_one(
    point: GridPoint,
    seed: int,
    train_examples: list[dict],
    val_examples: list[dict],
    test_examples: list[dict],
    force: bool = False,
    bad_seed_guard: bool = False,
) -> dict:
    if is_complete(point, seed) and not force:
        return json.loads(marker_path(point, seed).read_text(encoding="utf-8"))

    ckpt = train_grid_point(point, seed, train_examples, val_examples, force=force)
    meta_path = ckpt / "11_train_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bench = evaluate_benchmark_f1_from_ckpt(ckpt, test_examples)

    payload = {
        "model_id": "deberta_base",
        "run_key": point.key,
        "lr": point.lr,
        "warmup_label": point.warmup_label,
        "warmup_ratio": point.warmup_ratio,
        "seed": seed,
        "bad_seed_guard": bad_seed_guard,
        "best_epoch_val_f1": meta["best_epoch_val_f1"],
        "best_val_f1": meta["best_val_f1"],
        "benchmark_f1": bench["benchmark_f1"],
        "epoch_curve": meta["epoch_curve"],
        "degenerate": is_degenerate(meta["best_val_f1"], bench["benchmark_f1"]),
    }
    _write_marker(point, seed, payload)
    print(
        f"  DONE {point.key} seed={seed}: best_epoch={payload['best_epoch_val_f1']} "
        f"val_f1={payload['best_val_f1']:.4f} benchmark_f1={payload['benchmark_f1']:.4f}"
    )
    return payload


def run_grid(train: bool = True, force: bool = False) -> list[dict]:
    if not train:
        return _load_all_markers()

    train_examples, val_examples = build_train_val_examples()
    test_examples = build_biored_test_examples()
    rows: list[dict] = []

    for point in GRID_POINTS:
        row = run_one(point, PRIMARY_SEED, train_examples, val_examples, test_examples, force=force)
        rows.append(row)
        if row.get("degenerate"):
            print(f"  BAD-SEED GUARD triggered for {point.key}")
            for gseed in GUARD_SEEDS:
                grows = run_one(
                    point,
                    gseed,
                    train_examples,
                    val_examples,
                    test_examples,
                    force=force,
                    bad_seed_guard=True,
                )
                rows.append(grows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_grid_csv(rows)
    _print_stdout_summary(rows)
    return rows


def _load_all_markers() -> list[dict]:
    rows: list[dict] = []
    for point in GRID_POINTS:
        for seed_dir in sorted((RESULTS_DIR / point.key).glob("seed_*")) if (RESULTS_DIR / point.key).exists() else []:
            marker = seed_dir / COMPLETE_MARKER
            if marker.exists():
                rows.append(json.loads(marker.read_text(encoding="utf-8")))
    return rows


def _save_grid_csv(rows: list[dict]) -> None:
    import pandas as pd

    flat = []
    for r in rows:
        flat.append(
            {
                "run_key": r["run_key"],
                "lr": r["lr"],
                "warmup_label": r["warmup_label"],
                "warmup_ratio": r["warmup_ratio"],
                "seed": r["seed"],
                "bad_seed_guard": r.get("bad_seed_guard", False),
                "best_epoch_val_f1": r["best_epoch_val_f1"],
                "best_val_f1": r["best_val_f1"],
                "benchmark_f1": r["benchmark_f1"],
                "degenerate": r.get("degenerate", False),
            }
        )
    pd.DataFrame(flat).to_csv(OUTPUT_DIR / "grid_results.csv", index=False)

    curves = []
    for r in rows:
        for ep in r.get("epoch_curve", []):
            curves.append(
                {
                    "run_key": r["run_key"],
                    "lr": r["lr"],
                    "warmup_label": r["warmup_label"],
                    "seed": r["seed"],
                    **ep,
                }
            )
    if curves:
        pd.DataFrame(curves).to_csv(OUTPUT_DIR / "grid_epoch_curves.csv", index=False)


def _print_stdout_summary(rows: list[dict]) -> None:
    import pandas as pd

    from .config import ROUND10_DEGENERATE_CSV, ROUND10_ENCODER_SUMMARY

    print("\n=== Step 0: Round-1 degenerate runs ===")
    if ROUND10_DEGENERATE_CSV.exists():
        deg = pd.read_csv(ROUND10_DEGENERATE_CSV)
        for _, d in deg.iterrows():
            print(f"  {d['model_id']} seed={int(d['seed'])} flags={d.get('flags', '')}")
    else:
        print("  (degenerate CSV not found)")

    print("\n=== Recipe grid (primary seed 42 unless guard) ===")
    df = pd.DataFrame(
        [
            {
                "lr": r["lr"],
                "warmup": r["warmup_label"],
                "seed": r["seed"],
                "best_epoch": r["best_epoch_val_f1"],
                "benchmark_f1": r["benchmark_f1"],
                "guard": r.get("bad_seed_guard", False),
            }
            for r in rows
        ]
    )
    print(df.to_string(index=False))

    primary = [r for r in rows if r["seed"] == PRIMARY_SEED and not r.get("bad_seed_guard")]
    if primary:
        best = max(primary, key=lambda x: x["benchmark_f1"])
        enc = pd.read_csv(ROUND10_ENCODER_SUMMARY)
        others = enc[enc["model_id"] != "deberta_base"]
        print(
            f"\nDeBERTa grid best (seed {PRIMARY_SEED}): {best['benchmark_f1']:.3f} "
            f"at lr={best['lr']:.0e} warmup={best['warmup_label']}"
        )
        print(
            f"Round-1 eight encoders (excl. DeBERTa): min={others['benchmark_f1_mean'].min():.3f} "
            f"max={others['benchmark_f1_mean'].max():.3f}"
        )
        print(
            f"Round-1 DeBERTa mean (old recipe): "
            f"{enc.loc[enc['model_id'] == 'deberta_base', 'benchmark_f1_mean'].iloc[0]:.3f}"
        )


def run_fallback(force: bool = False) -> dict:
    """Optional fallback: lr=5e-6 + 10% warmup (not part of default grid)."""
    from .config import FALLBACK_POINT, FALLBACK_SEED

    train_examples, val_examples = build_train_val_examples()
    test_examples = build_biored_test_examples()
    row = run_one(FALLBACK_POINT, FALLBACK_SEED, train_examples, val_examples, test_examples, force=force)
    rows = _load_all_markers()
    _save_grid_csv(rows)
    _print_stdout_summary(rows)
    return row
