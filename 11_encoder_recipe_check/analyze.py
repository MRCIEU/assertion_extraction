"""Finalisation: compare grid to Round-1 encoder distribution (no KB)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    GRID_POINTS,
    OUTPUT_DIR,
    PRIMARY_SEED,
    ROUND10_DEBERTA_COLLAPSED_META,
    ROUND10_DEGENERATE_CSV,
    ROUND10_ENCODER_SUMMARY,
    RESULTS_DIR,
    COMPLETE_MARKER,
)
from .figures import plot_encoder_strip, plot_val_curves
from .report import write_report


def load_round10_degenerate() -> pd.DataFrame:
    if not ROUND10_DEGENERATE_CSV.exists():
        raise FileNotFoundError(f"Missing {ROUND10_DEGENERATE_CSV}")
    return pd.read_csv(ROUND10_DEGENERATE_CSV)


def load_round10_encoder_benchmark() -> pd.DataFrame:
    if not ROUND10_ENCODER_SUMMARY.exists():
        raise FileNotFoundError(f"Missing {ROUND10_ENCODER_SUMMARY}")
    return pd.read_csv(ROUND10_ENCODER_SUMMARY)


def load_grid_markers() -> list[dict]:
    from .config import FALLBACK_POINT

    rows: list[dict] = []
    for point in list(GRID_POINTS) + [FALLBACK_POINT]:
        base = RESULTS_DIR / point.key
        if not base.exists():
            continue
        for seed_dir in sorted(base.glob("seed_*")):
            marker = seed_dir / COMPLETE_MARKER
            if marker.exists():
                rows.append(json.loads(marker.read_text(encoding="utf-8")))
    return rows


def step0_degenerate_identification() -> pd.DataFrame:
    deg = load_round10_degenerate()
    out = deg.copy()
    out["is_deberta"] = out["model_id"] == "deberta_base"
    out.to_csv(OUTPUT_DIR / "degenerate_run_identification.csv", index=False)
    return out


def build_deberta_vs_group(grid_rows: list[dict]) -> pd.DataFrame:
    enc = load_round10_encoder_benchmark()
    r10_deberta = float(enc.loc[enc["model_id"] == "deberta_base", "benchmark_f1_mean"].iloc[0])

    primary = [r for r in grid_rows if r["seed"] == PRIMARY_SEED and not r.get("bad_seed_guard")]
    guard = [r for r in grid_rows if r.get("bad_seed_guard")]

    best_primary = max(primary, key=lambda x: x["benchmark_f1"]) if primary else None
    best_any = max(grid_rows, key=lambda x: x["benchmark_f1"]) if grid_rows else None

    rows: list[dict[str, Any]] = []
    for _, row in enc.iterrows():
        rows.append(
            {
                "label": row["short_name"],
                "model_id": row["model_id"],
                "source": "round1_encoder_mean",
                "benchmark_f1": float(row["benchmark_f1_mean"]),
                "is_deberta": row["model_id"] == "deberta_base",
            }
        )
    if best_primary:
        rows.append(
            {
                "label": "DeBERTa (grid best, seed 42)",
                "model_id": "deberta_base",
                "source": "recipe_grid_primary",
                "benchmark_f1": float(best_primary["benchmark_f1"]),
                "run_key": best_primary["run_key"],
                "is_deberta": True,
            }
        )
    if best_any and (not best_primary or best_any["benchmark_f1"] > best_primary["benchmark_f1"]):
        rows.append(
            {
                "label": "DeBERTa (grid best, any seed)",
                "model_id": "deberta_base",
                "source": "recipe_grid_best_any",
                "benchmark_f1": float(best_any["benchmark_f1"]),
                "run_key": best_any["run_key"],
                "seed": best_any["seed"],
                "is_deberta": True,
            }
        )
    rows.append(
        {
            "label": "DeBERTa (Round-1 mean, old recipe)",
            "model_id": "deberta_base",
            "source": "round1_mean_old",
            "benchmark_f1": r10_deberta,
            "is_deberta": True,
        }
    )
    df = pd.DataFrame(rows)
    others = enc[enc["model_id"] != "deberta_base"]["benchmark_f1_mean"]
    df.to_csv(OUTPUT_DIR / "deberta_vs_group.csv", index=False)
    return df, float(others.min()), float(others.max())


def warmup_contrast_table(grid_rows: list[dict]) -> pd.DataFrame:
    """Primary-seed none vs warmup at each lr."""
    rows = []
    for lr in sorted({p.lr for p in GRID_POINTS}):
        sub = [
            r
            for r in grid_rows
            if r["seed"] == PRIMARY_SEED
            and not r.get("bad_seed_guard")
            and abs(r["lr"] - lr) < 1e-12
        ]
        none = next((r for r in sub if r["warmup_label"] == "none"), None)
        warm = next((r for r in sub if r["warmup_label"] == "warmup_10pct"), None)
        if none and warm:
            rows.append(
                {
                    "lr": lr,
                    "benchmark_f1_none": none["benchmark_f1"],
                    "benchmark_f1_warmup_10pct": warm["benchmark_f1"],
                    "delta_warmup_minus_none": warm["benchmark_f1"] - none["benchmark_f1"],
                    "best_epoch_none": none["best_epoch_val_f1"],
                    "best_epoch_warmup": warm["best_epoch_val_f1"],
                }
            )
    return pd.DataFrame(rows)


def run_analysis() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_rows = load_grid_markers()
    if len(grid_rows) < 4:
        raise SystemExit(
            f"Analysis aborted: expected at least 4 grid markers, found {len(grid_rows)}. "
            "Run training grid first."
        )

    deg = step0_degenerate_identification()
    vs_group, others_min, others_max = build_deberta_vs_group(grid_rows)
    warmup_df = warmup_contrast_table(grid_rows)
    warmup_df.to_csv(OUTPUT_DIR / "warmup_contrast.csv", index=False)

    curves_path = OUTPUT_DIR / "grid_epoch_curves.csv"
    if curves_path.exists():
        curves = pd.read_csv(curves_path)
    else:
        curves = pd.DataFrame(
            [
                {**ep, "run_key": r["run_key"], "lr": r["lr"], "warmup_label": r["warmup_label"], "seed": r["seed"]}
                for r in grid_rows
                for ep in r.get("epoch_curve", [])
            ]
        )

    r10_deberta_epoch1 = _round1_deberta_reference_curve()
    plot_val_curves(curves, grid_rows, r10_reference=r10_deberta_epoch1)
    plot_encoder_strip(vs_group)

    write_report(
        degenerate=deg,
        grid_rows=grid_rows,
        vs_group=vs_group,
        warmup_df=warmup_df,
        others_min=others_min,
        others_max=others_max,
    )

    _print_analysis_stdout(deg, grid_rows, vs_group, warmup_df)


def _round1_deberta_reference_curve() -> pd.DataFrame | None:
    """Val curve from collapsed Round-1 DeBERTa seed 45 for contrast."""
    if not ROUND10_DEBERTA_COLLAPSED_META.exists():
        return None
    meta = json.loads(ROUND10_DEBERTA_COLLAPSED_META.read_text(encoding="utf-8"))
    curve = meta.get("epoch_curve", [])
    if not curve:
        return None
    df = pd.DataFrame(curve)
    df["run_key"] = "round1_deberta_seed45_old_recipe"
    df["warmup_label"] = "none (Round 1)"
    return df


def _print_analysis_stdout(
    deg: pd.DataFrame,
    grid_rows: list[dict],
    vs_group: pd.DataFrame,
    warmup_df: pd.DataFrame,
) -> None:
    print("\n=== Step 0: Round-1 degenerate runs ===")
    for _, d in deg.iterrows():
        print(f"  {d['model_id']} seed={int(d['seed'])}  is_deberta={d['is_deberta']}")

    print("\n=== Grid results ===")
    for r in sorted(grid_rows, key=lambda x: (x["run_key"], x["seed"])):
        print(
            f"  lr={r['lr']:.0e} warmup={r['warmup_label']} seed={r['seed']} "
            f"epoch={r['best_epoch_val_f1']} benchmark_f1={r['benchmark_f1']:.3f}"
            + (" [guard]" if r.get("bad_seed_guard") else "")
        )

    primary = [r for r in grid_rows if r["seed"] == PRIMARY_SEED and not r.get("bad_seed_guard")]
    if primary:
        best = max(primary, key=lambda x: x["benchmark_f1"])
        print(
            f"\nDeBERTa best (primary seeds): {best['benchmark_f1']:.3f} "
            f"({best['run_key']})"
        )
    enc = load_round10_encoder_benchmark()
    others = enc[enc["model_id"] != "deberta_base"]
    print(
        f"Eight-encoder Round-1 range: {others['benchmark_f1_mean'].min():.3f} "
        f"to {others['benchmark_f1_mean'].max():.3f}"
    )
    if not warmup_df.empty:
        print("\nWarmup contrast (seed 42):")
        print(warmup_df.to_string(index=False))

    print("\n=== Analysis complete ===")
