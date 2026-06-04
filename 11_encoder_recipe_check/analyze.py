"""Finalisation: DeBERTa recipe grid vs Round-1 reference (analysis only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    GRID_POINTS,
    OUTPUT_DIR,
    PRIMARY_SEED,
    RESULTS_DIR,
    ROUND10_DEBERTA_COLLAPSED_META,
    ROUND10_DEGENERATE_CSV,
    ROUND10_ENCODER_SUMMARY,
    ROUND10_PER_RUN_CSV,
    COMPLETE_MARKER,
)
from .figures import generate_figures
from .report import write_report


def load_round10_degenerate() -> pd.DataFrame:
    return pd.read_csv(ROUND10_DEGENERATE_CSV)


def load_round10_encoder_summary() -> pd.DataFrame:
    return pd.read_csv(ROUND10_ENCODER_SUMMARY)


def load_round10_deberta_means() -> dict[str, float]:
    """Six clean-seed and naive eight-seed DeBERTa means from folder-10 outputs."""
    enc = load_round10_encoder_summary()
    clean6 = float(enc.loc[enc["model_id"] == "deberta_base", "benchmark_f1_mean"].iloc[0])
    per_run = pd.read_csv(ROUND10_PER_RUN_CSV)
    deb = per_run[per_run["model_id"] == "deberta_base"]
    all8 = float(deb["benchmark_f1"].mean())
    return {"clean6": clean6, "all8": all8}


def load_grid_markers() -> list[dict]:
    rows: list[dict] = []
    for point in GRID_POINTS:
        base = RESULTS_DIR / point.key
        marker = base / f"seed_{PRIMARY_SEED}" / COMPLETE_MARKER
        if marker.exists():
            rows.append(json.loads(marker.read_text(encoding="utf-8")))
    return rows


def grid_to_dataframe(grid_rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_key": r["run_key"],
                "lr": r["lr"],
                "warmup_label": r["warmup_label"],
                "seed": r["seed"],
                "best_epoch_val_f1": r["best_epoch_val_f1"],
                "best_val_f1": r["best_val_f1"],
                "benchmark_f1": r["benchmark_f1"],
                "bad_seed_guard": bool(r.get("bad_seed_guard", False)),
            }
            for r in grid_rows
            if r["seed"] == PRIMARY_SEED and not r.get("bad_seed_guard")
        ]
    )


def build_epoch_curves(grid_rows: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for r in grid_rows:
        if r["seed"] != PRIMARY_SEED or r.get("bad_seed_guard"):
            continue
        for ep in r.get("epoch_curve", []):
            rows.append(
                {
                    "run_key": r["run_key"],
                    "lr": r["lr"],
                    "warmup_label": r["warmup_label"],
                    "seed": r["seed"],
                    "epoch": ep["epoch"],
                    "train_loss": ep.get("train_loss"),
                    "val_loss": ep["val_loss"],
                    "val_f1": ep["val_f1"],
                }
            )
    return pd.DataFrame(rows)


def lr_and_warmup_effects(grid: pd.DataFrame) -> pd.DataFrame:
    """Explicit contrasts: lr main effect, warmup secondary (seed 42)."""
    g = grid.copy()
    rows: list[dict[str, Any]] = []

    for lr in sorted(g["lr"].unique()):
        sub = g[g["lr"] == lr]
        none = sub[sub["warmup_label"] == "none"]["benchmark_f1"].iloc[0]
        warm = sub[sub["warmup_label"] == "warmup_10pct"]["benchmark_f1"].iloc[0]
        rows.append(
            {
                "contrast": "warmup_at_fixed_lr",
                "lr": lr,
                "benchmark_f1_none": none,
                "benchmark_f1_warmup": warm,
                "delta": warm - none,
            }
        )

    for warm in ["none", "warmup_10pct"]:
        sub = g[g["warmup_label"] == warm]
        lo = sub[sub["lr"] == 1e-5]["benchmark_f1"].iloc[0]
        hi = sub[sub["lr"] == 2e-5]["benchmark_f1"].iloc[0]
        rows.append(
            {
                "contrast": "lr_at_fixed_warmup",
                "warmup_label": warm,
                "benchmark_f1_1e5": lo,
                "benchmark_f1_2e5": hi,
                "delta_2e5_minus_1e5": hi - lo,
            }
        )

    at_1e5 = g[g["lr"] == 1e-5]["benchmark_f1"].mean()
    at_2e5 = g[g["lr"] == 2e-5]["benchmark_f1"].mean()
    with_warm = g[g["warmup_label"] == "warmup_10pct"]["benchmark_f1"].mean()
    no_warm = g[g["warmup_label"] == "none"]["benchmark_f1"].mean()

    rows.append(
        {
            "contrast": "summary_lr_averaged_over_warmup",
            "mean_1e5": at_1e5,
            "mean_2e5": at_2e5,
            "delta_2e5_minus_1e5": at_2e5 - at_1e5,
        }
    )
    rows.append(
        {
            "contrast": "summary_warmup_averaged_over_lr",
            "mean_none": no_warm,
            "mean_warmup_10pct": with_warm,
            "delta_warmup_minus_none": with_warm - no_warm,
        }
    )
    return pd.DataFrame(rows)


def build_encoder_placement(
    grid_best: dict,
    deberta_means: dict[str, float],
) -> tuple[pd.DataFrame, float, float]:
    enc = load_round10_encoder_summary()
    others = enc[enc["model_id"] != "deberta_base"]
    omin, omax = float(others["benchmark_f1_mean"].min()), float(others["benchmark_f1_mean"].max())

    rows: list[dict] = []
    for _, row in enc.iterrows():
        rows.append(
            {
                "short_name": row["short_name"],
                "model_id": row["model_id"],
                "source": "round1_encoder",
                "benchmark_f1": float(row["benchmark_f1_mean"]),
                "ci_lo": float(row.get("benchmark_f1_ci_lo", row["benchmark_f1_mean"])),
                "ci_hi": float(row.get("benchmark_f1_ci_hi", row["benchmark_f1_mean"])),
            }
        )
    rows.append(
        {
            "short_name": "DeBERTa (grid best)",
            "model_id": "deberta_base",
            "source": "grid_best_seed42",
            "benchmark_f1": float(grid_best["benchmark_f1"]),
        }
    )
    rows.append(
        {
            "short_name": "DeBERTa (Round-1 clean 6)",
            "model_id": "deberta_base",
            "source": "round1_clean6",
            "benchmark_f1": deberta_means["clean6"],
        }
    )
    rows.append(
        {
            "short_name": "DeBERTa (Round-1 all 8)",
            "model_id": "deberta_base",
            "source": "round1_all8",
            "benchmark_f1": deberta_means["all8"],
        }
    )
    return pd.DataFrame(rows), omin, omax


def run_analysis() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_rows = load_grid_markers()
    if len(grid_rows) < 4:
        raise SystemExit(
            f"Analysis aborted: expected 4 recipe markers at seed {PRIMARY_SEED}, found {len(grid_rows)}."
        )

    grid = grid_to_dataframe(grid_rows)
    grid.to_csv(OUTPUT_DIR / "recipe_grid.csv", index=False)

    curves = build_epoch_curves(grid_rows)
    curves.to_csv(OUTPUT_DIR / "grid_epoch_curves.csv", index=False)

    effects = lr_and_warmup_effects(grid)
    effects.to_csv(OUTPUT_DIR / "lr_warmup_effects.csv", index=False)

    deg = load_round10_degenerate()
    deg.to_csv(OUTPUT_DIR / "degenerate_run_identification.csv", index=False)

    deberta_means = load_round10_deberta_means()
    best = grid.loc[grid["benchmark_f1"].idxmax()].to_dict()
    placement, omin, omax = build_encoder_placement(best, deberta_means)
    placement.to_csv(OUTPUT_DIR / "deberta_placement.csv", index=False)

    generate_figures(
        grid,
        placement,
        curves,
        deberta_all8_mean=deberta_means["all8"],
        deberta_clean6_mean=deberta_means["clean6"],
        grid_best_f1=float(best["benchmark_f1"]),
    )

    write_report(
        degenerate=deg,
        grid=grid,
        effects=effects,
        placement=placement,
        others_min=omin,
        others_max=omax,
        deberta_means=deberta_means,
        grid_best=best,
    )

    _print_stdout(grid, effects, deberta_means, omin, omax, best)


def _print_stdout(
    grid: pd.DataFrame,
    effects: pd.DataFrame,
    deberta_means: dict[str, float],
    omin: float,
    omax: float,
    best: dict,
) -> None:
    print("\n=== Recipe grid (seed 42) ===")
    print(grid.to_string(index=False))

    lr_row = effects[effects["contrast"] == "summary_lr_averaged_over_warmup"].iloc[0]
    w_row = effects[effects["contrast"] == "summary_warmup_averaged_over_lr"].iloc[0]
    print(
        f"\nLearning-rate effect (averaged over warmup): "
        f"1e-5 mean={lr_row['mean_1e5']:.3f}, 2e-5 mean={lr_row['mean_2e5']:.3f}, "
        f"delta={lr_row['delta_2e5_minus_1e5']:+.3f}"
    )
    print(
        f"Warmup effect (averaged over lr): "
        f"none mean={w_row['mean_none']:.3f}, 10pct mean={w_row['mean_warmup_10pct']:.3f}, "
        f"delta={w_row['delta_warmup_minus_none']:+.3f}"
    )

    print(
        f"\nEight-encoder Round-1 band (excl. DeBERTa): {omin:.3f} to {omax:.3f}"
    )
    print(f"DeBERTa Round-1 clean 6-seed mean: {deberta_means['clean6']:.3f}")
    print(f"DeBERTa Round-1 8-seed mean (incl. collapsed): {deberta_means['all8']:.3f}")
    print(f"DeBERTa grid best (seed 42): {best['benchmark_f1']:.3f} at lr={best['lr']:.0e} warmup={best['warmup_label']}")
    print("\nSingle-seed evidence: all grid points use seed 42 only; no multi-seed stability claim.")
    print("\n=== Analysis complete ===")
