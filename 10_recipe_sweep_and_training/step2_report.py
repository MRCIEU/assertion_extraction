"""Step-2 matrix final report: CSVs, figure, and acceptance prose."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared.constants import TRAIN_SEEDS
from shared.models import MODELS

from .config import (
    MATRIX_FIGURE_DIR,
    MATRIX_OUTPUT_DIR,
    MATRIX_REPORT_PATH,
    REPORT_DIR,
    require_chosen_recipe,
)
from .step2_matrix_summary import (
    DEBERTA_MODEL_ID,
    deberta_gate_verdict,
    encoder_summary,
    load_matrix_table,
    matrix_footprint_gib,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _apply_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
        }
    )


def plot_benchmark_heatmap(df: pd.DataFrame) -> Path:
    _apply_figure_style()
    MATRIX_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    ok = df[~df["missing"].astype(bool)].copy()
    pivot = ok.pivot(index="short_name", columns="seed", values="benchmark_f1")
    order = [m.short_name for m in MODELS if m.short_name in pivot.index]
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = pivot.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=max(0.85, np.nanmax(data)))

    ax.set_xticks(range(len(TRAIN_SEEDS)))
    ax.set_xticklabels([str(s) for s in TRAIN_SEEDS])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Encoder")
    ax.set_title("Matrix benchmark F1 (BioRED test, best val_f1 checkpoint)")

    for i, name in enumerate(order):
        for j, seed in enumerate(TRAIN_SEEDS):
            val = data[i, j]
            if np.isnan(val):
                continue
            color = "white" if val < 0.35 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Benchmark F1")
    fig.tight_layout()

    out = MATRIX_FIGURE_DIR / "matrix_benchmark_f1_heatmap.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _prose_sections(
    df: pd.DataFrame,
    enc: pd.DataFrame,
    gate_pass: bool,
    gate_reasons: list[str],
    footprint_gib: float,
) -> list[str]:
    recipe = require_chosen_recipe()
    expected = len(MODELS) * len(TRAIN_SEEDS)
    n_done = int((~df["missing"].astype(bool)).sum())
    collapsed = df[df.get("collapsed", False).astype(bool) & ~df["missing"].astype(bool)]

    paras: list[str] = [
        (
            f"Step 2 trained {n_done}/{expected} runs at lr={recipe.lr}, warmup={recipe.warmup_label}, "
            "val_f1 checkpoint selection, on clean offset-marked training data (marker_method=offset). "
            f"Each run saved fp16 per-epoch weights, an fp32 best checkpoint, and one BioRED benchmark F1 "
            f"at the best checkpoint. Matrix storage footprint is {footprint_gib:.1f} GiB."
        )
    ]

    if n_done == expected and df["recipe_match"].all():
        paras.append(
            f"All 72 matrix_complete.json markers are present and record recipe_lr={recipe.lr}/"
            f"{recipe.warmup_label}. Checkpoints and training logs are under "
            "data/10_recipe_sweep_and_training/matrix/."
        )
    else:
        paras.append(
            f"Completion check: {n_done}/{expected} markers; "
            f"recipe mismatches={(~df['recipe_match']).sum() if 'recipe_match' in df else 'n/a'}."
        )

    top = enc.iloc[0]
    bottom = enc.iloc[-1]
    paras.append(
        f"Across seeds 42-49, mean benchmark F1 ranges from {bottom['benchmark_f1_mean']:.3f} "
        f"({bottom['short_name']}) to {top['benchmark_f1_mean']:.3f} ({top['short_name']}). "
        f"Encoder spread (max mean minus min mean) is "
        f"{float(enc['benchmark_f1_mean'].max() - enc['benchmark_f1_mean'].min()):.3f}."
    )

    deb = df[(df["model_id"] == DEBERTA_MODEL_ID) & ~df["missing"].astype(bool)].sort_values("seed")
    deb_lines = [f"seed {int(r['seed'])}: benchmark F1 {float(r['benchmark_f1']):.3f}" for _, r in deb.iterrows()]
    paras.append(
        "DeBERTa per-seed benchmark F1 (hard acceptance gate): "
        + "; ".join(deb_lines)
        + "."
    )

    if gate_pass:
        paras.append(
            "ACCEPTANCE GATE PASS: DeBERTa is stable across all eight seeds with no collapse and "
            f"no systematic suppression relative to peer encoders. Recipe {recipe.lr}/{recipe.warmup_label} "
            "is confirmed for folder 11, subject to your review."
        )
    else:
        paras.append(
            "ACCEPTANCE GATE FAIL: " + " ".join(gate_reasons)
            + f". Do not proceed to folder 11 with this matrix at lr={recipe.lr}/{recipe.warmup_label}. "
            "If even the lowest learning rate cannot stabilise DeBERTa across seeds, treat DeBERTa's "
            "cross-seed fragility as a deliberate finding rather than trying another recipe."
        )

    if not collapsed.empty:
        cl = ", ".join(
            f"{r['short_name']} seed={int(r['seed'])} (val_f1={float(r['best_val_f1']):.3f}, "
            f"benchmark F1={float(r['benchmark_f1']):.3f})"
            for _, r in collapsed.iterrows()
        )
        paras.append(f"Collapsed runs ({len(collapsed)}): {cl}.")

    return paras


def write_matrix_report(
    df: pd.DataFrame,
    enc: pd.DataFrame,
    gate_pass: bool,
    gate_reasons: list[str],
    figure_path: Path,
    footprint_gib: float,
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Step 2 matrix training report",
        "",
        "Full-matrix training on clean offset-marked data. Recipe from the clean-data step-1 advisory.",
        "",
        "## Summary",
        "",
    ]
    for para in _prose_sections(df, enc, gate_pass, gate_reasons, footprint_gib):
        lines.append(para)
        lines.append("")
    lines.extend(
        [
            "",
            "## Encoder mean benchmark F1 (seeds 42-49)",
            "",
            "| Encoder | mean F1 | std | min | max | collapsed |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, r in enc.iterrows():
        lines.append(
            f"| {r['short_name']} | {r['benchmark_f1_mean']:.3f} | {r['benchmark_f1_std']:.3f} | "
            f"{r['benchmark_f1_min']:.3f} | {r['benchmark_f1_max']:.3f} | {int(r['n_collapsed'])} |"
        )
    lines.extend(
        [
            f"See also the benchmark F1 heatmap ({figure_path.name}) and CSVs under outputs/10_recipe_sweep_and_training/matrix/.",
            "",
        ]
    )
    MATRIX_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return MATRIX_REPORT_PATH


def run_matrix_report() -> int:
    recipe = require_chosen_recipe()
    df = load_matrix_table()
    enc = encoder_summary(df)
    gate_pass, gate_reasons = deberta_gate_verdict(df, enc)
    footprint = matrix_footprint_gib()

    MATRIX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_run_path = MATRIX_OUTPUT_DIR / "matrix_per_run.csv"
    enc_path = MATRIX_OUTPUT_DIR / "matrix_encoder_summary.csv"
    df.to_csv(per_run_path, index=False)
    enc.to_csv(enc_path, index=False)

    figure_path = plot_benchmark_heatmap(df)
    report_path = write_matrix_report(df, enc, gate_pass, gate_reasons, figure_path, footprint)

    expected = len(MODELS) * len(TRAIN_SEEDS)
    n_done = int((~df["missing"].astype(bool)).sum())
    collapsed_n = int(df.get("collapsed", pd.Series(dtype=bool)).sum())

    _log("\n=== Step-2 matrix final report ===")
    _log(f"  Recipe: lr={recipe.lr}, warmup={recipe.warmup_label}")
    _log(f"  Completion: {n_done}/{expected}")
    _log(f"  Collapsed runs: {collapsed_n}")
    _log(f"  Footprint: {footprint:.2f} GiB")
    _log(f"  DeBERTa gate: {'PASS' if gate_pass else 'FAIL'}")
    for r in gate_reasons:
        _log(f"    {r}")
    _log(f"\n  Wrote {per_run_path}")
    _log(f"  Wrote {enc_path}")
    _log(f"  Wrote {figure_path}")
    _log(f"  Wrote {report_path}")

    if gate_pass:
        _log(
            f"\nDeBERTa stable across all 8 seeds (no collapse, no systematic suppression) "
            f"-> {recipe.lr}/{recipe.warmup_label} confirmed, cleared to proceed to folder 11 "
            "(your decision)."
        )
        return 0

    _log(
        f"\nDeBERTa collapses/suppressed on one or more seeds -> {recipe.lr}/{recipe.warmup_label} "
        "also unstable; DO NOT proceed; stop and report."
    )
    return 1
