"""Diagnostic figures (300 dpi, IEU-style)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared.models import MODELS

from .config import DPI, FIGURE_DIR, FOCUS_MODEL_IDS, MODEL_BY_ID, PALETTE


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": DPI,
            "figure.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_DIR / name,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)


def figure1_training_curves(mean_curves: pd.DataFrame) -> None:
    _apply_style()
    encoders = [m.model_id for m in MODELS]
    n = len(encoders)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.2 * nrow), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, mid in zip(axes, encoders):
        sub = mean_curves[mean_curves["model_id"] == mid]
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.plot(sub["epoch"], sub["val_f1"], "o-", color=PALETTE["neutral"], label="val F1", lw=1.5)
        ax2 = ax.twinx()
        ax2.plot(sub["epoch"], sub["val_loss"], "s--", color=PALETTE["accent"], label="val loss", lw=1.2)
        ax2.spines["top"].set_visible(False)
        name = MODEL_BY_ID[mid].short_name.replace("-base", "")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val F1")
        ax2.set_ylabel("Val loss")

    for ax in axes[len(encoders) :]:
        ax.set_visible(False)
    fig.suptitle("Validation curves by encoder (seed-averaged)", y=1.01, fontsize=13)
    fig.tight_layout()
    _save(fig, "fig1_training_curves.png")


def figure2_two_axis_timing(traj: pd.DataFrame) -> None:
    _apply_style()
    pe = traj[traj["source"] == "matrix_per_epoch"].copy()
    if pe.empty or pe["kb_mrr_hard"].isna().all() or pe["benchmark_f1"].isna().all():
        print("  Skipping fig2 (per-epoch benchmark/KB not scored yet)")
        return

    mean_pe = (
        pe.groupby(["model_id", "epoch"])[["benchmark_f1", "kb_mrr_hard"]]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    metrics = [
        ("benchmark_f1", "BioRED test F1"),
        ("kb_mrr_hard", "KB MRR (hard subset)"),
    ]

    for ax, (col, ylab) in zip(axes, metrics):
        for mid in FOCUS_MODEL_IDS:
            sub = mean_pe[mean_pe["model_id"] == mid].sort_values("epoch")
            if sub[col].isna().all():
                continue
            label = MODEL_BY_ID[mid].short_name.replace("-base", "")
            ax.plot(sub["epoch"], sub[col], "o-", lw=2, markersize=6, label=label)
        ax.set_xlabel("Training epoch")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Two-axis timing: benchmark vs hard-subset KB along per-epoch checkpoints "
        "(focus encoders, seed-averaged)",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "fig2_two_axis_timing.png")


def figure3_power(power_df: pd.DataFrame) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labels = [MODEL_BY_ID[m].short_name.replace("-base", "") for m in power_df["model_id"]]
    x = np.arange(len(labels))
    w = 0.28
    effect = power_df["estimated_training_effect_hard"].fillna(0).values
    detect = power_df["approx_detectable_effect_hard"].fillna(0).values
    r1_band = power_df["approx_detectable_vs_r1_pool_sd"].fillna(0).values
    ax.bar(x - w, effect, w, label="Est. training-amount effect (hard KB)", color=PALETTE["neutral"])
    ax.bar(x, detect, w, label="Detectable at 10 seeds (focus hard SD)", color=PALETTE["accent"], alpha=0.85)
    ax.bar(
        x + w,
        r1_band,
        w,
        label="Detectable vs Round 1 pool SD",
        color="#888888",
        alpha=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("KB MRR difference scale")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Training-amount effect vs seed-noise bands (hard subset)")
    fig.tight_layout()
    _save(fig, "fig3_power_check.png")


def figure4_three_point_paired(three_pt: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Paired deltas under multiple well-trained definitions (selection decoupling)."""
    from .three_point_timing import WELL_DEF_LABELS, WELL_DEF_VAL_F1, WELL_DEFS

    _apply_style()
    if three_pt.empty:
        print("  Skipping fig4 (no three-point data)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), sharey=True)
    colors = {
        "pubmedbert_base": PALETTE["neutral"],
        "roberta_base": "#228833",
        "distilbert_base": PALETTE["accent"],
    }

    bench_sd = float(summary["r1_benchmark_f1_sd"].median()) if not summary.empty else np.nan
    kb_sd = float(summary["r1_kb_hard_mrr_sd"].median()) if not summary.empty else np.nan

    for ax, well_def in zip(axes, WELL_DEFS):
        pair_col = f"pairable_{well_def}"
        pair = three_pt[three_pt[pair_col]].copy()
        title = WELL_DEF_LABELS[well_def].split(" (")[0]
        if pair.empty:
            ax.set_title(f"{title}\n(no pairable seeds)", fontsize=10)
            ax.axhline(0, color=PALETTE["grid"], lw=0.8)
            ax.axvline(0, color=PALETTE["grid"], lw=0.8)
            continue

        for mid in FOCUS_MODEL_IDS:
            sub = pair[pair["model_id"] == mid]
            if sub.empty:
                continue
            label = MODEL_BY_ID[mid].short_name.replace("-base", "")
            ax.scatter(
                sub[f"delta_benchmark_{well_def}"],
                sub[f"delta_kb_hard_{well_def}"],
                s=65,
                alpha=0.85,
                color=colors.get(mid, PALETTE["neutral"]),
                edgecolors="white",
                linewidths=0.6,
                label=label,
                zorder=3,
            )

        if not np.isnan(bench_sd) and bench_sd > 0:
            ax.axvline(bench_sd, color="#888888", ls=":", lw=1.0, alpha=0.75)
            ax.axvline(-bench_sd, color="#888888", ls=":", lw=1.0, alpha=0.75)
        if not np.isnan(kb_sd) and kb_sd > 0:
            ax.axhline(kb_sd, color="#888888", ls=":", lw=1.0, alpha=0.75)
            ax.axhline(-kb_sd, color="#888888", ls=":", lw=1.0, alpha=0.75)
        if not np.isnan(bench_sd) and not np.isnan(kb_sd):
            from matplotlib.patches import Rectangle

            ax.add_patch(
                Rectangle(
                    (-bench_sd, -kb_sd),
                    2 * bench_sd,
                    2 * kb_sd,
                    fill=False,
                    ls="--",
                    ec="#888888",
                    lw=0.9,
                    alpha=0.65,
                    zorder=1,
                )
            )

        ax.axhline(0, color=PALETTE["grid"], lw=0.8, zorder=0)
        ax.axvline(0, color=PALETTE["grid"], lw=0.8, zorder=0)
        ax.set_xlabel("Delta benchmark F1\n(well minus epoch 1)", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Delta KB MRR hard\n(well minus epoch 1)", fontsize=10)
        short = "val_f1-best" if well_def == WELL_DEF_VAL_F1 else well_def.replace("_", " ")
        ax.set_title(short, fontsize=10)
        if ax is axes[-1]:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Three-point paired timing: robustness to well-trained definition\n"
        "(dashed box = median Round 1 seed-noise band; per-seed dots)",
        fontsize=11,
        y=1.03,
    )
    fig.tight_layout()
    _save(fig, "fig4_three_point_paired.png")
