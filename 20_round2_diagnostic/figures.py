"""Diagnostic figures (300 dpi, IEU-style)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from importlib import import_module

from .config import DPI, FIGURE_DIR, FOCUS_MODEL_IDS, MODEL_BY_ID, PALETTE

_r1cfg = import_module("10_round1_benchmark_kb.config")


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
    encoders = [m.model_id for m in _r1cfg.MODELS]
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
    fig.suptitle("Validation curves by encoder (clean seeds, seed-averaged)", y=1.01, fontsize=13)
    fig.tight_layout()
    _save(fig, "fig1_training_curves.png")


def figure2_two_axis_timing(traj: pd.DataFrame) -> None:
    _apply_style()
    sweep = traj[traj["source"] == "round1_sweep_recipe_match"].copy()
    if sweep.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    metrics = [
        ("benchmark_f1", "BioRED test F1"),
        ("kb_mrr_hard", "KB MRR (hard subset)"),
    ]
    points = ["val_loss_best", "val_f1_best"]
    x = np.arange(len(points))

    for ax, (col, ylab) in zip(axes, metrics):
        for mid in FOCUS_MODEL_IDS:
            sub = sweep[sweep["model_id"] == mid].set_index("trajectory_point").reindex(points)
            if sub[col].isna().all():
                continue
            label = MODEL_BY_ID[mid].short_name.replace("-base", "")
            ax.plot(x, sub[col].values, "o-", lw=2, markersize=8, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(["Val-loss-best\n(under-trained)", "Val-F1-best\n(well-trained)"])
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8, loc="best")
        ax.set_title("Sweep seed 42, lr 2e-5, no warmup")

    fig.suptitle(
        "Two-axis timing: benchmark vs hard-subset KB at two saved checkpoints",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "fig2_two_axis_timing.png")


def figure3_power(power_df: pd.DataFrame) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [MODEL_BY_ID[m].short_name.replace("-base", "") for m in power_df["model_id"]]
    x = np.arange(len(labels))
    w = 0.35
    effect = power_df["estimated_training_effect_hard"].fillna(0).values
    detect = power_df["approx_detectable_effect_hard"].fillna(0).values
    ax.bar(x - w / 2, effect, w, label="Est. training-amount effect (hard KB)", color=PALETTE["neutral"])
    ax.bar(x + w / 2, detect, w, label="Approx. detectable at 10 seeds (2×SE)", color=PALETTE["accent"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("KB MRR difference scale")
    ax.legend(fontsize=9)
    ax.set_title("Training-amount effect vs seed-noise band (hard subset)")
    fig.tight_layout()
    _save(fig, "fig3_power_check.png")
