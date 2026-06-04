"""Figures for encoder recipe check."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURE_DIR, PRIMARY_SEED


def _save(fig, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_val_curves(
    curves: pd.DataFrame,
    grid_rows: list[dict],
    r10_reference: pd.DataFrame | None = None,
) -> None:
    """Per-epoch val_loss and val_f1 for primary-seed grid runs."""
    primary = curves[curves["seed"] == PRIMARY_SEED].copy()
    if primary.empty:
        primary = curves.copy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run_key, sub in primary.groupby("run_key"):
        label = sub["warmup_label"].iloc[0] if "warmup_label" in sub.columns else run_key
        lr = sub["lr"].iloc[0] if "lr" in sub.columns else ""
        tag = f"lr={lr:.0e} {label}"
        axes[0].plot(sub["epoch"], sub["val_loss"], marker="o", label=tag, alpha=0.85)
        axes[1].plot(sub["epoch"], sub["val_f1"], marker="o", label=tag, alpha=0.85)

    if r10_reference is not None and not r10_reference.empty:
        axes[0].plot(
            r10_reference["epoch"],
            r10_reference["val_loss"],
            "k--",
            alpha=0.6,
            label="Round-1 DeBERTa seed 45 (no warmup)",
        )
        axes[1].plot(
            r10_reference["epoch"],
            r10_reference["val_f1"],
            "k--",
            alpha=0.6,
            label="Round-1 DeBERTa seed 45 (no warmup)",
        )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation loss")
    axes[0].set_title("Validation loss curves (DeBERTa recipe grid)")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation F1")
    axes[1].set_title("Validation F1 curves (DeBERTa recipe grid)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "11_val_curves_grid.png")


def plot_encoder_strip(vs_group: pd.DataFrame) -> None:
    """Benchmark F1 range across Round-1 encoders with DeBERTa old/new marked."""
    r1 = vs_group[vs_group["source"] == "round1_encoder_mean"].sort_values("benchmark_f1")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(r1["label"], r1["benchmark_f1"], alpha=0.75, color="steelblue", label="Round-1 encoder mean")

    markers = vs_group[vs_group["source"] != "round1_encoder_mean"]
    colors = {"round1_mean_old": "red", "recipe_grid_primary": "darkgreen", "recipe_grid_best_any": "orange"}
    for _, row in markers.iterrows():
        ax.axvline(
            row["benchmark_f1"],
            color=colors.get(row["source"], "purple"),
            linestyle="--",
            linewidth=2,
            label=row["label"],
        )

    ax.set_xlabel("Self-measured BioRED test presence F1")
    ax.set_title("Encoder benchmark F1: Round-1 group vs DeBERTa diagnostic")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="lower right")
    _save(fig, "11_encoder_benchmark_strip.png")
