"""Figures for hyperparameter sweep diagnostic."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURE_DIR


def _save(fig, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_val_curves(curves: pd.DataFrame, summary: pd.DataFrame) -> None:
    for model_id in summary["model_id"].unique():
        sub = curves[curves["model_id"] == model_id]
        settings = sub.groupby(["lr", "warmup_label"]).ngroups
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for (lr, warmup), g in sub.groupby(["lr", "warmup_label"]):
            label = f"lr={lr:.0e} {warmup}"
            axes[0].plot(g["epoch"], g["val_loss"], marker="o", label=label, alpha=0.8)
            axes[1].plot(g["epoch"], g["val_f1"], marker="o", label=label, alpha=0.8)
        axes[0].set_ylabel("Validation loss")
        axes[1].set_ylabel("Validation F1")
        axes[1].set_xlabel("Epoch")
        axes[0].set_title(f"Validation curves — {model_id}")
        axes[1].legend(fontsize=7, ncol=2, loc="lower right")
        _save(fig, f"val_curves_{model_id}.png")


def plot_benchmark_spread(summary: pd.DataFrame, spread: pd.DataFrame) -> None:
    if spread.empty:
        return
    top = spread.head(6)
    fig, ax = plt.subplots(figsize=(9, 5))
    xlabels = [f"lr={r['lr']:.0e}\n{r['warmup_label']}" for _, r in top.iterrows()]
    for i, (_, row) in enumerate(top.iterrows()):
        sub = summary[
            (summary["lr"] == row["lr"]) & (summary["warmup_label"] == row["warmup_label"])
        ].sort_values("benchmark_f1_val_loss_ckpt")
        xs = [i + (j - 1) * 0.15 for j in range(len(sub))]
        ax.scatter(xs, sub["benchmark_f1_val_loss_ckpt"], s=60, alpha=0.85)
        for x, (_, r) in zip(xs, sub.iterrows()):
            ax.annotate(r["short_name"].split("-")[0], (x, r["benchmark_f1_val_loss_ckpt"]), fontsize=7, ha="center")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Self-measured BioRED test F1")
    ax.set_title("Benchmark F1 spread across architectures (top settings by spread)")
    _save(fig, "benchmark_f1_spread_by_setting.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        spread["warmup_label"] + " lr=" + spread["lr"].map(lambda x: f"{x:.0e}"),
        spread["benchmark_f1_spread"],
        alpha=0.85,
    )
    ax.set_xlabel("Benchmark F1 spread (max − min across 3 architectures)")
    ax.set_title("Benchmark gradient width by hyperparameter setting")
    _save(fig, "benchmark_f1_spread_all_settings.png")


def plot_best_epoch(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    agg = summary.groupby(["lr", "warmup_label"]).agg(
        mean_best_loss_epoch=("best_epoch_by_val_loss", "mean"),
        mean_best_f1_epoch=("best_epoch_by_val_f1", "mean"),
    ).reset_index()
    labels = [f"{r['warmup_label']}\nlr={r['lr']:.0e}" for _, r in agg.iterrows()]
    x = range(len(agg))
    w = 0.35
    ax.bar([i - w / 2 for i in x], agg["mean_best_loss_epoch"], width=w, label="best epoch (val_loss)", alpha=0.85)
    ax.bar([i + w / 2 for i in x], agg["mean_best_f1_epoch"], width=w, label="best epoch (val_f1)", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Epoch")
    ax.set_title("Checkpoint selection: val_loss vs val_f1 (mean across 3 architectures)")
    ax.legend()
    _save(fig, "best_epoch_by_criterion.png")


def plot_criterion_comparison(a1: dict) -> None:
    per_run = a1["per_run"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        per_run["benchmark_f1_val_loss_ckpt"],
        per_run["benchmark_f1_val_f1_ckpt"],
        alpha=0.8,
    )
    lo = min(per_run["benchmark_f1_val_loss_ckpt"].min(), per_run["benchmark_f1_val_f1_ckpt"].min()) - 0.02
    hi = max(per_run["benchmark_f1_val_loss_ckpt"].max(), per_run["benchmark_f1_val_f1_ckpt"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y=x")
    ax.set_xlabel("Benchmark F1 (val_loss checkpoint)")
    ax.set_ylabel("Benchmark F1 (val_f1 checkpoint)")
    ax.set_title("Paired checkpoint criterion comparison (24 runs)")
    ax.legend()
    _save(fig, "criterion_comparison_scatter.png")


def plot_spread_by_criterion(a2: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for criterion, sub in a2.groupby("criterion"):
        labels = [f"{r['lr']:.0e}\n{r['warmup_label'][:6]}" for _, r in sub.iterrows()]
        ax.plot(range(len(sub)), sub["benchmark_f1_spread"], marker="o", label=criterion, alpha=0.85)
    ax.axhline(0.05, color="gray", linestyle=":", label="spread threshold 0.05")
    ax.set_ylabel("Benchmark F1 spread (3 architectures)")
    ax.set_title("Cross-architecture spread by lr×warmup under each criterion")
    ax.legend()
    _save(fig, "spread_by_criterion.png")


def plot_architecture_lr_trend(trend: pd.DataFrame) -> None:
    for criterion in trend["criterion"].unique():
        sub = trend[trend["criterion"] == criterion]
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_id, g in sub.groupby("model_id"):
            # average across warmups for cleaner line
            agg = g.groupby("lr")["benchmark_f1"].mean().reset_index()
            ax.plot(agg["lr"], agg["benchmark_f1"], marker="o", label=model_id, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Benchmark F1")
        ax.set_title(f"Architecture benchmark F1 vs lr ({criterion} checkpoint)")
        ax.legend(fontsize=8)
        _save(fig, f"architecture_lr_trend_{criterion}.png")


def generate_objective_figures(results: dict) -> None:
    a1 = results["analysis1"]
    a2 = results["analysis2"]
    plot_criterion_comparison(a1)
    plot_spread_by_criterion(a2)
    plot_architecture_lr_trend(results["architecture_lr_trend"])


def generate_sweep_figures(summary: pd.DataFrame, curves: pd.DataFrame, spread: pd.DataFrame) -> None:
    if summary.empty:
        return
    plot_val_curves(curves, summary)
    plot_benchmark_spread(summary, spread)
    plot_best_epoch(summary)
