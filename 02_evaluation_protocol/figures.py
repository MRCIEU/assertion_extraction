"""Figures for step 02."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURE_DIR, OUTPUT_DIR


def plot_composition(targets_df: pd.DataFrame) -> None:
    comp = (
        targets_df.groupby(["label", "pair_type", "scope"])
        .size()
        .reset_index(name="count")
    )
    comp.to_csv(OUTPUT_DIR / "02_evaluation_protocol_composition.csv", index=False)

    primary = targets_df[targets_df["scope"] == "primary"]
    pivot = primary.groupby(["pair_type", "label"]).size().unstack(fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pivot.plot(kind="bar", ax=axes[0], color=["#DD8452", "#4C72B0"])
    axes[0].set_title("Primary scope composition")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=15)

    desc = targets_df[targets_df["scope"] == "descriptive_only"]
    if not desc.empty:
        dpivot = desc.groupby(["pair_type", "label"]).size().unstack(fill_value=0)
        dpivot.plot(kind="bar", ax=axes[1], color=["#DD8452", "#4C72B0"])
    axes[1].set_title("Descriptive-only scope")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_evaluation_protocol_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_difficulty(targets_df: pd.DataFrame) -> None:
    ok = targets_df[targets_df["difficulty_status"] == "ok"].copy()
    ok.to_csv(OUTPUT_DIR / "02_evaluation_protocol_difficulty_details.csv", index=False)

    summary = (
        ok.groupby("label")[["sentence_distance", "co_sentence"]]
        .agg(
            sentence_distance_mean=("sentence_distance", "mean"),
            sentence_distance_median=("sentence_distance", "median"),
            co_sentence_rate=("co_sentence", "mean"),
            n=("sentence_distance", "count"),
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "02_evaluation_protocol_difficulty_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for label, color in [("PRESENT", "#4C72B0"), ("ABSENT", "#DD8452")]:
        sub = ok[ok["label"] == label]["sentence_distance"].dropna()
        if len(sub):
            axes[0].hist(sub, bins=range(0, int(sub.max()) + 2), alpha=0.6, label=label, color=color)
    axes[0].set_xlabel("Sentence distance")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Positive vs negative sentence distance")
    axes[0].legend()

    labels = summary["label"].tolist()
    rates = summary["co_sentence_rate"].tolist()
    axes[1].bar(labels, rates, color=["#4C72B0", "#DD8452"])
    axes[1].set_ylabel("Co-sentence rate")
    axes[1].set_title("Entity pairs in same sentence")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_evaluation_protocol_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ceiling(baseline_df: pd.DataFrame) -> None:
    pivot = baseline_df.pivot(index="baseline", columns="metric", values="score")
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Trivial baseline scores (red=saturated ceiling risk)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_evaluation_protocol_ceiling_check.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
