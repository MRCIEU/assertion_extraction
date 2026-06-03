"""Figures for step 04."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURE_DIR, MODEL_BY_ID, OUTPUT_DIR


def plot_benchmark_vs_kb(table: pd.DataFrame, kb_metric: str = "mrr") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for _, row in table.iterrows():
        ax.scatter(row["benchmark_f1"], row[kb_metric], s=80)
        ax.annotate(row["short_name"], (row["benchmark_f1"], row[kb_metric]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Known benchmark F1")
    ax.set_ylabel(f"KB ranking metric ({kb_metric})")
    ax.set_title("Benchmark rank vs KB-ranking (B)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_pilot_study_benchmark_vs_kb.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ece_vs_benchmark(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for _, row in table.iterrows():
        ax.scatter(row["benchmark_f1"], row["ece"], s=80)
        ax.annotate(row["short_name"], (row["benchmark_f1"], row["ece"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Known benchmark F1")
    ax.set_ylabel("ECE (lower = better calibration to CIViC inclusion)")
    ax.set_title("Benchmark vs calibration (C2)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_pilot_study_ece_vs_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_diagrams(scores_df: pd.DataFrame, n_bins: int = 10) -> None:
    from .metrics_calibration import reliability_bins

    model_ids = sorted(scores_df["model_id"].unique())
    n = len(model_ids)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, model_id in zip(axes, model_ids):
        sub = scores_df[scores_df["model_id"] == model_id]
        y = sub["label_civic_curated_positive"].astype(int).values
        p = sub["score"].values
        bins = reliability_bins(y, p, n_bins=n_bins)
        valid = bins[bins["n"] > 0]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
        ax.bar(
            valid["bin_center"],
            valid["empirical_rate"],
            width=1.0 / n_bins * 0.9,
            alpha=0.5,
            label="empirical",
        )
        ax.plot(valid["bin_center"], valid["mean_confidence"], "o-", label="confidence")
        name = MODEL_BY_ID.get(model_id, model_id)
        ax.set_title(getattr(name, "short_name", model_id))
        ax.set_xlabel("Score bin")
        ax.set_ylim(0, 1)
        if ax is axes[0]:
            ax.set_ylabel("CIViC inclusion rate")
            ax.legend(fontsize=7)
    fig.suptitle("Reliability diagrams (calibration vs CIViC inclusion)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_pilot_study_reliability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions(scores_df: pd.DataFrame) -> None:
    model_ids = sorted(scores_df["model_id"].unique())
    fig, axes = plt.subplots(1, len(model_ids), figsize=(4 * len(model_ids), 4), sharey=True)
    if len(model_ids) == 1:
        axes = [axes]
    for ax, model_id in zip(axes, model_ids):
        sub = scores_df[scores_df["model_id"] == model_id]["score"].astype(float)
        ax.hist(sub, bins=30, range=(0, 1), alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.axvline(sub.mean(), color="red", linestyle="--", label=f"mean={sub.mean():.2f}")
        name = MODEL_BY_ID.get(model_id)
        ax.set_title(getattr(name, "short_name", model_id))
        ax.set_xlabel("P(present)")
        if ax is axes[0]:
            ax.set_ylabel("Count")
            ax.legend(fontsize=7)
    fig.suptitle("Score distributions (sanity: non-degenerate P(present))", y=1.02)
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "04_pilot_study_score_distributions.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def export_tables(
    ranking_df: pd.DataFrame,
    ranking_baselines: pd.DataFrame,
    calibration_df: pd.DataFrame,
    calibration_baselines: pd.DataFrame,
    bench_kb_table: pd.DataFrame,
    rank_flips_df: pd.DataFrame,
    score_dist_df: pd.DataFrame,
) -> None:
    ranking_df.to_csv(OUTPUT_DIR / "04_pilot_study_ranking_metrics.csv", index=False)
    ranking_baselines.to_csv(OUTPUT_DIR / "04_pilot_study_ranking_baselines.csv", index=False)
    calibration_df.to_csv(OUTPUT_DIR / "04_pilot_study_calibration_ece.csv", index=False)
    calibration_baselines.to_csv(OUTPUT_DIR / "04_pilot_study_calibration_baselines.csv", index=False)
    bench_kb_table.to_csv(OUTPUT_DIR / "04_pilot_study_benchmark_vs_kb.csv", index=False)
    rank_flips_df.to_csv(OUTPUT_DIR / "04_pilot_study_rank_flips.csv", index=False)
    score_dist_df.to_csv(OUTPUT_DIR / "04_pilot_study_score_distribution.csv", index=False)


def plot_distance_correlation(corr_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = corr_df["model_name"].tolist()
    x = np.arange(len(labels))
    ax.bar(x - 0.2, corr_df["pearson_r_proximity"], width=0.35, label="Pearson r (proximity score)")
    ax.bar(x + 0.2, corr_df["pointbiserial_r_co_sentence"], width=0.35, label="Point-biserial r (co-sentence)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Correlation with entity proximity")
    ax.set_title("Model score vs entity-distance correlation")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_distance_score_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_subset_ranking(subset_df: pd.DataFrame) -> None:
    hard = subset_df[subset_df["subset"] == "hard_cross_sentence"]
    if hard.empty:
        return
    models = [m for m in hard["ranker"].unique() if m != "distance_ranker"]
    model_labels = [
        MODEL_BY_ID[m].short_name if m in MODEL_BY_ID else m for m in models
    ] + ["Distance ranker"]
    mrr_vals = [hard[hard["ranker"] == m]["mrr"].iloc[0] for m in models]
    mrr_vals.append(hard[hard["ranker"] == "distance_ranker"]["mrr"].iloc[0])
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0"] * len(models) + ["#DD8452"]
    ax.bar(model_labels, mrr_vals, color=colors)
    ax.set_ylabel("MRR")
    ax.set_title("Hard subset (cross-sentence): trained models vs distance ranker")
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(mrr_vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_distance_hard_subset_mrr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_positive_distance_distribution(hist_df: pd.DataFrame, co_sentence_frac: float) -> None:
    if hist_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(hist_df["sentence_distance"].astype(str), hist_df["n_positives"], color="#55A868")
    ax.set_xlabel("Sentence distance between entities")
    ax.set_ylabel("CIViC-curated positives")
    ax.set_title(f"Positive distance distribution (co-sentence: {co_sentence_frac:.1%})")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_positive_distance_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
