"""Figures for Round 1."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURE_DIR, MODEL_BY_ID, PAIR_TYPES
from .metrics_calibration import reliability_bins


def _save(fig, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_benchmark_kb_scatter(encoder_df: pd.DataFrame) -> None:
    for pt, col in [
        ("gene-drug", "kb_mrr_gene_drug_mean"),
        ("gene-disease", "kb_mrr_gene_disease_mean"),
    ]:
        if col not in encoder_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        x = encoder_df["benchmark_f1_mean"]
        y = encoder_df[col]
        xerr = [
            encoder_df["benchmark_f1_mean"] - encoder_df.get("benchmark_f1_ci_lo", x),
            encoder_df.get("benchmark_f1_ci_hi", x) - encoder_df["benchmark_f1_mean"],
        ]
        yerr = [
            encoder_df[col] - encoder_df.get(f"{col.replace('_mean', '_ci_lo')}", y),
            encoder_df.get(f"{col.replace('_mean', '_ci_hi')}", y) - encoder_df[col],
        ]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=3, alpha=0.8)
        for _, row in encoder_df.iterrows():
            ax.annotate(row["short_name"], (row["benchmark_f1_mean"], row[col]), fontsize=7, alpha=0.8)
        ax.set_xlabel("Self-measured BioRED test presence F1")
        ax.set_ylabel(f"KB MRR ({pt})")
        ax.set_title(f"Benchmark F1 vs KB ranking ({pt})")
        _save(fig, f"10_benchmark_vs_kb_mrr_{pt.replace('-', '_')}.png")


def plot_benchmark_f1_range(encoder_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    order = encoder_df.sort_values("benchmark_f1_mean")
    ax.barh(order["short_name"], order["benchmark_f1_mean"], xerr=0.02, capsize=3, alpha=0.8)
    ax.set_xlabel("Self-measured BioRED test presence F1 (encoder mean)")
    ax.set_title("Benchmark quality gradient across encoders")
    _save(fig, "10_benchmark_f1_range.png")


def plot_easy_hard_bars(subset_df: pd.DataFrame) -> None:
    hard = subset_df[subset_df["subset"] == "hard_cross_sentence"].copy()
    if hard.empty:
        return
    enc = hard[hard["model_id"] != "distance_ranker"].groupby("model_id")["mrr"].mean().reset_index()
    enc["short_name"] = enc["model_id"].map(lambda m: MODEL_BY_ID[m].short_name if m in MODEL_BY_ID else m)
    dr = hard[hard["model_id"] == "distance_ranker"]["mrr"].iloc[0] if (hard["model_id"] == "distance_ranker").any() else 0

    fig, ax = plt.subplots(figsize=(9, 4))
    enc = enc.sort_values("mrr")
    ax.barh(enc["short_name"], enc["mrr"], alpha=0.8, label="trained models (hard subset mean)")
    ax.axvline(dr, color="red", linestyle="--", label=f"distance ranker ({dr:.3f})")
    ax.set_xlabel("MRR on hard (cross-sentence) subset")
    ax.set_title("Ranking validity: hard subset vs distance ranker")
    ax.legend()
    _save(fig, "10_easy_hard_hard_subset.png")


def plot_benchmark_ece_scatter(encoder_df: pd.DataFrame) -> None:
    if "ece_mean" not in encoder_df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(encoder_df["benchmark_f1_mean"], encoder_df["ece_mean"], s=60, alpha=0.8)
    for _, row in encoder_df.iterrows():
        ax.annotate(row["short_name"], (row["benchmark_f1_mean"], row["ece_mean"]), fontsize=7)
    ax.set_xlabel("Self-measured BioRED test presence F1")
    ax.set_ylabel("ECE vs CIViC inclusion")
    ax.set_title("Benchmark F1 vs calibration (lower ECE = better)")
    _save(fig, "10_benchmark_vs_ece.png")


def plot_reliability_diagrams(scores_df: pd.DataFrame, n_models: int = 3) -> None:
    top = (
        scores_df.groupby("run_id")["score"]
        .count()
        .sort_values(ascending=False)
        .head(n_models)
        .index.tolist()
    )
    for run_id in top[:3]:
        sub = scores_df[scores_df["run_id"] == run_id]
        y = sub["label_civic_curated_positive"].astype(int).values
        p = sub["score"].values.astype(float)
        bins = reliability_bins(y, p)
        fig, ax = plt.subplots(figsize=(5, 4))
        valid = bins["n"] > 0
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
        ax.bar(
            bins.loc[valid, "bin_center"],
            bins.loc[valid, "empirical_rate"],
            width=0.08,
            alpha=0.7,
            label="empirical rate",
        )
        ax.set_xlabel("Predicted P(relation present)")
        ax.set_ylabel("CIViC inclusion rate")
        ax.set_title(f"Reliability diagram: {run_id}")
        ax.legend(fontsize=8)
        safe = run_id.replace("/", "_")
        _save(fig, f"10_reliability_{safe}.png")


def generate_all_figures(
    encoder_df: pd.DataFrame,
    per_run: pd.DataFrame,
    subset_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    plot_benchmark_kb_scatter(encoder_df)
    plot_benchmark_f1_range(encoder_df)
    plot_easy_hard_bars(subset_df)
    plot_benchmark_ece_scatter(encoder_df)
    if not scores_df.empty:
        plot_reliability_diagrams(scores_df)
