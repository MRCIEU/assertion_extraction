"""Figures for step 03."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURE_DIR, OUTPUT_DIR


def _apply_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
        }
    )


def plot_pubtator_recall_gap(summary: dict) -> None:
    """Gap composition and miss rate by phrase length (read-only diagnostic)."""
    _apply_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    labels = ["Matched", "Entity absent", "Present but\nunmatched"]
    counts = [
        summary["n_matched"],
        summary["n_miss_entity_absent"],
        summary["n_miss_present_but_unmatched"],
    ]
    colors = ["#0072B2", "#E69F00", "#009E73"]
    y_pos = np.arange(len(labels))
    axes[0].barh(y_pos, counts, color=colors, height=0.55)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=11)
    axes[0].set_xlabel("CIViC primary relations (n)", fontsize=12)
    axes[0].set_title("Gap composition", fontsize=13)
    n_total = summary["n_total"]
    for i, c in enumerate(counts):
        axes[0].text(c + 8, i, f"{c} ({c/n_total:.1%})", va="center", fontsize=10)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    cats = ["All single-word\nentities", "Any multi-word\nentity"]
    miss_rates = [
        summary["relation_miss_rate_single_word"] * 100,
        summary["relation_miss_rate_any_multiword"] * 100,
    ]
    match_rates = [100 - mr for mr in miss_rates]
    x = np.arange(len(cats))
    width = 0.55
    axes[1].bar(x, match_rates, width, label="Matched in pool", color="#0072B2")
    axes[1].bar(x, miss_rates, width, bottom=match_rates, label="No pool positive", color="#E69F00")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cats, fontsize=11)
    axes[1].set_ylabel("Fraction of relations (%)", fontsize=12)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Miss rate vs phrase length", fontsize=13)
    axes[1].legend(loc="upper right", fontsize=9, frameon=False)
    for i, mr in enumerate(miss_rates):
        if mr >= 5:
            axes[1].text(
                i, match_rates[i] + mr / 2, f"{mr:.0f}% miss",
                ha="center", va="center", fontsize=10, color="white",
            )
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout(w_pad=2.0)
    out = FIGURE_DIR / "03_candidate_pool_pubtator_recall_gap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.08)
    plt.close(fig)


def plot_coverage(type_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["gene", "drug", "disease", "variant"]
    plot_df = type_summary.set_index("entity_type").reindex(order).dropna(subset=["coverage_rate"])
    ax.bar(plot_df.index, plot_df["coverage_rate"], color="#4C72B0")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("PubTator3 coverage rate")
    ax.set_xlabel("Entity type")
    ax.set_title("Positive coverage by entity type (A)")
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.text(i, row["coverage_rate"] + 0.02, f"{row['coverage_rate']:.0%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_candidate_pool_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pool_size(pool_size_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, pair_type in zip(axes, ["gene-drug", "gene-disease"]):
        sub = pool_size_df[pool_size_df["pair_type"] == pair_type]["pool_size"]
        if sub.empty:
            ax.set_title(f"{pair_type} (no data)")
            continue
        ax.hist(sub, bins=30, color="#55A868", edgecolor="white")
        ax.axvline(sub.median(), color="#C44E52", linestyle="--", label=f"median={sub.median():.0f}")
        ax.set_xlabel("Candidates per abstract")
        ax.set_ylabel("Abstract count")
        ax.set_title(f"Pool size: {pair_type}")
        ax.legend()
    fig.suptitle("Candidate pool size distribution (B)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_candidate_pool_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_composition(composition_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot = composition_df.pivot(index="scope", columns="pair_type", values="n_candidates").fillna(0)
    pivot.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_ylabel("Candidate count")
    ax.set_title("Pool composition by pair type and scope (C)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Pair type", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_candidate_pool_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_variant_root_cause(breakdown_df: pd.DataFrame) -> None:
    """D1: variant coverage root-cause breakdown."""
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = breakdown_df["label"].tolist()
    counts = breakdown_df["n"].tolist()
    colors = {"a_no_variant_annotated": "#4C72B0", "b_format_mismatch": "#DD8452", "c_matching_bug": "#C44E52"}
    bar_colors = [colors.get(c, "#888888") for c in breakdown_df["root_cause"]]
    ax.bar(range(len(labels)), counts, color=bar_colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(["(a) no variant", "(b) format mismatch", "(c) matching bug"][: len(labels)], rotation=0)
    ax.set_ylabel("Variant positives (n)")
    ax.set_title("Variant coverage root-cause (D1)")
    for i, c in enumerate(counts):
        ax.text(i, c + 1, str(c), ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_candidate_pool_variant_root_cause.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_systematic_loss(comparison_df: pd.DataFrame, gene_comparison_df: pd.DataFrame) -> None:
    """D2: covered vs missed feature comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    # Pair-type mix
    eval_row = comparison_df[comparison_df["group"] == "evaluable"].iloc[0]
    uneval_row = comparison_df[comparison_df["group"] == "unevaluable"].iloc[0]
    x = [0, 1]
    width = 0.35
    axes[0].bar([i - width / 2 for i in x], [eval_row["pct_gene_drug"], uneval_row["pct_gene_drug"]], width, label="gene-drug", color="#4C72B0")
    axes[0].bar([i + width / 2 for i in x], [eval_row["pct_gene_disease"], uneval_row["pct_gene_disease"]], width, label="gene-disease", color="#55A868")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Evaluable", "Unevaluable"])
    axes[0].set_ylabel("Fraction of primary positives")
    axes[0].set_title("Pair-type mix")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1)

    # Gene head length + symbol rate
    if not gene_comparison_df.empty:
        groups = gene_comparison_df["group"].tolist()
        axes[1].bar(
            [0, 1],
            gene_comparison_df["mean_entity_length"].tolist(),
            color=["#4C72B0", "#DD8452"],
        )
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(["Evaluable\n(gene head)", "Missed\n(gene head)"])
        axes[1].set_ylabel("Mean entity string length")
        axes[1].set_title("Gene name length")

        axes[2].bar(
            [0, 1],
            [v * 100 for v in gene_comparison_df["pct_symbol"].tolist()],
            color=["#4C72B0", "#DD8452"],
        )
        axes[2].set_xticks([0, 1])
        axes[2].set_xticklabels(["Evaluable\n(gene head)", "Missed\n(gene head)"])
        axes[2].set_ylabel("Symbol heuristic (%)")
        axes[2].set_title("Gene symbol vs full name")

    fig.suptitle("Systematic-loss check: covered vs missed primary positives (D2)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_candidate_pool_systematic_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_tables(
    type_summary: pd.DataFrame,
    pool_size_df: pd.DataFrame,
    composition_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> None:
    type_summary.to_csv(OUTPUT_DIR / "03_candidate_pool_coverage_by_entity_type.csv", index=False)
    pool_size_df.to_csv(OUTPUT_DIR / "03_candidate_pool_size_by_abstract.csv", index=False)
    composition_df.to_csv(OUTPUT_DIR / "03_candidate_pool_composition.csv", index=False)
    coverage_df.to_csv(OUTPUT_DIR / "03_candidate_pool_positive_coverage.csv", index=False)

    pool_stats = (
        pool_size_df.groupby("pair_type")["pool_size"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    pool_stats.to_csv(OUTPUT_DIR / "03_candidate_pool_size_distribution.csv", index=False)

    pos_frac = (
        pool_size_df.groupby("pair_type")
        .agg(
            n_abstracts=("pmid", "count"),
            mean_pool_size=("pool_size", "mean"),
            median_pool_size=("pool_size", "median"),
            mean_positive_fraction=("positive_fraction", "mean"),
            median_positive_fraction=("positive_fraction", "median"),
        )
        .reset_index()
    )
    pos_frac.to_csv(OUTPUT_DIR / "03_candidate_pool_ranking_room.csv", index=False)
