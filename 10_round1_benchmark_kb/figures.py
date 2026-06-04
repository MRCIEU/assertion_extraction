"""Four publication figures for Round 1 (300 dpi, IEU-style)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURE_DIR, MODEL_BY_ID

# Colour-blind-safe palette; single accent for reference lines only
PALETTE = {
    "encoder": "#4477AA",
    "encoder_fill": "#4477AA",
    "accent": "#EE6677",
    "grid": "#CCCCCC",
    "text": "#222222",
    "domain": "#66CCEE",
    "general": "#4477AA",
}

DPI = 300


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.edgecolor": PALETTE["text"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure1_benchmark_kb_faceted(encoder_df: pd.DataFrame) -> None:
    """Figure 1: benchmark F1 vs KB MRR, faceted by pair type, seed-based error bars."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    specs = [
        ("gene-drug", "kb_mrr_gene_drug_mean", "kb_mrr_gene_drug_ci_lo", "kb_mrr_gene_drug_ci_hi"),
        (
            "gene-disease",
            "kb_mrr_gene_disease_mean",
            "kb_mrr_gene_disease_ci_lo",
            "kb_mrr_gene_disease_ci_hi",
        ),
    ]
    for ax, (pt, ymean, ylo, yhi) in zip(axes, specs):
        if ymean not in encoder_df.columns:
            continue
        x = encoder_df["benchmark_f1_mean"]
        y = encoder_df[ymean]
        xerr = np.array(
            [
                x - encoder_df.get("benchmark_f1_ci_lo", x),
                encoder_df.get("benchmark_f1_ci_hi", x) - x,
            ]
        )
        yerr = np.array(
            [
                y - encoder_df.get(ylo, y),
                encoder_df.get(yhi, y) - y,
            ]
        )
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color=PALETTE["encoder"],
            ecolor=PALETTE["encoder"],
            elinewidth=1.2,
            capsize=3,
            markersize=7,
            alpha=0.9,
            zorder=3,
        )
        for _, row in encoder_df.iterrows():
            ax.annotate(
                row["short_name"].replace("-base", ""),
                (row["benchmark_f1_mean"], row[ymean]),
                fontsize=8,
                ha="left",
                va="bottom",
                xytext=(4, 4),
                textcoords="offset points",
                color=PALETTE["text"],
            )
        ax.set_xlabel("BioRED test presence F1 (self-measured)")
        ax.set_ylabel(f"KB mean reciprocal rank ({pt})")
        ax.set_title(pt)
    fig.suptitle(
        "Benchmark score versus knowledge-base ranking by entity-pair type\n"
        "(encoder means with seed-based uncertainty)",
        y=1.05,
        fontsize=13,
    )
    fig.subplots_adjust(wspace=0.32, top=0.82)
    _save(fig, "fig1_benchmark_kb_scatter.png")


def figure2_benchmark_f1_range(encoder_df: pd.DataFrame, deberta_old_mean: float | None = None) -> None:
    """Figure 2: ordered benchmark F1 strip for nine encoders."""
    _apply_style()
    order = encoder_df.sort_values("benchmark_f1_mean")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y_pos = np.arange(len(order))
    xerr = np.array(
        [
            order["benchmark_f1_mean"] - order.get("benchmark_f1_ci_lo", order["benchmark_f1_mean"]),
            order.get("benchmark_f1_ci_hi", order["benchmark_f1_mean"]) - order["benchmark_f1_mean"],
        ]
    )
    colors = [
        PALETTE["domain"] if "Bio" in n or "Sci" in n or "PubMed" in n else PALETTE["general"]
        for n in order["short_name"]
    ]
    ax.barh(
        y_pos,
        order["benchmark_f1_mean"],
        xerr=xerr,
        color=colors,
        edgecolor=PALETTE["text"],
        linewidth=0.6,
        height=0.65,
        capsize=3,
        alpha=0.85,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order["short_name"])
    ax.set_xlabel("BioRED test presence F1 (encoder mean, seed uncertainty)")
    ax.set_title("Benchmark quality gradient across nine encoders")
    spread = order["benchmark_f1_mean"].max() - order["benchmark_f1_mean"].min()
    ax.text(
        0.98,
        0.04,
        f"Spread = {spread:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
    )
    if deberta_old_mean is not None:
        deb_row = order[order["model_id"] == "deberta_base"]
        if not deb_row.empty:
            ax.axvline(
                deberta_old_mean,
                color=PALETTE["accent"],
                linestyle="--",
                linewidth=1.5,
                label=f"DeBERTa mean if collapsed seeds included ({deberta_old_mean:.3f})",
            )
            ax.legend(loc="lower right", frameon=True)
    _save(fig, "fig2_benchmark_f1_range.png")


def figure3_easy_hard_prerequisite(easy_hard_df: pd.DataFrame) -> None:
    """Figure 3: model MRR vs distance ranker on easy and hard subsets."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)
    panels = [
        ("easy_co_sentence", "Co-sentence pairs (easy)"),
        ("hard_cross_sentence", "Cross-sentence pairs (hard)"),
    ]
    for ax, (subset_key, title) in zip(axes, panels):
        sub = easy_hard_df[easy_hard_df["subset"] == subset_key]
        if sub.empty:
            continue
        dr = sub[sub["model_id"] == "distance_ranker"]["mrr"].iloc[0]
        enc = sub[sub["model_id"] != "distance_ranker"].groupby("model_id")["mrr"].mean().reset_index()
        enc["short_name"] = enc["model_id"].map(
            lambda m: MODEL_BY_ID[m].short_name if m in MODEL_BY_ID else m
        )
        enc = enc.sort_values("mrr")
        x = np.arange(len(enc))
        ax.bar(
            x,
            enc["mrr"],
            color=PALETTE["encoder"],
            edgecolor=PALETTE["text"],
            linewidth=0.5,
            width=0.72,
            label="Trained models (seed-averaged)",
        )
        ax.axhline(
            dr,
            color=PALETTE["accent"],
            linestyle="--",
            linewidth=2,
            label=f"Distance ranker ({dr:.3f})",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [s.replace("-base", "") for s in enc["short_name"]],
            rotation=45,
            ha="right",
        )
        ax.set_ylabel("Mean reciprocal rank")
        ax.set_title(title)
        ax.legend(loc="upper left", frameon=True)
    fig.suptitle(
        "Ranking validity: trained models compared with the proximity-only ranker",
        y=1.02,
        fontsize=13,
    )
    fig.subplots_adjust(wspace=0.28, bottom=0.22, top=0.88)
    _save(fig, "fig3_easy_hard_prerequisite.png")


def figure4_calibration_benchmark_ece(encoder_df: pd.DataFrame) -> None:
    """Figure 4: benchmark F1 vs ECE (calibration axis, opposite to ranking)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    x = encoder_df["benchmark_f1_mean"]
    y = encoder_df["ece_mean"]
    xerr = np.array(
        [
            x - encoder_df.get("benchmark_f1_ci_lo", x),
            encoder_df.get("benchmark_f1_ci_hi", x) - x,
        ]
    )
    yerr = np.array(
        [
            y - encoder_df.get("ece_ci_lo", y),
            encoder_df.get("ece_ci_hi", y) - y,
        ]
    )
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        color=PALETTE["encoder"],
        ecolor=PALETTE["encoder"],
        capsize=3,
        markersize=7,
    )
    for _, row in encoder_df.iterrows():
        ax.annotate(
            row["short_name"].replace("-base", ""),
            (row["benchmark_f1_mean"], row["ece_mean"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("BioRED test presence F1 (self-measured)")
    ax.set_ylabel("Expected calibration error vs CIViC curation inclusion")
    ax.set_title("Calibration tracks benchmark score (lower ECE is better)")
    ax.text(
        0.03,
        0.97,
        "Ground truth is curation inclusion, not objective biomedical truth.",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color=PALETTE["text"],
    )
    _save(fig, "fig4_calibration_benchmark_ece.png")


def generate_publication_figures(
    encoder_df: pd.DataFrame,
    easy_hard_df: pd.DataFrame,
    deberta_all_seeds_mean: float | None = None,
) -> None:
    """Emit exactly four PNG figures (replaces prior figure set)."""
    figure1_benchmark_kb_faceted(encoder_df)
    figure2_benchmark_f1_range(encoder_df, deberta_old_mean=deberta_all_seeds_mean)
    figure3_easy_hard_prerequisite(easy_hard_df)
    figure4_calibration_benchmark_ece(encoder_df)
