"""Four publication figures for Round 1 (300 dpi, IEU-style)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .config import FIGURE_DIR, MODEL_BY_ID

# Colour-blind-safe: neutral points, single accent for DeBERTa / reference lines
PALETTE = {
    "neutral": "#4477AA",
    "neutral_light": "#88AADD",
    "accent": "#CC6677",
    "grid": "#DDDDDD",
    "text": "#222222",
    "leader": "#888888",
}

DPI = 300

# Manual label offsets (points) to avoid clusters; keyed by model_id
_SCATTER_OFFSETS: dict[str, tuple[float, float]] = {
    "pubmedbert_base": (10, 12),
    "biomedbert_base": (-52, 10),
    "biolinkbert_base": (10, -14),
    "biobert_base": (-38, -12),
    "scibert_base": (-42, 8),
    "roberta_base": (12, -10),
    "bert_base": (-28, 14),
    "distilbert_base": (14, 6),
    "deberta_base": (-10, 16),
}

_ECE_OFFSETS: dict[str, tuple[float, float]] = {
    "pubmedbert_base": (10, 8),
    "biomedbert_base": (-58, 6),
    "biolinkbert_base": (12, -12),
    "biobert_base": (-36, 10),
    "scibert_base": (-40, -8),
    "roberta_base": (10, -12),
    "bert_base": (-30, -10),
    "distilbert_base": (12, 4),
    "deberta_base": (-48, -6),
}


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
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)


def _short_label(name: str) -> str:
    return name.replace("-base", "").replace("BERT", "BERT")


def _annotate_points(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    model_ids: list[str],
    offsets: dict[str, tuple[float, float]],
    *,
    fontsize: int = 9,
    use_leader: bool = True,
) -> None:
    for x, y, mid in zip(xs, ys, model_ids):
        dx, dy = offsets.get(mid, (8, 8))
        kw: dict = {
            "fontsize": fontsize,
            "color": PALETTE["text"],
            "xytext": (dx, dy),
            "textcoords": "offset points",
            "ha": "left" if dx >= 0 else "right",
            "va": "bottom" if dy >= 0 else "top",
        }
        if use_leader and (abs(dx) > 14 or abs(dy) > 14):
            kw["arrowprops"] = dict(
                arrowstyle="-",
                color=PALETTE["leader"],
                lw=0.7,
                shrinkA=2,
                shrinkB=2,
            )
        ax.annotate(_short_label(MODEL_BY_ID[mid].short_name if mid in MODEL_BY_ID else mid), (x, y), **kw)


def _encoder_point_colors(model_ids: list[str]) -> list[str]:
    return [PALETTE["accent"] if m == "deberta_base" else PALETTE["neutral"] for m in model_ids]


def figure1_benchmark_kb_faceted(encoder_df: pd.DataFrame) -> None:
    """Figure 1: benchmark F1 vs KB MRR, faceted, identical x-axis."""
    _apply_style()
    xlo = float(encoder_df["benchmark_f1_mean"].min()) - 0.02
    xhi = float(encoder_df["benchmark_f1_mean"].max()) + 0.02

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharex=True)
    specs = [
        ("Gene-drug", "kb_mrr_gene_drug_mean", "kb_mrr_gene_drug_ci_lo", "kb_mrr_gene_drug_ci_hi"),
        (
            "Gene-disease",
            "kb_mrr_gene_disease_mean",
            "kb_mrr_gene_disease_ci_lo",
            "kb_mrr_gene_disease_ci_hi",
        ),
    ]
    for ax, (title, ymean, ylo, yhi) in zip(axes, specs):
        mids = encoder_df["model_id"].tolist()
        x = encoder_df["benchmark_f1_mean"].astype(float).values
        y = encoder_df[ymean].astype(float).values
        xerr = np.array(
            [
                x - encoder_df.get("benchmark_f1_ci_lo", x).astype(float).values,
                encoder_df.get("benchmark_f1_ci_hi", x).astype(float).values - x,
            ]
        )
        yerr = np.array(
            [
                y - encoder_df.get(ylo, y).astype(float).values,
                encoder_df.get(yhi, y).astype(float).values - y,
            ]
        )
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color=PALETTE["neutral"],
            ecolor=PALETTE["neutral_light"],
            elinewidth=0.7,
            capsize=2,
            markersize=8,
            alpha=0.95,
            zorder=3,
        )
        for i, mid in enumerate(mids):
            c = PALETTE["accent"] if mid == "deberta_base" else PALETTE["neutral"]
            ax.plot(x[i], y[i], "o", color=c, markersize=8, zorder=4)
        _annotate_points(ax, x, y, mids, _SCATTER_OFFSETS, fontsize=9)
        ax.set_xlim(xlo, xhi)
        ax.set_xlabel("BioRED test presence F1 (self-measured)")
        ax.set_ylabel("KB mean reciprocal rank")
        ax.set_title(title)

    fig.suptitle(
        "Benchmark score versus knowledge-base ranking by entity-pair type\n"
        "(encoder means; seed uncertainty as light bars)",
        y=1.02,
        fontsize=13,
    )
    fig.subplots_adjust(wspace=0.28, top=0.86)
    _save(fig, "fig1_benchmark_kb_scatter.png")


def figure2_benchmark_f1_range(
    encoder_df: pd.DataFrame,
    deberta_all_seeds_mean: float | None = None,
) -> None:
    """Figure 2: horizontal dot plot with zoomed x-axis; DeBERTa accent + failed-seed marker."""
    _apply_style()
    order = encoder_df.sort_values("benchmark_f1_mean", ascending=True)
    y_pos = np.arange(len(order))
    x = order["benchmark_f1_mean"].astype(float).values
    xerr = np.array(
        [
            x - order.get("benchmark_f1_ci_lo", x).astype(float).values,
            order.get("benchmark_f1_ci_hi", x).astype(float).values - x,
        ]
    )
    colors = _encoder_point_colors(order["model_id"].tolist())

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.errorbar(
        x,
        y_pos,
        xerr=xerr,
        fmt="o",
        color=PALETTE["neutral"],
        ecolor=PALETTE["neutral_light"],
        elinewidth=0.7,
        capsize=2,
        markersize=9,
        zorder=3,
    )
    for xi, yi, c in zip(x, y_pos, colors):
        ax.plot(xi, yi, "o", color=c, markersize=9, zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([_short_label(s) for s in order["short_name"]])
    xmin = max(0.70, float(x.min()) - 0.015)
    xmax = min(0.80, float(x.max()) + 0.02)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("BioRED test presence F1 (encoder mean; seed uncertainty)")
    ax.set_title("Benchmark quality gradient across nine encoders")
    spread = float(x.max() - x.min())
    ax.text(0.98, 0.04, f"Spread = {spread:.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=10)

    deb_idx = order.index[order["model_id"] == "deberta_base"].tolist()
    if deberta_all_seeds_mean is not None and deb_idx:
        yi = deb_idx[0]
        ax.scatter(
            [deberta_all_seeds_mean],
            [yi],
            marker="D",
            s=55,
            facecolors="none",
            edgecolors=PALETTE["accent"],
            linewidths=1.4,
            zorder=5,
            label=f"DeBERTa with failed seeds included ({deberta_all_seeds_mean:.3f})",
        )
        ax.legend(loc="lower right", frameon=True, borderpad=0.6)

    _save(fig, "fig2_benchmark_f1_range.png")


def figure3_easy_hard_prerequisite(
    easy_hard_df: pd.DataFrame,
    encoder_order: list[str],
) -> None:
    """Figure 3: dot plots with shared encoder order and zoomed y-axis."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    panels = [
        ("easy_co_sentence", "Co-sentence pairs (easy)"),
        ("hard_cross_sentence", "Cross-sentence pairs (hard)"),
    ]
    for ax, (subset_key, title) in zip(axes, panels):
        sub = easy_hard_df[easy_hard_df["subset"] == subset_key]
        dr = float(sub[sub["model_id"] == "distance_ranker"]["mrr"].iloc[0])
        rows = []
        for mid in encoder_order:
            runs = sub[(sub["model_id"] == mid) & (sub["model_id"] != "distance_ranker")]
            if runs.empty:
                continue
            rows.append(
                {
                    "model_id": mid,
                    "mrr": float(runs["mrr"].mean()),
                    "mrr_sd": float(runs["mrr"].std(ddof=1)) if len(runs) > 1 else 0.0,
                }
            )
        enc = pd.DataFrame(rows)
        y = np.arange(len(enc))
        colors = _encoder_point_colors(enc["model_id"].tolist())
        xerr = enc["mrr_sd"].values if len(enc) else None
        ax.errorbar(
            enc["mrr"],
            y,
            xerr=xerr,
            fmt="o",
            color=PALETTE["neutral"],
            ecolor=PALETTE["neutral_light"],
            elinewidth=0.7,
            capsize=2,
            markersize=8,
            zorder=3,
        )
        for xi, yi, c in zip(enc["mrr"], y, colors):
            ax.plot(xi, yi, "o", color=c, markersize=8, zorder=4)
        ax.axhline(
            dr,
            color=PALETTE["accent"],
            linestyle="--",
            linewidth=1.8,
            label=f"Distance ranker ({dr:.3f})",
            zorder=2,
        )
        labels = [
            _short_label(MODEL_BY_ID[m].short_name if m in MODEL_BY_ID else m) for m in enc["model_id"]
        ]
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        mrr_vals = enc["mrr"].tolist() + [dr]
        ax.set_xlim(max(0.0, min(mrr_vals) - 0.04), min(1.0, max(mrr_vals) + 0.06))
        ax.set_xlabel("Mean reciprocal rank")
        ax.set_title(title)
        ax.legend(loc="lower right", frameon=True, fontsize=8)

    order_names = [
        _short_label(MODEL_BY_ID[m].short_name) for m in encoder_order if m in MODEL_BY_ID
    ]
    fig.suptitle(
        "Ranking validity: trained models versus the proximity-only ranker\n"
        f"(encoder order: high to low benchmark F1: {', '.join(order_names)})",
        y=1.03,
        fontsize=12,
    )
    fig.subplots_adjust(wspace=0.35, top=0.82, left=0.22)
    _save(fig, "fig3_easy_hard_prerequisite.png")


def figure4_calibration_benchmark_ece(encoder_df: pd.DataFrame) -> None:
    """Figure 4: benchmark F1 vs ECE; caveat below plot area."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    mids = encoder_df["model_id"].tolist()
    x = encoder_df["benchmark_f1_mean"].astype(float).values
    y = encoder_df["ece_mean"].astype(float).values
    xerr = np.array(
        [
            x - encoder_df.get("benchmark_f1_ci_lo", x).astype(float).values,
            encoder_df.get("benchmark_f1_ci_hi", x).astype(float).values - x,
        ]
    )
    yerr = np.array(
        [
            y - encoder_df.get("ece_ci_lo", y).astype(float).values,
            encoder_df.get("ece_ci_hi", y).astype(float).values - y,
        ]
    )
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        color=PALETTE["neutral"],
        ecolor=PALETTE["neutral_light"],
        elinewidth=0.7,
        capsize=2,
        markersize=8,
    )
    for i, mid in enumerate(mids):
        c = PALETTE["accent"] if mid == "deberta_base" else PALETTE["neutral"]
        ax.plot(x[i], y[i], "o", color=c, markersize=8, zorder=4)
    _annotate_points(ax, x, y, mids, _ECE_OFFSETS, fontsize=9)
    ax.set_xlabel("BioRED test presence F1 (self-measured)")
    ax.set_ylabel("Expected calibration error vs CIViC curation inclusion")
    ax.set_title("Calibration tracks benchmark score (lower ECE is better)")
    fig.text(
        0.5,
        0.01,
        "Ground truth is curation inclusion, not objective biomedical truth.",
        ha="center",
        fontsize=9,
        color=PALETTE["text"],
    )
    fig.subplots_adjust(bottom=0.14)
    _save(fig, "fig4_calibration_benchmark_ece.png")


def generate_publication_figures(
    encoder_df: pd.DataFrame,
    easy_hard_df: pd.DataFrame,
    deberta_all_seeds_mean: float | None = None,
) -> None:
    """Emit exactly four PNG figures."""
    encoder_order = (
        encoder_df.sort_values("benchmark_f1_mean", ascending=False)["model_id"].tolist()
    )
    figure1_benchmark_kb_faceted(encoder_df)
    figure2_benchmark_f1_range(encoder_df, deberta_all_seeds_mean=deberta_all_seeds_mean)
    figure3_easy_hard_prerequisite(easy_hard_df, encoder_order)
    figure4_calibration_benchmark_ece(encoder_df)
