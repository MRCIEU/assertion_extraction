"""Shared publication plotting style for all preparation and analysis steps."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito colour-blind-safe palette.
OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}

# Fixed semantic role mapping — identical in every figure across steps 00–20.
COLORS = {
    "benchmark": OKABE_ITO["blue"],
    "kb": OKABE_ITO["vermillion"],
    "gene_drug": OKABE_ITO["green"],
    "gene_disease": OKABE_ITO["purple"],
    "baseline": "#999999",
    "neutral": OKABE_ITO["black"],
    "neutral_light": "#BBBBBB",
    "deberta": OKABE_ITO["orange"],
    "positive": OKABE_ITO["green"],
    "negative": OKABE_ITO["orange"],
    "secondary": OKABE_ITO["sky"],
    "highlight": OKABE_ITO["orange"],
    "grid": "#E6E6E6",
}

# Fixed per-encoder colours (Okabe-Ito / Tol bright extension) — same mapping in every multi-encoder figure.
ENCODER_COLORS: dict[str, str] = {
    "pubmedbert_base": OKABE_ITO["blue"],
    "biomedbert_base": OKABE_ITO["vermillion"],
    "biolinkbert_base": OKABE_ITO["green"],
    "biobert_base": OKABE_ITO["purple"],
    "scibert_base": OKABE_ITO["orange"],
    "roberta_base": OKABE_ITO["sky"],
    "bert_base": "#666666",
    "distilbert_base": "#CC6677",
    "deberta_base": "#882255",
}

DPI = 300
FIG_SINGLE = (6.5, 4.5)
FIG_WIDE = (9.0, 5.0)
FIG_TALL = (6.5, 7.0)
FIG_HEATMAP = (7.0, 4.0)
FIG_PANEL = (10.0, 4.5)

FONT = {
    "family": "sans-serif",
    "size": 10,
    "title": 11,
    "label": 10,
    "tick": 9,
    "legend": 8,
}


def apply_style() -> None:
    """Apply shared matplotlib rcParams."""
    mpl.rcParams.update(
        {
            "font.family": FONT["family"],
            "font.size": FONT["size"],
            "axes.titlesize": FONT["title"],
            "axes.labelsize": FONT["label"],
            "xtick.labelsize": FONT["tick"],
            "ytick.labelsize": FONT["tick"],
            "legend.fontsize": FONT["legend"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "savefig.dpi": DPI,
            "figure.dpi": DPI,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Save figure at 300 dpi with generous margins."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.18,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    return path


def add_light_grid(ax: plt.Axes, axis: str = "y") -> None:
    """Add thin unobtrusive gridlines."""
    ax.grid(True, axis=axis, color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def encoder_point_color(model_id: str) -> str:
    """Per-encoder colour from the fixed study palette."""
    return ENCODER_COLORS.get(model_id, COLORS["neutral"])


def encoder_legend_handles(model_ids: list[str]):
    """Coloured marker handles for a compact encoder legend."""
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=encoder_point_color(mid),
            markeredgecolor=COLORS["neutral"], markeredgewidth=0.4, markersize=6,
            label=mid,
        )
        for mid in model_ids
    ]
