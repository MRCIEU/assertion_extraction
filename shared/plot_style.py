"""Shared publication plotting style for all preparation and analysis steps."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito colour-blind-safe palette with fixed semantic roles across the study.
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

# Semantic role mapping (constant across all figures).
COLORS = {
    "benchmark": OKABE_ITO["blue"],
    "kb": OKABE_ITO["vermillion"],
    "gene_drug": OKABE_ITO["green"],
    "gene_disease": OKABE_ITO["purple"],
    "positive": OKABE_ITO["green"],
    "negative": OKABE_ITO["orange"],
    "neutral": OKABE_ITO["black"],
    "neutral_light": "#999999",
    "accent": OKABE_ITO["vermillion"],
    "secondary": OKABE_ITO["sky"],
    "highlight": OKABE_ITO["orange"],
    "grid": "#E6E6E6",
}

DPI = 800
FIG_SINGLE = (6.5, 4.5)
FIG_WIDE = (9.0, 5.0)
FIG_TALL = (6.5, 7.0)
FIG_HEATMAP = (9.0, 5.5)

FONT = {
    "family": "sans-serif",
    "size": 10,
    "title": 11,
    "label": 10,
    "tick": 9,
    "legend": 9,
}


def apply_style() -> None:
    """Apply the shared matplotlib rcParams."""
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
    """Save figure at 800 dpi with no outer frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    return path


def add_light_grid(ax: plt.Axes, axis: str = "y") -> None:
    """Add thin unobtrusive gridlines."""
    ax.grid(True, axis=axis, color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
