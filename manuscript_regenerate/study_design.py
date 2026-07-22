"""Study-design schematic for Section 1 / paper front matter."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from shared.plot_style import COLORS, apply_style, save_figure

from .paths import OUTPUT_ROOT


def generate_study_design() -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, edge):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            linewidth=1.2,
            edgecolor=edge,
            facecolor="white",
        )
        ax.add_patch(patch)
        return patch

    def arrow(x0, y0, x1, y1):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color=COLORS["neutral"],
            )
        )

    # Stage 1
    box(0.35, 2.55, 2.55, 1.85, COLORS["benchmark"])
    ax.text(
        1.625,
        3.48,
        "Stage 1 · Fine-tuning",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=COLORS["neutral"],
    )
    ax.text(
        1.625,
        3.05,
        "9 encoders (4 general, 5 biomedical)\n"
        "BioRED + DrugProt training\n"
        "10 epochs × 8 seeds → 72 runs\n"
        "checkpoint saved every epoch",
        ha="center",
        va="center",
        fontsize=8.2,
        color=COLORS["neutral"],
        linespacing=1.25,
    )

    # Stage 2 outer
    box(3.45, 1.55, 3.1, 2.85, COLORS["kb"])
    ax.text(
        5.0,
        4.05,
        "Stage 2 · Dual-axis evaluation",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=COLORS["neutral"],
    )

    # Stage 2 inner: benchmark axis
    box(3.65, 2.55, 1.25, 1.15, COLORS["benchmark"])
    ax.text(
        4.275,
        3.12,
        "In-distribution\nbenchmark axis",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=COLORS["benchmark"],
    )
    ax.text(
        4.275,
        2.78,
        "Relation-presence F1\nAUPRC on held-out test",
        ha="center",
        va="center",
        fontsize=7.5,
        color=COLORS["neutral"],
        linespacing=1.2,
    )

    # Stage 2 inner: CIViC axis (evaluation only)
    box(5.1, 2.55, 1.25, 1.15, COLORS["gene_disease"])
    ax.text(
        5.725,
        3.12,
        "Out-of-distribution\nCIViC axis",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=COLORS["gene_disease"],
    )
    ax.text(
        5.725,
        2.78,
        "Ranking: MRR (primary)\nrecall@k & AUPRC checks",
        ha="center",
        va="center",
        fontsize=7.5,
        color=COLORS["neutral"],
        linespacing=1.2,
    )

    ax.text(
        5.0,
        1.95,
        "CIViC used for evaluation only — never in training",
        ha="center",
        va="center",
        fontsize=8,
        color=COLORS["neutral"],
        style="italic",
    )

    # Stage 3
    box(7.0, 2.55, 2.55, 1.85, COLORS["gene_drug"])
    ax.text(
        8.275,
        3.48,
        "Stage 3 · Three analyses",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=COLORS["neutral"],
    )
    ax.text(
        8.275,
        3.0,
        "(1) Do the two axes agree across models?\n"
        "(2) Does training drive them apart?\n"
        "(3) What do models miss?",
        ha="center",
        va="center",
        fontsize=8.2,
        color=COLORS["neutral"],
        linespacing=1.25,
    )

    arrow(2.95, 3.45, 3.42, 3.45)
    arrow(6.58, 3.45, 6.97, 3.45)

    ax.text(
        5.0,
        0.55,
        "Frozen PubTator–CIViC candidate pool links training encoders to both evaluation axes",
        ha="center",
        va="center",
        fontsize=8.5,
        color=COLORS["neutral"],
    )

    out_dir = OUTPUT_ROOT / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return save_figure(fig, out_dir / "study_design.png")
