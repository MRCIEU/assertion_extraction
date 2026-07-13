"""Study-design schematic for Section 1 / paper front matter."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from shared.plot_style import COLORS, apply_style, save_figure

from .paths import OUTPUT_ROOT


def generate_study_design() -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 2.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.4, 1.0, 2.2, 1.0, "Stage A\nData & training", COLORS["benchmark"]),
        (3.7, 1.0, 2.2, 1.0, "Stage B\nDual-axis scoring", COLORS["kb"]),
        (7.0, 1.0, 2.2, 1.0, "Stage C\nThree analyses", COLORS["gene_disease"]),
    ]
    for x, y, w, h, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.2,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color=COLORS["neutral"])

    for x0, x1 in ((2.7, 3.6), (6.0, 6.9)):
        arrow = FancyArrowPatch(
            (x0, 1.5),
            (x1, 1.5),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=COLORS["neutral"],
        )
        ax.add_patch(arrow)

    ax.text(
        5.0,
        0.25,
        "CIViC targets → frozen PubTator pool → 9×8 fine-tuning matrix → benchmark F1 + KB MRR",
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["neutral"],
    )

    out_dir = OUTPUT_ROOT / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return save_figure(fig, out_dir / "study_design.png")
