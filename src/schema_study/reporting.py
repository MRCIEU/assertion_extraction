"""Tables and figures for paired differences, retention, and audits.

Synthetic calls must name a destination outside the formal study result tree.
No composite score is added.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FORMAL_RESULT_PARTS = {"runs", "evaluation", "analysis", "final", "publication"}


def assert_figure_destination(path: Path, *, synthetic: bool) -> Path:
    resolved = Path(path).resolve()
    if synthetic and any(part in FORMAL_RESULT_PARTS for part in resolved.parts):
        raise ValueError(f"Synthetic figure cannot be written into a formal result directory: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def save_figure(figure, path: Path, *, synthetic: bool) -> dict:
    destination = assert_figure_destination(path, synthetic=synthetic)
    figure.savefig(destination.with_suffix(".pdf"))
    figure.savefig(destination.with_suffix(".png"), dpi=150)
    plt.close(figure)
    note = destination.with_suffix(".json")
    note.write_text(
        '{"synthetic": %s, "composite_score": false}\n' % ("true" if synthetic else "false"),
        encoding="utf-8",
    )
    return {"pdf": str(destination.with_suffix(".pdf")), "png": str(destination.with_suffix(".png")), "synthetic": synthetic}


def plot_paired_differences(labels: list[str], differences: list[float], lows: list[float], highs: list[float], path: Path, *, synthetic: bool) -> dict:
    figure, axis = plt.subplots(figsize=(7, 4))
    positions = range(len(labels))
    axis.axvline(0.0, color="0.5", linewidth=0.8)
    axis.errorbar(differences, list(positions), xerr=[
        [value - low for value, low in zip(differences, lows)],
        [high - value for value, high in zip(differences, highs)],
    ], fmt="o", color="black", capsize=3)
    axis.set_yticks(list(positions), labels)
    axis.set_xlabel("Paired difference")
    axis.set_title("Route or condition difference")
    figure.tight_layout()
    return save_figure(figure, path, synthetic=synthetic)


def plot_precision_retention(precision, retention, recall, coverage, path: Path, *, synthetic: bool) -> dict:
    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].scatter(retention, precision, color="black")
    axes[0].set_xlabel("Information retained")
    axes[0].set_ylabel("Positive micro-precision")
    axes[1].scatter(coverage, recall, color="black")
    axes[1].set_xlabel("Output coverage")
    axes[1].set_ylabel("Positive micro-recall")
    figure.tight_layout()
    return save_figure(figure, path, synthetic=synthetic)


def plot_loss_decomposition(schema_loss: float, prediction_loss: float, path: Path, *, synthetic: bool) -> dict:
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.bar(["schema loss", "prediction loss"], [schema_loss, prediction_loss], color=["0.35", "0.65"])
    axis.set_ylabel("Share of 1 - R")
    axis.set_ylim(0, 1)
    figure.tight_layout()
    return save_figure(figure, path, synthetic=synthetic)


def plot_named_bars(names: list[str], values: list[float], path: Path, ylabel: str, *, synthetic: bool) -> dict:
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(names, values, color="0.3")
    axis.set_ylabel(ylabel)
    figure.tight_layout()
    return save_figure(figure, path, synthetic=synthetic)
