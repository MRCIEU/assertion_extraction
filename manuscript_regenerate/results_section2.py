"""Paper Section 2 main-text figures: training trajectory and encoder heterogeneity."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from shared.models import MODELS
from shared.plot_style import (
    COLORS,
    add_light_grid,
    apply_style,
    encoder_point_color,
    save_figure_dual,
)

from .paths import STEPS, step_paths

_RS2_FONT = {"size": 11, "label": 11, "tick": 10, "legend": 9, "annot": 9}

_ENCODER_DISPLAY = {
    "bert_base": "BERT",
    "distilbert_base": "DistilBERT",
    "deberta_base": "DeBERTa",
    "roberta_base": "RoBERTa",
    "biobert_base": "BioBERT",
    "scibert_base": "SciBERT",
    "pubmedbert_base": "PubMedBERT",
    "bluebert_base": "BlueBERT",
    "biolinkbert_base": "BioLinkBERT",
}

_ENCODER_ORDER = [m.model_id for m in MODELS]

TRAJ_CAPTION_OVERALL = (
    "Solid lines: encoder-mean overall KB MRR across nine encoders. "
    "Shaded bands: ±1 SD across seeds. Dashed horizontals: untrained encoder means (step 11). "
    "Vertical dashed line: median validation-best epoch. "
    "Paired Δ annotated for epoch 1 → val-best (n pairable seeds). "
    "Epochs with fewer than 40 seed checkpoints omitted."
)

HET_CAPTION = (
    "Exploratory, n = 9 encoders; erosion magnitude is the negative of the mean "
    "gene-disease-hard paired change (epoch 1 to validation-best)."
)


def _apply_rs2_style() -> None:
    apply_style()
    plt.rcParams.update(
        {
            "font.size": _RS2_FONT["size"],
            "axes.labelsize": _RS2_FONT["label"],
            "xtick.labelsize": _RS2_FONT["tick"],
            "ytick.labelsize": _RS2_FONT["tick"],
            "legend.fontsize": _RS2_FONT["legend"],
        }
    )


def _fig_dir(step: str = "20") -> Path:
    return step_paths(STEPS[step])["figures"]


def _out_dir(step: str = "20") -> Path:
    return step_paths(STEPS[step])["outputs"]


def _write_sidecar(path: Path, text: str) -> Path:
    note = path.with_name(path.stem + "_caption_note.txt")
    note.write_text(text + "\n", encoding="utf-8")
    return note


def _aggregate_trajectory(traj: pd.DataFrame, metric_col: str, min_seeds: int = 40) -> pd.DataFrame:
    rows = []
    counts: dict[int, int] = {}
    for epoch in sorted(traj["epoch"].unique()):
        sub = traj[traj["epoch"] == epoch]
        counts[int(epoch)] = len(sub)
        if len(sub) < min_seeds:
            continue
        seed_vals = sub[metric_col].astype(float).values
        enc_means = [
            float(sub.loc[sub["model_id"] == mid, metric_col].mean())
            for mid in _ENCODER_ORDER
            if mid in sub["model_id"].values
        ]
        line = float(np.mean(enc_means))
        sd = float(np.std(seed_vals, ddof=1)) if len(seed_vals) > 1 else 0.0
        rows.append({"epoch": int(epoch), "mean": line, "sd": sd, "n_seeds": len(sub)})
    out = pd.DataFrame(rows)
    out["lo"] = out["mean"] - out["sd"]
    out["hi"] = out["mean"] + out["sd"]
    return out, counts


def _load_untrained_floors() -> dict[str, float]:
    lift = pd.read_csv(_out_dir("11") / "11_untrained_floor_lift.csv")
    return {
        "gene_drug": float(lift["untrained_kb_mrr_gene_drug"].mean()),
        "gene_disease": float(lift["untrained_kb_mrr_gene_disease"].mean()),
    }


def _load_paired_stats() -> tuple[dict[str, dict], float]:
    paired = pd.read_csv(_out_dir("20") / "20_within_seed_paired_changes.csv")
    p = paired[paired["pairable_val_f1_best"]].copy()
    gd = p["delta_kb_mrr_gene_drug_val_f1_best"].astype(float)
    gdis = p["delta_kb_mrr_gene_disease_val_f1_best"].astype(float)
    stats_out = {
        "gene_drug": {
            "delta": float(gd.mean()),
            "falls": int((gd < 0).sum()),
            "n": len(p),
        },
        "gene_disease": {
            "delta": float(gdis.mean()),
            "falls": int((gdis < 0).sum()),
            "n": len(p),
        },
    }
    median_epoch = float(p["epoch_well_val_f1_best"].median())
    return stats_out, median_epoch


def _encoder_legend(fig, model_ids: list[str], y_anchor: float = 0.02) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=encoder_point_color(mid),
            markeredgecolor=COLORS["neutral"],
            markeredgewidth=0.4,
            markersize=6,
            label=_ENCODER_DISPLAY.get(mid, mid),
        )
        for mid in model_ids
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y_anchor),
        ncol=5,
        frameon=False,
        fontsize=7,
        columnspacing=0.8,
        handletextpad=0.3,
    )


def _strip_x_positions(sub: pd.DataFrame, center: float, width: float = 0.24) -> np.ndarray:
    n = len(sub)
    if n <= 1:
        return np.array([center])
    order = sub["erosion"].astype(float).argsort().values
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    offsets = np.linspace(-width / 2, width / 2, n)
    return center + offsets[ranks]


def _plot_trajectory_pair(
    gd: pd.DataFrame,
    gdis: pd.DataFrame,
    *,
    ylabel: str,
    untrained: dict[str, float] | None = None,
    paired_stats: dict[str, dict] | None = None,
    median_val_best_epoch: float | None = None,
) -> plt.Figure:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    _apply_rs2_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.85))

    series = [
        (gd, COLORS["gene_drug"], "Gene-drug"),
        (gdis, COLORS["gene_disease"], "Gene-disease"),
    ]

    if untrained:
        ax.axhline(
            untrained["gene_drug"],
            color=COLORS["gene_drug"],
            linestyle=(0, (4, 3)),
            linewidth=1.0,
            alpha=0.55,
            zorder=1,
        )
        ax.axhline(
            untrained["gene_disease"],
            color=COLORS["gene_disease"],
            linestyle=(0, (4, 3)),
            linewidth=1.0,
            alpha=0.55,
            zorder=1,
        )

    if median_val_best_epoch is not None:
        ax.axvline(
            median_val_best_epoch,
            color=COLORS["neutral"],
            linestyle=(0, (2, 2)),
            linewidth=0.9,
            alpha=0.45,
            zorder=1,
        )

    for df, color, label in series:
        ax.fill_between(df["epoch"], df["lo"], df["hi"], color=color, alpha=0.22, linewidth=0, zorder=1)
        ax.plot(
            df["epoch"],
            df["mean"],
            "o-",
            color=color,
            lw=2.0,
            ms=4.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=label,
            zorder=3,
        )

    if paired_stats:
        gdis_stat = paired_stats["gene_disease"]
        gd_stat = paired_stats["gene_drug"]
        e1_gdis = float(gdis.loc[gdis["epoch"] == 1, "mean"].iloc[0])
        e1_gd = float(gd.loc[gd["epoch"] == 1, "mean"].iloc[0])
        ax.scatter([1], [e1_gdis], s=36, facecolors="none", edgecolors=COLORS["gene_disease"], linewidths=1.2, zorder=4)
        ax.scatter([1], [e1_gd], s=36, facecolors="none", edgecolors=COLORS["gene_drug"], linewidths=1.2, zorder=4)
        summary = (
            f"Paired Δ (epoch 1→val-best, n={gdis_stat['n']})\n"
            f"Gene-disease: {gdis_stat['delta']:+.3f} ({gdis_stat['falls']}/{gdis_stat['n']} seeds fall)\n"
            f"Gene-drug: {gd_stat['delta']:+.3f} ({gd_stat['falls']}/{gd_stat['n']} seeds fall)"
        )
        ax.text(
            0.03,
            0.03,
            summary,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color=COLORS["neutral"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["grid"], alpha=0.92),
            zorder=5,
        )

    ax.set_xlabel("Training epoch")
    ax.set_ylabel(ylabel)
    legend_items = [
        Line2D([0], [0], color=COLORS["gene_drug"], lw=2, marker="o", label="Gene-drug"),
        Line2D([0], [0], color=COLORS["gene_disease"], lw=2, marker="o", label="Gene-disease"),
        Patch(facecolor=COLORS["neutral_light"], edgecolor="none", alpha=0.35, label="±1 SD (seeds)"),
    ]
    if untrained:
        legend_items.append(
            Line2D([0], [0], color=COLORS["baseline"], lw=1, linestyle=(0, (4, 3)), label="Untrained encoder mean")
        )
    ax.legend(handles=legend_items, frameon=False, loc="upper right", fontsize=7, handlelength=1.4)
    add_light_grid(ax, "y")

    epochs = sorted(set(gd["epoch"]) | set(gdis["epoch"]))
    if epochs:
        ax.set_xlim(min(epochs) - 0.15, max(epochs) + 0.15)
        ax.set_xticks(epochs)

    y_vals = list(gd["lo"]) + list(gd["hi"]) + list(gdis["lo"]) + list(gdis["hi"])
    if untrained:
        y_vals.extend(untrained.values())
    ymin, ymax = min(y_vals) - 0.03, max(y_vals) + 0.05
    ax.set_ylim(ymin, ymax)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.98, bottom=0.16)
    return fig


def generate_fig5_training_trajectory() -> tuple[Path, Path, Path]:
    traj = pd.read_csv(_out_dir("20") / "20_epoch_trajectory.csv")
    gd, _ = _aggregate_trajectory(traj, "kb_mrr_gene_drug")
    gdis, _ = _aggregate_trajectory(traj, "kb_mrr_gene_disease")
    untrained = _load_untrained_floors()
    paired_stats, median_epoch = _load_paired_stats()

    fig = _plot_trajectory_pair(
        gd,
        gdis,
        ylabel="KB MRR (overall)",
        untrained=untrained,
        paired_stats=paired_stats,
        median_val_best_epoch=median_epoch,
    )
    stem = _fig_dir("20") / "fig5_training_trajectory"
    png, pdf = save_figure_dual(fig, stem)
    note = _write_sidecar(stem, TRAJ_CAPTION_OVERALL)
    return png, pdf, note


def generate_fig9_encoder_heterogeneity() -> tuple[Path, Path, Path]:
    enc = pd.read_csv(_out_dir("20") / "20_encoder_property_correlation.csv")
    enc = enc[enc["slug"] == "gene_disease_hard"].copy()
    enc["erosion"] = enc["erosion_magnitude"].astype(float)
    enc["bio"] = enc["biomedical_pretrain"].astype(int)
    rho, pval = stats.spearmanr(enc["bio"], enc["erosion"])

    _apply_rs2_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.4))

    for bio_val, center in ((0, 0.0), (1, 1.0)):
        sub = enc[enc["bio"] == bio_val].sort_values("erosion")
        xs = _strip_x_positions(sub, center)
        for (_, row), xi in zip(sub.iterrows(), xs):
            mid = row["model_id"]
            ax.scatter(
                xi,
                row["erosion"],
                s=58,
                c=encoder_point_color(mid),
                edgecolors=COLORS["neutral"],
                linewidths=0.45,
                zorder=3,
            )

    ax.axhline(0, color=COLORS["baseline"], lw=0.7, linestyle=":", zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["General-purpose", "Biomedical"])
    ax.set_xlabel("Biomedical pretraining")
    ax.set_ylabel("Erosion magnitude (−Δ MRR)")
    ax.set_xlim(-0.35, 1.35)
    add_light_grid(ax, "y")

    ax.text(
        0.97,
        0.05,
        f"Spearman ρ = {rho:.2f}\np = {pval:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=_RS2_FONT["annot"],
        color=COLORS["neutral"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.9),
        zorder=5,
    )

    fig.subplots_adjust(left=0.20, right=0.98, top=0.90, bottom=0.28)
    _encoder_legend(fig, enc["model_id"].tolist(), y_anchor=0.02)

    stem = _fig_dir("20") / "fig9_encoder_heterogeneity"
    png, pdf = save_figure_dual(fig, stem)
    note = _write_sidecar(stem, HET_CAPTION)
    return png, pdf, note
