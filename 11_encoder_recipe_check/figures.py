"""Publication figures for encoder recipe check (IEU style, matches folder 10)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURE_DIR, PRIMARY_SEED, ROUND1_RECIPE_LR

PALETTE = {
    "neutral": "#4477AA",
    "neutral_light": "#88AADD",
    "accent": "#CC6677",
    "grid": "#DDDDDD",
    "text": "#222222",
    "lr_low": "#4477AA",
    "lr_high": "#88AADD",
    "warmup_on": "#66CCEE",
    "warmup_off": "#999999",
}

DPI = 300


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "savefig.dpi": DPI,
            "figure.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_DIR / name,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)


def figure1_recipe_grid(
    grid: pd.DataFrame,
    *,
    deberta_all8_mean: float,
    deberta_clean6_mean: float,
) -> None:
    """Benchmark F1 by recipe: learning rate as main grouping, warmup secondary."""
    _apply_style()
    g = grid[grid["seed"] == PRIMARY_SEED].copy()
    g["lr_label"] = g["lr"].map({1e-5: "1e-5", 2e-5: "2e-5"})
    g["warmup_short"] = g["warmup_label"].map({"none": "no warmup", "warmup_10pct": "10% warmup"})

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x_positions: list[float] = []
    labels: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    colors: list[str] = []
    offsets = {"no warmup": -0.12, "10% warmup": 0.12}
    lr_centers = {"1e-5": 0, "2e-5": 1}

    for lr_lab in ["1e-5", "2e-5"]:
        sub = g[g["lr_label"] == lr_lab]
        for warm in ["no warmup", "10% warmup"]:
            row = sub[sub["warmup_short"] == warm]
            if row.empty:
                continue
            x = lr_centers[lr_lab] + offsets[warm]
            y = float(row["benchmark_f1"].iloc[0])
            xs.append(x)
            ys.append(y)
            colors.append(PALETTE["warmup_on"] if warm == "10% warmup" else PALETTE["warmup_off"])
            ep = int(row["best_epoch_val_f1"].iloc[0])
            ax.plot(x, y, "o", color=colors[-1], markersize=11, zorder=4)
            ax.annotate(
                f"{y:.3f}\n(ep {ep})",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                color=PALETTE["text"],
            )

    ax.axhline(deberta_all8_mean, color=PALETTE["accent"], linestyle=":", linewidth=1.2, label=f"Round-1 DeBERTa 8-seed mean ({deberta_all8_mean:.3f})")
    ax.axhline(deberta_clean6_mean, color=PALETTE["accent"], linestyle="--", linewidth=1.2, label=f"Round-1 DeBERTa 6 clean seeds ({deberta_clean6_mean:.3f})")
    r1_row = g[(g["lr"] == ROUND1_RECIPE_LR) & (g["warmup_label"] == "none")]
    if not r1_row.empty:
        ax.axhline(
            float(r1_row["benchmark_f1"].iloc[0]),
            color="#666666",
            linestyle="-.",
            linewidth=1.2,
            label=f"Round-1 recipe point ({float(r1_row['benchmark_f1'].iloc[0]):.3f})",
        )

    ymin = min(ys + [deberta_all8_mean, deberta_clean6_mean]) - 0.02
    ymax = max(ys + [deberta_clean6_mean]) + 0.04
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(max(0.68, ymin), min(0.80, ymax + 0.01))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Learning rate 1e-5", "Learning rate 2e-5"])
    ax.set_ylabel("BioRED test presence F1 (self-measured)")
    ax.set_title("DeBERTa recipe grid (seed 42): benchmark F1 by learning rate and warmup")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fig1_recipe_grid.png")


def _short_label(name: str) -> str:
    return name.replace("-base", "").replace("BioBERT", "BioBERT").strip()


def figure2_deberta_in_band(
    encoder_df: pd.DataFrame,
    grid_best_f1: float,
    *,
    deberta_all8_mean: float,
) -> None:
    """Round-1 encoder gradient strip with recovered DeBERTa and artifact markers."""
    _apply_style()
    r1 = encoder_df[encoder_df["source"] == "round1_encoder"].sort_values("benchmark_f1")
    y_pos = np.arange(len(r1))
    x = r1["benchmark_f1"].astype(float).values
    has_ci = "ci_lo" in r1.columns and r1["ci_lo"].notna().all()
    if has_ci:
        xerr = np.array(
            [
                x - r1["ci_lo"].astype(float).values,
                r1["ci_hi"].astype(float).values - x,
            ]
        )
    else:
        xerr = None

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if xerr is not None:
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
    colors = [
        PALETTE["accent"] if mid == "deberta_base" else PALETTE["neutral"]
        for mid in r1["model_id"]
    ]
    for xi, yi, c in zip(x, y_pos, colors):
        ax.plot(xi, yi, "o", color=c, markersize=9, zorder=4)

    deb_mask = r1["model_id"] == "deberta_base"
    if deb_mask.any():
        yi = int(np.flatnonzero(deb_mask.values)[0])
        ax.scatter(
            [deberta_all8_mean],
            [yi],
            marker="D",
            s=55,
            facecolors="none",
            edgecolors=PALETTE["accent"],
            linewidths=1.4,
            zorder=5,
            label=f"DeBERTa with failed seeds included ({deberta_all8_mean:.3f})",
        )
    ax.axvline(
        grid_best_f1,
        color=PALETTE["lr_low"],
        linestyle="-",
        linewidth=1.5,
        label=f"Grid best seed 42 ({grid_best_f1:.3f})",
        zorder=2,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([_short_label(s) for s in r1["short_name"]])
    xmin = max(0.52, float(x.min()) - 0.02)
    xmax = min(0.80, float(max(x.max(), grid_best_f1)) + 0.02)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("BioRED test presence F1 (encoder mean; seed uncertainty)")
    ax.set_title("Recovered DeBERTa sits inside the eight-encoder band, not at 0.554")
    ax.legend(loc="lower right", fontsize=7, frameon=True, borderpad=0.6)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fig2_deberta_in_band.png")


def figure3_val_curves(curves: pd.DataFrame) -> None:
    """Validation curves for four grid recipes (seed 42)."""
    _apply_style()
    g = curves[curves["seed"] == PRIMARY_SEED].copy()
    if g.empty:
        g = curves.copy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    order = [
        ("lr1e-05_warmup_none", "1e-5, no warmup"),
        ("lr2e-05_warmup_none", "2e-5, no warmup (Round-1 recipe)"),
        ("lr1e-05_warmup_warmup_10pct", "1e-5, 10% warmup"),
        ("lr2e-05_warmup_warmup_10pct", "2e-5, 10% warmup"),
    ]
    linestyles = {"2e-5, no warmup (Round-1 recipe)": "--"}
    for ax, col in zip(axes, ["val_f1", "val_loss"]):
        for rk, lab in order:
            sub = g[g["run_key"] == rk].sort_values("epoch")
            if sub.empty:
                continue
            ls = linestyles.get(lab, "-")
            lw = 2.0 if "Round-1" in lab else 1.5
            ax.plot(sub["epoch"], sub[col], marker="o", label=lab, linestyle=ls, linewidth=lw, alpha=0.9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation F1" if col == "val_f1" else "Validation loss")
        ax.set_title("Validation F1" if col == "val_f1" else "Validation loss")
        ax.legend(fontsize=7, loc="best")
        ax.grid(color=PALETTE["grid"], linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("DeBERTa trains later and higher under lr 1e-5 than under the Round-1 recipe", y=1.02)
    fig.tight_layout()
    _save(fig, "fig3_val_curves.png")


def generate_figures(
    grid: pd.DataFrame,
    encoder_df: pd.DataFrame,
    curves: pd.DataFrame,
    *,
    deberta_all8_mean: float,
    deberta_clean6_mean: float,
    grid_best_f1: float,
) -> None:
    figure1_recipe_grid(grid, deberta_all8_mean=deberta_all8_mean, deberta_clean6_mean=deberta_clean6_mean)
    figure2_deberta_in_band(
        encoder_df,
        grid_best_f1,
        deberta_all8_mean=deberta_all8_mean,
    )
    if not curves.empty:
        figure3_val_curves(curves)
