"""Publication figures for training-dynamics adjudication."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared.models import MODELS

from .config import DPI, FIGURE_DIR, MODEL_BY_ID, PALETTE
from .adjudication import WELL_DEFS, WELL_DEF_LABELS, WELL_DEF_VAL_F1


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
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


def figure1_per_seed_trajectories(traj: pd.DataFrame) -> None:
    """Benchmark F1 and KB hard MRR across epochs; faint per-seed + mean."""
    _apply_style()
    encoders = [m.model_id for m in MODELS]
    ncol = 3
    nrow = int(np.ceil(len(encoders) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.0, 2.2 * nrow))
    axes = np.atleast_1d(axes).flatten()

    for ax, mid in zip(axes, encoders):
        sub = traj[traj["model_id"] == mid].sort_values(["seed", "epoch"])
        if sub.empty:
            ax.set_visible(False)
            continue
        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)

        for seed, g in sub.groupby("seed"):
            ax.plot(
                g["epoch"],
                g["benchmark_f1"],
                color=PALETTE["neutral_light"],
                alpha=0.35,
                lw=0.9,
            )
            ax2.plot(
                g["epoch"],
                g["kb_mrr_hard"],
                color=PALETTE["accent"],
                alpha=0.25,
                lw=0.9,
            )

        mean_g = sub.groupby("epoch").agg(
            benchmark_f1=("benchmark_f1", "mean"),
            kb_mrr_hard=("kb_mrr_hard", "mean"),
        )
        ax.plot(
            mean_g.index,
            mean_g["benchmark_f1"],
            "o-",
            color=PALETTE["neutral"],
            lw=2.2,
            ms=4,
            label="Benchmark F1 (mean)",
        )
        ax2.plot(
            mean_g.index,
            mean_g["kb_mrr_hard"],
            "s-",
            color=PALETTE["accent"],
            lw=2.2,
            ms=4,
            label="KB hard MRR (mean)",
        )
        name = MODEL_BY_ID[mid].short_name.replace("-base", "")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("BioRED test F1")
        ax2.set_ylabel("CIViC hard-subset MRR")

    for ax in axes[len(encoders) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Within-model trajectories: in-distribution benchmark vs out-of-distribution KB (hard)",
        y=1.01,
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "fig1_per_seed_trajectories.png")


def figure2_paired_change_distribution(paired: pd.DataFrame) -> None:
    """Within-seed delta benchmark vs delta KB hard (epoch 1 -> best val F1)."""
    _apply_style()
    well = WELL_DEF_VAL_F1
    pc = f"pairable_{well}"
    sub = paired[paired[pc]].copy()
    if sub.empty:
        print("  Skipping fig2 (no pairable seeds)")
        return

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    x = sub[f"delta_benchmark_f1_{well}"].astype(float)
    y = sub[f"delta_kb_mrr_hard_{well}"].astype(float)
    colors = [PALETTE["accent"] if b > 0 and k < 0 else PALETTE["neutral"] for b, k in zip(x, y)]
    ax.scatter(x, y, c=colors, alpha=0.75, s=42, edgecolors="white", linewidths=0.4)
    ax.axhline(0, color=PALETTE["text"], lw=0.8, alpha=0.4)
    ax.axvline(0, color=PALETTE["text"], lw=0.8, alpha=0.4)
    ax.set_xlabel("Change in benchmark F1 (epoch 1 to best validation F1)")
    ax.set_ylabel("Change in KB hard-subset MRR")
    n_erosion = int(((x > 0) & (y < 0)).sum())
    ax.set_title(
        "Within-seed paired change\n"
        f"(each point = one seed; accent = benchmark up and KB hard down, n={n_erosion})"
    )
    _save(fig, "fig2_within_seed_paired_change.png")


def figure3_hard_easy_pair_type(hard_easy: pd.DataFrame, pair_type: pd.DataFrame) -> None:
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # Panel A: hard vs easy
    ax = axes[0]
    he = hard_easy[hard_easy["well_trained_definition"] == WELL_DEF_VAL_F1]
    labels = ["Hard\n(cross-sentence)", "Easy\n(co-sentence)"]
    keys = ["hard_cross_sentence", "easy_co_sentence"]
    means, los, his, cols = [], [], [], []
    for key in keys:
        row = he[he["subset"] == key]
        if row.empty:
            continue
        r = row.iloc[0]
        means.append(float(r["mean_delta_kb_mrr"]))
        los.append(float(r["mean_delta_kb_mrr"]) - float(r["ci_lo"]))
        his.append(float(r["ci_hi"]) - float(r["mean_delta_kb_mrr"]))
        cols.append(PALETTE["accent"] if key == "hard_cross_sentence" else PALETTE["neutral"])
    x = np.arange(len(means))
    ax.bar(x, means, color=cols, yerr=[los, his], capsize=4, width=0.55)
    ax.axhline(0, color=PALETTE["text"], lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels[: len(means)])
    ax.set_ylabel("Mean KB MRR change (within seed)")
    ax.set_title("Hard vs easy subset")

    # Panel B: gene-drug vs gene-disease
    ax = axes[1]
    pt = pair_type[pair_type["well_trained_definition"] == WELL_DEF_VAL_F1]
    labels2 = ["Gene-drug", "Gene-disease"]
    means2, los2, his2 = [], [], []
    for ptl in ["gene-drug", "gene-disease"]:
        row = pt[pt["pair_type"] == ptl]
        if row.empty:
            continue
        r = row.iloc[0]
        means2.append(float(r["mean_delta_kb_mrr"]))
        los2.append(float(r["mean_delta_kb_mrr"]) - float(r["ci_lo"]))
        his2.append(float(r["ci_hi"]) - float(r["mean_delta_kb_mrr"]))
    x2 = np.arange(len(means2))
    ax.bar(x2, means2, color=[PALETTE["neutral"], PALETTE["accent"]], yerr=[los2, his2], capsize=4, width=0.55)
    ax.axhline(0, color=PALETTE["text"], lw=0.8, alpha=0.5)
    ax.set_xticks(x2)
    ax.set_xticklabels(labels2[: len(means2)])
    ax.set_ylabel("Mean KB MRR change (within seed)")
    ax.set_title("Gene-drug vs gene-disease")

    fig.suptitle("KB erosion by subset and pair type (epoch 1 to best validation F1)", y=1.02, fontsize=13)
    fig.tight_layout()
    _save(fig, "fig3_hard_easy_pair_type.png")


def figure4_robustness_well_trained(robustness: pd.DataFrame) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    enc = robustness[robustness["model_id"] != "ALL"] if "ALL" in robustness["model_id"].values else robustness
    if enc.empty:
        enc = robustness
    x = np.arange(len(enc))
    width = 0.25
    for i, well_def in enumerate(WELL_DEFS):
        col = f"frac_erosion_{well_def}"
        if col not in enc.columns:
            continue
        offset = (i - 1) * width
        ax.bar(x + offset, enc[col].astype(float), width, label=WELL_DEF_LABELS[well_def])
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(s).replace("-base", "") for s in enc["short_name"]],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("Fraction of seeds with benchmark up, KB hard down")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_title("Robustness: erosion fraction under three well-trained definitions")
    fig.tight_layout()
    _save(fig, "fig4_robustness_well_trained.png")


def figure5_gene_disease_hard_trajectories(traj: pd.DataFrame) -> None:
    """Gene-disease-hard KB MRR vs benchmark F1 per seed (analysis E)."""
    _apply_style()
    if "kb_mrr_gene_disease_hard" not in traj.columns:
        print("  Skipping fig5 (missing gene-disease-hard cross metrics)")
        return

    encoders = [m.model_id for m in MODELS]
    ncol = 3
    nrow = int(np.ceil(len(encoders) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.0, 2.2 * nrow))
    axes = np.atleast_1d(axes).flatten()

    for ax, mid in zip(axes, encoders):
        sub = traj[traj["model_id"] == mid].sort_values(["seed", "epoch"])
        if sub.empty:
            ax.set_visible(False)
            continue
        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)

        for seed, g in sub.groupby("seed"):
            ax.plot(g["epoch"], g["benchmark_f1"], color=PALETTE["neutral_light"], alpha=0.35, lw=0.9)
            ax2.plot(
                g["epoch"],
                g["kb_mrr_gene_disease_hard"],
                color=PALETTE["accent"],
                alpha=0.25,
                lw=0.9,
            )

        mean_g = sub.groupby("epoch").agg(
            benchmark_f1=("benchmark_f1", "mean"),
            kb_mrr_gene_disease_hard=("kb_mrr_gene_disease_hard", "mean"),
        )
        ax.plot(
            mean_g.index,
            mean_g["benchmark_f1"],
            "o-",
            color=PALETTE["neutral"],
            lw=2.2,
            ms=4,
        )
        ax2.plot(
            mean_g.index,
            mean_g["kb_mrr_gene_disease_hard"],
            "s-",
            color=PALETTE["accent"],
            lw=2.2,
            ms=4,
        )
        name = MODEL_BY_ID[mid].short_name.replace("-base", "")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("BioRED test F1")
        ax2.set_ylabel("Gene-disease hard MRR")

    for ax in axes[len(encoders) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Gene-disease hard subset: benchmark vs knowledge-base ranking across training",
        y=1.01,
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "fig5_gene_disease_hard_trajectories.png")


def figure6_pair_type_subset_contrast(pair_subset: pd.DataFrame) -> None:
    """Contrast gene-drug vs gene-disease paired change by hard/easy subset."""
    _apply_style()
    sub = pair_subset[pair_subset["well_trained_definition"] == WELL_DEF_VAL_F1]
    if sub.empty:
        print("  Skipping fig6 (no pair×subset contrast data)")
        return

    order = ["gene_drug_hard", "gene_drug_easy", "gene_disease_hard", "gene_disease_easy"]
    labels = [
        "Gene-drug\nhard",
        "Gene-drug\neasy",
        "Gene-disease\nhard",
        "Gene-disease\neasy",
    ]
    colors = [PALETTE["neutral"], PALETTE["neutral_light"], PALETTE["accent"], "#EEBBB8"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    means, los, his, ns = [], [], [], []
    for slug in order:
        row = sub[sub["slug"] == slug]
        if row.empty:
            means.append(0)
            los.append(0)
            his.append(0)
            ns.append(0)
            continue
        r = row.iloc[0]
        means.append(float(r["mean_delta_kb_mrr"]))
        los.append(float(r["mean_delta_kb_mrr"]) - float(r["ci_lo"]))
        his.append(float(r["ci_hi"]) - float(r["mean_delta_kb_mrr"]))
        ns.append(int(r["n_kb_falls"]))

    x = np.arange(len(order))
    ax.bar(x, means, color=colors[: len(means)], yerr=[los, his], capsize=4, width=0.58)
    ax.axhline(0, color=PALETTE["text"], lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels[: len(means)])
    ax.set_ylabel("Mean KB MRR change (within seed, epoch 1 to best val F1)")
    ax.set_title("Paired change by pair type and subset")
    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + (0.003 if m >= 0 else -0.008), f"{n} fall", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig6_pair_type_subset_contrast.png")


    _save(fig, "fig6_pair_type_subset_contrast.png")


def figure7_kb_peak_timing(timing_summary: pd.DataFrame) -> None:
    _apply_style()
    sub = timing_summary[
        (timing_summary["slug"] == "gene_disease") & (timing_summary["timing_class"] != "all")
    ]
    if sub.empty:
        print("  Skipping fig7 (no timing data)")
        return
    labels = ["Before\nbest-val", "Coincident\n(±1 epoch)", "After\nbest-val"]
    keys = ["before_best_val", "coincident_best_val", "after_best_val"]
    vals = [float(sub.loc[sub["timing_class"] == k, "frac_seeds"].iloc[0]) if k in sub["timing_class"].values else 0 for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(range(3), vals, color=[PALETTE["accent"], PALETTE["neutral"], PALETTE["neutral_light"]], width=0.55)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of seeds")
    ax.set_ylim(0, 1.05)
    ax.set_title("When gene-disease KB ranking peaks relative to validation-best epoch")
    fig.tight_layout()
    _save(fig, "fig7_kb_peak_timing.png")


def figure8_pool_stratum(stratum_summary: pd.DataFrame) -> None:
    _apply_style()
    if stratum_summary.empty:
        print("  Skipping fig8 (no pool stratum data)")
        return
    order = ["small_pool", "large_pool", "comparable_to_gene_drug"]
    labels = ["Small pool\n(≤ median)", "Large pool\n(> median)", "Comparable to\ngene-drug size"]
    fig, ax = plt.subplots(figsize=(8, 5))
    means, los, his = [], [], []
    for key in order:
        row = stratum_summary[stratum_summary["stratum"] == key]
        if row.empty:
            continue
        r = row.iloc[0]
        means.append(float(r["mean_delta_mrr"]))
        los.append(float(r["mean_delta_mrr"]) - float(r["ci_lo"]))
        his.append(float(r["ci_hi"]) - float(r["mean_delta_mrr"]))
    x = np.arange(len(means))
    ax.bar(x, means, color=PALETTE["accent"], yerr=[los, his], capsize=4, width=0.55)
    ax.axhline(0, color=PALETTE["text"], lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels[: len(means)])
    ax.set_ylabel("Mean gene-disease MRR change (epoch 1 to best val)")
    ax.set_title("Gene-disease paired change by abstract pool-size stratum")
    fig.tight_layout()
    _save(fig, "fig8_pool_stratum_gene_disease.png")


def figure9_encoder_property_scatter(enc_table: pd.DataFrame) -> None:
    _apply_style()
    if enc_table.empty:
        print("  Skipping fig9 (no encoder correlation table)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    specs = [
        ("mean_benchmark_f1", "Mean benchmark F1"),
        ("biomedical_pretrain", "Biomedical pretrain (0/1)"),
        ("params_millions", "Parameters (millions)"),
    ]
    for ax, (col, xlab) in zip(axes, specs):
        x = enc_table[col].astype(float)
        y = enc_table["erosion_magnitude"].astype(float)
        ax.scatter(x, y, c=PALETTE["accent"], s=70, edgecolors="white", linewidths=0.5)
        for _, r in enc_table.iterrows():
            ax.annotate(
                str(r.get("encoder_name", r.get("short_name", ""))).replace("-base", ""),
                (float(r[col]), float(r["erosion_magnitude"])),
                fontsize=7,
                ha="left",
                va="bottom",
            )
        ax.set_xlabel(xlab)
        ax.set_ylabel("Gene-disease-hard erosion\n(−Δ MRR, epoch 1→best val)")
    fig.suptitle("Encoder properties vs gene-disease-hard erosion (n=9, exploratory)", y=1.02)
    fig.tight_layout()
    _save(fig, "fig9_encoder_property_scatter.png")


def figure10_failure_modes(patterns: pd.DataFrame, summary: dict) -> None:
    _apply_style()
    if patterns.empty:
        print("  Skipping fig10 (no failure-mode data)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax = axes[0]
    labels = ["Abstract-\nunsupported", "Genuine\nmodel error"]
    frac_unsup = float(summary.get("frac_abstract_unsupported", 0))
    vals = [frac_unsup, 1 - frac_unsup]
    ax.bar([0, 1], vals, color=[PALETTE["neutral_light"], PALETTE["accent"]], width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share of missed positives")
    ax.set_ylim(0, 1.05)
    ax.set_title("Missed CIViC positives: abstract ceiling vs model error")

    ax = axes[1]
    psub = patterns.copy()
    ax.barh(psub["pattern"], psub["rate_in_genuine_errors"], color=PALETTE["accent"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Rate among genuine errors")
    ax.set_title("Systematic failure modes (abstract-supported only)")
    fig.tight_layout()
    _save(fig, "fig10_failure_mode_summary.png")


def generate_all_figures(
    traj: pd.DataFrame,
    paired: pd.DataFrame,
    hard_easy: pd.DataFrame,
    pair_type: pd.DataFrame,
    robustness: pd.DataFrame,
    pair_subset: pd.DataFrame | None = None,
    timing_summary: pd.DataFrame | None = None,
    stratum_summary: pd.DataFrame | None = None,
    enc_table: pd.DataFrame | None = None,
    qual_patterns: pd.DataFrame | None = None,
    qual_summary: dict | None = None,
) -> None:
    figure1_per_seed_trajectories(traj)
    figure2_paired_change_distribution(paired)
    figure3_hard_easy_pair_type(hard_easy, pair_type)
    figure4_robustness_well_trained(robustness)
    figure5_gene_disease_hard_trajectories(traj)
    if pair_subset is not None:
        figure6_pair_type_subset_contrast(pair_subset)
    if timing_summary is not None:
        figure7_kb_peak_timing(timing_summary)
    if stratum_summary is not None:
        figure8_pool_stratum(stratum_summary)
    if enc_table is not None:
        figure9_encoder_property_scatter(enc_table)
    if qual_patterns is not None and qual_summary is not None:
        figure10_failure_modes(qual_patterns, qual_summary)
