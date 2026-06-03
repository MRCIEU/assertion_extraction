"""Figures for step 01."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .config import DATA_DIR, FIGURE_DIR, GRANULARITY_LEVELS


def plot_granularity_ladder(granularity_df: pd.DataFrame) -> None:
    train_df = granularity_df[granularity_df["corpus"].isin(["biored", "drugprot"])]
    level_short = {
        "1_coarse_association": "coarse",
        "2_directional_correlation": "directional",
        "3_fine_mechanism": "mechanism",
        "4_clinical_significance": "clinical",
    }
    pivot = train_df.groupby(["display_name", "granularity_level"])["count"].sum().reset_index()
    corpora = pivot["display_name"].unique()
    levels = [level_short[l] for l in GRANULARITY_LEVELS if l in level_short]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    x = range(len(levels))
    for i, corpus in enumerate(corpora):
        sub = pivot[pivot["display_name"] == corpus]
        counts = [sub[sub["granularity_level"] == l]["count"].sum() for l in GRANULARITY_LEVELS if l in level_short]
        offset = (i - 0.5) * width
        ax.bar([xi + offset for xi in x], counts, width, label=corpus)

    ax.set_xticks(list(x))
    ax.set_xticklabels(levels)
    ax.set_ylabel("Relation count")
    ax.set_title("Relation labels by granularity level")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_corpus_granularity_ladder.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_corpus_overview(inventory_long_path=None) -> None:
    path = inventory_long_path or (DATA_DIR / "corpus_inventory_long.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)
    rel = df[(df["table"] == "relation_type") & (df["split"] == "train")]
    top = rel.groupby(["display_name", "label"])["count"].sum().reset_index()
    top = top.sort_values("count", ascending=False).groupby("display_name").head(8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    for ax, name in zip(axes, ["BioRED", "DrugProt", "BC5CDR"]):
        sub = top[top["display_name"] == name]
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.barh(sub["label"], sub["count"], color="#4C72B0")
        ax.set_title(f"{name} (train)")
        ax.invert_yaxis()
    fig.suptitle("Top relation types per corpus (train split)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_corpus_relation_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pmid_overlap(overlap_row: dict) -> None:
    labels = ["BioRED only", "Both", "DrugProt only"]
    values = [
        overlap_row["biored_only"],
        overlap_row["intersection"],
        overlap_row["drugprot_only"],
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("PMID count")
    ax.set_title("Training-corpus PMID overlap (train + validation)")
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_corpus_pmid_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_conflict_summary(conflict: dict) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    agree = conflict["co_annotated_pairs"] - conflict["conflict_count"]
    ax.bar(
        ["Agree", "Conflict"],
        [agree, conflict["conflict_count"]],
        color=["#55A868", "#DD8452"],
    )
    rate = conflict["conflict_rate"]
    ax.set_ylabel("Co-annotated entity pairs")
    ax.set_title(f"Binary-presence agreement on shared PMIDs (conflict rate={rate:.1%})")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_corpus_pmid_conflicts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_leakage(leakage: dict) -> None:
    labels = ["BioRED", "DrugProt", "Combined"]
    values = [
        leakage["overlap_biored"],
        leakage["overlap_drugprot"],
        leakage["overlap_combined"],
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#C44E52" if v > 0 else "#55A868" for v in values]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("PMIDs overlapping CIViC eval set")
    ax.set_title(
        f"Train/eval PMID leakage (eval={leakage['eval_unique_pmids']} unique PMIDs)"
    )
    for i, v in enumerate(values):
        ax.text(i, v + 0.05, str(v), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_corpus_pmid_leakage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_oncology_fractions(fractions_df: pd.DataFrame) -> None:
    """Oncology fraction by criterion (primary figure uses conservative intersection separately)."""
    plot_df = fractions_df[fractions_df["fraction"].notna()].copy()
    if plot_df.empty:
        return
    crit_labels = {
        "disease_neoplasm": "Disease (NCIt neoplasm)",
        "gene_civic": "Gene (CIViC set)",
        "literature_mesh": "Literature (MeSH neoplasm)",
    }
    plot_df["criterion_label"] = plot_df["criterion"].map(crit_labels)
    corpora = ["biored", "drugprot"]
    pair_types = ["gene-drug", "gene-disease"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, pt in zip(axes, pair_types):
        sub = plot_df[plot_df["pair_type"] == pt]
        x = range(len(corpora))
        width = 0.25
        for i, crit in enumerate(["disease_neoplasm", "gene_civic", "literature_mesh"]):
            vals = [
                sub[(sub["corpus"] == c) & (sub["criterion"] == crit)]["fraction"].iloc[0] * 100
                if len(sub[(sub["corpus"] == c) & (sub["criterion"] == crit)]) else 0
                for c in corpora
            ]
            offset = (i - 1) * width
            ax.bar([xi + offset for xi in x], vals, width, label=crit_labels[crit])
        ax.set_xticks(list(x))
        ax.set_xticklabels(["BioRED", "DrugProt"])
        ax.set_title(pt.replace("-", "–"))
        ax.set_ylabel("Oncology-related fraction (%)")
        ax.set_ylim(0, 100)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Training-corpus oncology signal by independent criterion")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_oncology_fraction_by_criterion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_oncology_agreement(agreement_df: pd.DataFrame) -> None:
    """Conservative intersection (all criteria) as primary sufficiency figure."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = []
    values = []
    for _, row in agreement_df.iterrows():
        labels.append(f"{row['corpus']}\n{row['pair_type']}")
        values.append(100 * float(row["fraction_all_three"]))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]
    ax.bar(labels, values, color=colors[: len(labels)])
    ax.set_ylabel("Fraction meeting all criteria (%)")
    ax.set_title("Conservative oncology subset (all three criteria agree)")
    ax.set_ylim(0, max(values) * 1.2 if values else 100)
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_oncology_criteria_intersection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
