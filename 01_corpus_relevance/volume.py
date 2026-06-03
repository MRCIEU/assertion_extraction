"""Part C: trainable relation volume per CIViC pair type."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from .config import (
    CIVIC_PAIR_SHARES,
    CIVIC_PAIR_TYPES,
    FIGURE_DIR,
    INVENTORY_FILE,
    OUTPUT_DIR,
    TRAIN_STATS_FILE,
    TRAIN_VOLUME_THRESHOLD,
)


def _rq3_recommendation(pair: str, combined_n: int, drugprot_n: int) -> str:
    if pair.startswith("variant"):
        if combined_n < TRAIN_VOLUME_THRESHOLD:
            return "descriptive-only; insufficient for configuration experiments"
        if pair == "variant-drug" or (drugprot_n == 0 and combined_n < 2000):
            return "descriptive-only; BioRED-only and thin — focus RQ3 on gene pairs"
        return "suitable for configuration experiments (RQ3)"
    if combined_n >= TRAIN_VOLUME_THRESHOLD:
        return "suitable for configuration experiments (RQ3)"
    if combined_n >= 100:
        return "descriptive-only; thin training signal"
    return "descriptive-only; insufficient for configuration experiments"


def run_volume() -> pd.DataFrame:
    train_stats = json.loads(TRAIN_STATS_FILE.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY_FILE.read_text(encoding="utf-8"))

    rows = []
    for key, corpus in train_stats["corpora"].items():
        for pair in CIVIC_PAIR_TYPES:
            train_n = corpus["civic_pair_counts"].get(pair, 0)
            all_n = inventory["corpora"][key]["civic_pair_counts"].get(pair, 0)
            rows.append(
                {
                    "corpus": key,
                    "display_name": corpus["display_name"],
                    "split": "train",
                    "civic_pair_type": pair,
                    "train_relations": train_n,
                    "all_split_relations": all_n,
                    "civic_eval_share": CIVIC_PAIR_SHARES[pair],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "trainable_volume.csv", index=False)

    combined = {p: 0 for p in CIVIC_PAIR_TYPES}
    for corpus in train_stats["corpora"].values():
        for p, n in corpus["civic_pair_counts"].items():
            if p in combined:
                combined[p] += n

    assessment_rows = []
    for pair in CIVIC_PAIR_TYPES:
        biored_n = train_stats["corpora"]["biored"]["civic_pair_counts"].get(pair, 0)
        drugprot_n = train_stats["corpora"]["drugprot"]["civic_pair_counts"].get(pair, 0)
        combined_n = combined[pair]
        assessment_rows.append(
            {
                "civic_pair_type": pair,
                "civic_eval_share": CIVIC_PAIR_SHARES[pair],
                "biored_train_relations": biored_n,
                "drugprot_train_relations": drugprot_n,
                "bc5cdr_train_relations": 0,
                "combined_train_relations": combined_n,
                "threshold": TRAIN_VOLUME_THRESHOLD,
                "rq3_recommendation": _rq3_recommendation(pair, combined_n, drugprot_n),
            }
        )

    assessment = pd.DataFrame(assessment_rows)
    assessment.to_csv(OUTPUT_DIR / "volume_assessment.csv", index=False)

    _plot_volume(df, assessment)

    print("\n=== Part C: Trainable volume (train + validation splits) ===")
    print(assessment.to_string(index=False))
    return df


def _plot_volume(df: pd.DataFrame, assessment: pd.DataFrame) -> None:
    pivot = df.pivot(index="civic_pair_type", columns="display_name", values="train_relations")
    pivot = pivot.reindex(CIVIC_PAIR_TYPES)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(pivot.index))
    width = 0.35
    cols = list(pivot.columns)
    if len(cols) >= 1:
        ax.bar([i - width / 2 for i in x], pivot[cols[0]], width, label=cols[0])
    if len(cols) >= 2:
        ax.bar([i + width / 2 for i in x], pivot[cols[1]], width, label=cols[1])
    ax.axhline(TRAIN_VOLUME_THRESHOLD, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index, rotation=15)
    ax.set_ylabel("Train relations")
    ax.set_title("Trainable relations per CIViC pair type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_corpus_trainable_volume.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0" if "suitable" in r else "#DD8452" for r in assessment["rq3_recommendation"]]
    ax.bar(assessment["civic_pair_type"], assessment["combined_train_relations"], color=colors)
    ax.axhline(TRAIN_VOLUME_THRESHOLD, color="red", linestyle="--", linewidth=1)
    ax.set_ylabel("Combined train relations (BioRED + DrugProt)")
    ax.set_title("Combined training volume by CIViC pair type")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_corpus_combined_volume.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
