"""CIViC alignment matrix and coverage scores."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from .config import (
    CIVIC_PAIR_TYPES,
    CIVIC_PAIR_WEIGHTS_FILE,
    CORPORA,
    INVENTORY_FILE,
    OUTPUT_DIR,
    FIGURE_DIR,
)


def _load_civic_weights() -> dict[str, float]:
    df = pd.read_csv(CIVIC_PAIR_WEIGHTS_FILE)
    df["entity_pair_type"] = df["entity_pair_type"].str.replace("–", "-", regex=False).str.lower()
    return dict(zip(df["entity_pair_type"], df["share_of_evaluable"]))


def _admissibility(entry: dict) -> tuple[str, str]:
    if entry.get("load_status") != "ok":
        return "Not admissible", entry.get("load_error") or "Could not load corpus."
    civic_counts = entry.get("civic_pair_counts", {})
    covered = [p for p in CIVIC_PAIR_TYPES if civic_counts.get(p, 0) > 0]
    if not covered:
        return "Not admissible", "No training relations match any CIViC evaluation entity-pair type."
    if len(covered) == len(CIVIC_PAIR_TYPES):
        return "Admissible", "Covers all four CIViC evaluation entity-pair types."
    return "Partially admissible", f"Covers {len(covered)} of 4 CIViC pair types: {', '.join(covered)}."


def run_alignment(inventories: dict | None = None) -> dict:
    inventories = inventories or json.loads(INVENTORY_FILE.read_text(encoding="utf-8"))
    civic_weights = _load_civic_weights()
    corpus_keys = list(CORPORA.keys())

    matrix_rows = []
    for civic_pair in CIVIC_PAIR_TYPES:
        row = {"civic_pair_type": civic_pair, "civic_eval_share": civic_weights.get(civic_pair, 0.0)}
        for key in corpus_keys:
            count = inventories["corpora"][key]["civic_pair_counts"].get(civic_pair, 0)
            row[f"{key}_relation_count"] = count
            row[f"{key}_covers"] = count > 0
        matrix_rows.append(row)

    matrix_df = pd.DataFrame(matrix_rows)
    matrix_df.to_csv(OUTPUT_DIR / "corpus_alignment_matrix.csv", index=False)

    coverage_rows = []
    for key in corpus_keys:
        entry = inventories["corpora"][key]
        civic_counts = entry["civic_pair_counts"]
        score = sum(civic_weights.get(p, 0) for p in CIVIC_PAIR_TYPES if civic_counts.get(p, 0) > 0)
        adm, reason = _admissibility(entry)
        coverage_rows.append(
            {
                "corpus": key,
                "display_name": CORPORA[key]["display_name"],
                "language": CORPORA[key]["language"],
                "role": CORPORA[key]["role"],
                "total_relations": entry["total_relations"],
                "split_sizes": str(entry["split_sizes"]),
                "pairs_covered": sum(1 for p in CIVIC_PAIR_TYPES if civic_counts.get(p, 0) > 0),
                "pairs_covered_of_4": f"{sum(1 for p in CIVIC_PAIR_TYPES if civic_counts.get(p, 0) > 0)}/4",
                "civic_relevance_pct": round(100 * score, 1),
                "admissibility": adm,
                "admissibility_reason": reason,
                **{f"count_{p.replace('-', '_')}": civic_counts.get(p, 0) for p in CIVIC_PAIR_TYPES},
            }
        )

    coverage_df = pd.DataFrame(coverage_rows).sort_values("civic_relevance_pct", ascending=False)
    coverage_df.to_csv(OUTPUT_DIR / "corpus_civic_relevance.csv", index=False)

    _plot_heatmap(matrix_df, corpus_keys)
    _plot_relevance(coverage_df)

    print("\n=== Part A: CIViC alignment ===")
    print(coverage_df[["display_name", "civic_relevance_pct", "admissibility", "pairs_covered_of_4"]].to_string(index=False))
    return {"matrix": matrix_df, "coverage": coverage_df}


def _plot_heatmap(matrix_df: pd.DataFrame, corpus_keys: list[str]) -> None:
    data = matrix_df[[f"{k}_covers" for k in corpus_keys]].astype(int).values
    labels_x = [CORPORA[k]["display_name"] for k in corpus_keys]
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=20, ha="right")
    ax.set_yticks(range(len(matrix_df)))
    ax.set_yticklabels(matrix_df["civic_pair_type"])
    ax.set_title("CIViC pair type covered by each corpus")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            n = int(matrix_df.iloc[i][f"{corpus_keys[j]}_relation_count"])
            ax.text(j, i, f"yes\n(n={n})" if n else "no", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_corpus_alignment_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_relevance(coverage_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(coverage_df["display_name"], coverage_df["civic_relevance_pct"], color="#4C72B0")
    ax.set_ylabel("CIViC-relevance (%)")
    ax.set_title("Weighted CIViC evaluation coverage by corpus")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_corpus_relevance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
