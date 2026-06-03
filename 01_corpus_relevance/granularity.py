"""Part B: label-granularity ladder for BioRED, DrugProt, and CIViC."""

from __future__ import annotations

import json

import pandas as pd

from .config import CIVIC_INVENTORY_FILE, INVENTORY_FILE, OUTPUT_DIR

BIORED_LABELS = {
    "Association": ("1_coarse_association", "Undirected general association.", "high"),
    "Positive_Correlation": ("2_directional_correlation", "Positive correlation.", "high"),
    "Negative_Correlation": ("2_directional_correlation", "Negative correlation.", "high"),
    "Bind": ("3_fine_mechanism", "Physical binding.", "medium"),
    "Cotreatment": ("3_fine_mechanism", "Co-administration of treatments.", "high"),
    "Drug_Interaction": ("3_fine_mechanism", "Drug-drug interaction.", "high"),
    "Comparison": ("3_fine_mechanism", "Comparative statement.", "high"),
    "Conversion": ("3_fine_mechanism", "Chemical conversion.", "medium"),
}

DRUGPROT_LABELS = {
    "INHIBITOR": ("3_fine_mechanism", "Drug inhibits gene/protein.", "high"),
    "ACTIVATOR": ("3_fine_mechanism", "Drug activates gene/protein.", "high"),
    "ANTAGONIST": ("3_fine_mechanism", "Drug antagonises target.", "high"),
    "AGONIST": ("3_fine_mechanism", "Drug agonises target.", "high"),
    "AGONIST-ACTIVATOR": ("3_fine_mechanism", "Combined agonist/activator.", "high"),
    "AGONIST-INHIBITOR": ("3_fine_mechanism", "Combined agonist/inhibitor.", "high"),
    "INDIRECT-UPREGULATOR": ("3_fine_mechanism", "Indirect up-regulation.", "medium"),
    "INDIRECT-DOWNREGULATOR": ("3_fine_mechanism", "Indirect down-regulation.", "medium"),
    "DIRECT-REGULATOR": ("3_fine_mechanism", "Direct regulation.", "medium"),
    "SUBSTRATE": ("3_fine_mechanism", "Metabolic substrate.", "medium"),
    "PRODUCT-OF": ("3_fine_mechanism", "Metabolic product.", "medium"),
    "PART-OF": ("3_fine_mechanism", "Structural part-of.", "low"),
    "SUBSTRATE_PRODUCT-OF": ("3_fine_mechanism", "Combined substrate/product.", "medium"),
}


def _civic_labels() -> list[dict]:
    civic = pd.read_csv(CIVIC_INVENTORY_FILE)
    civic = civic[civic["is_evaluable_target"]]
    grouped = civic.groupby(
        ["evidence_type", "clinical_significance", "evidence_direction"], dropna=False
    ).size().reset_index(name="count")

    rows = []
    for _, row in grouped.iterrows():
        et = row["evidence_type"] if pd.notna(row["evidence_type"]) else "(missing)"
        sig = row["clinical_significance"] if pd.notna(row["clinical_significance"]) else "(missing)"
        direction = row["evidence_direction"]
        level = (
            "4_clinical_significance"
            if pd.notna(row["clinical_significance"]) and str(row["clinical_significance"]).strip()
            else "3_fine_mechanism"
        )
        rows.append(
            {
                "corpus": "civic",
                "display_name": "CIViC",
                "label": f"{et}|{sig}|{direction}",
                "granularity_level": level,
                "count": int(row["count"]),
                "maps_cleanly_to_training": False,
                "over_attribution_risk": "n/a",
                "mapping_note": "Evaluation KB — not projected onto training corpora.",
            }
        )
    return rows


def run_granularity(inventories: dict | None = None) -> pd.DataFrame:
    inventories = inventories or json.loads(INVENTORY_FILE.read_text(encoding="utf-8"))
    rows = []

    for key, label_defs in [("biored", BIORED_LABELS), ("drugprot", DRUGPROT_LABELS)]:
        counts = inventories["corpora"][key]["relation_type_counts"]
        for label, (level, desc, risk) in label_defs.items():
            rows.append(
                {
                    "corpus": key,
                    "display_name": inventories["corpora"][key]["display_name"],
                    "label": label,
                    "granularity_level": level,
                    "count": counts.get(label, 0),
                    "maps_cleanly_to_training": False,
                    "over_attribution_risk": risk,
                    "mapping_note": desc,
                }
            )

    rows.extend(_civic_labels())
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "granularity_ladder.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "question": "Can BioRED labels nest into CIViC labels?",
                "answer": "No — Association/Positive_Correlation are too coarse.",
                "rq1_implication": "Do not collapse CIViC labels onto BioRED types.",
            },
            {
                "question": "Can DrugProt mechanism labels nest into CIViC clinical significance?",
                "answer": "No — mechanism ≠ clinical outcome.",
                "rq1_implication": "Use DrugProt for gene-drug presence pretraining only.",
            },
            {
                "question": "Shared 4-level ladder across sources?",
                "answer": "Partially — levels exist but no 1:1 label mapping.",
                "rq1_implication": "RQ1 compares tiers descriptively, not as isomorphic labels.",
            },
        ]
    )
    summary.to_csv(OUTPUT_DIR / "granularity_summary.csv", index=False)

    risky = len(df[(df["corpus"].isin(["biored", "drugprot"])) & (df["over_attribution_risk"] == "high")])
    print("\n=== Part B: Label granularity ===")
    print(f"  labels catalogued: {len(df)}")
    print(f"  high over-attribution-risk training labels: {risky}")
    return df
