"""Part D: DrugProt mechanism -> CIViC clinical significance projectability."""

from __future__ import annotations

import json

import pandas as pd

from .config import CIVIC_INVENTORY_FILE, OUTPUT_DIR, TRAIN_STATS_FILE

DRUGPROT_TO_CIVIC = [
    ("INHIBITOR", "RESISTANCE", "PREDICTIVE", "ambiguous", "Mechanism ≠ clinical resistance outcome."),
    ("INHIBITOR", "SENSITIVITYRESPONSE", "PREDICTIVE", "ambiguous", "Conflicts with RESISTANCE mapping."),
    ("ACTIVATOR", "SENSITIVITYRESPONSE", "PREDICTIVE", "ambiguous", "Activation context-dependent."),
    ("ANTAGONIST", "SENSITIVITYRESPONSE", "PREDICTIVE", "ambiguous", "Pharmacology ≠ clinical response."),
    ("AGONIST", "SENSITIVITYRESPONSE", "PREDICTIVE", "ambiguous", "Pharmacology ≠ clinical response."),
    ("AGONIST-ACTIVATOR", "SENSITIVITYRESPONSE", "PREDICTIVE", "ambiguous", "Rare composite label."),
    ("INDIRECT-UPREGULATOR", "(none)", "(none)", "no_mapping", "Regulatory mechanism only."),
    ("INDIRECT-DOWNREGULATOR", "(none)", "(none)", "no_mapping", "Regulatory mechanism only."),
    ("DIRECT-REGULATOR", "(none)", "(none)", "no_mapping", "Broad regulatory mechanism."),
    ("SUBSTRATE", "(none)", "(none)", "no_mapping", "Metabolic relation."),
    ("PRODUCT-OF", "(none)", "(none)", "no_mapping", "Metabolic relation."),
    ("PART-OF", "(none)", "(none)", "no_mapping", "Structural relation."),
    ("AGONIST-INHIBITOR", "(none)", "(none)", "no_mapping", "Self-contradictory composite."),
    ("SUBSTRATE_PRODUCT-OF", "(none)", "(none)", "no_mapping", "Composite metabolic label."),
]


def run_drugprot_mapping() -> pd.DataFrame:
    train_stats = json.loads(TRAIN_STATS_FILE.read_text(encoding="utf-8"))
    counts = train_stats["corpora"]["drugprot"]["relation_type_counts"]

    rows = []
    for label, sig, etype, status, note in DRUGPROT_TO_CIVIC:
        rows.append(
            {
                "drugprot_label": label,
                "proposed_civic_significance": sig,
                "proposed_civic_evidence_type": etype,
                "mapping_status": status,
                "ambiguous_because": note,
                "drugprot_train_count": counts.get(label, 0),
                "deterministic_mapping_possible": status == "clean",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "drugprot_civic_mapping.csv", index=False)

    civic = pd.read_csv(CIVIC_INVENTORY_FILE)
    gene_drug = civic[
        (civic["is_evaluable_target"])
        & (civic["entity_pair_type"].str.replace("–", "-") == "gene-drug")
    ]
    civic_dist = (
        gene_drug.groupby(["evidence_type", "clinical_significance", "evidence_direction"], dropna=False)
        .size()
        .reset_index(name="civic_count")
    )
    civic_dist.to_csv(OUTPUT_DIR / "01_corpus_civic_gene_drug_labels.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "clean_mappings", "value": int((df["mapping_status"] == "clean").sum())},
            {"metric": "ambiguous_mappings", "value": int((df["mapping_status"] == "ambiguous").sum())},
            {"metric": "no_mapping", "value": int((df["mapping_status"] == "no_mapping").sum())},
            {"metric": "deterministic_projection_possible", "value": "no"},
        ]
    )
    summary.to_csv(OUTPUT_DIR / "01_corpus_drugprot_mapping_summary.csv", index=False)

    print("\n=== Part D: DrugProt -> CIViC projectability ===")
    print(f"  clean: 0 | ambiguous: {(df['mapping_status']=='ambiguous').sum()} | no_mapping: {(df['mapping_status']=='no_mapping').sum()}")
    return df
