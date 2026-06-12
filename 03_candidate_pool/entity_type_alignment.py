"""Read-only entity-type alignment diagnostic (PubTator, CIViC, training corpora)."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    FROZEN_POOL_CSV,
    FROZEN_PROTOCOL_JSON,
    OUTPUT_DIR,
    PUBTATOR_CACHE_JSON,
    PUBTATOR_TYPE_MAP,
    REPORT_DIR,
)
from .parse import parse_entities
from .pool_builder import load_frozen_positives

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_en = __import__(
    "01_corpus_relevance.entity_normalization",
    fromlist=["normalize_entity_type", "normalization_notes"],
)

ALIGNMENT_CSV = OUTPUT_DIR / "03_candidate_pool_entity_type_alignment.csv"
EXAMPLES_CSV = OUTPUT_DIR / "03_candidate_pool_entity_type_examples.csv"
SUMMARY_JSON = OUTPUT_DIR / "03_candidate_pool_entity_type_alignment_summary.json"


def _load_pubtator_docs() -> dict[str, dict]:
    return json.loads(PUBTATOR_CACHE_JSON.read_text(encoding="utf-8"))


def _enumerate_pubtator_types(docs: dict[str, dict]) -> tuple[pd.DataFrame, Counter, Counter]:
    raw_counts: Counter = Counter()
    civic_counts: Counter = Counter()
    db_by_civic: dict[str, Counter] = defaultdict(Counter)

    for doc in docs.values():
        for ent in parse_entities(doc):
            raw_counts[ent["pubtator_type"]] += 1
            civic_counts[ent["civic_type"]] += 1
            db_by_civic[ent["civic_type"]][str(ent.get("database"))] += 1

    rows: list[dict[str, Any]] = []
    for pt_label, civic in sorted(PUBTATOR_TYPE_MAP.items()):
        rows.append(
            {
                "source": "pubtator3",
                "raw_label": pt_label,
                "civic_role": civic,
                "n_annotations": int(raw_counts.get(pt_label, 0)),
            }
        )
    return pd.DataFrame(rows), raw_counts, civic_counts


def _enumerate_civic_types() -> pd.DataFrame:
    targets = load_frozen_positives()
    head = targets["head_type"].value_counts()
    tail = targets["tail_type"].value_counts()
    rows: list[dict[str, Any]] = []
    for role in ["gene", "drug", "disease"]:
        rows.append(
            {
                "source": "civic_targets",
                "raw_label": role,
                "civic_role": role,
                "n_slots": int(head.get(role, 0) + tail.get(role, 0)),
                "n_head": int(head.get(role, 0)),
                "n_tail": int(tail.get(role, 0)),
            }
        )
    return pd.DataFrame(rows)


def _enumerate_training_corpus_types() -> tuple[pd.DataFrame, dict[str, Counter]]:
    """Entity-type labels in BioRED and DrugProt train/validation splits (on disk via HuggingFace)."""
    from datasets import load_dataset

    corpus_rows: list[dict[str, Any]] = []
    raw_by_corpus: dict[str, Counter] = {}

    specs = [
        ("biored", "biored_bigbio_kb", "train"),
        ("biored", "biored_bigbio_kb", "validation"),
        ("drugprot", "drugprot_bigbio_kb", "train"),
        ("drugprot", "drugprot_bigbio_kb", "validation"),
    ]
    for corpus, cfg, split in specs:
        ds = load_dataset(f"bigbio/{corpus}", cfg, trust_remote_code=True, split=split)
        raw: Counter = Counter()
        mapped: Counter = Counter()
        for doc in ds:
            for ent in doc.get("entities") or []:
                t = str(ent.get("type") or "")
                raw[t] += 1
                m = _en.normalize_entity_type(t)
                if m:
                    mapped[m] += 1
        raw_by_corpus[f"{corpus}_{split}"] = raw
        for label, n in raw.items():
            civic = _en.normalize_entity_type(label)
            corpus_rows.append(
                {
                    "corpus": corpus,
                    "split": split,
                    "raw_label": label,
                    "civic_role": civic or "",
                    "n_entities": int(n),
                }
            )
    return pd.DataFrame(corpus_rows), raw_by_corpus


def _training_labels_for_role(corpus_df: pd.DataFrame, role: str) -> str:
    sub = corpus_df[(corpus_df["civic_role"] == role) & (corpus_df["n_entities"] > 0)]
    if sub.empty:
        return "(none stored)"
    parts: list[str] = []
    for corpus in ["biored", "drugprot"]:
        csub = sub[sub["corpus"] == corpus]
        if csub.empty:
            continue
        labels = sorted(set(csub["raw_label"]))
        parts.append(f"{corpus}: {', '.join(labels)}")
    return "; ".join(parts) if parts else "(none stored)"


def _pool_type_counts(pool: pd.DataFrame) -> dict[str, Any]:
    primary = pool[pool["scope"] == "primary"]
    gd = primary[primary["pair_type"] == "gene-drug"]
    gdis = primary[primary["pair_type"] == "gene-disease"]

    targets = load_frozen_positives()
    civic_drugs = set(targets.loc[targets["tail_type"] == "drug", "tail_entity"].astype(str).str.lower())
    civic_diseases = set(
        targets.loc[targets["tail_type"] == "disease", "tail_entity"].astype(str).str.lower()
    )

    gd_neg = gd[~gd["is_civic_positive"]]
    gd_neg_tail = gd_neg["tail_entity"].astype(str).str.lower()
    n_gd_neg_nontherapy = int((~gd_neg_tail.isin(civic_drugs)).sum())

    gdis_neg = gdis[~gdis["is_civic_positive"]]
    gdis_neg_tail = gdis_neg["tail_entity"].astype(str).str.lower()
    n_gdis_neg_noncivic = int((~gdis_neg_tail.isin(civic_diseases)).sum())

    return {
        "n_primary_candidates": int(len(primary)),
        "n_gene_drug_candidates": int(len(gd)),
        "n_gene_drug_positives": int(gd["is_civic_positive"].sum()),
        "n_gene_drug_neg_non_civic_drug_lexicon": n_gd_neg_nontherapy,
        "n_gene_disease_candidates": int(len(gdis)),
        "n_gene_disease_positives": int(gdis["is_civic_positive"].sum()),
        "n_gene_disease_neg_non_civic_disease_lexicon": n_gdis_neg_noncivic,
        "n_unique_pubtator_chemical_tails_gd": int(gd["tail_entity"].nunique()),
        "n_unique_civic_drugs_in_targets": int(len(civic_drugs)),
    }


def _build_correspondence_table(
    corpus_df: pd.DataFrame,
    pool_stats: dict[str, Any],
    civic_counts: Counter,
) -> pd.DataFrame:
    """Correspondence for gene, drug, disease roles across three systems."""
    roles = [
        {
            "civic_role": "gene",
            "pubtator_label": "Gene",
            "pubtator_n": int(civic_counts.get("gene", 0)),
            "mapping_status": "granularity_gap",
            "mapping_note": (
                "PubTator Gene maps one-to-one to CIViC gene by label, but BioRED "
                "GeneOrGeneProduct is broader (gene products and families)."
            ),
            "n_candidates_affected": pool_stats["n_gene_drug_candidates"] + pool_stats["n_gene_disease_candidates"],
            "n_positives_affected": pool_stats["n_gene_drug_positives"] + pool_stats["n_gene_disease_positives"],
        },
        {
            "civic_role": "drug",
            "pubtator_label": "Chemical",
            "pubtator_n": int(civic_counts.get("drug", 0)),
            "mapping_status": "granularity_gap",
            "mapping_note": (
                "PubTator Chemical is broader than CIViC drug/therapy; includes non-therapeutic "
                "compounds and reagents tagged with MeSH chemical IDs."
            ),
            "n_candidates_affected": pool_stats["n_gene_drug_candidates"],
            "n_positives_affected": pool_stats["n_gene_drug_positives"],
        },
        {
            "civic_role": "disease",
            "pubtator_label": "Disease",
            "pubtator_n": int(civic_counts.get("disease", 0)),
            "mapping_status": "granularity_gap",
            "mapping_note": (
                "PubTator Disease aligns to CIViC disease, but BioRED DiseaseOrPhenotypicFeature "
                "also includes phenotypic features during training."
            ),
            "n_candidates_affected": pool_stats["n_gene_disease_candidates"],
            "n_positives_affected": pool_stats["n_gene_disease_positives"],
        },
    ]
    rows: list[dict[str, Any]] = []
    for r in roles:
        rows.append(
            {
                "civic_role": r["civic_role"],
                "pubtator_label": r["pubtator_label"],
                "pubtator_annotation_count": r["pubtator_n"],
                "civic_correspondence": r["civic_role"],
                "biored_drugprot_correspondence": _training_labels_for_role(corpus_df, r["civic_role"]),
                "mapping_status": r["mapping_status"],
                "mapping_note": r["mapping_note"],
                "n_pool_candidates_on_role": r["n_candidates_affected"],
                "n_pool_positives_on_role": r["n_positives_affected"],
            }
        )
    rows.append(
        {
            "civic_role": "variant",
            "pubtator_label": "Variant, Mutation",
            "pubtator_annotation_count": int(civic_counts.get("variant", 0)),
            "civic_correspondence": "variant",
            "biored_drugprot_correspondence": "biored: SequenceVariant",
            "mapping_status": "inconsistent",
            "mapping_note": (
                "Variant pairing is descriptive-only; tmVar3 surface forms rarely match CIViC "
                "variant strings (see D1)."
            ),
            "n_pool_candidates_on_role": 0,
            "n_pool_positives_on_role": 0,
        }
    )
    return pd.DataFrame(rows)


def _collect_examples(
    pool: pd.DataFrame,
    docs: dict[str, dict],
    corpus_df: pd.DataFrame,
    max_per_class: int = 3,
) -> pd.DataFrame:
    """Short anonymised examples per mismatch class."""
    examples: list[dict[str, str]] = []
    primary = pool[pool["scope"] == "primary"]
    gd_neg = primary[(primary["pair_type"] == "gene-drug") & (~primary["is_civic_positive"])]

    targets = load_frozen_positives()
    civic_drugs = set(targets.loc[targets["tail_type"] == "drug", "tail_entity"].astype(str).str.lower())

    # Chemical breadth: gene-drug distractors whose tail is not any CIViC drug string
    chem_spurious = gd_neg[~gd_neg["tail_entity"].astype(str).str.lower().isin(civic_drugs)]
    seen: set[str] = set()
    for _, r in chem_spurious.iterrows():
        tail = str(r["tail_entity"])
        if tail.lower() in seen:
            continue
        seen.add(tail.lower())
        examples.append(
            {
                "mismatch_class": "chemical_broader_than_civic_drug",
                "entity_string": tail,
                "pubtator_type": "Chemical",
                "civic_or_corpus_type": "CIViC drug/therapy (curated name set)",
                "why_mismatch": (
                    "PubTator tags this as Chemical; pool pairs it as gene-drug, but the string "
                    "does not appear among CIViC curated drug names (reagent/metabolite-class distractor)."
                ),
                "pmid": str(r["pmid"]),
            }
        )
        if len(seen) >= max_per_class:
            break

    # Training phenotype breadth (from corpus labels, not pool)
    examples.append(
        {
            "mismatch_class": "disease_phenotype_training_breadth",
            "entity_string": "(BioRED entity type label)",
            "pubtator_type": "Disease (evaluation pool)",
            "civic_or_corpus_type": "DiseaseOrPhenotypicFeature (BioRED training)",
            "why_mismatch": (
                "Training negatives and positives include phenotypic features under one BigBio "
                "label; PubTator and CIViC use a narrower disease/therapy naming scope at evaluation."
            ),
            "pmid": "",
        }
    )

    # Gene product breadth
    examples.append(
        {
            "mismatch_class": "gene_product_training_breadth",
            "entity_string": "(e.g. full protein name in BioRED)",
            "pubtator_type": "Gene (evaluation pool)",
            "civic_or_corpus_type": "GeneOrGeneProduct (BioRED training)",
            "why_mismatch": (
                "BioRED gene arguments include gene products and symbols; CIViC gene features are "
                "curated symbols; PubTator Gene tags may differ in span granularity (see recall section)."
            ),
            "pmid": "",
        }
    )

    # DrugProt GENE-N (non-protein-coding) vs CIViC gene
    dp = corpus_df[(corpus_df["corpus"] == "drugprot") & (corpus_df["raw_label"] == "GENE-N")]
    if not dp.empty:
        examples.append(
            {
                "mismatch_class": "drugprot_gene_n_training",
                "entity_string": "(DrugProt GENE-N span)",
                "pubtator_type": "Gene (evaluation pool)",
                "civic_or_corpus_type": "GENE-N mapped to gene (DrugProt training)",
                "why_mismatch": (
                    "DrugProt distinguishes protein-coding (GENE-Y) from non-coding (GENE-N) genes; "
                    "both collapse to CIViC gene, and PubTator Gene does not preserve this distinction."
                ),
                "pmid": "",
            }
        )

    return pd.DataFrame(examples)


def _train_eval_consistency_note(corpus_df: pd.DataFrame) -> dict[str, str]:
    """Characterise training vs evaluation type semantics."""
    biored_drug = corpus_df[
        (corpus_df["corpus"] == "biored") & (corpus_df["raw_label"] == "ChemicalEntity")
    ]["n_entities"].sum()
    biored_dis = corpus_df[
        (corpus_df["corpus"] == "biored") & (corpus_df["raw_label"] == "DiseaseOrPhenotypicFeature")
    ]["n_entities"].sum()
    dp_chem = corpus_df[
        (corpus_df["corpus"] == "drugprot") & (corpus_df["raw_label"] == "CHEMICAL")
    ]["n_entities"].sum()

    return {
        "summary": (
            "Training and evaluation both collapse raw labels to three CIViC-aligned roles (gene, "
            "drug, disease) before pairing, but through different source ontologies. BioRED uses "
            "GeneOrGeneProduct, ChemicalEntity, and DiseaseOrPhenotypicFeature; DrugProt uses "
            "CHEMICAL and GENE-Y/GENE-N/GENE; the frozen pool uses PubTator Gene, Chemical, and "
            "Disease via a separate mapping table. Marked spans at training time carry BioRED/DrugProt "
            "boundaries; at evaluation the pool uses PubTator boundaries on the same PMIDs. Type "
            "semantics are therefore aligned at the role level but not at the ontology level."
        ),
        "direction_of_effect": (
            "Broader training categories (especially DiseaseOrPhenotypicFeature and ChemicalEntity) "
            "and broader PubTator Chemical tags at pool construction can inject gene-drug and "
            "gene-disease distractors that CIViC would not curate, inflating pools common-mode for "
            "all models. This is separate from the PubTator recall ceiling in the recall section."
        ),
        "biored_chemical_entities": int(biored_drug),
        "biored_disease_or_phenotype_entities": int(biored_dis),
        "drugprot_chemical_entities": int(dp_chem),
    }


def run_entity_type_alignment_diagnostic() -> dict[str, Any]:
    """Run full read-only diagnostic; write CSV, JSON, return summary for report."""
    docs = _load_pubtator_docs()
    pool = pd.read_csv(FROZEN_POOL_CSV)
    cls_path = OUTPUT_DIR / "03_candidate_pool_pubtator_recall_classification.csv"
    n_matched = int(pd.read_csv(cls_path)["matched_in_pool"].sum()) if cls_path.exists() else None

    pubtator_df, raw_counts, civic_counts = _enumerate_pubtator_types(docs)
    civic_df = _enumerate_civic_types()
    corpus_df, _raw_by_corpus = _enumerate_training_corpus_types()
    pool_stats = _pool_type_counts(pool)
    correspondence = _build_correspondence_table(corpus_df, pool_stats, civic_counts)
    train_eval = _train_eval_consistency_note(corpus_df)
    examples = _collect_examples(pool, docs, corpus_df)

    correspondence.to_csv(ALIGNMENT_CSV, index=False)
    examples.to_csv(EXAMPLES_CSV, index=False)

    summary: dict[str, Any] = {
        "pubtator_raw_types": dict(raw_counts),
        "pubtator_mapped_counts": dict(civic_counts),
        "pubtator_type_map": PUBTATOR_TYPE_MAP,
        "civic_target_types": civic_df.to_dict(orient="records"),
        "training_corpus_types": corpus_df.to_dict(orient="records"),
        "pool_stats": pool_stats,
        "train_eval_consistency": train_eval,
        "n_matched_relations": n_matched,
        "correspondence": correspondence.to_dict(orient="records"),
        "examples": examples.to_dict(orient="records"),
        "sources": {
            "frozen_pool": str(FROZEN_POOL_CSV),
            "pubtator_cache": str(PUBTATOR_CACHE_JSON),
                "civic_targets": str(FROZEN_PROTOCOL_JSON),
            "training_corpora": "bigbio/biored and bigbio/drugprot train+validation splits",
            "recall_classification": str(cls_path) if cls_path.exists() else None,
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Entity-type correspondence table ===")
    print(
        correspondence[
            [
                "civic_role",
                "pubtator_label",
                "civic_correspondence",
                "mapping_status",
                "n_pool_candidates_on_role",
                "n_pool_positives_on_role",
            ]
        ].to_string(index=False)
    )
    print("\n=== Affected-counts summary (primary pool, frozen) ===")
    print(f"  PubTator parsed annotations: Gene={civic_counts.get('gene',0)}, "
          f"Chemical→drug={civic_counts.get('drug',0)}, Disease={civic_counts.get('disease',0)}")
    print(f"  Primary gene-drug candidates: {pool_stats['n_gene_drug_candidates']} "
          f"({pool_stats['n_gene_drug_positives']} CIViC positives)")
    print(f"  Gene-drug negatives with tail outside CIViC drug name set: "
          f"{pool_stats['n_gene_drug_neg_non_civic_drug_lexicon']} "
          f"(source: pool_candidates.csv vs frozen_protocol.json drug strings)")
    print(f"  Primary gene-disease candidates: {pool_stats['n_gene_disease_candidates']} "
          f"({pool_stats['n_gene_disease_positives']} CIViC positives)")
    print(f"  Unique PubTator chemical tails in gene-drug pool: "
          f"{pool_stats['n_unique_pubtator_chemical_tails_gd']} vs "
          f"{pool_stats['n_unique_civic_drugs_in_targets']} unique CIViC drug strings in targets")
    if n_matched is not None:
        print(f"  Pool-positive relations (recall CSV): {n_matched}/1812 (unchanged by this diagnostic)")

    return summary


def alignment_report_section(summary: dict[str, Any]) -> str:
    """Plain prose section for step-03 report."""
    pool_stats = summary["pool_stats"]
    te = summary["train_eval_consistency"]
    corr = pd.DataFrame(summary["correspondence"])
    examples = summary.get("examples", [])

    raw = summary["pubtator_raw_types"]
    civic_ann = summary["pubtator_mapped_counts"]

    gd_neg_nc = pool_stats["n_gene_drug_neg_non_civic_drug_lexicon"]
    n_gd = pool_stats["n_gene_drug_candidates"]
    n_gd_pos = pool_stats["n_gene_drug_positives"]
    n_gdis = pool_stats["n_gene_disease_candidates"]

    lines = [
        "## Entity-type system alignment (PubTator, CIViC, training corpora)",
        "",
        "Relation extraction pairs entities by type when building the frozen candidate pool. "
        "Three type systems meet at this step, and they are not identical. PubTator3 supplies "
        f"entity labels used to enumerate pool candidates (Gene {raw.get('Gene', 0)}, Chemical "
        f"{raw.get('Chemical', 0)}, Disease {raw.get('Disease', 0)}, Variant {raw.get('Variant', 0)} "
        f"annotations across cached abstracts, mapped to CIViC-aligned roles before pairing). "
        "CIViC evaluation targets use gene, drug/therapy, and disease roles on curated entity "
        "strings (1812 primary relations: gene heads with drug or disease tails). BioRED and "
        "DrugProt supply the entity-type labels on marked argument spans during fine-tuning "
        "(BioRED: GeneOrGeneProduct, ChemicalEntity, DiseaseOrPhenotypicFeature; DrugProt: "
        "CHEMICAL, GENE-Y, GENE-N, GENE). Both training pipelines and the pool builder collapse "
        "these labels to the same three CIViC-aligned roles before forming gene-drug and "
        "gene-disease pairs, but the collapse uses different source ontologies.",
        "",
        "For gene, PubTator Gene maps to CIViC gene with the same role name, yet BioRED "
        "GeneOrGeneProduct is broader (gene products and related mentions). For drug, PubTator "
        "Chemical maps to CIViC drug/therapy, but Chemical is wider than curated therapy names: "
        f"{n_gd} primary gene-drug candidates are built from Gene–Chemical co-occurrence, of "
        f"which {n_gd_pos} are CIViC positives and {gd_neg_nc} negative candidates use a "
        "PubTator chemical surface form that does not appear among CIViC curated drug strings "
        "(read from the frozen pool and target list; a conservative proxy for non-therapy "
        "chemical distractors). For disease, PubTator Disease maps to CIViC disease, while "
        "BioRED DiseaseOrPhenotypicFeature also includes phenotypic features in training. "
        f"All {n_gdis} primary gene-disease candidates inherit the disease-side mapping.",
        "",
        te["summary"],
        "",
        te["direction_of_effect"],
        "",
        "Effect on the frozen pool is quantifiable but intentionally not repaired here. Every "
        "primary candidate inherits the PubTator-side type mapping; the gene-drug inflation from "
        "Chemical breadth affects all models equally because all encoders score the same frozen "
        "pool. The 1590 pool-positive relations and 222 recall-limited misses documented above "
        "are unchanged; type-granularity mismatch adds distractors and train/eval boundary "
        "shift on top of the recall ceiling already described in the PubTator recall section "
        "(multi-word drug span mismatches overlap with chemical-breadth distractors).",
        "",
        "This diagnostic does not identify pair-type-specific bias across model families: the "
        "pool is shared and type pairing rules are fixed. Any encoder-specific sensitivity would "
        "be indirect (through architecture), not through a different candidate set.",
        "",
        "Examples (anonymised):",
    ]

    for ex in examples[:6]:
        pmid_note = f" PMID {ex['pmid']}." if ex.get("pmid") else ""
        lines.append(
            f"- {ex['mismatch_class']}: \"{ex['entity_string']}\" — PubTator type {ex['pubtator_type']}, "
            f"corpus/target type {ex['civic_or_corpus_type']}. {ex['why_mismatch']}{pmid_note}"
        )

    lines.extend(
        [
            "",
            f"Auditable table: 03_candidate_pool_entity_type_alignment.csv ({len(corr)} role rows). "
            "Figure: 03_candidate_pool_entity_type_alignment.png.",
        ]
    )
    return "\n".join(lines)


def patch_report_with_alignment_section(section: str) -> None:
    """Insert or replace alignment section in step-03 report."""
    report_path = REPORT_DIR / "report.md"
    marker = "## Entity-type system alignment (PubTator, CIViC, training corpora)"
    if not report_path.exists():
        report_path.write_text(section.strip() + "\n", encoding="utf-8")
        print(f"\nReport written to {report_path}")
        return

    existing = report_path.read_text(encoding="utf-8")
    if marker in existing:
        before = existing.split(marker)[0].rstrip()
        after_parts = existing.split(marker, 1)[1]
        for end_marker in (
            "\n## Ranking-feasibility verdict",
            "\n## PubTator recall and entity-span limitation",
        ):
            if end_marker in after_parts:
                after = after_parts.split(end_marker, 1)[1]
                existing = before + "\n\n" + section.strip() + "\n\n" + end_marker.lstrip("\n") + after
                break
        else:
            existing = before + "\n\n" + section.strip() + "\n"
    else:
        insert_before = "## Ranking-feasibility verdict"
        if insert_before in existing:
            existing = existing.replace(
                insert_before,
                section.strip() + "\n\n" + insert_before,
            )
        elif "## PubTator recall and entity-span limitation" in existing:
            existing = existing.replace(
                "## PubTator recall and entity-span limitation",
                section.strip() + "\n\n## PubTator recall and entity-span limitation",
            )
        else:
            existing = existing.rstrip() + "\n\n" + section.strip() + "\n"

    report_path.write_text(existing, encoding="utf-8")
    print(f"\nReport updated: {report_path}")


def refresh_entity_type_alignment() -> dict[str, Any]:
    """Run diagnostic, figure, and patch report."""
    from .figures import plot_entity_type_alignment

    summary = run_entity_type_alignment_diagnostic()
    plot_entity_type_alignment(pd.DataFrame(summary["correspondence"]), summary["pool_stats"])
    section = alignment_report_section(summary)
    patch_report_with_alignment_section(section)
    return summary
