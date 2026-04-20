"""
Audit 01: Raw data availability across all source corpora.

Checks presence, structure, and file sizes for every raw dataset
used in the project. Outputs:
  - reports/raw_data_availability.json
  - reports/tables/raw_data_summary.csv
"""
from __future__ import annotations

import csv, json
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import RAW, REPORTS, TABLES, ensure_dirs


CORPUS_SPECS = {
    "biored": {
        "path": RAW / "biored",
        "expected_files": ["Train.BioC.JSON", "Dev.BioC.JSON", "Test.BioC.JSON"],
        "supervision_type": "supervised_RE_NER",
        "annotation_level": "document_level",
        "oncology_specificity": "general_biomedical",
        "training_role": "T1_backbone",
        "note": "BioCreative VIII Track 1 — multi-type entity+relation, novelty flag",
    },
    "drugprot": {
        "path": RAW / "drugprot",
        "expected_files": [
            "training/drugprot_training_abstracs.tsv",
            "training/drugprot_training_entities.tsv",
            "training/drugprot_training_relations.tsv",
            "development/drugprot_development_abstracs.tsv",
        ],
        "supervision_type": "supervised_RE_NER",
        "annotation_level": "abstract_level",
        "oncology_specificity": "indirect_via_cancer_context",
        "training_role": "T1_backbone",
        "note": "BioCreative VII DrugProt — 13-type drug-gene mechanism. NO official test split.",
    },
    "bc5cdr": {
        "path": RAW / "bc5cdr_hf",
        "expected_files": ["train.jsonl", "validation.jsonl", "test.jsonl"],
        "supervision_type": "supervised_RE_NER",
        "annotation_level": "document_level",
        "oncology_specificity": "mixed_chemical_disease",
        "training_role": "T1_backbone",
        "note": "BC5CDR via HuggingFace BigBio — CID (Chemical Induces Disease) only.",
    },
    "civic": {
        "path": RAW / "civic",
        "expected_files": [
            "nightly-AcceptedAssertionSummaries.tsv",
            "nightly-AcceptedClinicalEvidenceSummaries.tsv",
        ],
        "supervision_type": "kb_assertion_rows",
        "annotation_level": "kb_row_no_spans",
        "oncology_specificity": "very_high_clinical_genomics",
        "training_role": "T3a_weak_kb",
        "note": "CIViC nightly export — 5 assertion types. No token spans.",
    },
    "civicmine": {
        "path": RAW / "civicmine",
        "expected_files": ["civicmine_sentences.tsv", "civicmine_collated.tsv"],
        "supervision_type": "mined_sentence_weak",
        "annotation_level": "sentence_level_weak",
        "oncology_specificity": "high_biomarker_mining",
        "training_role": "T3b_weak_mined",
        "note": "CIViCmine Zenodo TSVs — 380K sentence rows. Mining noise present.",
    },
    "cancermine": {
        "path": RAW / "cancermine",
        "expected_files": ["cancermine_collated.tsv"],
        "supervision_type": "aggregated_kb_priors",
        "annotation_level": "aggregated_gene_cancer_roles",
        "oncology_specificity": "high_topic_gene_cancer",
        "training_role": "T3c_weak_aggregate",
        "note": "CancerMine — gene-cancer role aggregates. No spans, not instance-level.",
    },
    "oncology_lung_pubmed": {
        "path": RAW / "oncology_lung_pubmed_hf",
        "expected_files": ["train.jsonl"],
        "supervision_type": "unlabeled_text",
        "annotation_level": "unlabeled_abstract",
        "oncology_specificity": "high_topic_lung_oncology",
        "training_role": "T4_unlabeled_DA",
        "note": "Lung cancer PubMed abstracts — unlabeled; for domain adaptation only.",
    },
    "bronco": {
        "path": RAW / "bronco",
        "expected_files": [
            "BRONCO_20151221/BRONCO_FullText_Tabbed_20150602.txt",
            "BRONCO_20151221/BRONCO_MAPPED_final_20151118.txt",
        ],
        "supervision_type": "full_text_entity_mapped",
        "annotation_level": "full_text_tabbed_mapped",
        "oncology_specificity": "high_english_oncology",
        "training_role": "T5_engineering_blocked",
        "note": "English BRONCO 2015 — 108 PMC articles on disk; NO SPAN READER implemented.",
    },
}


def audit_corpus(name: str, spec: dict) -> dict:
    path: Path = spec["path"]
    record = {
        "corpus_id": name,
        "supervision_type": spec["supervision_type"],
        "annotation_level": spec["annotation_level"],
        "oncology_specificity": spec["oncology_specificity"],
        "training_role": spec["training_role"],
        "note": spec["note"],
        "directory_exists": path.exists(),
        "directory_size_mb": 0.0,
        "expected_files_found": [],
        "expected_files_missing": [],
        "status": "missing",
    }
    if not path.exists():
        record["expected_files_missing"] = spec["expected_files"]
        return record

    record["directory_size_mb"] = round(
        sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 1
    )
    for fname in spec["expected_files"]:
        fpath = path / fname
        if fpath.exists():
            record["expected_files_found"].append(fname)
        else:
            record["expected_files_missing"].append(fname)

    record["status"] = "ok" if not record["expected_files_missing"] else "partial"
    return record


def run() -> None:
    ensure_dirs()
    results = {}
    for name, spec in CORPUS_SPECS.items():
        results[name] = audit_corpus(name, spec)

    # Save JSON
    out_json = REPORTS / "raw_data_availability.json"
    out_json.write_text(json.dumps(results, indent=2))

    # Save CSV
    fieldnames = [
        "corpus_id", "status", "directory_exists", "directory_size_mb",
        "supervision_type", "annotation_level", "oncology_specificity",
        "training_role", "n_expected", "n_found", "n_missing", "note",
    ]
    rows = []
    for name, rec in results.items():
        rows.append({
            "corpus_id": name,
            "status": rec["status"],
            "directory_exists": rec["directory_exists"],
            "directory_size_mb": rec["directory_size_mb"],
            "supervision_type": rec["supervision_type"],
            "annotation_level": rec["annotation_level"],
            "oncology_specificity": rec["oncology_specificity"],
            "training_role": rec["training_role"],
            "n_expected": len(rec["expected_files_found"]) + len(rec["expected_files_missing"]),
            "n_found": len(rec["expected_files_found"]),
            "n_missing": len(rec["expected_files_missing"]),
            "note": rec["note"],
        })

    csv_out = TABLES / "raw_data_summary.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Print summary
    print("=== Raw Data Availability Audit ===\n")
    for name, rec in results.items():
        icon = "✓" if rec["status"] == "ok" else ("⚠" if rec["status"] == "partial" else "✗")
        sz = f"{rec['directory_size_mb']}MB"
        print(f"  {icon} {name:<25} [{rec['status']:<10}] {sz:>8}  {rec['training_role']}")
        if rec["expected_files_missing"]:
            for mf in rec["expected_files_missing"]:
                print(f"    MISSING: {mf}")
    print(f"\nOutputs: {out_json.name}, {csv_out.name}")
