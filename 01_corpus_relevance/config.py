"""Paths and constants for step 01."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _paths import step_dirs, OUTPUT_ROOT

STEP = "01_corpus_relevance"
_D = step_dirs(STEP)
DATA_DIR = _D["data"]
OUTPUT_DIR = _D["outputs"]
FIGURE_DIR = _D["figures"]
REPORT_DIR = _D["reports"]
RUNS_DIR = _D["runs"]

STEP00_DATA = OUTPUT_ROOT / "data" / "00_civic_feasibility"
STEP00_OUTPUTS = OUTPUT_ROOT / "outputs" / "00_civic_feasibility"
ENTITY_PAIR_BREAKDOWN_CSV = STEP00_OUTPUTS / "entity_pair_breakdown.csv"
EVALUABLE_INVENTORY_CSV = STEP00_OUTPUTS / "evaluable_inventory.csv"
CIVIC_INVENTORY_FILE = EVALUABLE_INVENTORY_CSV
CIVIC_PAIR_WEIGHTS_FILE = ENTITY_PAIR_BREAKDOWN_CSV

INVENTORY_FILE = DATA_DIR / "corpus_inventories.json"
TRAIN_STATS_FILE = DATA_DIR / "corpus_train_stats.json"

ONCOLOGY_FRACTIONS_CSV = OUTPUT_DIR / "oncology_fractions_by_criterion.csv"
ONCOLOGY_AGREEMENT_CSV = OUTPUT_DIR / "oncology_criteria_agreement.csv"
ONCOLOGY_METADATA_JSON = OUTPUT_DIR / "oncology_subset_metadata.json"
ONCOLOGY_PMID_MESH_JSON = DATA_DIR / "oncology" / "pmid_mesh_index.json"
EXCLUDED_PMIDS_JSON = OUTPUT_DIR / "excluded_pmids.json"
TRAINING_PMIDS_CLEAN_JSON = OUTPUT_DIR / "training_pmids_clean.json"
PMID_OVERLAP_CSV = OUTPUT_DIR / "pmid_overlap.csv"
PMID_CONFLICTS_CSV = OUTPUT_DIR / "pmid_conflicts.csv"
PMID_CONFLICT_EXAMPLES_CSV = OUTPUT_DIR / "pmid_conflict_examples.csv"
PMID_LEAKAGE_CSV = OUTPUT_DIR / "pmid_leakage.csv"

CIVIC_PAIR_TYPES = [
    "gene-drug",
    "gene-disease",
    "variant-disease",
    "variant-drug",
]

CIVIC_PAIR_SHARES = {
    "gene-drug": 0.5348,
    "gene-disease": 0.3064,
    "variant-disease": 0.0899,
    "variant-drug": 0.0689,
}

CORPORA = {
    "biored": {
        "hf_id": "bigbio/biored",
        "config": "biored_bigbio_kb",
        "display_name": "BioRED",
        "language": "English",
        "role": "primary",
        "train_splits": ["train", "validation"],
        "description": "Multi-type biomedical RE (PubMed abstracts).",
    },
    "drugprot": {
        "hf_id": "bigbio/drugprot",
        "config": "drugprot_bigbio_kb",
        "display_name": "DrugProt",
        "language": "English",
        "role": "supplement",
        "train_splits": ["train", "validation"],
        "description": "Drug/chemical–gene/protein relations (PubMed abstracts).",
    },
    "bc5cdr": {
        "hf_id": "bigbio/bc5cdr",
        "config": "bc5cdr_bigbio_kb",
        "display_name": "BC5CDR",
        "language": "English",
        "role": "reference",
        "train_splits": ["train"],
        "description": "Chemical–disease relations (PubMed articles).",
    },
}

TRAIN_VOLUME_THRESHOLD = 500
GRANULARITY_LEVELS = [
    "1_coarse_association",
    "2_directional_correlation",
    "3_fine_mechanism",
    "4_clinical_significance",
]
