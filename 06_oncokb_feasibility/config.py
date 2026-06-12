"""Paths and constants for step 06."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _paths import step_dirs

STEP = "06_oncokb_feasibility"
_D = step_dirs(STEP)
DATA_DIR = _D["data"]
OUTPUT_DIR = _D["outputs"]
REPORT_DIR = _D["reports"]
RUNS_DIR = _D["runs"]

ONCOKB_BASE_URL = "https://www.oncokb.org/api/v1"
ONCOKB_PRODUCTION_HOST = "https://www.oncokb.org"
REQUEST_TIMEOUT = 120
BATCH_SIZE = 50
REQUEST_PAUSE = 0.25

TRAINING_PMIDS_JSON = (
    REPO_ROOT.parent / "projects" / "project_1" / "outputs" / "01_corpus_relevance" / "training_pmids_clean.json"
)

# Optional user-supplied cancer gene list (TSV from oncokb.org/cancerGenes download).
WORKSPACE_CANCER_GENE_LIST = DATA_DIR / "cancer_gene_list.tsv"

INFO_JSON = DATA_DIR / "api_info.json"
CANCER_GENE_LIST_JSON = DATA_DIR / "cancer_gene_list.json"
FETCH_METADATA_JSON = DATA_DIR / "fetch_metadata.json"
ANNOTATIONS_JSONL = DATA_DIR / "annotation_responses.jsonl"
QUERY_MANIFEST_JSON = DATA_DIR / "query_manifest.json"
ASSOCIATIONS_CSV = OUTPUT_DIR / "associations_inventory.csv"
GROUNDING_SUMMARY_CSV = OUTPUT_DIR / "grounding_summary.csv"
STRUCTURAL_EVIDENCE_CSV = OUTPUT_DIR / "structural_pmid_evidence.csv"
GROUNDABLE_TRIPLES_CSV = OUTPUT_DIR / "groundable_triples.csv"
EVALUABLE_TRIPLES_CSV = OUTPUT_DIR / "evaluable_triples.csv"
PMID_RETRIEVABILITY_CSV = OUTPUT_DIR / "pmid_retrievability.csv"

# Umbrella / auto terms for byProteinChange (official API docs).
UMBRELLA_ALTERATIONS = (
    "Oncogenic Mutations",
    "Truncating Mutations",
    "Kinase Domain Duplication",
    "ITD",
    "Amplification",
    "Deletion",
    "Fusions",
)

CNA_TYPES = ("AMPLIFICATION", "DELETION")

ATYPICAL_NO_GENE = ("MSI-H", "TMB-H")

# Verdict thresholds for evaluable single-PMID triples after training-PMID exclusion.
MIN_EVALUABLE_GENE_DRUG = 80
MIN_EVALUABLE_GENE_DISEASE = 50
