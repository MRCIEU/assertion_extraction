"""Paths and constants for Round 1 (benchmark vs KB)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _paths import OUTPUT_ROOT, step_dirs

STEP = "10_round1_benchmark_kb"
_D = step_dirs(STEP)
DATA_DIR = _D["data"]
OUTPUT_DIR = _D["outputs"]
FIGURE_DIR = _D["figures"]
REPORT_DIR = _D["reports"]
RUNS_DIR = _D["runs"]

STEP01_OUTPUTS = OUTPUT_ROOT / "outputs" / "01_corpus_relevance"
STEP03_OUTPUTS = OUTPUT_ROOT / "outputs" / "03_candidate_pool"
STEP00_DATA = OUTPUT_ROOT / "data" / "00_civic_feasibility"

TRAINING_PMIDS_CLEAN_JSON = STEP01_OUTPUTS / "training_pmids_clean.json"
EXCLUDED_PMIDS_JSON = STEP01_OUTPUTS / "excluded_pmids.json"
FROZEN_POOL_JSON = STEP03_OUTPUTS / "frozen_pool.json"
EVIDENCE_JSON = STEP00_DATA / "evidence_items.json"
STEP03_RANKING_BASELINES_CSV = STEP03_OUTPUTS / "ranking_baselines.csv"

# Blocking leak check — must never appear in training data
LEAKED_PMIDS = frozenset({"16434489", "18794803", "23430109"})

CHECKPOINT_DIR = DATA_DIR / "checkpoints"
SCORES_DIR = DATA_DIR / "scores"
RESULTS_DIR = DATA_DIR / "model_results"
TRAIN_CACHE_TRAIN = DATA_DIR / "train_examples_train.jsonl"
TRAIN_CACHE_VAL = DATA_DIR / "train_examples_val.jsonl"

SAMPLING_SEED = 42
TRAIN_SEEDS = list(range(42, 50))  # 8 seeds
PRIMARY_SCOPE = "primary"
RECALL_K_VALUES = (1, 3, 5)
ECE_N_BINS = 10
POSITIVE_FRACTION_PRIOR = 0.148
PAIR_TYPES = ("gene-drug", "gene-disease")

MAX_TRAIN_EXAMPLES = 24_000
NEGATIVES_PER_POSITIVE = 2
TRAIN_BATCH_SIZE = 16
TRAIN_LR = 2e-5
TRAIN_WARMUP_RATIO = 0.06
MAX_SEQ_LENGTH = 256
INFER_BATCH_SIZE = 32
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
TRAIN_PAIR_TYPES = {"gene-drug", "gene-disease"}

COMPLETE_MARKER = "round1_complete.json"
BOOTSTRAP_N = 5000


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    short_name: str
    hf_name: str
    architecture: str


MODELS: list[ModelSpec] = [
    ModelSpec(
        "pubmedbert_base",
        "PubMedBERT-base",
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "domain BERT",
    ),
    ModelSpec(
        "biomedbert_base",
        "BioMedBERT-base",
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "domain BERT",
    ),
    ModelSpec(
        "biolinkbert_base",
        "BioLinkBERT-base",
        "michiyasunaga/BioLinkBERT-base",
        "domain BERT",
    ),
    ModelSpec(
        "biobert_base",
        "BioBERT-base",
        "dmis-lab/biobert-base-cased-v1.2",
        "domain BERT",
    ),
    ModelSpec(
        "scibert_base",
        "SciBERT",
        "allenai/scibert_scivocab_uncased",
        "domain BERT",
    ),
    ModelSpec(
        "roberta_base",
        "RoBERTa-base",
        "roberta-base",
        "general RoBERTa",
    ),
    ModelSpec(
        "bert_base",
        "BERT-base",
        "bert-base-uncased",
        "general BERT",
    ),
    ModelSpec(
        "distilbert_base",
        "DistilBERT-base",
        "distilbert-base-uncased",
        "lightweight",
    ),
    ModelSpec(
        "deberta_base",
        "DeBERTa-base",
        "microsoft/deberta-base",
        "general DeBERTa",
    ),
]

MODEL_BY_ID = {m.model_id: m for m in MODELS}
