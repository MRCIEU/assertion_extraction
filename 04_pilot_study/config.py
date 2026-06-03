"""Paths and constants for step 04 pilot study."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _paths import step_dirs, OUTPUT_ROOT

STEP = "04_pilot_study"
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
CIVIC_EVIDENCE_JSON = EVIDENCE_JSON
STEP03_RANKING_BASELINES_CSV = STEP03_OUTPUTS / "ranking_baselines.csv"
STEP03_RANKING_VERIFICATION_JSON = STEP03_OUTPUTS / "ranking_verification.json"

SCORES_DIR = DATA_DIR / "model_scores"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
TRAIN_CACHE_JSON = DATA_DIR / "train_examples.jsonl"

SAMPLING_SEED = 42
TRAIN_SEEDS = [42, 43, 44]
PRIMARY_SCOPE = "primary"
RECALL_K_VALUES = (1, 3, 5)
ECE_N_BINS = 10
POSITIVE_FRACTION_PRIOR = 0.148

MAX_TRAIN_EXAMPLES = 24_000
NEGATIVES_PER_POSITIVE = 2
TRAIN_BATCH_SIZE = 16
TRAIN_MAX_STEPS = 3_000
TRAIN_LR = 2e-5
TRAIN_WARMUP_RATIO = 0.06
MAX_SEQ_LENGTH = 256
INFER_BATCH_SIZE = 32

TRAIN_PAIR_TYPES = {"gene-drug", "gene-disease"}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    short_name: str
    hf_name: str
    benchmark_name: str
    benchmark_f1: float
    benchmark_source: str


MODELS: list[ModelSpec] = [
    ModelSpec(
        model_id="pubmedbert_base",
        short_name="PubMedBERT-base",
        hf_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        benchmark_name="BioRED document-level RE F1 (PubMedBERT family)",
        benchmark_f1=0.893,
        benchmark_source="BioRED paper / PubMedBERT baseline (~89.3% micro-F1 NER+RE suite)",
    ),
    ModelSpec(
        model_id="biolinkbert_base",
        short_name="BioLinkBERT-base",
        hf_name="michiyasunaga/BioLinkBERT-base",
        benchmark_name="BLURB biomedical benchmark avg.",
        benchmark_f1=0.840,
        benchmark_source="BioLinkBERT paper BLURB avg. (~84.0)",
    ),
    ModelSpec(
        model_id="roberta_base",
        short_name="RoBERTa-base",
        hf_name="roberta-base",
        benchmark_name="General-domain GLUE avg. (weak biomedical reference)",
        benchmark_f1=0.880,
        benchmark_source="RoBERTa-base GLUE (~88.0); not biomedical-specialised",
    ),
]

MODEL_BY_ID = {m.model_id: m for m in MODELS}
