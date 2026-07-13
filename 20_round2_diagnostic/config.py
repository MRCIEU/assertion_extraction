"""Paths for Round 2 training-dynamics diagnostic (folder-10 per-epoch checkpoints)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _paths import step_dirs
from shared.constants import RECALL_K_VALUES, TRAIN_SEEDS
from shared.models import MODELS, MODEL_BY_ID

STEP = "20_round2_diagnostic"
_D = step_dirs(STEP)
OUTPUT_DIR = _D["outputs"]
FIGURE_DIR = _D["figures"]
REPORT_DIR = _D["reports"]
DATA_DIR = _D["data"]
RUNS_DIR = _D["runs"]

ENRICHED_POOL_CACHE = DATA_DIR / "enriched_primary_pool.parquet"
SCORES_DIR = DATA_DIR / "scores"
SCORING_COMPLETE = DATA_DIR / "epoch_scoring_complete.json"

# Legacy cache (superseded by per-epoch JSON under SCORES_DIR)
EPOCH_KB_CACHE = DATA_DIR / "epoch_kb_trajectory.csv"

TRAIN_STEP = "10_recipe_sweep_and_training"
_T10 = step_dirs(TRAIN_STEP)
MATRIX_CKPT_DIR = _T10["data"] / "matrix" / "checkpoints"
MATRIX_RESULTS_DIR = _T10["data"] / "matrix" / "results"

# Legacy scripts may still pass biomedbert_base; on-disk matrix folders use bluebert_base.
_CHECKPOINT_MODEL_ALIASES = {"biomedbert_base": "bluebert_base"}


def resolve_checkpoint_model_id(model_id: str) -> str:
    return _CHECKPOINT_MODEL_ALIASES.get(model_id, model_id)

R11_STEP = "11_round1_analysis"
_R11 = step_dirs(R11_STEP)
R11_VARIANCE_CSV = _R11["outputs"] / "11_variance_components.csv"

EXPECTED_RECIPE_LR = 5e-6
SCORING_MODEL_IDS = tuple(m.model_id for m in MODELS)

TRAJECTORY_CSV = OUTPUT_DIR / "20_epoch_trajectory.csv"
PAIRED_CHANGES_CSV = OUTPUT_DIR / "20_within_seed_paired_changes.csv"
SEED_DISTRIBUTION_CSV = OUTPUT_DIR / "20_seed_erosion_distribution.csv"
HARD_EASY_CSV = OUTPUT_DIR / "20_hard_easy_breakdown.csv"
PAIR_TYPE_CSV = OUTPUT_DIR / "20_pair_type_breakdown.csv"
ROBUSTNESS_CSV = OUTPUT_DIR / "20_robustness_well_trained.csv"
GENE_DISEASE_SUBSET_CSV = OUTPUT_DIR / "20_gene_disease_subset_breakdown.csv"
GENE_DISEASE_ROBUSTNESS_CSV = OUTPUT_DIR / "20_gene_disease_robustness.csv"
GENE_DISEASE_SEED_CSV = OUTPUT_DIR / "20_gene_disease_seed_distribution.csv"
GENE_DISEASE_ENCODER_CSV = OUTPUT_DIR / "20_gene_disease_encoder_breakdown.csv"
PAIR_TYPE_SUBSET_CSV = OUTPUT_DIR / "20_pair_type_subset_contrast.csv"
INVENTORY_CSV = OUTPUT_DIR / "20_checkpoint_inventory.csv"

POOL_STEP = "03_candidate_pool"
_P03 = step_dirs(POOL_STEP)
POOL_SIZE_BY_ABSTRACT_CSV = _P03["outputs"] / "03_candidate_pool_size_by_abstract.csv"
PUBMED_RECALL_CSV = _P03["outputs"] / "03_candidate_pool_pubtator_recall_classification.csv"

R11_SCORES_DIR = _R11["data"] / "scores"
STRATUM_MRR_CACHE = DATA_DIR / "stratum_mrr_epoch1_cache.jsonl"

TIMING_CLASSIFICATION_CSV = OUTPUT_DIR / "20_kb_peak_timing.csv"
TIMING_SUMMARY_CSV = OUTPUT_DIR / "20_kb_peak_timing_summary.csv"
POOL_STRATUM_CSV = OUTPUT_DIR / "20_gene_disease_pool_stratum_paired.csv"
POOL_STRATUM_SUMMARY_CSV = OUTPUT_DIR / "20_gene_disease_pool_stratum_summary.csv"
ENCODER_CORRELATION_CSV = OUTPUT_DIR / "20_encoder_property_correlation.csv"
QUAL_ERROR_CASES_CSV = OUTPUT_DIR / "20_qualitative_error_cases.csv"
QUAL_ERROR_FLAGGED_CSV = OUTPUT_DIR / "20_qualitative_errors_flagged_manual.csv"
QUAL_ERROR_PATTERNS_CSV = OUTPUT_DIR / "20_qualitative_error_patterns.csv"
QUAL_ERROR_SUMMARY_CSV = OUTPUT_DIR / "20_qualitative_error_summary.csv"

DPI = 300
PALETTE = {
    "neutral": "#4477AA",
    "neutral_light": "#88AADD",
    "accent": "#CC6677",
    "grid": "#DDDDDD",
    "text": "#222222",
}
