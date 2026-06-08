"""Paths for Round 2 diagnostic (reads folder-10 per-epoch checkpoints)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _paths import OUTPUT_ROOT, step_dirs
from shared.constants import TRAIN_SEEDS
from shared.models import MODELS, MODEL_BY_ID

STEP = "20_round2_diagnostic"
_D = step_dirs(STEP)
OUTPUT_DIR = _D["outputs"]
FIGURE_DIR = _D["figures"]
REPORT_DIR = _D["reports"]
DATA_DIR = _D["data"]
ENRICHED_POOL_CACHE = DATA_DIR / "enriched_primary_pool.parquet"
EPOCH_KB_CACHE = DATA_DIR / "epoch_kb_trajectory.csv"

# Producer: folder 10 step-2 matrix
TRAIN_STEP = "10_recipe_sweep_and_training"
_T10 = step_dirs(TRAIN_STEP)
MATRIX_CKPT_DIR = _T10["data"] / "matrix" / "checkpoints"
MATRIX_RESULTS_DIR = _T10["data"] / "matrix" / "results"

# Round 1 analysis outputs (best-point KB for comparison)
R1_STEP = "11_round1_analysis"
_R11 = step_dirs(R1_STEP)
R11_OUTPUTS = _R11["outputs"]
R11_PER_RUN_CSV = R11_OUTPUTS / "11_per_run_scores.csv"
R11_EASY_HARD_CSV = R11_OUTPUTS / "11_easy_hard_ranking.csv"
R11_VARIANCE_CSV = R11_OUTPUTS / "11_variance_components.csv"
EPOCH_SCORE_COMPLETE = DATA_DIR / "epoch_scoring_complete.json"

FOCUS_MODEL_IDS = ("pubmedbert_base", "roberta_base", "distilbert_base")

DPI = 300
PALETTE = {
    "neutral": "#4477AA",
    "accent": "#CC6677",
    "grid": "#DDDDDD",
    "text": "#222222",
}
