"""Paths for Round 1 analysis (consumes folder-10 step-2 matrix outputs)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _paths import OUTPUT_ROOT, step_dirs
from shared.constants import BOOTSTRAP_N, TRAIN_SEEDS
from shared.models import MODELS, MODEL_BY_ID

STEP = "11_round1_analysis"
_D = step_dirs(STEP)
DATA_DIR = _D["data"]
OUTPUT_DIR = _D["outputs"]
FIGURE_DIR = _D["figures"]
REPORT_DIR = _D["reports"]
RUNS_DIR = _D["runs"]

# Producer: step-2 full matrix (folder 10)
TRAIN_STEP = "10_recipe_sweep_and_training"
_T10 = step_dirs(TRAIN_STEP)
MATRIX_DATA = _T10["data"] / "matrix"
MATRIX_CKPT_DIR = MATRIX_DATA / "checkpoints"
MATRIX_RESULTS_DIR = MATRIX_DATA / "results"
MATRIX_COMPLETE = "matrix_complete.json"

SCORES_DIR = DATA_DIR / "scores"
PER_RUN_CSV = OUTPUT_DIR / "11_per_run_scores.csv"

# Re-export for analysis modules
PAIR_TYPES = ("gene-drug", "gene-disease")
