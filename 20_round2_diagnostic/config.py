"""Paths for Round 2 diagnostic (reads Round 1 artifacts; writes under step 20)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _paths import OUTPUT_ROOT, step_dirs

STEP = "20_round2_diagnostic"
_D = step_dirs(STEP)
OUTPUT_DIR = _D["outputs"]
FIGURE_DIR = _D["figures"]
REPORT_DIR = _D["reports"]
DATA_DIR = _D["data"]

# Read-only Round 1 locations (do not write here)
R1_STEP = "10_round1_benchmark_kb"
_R1 = step_dirs(R1_STEP)
R1_CHECKPOINTS = _R1["data"] / "checkpoints"
R1_SCORES = _R1["data"] / "scores"
R1_OUTPUTS = _R1["outputs"]
R1_PER_RUN_CSV = R1_OUTPUTS / "10_per_run_scores.csv"
R1_EASY_HARD_CSV = R1_OUTPUTS / "10_easy_hard_ranking.csv"
R1_SWEEP_DATA = _R1["data"] / "sweep"
R1_SWEEP_RESULTS = R1_SWEEP_DATA / "results"
R1_SWEEP_CKPT_LOSS = R1_SWEEP_DATA / "checkpoints"
R1_SWEEP_CKPT_F1 = R1_SWEEP_DATA / "checkpoints_by_val_f1"

from importlib import import_module

_r1cfg = import_module("10_round1_benchmark_kb.config")
MODELS = _r1cfg.MODELS
MODEL_BY_ID = _r1cfg.MODEL_BY_ID
TRAIN_SEEDS = _r1cfg.TRAIN_SEEDS
TRAINING_STRATEGY = _r1cfg.TRAINING_STRATEGY
COLLAPSED_DEBERTA_SEEDS = frozenset({45, 49})

FOCUS_MODEL_IDS = ("pubmedbert_base", "roberta_base", "distilbert_base")
ROUND1_RECIPE_LR = 2e-5
ROUND1_RECIPE_WARMUP_LABEL = "none"

DPI = 300
PALETTE = {
    "neutral": "#4477AA",
    "accent": "#CC6677",
    "grid": "#DDDDDD",
    "text": "#222222",
}
