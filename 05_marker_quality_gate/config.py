"""Paths for step 05 marker quality gate."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _paths import OUTPUT_ROOT, step_dirs

STEP = "05_marker_quality_gate"
_D = step_dirs(STEP)
DATA_DIR = _D["data"]
OUTPUT_DIR = _D["outputs"]
FIGURE_DIR = _D["figures"]
REPORT_DIR = _D["reports"]
RUNS_DIR = _D["runs"]

TRAIN_CACHE_DIR = DATA_DIR / "train_cache"
QUALITY_RESULTS_JSON = OUTPUT_DIR / "quality_gate_results.json"
QUALITY_CHECKS_CSV = OUTPUT_DIR / "quality_gate_checks.csv"

# Canonical cache used by folder 10 (rebuilt here with force=True)
FOLDER10_TRAIN_CACHE = OUTPUT_ROOT / "data" / "10_recipe_sweep_and_training" / "cache"

# Reference numbers from pre-fix diagnostic (string-match pipeline)
BASELINE_TRAIN_HEAD_MARKER_MISMATCH = 0.593
BASELINE_TRAIN_SAME_SENTENCE_STRING = 0.266
BASELINE_TRAIN_SAME_SENTENCE_NATIVE = 0.393
BASELINE_CIVIC_POS_EASY = 0.512
BASELINE_CIVIC_POS_HARD = 0.449

SAME_SENT_TOLERANCE = 0.02
