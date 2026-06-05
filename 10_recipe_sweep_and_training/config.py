"""Paths and recipe constants for sweep + full-matrix training."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _paths import OUTPUT_ROOT, step_dirs
from shared.constants import GUARD_SEEDS, TRAIN_SEEDS
from shared.models import (
    MODELS,
    MODEL_BY_ID,
    SWEEP_LEARNING_RATES,
    SWEEP_MODEL_IDS,
    SWEEP_SEED,
    SWEEP_WARMUP_SETTINGS,
)
from shared.train_core import RecipeConfig

STEP = "10_recipe_sweep_and_training"
_D = step_dirs(STEP)
DATA_DIR = _D["data"]
OUTPUT_DIR = _D["outputs"]
FIGURE_DIR = _D["figures"]
REPORT_DIR = _D["reports"]
RUNS_DIR = _D["runs"]

# Step 1 sweep CSVs, figures, and decision artifacts
SWEEP_OUTPUT_DIR = OUTPUT_DIR / "sweep"
SWEEP_FIGURE_DIR = FIGURE_DIR / "sweep"
SWEEP_REPORT_PATH = REPORT_DIR / "sweep_report.md"

# Step 1: recipe sweep artifacts
SWEEP_DATA = DATA_DIR / "sweep"
SWEEP_CKPT_DIR = SWEEP_DATA / "checkpoints"
SWEEP_RESULTS_DIR = SWEEP_DATA / "results"
SWEEP_COMPLETE = "sweep_complete.json"

# Step 2: full matrix artifacts
MATRIX_DATA = DATA_DIR / "matrix"
MATRIX_CKPT_DIR = MATRIX_DATA / "checkpoints"
MATRIX_RESULTS_DIR = MATRIX_DATA / "results"
MATRIX_COMPLETE = "matrix_complete.json"

TRAIN_CACHE_DIR = DATA_DIR / "cache"

# =============================================================================
# USER INPUT: set after reviewing the step-1 advisory table (not locked by code)
# Assign a RecipeConfig after step 1, e.g.:
#   CHOSEN_RECIPE = RecipeConfig(lr=1e-5, warmup_ratio=0.0, warmup_label="none")
# Leave as None until then; step 2 will abort if still unset.
# =============================================================================
CHOSEN_RECIPE: RecipeConfig | None = RecipeConfig(lr=1e-5, warmup_ratio=0.0, warmup_label="none")

# Step-2 checkpoint storage (per-epoch dirs are fp16; best/ stays fp32 for folder 11)
SAVE_EPOCH_CHECKPOINTS_FP16 = True
# None = keep all epoch checkpoints; set to an integer N to retain only the N most recent
MAX_EPOCH_CHECKPOINTS_TO_KEEP: int | None = None

# Rough MiB per encoder checkpoint (weights + tokenizer); used for pre-run footprint estimate
ESTIMATED_FP32_CHECKPOINT_MIB = 440
ESTIMATED_FP16_CHECKPOINT_MIB = 220
ESTIMATED_AVG_EPOCHS_PER_RUN = 7


def require_chosen_recipe() -> RecipeConfig:
    if CHOSEN_RECIPE is None:
        raise SystemExit(
            "Set CHOSEN_RECIPE in config.py from the step 1 advisory table before running step 2."
        )
    return CHOSEN_RECIPE


@dataclass(frozen=True)
class SweepPoint:
    model_id: str
    lr: float
    warmup_label: str
    warmup_ratio: float
    seed: int = SWEEP_SEED

    @property
    def run_id(self) -> str:
        lr_s = f"{self.lr:.0e}".replace("+", "").replace("e-0", "e-")
        return f"{self.model_id}_lr{lr_s}_{self.warmup_label}_seed{self.seed}"

    def ckpt_dir(self) -> Path:
        return SWEEP_CKPT_DIR / self.run_id

    def result_path(self) -> Path:
        return SWEEP_RESULTS_DIR / self.run_id / SWEEP_COMPLETE


def all_sweep_points() -> list[SweepPoint]:
    points: list[SweepPoint] = []
    for model_id in SWEEP_MODEL_IDS:
        for lr in SWEEP_LEARNING_RATES:
            for warmup_label, warmup_ratio in SWEEP_WARMUP_SETTINGS:
                points.append(
                    SweepPoint(
                        model_id=model_id,
                        lr=lr,
                        warmup_label=warmup_label,
                        warmup_ratio=warmup_ratio,
                    )
                )
    return points


def matrix_run_root(model_id: str, seed: int) -> Path:
    return MATRIX_CKPT_DIR / model_id / f"seed_{seed}"


def matrix_result_path(model_id: str, seed: int) -> Path:
    return MATRIX_RESULTS_DIR / model_id / f"seed_{seed}" / MATRIX_COMPLETE


def matrix_best_checkpoint(model_id: str, seed: int) -> Path:
    return matrix_run_root(model_id, seed) / "best"
