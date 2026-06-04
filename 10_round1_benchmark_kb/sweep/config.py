"""Sweep paths, grid, and model subset."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _paths import OUTPUT_ROOT, step_dirs

STEP = "10_round1_benchmark_kb"
_D = step_dirs(STEP)
DATA_DIR = _D["data"]
OUTPUT_DIR = _D["outputs"] / "sweep"
FIGURE_DIR = _D["figures"] / "sweep"
REPORT_DIR = _D["reports"]
RUNS_DIR = _D["runs"] / "sweep"

SWEEP_DATA_DIR = DATA_DIR / "sweep"
SWEEP_CKPT_DIR = SWEEP_DATA_DIR / "checkpoints"
SWEEP_CKPT_F1_DIR = SWEEP_DATA_DIR / "checkpoints_by_val_f1"
SWEEP_RESULTS_DIR = SWEEP_DATA_DIR / "results"

from importlib import import_module

_r1 = import_module("10_round1_benchmark_kb.config")
MODEL_BY_ID = _r1.MODEL_BY_ID
TRAIN_CACHE_TRAIN = _r1.TRAIN_CACHE_TRAIN
TRAIN_CACHE_VAL = _r1.TRAIN_CACHE_VAL
MAX_EPOCHS = _r1.MAX_EPOCHS
EARLY_STOPPING_PATIENCE = _r1.EARLY_STOPPING_PATIENCE
MAX_SEQ_LENGTH = _r1.MAX_SEQ_LENGTH
TRAIN_BATCH_SIZE = _r1.TRAIN_BATCH_SIZE

SWEEP_SEED = 42
LEARNING_RATES = (5e-6, 1e-5, 2e-5, 3e-5)
WARMUP_SETTINGS: tuple[tuple[str, float], ...] = (
    ("none", 0.0),
    ("linear_10pct", 0.10),
)

SWEEP_MODEL_IDS = ("pubmedbert_base", "roberta_base", "distilbert_base")

COMPLETE_MARKER = "sweep_complete.json"


@dataclass(frozen=True)
class SweepRun:
    model_id: str
    lr: float
    warmup_label: str
    warmup_ratio: float
    seed: int = SWEEP_SEED

    @property
    def run_id(self) -> str:
        lr_s = f"{self.lr:.0e}".replace("+", "").replace("e-0", "e-")
        return f"{self.model_id}_lr{lr_s}_{self.warmup_label}_seed{self.seed}"

    def result_path(self) -> Path:
        return SWEEP_RESULTS_DIR / self.run_id / COMPLETE_MARKER


def all_runs() -> list[SweepRun]:
    runs: list[SweepRun] = []
    for model_id in SWEEP_MODEL_IDS:
        for lr in LEARNING_RATES:
            for warmup_label, warmup_ratio in WARMUP_SETTINGS:
                runs.append(
                    SweepRun(
                        model_id=model_id,
                        lr=lr,
                        warmup_label=warmup_label,
                        warmup_ratio=warmup_ratio,
                    )
                )
    return runs


def runs_for_model(model_id: str) -> list[SweepRun]:
    return [r for r in all_runs() if r.model_id == model_id]
