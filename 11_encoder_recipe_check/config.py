"""Paths and grid constants for encoder recipe check."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _paths import OUTPUT_ROOT, step_dirs

STEP = "11_encoder_recipe_check"
_D = step_dirs(STEP)
DATA_DIR = _D["data"]
OUTPUT_DIR = _D["outputs"]
FIGURE_DIR = _D["figures"]
REPORT_DIR = _D["reports"]
RUNS_DIR = _D["runs"]

ROUND10_OUTPUTS = OUTPUT_ROOT / "outputs" / "10_round1_benchmark_kb"
ROUND10_DATA = OUTPUT_ROOT / "data" / "10_round1_benchmark_kb"
ROUND10_DEGENERATE_CSV = ROUND10_OUTPUTS / "10_degenerate_runs.csv"
ROUND10_ENCODER_SUMMARY = ROUND10_OUTPUTS / "10_encoder_summary.csv"
ROUND10_DEBERTA_COLLAPSED_META = (
    ROUND10_DATA / "checkpoints" / "deberta_base" / "seed_45" / "10_train_metadata.json"
)

CHECKPOINT_DIR = DATA_DIR / "checkpoints"
RESULTS_DIR = DATA_DIR / "results"
COMPLETE_MARKER = "recipe_complete.json"

PRIMARY_SEED = 42
GUARD_SEEDS = (43, 44)
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
CHECKPOINT_CRITERION = "val_f1"
WARMUP_FRACTION = 0.1

# Prepared fallback (not run by default)
FALLBACK_LR = 5e-6
FALLBACK_WARMUP_RATIO = 0.1
FALLBACK_SEED = 42
FALLBACK_RUN_KEY = "lr5e-6_warmup_10pct"

DEGENERATE_VAL_F1_MAX = 1e-6
DEGENERATE_BENCHMARK_F1_MAX = 1e-6

GRID = (
    {"lr": 1e-5, "warmup_ratio": 0.0, "warmup_label": "none"},
    {"lr": 2e-5, "warmup_ratio": 0.0, "warmup_label": "none"},
    {"lr": 1e-5, "warmup_ratio": WARMUP_FRACTION, "warmup_label": "warmup_10pct"},
    {"lr": 2e-5, "warmup_ratio": WARMUP_FRACTION, "warmup_label": "warmup_10pct"},
)


def run_key(lr: float, warmup_label: str) -> str:
    lr_s = f"{lr:.0e}".replace("+", "")
    return f"lr{lr_s}_warmup_{warmup_label}"


@dataclass(frozen=True)
class GridPoint:
    lr: float
    warmup_ratio: float
    warmup_label: str
    key_override: str | None = None

    @property
    def key(self) -> str:
        if self.key_override:
            return self.key_override
        return run_key(self.lr, self.warmup_label)


GRID_POINTS: list[GridPoint] = [GridPoint(**g) for g in GRID]

FALLBACK_POINT = GridPoint(
    lr=FALLBACK_LR,
    warmup_ratio=FALLBACK_WARMUP_RATIO,
    warmup_label="warmup_10pct",
    key_override=FALLBACK_RUN_KEY,
)
