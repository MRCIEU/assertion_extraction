"""Phase B trainer package — clean .py rewrite of the .pyc-only Phase A trainer.

See `fine_tuning_experiments/phase_b/trainer_inventory/` for the API contracts
this package implements (scientific_trainer / scientific_data / minimal_trainer).

Public entry point:
  fine_tuning_experiments.phase_b.trainer.run_scientific_training
"""
from __future__ import annotations

from fine_tuning_experiments.phase_b.trainer.scientific_trainer import (
    run_scientific_training,
)

__all__ = ["run_scientific_training"]
