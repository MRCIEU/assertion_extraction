"""Independent best and last checkpoints.

Saved parameters are deep copies. Later changes to the live parameters do not
alter a checkpoint that has already been written.
"""

from __future__ import annotations

import copy
from typing import Mapping

from schema_study.evaluation import select_common_threshold, select_epoch


REQUIRED_RESUME_FIELDS = (
    "model_parameters",
    "optimizer_state",
    "scheduler_state",
    "completed_epochs",
    "optimizer_updates",
    "best_score",
    "best_epoch",
    "threshold",
    "rng_state",
    "data_order_seed",
    "amp_state",
    "training_config",
    "data_version",
)


def freeze(value):
    """Return an independent copy. The result does not share nested containers."""
    return copy.deepcopy(value)


class CheckpointBook:
    """Track best and last states for one training run."""

    def __init__(self) -> None:
        self.best: dict | None = None
        self.last: dict | None = None
        self.epoch_scores: list[dict] = []

    def observe(
        self,
        *,
        epoch: int,
        f1: float,
        parameters,
        optimizer_updates: int,
        scheduler_state,
        optimizer_state,
        rng_state,
        threshold: float | None = None,
        data_order_seed: int | None = None,
        amp_state=None,
        training_config: Mapping | None = None,
        data_version: str = "",
    ) -> dict:
        self.epoch_scores.append({"epoch": epoch, "f1": f1, "threshold": threshold})
        chosen = select_epoch(self.epoch_scores)
        state = {
            "model_parameters": freeze(parameters),
            "optimizer_state": freeze(optimizer_state),
            "scheduler_state": freeze(scheduler_state),
            "completed_epochs": epoch,
            "optimizer_updates": optimizer_updates,
            "best_score": chosen["f1"],
            "best_epoch": chosen["epoch"],
            "threshold": threshold,
            "best_threshold": chosen.get("threshold"),
            "rng_state": freeze(rng_state),
            "data_order_seed": data_order_seed,
            "amp_state": freeze(amp_state),
            "training_config": freeze(dict(training_config or {})),
            "data_version": data_version,
        }
        missing = [name for name in REQUIRED_RESUME_FIELDS if name not in state]
        if missing:
            raise ValueError(f"Checkpoint state is incomplete: {missing}")
        self.last = state
        if chosen["epoch"] == epoch:
            self.best = freeze(state)
        return {"best_epoch": self.best["completed_epochs"], "last_epoch": self.last["completed_epochs"]}

    def restore_last(self) -> dict:
        if self.last is None:
            raise ValueError("No last checkpoint")
        return freeze(self.last)


def threshold_for_epoch(gold_sets, probabilities, labels) -> dict:
    """DrugProt selection. An equal F1 keeps the higher threshold."""
    return select_common_threshold(gold_sets, probabilities, labels)
