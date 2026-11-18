"""Orchestrate validation, the fake or real step, and checkpoint records."""

from __future__ import annotations

from schema_study.evaluation import select_epoch
from schema_study.models.schedule import (
    assert_execution_allowed,
    run_accumulated_epoch,
    validate_training_config,
)


def dry_run_training(config: dict) -> dict:
    errors = validate_training_config(config)
    return {"ok": not errors, "errors": errors, "weights_loaded": False, "executed": False}


def execute_training(config: dict, examples: list[dict], *, requested: bool) -> dict:
    errors = validate_training_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    assert_execution_allowed(requested, "biomedbert training")
    from schema_study.models.torch_backend import run_biomedbert_training

    return run_biomedbert_training(config, examples, config["tokenizer"])


def select_checkpoint(epoch_scores: list[dict]) -> dict:
    """BioRED and the epoch half of DrugProt selection. Earlier epoch wins a tie."""
    return select_epoch(epoch_scores)


def checkpoint_record(name: str, epoch: int, metric: float) -> dict:
    return {
        "name": name,
        "epoch": epoch,
        "metric": metric,
        "weights_written_in_this_round": False,
    }


def run_fake_epoch(n_examples: int, micro_batch_size: int) -> dict:
    """Exercise accumulation with a substitute step. No tensor is updated."""
    seen = []

    def step(do_step: bool, _micro_count: int = 0) -> None:
        seen.append(do_step)

    result = run_accumulated_epoch(n_examples, micro_batch_size, step)
    result["step_flags"] = seen
    result["weights_loaded"] = False
    return result
