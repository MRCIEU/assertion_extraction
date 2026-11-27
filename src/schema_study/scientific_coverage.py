"""Canonical scientific coverage. A controller state name is not a completed experiment."""

from __future__ import annotations

from pathlib import Path

VALID_TRAINING = {"training_completed", "reused_historical_equivalent"}
VALID_INFERENCE = {"inference_completed"}


def completion_allowed(cells: list[dict]) -> bool:
    """Every required cell needs its own artifact. Supersession without a replacement is not completion."""
    if not cells:
        return False
    for cell in cells:
        if cell.get("superseded_without_replacement"):
            return False
        if not cell.get("artifact_present"):
            return False
        if cell.get("status") not in VALID_TRAINING | VALID_INFERENCE:
            return False
    return True


def canonical_groups() -> dict[str, list[dict]]:
    """The approved matrix. Operational task rows are not these denominators."""
    from schema_study.production_plan import (
        _inference_tasks,
        _supplementary_tasks,
        corrected_principal_tasks,
        corrected_prompt_tasks,
        principal_training_tasks,
    )

    training = corrected_principal_tasks()
    drugprot_training = [task for task in principal_training_tasks() if task.get("dataset") == "drugprot"]
    prompts = corrected_prompt_tasks() + [task for task in _inference_tasks() if task.get("dataset") == "drugprot"]
    evaluations = [
        task for task in _supplementary_tasks(training)
        if task["family"] in {"adapter_inference", "final_evaluation"}
    ] + [
        task for task in _supplementary_tasks(drugprot_training)
        if task["family"] in {"adapter_inference", "final_evaluation"} and task.get("dataset") == "drugprot"
    ]
    return {
        "training": training + drugprot_training,
        "prompts": prompts,
        "evaluations": evaluations,
    }


def canonical_counts() -> dict[str, int]:
    groups = canonical_groups()
    drugprot_training = sum(1 for task in groups["training"] if task.get("dataset") == "drugprot")
    return {
        "principal_training": len(groups["training"]),
        "principal_prompts": len(groups["prompts"]),
        "principal_evaluations": len(groups["evaluations"]),
        "biored_training": len(groups["training"]) - drugprot_training,
        "drugprot_training": drugprot_training,
    }


def _artifact_present(task: dict, entry: dict) -> bool:
    if entry.get("scientific") == "reused_historical_equivalent" and entry.get("artifact"):
        return Path(entry["artifact"]).exists()
    output = Path(task.get("output") or "")
    if task.get("family") == "principal_training":
        return (output / "adapter_checkpoint.pt").is_file() or (output / "checkpoints" / "best" / "checkpoint.pt").is_file()
    return (output / "scientific_status.json").is_file() and (output / "predictions.jsonl").is_file()


def coverage_report(tasks: list[dict], state: dict) -> dict:
    """Count canonical cells. passed is false while any required artifact is missing or invalid."""
    by_id = {task["task_id"]: task for task in tasks}
    groups = canonical_groups()
    cells = []
    for task in groups["training"] + groups["prompts"] + groups["evaluations"]:
        annotated = by_id.get(task["task_id"], task)
        entry = state.get(task["task_id"], {})
        status = entry.get("scientific") or entry.get("state") or "unsubmitted"
        cells.append({
            "task_id": task["task_id"],
            "family": task["family"],
            "status": status,
            "artifact_present": _artifact_present(annotated, entry),
            "superseded_without_replacement": status == "superseded_stored_representation",
        })
    running = sum(1 for cell in cells if cell["status"] in {"running", "pending", "submitted"})
    blocked = sum(1 for cell in cells if cell["status"] == "terminal_failure")
    expected = canonical_counts()
    required = expected["principal_training"] + expected["principal_prompts"] + expected["principal_evaluations"]
    return {
        "passed": len(cells) == required and completion_allowed(cells),
        "cells": len(cells),
        "running": running,
        "blocked": blocked,
        "expected": expected,
    }
