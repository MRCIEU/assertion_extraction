"""Canonical scientific coverage. A controller state name is not a completed experiment."""

from __future__ import annotations

import json
from pathlib import Path

from schema_study.programme_closure import artifact_valid, expected_cells

VALID_TRAINING = {"training_completed", "reused_historical_equivalent"}
VALID_INFERENCE = {"inference_completed"}
CORRECTED = "mesh_ancestor_v1"


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
    cells = expected_cells()
    return {
        "principal_training": len(cells["training"]),
        "principal_prompts": len(cells["prompts"]),
        "principal_evaluations": len(cells["evaluations"]),
        "biored_training": sum(1 for row in cells["training"] if row["dataset"] == "biored"),
        "drugprot_training": sum(1 for row in cells["training"] if row["dataset"] == "drugprot"),
        "biored_evaluations": sum(1 for row in cells["evaluations"] if row["dataset"] == "biored"),
        "bc8_evaluations": sum(1 for row in cells["evaluations"] if row["dataset"] == "bc8"),
        "drugprot_evaluations": sum(1 for row in cells["evaluations"] if row["dataset"] == "drugprot"),
    }


def _status_payload(task: dict) -> dict:
    path = Path(task.get("output") or "") / "scientific_status.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _matches(task: dict, cell: dict, kind: str) -> bool:
    if task.get("model") != cell["model"] or str(task.get("entity")) != cell["entity"] or str(task.get("predicate")) != cell["predicate"]:
        return False
    corrected = cell["dataset"] != "drugprot"
    if corrected and task.get("representation_version") != CORRECTED:
        return False
    if not corrected and task.get("representation_version"):
        return False
    if kind == "training":
        return task.get("family") == "principal_training" and task.get("dataset") == cell["dataset"] and int(task.get("seed")) == cell["seed"]
    if kind == "prompts":
        return task.get("family") == cell["strategy"] and task.get("dataset") == cell["dataset"]
    family = "final_evaluation" if cell["model"] == "biomedbert" else "adapter_inference"
    return task.get("family") == family and task.get("dataset") == cell["dataset"] and int(task.get("seed")) == cell["seed"]


def _population_covers(payload: dict, corpus: str) -> bool:
    if payload.get("predictions_usable") is not True or payload.get("population_complete") is not True:
        return False
    populations = payload.get("populations") or {}
    if populations:
        return int(populations.get(corpus) or 0) > 0
    return payload.get("dataset") == corpus


def _cell_accepted(kind: str, cell: dict, task: dict, entry: dict) -> bool:
    if kind == "training" and entry.get("scientific") == "reused_historical_equivalent":
        return artifact_valid(Path(str(entry.get("artifact") or "")))
    payload = _status_payload(task)
    if kind == "training":
        if entry.get("scientific") != "training_completed" and entry.get("state") != "training_completed":
            return False
        if payload.get("selection_recorded") is not True or payload.get("checkpoint_validated") is not True:
            return False
        output = Path(task.get("output") or "")
        checkpoint = output / ("checkpoints/best/checkpoint.pt" if cell["model"] == "biomedbert" else "best_adapter")
        if cell["model"] == "biomedbert":
            return checkpoint.is_file() and checkpoint.stat().st_size > 0
        return artifact_valid(checkpoint) or (output / "adapter_checkpoint.pt").is_file()
    return _population_covers(payload, cell["dataset"])


def coverage_report(tasks: list[dict], state: dict) -> dict:
    """Accept cells from the frozen matrix. One encoder ledger can carry BioRED and BC8, and neither is inferred from a filename."""
    cells = []
    inventory = expected_cells()
    for kind, rows in inventory.items():
        for cell in rows:
            hits = [task for task in tasks if _matches(task, cell, kind)]
            task = hits[0] if len(hits) == 1 else None
            if task is None and kind == "evaluations" and cell["dataset"] == "bc8" and cell["model"] == "biomedbert":
                internal = {**cell, "dataset": "biored"}
                internal_hits = [item for item in tasks if _matches(item, internal, kind)]
                task = internal_hits[0] if len(internal_hits) == 1 else None
            entry = state.get(task["task_id"], {}) if task else {}
            status = entry.get("scientific") or entry.get("state") or ("missing_task" if task is None else "unsubmitted")
            accepted = bool(task) and len(hits) == 1 and _cell_accepted(kind, cell, task, entry)
            if not accepted and kind == "evaluations" and cell["dataset"] == "bc8" and cell["model"] == "biomedbert":
                internal = {**cell, "dataset": "biored"}
                internal_hits = [item for item in tasks if _matches(item, internal, kind)]
                if len(internal_hits) == 1 and _cell_accepted(kind, cell, internal_hits[0], state.get(internal_hits[0]["task_id"], {})):
                    task = internal_hits[0]
                    entry = state.get(task["task_id"], {})
                    status = entry.get("scientific") or entry.get("state") or status
                    accepted = True
            cells.append({
                "task_id": None if task is None else task["task_id"],
                "kind": kind,
                "dataset": cell["dataset"],
                "status": status if not accepted else (entry.get("scientific") or entry.get("state")),
                "artifact_present": accepted,
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
        "missing": sum(1 for cell in cells if not cell["artifact_present"]),
    }
