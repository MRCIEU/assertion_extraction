"""Inference request recovery and supplementary generation entry points.

These functions do not load a generative model. Formal generation stays behind
the approval gate in the dispatcher.
"""

from __future__ import annotations


def pending_requests(request_ids: list[str], completed: dict[str, str]) -> list[str]:
    """Resume missing and explicitly retryable requests. A valid empty output is complete."""
    pending = []
    for request_id in request_ids:
        status = completed.get(request_id)
        if status is None or status == "retryable":
            pending.append(request_id)
        elif status not in {"valid", "valid_empty"}:
            pending.append(request_id)
    return pending


def release_dependents(tasks: list[dict], completed_ids: set[str]) -> list[str]:
    """A dependent task is released only when every dependency has a verified completion."""
    external = {"inputs_prepared", "model_assets", "approval", "historical_drugprot_checkpoint"}
    released = []
    for task in tasks:
        dependencies = [item for item in task.get("depends_on", []) if item not in external]
        if dependencies and all(item in completed_ids for item in dependencies):
            released.append(task["task_id"])
    return released


def _not_authorised(name: str) -> dict:
    return {"family": name, "submitted": False, "reason": "formal_gpu_execution_is_not_authorised"}


def constrained_generate(*_args, **_kwargs) -> dict:
    return _not_authorised("constrained")


def thinking_generate(*_args, **_kwargs) -> dict:
    return _not_authorised("thinking")


def self_consistency(*_args, **_kwargs) -> dict:
    return _not_authorised("self_consistency")


def evidence_generate(*_args, **_kwargs) -> dict:
    return _not_authorised("evidence")


def qualified_generate(*_args, **_kwargs) -> dict:
    return _not_authorised("qualified")
