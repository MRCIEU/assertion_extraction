"""Inference ledgers and supplementary generation.

The functions below build requests, call an injected backend, and record
statuses. They do not load vLLM unless `backend` is omitted on an authorised
GPU run. A valid empty answer is complete. An invalid answer is not retried.
"""

from __future__ import annotations

import json
from pathlib import Path

RETRYABLE_STATUSES = {"retryable", "filesystem", "timeout", "node_fail"}
TERMINAL_STATUSES = {"valid", "valid_empty", "invalid", "incorrect"}
EXTERNAL = {"inputs_prepared", "model_assets", "approval", "historical_drugprot_checkpoint"}


def pending_requests(request_ids: list[str], completed: dict[str, str]) -> list[str]:
    """Resume missing and retryable execution failures. Scientific invalidity is not retried."""
    pending = []
    for request_id in request_ids:
        status = completed.get(request_id)
        if status is None or status in RETRYABLE_STATUSES:
            pending.append(request_id)
    return pending


def release_dependents(tasks: list[dict], completed_ids: set[str], verified_external: set[str] | None = None) -> list[str]:
    """Release a task only when task dependencies and named external prerequisites are verified."""
    verified_external = verified_external or set()
    released = []
    for task in tasks:
        dependencies = list(task.get("depends_on", []))
        if not dependencies:
            continue
        ready = True
        for item in dependencies:
            if item in EXTERNAL:
                ready = item in verified_external
            elif item not in completed_ids:
                ready = False
            if not ready:
                break
        if ready:
            released.append(task["task_id"])
    return released


def group_requests(rows: list[dict]) -> list[list[dict]]:
    """Group requests that can share one loaded model and adapter."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row.get("model"), row.get("revision"), row.get("adapter"), row.get("decoding"))
        groups.setdefault(key, []).append(row)
    return list(groups.values())


def classify_output(raw: str | None, labels: list[str] | None, failed: bool = False) -> str:
    if failed:
        return "retryable"
    if raw is None:
        return "invalid"
    if labels is None:
        return "invalid"
    if labels == [] and raw.strip() in {"", "{}", '{"labels": []}'}:
        return "valid_empty"
    if labels == [] and raw.strip():
        return "invalid"
    return "valid"


def apply_requests(requests: list[dict], completed: dict[str, str], generate) -> dict[str, str]:
    """Generate only the pending requests and append them to the ledger."""
    updated = dict(completed)
    for request in requests:
        if request["request_id"] not in pending_requests([request["request_id"]], updated):
            continue
        try:
            result = generate(request)
        except OSError:
            updated[request["request_id"]] = "retryable"
            continue
        except RuntimeError as exc:
            if "retryable" in str(exc):
                updated[request["request_id"]] = "retryable"
            else:
                updated[request["request_id"]] = "invalid"
            continue
        updated[request["request_id"]] = classify_output(result.get("raw"), result.get("labels"), failed=bool(result.get("failed")))
    return updated


def write_ledger(path: Path, completed: dict[str, str], rows: list[dict]) -> None:
    by_id = {row["request_id"]: row for row in rows}
    lines = []
    for request_id, status in completed.items():
        source = by_id.get(request_id, {"request_id": request_id})
        lines.append(json.dumps({**source, "status": status}, sort_keys=True))
    temporary = path.with_suffix(".jsonl.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    temporary.replace(path)


def vllm_generate(prompts: list[str], sampling: dict, lora_path: str | None = None) -> list[dict]:
    """Call the installed vLLM 0.13 `LLM.generate` interface. Tests inject a backend instead."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    engine = LLM(model=sampling["model_path"], dtype="bfloat16", enable_lora=lora_path is not None)
    params = SamplingParams(temperature=sampling.get("temperature", 0.0), max_tokens=sampling.get("max_tokens", 256))
    lora = LoRARequest("adapter", 1, lora_path) if lora_path else None
    outputs = engine.generate(prompts, params, lora_request=lora)
    return [{"raw": output.outputs[0].text if output.outputs else "", "labels": None} for output in outputs]


def _run(name: str, requests: list[dict], completed: dict[str, str], backend, ledger: Path | None) -> dict:
    if backend is None:
        return {"family": name, "submitted": False, "reason": "formal_gpu_execution_is_not_authorised"}
    updated = apply_requests(requests, completed, backend)
    if ledger is not None:
        write_ledger(ledger, updated, requests)
    return {"family": name, "submitted": False, "executed": True, "ledger": updated}


def constrained_generate(requests, completed=None, backend=None, ledger=None) -> dict:
    return _run("constrained", requests, completed or {}, backend, ledger)


def thinking_generate(requests, completed=None, backend=None, ledger=None) -> dict:
    return _run("thinking", requests, completed or {}, backend, ledger)


def self_consistency(requests, completed=None, backend=None, ledger=None) -> dict:
    return _run("self_consistency", requests, completed or {}, backend, ledger)


def evidence_generate(requests, completed=None, backend=None, ledger=None) -> dict:
    return _run("evidence", requests, completed or {}, backend, ledger)


def qualified_generate(requests, completed=None, backend=None, ledger=None) -> dict:
    return _run("qualified", requests, completed or {}, backend, ledger)


def candidate_scores(requests, completed=None, backend=None, ledger=None) -> dict:
    return _run("candidate_scoring", requests, completed or {}, backend, ledger)
