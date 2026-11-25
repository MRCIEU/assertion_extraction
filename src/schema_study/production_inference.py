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
    """Call the installed vLLM 0.13 `LLM.generate` interface and parse each answer."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams

    from schema_study.production_runtime import interpret_generation

    engine = LLM(model=sampling["model_path"], dtype="bfloat16", enable_lora=lora_path is not None)
    structured = sampling.get("structured_outputs")
    params = SamplingParams(
        temperature=sampling.get("temperature", 0.0),
        max_tokens=sampling.get("max_tokens", 256),
        n=int(sampling.get("n", 1)),
        structured_outputs=StructuredOutputsParams(**structured) if structured else None,
    )
    lora = LoRARequest("adapter", 1, lora_path) if lora_path else None
    outputs = engine.generate(prompts, params, lora_request=lora)
    rows = []
    for output in outputs:
        raw = output.outputs[0].text if output.outputs else ""
        labels = None
        parsed = None
        if sampling.get("schema") is not None and sampling.get("candidate_id") and sampling.get("level"):
            parsed = interpret_generation(raw, sampling["candidate_id"], sampling["schema"], sampling["level"])
            if parsed["ledger_status"] != "invalid":
                labels = parsed["predicted_labels"]
        rows.append({"raw": raw, "labels": labels, "parsed": parsed})
    return rows


def _run(name: str, requests: list[dict], completed: dict[str, str], backend, ledger: Path | None) -> dict:
    if backend is None:
        return {"family": name, "submitted": False, "reason": "formal_gpu_execution_is_not_authorised"}
    updated = apply_requests(requests, completed, backend)
    if ledger is not None:
        write_ledger(ledger, updated, requests)
    return {"family": name, "submitted": False, "executed": True, "ledger": updated}


def constrained_generate(requests, completed=None, backend=None, ledger=None) -> dict:
    from schema_study.production_runtime import constrained_sampling

    prepared = []
    for request in requests:
        schema_json = request.get("schema_json") or {"type": "object", "properties": {"labels": {"type": "array"}}, "required": ["labels"]}
        prepared.append({**request, "sampling": constrained_sampling(schema_json), "decoding": "constrained"})
    if backend is None:
        return {"family": "constrained", "submitted": False, "reason": "formal_gpu_execution_is_not_authorised", "requests": prepared}
    return _run("constrained", prepared, completed or {}, backend, ledger)


def thinking_generate(requests, completed=None, backend=None, ledger=None) -> dict:
    from schema_study.production_runtime import thinking_request

    prepared = [{**request, **thinking_request(request.get("prompt") or request.get("user") or "")} for request in requests]

    def generate(request):
        result = backend(request)
        raw = result.get("raw") or ""
        final = raw.split("</think>")[-1].strip() if "</think>" in raw else raw
        return {**result, "raw": final, "reasoning": raw, "final_answer_separated": True}

    if backend is None:
        return {"family": "thinking", "submitted": False, "reason": "formal_gpu_execution_is_not_authorised", "requests": prepared}
    return _run("thinking", prepared, completed or {}, generate, ledger)


def self_consistency(requests, completed=None, backend=None, ledger=None) -> dict:
    from schema_study.production_runtime import self_consistency as aggregate

    if backend is None:
        return {"family": "self_consistency", "submitted": False, "reason": "formal_gpu_execution_is_not_authorised", "samples": 5}

    def generate(request):
        samples = [backend({**request, "sample_index": index}) for index in range(5)]
        for sample in samples:
            sample.setdefault("ledger_status", "valid" if sample.get("labels") is not None else "invalid")
            sample.setdefault("predicted_labels", sample.get("labels") or [])
        aggregated = aggregate(samples, bool(request.get("multi")))
        return {"raw": json.dumps(samples), "labels": aggregated["labels"], "samples": samples}

    return _run("self_consistency", requests, completed or {}, generate, ledger)


def evidence_generate(requests, completed=None, backend=None, ledger=None) -> dict:
    from schema_study.production_runtime import evidence_record

    if backend is None:
        return {"family": "evidence", "submitted": False, "reason": "formal_gpu_execution_is_not_authorised"}

    def generate(request):
        result = backend(request)
        record = evidence_record(result.get("raw") or "", request.get("source_text") or "", result.get("spans") or request.get("spans") or [])
        labels = result.get("labels") if record["complete"] else None
        return {**result, "labels": labels, "evidence": record}

    return _run("evidence", requests, completed or {}, generate, ledger)


def qualified_generate(requests, completed=None, backend=None, ledger=None) -> dict:
    from schema_study.production_runtime import evidence_record

    if backend is None:
        return {"family": "qualified", "submitted": False, "reason": "formal_gpu_execution_is_not_authorised"}

    def generate(request):
        result = backend(request)
        record = evidence_record(result.get("raw") or "", request.get("source_text") or "", result.get("spans") or [])
        qualifications = result.get("qualifications") or []
        complete = record["complete"] and bool(qualifications)
        return {**result, "labels": result.get("labels") if complete else None, "qualifications": qualifications, "evidence": record}

    return _run("qualified", requests, completed or {}, generate, ledger)


def candidate_scores(requests, completed=None, backend=None, ledger=None) -> dict:
    from schema_study.production_runtime import score_candidates

    if backend is None:
        return {"family": "candidate_scoring", "submitted": False, "reason": "formal_gpu_execution_is_not_authorised"}

    def generate(request):
        result = backend(request)
        scored = score_candidates(result.get("scores") or {}, bool(request.get("multi")))
        return {"raw": json.dumps(scored["scores"]), "labels": scored["labels"], "scores": scored}

    return _run("candidate_scoring", requests, completed or {}, generate, ledger)
