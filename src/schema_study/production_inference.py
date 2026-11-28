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
                continue
            raise
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


def vllm_generate(prompts: list[str], sampling: dict, lora_path: str | None = None, engine=None) -> list[dict]:
    """Generate one batch on a supplied engine. A new engine is built only when the caller has not loaded one."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams

    from schema_study.production_runtime import interpret_generation

    if engine is None:
        if not sampling.get("model_path"):
            raise RuntimeError("A production engine requires the pinned model path.")
        engine = LLM(model=sampling["model_path"], dtype="bfloat16", enable_lora=lora_path is not None)
    structured = sampling.get("structured_outputs")
    params = SamplingParams(
        temperature=sampling.get("temperature", 0.0),
        max_tokens=sampling.get("max_tokens", 256),
        n=int(sampling.get("n", 1)),
        structured_outputs=StructuredOutputsParams(**structured) if structured else None,
        prompt_logprobs=int(sampling["prompt_logprobs"]) if sampling.get("prompt_logprobs") else None,
    )
    lora = LoRARequest("adapter", 1, lora_path) if lora_path else None
    outputs = engine.generate(prompts, params, lora_request=lora)
    rows = []
    for output in outputs:
        completions = list(output.outputs or [])
        raw = completions[0].text if completions else ""
        samples = []
        for completion in completions:
            sample_raw = completion.text
            sample = {"raw": sample_raw, "finish_reason": completion.finish_reason}
            if sampling.get("schema") is not None and sampling.get("candidate_id") and sampling.get("level"):
                parsed = interpret_generation(sample_raw, sampling["candidate_id"], sampling["schema"], sampling["level"])
                sample["labels"] = None if parsed["ledger_status"] == "invalid" else parsed["predicted_labels"]
                sample["parsed"] = parsed
                sample["ledger_status"] = parsed["ledger_status"]
                sample["predicted_labels"] = parsed["predicted_labels"]
            samples.append(sample)
        labels = samples[0].get("labels") if samples else None
        parsed = samples[0].get("parsed") if samples else None
        prompt_token_logprobs = []
        for entry in output.prompt_logprobs or []:
            if not entry:
                continue
            chosen = max(entry.values(), key=lambda item: getattr(item, "rank", 0) == 1)
            prompt_token_logprobs.append(float(getattr(chosen, "logprob", chosen)))
        rows.append({
            "raw": raw,
            "labels": labels,
            "parsed": parsed,
            "samples": samples,
            "prompt_token_logprobs": prompt_token_logprobs,
            "completion_tokens": sum(len(getattr(completion, "token_ids", []) or []) for completion in completions),
        })
    return rows


def transformers_generate(model, tokenizer, prompts: list[str], max_new_tokens: int, temperature: float, samples: int) -> list[dict]:
    """Greedy or sampled generation with left padding. The continuation starts at the padded width."""
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    width = int(encoded["input_ids"].shape[1])
    prompt_lengths = [int(encoded["attention_mask"][index].sum()) for index in range(len(prompts))]
    generate = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "num_return_sequences": samples,
    }
    if temperature and temperature > 0:
        generate.update({"do_sample": True, "temperature": temperature})
    else:
        generate["do_sample"] = False
    with torch.inference_mode():
        output = model.generate(**encoded, **generate)
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    rows = []
    for index, _prompt in enumerate(prompts):
        samples_out = []
        for sample_index in range(samples):
            continuation = output[index * samples + sample_index][width:].tolist()
            produced = []
            reason = "other"
            for token in continuation:
                if pad is not None and token == pad and token != eos:
                    break
                produced.append(int(token))
                if eos is not None and token == eos:
                    reason = "eos"
                    break
            if reason != "eos" and len(produced) >= max_new_tokens:
                reason = "length"
            text = tokenizer.decode(produced, skip_special_tokens=True)
            samples_out.append({
                "raw": text,
                "finish_reason": reason,
                "completion_tokens": len(produced),
                "prompt_tokens": prompt_lengths[index],
            })
        rows.append({
            "raw": samples_out[0]["raw"] if samples_out else "",
            "labels": None,
            "parsed": None,
            "samples": samples_out,
            "prompt_token_logprobs": [],
            "prompt_tokens": samples_out[0]["prompt_tokens"] if samples_out else prompt_lengths[index],
            "completion_tokens": samples_out[0]["completion_tokens"] if samples_out else 0,
            "finish_reason": samples_out[0]["finish_reason"] if samples_out else "other",
            "backend": "transformers",
        })
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
    if backend is None:
        return {"family": "candidate_scoring", "submitted": False, "reason": "formal_gpu_execution_is_not_authorised"}

    def generate(request):
        from schema_study.production_runtime import require_legal_scores, sequence_logprobs

        result = backend(request)
        legal = request.get("legal_labels") or []
        if not legal:
            raise RuntimeError("Candidate scoring has no legal output schema.")
        if "token_logprobs" not in result:
            raise RuntimeError("Candidate scoring requires token log probabilities for each legal sequence.")
        scored = require_legal_scores(sequence_logprobs(result["token_logprobs"]), legal, bool(request.get("multi")))
        return {"raw": json.dumps({"labels": scored["labels"]}), "labels": scored["labels"], "scores": scored}

    return _run("candidate_scoring", requests, completed or {}, generate, ledger)
