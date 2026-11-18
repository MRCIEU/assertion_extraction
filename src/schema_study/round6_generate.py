"""Generate one frozen round-6 prompt version. Scoring is a separate command."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from schema_study.batch_plan import adopted_answers, batches_to_execute, classify_attempt, prompt_sha256
from schema_study.generation_ledger import append_jsonl, read_jsonl
from schema_study.predictions import parse_prediction_text
from schema_study.process_record import process_directory, write_process_json
from schema_study.schemas import load_schema

CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
ROOT = STORAGE / "studies/schema_granularity_v1/runs/task_suitability/round6_20261115"
MODELS = {
    "qwen8b": {
        "model_id": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "path": STORAGE / "models/Qwen__Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218",
    },
    "qwen32b": {
        "model_id": "Qwen/Qwen3-32B",
        "revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "path": STORAGE / "models/Qwen__Qwen3-32B/9216db5781bf21249d130ec9da846c4624c16137",
    },
}


def _manifest(version: str, model_name: str) -> dict:
    specific = ROOT / "plan" / f"batch_manifest_{version}_{model_name}.json"
    shared = ROOT / "plan" / f"batch_manifest_{version}.json"
    path = specific if specific.is_file() else shared
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_device(stop: threading.Event, peaks: dict) -> None:
    import subprocess

    while not stop.is_set():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=5,
            )
            used = max(int(part.strip()) for part in result.stdout.splitlines() if part.strip())
            peaks["device_mib"] = max(peaks["device_mib"], used)
            peaks["samples"].append(used)
        except Exception:
            peaks["sample_errors"] += 1
        stop.wait(2.0)


def generate(version: str, model_name: str) -> None:
    if version not in {"v0", "v1"} or model_name not in MODELS:
        raise SystemExit("Usage: round6_generate generate v0|v1 qwen8b|qwen32b")
    spec = MODELS[model_name]
    requests = {row["request_id"]: row for row in read_jsonl(ROOT / "samples/requests.jsonl")}
    manifest = _manifest(version, model_name)
    output = ROOT / "generation" / version / model_name
    output.mkdir(parents=True, exist_ok=True)
    adopted = adopted_answers(read_jsonl(output / "attempts.jsonl"))
    pending = batches_to_execute(manifest, adopted)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    attempt_id = "generate"
    suffix = 1
    while (output / "processes" / f"{job_id}_{attempt_id}").exists():
        suffix += 1
        attempt_id = f"generate-{suffix}"
    record = process_directory(output, job_id, attempt_id)
    if not pending:
        write_process_json(record, "status.json", {"model_called": False, "reason": "ledger_already_complete", "calls": 0})
        write_process_json(record, "run_clock.json", {"wall_seconds": 0, "slurm_job_id": job_id, "attempt_id": attempt_id})
        write_process_json(record, "resource_usage.json", {"model_called": False, "calls": 0, "slurm_job_id": job_id})
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stopping: this GPU does not support bf16.")
    peaks = {"device_mib": 0, "samples": [], "sample_errors": 0}
    stop = threading.Event()
    thread = threading.Thread(target=_sample_device, args=(stop, peaks), daemon=True)
    thread.start()
    started = time.perf_counter()
    calls = 0
    allocated = 0
    reserved = 0
    error = ""
    try:
        tokenizer = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(spec["path"], local_files_only=True, torch_dtype=torch.bfloat16)
        model.to("cuda")
        model.eval()
        schemas = {name: load_schema(CODE / "configs/schemas" / f"{name}.yaml") for name in ("biored", "drugprot")}
        field = f"rendered_{version}_{model_name}"

        def run_batch(batch: dict, warmup: bool) -> None:
            nonlocal calls
            group = [requests[request_id] for request_id in batch["request_ids"]]
            encoded = tokenizer([row[field] for row in group], return_tensors="pt", padding=True, truncation=False)
            width = int(encoded["input_ids"].shape[1])
            if width + 256 > int(model.config.max_position_embeddings):
                raise RuntimeError("A prompt plus 256 tokens exceeds the model position limit.")
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            calls += len(group)
            with torch.no_grad():
                output_ids = model.generate(**encoded, do_sample=False, max_new_tokens=256)
            current = adopted_answers(read_jsonl(output / "attempts.jsonl"))
            for index, request in enumerate(group):
                new_tokens = output_ids[index, width:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                parsed = parse_prediction_text(text, expected_candidate_id=request["candidate_id"], schema=schemas[request["dataset"]], level=request["predicate_level"])
                member = next(item for item in batch["members"] if item["request_id"] == request["request_id"])
                digest = prompt_sha256(request[field])
                if digest != member["prompt_sha256"]:
                    raise RuntimeError(f"Prompt hash changed for {request['request_id']}.")
                row = {
                    "request_id": request["request_id"],
                    "candidate_id": request["candidate_id"],
                    "document_id": request["document_id"],
                    "dataset": request["dataset"],
                    "predicate_level": request["predicate_level"],
                    "prompt_sha256": digest,
                    "batch_id": batch["batch_id"],
                    "pad_width": width,
                    "is_warmup": warmup,
                    "status": "completed",
                    "raw_answer": text,
                    "output_status": parsed["output_status"],
                    "predicted_labels": parsed["predicted_labels"],
                    "illegal_labels": parsed["illegal_labels"],
                    "truncated": int(new_tokens.shape[0]) >= 256,
                    "model_revision": spec["revision"],
                    "slurm_job_id": job_id,
                }
                decision = classify_attempt(batch, row, current)
                row["adoption"] = decision
                append_jsonl(output / "attempts.jsonl", row)
                if decision == "adopt":
                    current[request["request_id"]] = row
            append_jsonl(record / "batches.jsonl", {"batch_id": batch["batch_id"], "warmup": warmup, "pad_width": width, "request_ids": batch["request_ids"]})

        run_batch(manifest["batches"][0], True)
        for batch in pending:
            run_batch(batch, False)
        allocated = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop.set()
        thread.join(timeout=3)
        elapsed = time.perf_counter() - started
        write_process_json(record, "run_clock.json", {
            "wall_seconds": elapsed, "device_peak_mib": peaks["device_mib"], "device_samples": peaks["samples"],
            "sample_errors": peaks["sample_errors"], "slurm_job_id": job_id, "attempt_id": attempt_id,
            "version": version, "model": model_name,
        })
        write_process_json(record, "resource_usage.json", {
            "model_called": True, "calls": calls, "wall_seconds": elapsed, "framework_allocated_bytes": allocated,
            "framework_reserved_bytes": reserved, "device_peak_mib": peaks["device_mib"], "slurm_job_id": job_id,
            "attempt_id": attempt_id, "dtype": "bf16", "batch_size": 16, "do_sample": False, "max_new_tokens": 256,
            "error": error,
        })
        write_process_json(record, "status.json", {
            "model_called": True, "calls": calls, "pending_batches": [batch["batch_id"] for batch in pending], "error": error,
        })


def main() -> None:
    import sys
    if len(sys.argv) != 4 or sys.argv[1] != "generate":
        raise SystemExit("Usage: python -m schema_study.round6_generate generate v0|v1 qwen8b|qwen32b")
    generate(sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
