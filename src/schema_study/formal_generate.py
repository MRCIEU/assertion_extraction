"""Generate one frozen formal shard. Scoring is separate and does not call the model."""

from __future__ import annotations

import json
import os
from pathlib import Path

from schema_study.batch_plan import adopted_answers, batches_to_execute, classify_attempt, prompt_sha256
from schema_study.generation_ledger import append_jsonl, read_jsonl
from schema_study.predictions import parse_prediction_text
from schema_study.process_record import process_directory, write_process_json
from schema_study.schemas import load_schema

CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
ROOT = STORAGE / "studies/schema_granularity_v1/runs/formal/formal_20261115"
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


def _manifest_for(rows: list[dict], field: str) -> dict:
    batches = []
    for start in range(0, len(rows), 16):
        group = rows[start:start + 16]
        batches.append({
            "batch_id": f"batch-{start // 16:05d}",
            "request_ids": [row["request_id"] for row in group],
            "members": [{
                "request_id": row["request_id"],
                "candidate_id": row["candidate_id"],
                "dataset": row["dataset"],
                "predicate_level": row["predicate_level"],
                "prompt_sha256": row["rendered_sha256"] if field == "rendered" else row.get("rendered_qwen32b_sha256", row["rendered_sha256"]),
            } for row in group],
        })
    return {"immutable": True, "batch_size": 16, "batches": batches}


def generate(model_name: str, shard_path: str) -> None:
    import time
    import threading
    spec = MODELS[model_name]
    source = Path(shard_path)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("Shard is empty.")
    first = rows[0]
    field = "rendered_qwen32b" if model_name == "qwen32b" and "rendered_qwen32b" in first else "rendered"
    output = ROOT / "generation/outputs" / model_name / first["dataset"] / first["entity_level"] / first["predicate_level"] / source.stem
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "batch_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(_manifest_for(rows, field), ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {row["request_id"]: row for row in rows}
    adopted = adopted_answers(read_jsonl(output / "attempts.jsonl"))
    pending = batches_to_execute(manifest, adopted)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    attempt = "generate"
    suffix = 1
    while (output / "processes" / f"{job_id}_{attempt}").exists():
        suffix += 1
        attempt = f"generate-{suffix}"
    record = process_directory(output, job_id, attempt)
    if not pending:
        write_process_json(record, "status.json", {"model_called": False, "reason": "ledger_already_complete", "calls": 0, "slurm_job_id": job_id})
        write_process_json(record, "resource_usage.json", {"model_called": False, "calls": 0, "slurm_job_id": job_id})
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stopping: this GPU does not support bf16.")
    peaks = {"device_mib": 0, "samples": 0, "errors": 0}
    stop = threading.Event()

    def sample() -> None:
        import subprocess
        while not stop.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=5)
                used = max(int(part.strip()) for part in result.stdout.splitlines() if part.strip())
                peaks["device_mib"] = max(peaks["device_mib"], used)
                peaks["samples"] += 1
            except Exception:
                peaks["errors"] += 1
            stop.wait(2.0)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = time.perf_counter()
    calls = 0
    allocated = 0
    error = ""
    try:
        tokenizer = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(spec["path"], local_files_only=True, torch_dtype=torch.bfloat16)
        model.to("cuda")
        model.eval()
        limit = int(model.config.max_position_embeddings)
        schemas = {
            "biored": load_schema(CODE / "configs/schemas/biored.yaml"),
            "bc8": load_schema(CODE / "configs/schemas/biored.yaml"),
            "drugprot": load_schema(CODE / "configs/schemas/drugprot.yaml"),
        }
        for batch in pending:
            group = [by_id[request_id] for request_id in batch["request_ids"]]
            texts = [row[field] for row in group]
            for row, text in zip(group, texts):
                if prompt_sha256(text) != (row.get("rendered_qwen32b_sha256") if field == "rendered_qwen32b" else row["rendered_sha256"]):
                    raise RuntimeError(f"Prompt hash changed for {row['request_id']}.")
            encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
            width = int(encoded["input_ids"].shape[1])
            if width + 256 > limit:
                raise RuntimeError("A prompt plus 256 tokens exceeds the model position limit.")
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            calls += len(group)
            with torch.no_grad():
                output_ids = model.generate(**encoded, do_sample=False, max_new_tokens=256)
            current = adopted_answers(read_jsonl(output / "attempts.jsonl"))
            for index, request in enumerate(group):
                new_tokens = output_ids[index, width:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                parsed = parse_prediction_text(
                    text,
                    expected_candidate_id=request["candidate_id"],
                    schema=schemas[request["dataset"]],
                    level=request["predicate_level"],
                )
                member = batch["members"][index]
                row = {
                    "request_id": request["request_id"],
                    "candidate_id": request["candidate_id"],
                    "document_id": request["document_id"],
                    "dataset": request["dataset"],
                    "split": request["split"],
                    "entity_level": request["entity_level"],
                    "predicate_level": request["predicate_level"],
                    "prompt_sha256": member["prompt_sha256"],
                    "batch_id": batch["batch_id"],
                    "pad_width": width,
                    "is_warmup": False,
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
            append_jsonl(record / "batches.jsonl", {"batch_id": batch["batch_id"], "pad_width": width, "request_ids": batch["request_ids"]})
        allocated = int(torch.cuda.max_memory_allocated())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop.set()
        thread.join(timeout=3)
        elapsed = time.perf_counter() - started
        write_process_json(record, "run_clock.json", {"wall_seconds": elapsed, "device_peak_mib": peaks["device_mib"], "device_samples": peaks["samples"], "sample_errors": peaks["errors"], "slurm_job_id": job_id, "attempt_id": attempt})
        write_process_json(record, "resource_usage.json", {"model_called": True, "calls": calls, "wall_seconds": elapsed, "framework_allocated_bytes": allocated, "device_peak_mib": peaks["device_mib"], "error": error, "slurm_job_id": job_id, "attempt_id": attempt, "dtype": "bf16", "batch_size": 16, "do_sample": False, "max_new_tokens": 256})
        write_process_json(record, "status.json", {"model_called": True, "calls": calls, "error": error, "slurm_job_id": job_id})


def main() -> None:
    import sys
    if len(sys.argv) != 3:
        raise SystemExit("Usage: formal_generate model shard_path")
    generate(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
