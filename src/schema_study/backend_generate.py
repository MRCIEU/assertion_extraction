"""Generate the frozen round-4 prompts with one backend and one batch setting.

Torch or vLLM is imported only after the output path and request list are checked.
Scoring is a separate command and never calls the model.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from schema_study.generation_ledger import append_jsonl, mark_generation, read_jsonl, resource_path, write_json

CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
ROOT = STORAGE / "studies/schema_granularity_v1/runs/backend_validation/round4_20261115"
OUTPUT_ROOT = Path(os.environ.get("SCHEMA_STUDY_GENERATION_ROOT", str(ROOT)))
MODELS = {
    "8b": {
        "model_id": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "path": STORAGE / "models/Qwen__Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218",
        "rendered_field": "rendered_qwen8b",
    },
    "32b": {
        "model_id": "Qwen/Qwen3-32B",
        "revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "path": STORAGE / "models/Qwen__Qwen3-32B/9216db5781bf21249d130ec9da846c4624c16137",
        "rendered_field": "rendered_qwen32b",
    },
}
WARMUP_COUNT = 4


def _requests() -> list[dict]:
    return read_jsonl(ROOT / "samples/requests.jsonl")


def _output(model: str, backend: str, batch: int, repeat: int) -> Path:
    path = OUTPUT_ROOT / f"qwen{model}" / backend / f"batch{batch}_repeat{repeat}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sample_device(stop: threading.Event, peaks: dict) -> None:
    import subprocess

    while not stop.is_set():
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            used = max(int(part) for part in completed.stdout.split())
            peaks["device_mib"] = max(peaks.get("device_mib", 0), used)
            peaks["device_samples"] = peaks.get("device_samples", 0) + 1
        stop.wait(2.0)


def _record(output: Path, request: dict, text: str, parsed: dict, meta: dict) -> None:
    append_jsonl(output / "attempts.jsonl", {
        "request_id": request["request_id"],
        "candidate_id": request["candidate_id"],
        "dataset": request["dataset"],
        "predicate_level": request["predicate_level"],
        "model_revision": meta["revision"],
        "backend": meta["backend"],
        "backend_version": meta["backend_version"],
        "batch_or_concurrency": meta["batch"],
        "measurement_repeat": meta["repeat"],
        "attempt": 1,
        "is_warmup": meta["is_warmup"],
        "included_in_timing": meta["included_in_timing"],
        "included_in_formal_analysis": False,
        "status": "completed",
        "model_called": True,
        "raw_answer": text,
        "output_status": parsed["output_status"],
        "predicted_labels": parsed["predicted_labels"],
        "illegal_labels": parsed["illegal_labels"],
        "input_tokens": meta["input_tokens"],
        "output_tokens": meta["output_tokens"],
        "truncated": meta["output_tokens"] >= 256,
    })


def _pending(output: Path, requests: list[dict], warmup: bool) -> list[dict]:
    done = {
        row["request_id"]
        for row in read_jsonl(output / "attempts.jsonl")
        if row.get("status") == "completed" and row.get("is_warmup") is warmup
    }
    selected = requests[:WARMUP_COUNT] if warmup else requests
    return [row for row in selected if row["request_id"] not in done]


def generate(model: str, backend: str, batch: int, repeat: int, only_ids: set[str] | None = None) -> None:
    spec = MODELS[model]
    if backend not in {"transformers", "vllm"}:
        raise ValueError(backend)
    requests = _requests()
    if only_ids is not None:
        requests = [row for row in requests if row["request_id"] in only_ids]
    missing_render = [row["request_id"] for row in requests if spec["rendered_field"] not in row]
    if missing_render:
        raise RuntimeError("Rendered prompts are not frozen for this model.")
    output = _output(model, backend, batch, repeat)
    if not _pending(output, requests, True) and not _pending(output, requests, False):
        mark_generation(output, {"complete": True, "model_called": False, "reason": "ledger_already_complete"})
        return
    capacity = json.loads((ROOT / "samples/context_capacity.json").read_text(encoding="utf-8"))
    write_json(output / "resolved_config.json", {"model": spec["model_id"], "revision": spec["revision"], "backend": backend, "batch_or_concurrency": batch, "measurement_repeat": repeat, "dtype": "bf16", "max_new_tokens": 256, "do_sample": False, "temperature": 0, "max_model_len": capacity["max_model_len"], "included_in_formal_analysis": False})
    write_json(output / "input_sources.json", {"requests": str(ROOT / "samples/requests.jsonl"), "weights": str(spec["path"]), "revision": spec["revision"]})
    peaks = {"device_mib": 0, "device_samples": 0}
    stop = threading.Event()
    thread = threading.Thread(target=_sample_device, args=(stop, peaks), daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        if backend == "transformers":
            _transformers(spec, requests, output, batch, repeat, capacity)
        else:
            _vllm(spec, requests, output, batch, repeat, capacity)
    finally:
        stop.set()
        thread.join(timeout=3)
    elapsed = time.perf_counter() - started
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    clock = {"wall_seconds_including_load_and_write": elapsed, "device_peak_mib": peaks["device_mib"], "device_samples": peaks["device_samples"], "sample_interval_seconds": 2.0, "slurm_job_id": job_id}
    write_json(output / f"run_clock_{job_id}.json", clock)


def _transformers(spec: dict, requests: list[dict], output: Path, batch: int, repeat: int, capacity: dict) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from schema_study.predictions import parse_prediction_text
    from schema_study.schemas import load_schema

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stopping: this GPU does not support bf16.")
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec["path"], local_files_only=True, torch_dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()
    load_seconds = time.perf_counter() - load_started
    version = __import__("transformers").__version__
    schemas = {name: load_schema(CODE / "configs/schemas" / f"{name}.yaml") for name in ("biored", "drugprot")}
    meta = {"revision": spec["revision"], "backend": "transformers", "backend_version": version, "batch": batch, "repeat": repeat}

    def run_group(pending: list[dict], warmup: bool) -> float:
        pure = 0.0
        for start in range(0, len(pending), batch):
            group = pending[start:start + batch]
            encoded = tokenizer(
                [row[spec["rendered_field"]] for row in group],
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            append_jsonl(output / "batches.jsonl", {
                "warmup": warmup,
                "batch_index": start // batch,
                "width": int(encoded["input_ids"].shape[1]),
                "request_ids": [row["request_id"] for row in group],
            })
            if int(encoded["input_ids"].shape[1]) + 256 > capacity["max_model_len"]:
                raise RuntimeError("A frozen prompt exceeds the recorded context capacity.")
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            lengths = encoded["attention_mask"].sum(dim=1).tolist()
            tick = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(**encoded, do_sample=False, max_new_tokens=256)
            pure += time.perf_counter() - tick
            for index, request in enumerate(group):
                new_tokens = output_ids[index, encoded["input_ids"].shape[1]:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                parsed = parse_prediction_text(text, expected_candidate_id=request["candidate_id"], schema=schemas[request["dataset"]], level=request["predicate_level"])
                _record(output, request, text, parsed, {**meta, "is_warmup": warmup, "included_in_timing": not warmup, "input_tokens": int(lengths[index]), "output_tokens": int(new_tokens.shape[0])})
        return pure

    warm_pending = _pending(output, requests, True)
    warm_seconds = run_group(warm_pending, True)
    timed_pending = _pending(output, requests, False)
    pure_seconds = run_group(timed_pending, False)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    write_json(resource_path(output, f"{job_id}_{spec['model_id'].split('/')[-1]}_{batch}_{repeat}"), {
        "process_id": f"transformers_batch{batch}_repeat{repeat}_job{job_id}",
        "backend": "transformers",
        "load_seconds": load_seconds,
        "warmup_generate_seconds": warm_seconds,
        "timed_generate_seconds": pure_seconds,
        "framework_allocated_bytes": torch.cuda.max_memory_allocated(),
        "framework_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_memory_note": "Device-wide peak is sampled from nvidia-smi every 2 seconds and stored in run_clock.json.",
        "scheduler_max_rss_bytes": None,
        "completed_timing_requests": len([row for row in read_jsonl(output / "attempts.jsonl") if row.get("included_in_timing")]),
        "dtype": "bf16",
        "greedy": "do_sample=false",
    })
    expected = len(requests)
    done = len([row for row in read_jsonl(output / "attempts.jsonl") if row.get("included_in_timing")])
    mark_generation(output, {"complete": done == expected, "completed_timing_requests": done, "expected": expected})


def _vllm(spec: dict, requests: list[dict], output: Path, batch: int, repeat: int, capacity: dict) -> None:
    from vllm import LLM, SamplingParams

    from schema_study.predictions import parse_prediction_text
    from schema_study.schemas import load_schema

    load_started = time.perf_counter()
    engine = LLM(
        model=str(spec["path"]),
        dtype="bfloat16",
        max_model_len=capacity["max_model_len"],
        max_num_seqs=batch,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=False,
    )
    load_seconds = time.perf_counter() - load_started
    params = SamplingParams(temperature=0, max_tokens=256)
    version = __import__("vllm").__version__
    schemas = {name: load_schema(CODE / "configs/schemas" / f"{name}.yaml") for name in ("biored", "drugprot")}
    meta = {"revision": spec["revision"], "backend": "vllm", "backend_version": version, "batch": batch, "repeat": repeat}

    def run_group(pending: list[dict], warmup: bool) -> float:
        if not pending:
            return 0.0
        prompts = [row[spec["rendered_field"]] for row in pending]
        tick = time.perf_counter()
        results = engine.generate(prompts, params)
        pure = time.perf_counter() - tick
        for request, result in zip(pending, results):
            text = result.outputs[0].text
            parsed = parse_prediction_text(text, expected_candidate_id=request["candidate_id"], schema=schemas[request["dataset"]], level=request["predicate_level"])
            _record(output, request, text, parsed, {
                **meta,
                "is_warmup": warmup,
                "included_in_timing": not warmup,
                "input_tokens": len(result.prompt_token_ids),
                "output_tokens": len(result.outputs[0].token_ids),
            })
        return pure

    warm_seconds = run_group(_pending(output, requests, True), True)
    pure_seconds = run_group(_pending(output, requests, False), False)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    write_json(resource_path(output, f"{job_id}_{spec['model_id'].split('/')[-1]}_{batch}_{repeat}"), {
        "process_id": f"vllm_concurrency{batch}_repeat{repeat}_job{job_id}",
        "backend": "vllm",
        "load_seconds": load_seconds,
        "warmup_generate_seconds": warm_seconds,
        "timed_generate_seconds": pure_seconds,
        "framework_allocated_bytes": None,
        "framework_reserved_bytes": None,
        "framework_note": "vLLM worker memory is not read from the parent torch process.",
        "device_memory_note": "Device-wide peak is sampled from nvidia-smi every 2 seconds and stored in run_clock.json.",
        "scheduler_max_rss_bytes": None,
        "completed_timing_requests": len([row for row in read_jsonl(output / "attempts.jsonl") if row.get("included_in_timing")]),
        "dtype": "bf16",
        "greedy": "SamplingParams temperature=0 corresponds to the study greedy setting",
        "prefix_caching": False,
    })
    expected = len(requests)
    done = len([row for row in read_jsonl(output / "attempts.jsonl") if row.get("included_in_timing")])
    mark_generation(output, {"complete": done == expected, "completed_timing_requests": done, "expected": expected})


def score(model: str, backend: str, batch: int, repeat: int) -> None:
    from schema_study.formal_eval import score_predictions
    from schema_study.predictions import rollup_predicted_labels
    from schema_study.schemas import load_schema

    import pyarrow.dataset as ds

    output = _output(model, backend, batch, repeat)
    rows = [row for row in read_jsonl(output / "attempts.jsonl") if row.get("included_in_timing")]
    started = time.perf_counter()
    summaries = []
    for dataset in ("biored", "drugprot"):
        schema = load_schema(CODE / "configs/schemas" / f"{dataset}.yaml")
        version = "biored_candidates_v2" if dataset == "biored" else "drugprot_candidates_v1"
        wanted = {row["candidate_id"] for row in rows if row["dataset"] == dataset}
        table = ds.dataset(str(STORAGE / "data/prepared" / dataset / version / "candidates.parquet")).to_table(
            columns=["candidate_id", "document_id", "split", "gold_labels"],
            filter=ds.field("candidate_id").isin(wanted),
        )
        gold = {row["candidate_id"]: row for row in table.to_pylist()}
        for level in ("P4", "P2"):
            selected = [row for row in rows if row["dataset"] == dataset and row["predicate_level"] == level]
            population = []
            predictions = []
            for row in selected:
                source = gold[row["candidate_id"]]
                population.append({"candidate_id": row["candidate_id"], "document_id": source["document_id"], "split": "dev", "gold_labels": list(source["gold_labels"] or []), "included_in_scoring": True})
                predictions.append({"candidate_id": row["candidate_id"], "document_id": source["document_id"], "split": "dev", "output_status": row["output_status"], "predicted_labels": row["predicted_labels"], "illegal_labels": row["illegal_labels"]})
            summaries.append({"dataset": dataset, "level": level, "summary": score_predictions(population, predictions, schema, level, allowed_splits={"dev"}, data_version=version, input_version="round4_technical", prediction_version=f"{backend}_batch{batch}", mapping_version=schema.schema_id)["summary"]})
            if level == "P4":
                rolled = [{**prediction, "predicted_labels": rollup_predicted_labels(schema, prediction["predicted_labels"], source_level="P4", target_level="P2")} for prediction in predictions]
                summaries.append({"dataset": dataset, "level": "P2_rollup", "summary": score_predictions(population, rolled, schema, "P2", allowed_splits={"dev"}, data_version=version, input_version="round4_technical", prediction_version=f"{backend}_rollup", mapping_version=schema.schema_id)["summary"]})
    write_json(output / "score_summary.json", {"scores": summaries, "score_seconds": time.perf_counter() - started, "included_in_formal_analysis": False, "study_phase": "backend_validation"})
    from schema_study.generation_ledger import mark_score

    mark_score(output, {"complete": True})


if __name__ == "__main__":
    import sys

    command, model, backend, batch, repeat = sys.argv[1:6]
    if command == "generate":
        only = None
        if len(sys.argv) > 6:
            extra = Path(sys.argv[6])
            text = extra.read_text(encoding="utf-8") if extra.is_file() else sys.argv[6]
            only = {item for item in text.replace(",", "\n").split() if item}
        generate(model, backend, int(batch), int(repeat), only)
    elif command == "score":
        score(model, backend, int(batch), int(repeat))
    else:
        raise SystemExit(f"Unknown command {command}")
