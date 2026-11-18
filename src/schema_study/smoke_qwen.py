"""Qwen3-8B smoke generation with the local Transformers path.

This does not validate vLLM. Torch is imported only after the smoke permit check.
"""

from __future__ import annotations

from pathlib import Path

from schema_study.generation_ledger import (
    append_jsonl,
    completed_ids,
    mark_generation,
    mark_score,
    read_jsonl,
    resource_path,
    select_pending,
    write_json,
)
from schema_study.predictions import parse_prediction_text, rollup_predicted_labels
from schema_study.schemas import load_schema
from schema_study.smoke import SMOKE_MODELS, assert_smoke_config

CODE_ROOT = Path("/home/b5ac/freddieyu.b5ac/project_1")
STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
MODEL_DIR = STORAGE / "models/Qwen__Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218"
SMOKE_ROOT = STORAGE / "studies/schema_granularity_v1/runs/smoke/round3_20261115"
MODEL_ID = "Qwen/Qwen3-8B"
PROMPT = (CODE_ROOT / "prompts/relation_extraction.txt").read_text(encoding="utf-8")


def config() -> dict:
    return {
        "phase": "smoke",
        "study_phase": "smoke",
        "included_in_formal_analysis": False,
        "formal_execution_authorized": False,
        "model_id": MODEL_ID,
        "revision": SMOKE_MODELS[MODEL_ID],
        "splits": ["train", "dev"],
        "output_dir": str(SMOKE_ROOT / "qwen3_8b"),
        "max_generation_requests": 50,
        "max_updates": 0,
    }


def _load_requests() -> list[dict]:
    return read_jsonl(SMOKE_ROOT / "samples/qwen_requests.jsonl")


def generate(stage: str) -> None:
    settings = config()
    assert_smoke_config(settings)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stopping: this GPU does not support bf16. Precision was not changed.")
    output = Path(settings["output_dir"])
    for name in ("requests", "responses", "logs"):
        (output / name).mkdir(parents=True, exist_ok=True)
    ledger_path = output / "attempts.jsonl"
    done = completed_ids(read_jsonl(ledger_path))
    calls = sum(1 for row in read_jsonl(ledger_path) if row.get("model_called"))
    requests = _load_requests()
    pending = select_pending(requests, done, stage, 24)
    if not pending:
        mark_generation(output, {"complete": len(done) == 48, "stage": stage, "completed_nonrepeat": len(done), "model_called": False})
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16)
    model.to("cuda")
    template_ok = False
    for request in pending:
        if calls >= 50:
            raise RuntimeError("Qwen smoke generation cap of 50 was reached.")
        if stage != "repeat" and request["request_id"] in done:
            continue
        messages = [{"role": "user", "content": request["prompt"]}]
        try:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            template_ok = True
        except TypeError as exc:
            raise RuntimeError("The installed chat template does not accept enable_thinking=false.") from exc
        (output / "requests" / f"{request['request_id'].replace('|', '_')}.txt").write_text(rendered, encoding="utf-8")
        encoded = tokenizer(rendered, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            output_ids = model.generate(**encoded, do_sample=False, max_new_tokens=256)
        new_tokens = output_ids[0, encoded["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        schema = load_schema(CODE_ROOT / "configs/schemas" / f"{request['dataset']}.yaml")
        parsed = parse_prediction_text(text, expected_candidate_id=request["candidate_id"], schema=schema, level=request["level"])
        record = {
            "request_id": request["request_id"] if stage != "repeat" else request["request_id"] + "#repeat",
            "candidate_id": request["candidate_id"],
            "status": "completed",
            "model_called": True,
            "repeat_check": stage == "repeat",
            "output_status": parsed["output_status"],
            "predicted_labels": parsed["predicted_labels"],
            "illegal_labels": parsed["illegal_labels"],
            "raw_answer": text,
            "input_tokens": int(encoded["input_ids"].shape[1]),
            "output_tokens": int(new_tokens.shape[0]),
            "truncated": int(new_tokens.shape[0]) >= 256,
            "enable_thinking_argument_accepted": template_ok,
            "generation": "do_sample=false corresponds to temperature 0 and top_p 1",
        }
        append_jsonl(ledger_path, record)
        (output / "responses" / f"{record['request_id'].replace('|', '_')}.txt").write_text(text, encoding="utf-8")
        calls += 1
        done.add(request["request_id"])
    usage = {
        "process_id": stage,
        "gpu_name": torch.cuda.get_device_name(0),
        "framework_allocated_bytes": torch.cuda.max_memory_allocated(),
        "framework_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_memory_bytes": None,
        "device_memory_note": "Round 3 did not sample device memory. This field is not inferred.",
        "scheduler_max_rss_bytes": None,
        "model_calls": calls,
        "completed_requests": len(completed_ids(read_jsonl(ledger_path))),
        "dtype": "bf16",
        "backend": "transformers",
        "measurement": "torch.cuda.max_memory_allocated and max_memory_reserved for this process only",
    }
    write_json(resource_path(output, stage), usage)
    finished = len(completed_ids(read_jsonl(ledger_path))) == 48
    mark_generation(output, {"complete": finished, "stage": stage, "completed_nonrepeat": len(completed_ids(read_jsonl(ledger_path))), "model_calls": calls})


def score_saved(report_dir: Path | None = None) -> None:
    settings = config()
    assert_smoke_config(settings)
    output = Path(settings["output_dir"])
    report = Path(report_dir) if report_dir is not None else output
    report.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_jsonl(output / "attempts.jsonl") if not row.get("repeat_check")]
    if len(rows) != 48:
        raise RuntimeError(f"Expected 48 completed smoke requests, found {len(rows)}.")
    repeats = [row for row in read_jsonl(output / "attempts.jsonl") if row.get("repeat_check")]
    comparisons = []
    for row in repeats:
        base_id = row["request_id"].replace("#repeat", "")
        base = next(item for item in rows if item["request_id"] == base_id)
        comparisons.append({
            "request_id": base_id,
            "same_raw_answer": row["raw_answer"] == base["raw_answer"],
            "same_parsed_labels": row["predicted_labels"] == base["predicted_labels"],
        })
    from schema_study.formal_eval import score_predictions

    summaries = []
    requests = {row["request_id"]: row for row in _load_requests()}
    for dataset in ("biored", "drugprot"):
        schema = load_schema(CODE_ROOT / "configs/schemas" / f"{dataset}.yaml")
        candidates = []
        for line in (SMOKE_ROOT / "samples/smoke_candidates.tsv").read_text(encoding="utf-8").splitlines()[1:]:
            corpus, split, _role, document_id, candidate_id, gold_status, gold_labels = line.split("\t")
            if corpus == dataset and split == "dev":
                candidates.append({"candidate_id": candidate_id, "document_id": document_id, "split": "dev", "gold_labels": [label for label in gold_labels.split("|") if label], "included_in_scoring": True})
        by_id = {row["candidate_id"]: row for row in candidates}
        for level in ("P4", "P2"):
            population_ids = []
            predictions = []
            for request_id, request in requests.items():
                if request["dataset"] != dataset or request["level"] != level:
                    continue
                population_ids.append(request["candidate_id"])
                answer = next(row for row in rows if row["request_id"] == request_id)
                predictions.append({"candidate_id": request["candidate_id"], "document_id": by_id[request["candidate_id"]]["document_id"], "split": "dev", "output_status": answer["output_status"], "predicted_labels": answer["predicted_labels"], "illegal_labels": answer["illegal_labels"]})
            population = [by_id[candidate_id] for candidate_id in population_ids]
            summary = score_predictions(population, predictions, schema, level, allowed_splits={"dev"}, data_version="smoke", input_version="smoke", prediction_version="qwen_smoke", mapping_version=schema.schema_id)
            summaries.append({"dataset": dataset, "level": level, "summary": summary["summary"]})
            if level == "P4":
                rolled = []
                for prediction in predictions:
                    rolled.append({**prediction, "predicted_labels": rollup_predicted_labels(schema, prediction["predicted_labels"], source_level="P4", target_level="P2")})
                rollup = score_predictions(population, rolled, schema, "P2", allowed_splits={"dev"}, data_version="smoke", input_version="smoke", prediction_version="qwen_rollup", mapping_version=schema.schema_id)
                summaries.append({"dataset": dataset, "level": "P2_rollup", "summary": rollup["summary"]})
    nonrepeat_ids = [row["request_id"] for row in rows]
    write_json(report / "resume_check.json", {
        "completed_nonrepeat": len(rows),
        "unique_nonrepeat": len(set(nonrepeat_ids)),
        "repeat_checks": len(repeats),
        "duplicate_nonrepeat": len(nonrepeat_ids) != len(set(nonrepeat_ids)),
        "answers_read_from": str(output / "attempts.jsonl"),
    })
    write_json(report / "parsing_summary.json", {"calls": len(read_jsonl(output / "attempts.jsonl")), "repeats": comparisons, "scores": summaries, "study_phase": "smoke", "included_in_formal_analysis": False})
    mark_score(report, {"complete": True, "source_generation": str(output)})


def settle(report_dir: Path | None = None) -> None:
    """Finish scoring without calling the model when the answer ledger is complete."""
    output = Path(config()["output_dir"])
    status_dir = Path(report_dir) if report_dir is not None else output
    rows = read_jsonl(output / "attempts.jsonl")
    done = completed_ids(rows)
    repeats = [row for row in rows if row.get("repeat_check")]
    if len(done) < 48:
        generate("resume")
        rows = read_jsonl(output / "attempts.jsonl")
        done = completed_ids(rows)
        repeats = [row for row in rows if row.get("repeat_check")]
    mark_generation(status_dir, {
        "complete": len(done) == 48,
        "completed_nonrepeat": len(done),
        "repeat_rows": len(repeats),
        "reconciled_from_existing_ledger": len(done) == 48,
        "model_called": False,
        "answers": str(output / "attempts.jsonl"),
    })
    if len(repeats) < 2 and len(done) >= 48:
        generate("repeat")
    score_saved(status_dir)


if __name__ == "__main__":
    import sys

    command = sys.argv[1]
    destination = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if command == "score":
        try:
            score_saved(destination)
        except Exception as exc:
            target = destination if destination is not None else Path(config()["output_dir"])
            mark_score(target, {"complete": False, "error_type": type(exc).__name__})
            raise
    elif command == "settle":
        try:
            settle(destination)
        except Exception as exc:
            target = destination if destination is not None else Path(config()["output_dir"])
            mark_score(target, {"complete": False, "error_type": type(exc).__name__})
            raise
    else:
        generate(command)
