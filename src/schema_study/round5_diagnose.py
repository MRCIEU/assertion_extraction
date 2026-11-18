"""Compare unpadded, left-padded, and batched greedy generation on a frozen diagnostic set."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
ROUND4 = STORAGE / "studies/schema_granularity_v1/runs/backend_validation/round4_20261115"
OUT_ROOT = STORAGE / "studies/schema_granularity_v1/runs/backend_validation/round5_20261115/diagnose"
MODELS = {
    "8b": {
        "path": STORAGE / "models/Qwen__Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "field": "rendered_qwen8b",
    },
    "32b": {
        "path": STORAGE / "models/Qwen__Qwen3-32B/9216db5781bf21249d130ec9da846c4624c16137",
        "revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "field": "rendered_qwen32b",
    },
}


def _sample(stop: threading.Event, peaks: dict) -> None:
    import subprocess

    while not stop.is_set():
        completed = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], check=False, capture_output=True, text=True)
        if completed.returncode == 0 and completed.stdout.strip():
            used = max(int(part) for part in completed.stdout.split())
            peaks["device_mib"] = max(peaks["device_mib"], used)
            peaks["samples"] += 1
        stop.wait(2.0)


def _content(ids: list[int], mask: list[int]) -> list[int]:
    return [token for token, keep in zip(ids, mask) if keep == 1]


def _positions(mask: list[int]) -> list[int]:
    cumulative = 0
    positions = []
    for keep in mask:
        cumulative += keep
        positions.append(cumulative - 1)
    return [1 if keep == 0 else position for keep, position in zip(mask, positions)]


def main(model_key: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from schema_study.predictions import parse_prediction_text
    from schema_study.schemas import load_schema

    spec = MODELS[model_key]
    ids = (OUT_ROOT.parent / "audit" / f"diagnostic_ids_{'qwen' + model_key}.txt").read_text(encoding="utf-8").split()
    if len(ids) != 32:
        raise RuntimeError(f"Diagnostic list for {model_key} has {len(ids)} ids.")
    requests = {json.loads(line)["request_id"]: json.loads(line) for line in (ROUND4 / "samples/requests.jsonl").open(encoding="utf-8")}
    ordered = [requests[request_id] for request_id in ids]
    schemas = {name: load_schema(CODE / "configs/schemas" / f"{name}.yaml") for name in ("biored", "drugprot")}
    output = OUT_ROOT / model_key
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stopping: this GPU does not support bf16.")
    peaks = {"device_mib": 0, "samples": 0}
    stop = threading.Event()
    thread = threading.Thread(target=_sample, args=(stop, peaks), daemon=True)
    thread.start()
    started = time.perf_counter()
    calls = {"generate": 0, "forward": 0}
    load_seconds = None
    summary = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True)
        default_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(spec["path"], local_files_only=True, torch_dtype=torch.bfloat16)
        model.to("cuda")
        model.eval()
        load_seconds = time.perf_counter() - load_started
        config = model.generation_config
        facts = {
            "padding_side_default": default_side,
            "padding_side_used": tokenizer.padding_side,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "generation_config_pad_token_id": config.pad_token_id,
            "generation_config_eos_token_id": config.eos_token_id,
            "generation_config_do_sample": config.do_sample,
            "call_override": {"do_sample": False, "max_new_tokens": 256, "truncation": False},
            "model_training": model.training,
            "position_ids": "Not supplied. Transformers builds them from attention_mask.cumsum when the mask is passed.",
            "revision": spec["revision"],
        }
        groups = [ordered[:16], ordered[16:]]

        def generate_batch(group: list[dict]) -> list[dict]:
            encoded = tokenizer([row[spec["field"]] for row in group], return_tensors="pt", padding=True, truncation=False)
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            calls["generate"] += len(group)
            with torch.no_grad():
                output_ids = model.generate(**encoded, do_sample=False, max_new_tokens=256)
            rows = []
            for index, request in enumerate(group):
                ids_row = encoded["input_ids"][index].tolist()
                mask_row = encoded["attention_mask"][index].tolist()
                new_tokens = output_ids[index, encoded["input_ids"].shape[1]:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                parsed = parse_prediction_text(text, expected_candidate_id=request["candidate_id"], schema=schemas[request["dataset"]], level=request["predicate_level"])
                rows.append({
                    "request_id": request["request_id"],
                    "input_ids": ids_row,
                    "attention_mask": mask_row,
                    "content_ids": _content(ids_row, mask_row),
                    "text": text,
                    "output_status": parsed["output_status"],
                    "predicted_labels": parsed["predicted_labels"],
                    "truncated": int(new_tokens.shape[0]) >= 256,
                })
            return rows

        warmup = tokenizer(ordered[0][spec["field"]], return_tensors="pt", truncation=False).to("cuda")
        with torch.no_grad():
            model.generate(**warmup, do_sample=False, max_new_tokens=8)
        calls["generate"] += 1
        unpadded = []
        for request in ordered:
            encoded = tokenizer(request[spec["field"]], return_tensors="pt", padding=False, truncation=False)
            ids_row = encoded["input_ids"][0].tolist()
            mask_row = encoded["attention_mask"][0].tolist()
            unpadded.append({"request_id": request["request_id"], "input_ids": ids_row, "attention_mask": mask_row, "content_ids": _content(ids_row, mask_row)})
        by_id = {row["request_id"]: row for row in unpadded}
        widths = []
        for group in groups:
            widths.append(max(len(by_id[row["request_id"]]["content_ids"]) for row in group))
        padded_rows = []
        for group, width in zip(groups, widths):
            for request in group:
                encoded = tokenizer(request[spec["field"]], return_tensors="pt", padding="max_length", max_length=width, truncation=False)
                ids_row = encoded["input_ids"][0].tolist()
                mask_row = encoded["attention_mask"][0].tolist()
                padded_rows.append({
                    "request_id": request["request_id"],
                    "width": width,
                    "input_ids": ids_row,
                    "attention_mask": mask_row,
                    "content_ids": _content(ids_row, mask_row),
                    "positions_head": _positions(mask_row)[:4],
                    "positions_tail": _positions(mask_row)[-4:],
                })
        condition_a = []
        for request in ordered:
            condition_a.extend(generate_batch([request]))
        condition_b = []
        for group, width in zip(groups, widths):
            for request in group:
                encoded = tokenizer(request[spec["field"]], return_tensors="pt", padding="max_length", max_length=width, truncation=False).to("cuda")
                calls["generate"] += 1
                with torch.no_grad():
                    output_ids = model.generate(**encoded, do_sample=False, max_new_tokens=256)
                new_tokens = output_ids[0, encoded["input_ids"].shape[1]:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                parsed = parse_prediction_text(text, expected_candidate_id=request["candidate_id"], schema=schemas[request["dataset"]], level=request["predicate_level"])
                condition_b.append({"request_id": request["request_id"], "text": text, "output_status": parsed["output_status"], "predicted_labels": parsed["predicted_labels"], "truncated": int(new_tokens.shape[0]) >= 256})
        condition_c = []
        for group in groups:
            condition_c.extend(generate_batch(group))
        swapped = [groups[0][:12] + groups[1][12:], groups[1][:12] + groups[0][12:]]
        condition_d = []
        for group in swapped:
            condition_d.extend(generate_batch(group))
        a_map = {row["request_id"]: row for row in condition_a}
        b_map = {row["request_id"]: row for row in condition_b}
        c_map = {row["request_id"]: row for row in condition_c}
        d_map = {row["request_id"]: row for row in condition_d}
        pad_map = {row["request_id"]: row for row in padded_rows}
        comparisons = []
        for request in ordered:
            request_id = request["request_id"]
            base = by_id[request_id]
            padded = pad_map[request_id]
            same_content = base["content_ids"] == padded["content_ids"]
            real_pad_kept = True
            if tokenizer.pad_token_id in base["content_ids"]:
                real_pad_kept = padded["attention_mask"][padded["content_ids"].index(tokenizer.pad_token_id) + (len(padded["input_ids"]) - len(padded["content_ids"]))] == 1
            comparisons.append({
                "request_id": request_id,
                "content_matches_unpadded": same_content,
                "real_pad_id_in_content_kept": real_pad_kept,
                "a_equals_b_text": a_map[request_id]["text"] == b_map[request_id]["text"],
                "a_equals_c_text": a_map[request_id]["text"] == c_map[request_id]["text"],
                "c_equals_d_text": c_map[request_id]["text"] == d_map[request_id]["text"],
                "a_labels": a_map[request_id]["predicted_labels"],
                "b_labels": b_map[request_id]["predicted_labels"],
                "c_labels": c_map[request_id]["predicted_labels"],
                "d_labels": d_map[request_id]["predicted_labels"],
                "a_status": a_map[request_id]["output_status"],
                "b_status": b_map[request_id]["output_status"],
                "c_status": c_map[request_id]["output_status"],
                "d_status": d_map[request_id]["output_status"],
                "a_truncated": a_map[request_id]["truncated"],
                "c_truncated": c_map[request_id]["truncated"],
            })
        margins = []
        interesting = [row["request_id"] for row in comparisons if not row["a_equals_b_text"] or not row["a_equals_c_text"]][:4]
        for request_id in interesting:
            request = requests[request_id]
            encoded = tokenizer(request[spec["field"]], return_tensors="pt", truncation=False).to("cuda")
            calls["forward"] += 1
            with torch.no_grad():
                logits = model(**encoded).logits[0, -1].float()
            top = torch.topk(logits, 5)
            values = top.values.tolist()
            margins.append({
                "request_id": request_id,
                "top_tokens": [tokenizer.decode([int(token)]) for token in top.indices.tolist()],
                "top_ids": [int(token) for token in top.indices.tolist()],
                "margin_top1_top2": values[0] - values[1],
            })
        summary = {
            "facts": facts,
            "load_seconds": load_seconds,
            "widths": widths,
            "swapped_groups": [[row["request_id"] for row in group] for group in swapped],
            "content_mismatch": sum(not row["content_matches_unpadded"] for row in comparisons),
            "a_b_text_mismatch": sum(not row["a_equals_b_text"] for row in comparisons),
            "a_c_text_mismatch": sum(not row["a_equals_c_text"] for row in comparisons),
            "a_c_label_mismatch": sum(row["a_labels"] != row["c_labels"] or row["a_status"] != row["c_status"] for row in comparisons),
            "c_d_label_mismatch": sum(row["c_labels"] != row["d_labels"] or row["c_status"] != row["d_status"] for row in comparisons),
            "rows": comparisons,
            "first_token_margins": margins,
            "calls": calls,
            "included_in_formal_analysis": False,
        }
        (output / "diagnosis.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (output / "batches.json").write_text(json.dumps({
            "order": ids,
            "groups": [[row["request_id"] for row in group] for group in groups],
            "swapped_groups": summary["swapped_groups"],
        }, indent=2) + "\n", encoding="utf-8")
    finally:
        stop.set()
        thread.join(timeout=3)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    (output / f"resource_usage_{job_id}.json").write_text(json.dumps({
        "slurm_job_id": job_id,
        "load_seconds": load_seconds,
        "wall_seconds": time.perf_counter() - started,
        "framework_allocated_bytes": torch.cuda.max_memory_allocated(),
        "framework_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_peak_mib": peaks["device_mib"],
        "device_samples": peaks["samples"],
        "sample_interval_seconds": 2.0,
        "generate_requests": calls["generate"],
        "diagnostic_forwards": calls["forward"],
        "scheduler_max_rss_bytes": None,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"content_mismatch": summary["content_mismatch"], "a_b": summary["a_b_text_mismatch"], "a_c_labels": summary["a_c_label_mismatch"], "c_d_labels": summary["c_d_label_mismatch"], "calls": calls}))


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
