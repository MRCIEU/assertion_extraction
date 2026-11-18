"""Separate padding-width effects from batch-membership effects on the frozen diagnostic set."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
ROUND4 = STORAGE / "studies/schema_granularity_v1/runs/backend_validation/round4_20261115"
DIAG = STORAGE / "studies/schema_granularity_v1/runs/backend_validation/round5_20261115/diagnose"
MODELS = {
    "8b": (STORAGE / "models/Qwen__Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218", "rendered_qwen8b"),
    "32b": (STORAGE / "models/Qwen__Qwen3-32B/9216db5781bf21249d130ec9da846c4624c16137", "rendered_qwen32b"),
}


def _decode_new(tokenizer, output_ids, prompt_width: int):
    new_tokens = output_ids[prompt_width:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), [int(token) for token in new_tokens]


def main(model_key: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from schema_study.predictions import parse_prediction_text
    from schema_study.schemas import load_schema

    path, field = MODELS[model_key]
    diagnosis = json.loads((DIAG / model_key / "diagnosis.json").read_text(encoding="utf-8"))
    batches = json.loads((DIAG / model_key / "batches.json").read_text(encoding="utf-8"))
    requests = {json.loads(line)["request_id"]: json.loads(line) for line in (ROUND4 / "samples/requests.jsonl").open(encoding="utf-8")}
    schemas = {name: load_schema(CODE / "configs/schemas" / f"{name}.yaml") for name in ("biored", "drugprot")}
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    calls = 0
    width = max(diagnosis["widths"])
    margins = []
    for row in diagnosis["rows"]:
        if row["a_equals_b_text"]:
            continue
        request = requests[row["request_id"]]
        sequences = []
        for padding, length in ((False, None), ("max_length", width)):
            kwargs = {"return_tensors": "pt", "truncation": False}
            if padding:
                kwargs.update(padding=padding, max_length=length)
            else:
                kwargs["padding"] = False
            encoded = tokenizer(request[field], **kwargs).to("cuda")
            calls += 1
            with torch.no_grad():
                output_ids = model.generate(**encoded, do_sample=False, max_new_tokens=256)
            text, tokens = _decode_new(tokenizer, output_ids[0].tolist(), encoded["input_ids"].shape[1])
            sequences.append((text, tokens, encoded))
        (text_a, tokens_a, encoded_a), (text_b, tokens_b, encoded_b) = sequences
        limit = min(len(tokens_a), len(tokens_b))
        split = next((index for index in range(limit) if tokens_a[index] != tokens_b[index]), limit)
        prefix = tokens_a[:split]
        def margin(encoded, prefix_tokens):
            nonlocal calls
            ids = torch.cat([encoded["input_ids"][0], torch.tensor(prefix_tokens, device="cuda")], dim=0).unsqueeze(0)
            mask = torch.cat([encoded["attention_mask"][0], torch.ones(len(prefix_tokens), dtype=torch.long, device="cuda")], dim=0).unsqueeze(0)
            calls += 1
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=mask).logits[0, -1].float()
            top = torch.topk(logits, 2)
            values = top.values.tolist()
            return {"ids": [int(token) for token in top.indices.tolist()], "margin": values[0] - values[1]}
        margins.append({
            "request_id": row["request_id"],
            "first_different_output_token": split,
            "unpadded_margin": margin(encoded_a, prefix),
            "padded_margin": margin(encoded_b, prefix),
            "texts_still_differ": text_a != text_b,
        })

    def run_groups(groups: list[list[str]]) -> dict[str, dict]:
        nonlocal calls
        found = {}
        for group_ids in groups:
            group = [requests[request_id] for request_id in group_ids]
            encoded = tokenizer([row[field] for row in group], return_tensors="pt", padding="max_length", max_length=width, truncation=False).to("cuda")
            calls += len(group)
            with torch.no_grad():
                output_ids = model.generate(**encoded, do_sample=False, max_new_tokens=256)
            for index, request in enumerate(group):
                text, _tokens = _decode_new(tokenizer, output_ids[index].tolist(), encoded["input_ids"].shape[1])
                parsed = parse_prediction_text(text, expected_candidate_id=request["candidate_id"], schema=schemas[request["dataset"]], level=request["predicate_level"])
                found[request["request_id"]] = {"text": text, "labels": parsed["predicted_labels"], "status": parsed["output_status"]}
        return found

    original = run_groups(batches["groups"])
    swapped = run_groups(batches["swapped_groups"])
    membership = []
    for request_id in batches["order"]:
        membership.append({
            "request_id": request_id,
            "same_text": original[request_id]["text"] == swapped[request_id]["text"],
            "same_labels": original[request_id]["labels"] == swapped[request_id]["labels"] and original[request_id]["status"] == swapped[request_id]["status"],
            "original_labels": original[request_id]["labels"],
            "swapped_labels": swapped[request_id]["labels"],
        })
    output = DIAG / model_key / "followup.json"
    output.write_text(json.dumps({
        "fixed_width": width,
        "padding_divergence": margins,
        "membership_same_width_label_mismatches": sum(not row["same_labels"] for row in membership),
        "membership_rows": membership,
        "generate_requests": calls,
        "included_in_formal_analysis": False,
    }, indent=2) + "\n", encoding="utf-8")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    (DIAG / model_key / f"resource_usage_{job_id}_followup.json").write_text(json.dumps({
        "slurm_job_id": job_id,
        "generate_requests": calls,
        "framework_allocated_bytes": torch.cuda.max_memory_allocated(),
        "framework_reserved_bytes": torch.cuda.max_memory_reserved(),
        "wall_note": "Device samples were not repeated in this follow-up. Scheduler MaxRSS is filled from sacct.",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"divergences": len(margins), "membership_label_mismatches": sum(not row["same_labels"] for row in membership), "calls": calls}))


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
