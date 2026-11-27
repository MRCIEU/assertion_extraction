"""Matched vLLM-versus-Transformers comparison. Only the backend is intended to differ."""

from __future__ import annotations

import json
from pathlib import Path


def select_rows(rows: list[dict], count: int = 8) -> list[dict]:
    """Fixed positions. Gold labels are not read and are not a selection criterion."""
    if not rows:
        return []
    if len(rows) <= count:
        return list(rows)
    indexes = sorted({int(position * (len(rows) - 1) / (count - 1)) for position in range(count)})
    return [rows[index] for index in indexes]


def label_agreement(reference: list[dict], generated: list[dict]) -> dict:
    """Agreement on the paired rows. High overall agreement is not reported as equivalence by itself."""
    paired = min(len(reference), len(generated))
    same = validity = positive = 0
    for left, right in zip(reference, generated):
        if tuple(left.get("predicted_labels") or []) == tuple(right.get("predicted_labels") or []):
            same += 1
        if (left.get("status") == "valid") == (right.get("status") == "valid"):
            validity += 1
        if bool(left.get("predicted_labels")) == bool(right.get("predicted_labels")):
            positive += 1
    return {
        "rows": paired,
        "label_agreement": same / paired if paired else None,
        "validity_agreement": validity / paired if paired else None,
        "positive_decision_agreement": positive / paired if paired else None,
        "token_ids_retained_by_vllm": False,
    }


def compare_backends(task: dict, workdir: Path) -> dict:
    """Re-generate a fixed base-model sample with Transformers and compare it with the retained vLLM ledger."""
    from schema_study.production_plan import task_directory
    from schema_study.production_runtime import publish_scientific_status

    workdir.mkdir(parents=True, exist_ok=True)
    from schema_study.production_plan import _task

    identity = _task("zero_shot", dataset="biored", entity="E0", predicate="P0", model="qwen8b", paradigm="zero_shot", seed=None)
    ledger = Path(task_directory(identity)) / "predictions.jsonl"
    if not ledger.is_file():
        report = {"limitation": "no_retained_qwen8b_ledger", "agreement": None}
        (workdir / "backend_control.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        publish_scientific_status(workdir, {"scientific": "diagnostic_recorded", **report})
        return {"completed": True, "status": "diagnostic_recorded", "report": report}
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen = select_rows(rows)
    generated = _transformers_rows(chosen)
    report = {"ledger": str(ledger), "agreement": label_agreement(chosen, generated), "limitation": "original_vllm_token_ids_were_not_stored"}
    (workdir / "backend_control.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    publish_scientific_status(workdir, {"scientific": "diagnostic_recorded", "rows": report["agreement"]["rows"]})
    return {"completed": True, "status": "diagnostic_recorded", "report": report}


def _transformers_rows(rows: list[dict]) -> list[dict]:
    import torch

    from schema_study.paths import get_paths
    from schema_study.production_lora import load_causal_lm
    from schema_study.production_runtime import interpret_generation
    from schema_study.schemas import default_schema_path, load_schema

    schema = load_schema(default_schema_path(get_paths().code_root, "biored"))
    model, tokenizer = load_causal_lm("qwen8b")
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    produced = []
    for row in rows:
        encoded = tokenizer(row["rendered"], return_tensors="pt")
        if torch.cuda.is_available():
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
        with torch.no_grad():
            output = model.generate(**encoded, max_new_tokens=64, do_sample=False)
        text = tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = interpret_generation(text, row["candidate_id"], schema, row.get("entity") or "E0")
        produced.append({"predicted_labels": parsed.get("predicted_labels"), "status": parsed.get("output_status")})
    return produced
