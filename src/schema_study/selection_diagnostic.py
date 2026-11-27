"""Rescore a historical best checkpoint and the last checkpoint with the corrected decoder.

This does not run optimiser steps and it does not read final test labels.
"""

from __future__ import annotations

import json
from pathlib import Path


def _weights(path: Path) -> tuple[dict, dict]:
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=False)
    weights = saved.get("best_adapter") or saved.get("adapter") or {}
    return saved, weights


def _same(left: dict, right: dict) -> bool:
    import torch

    if set(left) != set(right):
        return False
    return all(torch.equal(left[name], right[name]) for name in left)


def _agreement(left_path: Path, right_path: Path) -> dict:
    def read(path: Path) -> dict:
        found = {}
        if not path.is_file():
            return found
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            found[tuple(row.get("request_key") or [])] = tuple(row.get("predicted") or [])
        return found

    left, right = read(left_path), read(right_path)
    keys = set(left) & set(right)
    same = sum(1 for key in keys if left[key] == right[key])
    return {"shared_requests": len(keys), "identical_predictions": same}


def score_pair(task: dict, workdir: Path) -> dict:
    """Score the recoverable checkpoints. A missing file is a recorded limitation."""
    import torch

    from schema_study.production_dispatch import _lora_development_score, _metadata_rows
    from schema_study.production_lora import load_causal_lm, restore_into_new_model
    from schema_study.production_plan import execution_manifest
    from schema_study.production_runtime import publish_scientific_status, select_population

    source = next(item for item in execution_manifest() if item["task_id"] == task["source_task"])
    directory = Path(source["output"])
    best_path = directory / "best" / "adapter_checkpoint.pt"
    last_path = directory / "adapter_checkpoint.pt"
    report = {
        "source_task": task["source_task"],
        "best_exists": best_path.is_file(),
        "last_exists": last_path.is_file(),
        "decoder": "left_pad_slice_at_padded_width",
    }
    workdir.mkdir(parents=True, exist_ok=True)
    if not best_path.is_file() or not last_path.is_file():
        report["limitation"] = "checkpoint_unavailable"
        (workdir / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        publish_scientific_status(workdir, {"scientific": "inference_completed", "limitation": report["limitation"], "predictions_usable": False})
        return {"completed": True, "status": "inference_completed", "limitation": report["limitation"]}
    best_saved, best_weights = _weights(best_path)
    last_saved, last_weights = _weights(last_path)
    report["checkpoints_identical"] = _same(best_weights, last_weights)
    report["best_step"] = best_saved.get("step")
    report["last_step"] = last_saved.get("step")
    population = select_population(source, _metadata_rows(source))
    development = population.get("development") or []
    if any(row.get("split") == "test" for row in development):
        raise RuntimeError("A test row entered the selection diagnostic.")
    model, tokenizer = load_causal_lm(task["model"])
    if torch.cuda.is_available():
        model.to("cuda")
    scores = {}
    plan = [("best", best_path, best_saved)]
    if not report["checkpoints_identical"]:
        plan.append(("last", last_path, last_saved))
    for name, _path, saved in plan:
        restore_into_new_model(saved, model, None, load_optimizer=False)
        ledger = workdir / f"{name}_development_predictions.jsonl"
        metric = _lora_development_score(source, development, tokenizer, ledger)(model, int(saved.get("step") or 0))
        scores[name] = metric
    if report["checkpoints_identical"]:
        scores["last"] = scores["best"]
    report["corrected_development_micro_f1"] = scores
    report["agreement"] = _agreement(workdir / "best_development_predictions.jsonl", workdir / "last_development_predictions.jsonl")
    report["selected_checkpoint"] = "best" if scores["best"] >= scores["last"] else "last"
    report["complete_historical_trajectory"] = False
    (workdir / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    publish_scientific_status(workdir, {
        "scientific": "inference_completed",
        "predictions_usable": True,
        "selected_checkpoint": report["selected_checkpoint"],
        "checkpoints_identical": report["checkpoints_identical"],
    })
    return {"completed": True, "status": "inference_completed", "report": report}
