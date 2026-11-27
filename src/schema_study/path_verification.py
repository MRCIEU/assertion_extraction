"""Qwen3-32B release evidence. Continuation and selected-model inference stay separate."""

from __future__ import annotations

import json
from pathlib import Path


def continuation_package(saved: dict) -> dict:
    """The rolling checkpoint. Its optimiser belongs to these weights, not to an earlier selected adapter."""
    if not isinstance(saved.get("optimizer"), dict) or not saved.get("optimizer"):
        raise ValueError("The continuation checkpoint has no optimiser state.")
    scheduler_step = int((saved.get("scheduler") or {}).get("step", -1))
    if scheduler_step != int(saved.get("step", -2)):
        raise ValueError("The scheduler step does not match the checkpoint step.")
    return saved


def selected_adapter(saved: dict) -> dict | None:
    """Selected weights only. A later rolling adapter is not a substitute."""
    best = saved.get("best_adapter")
    if not isinstance(best, dict) or not best:
        return None
    return best


def verify_representative(task: dict, workdir: Path) -> dict:
    """Continue the rolling checkpoint, then score selected weights through the production inference path."""
    import torch

    from schema_study.production_lora import (
        export_peft_adapter,
        fit,
        load_causal_lm,
        peft_key,
        prepare_process,
        production_configuration,
    )
    from schema_study.production_plan import execution_manifest
    from schema_study.production_runtime import publish_scientific_status

    workdir.mkdir(parents=True, exist_ok=True)
    dataset = task["dataset"]
    entity = "native" if dataset == "drugprot" else "E0"
    source = next(
        item for item in execution_manifest()
        if item.get("family") == "principal_training"
        and item.get("model") == "qwen32b"
        and item.get("dataset") == dataset
        and item.get("entity") == entity
        and item.get("predicate") == "P0"
        and int(item.get("seed") or 0) == 13
        and not item.get("representation_version")
    )
    checkpoint = Path(source["output"]) / "adapter_checkpoint.pt"
    checks = {name: False for name in (
        "model_identity", "sharding", "effective_batch", "checkpoint_restored",
        "optimiser_state_present", "development_evaluation", "selected_checkpoint",
        "adapter_export", "adapter_restored", "inference_diagnostic",
    )}
    if not checkpoint.is_file():
        publish_scientific_status(workdir, {"scientific": "diagnostic_recorded", "checks": checks, "reason": "checkpoint_absent"})
        return {"completed": True, "status": "diagnostic_recorded", "verified": False}
    before = checkpoint.stat().st_mtime_ns
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    rolling = continuation_package(saved)
    best = selected_adapter(saved)
    checks["selected_checkpoint"] = best is not None
    checks["model_identity"] = bool((saved.get("fingerprints") or {}).get("base_revision") or saved.get("base_revision"))
    recorded_batch = (saved.get("configuration") or {}).get("effective_batch_size")
    checks["effective_batch"] = int(recorded_batch or 0) == 16
    boundary = Path(source["output"]) / "development_boundaries.json"
    if boundary.is_file():
        record = json.loads(boundary.read_text(encoding="utf-8"))
        checks["development_evaluation"] = bool(record) and all(
            isinstance(value, dict) and value.get("complete") is True and int(value.get("requests") or 0) > 0
            for value in record.values()
        )
    from schema_study.production_assets import GENERATIVE

    configuration = production_configuration()
    configuration.update({
        "base_revision": source.get("model_revision"),
        "base_model_path": str(GENERATIVE["qwen32b"]["path"]),
        "world_size": int(task["resources"]["gpus"]),
        "requested_gpus": int(task["resources"]["gpus"]),
        "device_policy": "fsdp",
        "placement": "fsdp_full_shard",
        "require_fingerprints": True,
        "micro_batch_size": 1,
        "retain_consumed_prefixes": False,
    })
    process = prepare_process(int(task["resources"]["gpus"]))
    model, tokenizer = load_causal_lm("qwen32b")
    from schema_study.production_dispatch import _development_from_candidates, _lora_development_score, _supervised_batches

    start = int(rolling["step"])
    isolated = workdir / "continuation"
    isolated.mkdir(parents=True, exist_ok=True)
    result = fit(
        model,
        _supervised_batches(source, tokenizer, configuration),
        configuration=configuration,
        steps=start + 2,
        total_steps=int((rolling.get("scheduler") or {}).get("total_steps") or start + 2),
        resume=rolling,
        output_dir=isolated,
        fingerprints=rolling.get("fingerprints") or {},
        fresh=model,
        stop_after=start + 1,
    )
    advanced = int(result.get("steps") or 0) > start and all(
        __import__("math").isfinite(value) for value in (result.get("losses") or [0.0])
    )
    checks["optimiser_state_present"] = advanced
    checks["checkpoint_restored"] = (isolated / "adapter_checkpoint.pt").is_file() and checkpoint.stat().st_mtime_ns == before
    checks["sharding"] = int(process["world"]) == int(task["resources"]["gpus"])
    if not checks["development_evaluation"]:
        development = [row for row in _development_from_candidates(source) if row.get("split") != "test"][:4]
        if development and all(row.get("split") in {"dev", "development"} for row in development):
            score = _lora_development_score(source, development, tokenizer, isolated / "development_predictions.jsonl")
            metric = float(score(model, int(result["steps"])))
            checks["development_evaluation"] = metric >= 0 and (isolated / "development_predictions.jsonl").is_file()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if best is not None:
        exported = export_peft_adapter(workdir / "peft_standard", best, configuration)
        checks["adapter_export"] = (exported / "adapter_model.safetensors").is_file()
        requests = _production_requests(dataset, entity, limit=4)
        if requests:
            from schema_study.production_dispatch import GenerationSession
            from schema_study.production_runtime import interpret_generation
            from schema_study.schemas import default_schema_path, load_schema

            probe = {
                **source,
                "family": "adapter_inference",
                "adapter_path": str(exported),
                "resources": task.get("resources") or {},
            }
            session = GenerationSession(probe)
            generated = session.generate_batch(requests, batch_size=1)
            loaded, _tokenizer = session._transformers()
            loaded_pairs = {
                peft_key(name): parameter.detach().cpu()
                for name, parameter in loaded.named_parameters()
                if "lora_" in name and parameter.requires_grad
            }
            checks["adapter_restored"] = bool(best) and all(
                peft_key(name) in loaded_pairs and torch.equal(loaded_pairs[peft_key(name)], tensor.detach().cpu())
                for name, tensor in best.items() if "lora_" in name
            )
            from schema_study.paths import get_paths

            schema = load_schema(default_schema_path(get_paths().code_root, "drugprot" if dataset == "drugprot" else "biored"))
            written = []
            for request, row in zip(requests, generated):
                interpreted = interpret_generation(row.get("raw") or row.get("text"), request["candidate_id"], schema, source.get("predicate") or "P0")
                written.append({
                    "request_id": request["request_id"],
                    "candidate_id": request["candidate_id"],
                    "dataset": dataset,
                    "rendered_sha256": request.get("rendered_sha256"),
                    "output_status": interpreted["ledger_status"],
                    "predicted_labels": interpreted.get("predicted_labels") or [],
                    "raw": interpreted.get("raw"),
                })
            ledger = workdir / "predictions.jsonl"
            ledger.write_text("".join(json.dumps(row) + "\n" for row in written), encoding="utf-8")
            from schema_study.programme_closure import assess_prediction_rows, publish_ledger_acceptance

            population = assess_prediction_rows(written, {row["candidate_id"] for row in requests}, dataset)
            legal = all(row["output_status"] in {"valid", "valid_empty", "invalid"} for row in written)
            checks["inference_diagnostic"] = population["passed"] and legal and all(row.get("rendered_sha256") for row in written)
            publish_ledger_acceptance(workdir, dataset, written, {row["candidate_id"] for row in requests})
    evidence = {
        "verified": all(checks.values()),
        "dataset": dataset,
        "representative": source["task_id"],
        "execution_revision": (saved.get("fingerprints") or {}).get("revision"),
        "continuation_directory": str(isolated),
        "checks": checks,
    }
    temporary = workdir / ".release_evidence.json.tmp"
    temporary.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    temporary.replace(workdir / "release_evidence.json")
    publish_scientific_status(workdir, {"scientific": "diagnostic_recorded", "verified": evidence["verified"], "checks": checks})
    return {"completed": True, "status": "diagnostic_recorded", "verified": evidence["verified"]}


def _production_requests(dataset: str, entity: str, predicate: str = "P0", limit: int = 4) -> list[dict]:
    """A bounded prefix of the accepted generation shards. Context text is not this prompt."""
    root = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/runs/formal/formal_matched_e0_e7/generation_inputs")
    directories = [
        root / "mesh_ancestor_v1" / "zero_shot" / dataset / entity / predicate,
        root / "zero_shot" / dataset / entity / predicate,
    ]
    rows = []
    for directory in directories:
        if not directory.is_dir():
            continue
        for shard in sorted(directory.glob("shard_*.jsonl")):
            for line in shard.open(encoding="utf-8"):
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if len(rows) == limit:
                    return rows
        if rows:
            return rows
    return rows
