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


def continuation_advanced(start: int, end: int | None, losses: list) -> bool:
    """A missing loss record is not a finite loss. The step must move forward inside the horizon."""
    import math

    if end is None or not losses:
        return False
    return int(end) > int(start) and all(math.isfinite(value) for value in losses)


def adapter_names_match(expected_names, loaded_names) -> bool:
    """Inference-mode adapters are frozen. Gradient flags are not the parameter inventory."""
    expected = {name for name in expected_names if "lora_" in name}
    loaded = {name for name in loaded_names if "lora_" in name}
    return bool(expected) and expected == loaded


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
    from schema_study.production_dispatch import _supervised_batches

    start = int(rolling["step"])
    horizon = int((rolling.get("scheduler") or {}).get("total_steps") or 0)
    isolated = workdir / "continuation"
    isolated.mkdir(parents=True, exist_ok=True)
    prior_path = workdir / "release_evidence.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.is_file() else {}
    prior_checks = prior.get("checks") or {}
    continuation_current = (
        prior.get("source_mtime_ns") == before
        and prior_checks.get("optimiser_state_present") is True
        and prior_checks.get("checkpoint_restored") is True
        and prior_checks.get("sharding") is True
    )
    if continuation_current:
        result = {"steps": prior.get("continued_step"), "losses": prior.get("losses") or [], "position": prior.get("position"), "checkpoint": prior.get("checkpoint") or {}}
        checks["optimiser_state_present"] = True
        checks["checkpoint_restored"] = True
        checks["sharding"] = True
    elif horizon <= start:
        result = {"steps": start, "losses": [], "position": rolling.get("position"), "checkpoint": {}}
    else:
        model, tokenizer = load_causal_lm("qwen32b")
        result = fit(
            model,
            _supervised_batches(source, tokenizer, configuration),
            configuration=configuration,
            steps=min(start + 2, horizon),
            total_steps=horizon,
            resume=rolling,
            output_dir=isolated,
            fingerprints=rolling.get("fingerprints") or {},
            fresh=model,
            stop_after=start + 1,
        )
        trained = result.get("model") or model
        placement = (result.get("checkpoint") or {}).get("placement") or {}
        checks["optimiser_state_present"] = continuation_advanced(start, result.get("steps"), result.get("losses") or [])
        checks["checkpoint_restored"] = (isolated / "adapter_checkpoint.pt").is_file() and checkpoint.stat().st_mtime_ns == before
        checks["sharding"] = (
            bool(getattr(trained, "_schema_study_fsdp2", False))
            and int(process["world"]) == int(task["resources"]["gpus"])
            and placement.get("parameter_sharding") is True
            and placement.get("moved_unsharded_before_wrap") is False
        )
        del model, trained
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rank = int(process["rank"])
    world = int(process["world"])
    if world > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if rank != 0:
        return {"completed": True, "status": "diagnostic_recorded", "verified": False, "rank": rank}
    requests, request_provenance = _production_requests(dataset, entity, limit=4)
    if best is not None:
        exported = export_peft_adapter(workdir / "peft_standard", best, configuration)
        checks["adapter_export"] = (exported / "adapter_model.safetensors").is_file()
        if requests:
            from schema_study.learning_conditions import HISTORICAL_PREDICATES
            from schema_study.production_dispatch import GenerationSession, _parse_generated, _write_predictions
            from schema_study.production_dispatch import _generation_schema
            from schema_study.production_runtime import prediction_record

            historical = source.get("schema_predicate") or HISTORICAL_PREDICATES.get(source.get("predicate"), source.get("predicate"))
            probe = {
                **source,
                "family": "adapter_inference",
                "adapter_path": str(exported),
                "schema_predicate": historical,
                "resources": {"gpus": 1},
            }
            session = GenerationSession(probe)
            generated = session.generate_batch(requests, batch_size=1)
            loaded, _tokenizer = session._transformers()
            loaded_names = [name for name, _parameter in loaded.named_parameters()]
            expected_names = [name for name in best if "lora_" in name]
            loaded_pairs = {
                peft_key(name): parameter.detach().cpu()
                for name, parameter in loaded.named_parameters()
                if "lora_" in name
            }
            expected_pairs = {peft_key(name): tensor.detach().cpu() for name, tensor in best.items() if "lora_" in name}
            config_path = exported / "adapter_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
            active = getattr(loaded, "active_adapters", None) or getattr(loaded, "active_adapter", None)
            checks["adapter_restored"] = (
                adapter_names_match(expected_names, loaded_names)
                and bool(expected_pairs)
                and set(expected_pairs) == set(loaded_pairs)
                and all(torch.equal(loaded_pairs[name], tensor) for name, tensor in expected_pairs.items())
                and int(config.get("r") or 0) == int(configuration["rank"])
                and int(config.get("lora_alpha") or 0) == int(configuration["alpha"])
                and str(config.get("base_model_name_or_path") or "") == str(configuration["base_model_path"])
                and active not in {None, "", []}
            )
            schema = _generation_schema(probe)
            published = {}
            legal = True
            for request, generated_row in zip(requests, generated):
                shaped = _parse_generated(probe, request, generated_row, schema)
                status = (shaped.get("parsed") or {}).get("ledger_status")
                if status not in {"valid", "valid_empty", "invalid"}:
                    legal = False
                    status = "invalid"
                record = prediction_record(request, shaped, probe, session.specification, status)
                record["output_status"] = status
                record["dataset"] = dataset
                published[request["request_id"]] = record
            _write_predictions(workdir / "predictions.jsonl", published)
            checks["inference_diagnostic"] = legal and bool(published) and all(row.get("rendered_sha256") for row in requests) and request_provenance != "missing"
    evidence = {
        "verified": all(checks.values()),
        "dataset": dataset,
        "representative": source["task_id"],
        "execution_revision": (saved.get("fingerprints") or {}).get("revision"),
        "continuation_directory": str(isolated),
        "source_mtime_ns": before,
        "continued_step": result.get("steps"),
        "losses": result.get("losses") or [],
        "position": result.get("position"),
        "request_provenance": request_provenance,
        "development_source": "development_boundaries" if checks["development_evaluation"] else "not_scheduled",
        "checks": checks,
    }
    import os

    temporary = workdir / f".release_evidence.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    temporary.replace(workdir / "release_evidence.json")
    complete = all(checks.values())
    publish_scientific_status(workdir, {
        "scientific": "diagnostic_recorded" if complete else "valid_continuation_required",
        "verified": complete,
        "checks": checks,
        "steps": result.get("steps"),
        "checkpoint_fingerprint": str(before),
    })
    return {"completed": True, "status": "diagnostic_recorded" if complete else "valid_continuation_required", "verified": complete}


def _production_requests(dataset: str, entity: str, predicate: str = "P0", limit: int = 4) -> tuple[list[dict], str]:
    """Accepted production requests only. A corrected corpus does not fall back to a historical shard."""
    root = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/runs/formal/formal_matched_e0_e7/generation_inputs")
    if dataset == "drugprot":
        directory = root / "zero_shot" / dataset / entity / predicate
        provenance = "native"
    else:
        directory = root / "mesh_ancestor_v1" / "zero_shot" / dataset / entity / predicate
        provenance = "mesh_ancestor_v1"
    rows = []
    if not directory.is_dir():
        return rows, "missing"
    for shard in sorted(directory.glob("shard_*.jsonl")):
        for line in shard.open(encoding="utf-8"):
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) == limit:
                return rows, provenance
    return rows, provenance if rows else "missing"
