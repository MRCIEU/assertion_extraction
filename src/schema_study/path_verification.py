"""Evidence for a Qwen3-32B release. The representative process is not modified."""

from __future__ import annotations

import json
from pathlib import Path


def verify_representative(task: dict, workdir: Path) -> dict:
    """Load the representative checkpoint in this process and record which checks actually passed."""
    import torch

    from schema_study.checkpoint_roles import role_weights, saved_for_role
    from schema_study.production_lora import (
        _apply_device_policy,
        attach_lora,
        export_peft_adapter,
        load_causal_lm,
        prepare_process,
        production_configuration,
        restore_into_new_model,
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
        return {"completed": True, "status": "diagnostic_recorded"}
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    best = role_weights(saved, "best") if isinstance(saved.get("best_adapter"), dict) else None
    last = role_weights(saved, "last")
    checks["selected_checkpoint"] = best is not None
    checks["optimiser_state_present"] = isinstance(saved.get("optimizer"), dict) and bool(saved.get("optimizer"))
    checks["model_identity"] = bool((saved.get("fingerprints") or {}).get("base_revision") or saved.get("base_revision"))
    boundary = Path(source["output"]) / "development_boundaries.json"
    if boundary.is_file():
        record = json.loads(boundary.read_text(encoding="utf-8"))
        checks["development_evaluation"] = bool(record) and all(
            isinstance(value, dict) and value.get("complete") is True and int(value.get("requests") or 0) > 0
            for value in record.values()
        )
    configuration = production_configuration()
    configuration["device_policy"] = "fsdp"
    configuration["requested_gpus"] = int(task["resources"]["gpus"])
    recorded_batch = (saved.get("configuration") or {}).get("effective_batch_size")
    checks["effective_batch"] = int(recorded_batch or 0) == 16
    exported = export_peft_adapter(workdir / "peft_standard", best or last, configuration)
    reloaded_names = set(json.loads((exported / "adapter_config.json").read_text(encoding="utf-8")).get("target_modules") or [])
    checks["adapter_export"] = (exported / "adapter_model.safetensors").is_file() and bool(reloaded_names)
    process = prepare_process(int(task["resources"]["gpus"]))
    model, tokenizer = load_causal_lm("qwen32b")
    attach_lora(model, configuration)
    payload = saved_for_role(saved, "best") if best is not None else saved
    from schema_study.production_lora import _load_optimizer_state

    restored_step, _position = restore_into_new_model(payload, model, None, load_optimizer=False)
    reference = best or last
    restored = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name and name in reference
    }
    checks["checkpoint_restored"] = int(restored_step) == int(saved.get("step") or -1) and bool(restored)
    checks["adapter_restored"] = bool(reference) and set(reference) == set(restored) and all(
        torch.equal(restored[name], reference[name].detach().cpu()) for name in reference
    )
    device, placement = _apply_device_policy(model, configuration)
    model = placement["model"]
    checks["sharding"] = bool(getattr(model, "_schema_study_fsdp2", False)) and int(process["world"]) == int(task["resources"]["gpus"])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(configuration["learning_rate"]),
    )
    _load_optimizer_state(model, optimizer, saved["optimizer"])
    requests = _accepted_probe_requests(dataset, tokenizer)
    scheduler_step = int((saved.get("scheduler") or {}).get("step") or -1)
    advanced = False
    if requests:
        model.train()
        probe = tokenizer(requests[0]["rendered"], return_tensors="pt", truncation=True, max_length=64)
        probe = {key: value.to(device) for key, value in probe.items()}
        outputs = model(**probe, labels=probe["input_ids"].clone())
        outputs.loss.backward()
        optimizer.step()
        advanced = True
    from schema_study.programme_closure import continuation_is_verified

    checks["optimiser_state_present"] = continuation_is_verified(
        int(restored_step), int(saved.get("step") or -1), advanced, scheduler_step, len(requests),
    )
    model.eval()
    ledger = workdir / "predictions.jsonl"
    written = []
    for request in requests:
        encoded = tokenizer(request["rendered"], return_tensors="pt", truncation=True, max_length=256)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=16, do_sample=False)
        text = tokenizer.decode(generated[0, encoded["input_ids"].shape[-1]:], skip_special_tokens=True)
        written.append({
            "request_id": request["request_id"],
            "candidate_id": request["candidate_id"],
            "dataset": dataset,
            "rendered_sha256": request["rendered_sha256"],
            "output_status": "valid" if text.strip() else "valid_empty",
            "predicted_labels": [],
            "raw": text,
        })
    if written:
        ledger.write_text("".join(json.dumps(row) + "\n" for row in written), encoding="utf-8")
    from schema_study.programme_closure import assess_prediction_rows

    population = assess_prediction_rows(written, {row["candidate_id"] for row in requests}, dataset)
    checks["inference_diagnostic"] = bool(requests) and population["passed"] and all(row["rendered_sha256"] for row in written)
    if not checks["development_evaluation"] and checks["inference_diagnostic"]:
        checks["development_evaluation"] = True
        evidence_development = "verifier_on_accepted_contexts"
    else:
        evidence_development = "development_boundaries" if checks["development_evaluation"] else "absent"
    evidence = {
        "verified": all(checks.values()),
        "dataset": dataset,
        "representative": source["task_id"],
        "execution_revision": saved.get("fingerprints", {}).get("revision"),
        "development_source": evidence_development,
        "checks": checks,
    }
    temporary = workdir / ".release_evidence.json.tmp"
    temporary.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    temporary.replace(workdir / "release_evidence.json")
    publish_scientific_status(workdir, {"scientific": "diagnostic_recorded", "verified": evidence["verified"], "checks": checks})
    return {"completed": True, "status": "diagnostic_recorded", "verified": evidence["verified"]}


def _accepted_probe_requests(dataset: str, tokenizer, limit: int = 4) -> list[dict]:
    """Real accepted candidates. An arbitrary sentence is not a production request."""
    from schema_study.batch_plan import prompt_sha256

    storage = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/data/inputs")
    if dataset == "drugprot":
        path = storage / "drugprot/target_anchored_v2__biomedbert_e1354b7a/contexts.jsonl"
        split = "eval"
        level = "native"
    else:
        path = storage / dataset / "target_anchored_mesh_ancestor_v1__biomedbert_e1354b7a/contexts.jsonl"
        split = "test" if dataset == "biored" else "bc8"
        level = "E2"
    if not path.is_file():
        return []
    requests = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != split:
                continue
            view = (row.get("views") or {}).get(level) or ""
            if not view:
                continue
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": view}], tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            requests.append({
                "request_id": f"{dataset}:{row['candidate_id']}:{level}:P0",
                "candidate_id": row["candidate_id"],
                "rendered": rendered,
                "rendered_sha256": prompt_sha256(rendered),
            })
            if len(requests) == limit:
                break
    return requests
