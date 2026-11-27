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
    model.eval()
    generated_requests = 0
    for prompt in ("No relation.", "The gene inhibits the disease."):
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=4, do_sample=False)
        if generated.shape[-1] > encoded["input_ids"].shape[-1]:
            generated_requests += 1
    checks["inference_diagnostic"] = generated_requests == 2
    evidence = {
        "verified": all(checks.values()),
        "dataset": dataset,
        "representative": source["task_id"],
        "execution_revision": saved.get("fingerprints", {}).get("revision"),
        "checks": checks,
    }
    temporary = workdir / ".release_evidence.json.tmp"
    temporary.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    temporary.replace(workdir / "release_evidence.json")
    publish_scientific_status(workdir, {"scientific": "diagnostic_recorded", "verified": evidence["verified"], "checks": checks})
    return {"completed": True, "status": "diagnostic_recorded", "verified": evidence["verified"]}
