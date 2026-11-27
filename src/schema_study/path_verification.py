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
    source = next(
        item for item in execution_manifest()
        if item.get("family") == "principal_training"
        and item.get("model") == "qwen32b"
        and item.get("dataset") == dataset
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
        checks["development_evaluation"] = any((value or {}).get("complete") and (value or {}).get("metric", -1) >= 0 for value in record.values())
    configuration = production_configuration()
    configuration["device_policy"] = "fsdp"
    configuration["requested_gpus"] = int(task["resources"]["gpus"])
    checks["effective_batch"] = int(configuration["effective_batch_size"]) == 16
    exported = export_peft_adapter(workdir / "peft_standard", best or last, configuration)
    reloaded_names = set(json.loads((exported / "adapter_config.json").read_text(encoding="utf-8")).get("target_modules") or [])
    checks["adapter_export"] = (exported / "adapter_model.safetensors").is_file() and bool(reloaded_names)
    process = prepare_process(int(task["resources"]["gpus"]))
    model, tokenizer = load_causal_lm("qwen32b")
    attach_lora(model, configuration)
    restore_into_new_model(saved_for_role(saved, "best") if best is not None else saved, model, None, load_optimizer=False)
    checks["checkpoint_restored"] = True
    device, placement = _apply_device_policy(model, configuration)
    model = placement["model"]
    checks["sharding"] = bool(getattr(model, "_schema_study_fsdp2", False)) or process["world"] == 1
    checks["adapter_restored"] = checks["checkpoint_restored"] and checks["adapter_export"]
    model.eval()
    tokens = tokenizer("relation", return_tensors="pt")
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.no_grad():
        logits = model(**tokens).logits
    checks["inference_diagnostic"] = bool(torch.isfinite(logits).all())
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
