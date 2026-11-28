"""Compare an injected adapter with the standard PEFT export. This does not retrain."""

from __future__ import annotations

import json
from pathlib import Path


def mapped_keys(names: list[str], translate) -> list[str]:
    """Reject two source names that would overwrite one exported key."""
    seen = {}
    for name in names:
        key = translate(name)
        if key in seen:
            raise ValueError(f"Parameter-name collision: {seen[key]} and {name} both export as {key}.")
        seen[key] = name
    return list(seen)


def compare_adapters(task: dict, workdir: Path) -> dict:
    """Tensor identity for the available rank-8, rank-16 and rank-32 adapters, plus a forward when CUDA is present."""
    import torch

    from schema_study.checkpoint_roles import role_weights
    from schema_study.production_lora import export_peft_adapter, peft_key, production_configuration
    from schema_study.production_plan import _historical_principal, sensitivity_tasks, task_directory
    from schema_study.production_runtime import publish_scientific_status

    workdir.mkdir(parents=True, exist_ok=True)
    locations = {}
    for item in sensitivity_tasks():
        if item.get("change") in {"rank_8", "rank_32"} and item.get("seed") == 13:
            locations[item["change"]] = Path(task_directory(item)) / "best" / "adapter_checkpoint.pt"
    principal = _historical_principal("qwen8b", "biored", "E0", "P0", 13)
    locations["rank_16"] = Path(task_directory(principal)) / "best" / "adapter_checkpoint.pt"
    report = {}
    for name, path in locations.items():
        if not path.is_file():
            report[name] = {"present": False}
            continue
        saved = torch.load(path, map_location="cpu", weights_only=False)
        weights = role_weights(saved, "best")
        mapped_keys([key for key in weights if "lora_" in key], peft_key)
        configuration = production_configuration(rank=8 if name == "rank_8" else 32 if name == "rank_32" else 16)
        exported = export_peft_adapter(workdir / name, weights, configuration)
        from safetensors.torch import load_file

        restored = load_file(exported / "adapter_model.safetensors")
        mismatched = [
            peft_key(key) for key, tensor in weights.items()
            if "lora_" in key and not torch.equal(tensor.cpu(), restored[peft_key(key)].cpu())
        ]
        report[name] = {"present": True, "tensors": len(restored), "mismatched": mismatched, "forward": None}
        if torch.cuda.is_available() and not mismatched:
            report[name]["forward"] = _forward_agreement(weights, exported, configuration)
    (workdir / "adapter_numeric.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    publish_scientific_status(workdir, {"scientific": "diagnostic_recorded", "ranks": report})
    return {"completed": True, "status": "diagnostic_recorded", "report": report}


def _forward_agreement(weights: dict, exported: Path, configuration: dict) -> dict:
    """One fixed prompt through the injected adapter and through PEFT. BF16 tolerance is explicit."""
    import torch
    from peft import PeftModel

    from schema_study.production_lora import attach_lora, load_causal_lm

    model, tokenizer = load_causal_lm("qwen8b")
    attach_lora(model, configuration)
    current = model.state_dict()
    current.update(weights)
    model.load_state_dict(current)
    model.to("cuda").eval()
    tokens = tokenizer("candidate", return_tensors="pt").to("cuda")
    with torch.no_grad():
        injected = model(**tokens).logits
    base, _tokenizer = load_causal_lm("qwen8b")
    peft_model = PeftModel.from_pretrained(base, exported).to("cuda").eval()
    with torch.no_grad():
        exported_logits = peft_model(**tokens).logits
    difference = (injected - exported_logits).abs().max().item()
    return {"max_abs_difference": difference, "within_bf16_tolerance": difference <= 1e-2}
