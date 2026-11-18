"""Local Qwen loading. Called only after the execution permit is granted."""

from __future__ import annotations


def load_qwen(model_id: str, revision: str, settings: dict):
    """Load one Qwen checkpoint. This round never calls this function."""
    supported = {"temperature", "top_p", "max_new_tokens", "enable_thinking"}
    unknown = sorted(set(settings) - supported)
    if unknown:
        raise ValueError(f"Unsupported generation settings are not replaced: {unknown}")
    if settings.get("enable_thinking") is not False:
        raise ValueError("The study setting requires enable_thinking false.")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    return {"tokenizer": tokenizer, "model": model, "settings": dict(settings)}
