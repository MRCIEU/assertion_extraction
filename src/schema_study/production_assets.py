"""Pinned local model locations. This module does not load weights."""

from __future__ import annotations

import json
from pathlib import Path

STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
MODELS = STORAGE / "models"

BIOMEDBERT_REVISION = "e1354b7a3a09615f6aba48dfad4b7a613eef7062"
BIOLINKBERT_REVISION = "b71f5d70f063d1c8f1124070ce86f1ee463ca1fe"
QWEN8_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
QWEN32_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
GPT_OSS_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
QWEN36_REVISION = "1cda14dd71a3f47115850eb790f416f165da1cb5"
DEBERTA_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
CERTAINTY_REVISION = "57bb19e0804a77689ca02f2b1d408d162413cdc2"


def _revision(root: Path, revision: str) -> Path:
    return root / revision


ENCODERS = {
    "biomedbert": {
        "model_id": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "revision": BIOMEDBERT_REVISION,
        "model": _revision(MODELS / "microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", BIOMEDBERT_REVISION) / "model_safetensors",
        "tokenizer": _revision(MODELS / "microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", BIOMEDBERT_REVISION) / "tokenizer",
        "architecture": "BertForMaskedLM",
    },
    "biolinkbert": {
        "model_id": "michiyasunaga/BioLinkBERT-base",
        "revision": BIOLINKBERT_REVISION,
        "model": _revision(MODELS / "michiyasunaga__BioLinkBERT-base", BIOLINKBERT_REVISION),
        "tokenizer": _revision(MODELS / "michiyasunaga__BioLinkBERT-base", BIOLINKBERT_REVISION),
        "architecture": "BertModel",
    },
}

GENERATIVE = {
    "qwen8b": {
        "model_id": "Qwen/Qwen3-8B",
        "revision": QWEN8_REVISION,
        "path": _revision(MODELS / "Qwen__Qwen3-8B", QWEN8_REVISION),
        "family": "qwen3",
        "chat_template": True,
    },
    "qwen32b": {
        "model_id": "Qwen/Qwen3-32B",
        "revision": QWEN32_REVISION,
        "path": _revision(MODELS / "Qwen__Qwen3-32B", QWEN32_REVISION),
        "family": "qwen3",
        "chat_template": True,
    },
    "gpt_oss_20b": {
        "model_id": "openai/gpt-oss-20b",
        "revision": GPT_OSS_REVISION,
        "path": _revision(MODELS / "openai__gpt-oss-20b", GPT_OSS_REVISION),
        "family": "gpt_oss",
        "chat_template": True,
        "licence": "Apache-2.0",
    },
    "qwen36": {
        "model_id": "Qwen/Qwen3.6-27B",
        "revision": QWEN36_REVISION,
        "path": _revision(MODELS / "Qwen__Qwen3.6-27B", QWEN36_REVISION),
        "family": "qwen3_5",
        "licence": "Apache-2.0",
    },
}


def encoder_layout(model_name: str) -> dict:
    if model_name not in ENCODERS:
        raise KeyError(model_name)
    layout = dict(ENCODERS[model_name])
    layout["model_present"] = layout["model"].exists()
    layout["tokenizer_present"] = layout["tokenizer"].exists()
    return layout


def read_config(model_name: str) -> dict:
    """Read config.json. This does not construct a model."""
    spec = GENERATIVE[model_name]
    path = spec["path"] / "config.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def architecture_status(model_name: str) -> dict:
    """Record whether the installed Transformers release can name the architecture."""
    if model_name == "qwen36":
        config = read_config(model_name)
        return {
            "model": model_name,
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "weights_present": spec_present(model_name),
            "blocked": True,
            "reason": "Transformers 4.57.5 does not recognise model_type qwen3_5. The environment is not upgraded.",
        }
    if model_name == "gpt_oss_20b":
        config = read_config(model_name)
        return {
            "model": model_name,
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "weights_present": spec_present(model_name),
            "blocked": False,
            "weight_loading": "requires_authorised_gpu_preflight",
            "reason": "GptOssForCausalLM is the recorded class. Weights have not been loaded in this preparation.",
        }
    spec = GENERATIVE[model_name]
    return {"model": model_name, "revision": spec["revision"], "weights_present": spec_present(model_name), "blocked": False}


def spec_present(model_name: str) -> bool:
    path = GENERATIVE[model_name]["path"]
    return path.is_dir() and (path / "config.json").is_file()
