"""Compare the BiomedBERT bin with the safetensors copy.

This reads tensors only. It does not construct a model or run a forward pass.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import torch
from safetensors.torch import load_file

STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
REVISION = "e1354b7a3a09615f6aba48dfad4b7a613eef7062"
MODEL = STORAGE / "models/microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext" / REVISION
BIN_PATH = MODEL / "model/pytorch_model.bin"
SAFE_PATH = MODEL / "model_safetensors/model.safetensors"
REPORT = STORAGE / "studies/schema_granularity_v1/runs/validation/round4_cpu/weight_verification.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shared_groups(state: dict) -> list[dict]:
    groups: dict[int, list[str]] = {}
    for name, value in state.items():
        groups.setdefault(value.data_ptr(), []).append(name)
    return [{"names": names, "shape": list(state[names[0]].shape), "dtype": str(state[names[0]].dtype)} for names in groups.values() if len(names) > 1]


def main() -> None:
    bin_sha = sha256(BIN_PATH)
    safe_sha = sha256(SAFE_PATH)
    source = torch.load(BIN_PATH, map_location="cpu", weights_only=True)
    converted = load_file(SAFE_PATH)
    missing = sorted(set(source) - set(converted))
    unexpected = sorted(set(converted) - set(source))
    mismatches = []
    for name, value in source.items():
        other = converted[name]
        if value.shape != other.shape or value.dtype != other.dtype or not torch.equal(value, other):
            mismatches.append(name)
    encoder = sorted(name for name in source if name.startswith("bert."))
    head = sorted(name for name in source if name.startswith("cls."))
    other = sorted(set(source) - set(encoder) - set(head))
    shared = shared_groups(source)
    cloned = []
    for group in shared:
        names = group["names"]
        for name in names[1:]:
            same_value = bool(torch.equal(source[names[0]], source[name]))
            stored_separately = converted[name].data_ptr() != converted[names[0]].data_ptr()
            cloned.append({"kept": names[0], "cloned": name, "values_equal": same_value, "safetensors_not_aliased": stored_separately})
    config = json.loads((MODEL / "model/config.json").read_text(encoding="utf-8"))
    embedding = source["bert.embeddings.word_embeddings.weight"]
    report = {
        "repository": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "revision": REVISION,
        "bin_sha256": bin_sha,
        "safetensors_sha256": safe_sha,
        "read_method": "torch.load(..., map_location='cpu', weights_only=True)",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "safetensors": __import__("safetensors").__version__,
        "tensor_count": len(source),
        "missing_from_safetensors": missing,
        "unexpected_in_safetensors": unexpected,
        "value_mismatch_count": len(mismatches),
        "value_mismatches": mismatches[:20],
        "shared_in_bin": shared,
        "shared_handling": cloned,
        "encoder_parameter_count": len(encoder),
        "pretrained_head_parameter_count": len(head),
        "other_parameter_names": other,
        "encoder_keys_present_in_conversion": missing == [] and all(name in converted for name in encoder),
        "vocab_rows": list(embedding.shape),
        "config_vocab_size": config.get("vocab_size"),
        "config_architectures": config.get("architectures"),
        "load_interpretation": {
            "encoder_prefix": "bert.",
            "ignored_pretrained_head_prefix": "cls.",
            "reason": "AutoModel loads the Bert encoder. The cls parameters are the masked-language-model head.",
            "extra_marker_rows_and_classifier": "The 14 added marker rows and the new classifier are created after loading. They are not part of this conversion.",
        },
        "equivalent": not missing and not unexpected and not mismatches and not other,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"equivalent": report["equivalent"], "tensors": report["tensor_count"], "mismatches": report["value_mismatch_count"], "encoder": len(encoder), "head": len(head)}, indent=2))
    if not report["equivalent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
