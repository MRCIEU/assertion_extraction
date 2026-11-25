"""Executable definitions for the formal programme.

These functions construct losses, masks, schedules and inventories. They do not
load model weights and they do not submit jobs. Formal GPU execution stays
closed until a researcher approval file is present.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

APPROVAL_NAME = "formal_execution_approval.json"
CHECKPOINT_FIELDS = (
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "scaler_state",
    "rng_state",
    "step",
    "epoch",
    "data_fingerprint",
    "configuration_fingerprint",
)
MULTITASK_LOSS_WEIGHTS = {"existence": 1.0, "coarse": 1.0, "native": 1.0}
CONSISTENCY_WEIGHT = 0.1
NHR_PER_GPU_HOUR = 0.25


def execution_authorised(study_directory: Path) -> bool:
    """Approval is an explicit file, not the presence of a test suite."""
    path = Path(study_directory) / APPROVAL_NAME
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("approved") is True and bool(str(payload.get("statement") or "").strip())


def distributed_accumulation(micro_batch: int, world_size: int, effective_batch: int = 16) -> int:
    """Gradient accumulation that keeps the global batch at the study value."""
    if micro_batch <= 0 or world_size <= 0:
        raise ValueError("Micro-batch and world size must be positive.")
    per_update = micro_batch * world_size
    if effective_batch % per_update != 0:
        raise ValueError("The micro-batch and world size must divide the effective batch.")
    return effective_batch // per_update


def assistant_labels(token_ids: list[int], prompt_tokens: int, pad_id: int) -> list[int]:
    """Supervise the assistant span only. Prompt and padding receive the ignore index."""
    if prompt_tokens < 0 or prompt_tokens > len(token_ids):
        raise ValueError("The prompt length is outside the token sequence.")
    labels = [-100 if index < prompt_tokens or token == pad_id else token for index, token in enumerate(token_ids)]
    if all(label == -100 for label in labels):
        raise ValueError("The supervised span is empty.")
    return labels


def combine_head_losses(losses: dict[str, float]) -> float:
    """A frozen weighted sum. The weights are not tuned on the test set."""
    missing = [name for name in MULTITASK_LOSS_WEIGHTS if name not in losses]
    if missing:
        raise KeyError(", ".join(missing))
    return sum(MULTITASK_LOSS_WEIGHTS[name] * float(losses[name]) for name in MULTITASK_LOSS_WEIGHTS)


def kl_consistency(target: list[float], prediction: list[float]) -> float:
    """Divergence from the mapped native distribution to the direct coarse distribution."""
    if len(target) != len(prediction) or not target:
        raise ValueError("The two distributions must share a non-empty label space.")
    if any(value <= 0 for value in prediction):
        raise ValueError("The coarse distribution must be strictly positive.")
    total = sum(target)
    if abs(total - 1.0) > 1e-6:
        raise ValueError("The mapped native distribution must sum to one.")
    return sum(0.0 if left == 0 else left * math.log(left / right) for left, right in zip(target, prediction))


def parent_event_probability(child_probabilities: list[float]) -> float:
    """A parent event occurs if any child event occurs. Sigmoid outputs are not added."""
    remaining = 1.0
    for probability in child_probabilities:
        if probability != probability or not 0.0 <= float(probability) <= 1.0:
            raise ValueError("A child probability must be finite and inside [0, 1].")
        remaining *= 1.0 - float(probability)
    return 1.0 - remaining


def schema_view(candidate_id: str, epoch: int, views: list[str], seed: int = 20261125) -> str:
    """One representation of a candidate in an epoch. The views are not extra findings."""
    if epoch < 0 or not views:
        raise ValueError("An epoch and at least one view are required.")
    digest = int(hashlib.sha256(f"{seed}:{epoch}:{candidate_id}".encode("utf-8")).hexdigest()[:8], 16)
    return views[digest % len(views)]


def curriculum_stages(order: str) -> list[str]:
    """Every order ends on the same native block, so the final target is not confounded."""
    prefixes = {
        "coarse_to_fine": ["coarse", "fine"],
        "fine_to_coarse": ["fine", "coarse"],
        "mixed": ["mixed"],
    }
    if order not in prefixes:
        raise ValueError(order)
    return prefixes[order] + ["native"]


def nested_document_subsets(document_ids: list[str], seed: int = 13) -> dict[str, list[str]]:
    """10 percent is contained in 25, 50 and 100 percent of the same shuffled documents."""
    identifiers = sorted(set(document_ids))
    generator = random.Random(seed)
    generator.shuffle(identifiers)
    fractions = (0.1, 0.25, 0.5, 1.0)
    counts = {fraction: max(1, int(round(fraction * len(identifiers)))) if identifiers else 0 for fraction in fractions}
    counts[1.0] = len(identifiers)
    return {f"{int(fraction * 100)}": identifiers[: counts[fraction]] for fraction in fractions}


def training_requirement(dataset: str, entity_level: str, changed_training_inputs: int, historical_checkpoint: bool) -> str:
    """A changed training population is retrained. An unchanged DrugProt checkpoint can be reused."""
    if changed_training_inputs > 0:
        return "retrain"
    if dataset == "drugprot" and entity_level == "native" and historical_checkpoint:
        return "reuse_historical_checkpoint"
    if historical_checkpoint and entity_level in {"E0", "E1", "E2", "native"}:
        return "reuse_if_input_fingerprint_matches"
    return "new_training"


def checkpoint_can_resume(saved: dict, configuration_fingerprint: str, data_fingerprint: str) -> bool:
    """A continuation is the same run only when both fingerprints and the step agree."""
    missing = [field for field in CHECKPOINT_FIELDS if field not in saved]
    if missing:
        return False
    return saved["configuration_fingerprint"] == configuration_fingerprint and saved["data_fingerprint"] == data_fingerprint and int(saved["step"]) >= 0


def node_hours(gpu_hours: float) -> float:
    """Four GPU-hours are one node-hour when a job uses one GPU on a four-GPU node."""
    return gpu_hours * NHR_PER_GPU_HOUR


def supplementary_inventory() -> list[dict]:
    """Supplementary runs stay outside the 324-run principal design."""
    return [
        {"id": "biolinkbert", "priority": "supplementary", "purpose": "Encoder replication at E0, E2 and E7, plus native DrugProt.", "new_training_runs": 60, "weights_local": False},
        {"id": "hierarchical_encoder", "priority": "supplementary", "purpose": "Shared encoder with existence, coarse and native heads.", "new_training_runs": 12, "loss_weights": dict(MULTITASK_LOSS_WEIGHTS)},
        {"id": "hierarchical_consistency", "priority": "supplementary", "purpose": "The multi-task model plus a frozen consistency term.", "new_training_runs": 12, "consistency_weight": CONSISTENCY_WEIGHT},
        {"id": "schema_conditioned_lora", "priority": "supplementary", "purpose": "One adapter per corpus and model size, not 32 independent adapters.", "new_training_runs": 4},
        {"id": "curriculum", "priority": "supplementary", "purpose": "Coarse-to-fine, fine-to-coarse and mixed order, with a shared final native block.", "new_training_runs": 9},
        {"id": "class_imbalance", "priority": "sensitivity", "purpose": "Weighted and focal losses against the unchanged principal objective.", "replaces_principal_loss": False},
        {"id": "negative_sampling", "priority": "sensitivity", "purpose": "DrugProt training subsamples. The evaluation population stays complete.", "evaluation_population_changed": False},
        {"id": "data_efficiency", "priority": "sensitivity", "purpose": "Nested 10, 25, 50 and 100 percent document subsets. The 100 percent run is the principal run.", "duplicate_full_run": False},
        {"id": "unseen_schema", "priority": "sensitivity", "purpose": "A withheld schema instruction. Native distinctions may still have been seen under another view."},
        {"id": "gpt_oss_20b", "priority": "external", "purpose": "Zero-shot and few-shot at representative conditions. Not the full factorial.", "weights_local": False, "full_matrix": False},
        {"id": "qwen36_27b", "priority": "optional", "purpose": "Cross-generation check if the checkpoint, licence and budget are later approved.", "weights_local": False},
    ]


def manifest_payload() -> dict:
    """The inventory that must be approved before any formal GPU submission."""
    return {
        "approval": "not_approved",
        "gpu_submission": "blocked",
        "principal": principal_training_counts(True, 12),
        "supplementary": supplementary_inventory(),
        "nli_baseline": {
            "model_id": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            "revision": "6f5cf0a2b59cabb106aca4c287eed12e357e90eb",
            "id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"},
            "weights_local": False,
            "xsmall_checkpoint": "anonymous_access_returned_401",
        },
        "biolinkbert": {
            "model_id": "michiyasunaga/BioLinkBERT-base",
            "revision": "b71f5d70f063d1c8f1124070ce86f1ee463ca1fe",
            "weights_local": False,
        },
        "observed_drugprot_biomedbert": {
            "source": "formal_20261115/train/drugprot/native/P1/seed13/training_history.jsonl",
            "updates_per_epoch": 28212,
            "epoch_seconds": [1324.0, 1340.7, 1313.4, 1344.8, 1328.5],
            "forecast_status": "not_a_forecast_of_the_qwen_or_biored_matrix",
        },
    }


def principal_training_counts(biored_training_inputs_changed: bool, drugprot_historical_runs: int) -> dict:
    """324 designed supervised runs. Unchanged DrugProt BiomedBERT checkpoints are not repeated."""
    biored_encoder = 8 * 4 * 3
    biored_lora = biored_encoder * 2
    drugprot_encoder = 4 * 3
    drugprot_lora = drugprot_encoder * 2
    drugprot_new_encoder = 0 if drugprot_historical_runs >= drugprot_encoder else drugprot_encoder - drugprot_historical_runs
    biored_new_encoder = biored_encoder if biored_training_inputs_changed else 5 * 4 * 3
    new_runs = biored_new_encoder + biored_lora + drugprot_new_encoder + drugprot_lora
    return {
        "designed_supervised_runs": biored_encoder * 3 + drugprot_encoder * 3,
        "biored_encoder_runs": biored_new_encoder,
        "biored_lora_runs": biored_lora,
        "drugprot_historical_encoder_runs": min(drugprot_historical_runs, drugprot_encoder),
        "drugprot_new_encoder_runs": drugprot_new_encoder,
        "drugprot_new_lora_runs": drugprot_lora,
        "new_principal_training_runs": new_runs,
        "bc8_training_runs": 0,
        "deterministic_abstraction_is_inference": False,
    }
