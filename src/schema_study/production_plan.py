"""Task enumeration, approval binding, and CPU dispatch for the formal programme.

Counts are produced by iterating the agreed conditions. This module does not
load a model and it does not submit a GPU job. An approval file is accepted
only when it names this manifest, the repository commit, and the project account.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SEEDS = (13, 42, 2026)
BIORED_ENTITIES = tuple(f"E{level}" for level in range(8))
PREDICATES = ("P0", "P1", "P2", "P3")
CURRICULUM_EXPOSURE = {
    "coarse_to_fine": ("P0", "P1", "P2", "P3"),
    "fine_to_coarse": ("P3", "P2", "P1", "P0"),
    "mixed": ("P1", "P3", "P0", "P2"),
}
FINAL_NATIVE_PREDICATE = "P3"
ACCOUNT = "brics.b5ac"
MAX_RUNNING_JOBS = 256
MAX_QUEUED_JOBS = 512
MAX_ARRAY_TASKS = 1000
RETRY_POLICY = {
    "timeout": {"action": "resume_from_verified_checkpoint", "max_attempts": 3},
    "out_of_memory": {"action": "predefined_micro_batch_reduction", "preserve_effective_batch": 16, "max_attempts": 1},
    "nan_loss": {"action": "stop_condition", "max_attempts": 0},
    "input_integrity": {"action": "stop_dependents", "max_attempts": 0},
    "filesystem": {"action": "retry_unchanged", "max_attempts": 2},
}


def _task(family: str, **fields) -> dict:
    identity = "|".join(str(fields[key]) for key in sorted(fields))
    return {
        "task_id": hashlib.sha256(f"{family}|{identity}".encode("utf-8")).hexdigest()[:16],
        "family": family,
        "executable": True,
        "continuation_policy": "resume_the_same_world_size_from_the_last_completed_optimiser_step",
        "retry_policy": dict(RETRY_POLICY),
        **fields,
    }


def principal_training_tasks(reuse_drugprot_encoder: bool = True) -> list[dict]:
    """288 BioRED runs and 36 DrugProt runs. BC8 is not a training corpus."""
    systems = (
        ("M1", "biomedbert", "full_finetune"),
        ("M4", "qwen8b", "lora_sft"),
        ("M7", "qwen32b", "lora_sft"),
    )
    tasks = []
    for system, model, adaptation in systems:
        for entity in BIORED_ENTITIES:
            for predicate in PREDICATES:
                for seed in SEEDS:
                    tasks.append(_task("principal_training", dataset="biored", entity=entity, predicate=predicate, system=system, model=model, adaptation=adaptation, seed=seed, new_training=True))
        for predicate in PREDICATES:
            for seed in SEEDS:
                reuse = bool(reuse_drugprot_encoder and model == "biomedbert")
                tasks.append(_task(
                    "principal_training", dataset="drugprot", entity="native", predicate=predicate,
                    system=system, model=model, adaptation=adaptation, seed=seed, new_training=not reuse,
                    historical_predicate={"P0": "P1", "P1": "P2", "P2": "P3", "P3": "P4"}[predicate],
                ))
    return tasks


def biolinkbert_tasks() -> list[dict]:
    tasks = []
    for entity in ("E0", "E2", "E7"):
        for predicate in PREDICATES:
            for seed in SEEDS:
                tasks.append(_task("biolinkbert", dataset="biored", entity=entity, predicate=predicate, seed=seed, model="biolinkbert", adaptation="full_finetune"))
    for predicate in PREDICATES:
        for seed in SEEDS:
            tasks.append(_task("biolinkbert", dataset="drugprot", entity="native", predicate=predicate, seed=seed, model="biolinkbert", adaptation="full_finetune"))
    return tasks


def hierarchical_tasks(consistency: bool) -> list[dict]:
    family = "hierarchical_consistency" if consistency else "hierarchical_multitask"
    tasks = []
    for entity in ("E0", "E2", "E7"):
        for seed in SEEDS:
            tasks.append(_task(family, dataset="biored", entity=entity, seed=seed, heads=list(PREDICATES), consistency=consistency, consistency_weight=0.1 if consistency else 0.0))
    for seed in SEEDS:
        tasks.append(_task(family, dataset="drugprot", entity="native", seed=seed, heads=list(PREDICATES), consistency=consistency, consistency_weight=0.1 if consistency else 0.0))
    return tasks


def schema_conditioned_tasks() -> list[dict]:
    tasks = []
    for model in ("qwen8b", "qwen32b"):
        for dataset in ("biored", "drugprot"):
            for seed in SEEDS:
                for exposure in ("resource_efficient_one_view", "exposure_controlled_all_views"):
                    tasks.append(_task("schema_conditioned_lora", dataset=dataset, model=model, seed=seed, exposure=exposure))
    return tasks


def curriculum_tasks() -> list[dict]:
    tasks = []
    conditions = (("biored", "E2"), ("biored", "E7"), ("drugprot", "native"))
    for dataset, entity in conditions:
        for order in CURRICULUM_EXPOSURE:
            for seed in SEEDS:
                tasks.append(_task("curriculum", dataset=dataset, entity=entity, order=order, seed=seed, model="qwen8b", final_block=FINAL_NATIVE_PREDICATE))
    return tasks


def sensitivity_tasks() -> list[dict]:
    tasks = []
    for entity in ("E0", "E2", "E7"):
        for loss in ("class_weighted", "focal"):
            for seed in SEEDS:
                tasks.append(_task("class_imbalance", dataset="biored", entity=entity, predicate="P3", loss=loss, seed=seed, model="biomedbert", standard_loss_reused=True))
    for model in ("biomedbert", "qwen8b"):
        for strategy in ("random_negative", "hard_negative"):
            for seed in SEEDS:
                tasks.append(_task("negative_sampling", dataset="drugprot", entity="native", predicate="P3", model=model, strategy=strategy, seed=seed, evaluation_population="complete"))
    for model in ("biomedbert", "qwen8b"):
        for entity in ("E2", "E7"):
            for predicate in ("P0", "P3"):
                for fraction in (10, 25, 50):
                    for seed in SEEDS:
                        tasks.append(_task("data_efficiency", dataset="biored", entity=entity, predicate=predicate, model=model, fraction=fraction, seed=seed, full_run_reused=True))
    for withheld in ("P1", "P2", "E3"):
        for seed in SEEDS:
            tasks.append(_task("unseen_schema", dataset="biored", model="qwen8b", entity="E2", withheld=withheld, seed=seed))
    for learning_rate in (1e-5, 5e-5):
        for seed in SEEDS:
            tasks.append(_task("encoder_sensitivity", dataset="biored", entity="E2", predicate="P3", model="biomedbert", learning_rate=learning_rate, seed=seed, baseline_reused=True))
    for change in ("rank_8", "rank_32", "lr_5e-5", "lr_2e-4"):
        for seed in SEEDS:
            tasks.append(_task("lora_sensitivity", dataset="biored", entity="E2", predicate="P3", model="qwen8b", change=change, seed=seed, baseline_reused=True))
    return tasks


def curriculum_sequence(order: str, examples_by_predicate: dict[str, list[str]]) -> list[tuple[str, str, str]]:
    """Exposure blocks use the same example lists. Every order then repeats the native block."""
    if order not in CURRICULUM_EXPOSURE:
        raise ValueError(order)
    missing = [name for name in PREDICATES if name not in examples_by_predicate]
    if missing:
        raise KeyError(", ".join(missing))
    sequence = [(predicate, example, "exposure") for predicate in CURRICULUM_EXPOSURE[order] for example in examples_by_predicate[predicate]]
    sequence.extend((FINAL_NATIVE_PREDICATE, example, "final_native") for example in examples_by_predicate[FINAL_NATIVE_PREDICATE])
    return sequence


def family_counts(reuse_drugprot_encoder: bool = True) -> dict:
    principal = principal_training_tasks(reuse_drugprot_encoder)
    groups = {
        "principal_training": principal,
        "biolinkbert": biolinkbert_tasks(),
        "hierarchical_multitask": hierarchical_tasks(False),
        "hierarchical_consistency": hierarchical_tasks(True),
        "schema_conditioned_lora": schema_conditioned_tasks(),
        "curriculum": curriculum_tasks(),
        "sensitivity": sensitivity_tasks(),
    }
    counts = {name: len(rows) for name, rows in groups.items()}
    counts["principal_designed"] = len(principal)
    counts["principal_new_training"] = sum(1 for row in principal if row["new_training"])
    counts["principal_reused_drugprot_encoder"] = sum(1 for row in principal if not row["new_training"])
    counts["bc8_training"] = 0
    return counts


def programme_definition(commit: str, reuse_drugprot_encoder: bool = True) -> dict:
    return {
        "commit": commit,
        "account": ACCOUNT,
        "counts": family_counts(reuse_drugprot_encoder),
        "scope": "principal_and_agreed_supplementary",
        "gpu_execution": "requires_bound_approval",
        "allocation_balance": "not_retrieved_from_slurm_researcher_confirmed_scope_is_not_reduced",
    }


def manifest_sha256(commit: str, reuse_drugprot_encoder: bool = True) -> str:
    payload = json.dumps(programme_definition(commit, reuse_drugprot_encoder), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def gpu_authorised(study_directory: Path, commit: str, reuse_drugprot_encoder: bool = True) -> bool:
    """Approval must name this manifest. A non-empty statement alone is not sufficient."""
    path = Path(study_directory) / "formal_execution_approval.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest_sha256(commit, reuse_drugprot_encoder)
    return (
        payload.get("approved") is True
        and payload.get("manifest_sha256") == expected
        and payload.get("commit") == commit
        and payload.get("account") == ACCOUNT
        and payload.get("scope") == "principal_and_agreed_supplementary"
        and bool(str(payload.get("statement") or "").strip())
    )


def submission_allowed(running: int, pending: int, proposed: int) -> tuple[bool, str]:
    """A 1,000-task array is still rejected when it would pass the 512-job user limit."""
    if proposed > MAX_ARRAY_TASKS:
        return False, "array_exceeds_1000"
    if running + pending + proposed > MAX_QUEUED_JOBS:
        return False, "user_job_limit_512"
    if running + proposed > MAX_RUNNING_JOBS:
        return False, "defer_until_running_jobs_fall"
    return True, "accepted"


def validate_dependencies(tasks: list[dict]) -> list[str]:
    """Return dependency errors. External prerequisites are not task identifiers."""
    known = {task["task_id"] for task in tasks}
    external = {"inputs_prepared", "model_assets", "approval", "historical_drugprot_checkpoint"}
    errors = []
    for task in tasks:
        for dependency in task.get("depends_on", []):
            if dependency not in known and dependency not in external:
                errors.append(f"{task['task_id']} missing {dependency}")
        if task["task_id"] in task.get("depends_on", []):
            errors.append(f"{task['task_id']} depends on itself")
    return errors


def drugprot_checkpoint_reusable(directory: Path) -> bool:
    """File presence is not enough: the completion record and the best checkpoint must both exist."""
    completion = directory / "prediction_complete.json"
    checkpoint = directory / "checkpoints" / "best" / "checkpoint.pt"
    if not completion.is_file() or not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        return False
    payload = json.loads(completion.read_text(encoding="utf-8"))
    return payload.get("rows_written") is True and payload.get("checkpoint") == "best"


def audit_drugprot_encoder(root: Path) -> dict:
    reusable = []
    rejected = []
    for predicate in ("P1", "P2", "P3", "P4"):
        for seed in SEEDS:
            directory = root / "drugprot" / "native" / predicate / f"seed{seed}"
            record = {"predicate": predicate, "seed": seed}
            if drugprot_checkpoint_reusable(directory):
                reusable.append(record)
            else:
                rejected.append(record)
    return {"reusable": reusable, "rejected": rejected, "reuse_count": len(reusable)}
