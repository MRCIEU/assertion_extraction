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
        "executable": False,
        "statuses": ["specified"],
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
            tasks.append(_task(
                "unseen_schema", dataset="biored", model="qwen8b", withheld=withheld, seed=seed,
                training_views=unseen_training_views(withheld),
            ))
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
    if order == "mixed":
        indices = {predicate: 0 for predicate in PREDICATES}
        sequence = []
        remaining = sum(len(examples_by_predicate[predicate]) for predicate in PREDICATES)
        while remaining:
            for predicate in PREDICATES:
                index = indices[predicate]
                if index < len(examples_by_predicate[predicate]):
                    sequence.append((predicate, examples_by_predicate[predicate][index], "exposure"))
                    indices[predicate] = index + 1
                    remaining -= 1
    else:
        sequence = [(predicate, example, "exposure") for predicate in CURRICULUM_EXPOSURE[order] for example in examples_by_predicate[predicate]]
    sequence.extend((FINAL_NATIVE_PREDICATE, example, "final_native") for example in examples_by_predicate[FINAL_NATIVE_PREDICATE])
    return sequence


def membership_rows(schema, coarse_level: str, native_level: str) -> list[list[int]]:
    """Map each native label, including the negative outcome, into exactly one coarse label."""
    native_ids = [label.stable_id for label in schema.level_labels(native_level)] + [schema.no_relation_id]
    rows = []
    for label in schema.level_labels(coarse_level):
        rows.append([1 if identifier in label.members else 0 for identifier in native_ids])
    rows.append([1 if identifier == schema.no_relation_id else 0 for identifier in native_ids])
    if any(sum(column) != 1 for column in zip(*rows)):
        raise ValueError(f"{schema.schema_id} {coarse_level} does not partition {native_level}.")
    return rows


def unseen_training_views(withheld: str) -> list[str]:
    """E3 is withheld from a multi-level inventory. Training only at E2 would not test that exclusion."""
    if withheld in BIORED_ENTITIES:
        views = [entity for entity in BIORED_ENTITIES if entity != withheld]
    elif withheld in PREDICATES:
        views = [predicate for predicate in PREDICATES if predicate != withheld]
    else:
        raise ValueError(withheld)
    if withheld in views or len(views) < 2:
        raise ValueError("The withheld view must leave more than one training view.")
    return views


EXECUTORS = {
    "full_finetune": "schema_study.formal_train:train",
    "lora_sft": "schema_study.production_lora:fit",
    "biolinkbert": "schema_study.formal_train:train",
    "hierarchical_multitask": "schema_study.production_encoder:train_hierarchical",
    "hierarchical_consistency": "schema_study.production_encoder:train_hierarchical",
    "schema_conditioned_lora": "schema_study.production_lora:fit",
    "curriculum": "schema_study.production_lora:fit",
    "class_imbalance": "schema_study.production_encoder:train_encoder",
    "negative_sampling": "schema_study.production_encoder:train_encoder",
    "data_efficiency": "schema_study.production_encoder:train_encoder",
    "unseen_schema": "schema_study.production_lora:fit",
    "encoder_sensitivity": "schema_study.formal_train:train",
    "lora_sensitivity": "schema_study.production_lora:fit",
    "zero_shot": "schema_study.formal_generate:generate",
    "few_shot": "schema_study.formal_generate:generate",
    "adapter_inference": "schema_study.formal_generate:generate",
    "constrained": "schema_study.production_inference:constrained_generate",
    "thinking": "schema_study.production_inference:thinking_generate",
    "self_consistency": "schema_study.production_inference:self_consistency",
    "evidence": "schema_study.production_inference:evidence_generate",
    "qualified": "schema_study.production_inference:qualified_generate",
    "candidate_scoring": "schema_study.production_inference:candidate_scores",
    "certainty": "schema_study.measurement_cpu:main",
    "nli": "schema_study.measurement_cpu:main",
    "verifier_adaptation": "schema_study.measurement_cpu:adapt_verifier",
    "analysis": "schema_study.analysis_plan:analyse_records",
    "gpt_oss": "schema_study.production_lora:load_causal_lm",
    "qwen36": None,
}

IMPLEMENTED_FAMILIES = {
    "principal_training", "biolinkbert", "hierarchical_multitask", "hierarchical_consistency",
    "schema_conditioned_lora", "curriculum", "class_imbalance", "negative_sampling",
    "data_efficiency", "unseen_schema", "encoder_sensitivity", "lora_sensitivity",
    "zero_shot", "few_shot", "adapter_inference", "constrained", "thinking",
    "self_consistency", "evidence", "qualified", "candidate_scoring", "certainty",
    "nli", "verifier_adaptation", "analysis", "gpt_oss",
}
BLOCKED_MODELS = {
    "qwen36": "Transformers 4.57.5 does not recognise model_type qwen3_5. The optional comparison does not block the principal study.",
}


def executor_for(task: dict) -> str | None:
    if task.get("model") == "qwen36":
        return None
    if task["family"] == "principal_training":
        return EXECUTORS[task["adaptation"]]
    if task.get("model", "").startswith("qwen") and task["family"] in {"negative_sampling", "data_efficiency", "lora_sensitivity"}:
        return EXECUTORS["lora_sft"]
    if task.get("model") == "qwen8b" and task["family"] == "negative_sampling":
        return EXECUTORS["lora_sft"]
    return EXECUTORS.get(task["family"])


def _configuration(task: dict) -> dict:
    """Scientific settings that change the result. Walltime is not one of them."""
    from schema_study.learning_conditions import lora_configuration
    from schema_study.models.schedule import STUDY_HYPERPARAMETERS

    if task.get("adaptation") == "lora_sft" or task["family"] in {"schema_conditioned_lora", "curriculum", "unseen_schema", "lora_sensitivity"} or (
        str(task.get("model", "")).startswith("qwen") and task["family"] in {"negative_sampling", "data_efficiency"}
    ):
        configuration = lora_configuration()
        if task.get("change") == "rank_8":
            configuration["rank"] = 8
        elif task.get("change") == "rank_32":
            configuration["rank"] = 32
        elif task.get("change") == "lr_5e-5":
            configuration["learning_rate"] = 5e-5
        elif task.get("change") == "lr_2e-4":
            configuration["learning_rate"] = 2e-4
        configuration["seed"] = task.get("seed")
        configuration["incomplete_groups"] = "dropped"
        configuration["schedule_scope"] = "full_intended_horizon"
        if task["family"] == "curriculum":
            configuration["max_epochs"] = 1
            configuration["data_schedule"] = "single_matched_curriculum_pass"
        elif task["family"] == "schema_conditioned_lora":
            configuration["data_schedule"] = task.get("exposure")
        else:
            configuration["data_schedule"] = "independent_epochs"
        return {key: configuration[key] for key in (
            "precision", "quantization", "rank", "alpha", "dropout", "learning_rate", "warmup_ratio",
            "max_epochs", "effective_batch_size", "seed", "schedule", "incomplete_groups", "schedule_scope", "data_schedule",
        ) if key in configuration}
    configuration = {
        "precision": "bf16",
        "learning_rate": task.get("learning_rate", STUDY_HYPERPARAMETERS["learning_rate"]),
        "max_epochs": STUDY_HYPERPARAMETERS["max_epochs"],
        "effective_batch_size": STUDY_HYPERPARAMETERS["effective_batch_size"],
        "objective": task.get("loss") or task.get("adaptation") or "standard",
        "seed": task.get("seed"),
    }
    if task.get("fraction"):
        configuration["fraction"] = task["fraction"]
    if task.get("strategy"):
        configuration["negative_strategy"] = task["strategy"]
    if "consistency" in task:
        configuration["consistency_weight"] = task["consistency_weight"]
    return configuration


def _resources(task: dict) -> dict:
    model = task.get("model", "")
    if str(model).startswith("qwen32") or model == "qwen32b":
        return {"gpus": 2, "gpu_options": [1, 2, 4], "cpus": 8, "strategy": "fsdp_full_shard", "parameter_sharding": True, "partition": "workq", "qos": "normal"}
    if str(model).startswith("qwen") or model in {"gpt_oss_20b", "qwen36"}:
        return {"gpus": 1, "gpu_options": [1, 2, 4], "cpus": 8, "strategy": "single_process", "partition": "workq", "qos": "normal"}
    return {"gpus": 1, "gpu_options": [1], "cpus": 8, "strategy": "single_gpu", "partition": "workq", "qos": "normal"}


def _walltime(task: dict) -> str:
    if task.get("dataset") == "drugprot" and task["family"] in {"principal_training", "biolinkbert"}:
        return "08:00:00"
    if task["family"] in {"schema_conditioned_lora", "curriculum", "lora_sensitivity"} or task.get("adaptation") == "lora_sft":
        return "24:00:00"
    return "04:00:00"


def annotate_task(task: dict) -> dict:
    """A completed implementation is not authorisation. Executable stays false."""
    from schema_study.learning_conditions import HISTORICAL_PREDICATES
    from schema_study.production_assets import ENCODERS, GENERATIVE
    from schema_study.production_records import input_reference

    executor = executor_for(task)
    task["executor"] = executor
    task["executable"] = False
    task["authorised"] = False
    model = task.get("model")
    if model in ENCODERS:
        task["model_revision"] = ENCODERS[model]["revision"]
        task["tokenizer_revision"] = ENCODERS[model]["revision"]
    elif model in GENERATIVE:
        task["model_revision"] = GENERATIVE[model]["revision"]
        task["tokenizer_revision"] = GENERATIVE[model]["revision"]
    elif model == "deberta_base":
        from schema_study.production_assets import DEBERTA_REVISION
        task["model_revision"] = DEBERTA_REVISION
    if task.get("predicate"):
        task["schema_predicate"] = task.get("historical_predicate") or HISTORICAL_PREDICATES.get(task["predicate"], task["predicate"])
    dataset = task.get("dataset") or "biored"
    kind = "sft" if task.get("adaptation") in {"lora_sft", "full_finetune"} or task["family"] in {
        "schema_conditioned_lora", "curriculum", "unseen_schema", "lora_sensitivity", "hierarchical_multitask", "hierarchical_consistency",
    } else "contexts"
    if dataset in {"biored", "drugprot", "bc8"}:
        try:
            task["input"] = input_reference(dataset, "sft" if kind == "sft" and dataset != "bc8" else "contexts")
        except KeyError:
            task["input"] = input_reference(dataset, "contexts")
    task["configuration"] = _configuration(task)
    entity = task.get("entity", "all")
    predicate = task.get("predicate", "all")
    seed = task.get("seed", "none")
    root = f"/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/runs/formal/formal_matched_e0_e7/{task['family']}/{dataset}/{entity}/{predicate}/seed{seed}/{task['task_id']}"
    task["output"] = root
    task["checkpoint"] = root + "/checkpoints"
    task["validator"] = "completion record with rows_written true, finite loss, and the saved adapter or encoder state"
    dependencies = ["inputs_prepared", "model_assets", "approval"]
    if task.get("new_training") is False:
        dependencies.append("historical_drugprot_checkpoint")
    if task.get("trains_from"):
        dependencies.append(task["trains_from"])
    task["depends_on"] = dependencies
    task["resources"] = _resources(task)
    task["walltime"] = _walltime(task)
    task["command"] = f"python -m schema_study.production_dispatch run {task['family']} {task['task_id']}"
    reason = BLOCKED_MODELS.get(model or "")
    statuses = ["specified"]
    if reason or not executor or task["family"] not in IMPLEMENTED_FAMILIES:
        statuses.append("blocked")
        task["blocked_reason"] = reason or "production_executor_missing"
    else:
        statuses.extend(["implementation_complete", "awaiting_production_gpu_preflight"])
    task["statuses"] = statuses
    return task


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


SCIENTIFIC_FIELDS = (
    "family", "dataset", "entity", "predicate", "system", "model", "adaptation", "seed",
    "exposure", "order", "loss", "strategy", "fraction", "withheld", "training_views",
    "learning_rate", "change", "heads", "consistency", "consistency_weight", "final_block",
    "new_training", "historical_predicate",
)


def resolved_tasks(reuse_drugprot_encoder: bool = True) -> list[dict]:
    groups = [
        principal_training_tasks(reuse_drugprot_encoder),
        biolinkbert_tasks(),
        hierarchical_tasks(False),
        hierarchical_tasks(True),
        schema_conditioned_tasks(),
        curriculum_tasks(),
        sensitivity_tasks(),
    ]
    return [annotate_task(task) for group in groups for task in group]


def scientific_record(task: dict) -> dict:
    """Fields that change the scientific result. Scheduling fields are omitted."""
    row = {field: task[field] for field in SCIENTIFIC_FIELDS if field in task}
    row["task_id"] = task["task_id"]
    row["executor"] = task.get("executor")
    row["model_revision"] = task.get("model_revision")
    row["schema_predicate"] = task.get("schema_predicate")
    row["input_fingerprint"] = (task.get("input") or {}).get("fingerprint")
    row["configuration"] = task.get("configuration")
    return row


def approval_digest(tasks: list[dict], commit: str) -> str:
    rows = [scientific_record(task) for task in sorted(tasks, key=lambda item: item["task_id"])]
    payload = {"account": ACCOUNT, "commit": commit, "scope": "principal_and_agreed_supplementary", "tasks": rows}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _inference_tasks() -> list[dict]:
    tasks = []
    for model in ("qwen8b", "qwen32b"):
        for dataset, entities in (("biored", BIORED_ENTITIES), ("bc8", BIORED_ENTITIES), ("drugprot", ("native",))):
            for entity in entities:
                for predicate in PREDICATES:
                    for paradigm, family in (("zero_shot", "zero_shot"), ("few_shot", "few_shot")):
                        tasks.append(_task(family, dataset=dataset, entity=entity, predicate=predicate, model=model, paradigm=paradigm, seed=None))
    return tasks


def _supplementary_tasks(training: list[dict]) -> list[dict]:
    tasks = []
    for strategy in ("constrained", "candidate_scoring", "thinking", "self_consistency", "evidence", "qualified"):
        for dataset, entity in (("biored", "E2"), ("biored", "E7"), ("drugprot", "native")):
            tasks.append(_task(strategy, dataset=dataset, entity=entity, predicate="P3", model="qwen8b", scope="supplementary_decoding"))
    for dataset in ("biored", "drugprot"):
        for paradigm in ("zero_shot", "few_shot"):
            tasks.append(_task("gpt_oss", dataset=dataset, entity="E2" if dataset == "biored" else "native", predicate="P3", model="gpt_oss_20b", paradigm=paradigm))
            tasks.append(_task("qwen36", dataset=dataset, entity="E2" if dataset == "biored" else "native", predicate="P3", model="qwen36", paradigm=paradigm))
    for source in training:
        if source.get("adaptation") == "lora_sft" or source["family"] in {"schema_conditioned_lora", "curriculum", "unseen_schema", "lora_sensitivity"}:
            tasks.append(_task(
                "adapter_inference", dataset=source.get("dataset"), entity=source.get("entity"), predicate=source.get("predicate"),
                model=source.get("model"), seed=source.get("seed"), trains_from=source["task_id"],
            ))
    tasks.append(_task("verifier_adaptation", dataset="scifact", model="deberta_base", split="train_development_held_out_from_reported_baseline"))
    for dataset in ("biored", "drugprot", "bc8"):
        tasks.append(_task("analysis", dataset=dataset, statistic="corpus_micro_f1"))
    tasks.append(_task("certainty", dataset="scifact", model="sentence_level_certainty", labelled_accuracy=False))
    tasks.append(_task("nli", dataset="scifact", model="deberta_base", role="general_domain_baseline"))
    return tasks


def execution_manifest(reuse_drugprot_encoder: bool = True) -> list[dict]:
    """Training, inference, measurement and analysis. Continuation segments are not extra training runs."""
    training = resolved_tasks(reuse_drugprot_encoder)
    return training + [annotate_task(task) for task in _inference_tasks() + _supplementary_tasks(training)]


def manifest_sha256(commit: str, reuse_drugprot_encoder: bool = True) -> str:
    """Digest the resolved execution definitions. A count summary is not the approval record."""
    return approval_digest(execution_manifest(reuse_drugprot_encoder), commit)


def executing_revision(code_root: Path) -> str | None:
    """The checkout that will run, not an environment variable supplied by the caller."""
    import subprocess

    try:
        output = subprocess.check_output(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.strip() or None


def gpu_authorised(study_directory: Path, code_root: Path, reuse_drugprot_encoder: bool = True) -> bool:
    """Approval must name the resolved manifest and the revision git reports for this checkout."""
    path = Path(study_directory) / "formal_execution_approval.json"
    revision = executing_revision(code_root)
    if not path.is_file() or revision is None:
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        payload.get("approved") is True
        and payload.get("manifest_sha256") == manifest_sha256(revision, reuse_drugprot_encoder)
        and payload.get("commit") == revision
        and payload.get("account") == ACCOUNT
        and payload.get("scope") == "principal_and_agreed_supplementary"
        and bool(str(payload.get("statement") or "").strip())
    )


def retry_decision(kind: str, attempts: int) -> str:
    """Bounded recovery. An attempt count at the limit stops the condition."""
    policy = RETRY_POLICY[kind]
    if attempts >= int(policy["max_attempts"]):
        return "stop"
    return str(policy["action"])


def submission_allowed(running: int, pending: int, proposed: int) -> tuple[bool, str]:
    """Proposed jobs join the queue. They are not counted as already running."""
    if proposed > MAX_ARRAY_TASKS:
        return False, "array_exceeds_1000"
    if running + pending + proposed > MAX_QUEUED_JOBS:
        return False, "user_job_limit_512"
    return True, "accepted"


def runnable_slots(running: int) -> int:
    return max(0, MAX_RUNNING_JOBS - running)


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
    if find_cycles(tasks):
        errors.append("dependency cycle")
    return errors


def find_cycles(tasks: list[dict]) -> list[list[str]]:
    """Return one node list for each cycle. A self-loop is a cycle."""
    known = {task["task_id"] for task in tasks}
    graph = {task["task_id"]: [item for item in task.get("depends_on", []) if item in known] for task in tasks}
    colour = {task_id: 0 for task_id in graph}
    cycles = []

    def visit(task_id: str, stack: list[str]) -> None:
        colour[task_id] = 1
        stack.append(task_id)
        for dependency in graph[task_id]:
            if colour[dependency] == 1:
                cycles.append(stack[stack.index(dependency):] + [dependency])
            elif colour[dependency] == 0:
                visit(dependency, stack)
        stack.pop()
        colour[task_id] = 2

    for task_id in graph:
        if colour[task_id] == 0:
            visit(task_id, [])
    return cycles


def submission_groups(tasks: list[dict]) -> dict[str, list[str]]:
    """Group task ids by resource request. This does not submit them."""
    groups: dict[str, list[str]] = {}
    for task in tasks:
        resources = task.get("resources") or {}
        key = f"gpu{resources.get('gpus', 0)}:{task.get('walltime', 'unset')}"
        groups.setdefault(key, []).append(task["task_id"])
    return groups


def recover_drugprot_provenance(directory: Path) -> dict:
    """Separate files that were read from values reconstructed from the writer."""
    completion = directory / "prediction_complete.json"
    history = directory / "training_history.jsonl"
    process = next(directory.glob("processes/*_train/status.json"), None)
    verified = {
        "completion_record": completion.is_file(),
        "history_present": history.is_file(),
        "process_status_present": process is not None,
        "checkpoint_bytes": (directory / "checkpoints" / "best" / "checkpoint.pt").stat().st_size if (directory / "checkpoints" / "best" / "checkpoint.pt").is_file() else 0,
    }
    if process is not None:
        verified["process"] = json.loads(process.read_text(encoding="utf-8"))
    reconstructed = {
        "model_id": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "revision": "e1354b7a3a09615f6aba48dfad4b7a613eef7062",
        "input": "data/inputs/drugprot/target_anchored_v1__biomedbert_e1354b7a/contexts.jsonl",
        "source": "formal_train.py constants for formal_20261115",
        "stored_inside_checkpoint": False,
    }
    return {
        "directly_verified": verified,
        "reconstructed_from_writer": reconstructed,
        "limitation": "The checkpoint does not store an input or model fingerprint, so reuse is defensible only as a completed run of that writer.",
    }


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
