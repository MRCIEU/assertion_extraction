"""Training horizon, population selection, generation requests and scheduler commands.

This module does not load a model and it does not call sbatch. A requested GPU
count is not a distributed world size.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from schema_study.learning_conditions import RESEARCH_PREDICATES
from schema_study.predictions import parse_prediction_text
from schema_study.production_records import (
    all_view_records,
    curriculum_records,
    negative_population,
    one_view_records,
    subset_document_ids,
    unseen_split,
    view_names,
)

ACCOUNT = "brics.b5ac"
PARTITION = "workq"
QOS = "normal"
EFFECTIVE_BATCH = 16


def assert_runtime_world(requested_gpus: int, initialised_world: int) -> None:
    """The process group must already match the launch. A GPU request is not a world size."""
    if requested_gpus != initialised_world:
        raise RuntimeError(
            f"Requested {requested_gpus} GPUs but the initialised world size is {initialised_world}."
        )


def placement(model_name: str, requested_gpus: int) -> dict:
    """DDP replicates parameters. It does not divide the base weights across devices."""
    if model_name == "qwen32b" and requested_gpus >= 2:
        return {
            "method": "fsdp_full_shard",
            "parameter_sharding": True,
            "replicates_full_parameters": False,
            "device_map": None,
            "quantization": "none",
            "precision": "bf16",
            "reason": "BF16 Qwen3-32B is not assumed to fit on one GH200 with activations. FSDP shards parameters. DDP would replicate them.",
        }
    if requested_gpus > 1:
        return {
            "method": "distributed_data_parallel",
            "parameter_sharding": False,
            "replicates_full_parameters": True,
            "device_map": None,
            "quantization": "none",
            "precision": "bf16",
            "reason": "DDP keeps a full copy of the parameters on every participating GPU.",
        }
    return {
        "method": "single_process",
        "parameter_sharding": False,
        "replicates_full_parameters": False,
        "device_map": None,
        "quantization": "none",
        "precision": "bf16",
        "reason": "One process owns the full BF16 model.",
    }


def optimiser_horizon(examples_per_epoch: int, epochs: int, micro_batch: int, world_size: int, effective_batch: int = EFFECTIVE_BATCH) -> dict:
    """Drop a trailing group that does not fill the global effective batch."""
    if examples_per_epoch < 0 or epochs < 1 or micro_batch < 1 or world_size < 1:
        raise ValueError("The training horizon needs a positive population, epoch count and world size.")
    if effective_batch % (micro_batch * world_size) != 0:
        raise ValueError("Micro-batch and world size must divide the global effective batch of 16.")
    accumulation = effective_batch // (micro_batch * world_size)
    updates_per_epoch = examples_per_epoch // effective_batch
    remainder = examples_per_epoch % effective_batch
    if updates_per_epoch < 1:
        raise ValueError("The population does not fill one global effective batch.")
    return {
        "examples_per_epoch": examples_per_epoch,
        "epochs": epochs,
        "micro_batch": micro_batch,
        "world_size": world_size,
        "accumulation": accumulation,
        "effective_batch": effective_batch,
        "updates_per_epoch": updates_per_epoch,
        "remainder_dropped": remainder,
        "total_updates": updates_per_epoch * epochs,
        "incomplete_groups": "dropped",
    }


def horizon_for_task(task: dict, candidates: int, views: int, world_size: int, sequence_examples: int | None = None) -> dict:
    """Principal Qwen LoRA uses three epochs. Curriculum is one matched pass, not three copies."""
    family = task["family"]
    if family == "curriculum":
        if not sequence_examples:
            raise ValueError("Curriculum needs the length of its data stream.")
        horizon = optimiser_horizon(sequence_examples, 1, 1, world_size)
        horizon["schedule"] = "single_matched_curriculum_pass"
    elif family == "schema_conditioned_lora":
        per_epoch = candidates if task.get("exposure") == "resource_efficient_one_view" else candidates * views
        horizon = optimiser_horizon(per_epoch, 3, 1, world_size)
        horizon["schedule"] = task.get("exposure")
    else:
        horizon = optimiser_horizon(candidates, int(task.get("configuration", {}).get("max_epochs") or 3), 1, world_size)
        horizon["schedule"] = "independent_epochs"
    horizon["family"] = family
    return horizon


def select_population(task: dict, rows: list[dict]) -> dict:
    """Return the supervised rows for one family. BioRED shared-schema training is not the DrugProt native view."""
    rows = [_normalise(row) for row in rows]
    family = task["family"]
    if family == "schema_conditioned_lora":
        views = view_names("drugprot" if task.get("dataset") == "drugprot" else "biored")
        train_rows = [row for row in rows if row["split"] == "train"]
        dev_rows = [row for row in rows if row["split"] == "dev"]
        test_rows = [row for row in rows if row["split"] == "test"]
        labels = {(row["candidate_id"], row["entity"], row["predicate"]): (row.get("assistant") or {}).get("labels", []) for row in train_rows + dev_rows + test_rows}
        candidates = _unique_candidates(train_rows)
        dev_candidates = _unique_candidates(dev_rows)
        if task.get("exposure") == "resource_efficient_one_view":
            selected = []
            for epoch in range(3):
                selected.extend(one_view_records(candidates, views, epoch, int(task.get("seed") or 13)))
            evaluation = one_view_records(dev_candidates, views, 0, int(task.get("seed") or 13))
            kind = "resource_efficient_one_view"
        elif task.get("exposure") == "exposure_controlled_all_views":
            selected = all_view_records(candidates, views)
            evaluation = all_view_records(dev_candidates, views)
            kind = "exposure_controlled_all_views"
        else:
            raise ValueError("The shared-schema exposure is not one of the two agreed definitions.")
        test_candidates = _unique_candidates(test_rows)
        final_rows = one_view_records(test_candidates, views, 0, int(task.get("seed") or 13)) if kind == "resource_efficient_one_view" else all_view_records(test_candidates, views)
        development = _attach_view_labels(evaluation, labels) if evaluation else []
        final_evaluation = _attach_view_labels(final_rows, labels) if final_rows else []
        return {
            "kind": kind,
            "rows": _attach_view_labels(selected, labels),
            "development": development,
            "final_evaluation": final_evaluation,
            "evaluation": development,
            "views": views,
            "candidates": len(candidates),
        }
    if family == "curriculum":
        train_rows = [row for row in rows if row["split"] == "train" and row["entity"] == task["entity"]]
        grouped = {predicate: [row for row in train_rows if row.get("predicate") == predicate] for predicate in RESEARCH_PREDICATES}
        ordered = curriculum_records(task["order"], grouped)
        development = [row for row in rows if row["split"] == "dev" and row["entity"] == task["entity"] and row["predicate"] == "P3"]
        final_evaluation = [row for row in rows if row["split"] == "test" and row["entity"] == task["entity"] and row["predicate"] == "P3"]
        return {"kind": task["order"], "rows": ordered, "development": development, "final_evaluation": final_evaluation, "evaluation": development, "candidates": len({row["candidate_id"] for row in train_rows})}
    if family == "unseen_schema":
        train_rows = [row for row in rows if row["split"] == "train"]
        split = unseen_split(task["withheld"], train_rows)
        def withheld_row(row: dict, split: str) -> bool:
            return row["split"] == split and (row["entity"] == task["withheld"] or row["predicate"] == task["withheld"])

        development = [row for row in rows if withheld_row(row, "dev")]
        final_evaluation = [row for row in rows if withheld_row(row, "test")]
        return {
            "kind": "unseen",
            "rows": split["train"],
            "development": development,
            "final_evaluation": final_evaluation,
            "evaluation": development,
            "withheld": task["withheld"],
        }
    if family in {"hierarchical_multitask", "hierarchical_consistency"}:
        return _hierarchical_population(task, rows)
    if family == "data_efficiency":
        scoped = [row for row in rows if _same_condition(row, task.get("entity"), task.get("predicate"))]
        documents = subset_document_ids([row["document_id"] for row in scoped if row.get("split", "train") == "train"], int(task["fraction"]), int(task["seed"]))
        allowed = set(documents)
        train = [row for row in scoped if row.get("split", "train") == "train" and row["document_id"] in allowed]
        development = [row for row in scoped if row.get("split") == "dev"]
        final_evaluation = [row for row in scoped if row.get("split") == "test"]
        if any(row["document_id"] not in allowed for row in train):
            raise RuntimeError("A training row is outside the nested document subset.")
        if any(row["split"] == "test" for row in development):
            raise RuntimeError("A test row entered development selection.")
        return {"kind": f"fraction_{task['fraction']}", "rows": train, "development": development, "final_evaluation": final_evaluation, "evaluation": development, "documents": documents}
    if family == "negative_sampling":
        scoped = [row for row in rows if _same_condition(row, task.get("entity"), task.get("predicate"))]
        train = [row for row in scoped if row.get("split", "train") == "train"]
        chosen = negative_population(train, task["strategy"], int(task["seed"]))
        development = [row for row in scoped if row.get("split") == "dev"]
        final_evaluation = [row for row in scoped if row.get("split") == "test"]
        return {"kind": task["strategy"], "rows": chosen["rows"], "development": development, "final_evaluation": final_evaluation, "evaluation": development, "negative_ids": chosen["negative_ids"]}
    entity = task.get("entity")
    predicate = task.get("predicate")
    matched = [row for row in rows if _same_condition(row, entity, predicate)]
    selected = [row for row in matched if row.get("split", "train") == "train"]
    development = [row for row in matched if row.get("split") == "dev"]
    final_evaluation = [row for row in matched if row.get("split") == "test"]
    external = []
    if task.get("dataset") == "biored":
        external = [row for row in matched if row.get("split") == "bc8"]
    return {
        "kind": "independent",
        "rows": selected,
        "development": development,
        "final_evaluation": final_evaluation,
        "external_evaluation": external,
        "evaluation": development,
        "candidates": len(selected),
    }


def _hierarchical_population(task: dict, rows: list[dict]) -> dict:
    """One training example carries the four predicate targets for the same candidate."""
    def collect(split: str) -> list[dict]:
        grouped: dict[str, dict] = {}
        for row in rows:
            if row["split"] != split or row["entity"] != task["entity"]:
                continue
            slot = grouped.setdefault(row["candidate_id"], {"candidate_id": row["candidate_id"], "entity": task["entity"], "document_id": row["document_id"], "labels_by_predicate": {}})
            slot["labels_by_predicate"][row["predicate"]] = (row.get("assistant") or {}).get("labels", [])
        complete = [row for row in grouped.values() if set(RESEARCH_PREDICATES).issubset(row["labels_by_predicate"])]
        return complete

    train = collect("train")
    development = collect("dev")
    final_evaluation = collect("test")
    return {"kind": task["family"], "rows": train, "development": development, "final_evaluation": final_evaluation, "evaluation": development, "candidates": len(train)}


def _unique_candidates(rows: list[dict]) -> list[dict]:
    chosen = []
    seen = set()
    for row in rows:
        if row["candidate_id"] in seen:
            continue
        seen.add(row["candidate_id"])
        chosen.append({"candidate_id": row["candidate_id"], "document_id": row.get("document_id")})
    return chosen


def _attach_view_labels(records: list[dict], labels: dict[tuple, list]) -> list[dict]:
    attached = []
    for record in records:
        entity, predicate = str(record["view"]).split("/")
        key = (record["candidate_id"], entity, predicate)
        if key not in labels:
            raise KeyError(f"{record['candidate_id']} has no {entity}/{predicate} supervision.")
        attached.append({**record, "entity": entity, "predicate": predicate, "assistant": {"labels": labels[key]}})
    return attached


def _same_condition(row: dict, entity: str | None, predicate: str | None) -> bool:
    if entity and row.get("entity") not in {None, entity}:
        return False
    if predicate and row.get("predicate") not in {None, predicate}:
        return False
    return True


def _normalise(row: dict) -> dict:
    document = row.get("document_id")
    if not document and row.get("candidate_id"):
        parts = str(row["candidate_id"]).split(":")
        document = parts[1] if len(parts) > 2 else row["candidate_id"]
    labels = (row.get("assistant") or {}).get("labels", [])
    positive = row["positive"] if "positive" in row else bool(labels)
    return {
        **row,
        "entity": row.get("entity") or row.get("entity_level"),
        "predicate": row.get("predicate") or row.get("predicate_level"),
        "document_id": document,
        "split": row.get("split", "train"),
        "positive": positive,
    }


def sensitivity_changes(baseline: dict, proposed: dict) -> dict:
    """A sensitivity task must change the named factor and leave the other principal settings in place."""
    changed = {key: proposed[key] for key in ("learning_rate", "rank") if proposed.get(key) != baseline.get(key)}
    if not changed:
        raise ValueError("The sensitivity configuration is identical to the principal baseline.")
    return changed


def training_status(step: int, total_updates: int, selection_recorded: bool, failed: bool = False, invalid_checkpoint: bool = False) -> str:
    if invalid_checkpoint:
        return "invalid_or_incomplete_checkpoint"
    if failed:
        return "failed"
    if step < total_updates:
        return "valid_continuation_required"
    if not selection_recorded:
        return "valid_continuation_required"
    return "training_completed"


NEGATIVE_LABELS = {"NO_RELATION", "No_Relation", "no_relation"}
DRUGPROT_DEVELOPMENT_DECISION = {
    "rule": "independent_sigmoid",
    "threshold": 0.5,
    "negative_class": None,
    "negative_class_counted": False,
}


def positive_labels(labels: list[str]) -> set[str]:
    """The negative class is the absence of a positive relation. It is not a positive prediction."""
    return {label for label in labels if label not in NEGATIVE_LABELS}


def positive_micro_f1(rows: list[dict]) -> dict:
    """Corpus micro-F1 from pooled positive labels. Empty predictions contribute false negatives."""
    true_positive = false_positive = false_negative = 0
    for row in rows:
        gold = positive_labels(list(row.get("gold") or []))
        predicted = positive_labels(list(row.get("predicted") or [])) if row.get("status") == "valid" else set()
        true_positive += len(gold & predicted)
        false_positive += len(predicted - gold)
        false_negative += len(gold - predicted)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "document_f1_averaged": False, "negative_class_counted": False}


def require_production_device(policy: str, cuda_available: bool, initialised_world: int) -> dict:
    """A production GPU policy is rejected when the process is still on CPU."""
    if policy == "cpu_fixture":
        return {"device": "cpu", "precision": "fp32", "world": 1, "parameter_sharding": False, "moved_unsharded_before_wrap": False}
    if policy == "single_gpu":
        if not cuda_available or initialised_world != 1:
            raise RuntimeError("A single-GPU production task remained on CPU or was given a distributed world.")
        return {"device": "cuda:0", "precision": "bf16", "world": 1, "parameter_sharding": False, "moved_unsharded_before_wrap": False}
    if policy == "ddp":
        if not cuda_available or initialised_world < 2:
            raise RuntimeError("Data-parallel training requires an initialised CUDA process group.")
        return {"device": "cuda:local", "precision": "bf16", "world": initialised_world, "parameter_sharding": False, "replicates_full_parameters": True, "moved_unsharded_before_wrap": True}
    if policy == "fsdp":
        if not cuda_available or initialised_world < 2:
            raise RuntimeError("Parameter sharding requires an initialised CUDA process group.")
        return {"device": "cuda:local", "precision": "bf16", "world": initialised_world, "parameter_sharding": True, "moved_unsharded_before_wrap": False}
    raise ValueError(policy)


def rank_partitions(example_ids: list[str], *, world: int, micro_batch: int, seed: int, epoch: int, shuffle: bool) -> dict:
    """Split one global order into disjoint rank sequences that rebuild the effective batch of 16."""
    if world < 1 or micro_batch < 1:
        raise ValueError("Rank partitioning needs a positive world size and micro-batch.")
    if EFFECTIVE_BATCH % (micro_batch * world) != 0:
        raise ValueError("Micro-batch and world size must divide the global effective batch of 16.")
    ordered = list(example_ids)
    if shuffle:
        random.Random(seed + epoch).shuffle(ordered)
    accumulation = EFFECTIVE_BATCH // (micro_batch * world)
    kept_count = (len(ordered) // EFFECTIVE_BATCH) * EFFECTIVE_BATCH
    kept = ordered[:kept_count]
    ranks = {rank: [] for rank in range(world)}
    for start in range(0, kept_count, EFFECTIVE_BATCH):
        window = kept[start:start + EFFECTIVE_BATCH]
        for step in range(accumulation):
            for rank in range(world):
                offset = step * world * micro_batch + rank * micro_batch
                ranks[rank].extend(window[offset:offset + micro_batch])
    assigned = [example for rank in range(world) for example in ranks[rank]]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(kept):
        raise RuntimeError("Rank assignment repeated or dropped a kept training example.")
    return {
        "ordered_before_drop": ordered,
        "kept": kept,
        "dropped": ordered[kept_count:],
        "ranks": ranks,
        "shuffle": shuffle,
        "epoch": epoch,
        "seed": seed,
        "effective_batch": EFFECTIVE_BATCH,
        "accumulation": accumulation,
        "micro_batch": micro_batch,
        "world": world,
        "remainder_policy": "dropped_after_epoch_order",
    }


def resume_rank_examples(rank_examples: list[str], position: int) -> list[str]:
    """Continuation starts at the next unconsumed rank example. Completed examples are not repeated."""
    if position < 0 or position > len(rank_examples):
        raise ValueError("The checkpoint position is outside the rank example sequence.")
    return rank_examples[position:]


ACTIVE_CONTROLLER_STATES = {"submitted", "pending", "running"}
GROUP_RELEASE_STATES = {"preflight_passed", "training_completed", "inference_completed", "completed"}
SCIENTIFIC_COMPLETE = {"training_completed", "inference_completed", "completed"}
RETRYABLE_SLURM = {"TIMEOUT", "NODE_FAIL", "PREEMPTED", "DEADLINE"}
TERMINAL_SLURM = {"FAILED", "CANCELLED", "OUT_OF_MEMORY", "BOOT_FAIL"}
MONITORING_SECONDS = 300
HELD_FAMILIES = set()


def _group_released(entry: dict) -> bool:
    """A passed preflight releases the group. Adapter inference still waits for the selected checkpoint."""
    return entry.get("scientific") in GROUP_RELEASE_STATES or entry.get("state") in GROUP_RELEASE_STATES


def _ready_name(name: str, evidence) -> bool:
    if evidence is None:
        return False
    if isinstance(evidence, dict):
        return bool((evidence.get(name) or {}).get("ready"))
    return name in evidence


def unresolved_prerequisites(task: dict, state: dict, evidence=None) -> list[str]:
    """A named input is ready only when its own evidence says so. Another corpus does not satisfy it."""
    missing = []
    for name in task.get("depends_on", []):
        if _ready_name(name, evidence):
            continue
        dependency = state.get(name, {})
        if name == task.get("trains_from") or name not in {"approval", "inputs_prepared", "model_assets", "historical_drugprot_checkpoint"} and dependency:
            if dependency.get("state") in SCIENTIFIC_COMPLETE or dependency.get("scientific") in SCIENTIFIC_COMPLETE:
                continue
        missing.append(name)
    return missing


def interpret_slurm(job_id: str, queued: dict[str, str], sacct_state: str | None) -> str:
    """A missing queue entry is not completion. sacct must name a terminal state."""
    if job_id in queued:
        observed = queued[job_id]
        if observed == "RUNNING":
            return "running"
        return "pending"
    if not sacct_state:
        return "unknown"
    base = sacct_state.split()[0]
    if base == "COMPLETED":
        return "slurm_completed"
    if base in RETRYABLE_SLURM:
        return "retryable"
    if base in TERMINAL_SLURM:
        return "terminal"
    if base in {"PENDING", "RUNNING", "CONFIGURING", "SUSPENDED"}:
        return "running" if base == "RUNNING" else "pending"
    return "unknown"


def _failure_entry(current: dict, task_id: str, reason: str, *, systematic: bool = False, group: str | None = None) -> dict:
    """Apply the bounded recovery policy and keep a systematic group block with the entry."""
    failed = record_failure({task_id: current}, task_id, reason, systematic=systematic, group=group)
    entry = failed[task_id]
    entry["job_id"] = current.get("job_id")
    if "_groups" in failed:
        entry["_group_block"] = failed["_groups"]
    return entry


def merge_group_block(state: dict, entry: dict) -> dict:
    """Copy a systematic block into the persistent state. The entry itself stays one task."""
    block = entry.pop("_group_block", None)
    if block:
        for key, value in block.items():
            state.setdefault("_groups", {}).setdefault(key, {}).update(value)
    return entry


def reconcile_entry(entry: dict, scheduler: str, output: dict | None, *, task_id: str = "", group: str | None = None, slurm_state: str = "", elapsed_seconds: int = 0, observation_seconds: float | None = None) -> dict:
    """Join one Slurm observation with the scientific output. Queue time is not useful computation."""
    del elapsed_seconds
    current = dict(entry)
    if slurm_state:
        current["slurm_state"] = slurm_state.split()[0]
    if scheduler == "pending":
        current["state"] = "pending"
        return current
    if scheduler == "running":
        current["state"] = "running"
        started = None if not output else output.get("useful_computation_started")
        if output and output.get("technical_preflight") and started is not None:
            observed = float(observation_seconds) if observation_seconds is not None else time.time() - float(started)
            if observed >= MONITORING_SECONDS:
                current["scientific"] = "preflight_passed"
                current["monitoring_guarantees_completion"] = False
                current["useful_observation_seconds"] = observed
        return current
    if scheduler == "unknown":
        current["scheduler"] = "unknown"
        return current
    if scheduler == "retryable":
        return _failure_entry(current, task_id, current.get("slurm_state", "retryable"), group=group)
    if scheduler == "terminal":
        return _failure_entry(current, task_id, current.get("slurm_state", "terminal"), systematic=current.get("slurm_state") == "OUT_OF_MEMORY", group=group)
    status = (output or {}).get("scientific")
    if status == "training_completed" and not (output.get("selection_recorded") and output.get("checkpoint_validated")):
        current["state"] = "valid_continuation_required"
        current["scientific"] = "preflight_passed" if output.get("technical_preflight") or output.get("preflight_passed") else "valid_continuation_required"
        current["reason"] = "training_completion_missing_selection_or_checkpoint"
        return current
    if status == "inference_completed" and not output.get("predictions_usable"):
        current["state"] = "valid_continuation_required"
        current["reason"] = "inference_completion_missing_prediction_content"
        return current
    if status in {"preflight_passed", "training_completed", "inference_completed", "valid_continuation_required"}:
        current["state"] = status
        if status == "valid_continuation_required" and (output.get("preflight_passed") or output.get("technical_preflight") or current.get("scientific") == "preflight_passed"):
            current["scientific"] = "preflight_passed"
        else:
            current["scientific"] = status
        current["monitoring_guarantees_completion"] = False
        return current
    if status == "invalid":
        return _failure_entry(current, task_id, "scientific_invalid", systematic=True, group=group)
    return _failure_entry(current, task_id, "slurm_completed_without_scientific_record", group=group)


def programme_remaining(tasks: list[dict], state: dict) -> bool:
    """True while any approved task is unfinished. An idle submission turn does not clear it."""
    for task in tasks:
        if "blocked" in task.get("statuses", []) or task.get("family") in HELD_FAMILIES:
            continue
        current = state.get(task["task_id"], {}).get("state", "unsubmitted")
        if current not in SCIENTIFIC_COMPLETE | {"terminal_failure"}:
            return True
    return False


def rank0_generation(rank: int, generate, barrier):
    """Rank 0 generates locally. Every rank enters the barrier before anyone leaves the collective."""
    produced = generate() if rank == 0 else None
    barrier()
    return produced


def coordinate_rank_evaluation(rank: int, compute, barrier, box: dict) -> float:
    """Every rank waits before the score is read. Rank 0 does not publish early."""
    if rank == 0:
        box["score"] = float(compute())
    barrier()
    if "score" not in box:
        raise RuntimeError("Development evaluation broadcast started before every rank had entered the collective.")
    return float(box["score"])


def validate_request_shards(directory: Path) -> dict:
    """Reject a shard directory that has no completion record, a bad checksum, or a repeated request."""
    import hashlib

    marker = directory / "COMPLETE"
    if not marker.is_file():
        return {"ready": False, "reason": "incomplete_shards"}
    record = json.loads(marker.read_text(encoding="utf-8"))
    identifiers = []
    mismatches = 0
    for name in record.get("shards") or []:
        path = directory / name
        if not path.is_file():
            return {"ready": False, "reason": "missing_shard", "shard": name}
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            digest = hashlib.sha256(row["rendered"].encode("utf-8")).hexdigest()
            if digest != row.get("rendered_sha256"):
                mismatches += 1
            identifiers.append(row["request_id"])
    duplicate = len(identifiers) - len(set(identifiers))
    expected = int(record.get("requests", -1))
    ready = mismatches == 0 and duplicate == 0 and len(identifiers) == expected
    return {
        "ready": ready,
        "requests": len(identifiers),
        "expected": expected,
        "duplicate_requests": duplicate,
        "checksum_failures": mismatches,
        "reason": "ok" if ready else "population_mismatch",
    }


def publish_validation(directory: Path) -> dict:
    """Write the checksum record beside the shards. A directory without one is not ready."""
    result = validate_request_shards(directory)
    temporary = directory / ".VALIDATION.json.tmp"
    temporary.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(directory / "VALIDATION.json")
    return result


def publish_scientific_status(directory: Path, payload: dict) -> None:
    """Publish the task's own status. A Slurm accounting line is not this record."""
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / ".scientific_status.json.tmp"
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(directory / "scientific_status.json")


def verify_peft_adapter(directory: Path, expected_base: str | None = None) -> dict:
    """The exported adapter must be a PEFT LoRA directory for the pinned base revision."""
    config_path = directory / "adapter_config.json"
    weights = directory / "adapter_model.bin"
    if not config_path.is_file() or not weights.is_file() or weights.stat().st_size <= 0:
        return {"ready": False, "reason": "adapter_export_missing"}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA":
        return {"ready": False, "reason": "unsupported_peft_type"}
    base = config.get("base_model_name_or_path")
    if expected_base and base != expected_base:
        return {"ready": False, "reason": "base_model_mismatch", "base_model_name_or_path": base}
    return {"ready": True, "peft_type": "LORA", "base_model_name_or_path": base}


def probe_readiness(task: dict) -> dict:
    """Check this task's own files. A ready BioRED input does not mark DrugProt ready."""
    evidence = {}
    info = task.get("input") or {}
    if task.get("dataset"):
        key = f"inputs:{task['dataset']}:{info.get('kind', 'contexts')}"
        path = Path(info["path"]) if info.get("path") else None
        evidence[key] = {"ready": bool(path and path.is_file() and info.get("fingerprint")), "fingerprint": info.get("fingerprint")}
    model = task.get("model")
    if model:
        ready = False
        try:
            from schema_study.production_assets import ENCODERS, GENERATIVE

            spec = GENERATIVE.get(model) or ENCODERS.get(model)
            if model == "deberta_base":
                from schema_study.production_assets import DEBERTA_REVISION, MODELS

                location = MODELS / "MoritzLaurer__DeBERTa-v3-base-mnli-fever-anli" / DEBERTA_REVISION
                ready = location.is_dir() and task.get("model_revision") == DEBERTA_REVISION
            elif spec:
                location = Path(spec.get("path") or spec["model"])
                ready = location.is_dir() and task.get("model_revision") == spec.get("revision")
        except Exception:
            ready = False
        evidence[f"model:{model}"] = {"ready": ready, "revision": task.get("model_revision")}
    if task.get("new_training") is False:
        from schema_study.production_plan import drugprot_checkpoint_reusable

        historical = task.get("historical_predicate")
        directory = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/runs/formal/formal_20261115/train/drugprot/native") / str(historical) / f"seed{task.get('seed')}"
        evidence[f"historical:{historical}:{task.get('seed')}"] = {
            "ready": drugprot_checkpoint_reusable(directory),
            "fingerprint": None,
            "provenance": "reconstructed_completion_record",
        }
    if task.get("population_filter"):
        manifest = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/runs/formal/formal_matched_e0_e7/generation_inputs/validation_manifest.json")
        ready = manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")).get("ready") is True
        evidence[f"population:{task['population_filter']}:{task.get('dataset')}"] = {"ready": ready}
    elif task["family"] in {"zero_shot", "few_shot", "adapter_inference", "constrained", "thinking", "self_consistency", "evidence", "qualified", "candidate_scoring", "gpt_oss"}:
        source = "few_shot" if task["family"] == "few_shot" else "zero_shot"
        directory = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/runs/formal/formal_matched_e0_e7/generation_inputs") / source / str(task.get("dataset")) / str(task.get("entity")) / str(task.get("predicate"))
        validation = directory / "VALIDATION.json"
        ready = False
        if validation.is_file():
            ready = json.loads(validation.read_text(encoding="utf-8")).get("ready") is True
        evidence[f"shards:{source}:{task.get('dataset')}:{task.get('entity')}:{task.get('predicate')}"] = {"ready": ready}
    if task.get("adapter_path"):
        evidence[f"adapter:{task.get('trains_from')}"] = verify_peft_adapter(Path(task["adapter_path"]), task.get("model_revision"))
    if task.get("checkpoint_source"):
        checkpoint = Path(task["checkpoint_source"])
        evidence[f"checkpoint:{task.get('trains_from')}"] = {"ready": checkpoint.is_file() and checkpoint.stat().st_size > 0}
    return evidence


def controller_decisions(tasks: list[dict], state: dict, running: int, pending: int, verified_external: set[str] | None = None) -> list[dict]:
    """Choose submissions from persistent state. An active or completed task is not submitted again."""
    from schema_study.production_plan import MAX_QUEUED_JOBS

    verified = verified_external or set()
    seen_groups = set()
    for task in tasks:
        if "representative" in task:
            continue
        if "blocked" in task.get("statuses", []):
            task["representative"] = False
            continue
        key = f"{task.get('family')}|{task.get('model')}|{(task.get('resources') or {}).get('gpus')}"
        task["representative_group"] = key
        task["representative"] = key not in seen_groups
        seen_groups.add(key)
    ready_representatives = {
        task.get("representative_group")
        for task in tasks
        if task.get("representative") and _group_released(state.get(task["task_id"], {}))
    }
    decisions = []
    proposed = 0
    for task in tasks:
        current = state.get(task["task_id"], {}).get("state", "unsubmitted")
        if current in ACTIVE_CONTROLLER_STATES or current in SCIENTIFIC_COMPLETE:
            decisions.append({"task_id": task["task_id"], "action": "skip", "reason": current})
            continue
        if task.get("family") in HELD_FAMILIES:
            decisions.append({"task_id": task["task_id"], "action": "skip", "reason": "held_pending_independent_evidence"})
            continue
        if current in {"terminal_failure", "blocked"} or "blocked" in task.get("statuses", []):
            decisions.append({"task_id": task["task_id"], "action": "skip", "reason": current if current != "unsubmitted" else "blocked"})
            continue
        group_block = (state.get("_groups") or {}).get(task.get("representative_group") or "")
        if group_block and group_block.get("blocked"):
            decisions.append({"task_id": task["task_id"], "action": "skip", "reason": "systematic_group_failure"})
            continue
        if task.get("representative") is False and task.get("representative_group") not in ready_representatives:
            decisions.append({"task_id": task["task_id"], "action": "skip", "reason": "representative_pending"})
            continue
        missing = unresolved_prerequisites(task, state, verified)
        if missing:
            decisions.append({"task_id": task["task_id"], "action": "skip", "reason": "prerequisite", "missing": missing})
            continue
        if running + pending + proposed + 1 > MAX_QUEUED_JOBS:
            decisions.append({"task_id": task["task_id"], "action": "defer", "reason": "user_job_limit_512"})
            continue
        proposed += 1
        decisions.append({"task_id": task["task_id"], "action": "submit"})
    return decisions


def record_submission(state: dict, task_id: str, job_id: str, manifest_revision: str) -> dict:
    """Persist one Slurm identifier. A second submission of an active task is rejected."""
    current = state.get(task_id, {}).get("state", "unsubmitted")
    if current in ACTIVE_CONTROLLER_STATES or current in SCIENTIFIC_COMPLETE:
        raise RuntimeError(f"{task_id} is already {current}.")
    previous = state.get(task_id, {})
    state[task_id] = {
        "state": "submitted",
        "job_id": job_id,
        "manifest_revision": manifest_revision,
        "attempts": int(previous.get("attempts", 0)),
        "group": previous.get("group"),
    }
    if previous.get("scientific"):
        state[task_id]["scientific"] = previous["scientific"]
    return state


def record_failure(state: dict, task_id: str, reason: str, *, systematic: bool = False, group: str | None = None, limit: int = 2) -> dict:
    """Count a transient failure. A systematic failure closes the homogeneous group."""
    previous = state.get(task_id, {})
    attempts = int(previous.get("attempts", 0)) + 1
    assigned = group or previous.get("group")
    closed = systematic or attempts > limit
    state[task_id] = {
        "state": "terminal_failure" if closed else "retryable_failure",
        "attempts": attempts,
        "reason": reason,
        "systematic": systematic,
        "group": assigned,
    }
    if systematic and assigned:
        state.setdefault("_groups", {}).setdefault(assigned, {})["blocked"] = reason
    return state


def controller_transaction(path: Path):
    """Hold one lock across reconciliation and submission, not only across the last write."""
    from contextlib import contextmanager

    @contextmanager
    def _locked():
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix(".transaction.lock")
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError("Another controller invocation is reconciling this programme.") from exc
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            yield
        finally:
            os.close(descriptor)
            lock.unlink(missing_ok=True)

    return _locked()


def save_controller_state(path: Path, state: dict, *, external_lock: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    descriptor = None
    if not external_lock:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError("Another controller is writing the task state.") from exc
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
            lock.unlink(missing_ok=True)


def load_controller_state(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def attach_stored_candidate(raw: str | None, candidate_id: str) -> str | None:
    """The loss target does not contain the candidate identifier. Attach the stored one before parsing."""
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(payload, dict) and "labels" in payload and "candidate_id" not in payload:
        payload = {"candidate_id": candidate_id, **payload}
        return json.dumps(payload, ensure_ascii=False)
    return raw


def interpret_generation(raw: str | None, candidate_id: str, schema, level: str) -> dict:
    """Parse with the corpus parser. Invalid science is not a retryable execution failure."""
    parsed = parse_prediction_text(attach_stored_candidate(raw, candidate_id), expected_candidate_id=candidate_id, schema=schema, level=level)
    labels = parsed["predicted_labels"]
    if parsed["output_status"] == "valid" and not labels:
        ledger = "valid_empty"
    elif parsed["output_status"] == "valid":
        ledger = "valid"
    else:
        ledger = "invalid"
    return {**parsed, "raw": raw, "ledger_status": ledger, "retry": False}


def constrained_sampling(schema_json: dict) -> dict:
    """vLLM 0.33 structured outputs. This is not an unconstrained sample."""
    return {"temperature": 0.0, "max_tokens": 256, "structured_outputs": {"json": schema_json}}


def thinking_request(user: str) -> dict:
    return {"enable_thinking": True, "messages": [{"role": "user", "content": user}], "final_answer_separated": True}


def self_consistency(samples: list[dict], multi: bool) -> dict:
    """Five retained samples. One ordinary generation is not a self-consistency result."""
    if len(samples) != 5:
        raise ValueError("Self-consistency requires five samples.")
    valid = [sample for sample in samples if sample.get("ledger_status") in {"valid", "valid_empty"}]
    if not multi:
        counts: dict[tuple, int] = {}
        for sample in valid:
            key = tuple(sample.get("predicted_labels") or [])
            counts[key] = counts.get(key, 0) + 1
        chosen = max(counts, key=lambda item: (counts[item], item)) if counts else tuple()
        labels = list(chosen)
    else:
        votes: dict[str, int] = {}
        for sample in valid:
            for label in sample.get("predicted_labels") or []:
                votes[label] = votes.get(label, 0) + 1
        labels = [label for label, count in sorted(votes.items()) if count * 2 >= len(valid) and valid]
    return {"samples": samples, "valid_samples": len(valid), "labels": labels, "aggregation": "majority"}


def sequence_logprobs(token_logprobs: dict[str, list[float]]) -> dict[str, float]:
    """Sum the token log probabilities of each legal sequence. A supplied label score is not accepted."""
    if not token_logprobs or any(not values for values in token_logprobs.values()):
        raise RuntimeError("Candidate scoring requires token log probabilities for each legal sequence.")
    return {label: float(sum(values)) for label, values in token_logprobs.items()}


def require_legal_scores(scores: dict[str, float], legal_labels: list[str], multi: bool = False) -> dict:
    """Every legal candidate needs its own score. A prechosen label is not a score."""
    missing = [label for label in legal_labels if label not in scores]
    if missing or not scores:
        raise RuntimeError("Candidate scoring is missing a score for a legal output.")
    return score_candidates({label: scores[label] for label in legal_labels}, multi)


def score_candidates(scores: dict[str, float], multi: bool) -> dict:
    if not scores:
        raise ValueError("Candidate scoring needs a score for every legal output.")
    if not multi:
        chosen = max(scores, key=lambda label: (scores[label], label))
        return {"method": "argmax_legal_candidate", "scores": scores, "labels": [chosen]}
    return {"method": "positive_score", "scores": scores, "labels": [label for label, score in scores.items() if score > 0]}


def evidence_record(raw: str, source_text: str, spans: list[str]) -> dict:
    """A label without a source span is not an evidence-grounded assertion."""
    if not spans:
        return {"raw": raw, "spans": [], "unsupported": [], "complete": False, "reason": "label_only_response"}
    unsupported = [span for span in spans if span not in source_text]
    return {"raw": raw, "spans": spans, "unsupported": unsupported, "complete": not unsupported}


def prediction_usable(record: dict) -> bool:
    """A status without the generated response is not a scientific prediction."""
    return bool(
        record.get("request_id")
        and record.get("candidate_id")
        and "raw" in record
        and "predicted_labels" in record
        and record.get("status") in {"valid", "valid_empty", "invalid"}
    )


def prediction_record(request: dict, result: dict, task: dict, specification: dict, status: str) -> dict:
    """Retain the original response and the fields needed to reproduce the evaluation."""
    parsed = result.get("parsed") or {}
    return {
        "request_id": request["request_id"],
        "candidate_id": request.get("candidate_id"),
        "model": task.get("model") or request.get("model"),
        "model_revision": task.get("model_revision"),
        "tokenizer_revision": task.get("tokenizer_revision"),
        "adapter": task.get("adapter_path"),
        "checkpoint": task.get("adapter_path") or task.get("checkpoint"),
        "family": task.get("family"),
        "learning_condition": task.get("family"),
        "entity": task.get("entity") or request.get("entity_level"),
        "predicate": task.get("predicate") or request.get("predicate_level"),
        "seed": task.get("seed"),
        "raw": result.get("raw"),
        "predicted_labels": result.get("labels") if result.get("labels") is not None else result.get("predicted_labels"),
        "illegal_labels": result.get("illegal_labels", parsed.get("illegal_labels", [])),
        "status": status,
        "prompt_tokens": result.get("prompt_tokens"),
        "completion_tokens": result.get("completion_tokens"),
        "scores": result.get("scores"),
        "input_fingerprint": (task.get("input") or {}).get("fingerprint") or request.get("rendered_sha256"),
        "rendered": request.get("rendered") or request.get("prompt"),
        "rendered_sha256": request.get("rendered_sha256"),
        "decoding": specification.get("decoding"),
    }


def generation_requests(rows: list[dict]) -> list[dict]:
    """Every request carries the model input. A candidate identifier is not a prompt."""
    requests = []
    for row in rows:
        prompt = row.get("prompt")
        if not prompt or prompt == row.get("candidate_id"):
            raise ValueError(f"{row.get('candidate_id')} has no model input.")
        request = {
            "request_id": row["request_id"],
            "candidate_id": row["candidate_id"],
            "prompt": prompt,
            "model": row["model"],
            "decoding": row.get("decoding", "greedy"),
        }
        for key in ("schema_json", "source_text", "legal_labels", "multi", "spans"):
            if key in row:
                request[key] = row[key]
        requests.append(request)
    return requests


def write_shards(requests: list[dict], directory: Path, shard_size: int) -> list[Path]:
    if shard_size < 1:
        raise ValueError("Shard size must be positive.")
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(0, len(requests), shard_size):
        path = directory / f"shard_{index // shard_size:05d}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        chunk = requests[index:index + shard_size]
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in chunk), encoding="utf-8")
        temporary.replace(path)
        paths.append(path)
    return paths


def sbatch_argv(task: dict, script: str) -> list[str]:
    """The command that would submit one task. Calling this function does not submit it."""
    resources = task.get("resources") or {}
    gpus = int(resources.get("gpus") or 0)
    command = [
        "sbatch",
        f"--account={ACCOUNT}",
        f"--partition={PARTITION}",
        f"--qos={QOS}",
        f"--time={task.get('walltime', '04:00:00')}",
        "--cpus-per-task=8",
        f"--job-name=p1-{task['family'][:8]}",
        script,
        task["family"],
        task["task_id"],
    ]
    if gpus:
        command.insert(5, f"--gpus={gpus}")
    return command


def controller_requeue(job_id: str, authorised: bool, finished: bool, active_job_ids: list[str] | None = None) -> list[str] | None:
    """Schedule another bounded step while work remains. A quiet submission turn is not completion."""
    from schema_study.formal_launch import requeue_permitted

    if not requeue_permitted(authorised, finished):
        return None
    command = [
        "sbatch",
        "--parsable",
        "--begin=now+10minutes",
        "--account=brics.b5ac",
        "--partition=workq",
        "--qos=normal",
        "--time=01:00:00",
        "scripts/slurm/programme_controller.sbatch",
    ]
    if job_id and job_id != "0":
        command.insert(2, f"--dependency=afterany:{job_id}")
    return command


def context_sources(corpus: str) -> tuple[str, ...]:
    """BC8 targets are in the BC8 file. Their retrieved demonstrations are BioRED training contexts."""
    if corpus == "bc8":
        return ("bc8", "biored")
    return (corpus,)


def external_model_convention(model_name: str) -> dict:
    """GPT-OSS does not inherit Qwen LoRA modules or the Qwen chat template."""
    if model_name == "qwen36":
        return {"executable": False, "reason": "Transformers 4.57.5 does not recognise model_type qwen3_5."}
    if model_name == "gpt_oss_20b":
        return {
            "executable": False,
            "lora_target_modules": None,
            "uses_qwen_chat_template": False,
            "weight_loading": "requires_authorised_gpu_preflight",
        }
    raise KeyError(model_name)


def evaluation_released(dependency_status: str) -> bool:
    """A continuation segment does not release final evaluation."""
    return dependency_status == "training_completed"
