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


def scheduled_development_steps(total_updates: int, interval: int) -> tuple[int, ...]:
    """Epoch boundaries only. A short continuation segment does not add a selection step."""
    total = int(total_updates)
    every = int(interval)
    if total < 1 or every < 1:
        raise ValueError("Development scheduling needs a positive horizon and interval.")
    return tuple(step for step in range(every, total + 1, every))


def development_request_key(row: dict, predicate: str) -> tuple[str, str, str]:
    """One development request. The candidate id is not unique across views or predicates."""
    return (
        str(row.get("candidate_id")),
        str(row.get("entity_level") or row.get("entity") or ""),
        str(predicate),
    )


def development_evaluation_identity(
    task_id: str,
    optimiser_step: int,
    request_keys: list[tuple],
    *,
    checkpoint_fingerprint: str = "",
    population_fingerprint: str = "",
    prompt_fingerprint: str = "",
    generation_fingerprint: str = "",
    parser_version: str = "",
) -> str:
    """Bind a ledger to one checkpoint, one request population, and one decoding contract."""
    import hashlib

    payload = {
        "task_id": str(task_id),
        "optimiser_step": int(optimiser_step),
        "checkpoint_fingerprint": str(checkpoint_fingerprint),
        "population_fingerprint": str(population_fingerprint),
        "prompt_fingerprint": str(prompt_fingerprint),
        "generation_fingerprint": str(generation_fingerprint),
        "parser_version": str(parser_version),
        "requests": [list(key) for key in sorted(request_keys)],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:20]


def load_development_ledger(path: Path, evaluation_id: str, expected: set[tuple]) -> dict[tuple, dict]:
    """Read one checkpoint's ledger. Another checkpoint's rows are not completions."""
    found: dict[tuple, dict] = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("evaluation_id") != evaluation_id:
            raise RuntimeError("A development ledger contains a row from a different checkpoint or population.")
        key = tuple(row.get("request_key") or [])
        if len(key) != 3 or key not in expected:
            raise RuntimeError("A development ledger contains a request that is not in this evaluation.")
        if key in found:
            raise RuntimeError("A development ledger repeats a request.")
        found[key] = row
    return found


def record_development_boundary(directory: Path, optimiser_step: int, evaluation_id: str, complete: bool, requests: int) -> None:
    path = directory / "development_boundaries.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    current[str(int(optimiser_step))] = {
        "evaluation_id": evaluation_id,
        "complete": bool(complete),
        "requests": int(requests),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(current, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def development_boundary_complete(directory: Path, optimiser_step: int) -> bool:
    path = directory / "development_boundaries.json"
    if not path.is_file():
        return False
    row = json.loads(path.read_text(encoding="utf-8")).get(str(int(optimiser_step))) or {}
    return row.get("complete") is True


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
SCIENTIFIC_COMPLETE = {"training_completed", "inference_completed", "completed", "diagnostic_recorded"}
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


def _continuation_stalled(current: dict, output: dict) -> bool:
    """Two exits at the same optimiser step are not useful continuation."""
    end = output.get("steps")
    if end is None:
        end = output.get("optimiser_steps")
    start = current.get("continuation_step")
    current["continuation_started_step"] = start
    current["continuation_ended_step"] = end
    current["failure_signature"] = output.get("scientific")
    fingerprint = output.get("checkpoint_fingerprint")
    current["continuation_started_fingerprint"] = current.get("continuation_fingerprint")
    if fingerprint:
        current["continuation_ended_fingerprint"] = fingerprint
        current["continuation_fingerprint"] = fingerprint
    population_advanced = False
    if "population_covered" in output:
        covered = output.get("population_covered")
        previous_covered = current.get("population_covered")
        current["population_covered_started"] = previous_covered
        current["population_covered"] = covered
        population_advanced = covered is not None and (previous_covered is None or int(covered) > int(previous_covered))
    step_advanced = end is not None and (start is None or int(end) > int(start))
    if step_advanced or (end is None and population_advanced):
        stalled = False
    else:
        stalled = True
    if stalled:
        current["unproductive_continuations"] = int(current.get("unproductive_continuations") or 0) + 1
    else:
        current["unproductive_continuations"] = 0
        if end is not None:
            current["continuation_step"] = int(end)
    return int(current.get("unproductive_continuations") or 0) >= 2


def queue_ledger_acceptance(state: dict, tasks: list[dict]) -> bool:
    """A finished prediction task still needs identity acceptance. Completion does not publish that record."""
    from schema_study.programme_closure import ledger_acceptance_valid, scientific_provenance_valid

    needed = False
    for task in tasks:
        if task.get("family") not in {"zero_shot", "few_shot", "adapter_inference", "final_evaluation"}:
            continue
        entry = state.get(task["task_id"], {})
        if entry.get("state") != "inference_completed":
            continue
        corpus = str(task.get("dataset") or "")
        directory = Path(task.get("output") or "")
        if ledger_acceptance_valid(directory, corpus) and scientific_provenance_valid(directory, corpus):
            continue
        record = directory / "population_acceptance.json"
        if record.is_file():
            body = json.loads(record.read_text(encoding="utf-8"))
            cell = body.get(corpus) if isinstance(body.get(corpus), dict) else {}
            if cell.get("passed") is False and not entry.get("ledger_replacement"):
                state[task["task_id"]] = {
                    "state": "unsubmitted",
                    "attempts": int(entry.get("attempts") or 0),
                    "ledger_replacement": 1,
                    "reason": cell.get("reason") or "ledger_incompatible",
                    "job_id": entry.get("job_id"),
                }
                continue
        needed = True
    if needed:
        for task in tasks:
            if task.get("family") != "ledger_acceptance":
                continue
            current = state.get(task["task_id"], {}).get("state")
            if current in SCIENTIFIC_COMPLETE:
                state[task["task_id"]] = {"state": "unsubmitted", "attempts": 0}
    return needed


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
    if scheduler == "retryable" and str(current.get("slurm_state", "")).startswith("TIMEOUT"):
        if _continuation_stalled(current, output or {}):
            current["state"] = "terminal_failure"
            current["reason"] = "unproductive_continuation"
            current["systematic"] = False
            return current
        current["state"] = "valid_continuation_required"
        current["scientific"] = "preflight_passed" if (output or {}).get("technical_preflight") else current.get("scientific")
        current["continuation_segments"] = int(current.get("continuation_segments", 0)) + 1
        current["reason"] = "planned_walltime_continuation"
        return current
    if scheduler == "retryable":
        return _failure_entry(current, task_id, current.get("slurm_state", "retryable"), group=group)
    output_record = output or {}
    checkpoint_continues = output_record.get("checkpoint_validated") is True and (
        output_record.get("scientific") == "valid_continuation_required" or output_record.get("technical_preflight") is True
    )
    if scheduler == "terminal" and checkpoint_continues:
        if _continuation_stalled(current, output_record):
            current["state"] = "terminal_failure"
            current["reason"] = "unproductive_continuation"
            current["systematic"] = False
            return current
        current["state"] = "valid_continuation_required"
        current["scientific"] = "preflight_passed" if output.get("technical_preflight") else "valid_continuation_required"
        current["continuation_segments"] = int(current.get("continuation_segments", 0)) + 1
        current["reason"] = "checkpointed_exit_continues"
        current["attempts"] = int(current.get("attempts") or 0)
        return current
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
    if status == "diagnostic_recorded":
        current["state"] = "diagnostic_recorded"
        current["scientific"] = "diagnostic_recorded"
        current["monitoring_guarantees_completion"] = False
        return current
    if status == "invalid":
        return _failure_entry(current, task_id, "scientific_invalid", systematic=True, group=group)
    return _failure_entry(current, task_id, "slurm_completed_without_scientific_record", group=group)


def resolve_task_equivalence(state: dict, task: dict, manifest: dict) -> bool:
    """Apply one corrected condition's current predecessor state. Returns whether this task was handled."""
    from schema_study.input_acceptance import reuse_decision
    from schema_study.programme_closure import EQUIVALENCE_METHODS, artifact_valid, resolve_equivalence

    if not (
        task.get("representation_version") == CORRECTED_REPRESENTATION
        and task.get("family") == "principal_training"
        and str(task.get("entity") or "") in {"E0", "E1", "E2"}
    ):
        return False
    current = state.get(task["task_id"], {})
    info = (manifest.get("equivalence") or {}).get(task["task_id"]) or {}
    predecessor = info.get("predecessor") or task.get("replaces")
    predecessor_state = state.get(predecessor, {}).get("state") if predecessor else None
    evidence_ok = (
        info.get("equivalence_method") in EQUIVALENCE_METHODS
        and info.get("rendered_input_match") is True
        and info.get("selection_protocol") != "unrecoverable"
    )
    resolution = resolve_equivalence(
        (manifest.get("cell_disposition") or {}).get(task["task_id"]),
        predecessor_state,
        artifact_valid(Path(str(info.get("artifact") or ""))),
        evidence_ok,
    )
    if resolution == "reuse":
        decision = reuse_decision(task, info)
        state[task["task_id"]] = decision or {"state": "unsubmitted", "equivalence_resolution": "retrain"}
    elif resolution == "wait":
        state[task["task_id"]] = {**current, "state": current.get("state", "unsubmitted"), "equivalence_resolution": "wait"}
    else:
        state[task["task_id"]] = {**current, "state": "unsubmitted", "equivalence_resolution": "retrain"}
    return True


def account_representation(state: dict, tasks: list[dict]) -> None:
    """Archive unstarted stored deeper tasks and resolve a predecessor that finished after classification."""
    from schema_study.input_acceptance import contexts_accepted, reuse_decision

    manifest = corrected_input_manifest()
    equivalence = manifest.get("equivalence") or {}
    for task in tasks:
        current = state.get(task["task_id"], {})
        if current.get("scientific") == "reused_historical_equivalent" or current.get("state") == "reused_historical_equivalent":
            decision = reuse_decision(task, equivalence.get(task["task_id"]))
            if decision is None:
                state[task["task_id"]] = {"state": "unsubmitted", "scientific": None, "reuse_revoked": True, "equivalence_resolution": "retrain"}
            continue
        if current.get("state") in ACTIVE_OR_DONE | {"terminal_failure", "superseded_stored_representation"}:
            continue
        if representation_hold(task, current) == "superseded_stored_representation":
            state[task["task_id"]] = {
                "state": "superseded_stored_representation",
                "scientific": "historical_stored_representation",
                "preserved": True,
            }
            continue
        if not contexts_accepted(manifest):
            continue
        if resolve_task_equivalence(state, task, manifest):
            continue
        decision = reuse_decision(task, equivalence.get(task["task_id"]))
        if decision is not None:
            state[task["task_id"]] = decision


def observe_qwen32_release(state: dict, tasks: list[dict]) -> None:
    """Release one dataset only from a verification record whose checks all passed."""
    from schema_study.checkpoint_roles import release_evidence_complete

    for dataset in ("biored", "drugprot"):
        path = qwen32_release_file(dataset)
        if release_evidence_complete(_release_record(path)):
            continue
        verification = next(
            (
                task for task in tasks
                if task.get("family") == "path_verification" and task.get("dataset") == dataset
            ),
            None,
        )
        if verification is None:
            continue
        evidence = Path(verification["output"]) / "release_evidence.json"
        if not evidence.is_file():
            continue
        payload = json.loads(evidence.read_text(encoding="utf-8"))
        if not release_evidence_complete(payload):
            continue
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


def write_programme_status(tasks: list[dict], state: dict, controller_job: str) -> None:
    """Persistent status. Queue time is not reported as computation."""
    counts = {}
    for task in tasks:
        if "blocked" in task.get("statuses", []):
            label = "blocked"
        else:
            label = state.get(task["task_id"], {}).get("state", "unsubmitted")
        counts[label] = counts.get(label, 0) + 1
    principal = [
        task for task in tasks
        if task.get("family") in {"principal_training", "zero_shot", "few_shot", "adapter_inference", "final_evaluation"}
        and "blocked" not in task.get("statuses", [])
    ]
    unsettled = [
        task["task_id"] for task in principal
        if state.get(task["task_id"], {}).get("state", "unsubmitted") not in SCIENTIFIC_COMPLETE | {"terminal_failure", "superseded_stored_representation", "reused_historical_equivalent"}
    ]
    blocked = [task["task_id"] for task in principal if state.get(task["task_id"], {}).get("state") == "terminal_failure"]
    from schema_study.scientific_coverage import coverage_report

    coverage = coverage_report(tasks, state)
    if coverage["passed"]:
        status_name = "COMPLETE"
    elif blocked and not unsettled:
        status_name = "BLOCKED_REQUIRES_DECISION"
    elif not unsettled:
        status_name = "BLOCKED_REQUIRES_DECISION"
    elif blocked:
        status_name = "RUNNING_WITH_ISOLATED_EXCEPTIONS"
    else:
        status_name = "RUNNING"
    payload = {
        "status": status_name,
        "recorded_at": time.time(),
        "controller_job": controller_job,
        "counts": counts,
        "principal_unsettled": len(unsettled),
        "principal_terminal_failures": blocked[:20],
        "corrected_inputs": bool((corrected_input_manifest().get("acceptance") or {}).get("contexts_passed")),
        "coverage": {key: value for key, value in coverage.items() if key != "passed"} | {"passed": coverage["passed"]},
        "qwen32_biored_released": qwen32_core_released("biored"),
        "qwen32_drugprot_released": qwen32_core_released("drugprot"),
    }
    destination = PROGRAMME_ROOT / "programme_status.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    if blocked:
        exception = PROGRAMME_ROOT / "exception_report.json"
        exception.write_text(json.dumps({"status": status_name, "principal_terminal_failures": blocked, "recorded_at": payload["recorded_at"]}, indent=2) + "\n", encoding="utf-8")


def programme_remaining(tasks: list[dict], state: dict) -> bool:
    """True while any approved task is unfinished. An idle submission turn does not clear it."""
    for task in tasks:
        if "blocked" in task.get("statuses", []) or task.get("family") in HELD_FAMILIES:
            continue
        current = state.get(task["task_id"], {}).get("state", "unsubmitted")
        if current not in SCIENTIFIC_COMPLETE | {"terminal_failure", "superseded_stored_representation", "reused_historical_equivalent"}:
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
    """Publish the task's own status. Only one rank writes; a shared temporary name is not safe."""
    import os

    if int(os.environ.get("RANK", "0")) != 0:
        return
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".scientific_status.json.tmp.{os.getpid()}"
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
            if model == "sentence_level_certainty":
                location = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/cache/huggingface/pedropei-sentence-level-certainty")
                ready = (location / "config.json").is_file() and (location / "pytorch_model.bin").is_file()
            elif model == "deberta_base":
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
        directory = Path(task["shard_directory"]) if task.get("shard_directory") else Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/runs/formal/formal_matched_e0_e7/generation_inputs") / source / str(task.get("dataset")) / str(task.get("entity")) / str(task.get("predicate"))
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


DEEPER_ENTITIES = {"E3", "E4", "E5", "E6", "E7", "E8"}
STORED_DEEPER_ENTITIES = {"E3", "E4", "E5", "E6", "E7"}
CORRECTED_REPRESENTATION = "mesh_ancestor_v1"
PROGRAMME_ROOT = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/runs/formal/formal_matched_e0_e7")
QWEN32_CORE_RELEASE = PROGRAMME_ROOT / "qwen32b_core_release.json"
CORRECTED_INPUT_MANIFEST = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/reports/representation_corrected/input_manifest.json")
ACTIVE_OR_DONE = {"running", "pending", "submitted", "valid_continuation_required", "training_completed", "inference_completed", "completed"}


def external_input_build_active() -> bool:
    """A named input job, not one historical job identifier."""
    import os
    import subprocess

    user = os.environ.get("USER") or "brics.b5ac"
    result = subprocess.run(
        ["squeue", "-u", user, "-n", "p1-inputs", "-h", "-o", "%T"],
        capture_output=True, text=True, check=False,
    )
    return bool(result.stdout.strip())


def prompt_shards_present() -> bool:
    root = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/runs/formal/formal_matched_e0_e7/generation_inputs/mesh_ancestor_v1")
    present = 0
    for source in ("zero_shot", "few_shot"):
        for corpus in ("biored", "bc8"):
            for level in range(9):
                for predicate in range(4):
                    directory = root / source / corpus / f"E{level}" / f"P{predicate}"
                    if any(directory.glob("shard_*.jsonl")):
                        present += 1
    return present == 144


def prompts_ready_to_build() -> bool:
    from schema_study.input_acceptance import contexts_accepted

    return contexts_accepted(corrected_input_manifest())


def _corrected_context_files() -> bool:
    root = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/data/inputs")
    version = "target_anchored_mesh_ancestor_v1__biomedbert_e1354b7a"
    return all((root / corpus / version / "contexts.jsonl").is_file() for corpus in ("biored", "bc8"))


def corrected_input_manifest() -> dict:
    if not CORRECTED_INPUT_MANIFEST.is_file():
        return {}
    return json.loads(CORRECTED_INPUT_MANIFEST.read_text(encoding="utf-8"))


def representation_hold(task: dict, entry: dict | None = None) -> str | None:
    """Unstarted stored E3–E7 stays superseded. Corrected work waits for the frozen inputs."""
    current = (entry or {}).get("state")
    if current in ACTIVE_OR_DONE:
        return None
    if task.get("family") == "selection_diagnostic":
        return None
    if task.get("representation_version") == CORRECTED_REPRESENTATION:
        from schema_study.input_acceptance import contexts_accepted, prompts_accepted

        manifest = corrected_input_manifest()
        if not contexts_accepted(manifest):
            return "corrected_inputs_not_ready"
        if task.get("family") == "principal_training" and (manifest.get("acceptance") or {}).get("model_inputs_passed") is not True:
            return "model_inputs_unverified"
        if task.get("family") in {"zero_shot", "few_shot", "adapter_inference"} and not prompts_accepted(manifest):
            return "corrected_prompts_not_ready"
        if task.get("family") == "principal_training" and str(task.get("entity") or "") in {"E0", "E1", "E2"}:
            resolution = (entry or {}).get("equivalence_resolution")
            disposition = (manifest.get("cell_disposition") or {}).get(task["task_id"])
            if resolution == "retrain" or disposition == "retrain":
                return None
            if resolution == "wait" or disposition in {"reuse", "reuse_when_complete"}:
                return "equivalent_predecessor_unresolved"
            return "equivalence_not_classified"
        return None
    if task.get("family") == "final_evaluation" and not task.get("representation_version"):
        return None
    entity = str(task.get("entity") or "")
    if entity in STORED_DEEPER_ENTITIES or task.get("exposure") == "exposure_controlled_all_views" or task.get("withheld") == "E3":
        return "superseded_stored_representation"
    return None


def representative_group(task: dict) -> str:
    if task.get("family") == "selection_diagnostic":
        return f"selection_diagnostic|{task.get('task_id')}"
    band = "deeper" if representation_hold(task, {}) else "core"
    dataset = task.get("dataset") or "na"
    if task.get("model") == "qwen32b":
        return f"{task.get('family')}|qwen32b|{dataset}|{band}|{(task.get('resources') or {}).get('gpus')}"
    return f"{task.get('family')}|{task.get('model')}|{band}|{(task.get('resources') or {}).get('gpus')}"


def qwen32_release_file(dataset: str) -> Path:
    return PROGRAMME_ROOT / f"qwen32b_{dataset}_release.json"


def _release_record(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _release_payload(path: Path) -> bool:
    from schema_study.checkpoint_roles import release_evidence_complete

    return release_evidence_complete(_release_record(path))


def qwen32_core_released(dataset: str = "biored") -> bool:
    """Release one dataset when its own representative has recorded the four checks."""
    if _release_payload(qwen32_release_file(dataset)):
        return True
    return dataset == "biored" and _release_payload(QWEN32_CORE_RELEASE)


def controller_decisions(tasks: list[dict], state: dict, running: int, pending: int, verified_external: set[str] | None = None, active_job_ids: set[str] | None = None) -> list[dict]:
    """Choose submissions from persistent state. An active or completed task is not submitted again."""
    from schema_study.production_plan import MAX_QUEUED_JOBS

    priority = {"input_preparation", "input_validation", "ledger_acceptance", "path_verification"}
    tasks = sorted(tasks, key=lambda item: 0 if item.get("family") in priority else 1)
    verified = verified_external or set()
    input_build_running = external_input_build_active()
    seen_groups = set()
    for task in tasks:
        if "representative" in task:
            continue
        if "blocked" in task.get("statuses", []):
            task["representative"] = False
            continue
        key = representative_group(task)
        task["representative_group"] = key
        task["representative"] = key not in seen_groups
        seen_groups.add(key)
    ready_representatives = {
        task.get("representative_group")
        for task in tasks
        if task.get("representative") and _group_released(state.get(task["task_id"], {}))
    }
    observe_qwen32_release(state, tasks)
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
        hold = representation_hold(task, state.get(task["task_id"], {}))
        if hold:
            decisions.append({"task_id": task["task_id"], "action": "skip", "reason": hold})
            continue
        if task.get("family") in {"input_preparation", "input_validation"} and input_build_running:
            decisions.append({"task_id": task["task_id"], "action": "skip", "reason": "external_input_build_running"})
            continue
        if task.get("family") in {"input_preparation", "input_validation"}:
            from schema_study.input_acceptance import context_files_complete

            prompts = task.get("role") == "mesh_ancestor_v1_prompts"
            if prompts and not prompts_ready_to_build():
                decisions.append({"task_id": task["task_id"], "action": "skip", "reason": "contexts_not_accepted"})
                continue
            if prompts and task.get("family") == "input_validation" and any(
                state.get(item["task_id"], {}).get("state") in ACTIVE_CONTROLLER_STATES
                for item in tasks
                if item.get("family") == "input_preparation" and item.get("role") == "mesh_ancestor_v1_prompts"
            ):
                decisions.append({"task_id": task["task_id"], "action": "skip", "reason": "prompt_preparation_running"})
                continue
            if prompts and task.get("family") == "input_validation" and not prompt_shards_present():
                decisions.append({"task_id": task["task_id"], "action": "skip", "reason": "prompts_not_written"})
                continue
            if prompts and task.get("family") == "input_preparation" and prompt_shards_present():
                prompts_passed = (corrected_input_manifest().get("acceptance") or {}).get("prompts_passed")
                if prompts_passed is not False:
                    decisions.append({"task_id": task["task_id"], "action": "skip", "reason": "prompts_already_present"})
                    continue
            if not prompts and task.get("family") == "input_validation" and not context_files_complete():
                decisions.append({"task_id": task["task_id"], "action": "skip", "reason": "contexts_not_written"})
                continue
            if not prompts and task.get("family") == "input_preparation" and context_files_complete():
                contexts_passed = (corrected_input_manifest().get("acceptance") or {}).get("contexts_passed")
                if contexts_passed is not False:
                    decisions.append({"task_id": task["task_id"], "action": "skip", "reason": "contexts_already_present"})
                    continue
        if task.get("family") == "path_verification":
            source = next(
                (
                    item for item in tasks
                    if item.get("family") == "principal_training"
                    and item.get("model") == "qwen32b"
                    and item.get("dataset") == task.get("dataset")
                    and not item.get("representation_version")
                ),
                None,
            )
            source_state = None if source is None else state.get(source["task_id"], {}).get("state")
            checkpoint = None if source is None else Path(source["output"]) / "adapter_checkpoint.pt"
            if source_state == "running":
                decisions.append({"task_id": task["task_id"], "action": "skip", "reason": "representative_writing"})
                continue
            if checkpoint is None or not checkpoint.is_file():
                decisions.append({"task_id": task["task_id"], "action": "skip", "reason": "checkpoint_absent"})
                continue
        if task.get("family") == "selection_diagnostic" and task.get("dataset") == "biored":
            decisions.append({"task_id": task["task_id"], "action": "skip", "reason": "superseded_corrected_retraining"})
            continue
        if task.get("family") == "selection_diagnostic":
            source_state = state.get(task.get("source_task"), {}).get("state")
            if source_state not in SCIENTIFIC_COMPLETE:
                decisions.append({"task_id": task["task_id"], "action": "skip", "reason": "source_training_not_finished"})
                continue
        if (
            task.get("family") == "principal_training"
            and task.get("model") == "qwen32b"
            and task.get("representative") is False
            and not qwen32_core_released(str(task.get("dataset") or "biored"))
        ):
            decisions.append({"task_id": task["task_id"], "action": "skip", "reason": "execution_path_unverified"})
            continue
        if current in {"terminal_failure", "blocked"} or "blocked" in task.get("statuses", []):
            decisions.append({"task_id": task["task_id"], "action": "skip", "reason": current if current != "unsubmitted" else "blocked"})
            continue
        group_block = (state.get("_groups") or {}).get(task.get("representative_group") or "")
        if group_block and group_block.get("blocked"):
            decisions.append({"task_id": task["task_id"], "action": "skip", "reason": "systematic_group_failure"})
            continue
        if (
            task.get("representative") is False
            and task.get("family") not in {"input_preparation", "input_validation", "ledger_acceptance"}
            and task.get("representative_group") not in ready_representatives
        ):
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
        "inference_backend": specification.get("backend") or result.get("backend"),
        "inference_backend_version": specification.get("backend_version"),
        "finish_reason": result.get("finish_reason"),
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
    if str(task.get("model")) == "qwen32b" and gpus:
        command.insert(5, "--mem=120G")
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
