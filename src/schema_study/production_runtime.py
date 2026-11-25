"""Training horizon, population selection, generation requests and scheduler commands.

This module does not load a model and it does not call sbatch. A requested GPU
count is not a distributed world size.
"""

from __future__ import annotations

import json
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
        labels = {(row["candidate_id"], row["entity"], row["predicate"]): (row.get("assistant") or {}).get("labels", []) for row in train_rows + dev_rows}
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
        return {
            "kind": kind,
            "rows": _attach_view_labels(selected, labels),
            "evaluation": _attach_view_labels(evaluation, labels),
            "views": views,
            "candidates": len(candidates),
        }
    if family == "curriculum":
        train_rows = [row for row in rows if row["split"] == "train" and row["entity"] == task["entity"]]
        grouped = {predicate: [row for row in train_rows if row.get("predicate") == predicate] for predicate in RESEARCH_PREDICATES}
        ordered = curriculum_records(task["order"], grouped)
        evaluation = [row for row in rows if row["split"] == "dev" and row["entity"] == task["entity"] and row["predicate"] == "P3"]
        return {"kind": task["order"], "rows": ordered, "evaluation": evaluation, "candidates": len({row["candidate_id"] for row in train_rows})}
    if family == "unseen_schema":
        train_rows = [row for row in rows if row["split"] == "train"]
        split = unseen_split(task["withheld"], train_rows)
        evaluation = [
            row for row in rows
            if row["split"] in {"dev", "test"} and (row["entity"] == task["withheld"] or row["predicate"] == task["withheld"])
        ]
        return {"kind": "unseen", "rows": split["train"], "evaluation": evaluation, "withheld": task["withheld"]}
    if family in {"hierarchical_multitask", "hierarchical_consistency"}:
        return _hierarchical_population(task, rows)
    if family == "data_efficiency":
        scoped = [row for row in rows if _same_condition(row, task.get("entity"), task.get("predicate"))]
        documents = subset_document_ids([row["document_id"] for row in scoped if row.get("split", "train") == "train"], int(task["fraction"]), int(task["seed"]))
        allowed = set(documents)
        train = [row for row in scoped if row.get("split", "train") == "train" and row["document_id"] in allowed]
        held = [row for row in scoped if row.get("split") in {"dev", "test"}]
        if any(row["document_id"] not in allowed for row in train):
            raise RuntimeError("A training row is outside the nested document subset.")
        return {"kind": f"fraction_{task['fraction']}", "rows": train, "evaluation": held, "documents": documents}
    if family == "negative_sampling":
        scoped = [row for row in rows if _same_condition(row, task.get("entity"), task.get("predicate"))]
        train = [row for row in scoped if row.get("split", "train") == "train"]
        chosen = negative_population(train, task["strategy"], int(task["seed"]))
        evaluation = [row for row in scoped if row.get("split") in {"dev", "test"}]
        return {"kind": task["strategy"], "rows": chosen["rows"], "evaluation": evaluation, "negative_ids": chosen["negative_ids"]}
    entity = task.get("entity")
    predicate = task.get("predicate")
    matched = [row for row in rows if _same_condition(row, entity, predicate)]
    selected = [row for row in matched if row.get("split", "train") == "train"]
    evaluation = [row for row in matched if row.get("split") == "dev"]
    return {"kind": "independent", "rows": selected, "evaluation": evaluation, "candidates": len(selected)}


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
    evaluation = collect("dev")
    return {"kind": task["family"], "rows": train, "evaluation": evaluation, "candidates": len(train)}


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


def positive_micro_f1(rows: list[dict]) -> dict:
    """Corpus micro-F1 from pooled counts. Empty predictions contribute false negatives, not a mean of document F1."""
    true_positive = false_positive = false_negative = 0
    for row in rows:
        gold = set(row.get("gold") or [])
        predicted = set(row.get("predicted") or []) if row.get("status") == "valid" else set()
        true_positive += len(gold & predicted)
        false_positive += len(predicted - gold)
        false_negative += len(gold - predicted)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "document_f1_averaged": False}


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
    """vLLM 0.13 structured outputs. This is not an unconstrained sample."""
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


def generation_requests(rows: list[dict]) -> list[dict]:
    """Every request carries the model input. A candidate identifier is not a prompt."""
    requests = []
    for row in rows:
        prompt = row.get("prompt")
        if not prompt or prompt == row.get("candidate_id"):
            raise ValueError(f"{row.get('candidate_id')} has no model input.")
        requests.append({
            "request_id": row["request_id"],
            "candidate_id": row["candidate_id"],
            "prompt": prompt,
            "model": row["model"],
            "decoding": row.get("decoding", "greedy"),
        })
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


def controller_requeue(job_id: str, authorised: bool, finished: bool) -> list[str] | None:
    """Requeue only after approval and only while work remains. This does not call sbatch."""
    from schema_study.formal_launch import requeue_permitted

    if not requeue_permitted(authorised, finished):
        return None
    return ["sbatch", f"--dependency=afterany:{job_id}", "--account=brics.b5ac", "--partition=workq", "--qos=normal", "scripts/slurm/programme_controller.sbatch"]


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
