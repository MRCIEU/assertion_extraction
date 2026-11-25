"""Dispatch a resolved task to the production executor.

Importing this module does not import PyTorch. An unauthorised GPU task exits
3. A CPU fixture task uses the same `execute_task` path and must publish a record.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from schema_study.paths import get_paths
from schema_study.production_plan import execution_manifest, gpu_authorised, resolved_tasks
from schema_study.records import publish_file


def resolve_all() -> dict:
    tasks = resolved_tasks()
    manifest = execution_manifest()
    missing = [task["task_id"] for task in manifest if task["family"] != "qwen36" and not task.get("executor")]
    executable = [task["task_id"] for task in manifest if task["executable"]]
    blocked = [task["task_id"] for task in manifest if "blocked" in task["statuses"]]
    return {
        "training_tasks": len(tasks),
        "execution_tasks": len(manifest),
        "missing_executor": missing,
        "marked_executable": executable,
        "blocked": len(blocked),
    }


def _lookup(family: str, task_id: str) -> dict | None:
    for task in execution_manifest():
        if task["task_id"] == task_id and task["family"] == family:
            return task
    return None


def execute_task(task: dict, workdir: Path, *, gpu_permitted: bool) -> dict:
    """Run one task. A production model is loaded only when GPU execution is permitted."""
    if task.get("model") == "cpu_fixture":
        from schema_study.production_lora import FixtureCausalModel, compare_interrupted_run, publish_prediction

        result = compare_interrupted_run(workdir / "continuation")
        publish_prediction(workdir / "predictions.jsonl", [{
            "request_id": "cpu-fixture",
            "status": "valid" if result["weights_match"] and result["consumed_match"] else "invalid",
            "raw": json.dumps({key: result[key] for key in ("weights_match", "losses_match", "consumed_match", "steps")}),
            "labels": ["continued"] if result["weights_match"] else [],
        }])
        completed = workdir / "prediction_complete.json"
        completed.write_text(json.dumps({"rows_written": True, "checkpoint": "adapter", "empirical_qwen_result": False}) + "\n", encoding="utf-8")
        result["completed"] = bool(result["weights_match"] and result["losses_match"] and result["consumed_match"])
        return result
    cpu_executors = {"schema_study.analysis_plan:analyse_records", "schema_study.measurement_cpu:adapt_verifier"}
    if not gpu_permitted and task.get("executor") not in cpu_executors:
        raise PermissionError("formal_gpu_execution_is_not_authorised")
    if not task.get("executor"):
        raise RuntimeError("The task has no production executor.")
    return _invoke_production(task, workdir, gpu_permitted=gpu_permitted)


def _invoke_production(task: dict, workdir: Path, *, gpu_permitted: bool = False) -> dict:
    """Call the registered executor with the task's model, inputs and configuration."""
    executor = task["executor"]
    if task.get("model") == "qwen36" or "blocked" in task.get("statuses", []):
        return {"completed": False, "status": "blocked", "reason": task.get("blocked_reason", "blocked")}
    if executor == "schema_study.formal_train:train":
        from schema_study.formal_train import train

        from schema_study.experiments import training_hyperparameters

        learning_rate = task.get("learning_rate")
        result = train(
            task["dataset"],
            task["entity"],
            task.get("schema_predicate") or task["predicate"],
            int(task["seed"]),
            model_name=task.get("model") or "biomedbert",
            objective=task.get("loss") or "standard",
            fraction=int(task.get("fraction") or 100),
            negative_strategy=task.get("strategy"),
            learning_rate=learning_rate,
            output_dir=workdir,
        )
        expected = int(training_hyperparameters()["max_epochs"])
        complete = int(result["completed_epochs"]) >= expected and result.get("best_score") is not None
        return {
            "completed": complete,
            "status": "training_completed" if complete else "valid_continuation_required",
            "epochs": result["completed_epochs"],
            "development_score": result.get("best_score"),
            "empirical_result": False,
        }
    if executor == "schema_study.production_lora:fit":
        from schema_study.production_lora import fit, load_causal_lm, prepare_process, production_configuration
        from schema_study.production_runtime import horizon_for_task, placement, select_population

        requested = int(task["resources"]["gpus"])
        process = prepare_process(requested)
        population = select_population(task, _metadata_rows(task))
        views = len(population.get("views") or [1])
        sequence = len(population["rows"]) if task["family"] == "curriculum" else None
        horizon = horizon_for_task(task, population.get("candidates") or len(population["rows"]), views, process["world"], sequence)
        model, tokenizer = load_causal_lm(task["model"])
        configuration = production_configuration()
        configuration.update({key: task["configuration"][key] for key in task["configuration"] if key in configuration})
        configuration["base_revision"] = task.get("model_revision")
        configuration["tokenizer_revision"] = task.get("tokenizer_revision")
        configuration["micro_batch_size"] = 1
        configuration["world_size"] = process["world"]
        configuration["requested_gpus"] = requested
        method = placement(task["model"], requested)["method"]
        configuration["placement"] = method
        configuration["device_policy"] = {"single_process": "single_gpu", "distributed_data_parallel": "ddp", "fsdp_full_shard": "fsdp"}[method]
        configuration["require_fingerprints"] = True
        configuration["eval_interval"] = horizon["updates_per_epoch"]
        development = population.get("development") or _development_from_candidates(task)
        if any(row.get("split") == "test" for row in development):
            raise RuntimeError("A test row entered development selection.")
        resume, fresh = _resume_lora(workdir, model, configuration, {
            "input": (task.get("input") or {}).get("fingerprint"),
            "configuration": json.dumps(task.get("configuration"), sort_keys=True),
        })
        result = fit(
            model,
            _supervised_batches(task, tokenizer, configuration),
            configuration=configuration,
            steps=horizon["total_updates"],
            total_steps=horizon["total_updates"],
            output_dir=workdir,
            fingerprints={"input": (task.get("input") or {}).get("fingerprint"), "configuration": json.dumps(task.get("configuration"), sort_keys=True)},
            score_fn=_lora_development_score(task, development, tokenizer, workdir / "development_predictions.jsonl"),
            resume=resume,
            fresh=fresh,
        )
        return {"completed": result["completed"], "status": result["status"], "total_updates": horizon["total_updates"], "empirical_qwen_result": False}
    if executor == "schema_study.production_encoder:train_hierarchical":
        from schema_study.production_encoder import SchemaHierarchicalModel, prepare_marker_encoder, train_hierarchical
        from schema_study.production_records import head_sizes
        from schema_study.schemas import default_schema_path, load_schema

        from schema_study.production_runtime import horizon_for_task, select_population

        corpus = task.get("dataset", "biored")
        schema = load_schema(default_schema_path(get_paths().code_root, "drugprot" if corpus == "drugprot" else "biored"))
        population = select_population(task, _metadata_rows(task))
        horizon = horizon_for_task(task, population.get("candidates") or len(population["rows"]), 1, 1)
        encoder_name = "biolinkbert" if task.get("model") == "biolinkbert" else "biomedbert"
        encoder, _tokenizer = prepare_marker_encoder(encoder_name, int(task.get("seed") or 13))
        model = SchemaHierarchicalModel(encoder, int(encoder.config.hidden_size), head_sizes(schema, corpus), corpus)
        evaluation = population.get("evaluation") or _development_from_candidates(task, schema=schema, hierarchical=True)
        def batch_fn():
            return _hierarchical_batches(task, population["rows"], schema, corpus)

        return train_hierarchical(
            model,
            steps=horizon["total_updates"],
            total_steps=horizon["total_updates"],
            consistency=bool(task.get("consistency")),
            corpus=corpus,
            batch_fn=batch_fn,
            score_fn=_hierarchical_development_score(task, evaluation, schema, corpus),
            output_dir=workdir,
            epoch_updates=horizon["updates_per_epoch"],
        )
    if executor == "schema_study.formal_train:evaluate":
        from schema_study.formal_train import evaluate

        return evaluate(task)
    if executor == "schema_study.production_encoder:evaluate_hierarchical":
        return _evaluate_hierarchical(task, workdir)
    if executor.startswith("schema_study.production_inference:") or executor == "schema_study.formal_generate:generate":
        return _run_generation(task, workdir)
    if executor == "schema_study.analysis_plan:analyse_records":
        from schema_study.analysis_plan import analyse_records

        documents = task.get("documents") or []
        if not documents or any(not row.get("document_id") for row in documents):
            return {"completed": False, "status": "waiting_for_predictions", "reason": "Analysis needs prediction records. An empty document field is not a result."}
        report = analyse_records(documents)
        return {"completed": True, "status": "analysis_completed", "report": report}
    if executor == "schema_study.measurement_cpu:adapt_verifier":
        if not gpu_permitted:
            return {
                "started": False,
                "completed": False,
                "status": "not_started",
                "reason": "Verifier adaptation runs on the scheduled GPU.",
            }
        from schema_study.measurement_cpu import train_adapted_verifier

        return train_adapted_verifier(task, workdir)
    module_name, function_name = executor.split(":")
    importlib.import_module(module_name)
    return {"completed": False, "status": "failed", "reason": f"{function_name} has no completed production invocation"}


def _epoch_repeats(task: dict) -> int:
    """Resource-efficient records already contain one row per epoch. Other families repeat the selected epoch."""
    if task["family"] == "curriculum":
        return 1
    if task["family"] == "schema_conditioned_lora" and task.get("exposure") == "resource_efficient_one_view":
        return 1
    return int(task.get("configuration", {}).get("max_epochs") or 3)


def _metadata_rows(task: dict) -> list[dict]:
    from schema_study.production_records import INPUTS

    path = INPUTS[f"sft_{task['dataset']}"]
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append({
                "candidate_id": row["candidate_id"],
                "entity_level": row.get("entity_level"),
                "predicate_level": row.get("predicate_level"),
                "split": row.get("split", "train"),
                "assistant": row.get("assistant") or {"labels": []},
            })
    return rows


def _context_texts(dataset: str, needed: set[tuple[str, str]]) -> dict[tuple[str, str], str]:
    from schema_study.production_records import INPUTS

    found = {}
    with INPUTS[f"contexts_{dataset}"].open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            identifier = row["candidate_id"]
            for entity, text in (row.get("views") or {}).items():
                key = (identifier, entity)
                if key in needed:
                    found[key] = text
            if len(found) == len(needed):
                break
    missing = needed - found.keys()
    if missing:
        sample = next(iter(missing))
        raise RuntimeError(f"{sample[0]} has no source text for {sample[1]}.")
    return found


def _view_entity(task: dict, row: dict) -> str:
    if row.get("view"):
        return str(row["view"]).split("/")[0]
    entity = row.get("entity") or task.get("entity")
    if not entity:
        raise RuntimeError("The training row has no entity representation. BioRED shared-schema training does not default to native.")
    return entity


def _supervised_batches(task: dict, tokenizer, configuration: dict | None = None):
    """Join source text to the selected population, then tokenise. An identifier is not a training string."""
    from schema_study.formal_prepare import _fill
    from schema_study.learning_conditions import HISTORICAL_PREDICATES
    from schema_study.production_lora import batches_from_records
    from schema_study.production_runtime import select_population
    from schema_study.schemas import default_schema_path, load_schema

    dataset = task["dataset"]
    population = select_population(task, _metadata_rows(task))
    selected = population["rows"]
    schema_name = "drugprot" if dataset == "drugprot" else "biored"
    schema = load_schema(default_schema_path(get_paths().code_root, schema_name))
    blocks = {}
    needed = set()
    for row in selected:
        entity = _view_entity(task, row)
        predicate = row.get("predicate") or task.get("predicate")
        historical = HISTORICAL_PREDICATES.get(predicate, predicate)
        blocks.setdefault(historical, "\n".join(f"- {name}" for name in schema.positive_display_names(historical)))
        needed.add((row["candidate_id"], entity))
    texts = _context_texts(dataset, needed)

    def records():
        import os

        from schema_study.production_runtime import rank_partitions

        world = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        shuffle = task["family"] not in {"curriculum", "schema_conditioned_lora"}
        for epoch in range(_epoch_repeats(task)):
            keyed = [(f"{index}:{row['candidate_id']}:{row.get('view', '')}:{row.get('epoch', '')}", row) for index, row in enumerate(selected)]
            partition = rank_partitions([key for key, _row in keyed], world=world, micro_batch=1, seed=int(task.get("seed") or 13), epoch=epoch, shuffle=shuffle)
            if configuration is not None:
                configuration["data_partition"] = {
                    "shuffle": partition["shuffle"],
                    "epoch": partition["epoch"],
                    "seed": partition["seed"],
                    "dropped": partition["dropped"],
                    "remainder_policy": partition["remainder_policy"],
                    "same_remainder_each_epoch": not partition["shuffle"],
                    "effective_batch": partition["effective_batch"],
                    "accumulation": partition["accumulation"],
                    "micro_batch": partition["micro_batch"],
                    "world": partition["world"],
                    "rank": rank,
                    "rank_examples": len(partition["ranks"][rank]),
                }
            by_key = {key: row for key, row in keyed}
            for key in partition["ranks"][rank]:
                row = by_key[key]
                entity = _view_entity(task, row)
                predicate = row.get("predicate") or task.get("predicate")
                historical = HISTORICAL_PREDICATES.get(predicate, predicate)
                labels = (row.get("assistant") or {}).get("labels", [])
                yield {
                    "user": _fill(dataset, row["candidate_id"], historical, texts[(row["candidate_id"], entity)], {(dataset, historical): blocks[historical]}),
                    "assistant": json.dumps({"labels": labels}, ensure_ascii=False),
                }

    yield from batches_from_records(records(), tokenizer, 1)


def _development_from_candidates(task: dict, schema=None, hierarchical: bool = False) -> list[dict]:
    """Development gold comes from the candidate table. The supervised file contains training rows only."""
    from schema_study.formal_eval import gold_groups
    from schema_study.formal_train import _candidates
    from schema_study.learning_conditions import HISTORICAL_PREDICATES, RESEARCH_PREDICATES
    from schema_study.schemas import default_schema_path, load_schema

    dataset = task["dataset"]
    if schema is None:
        schema = load_schema(default_schema_path(get_paths().code_root, "drugprot" if dataset == "drugprot" else "biored"))
    found = _candidates(dataset, {"dev"})
    if not found:
        raise RuntimeError("The development population is empty.")
    if task["family"] == "unseen_schema":
        withheld = task["withheld"]
        entities = [withheld] if str(withheld).startswith("E") else [f"E{level}" for level in range(8)]
        predicates = [withheld] if str(withheld).startswith("P") else list(RESEARCH_PREDICATES)
        rows = []
        for item in found.values():
            for entity in entities:
                for predicate in predicates:
                    labels = gold_groups(schema, item["gold_labels"], HISTORICAL_PREDICATES[predicate])
                    rows.append({**item, "entity": entity, "predicate": predicate, "split": "dev", "assistant": {"labels": labels}})
        return rows
    if task["family"] == "schema_conditioned_lora":
        from schema_study.production_records import all_view_records, one_view_records, view_names

        views = view_names("drugprot" if dataset == "drugprot" else "biored")
        candidates = [{"candidate_id": item["candidate_id"], "document_id": item["document_id"], "gold_labels": item["gold_labels"]} for item in found.values()]
        if task.get("exposure") == "resource_efficient_one_view":
            chosen = one_view_records(candidates, views, 0, int(task.get("seed") or 13))
        else:
            chosen = all_view_records(candidates, views)
        rows = []
        for record in chosen:
            entity, predicate = record["view"].split("/")
            labels = gold_groups(schema, record["gold_labels"], HISTORICAL_PREDICATES[predicate])
            rows.append({**record, "entity": entity, "predicate": predicate, "split": "dev", "assistant": {"labels": labels}})
        return rows
    rows = []
    for item in found.values():
        if hierarchical:
            labels = {predicate: gold_groups(schema, item["gold_labels"], HISTORICAL_PREDICATES[predicate]) for predicate in RESEARCH_PREDICATES}
            rows.append({**item, "entity": task["entity"], "split": "dev", "labels_by_predicate": labels, "assistant": {"labels": labels.get("P3", [])}})
        else:
            predicate = task.get("predicate") or "P3"
            historical = task.get("schema_predicate") or HISTORICAL_PREDICATES.get(predicate, predicate)
            labels = gold_groups(schema, item["gold_labels"], historical)
            rows.append({**item, "entity": task.get("entity"), "predicate": predicate, "split": "dev", "assistant": {"labels": labels}})
    return rows


def backend_specification(task: dict) -> dict:
    """The scientific decoding settings. A generic call does not stand in for a named strategy."""
    family = task.get("family")
    specification = {
        "model": task.get("model"),
        "revision": task.get("model_revision"),
        "adapter": task.get("adapter_path"),
        "decoding": "greedy",
        "max_new_tokens": 256,
        "samples": 1,
        "temperature": 0.0,
    }
    if family == "constrained":
        specification.update({"decoding": "constrained", "structured_outputs": True})
    elif family == "thinking":
        specification.update({"decoding": "thinking", "enable_thinking": True})
    elif family == "self_consistency":
        specification.update({"decoding": "self_consistency", "samples": 5, "temperature": 0.7})
    elif family == "candidate_scoring":
        specification.update({"decoding": "candidate_scoring", "token_logprobs": True})
    elif family == "evidence":
        specification.update({"decoding": "evidence", "requires_spans": True})
    elif family == "qualified":
        specification.update({"decoding": "qualified", "requires_qualifications": True})
    return specification


class GenerationSession:
    """One loaded model serves every compatible request. A request does not build its own engine."""

    def __init__(self, task: dict):
        self.task = task
        self.specification = backend_specification(task)
        self.engine = None
        self.tokenizer = None
        self.loads = 0
        self.fixture = task.get("backend_name") == "fixture"
        self.responses = task.get("fixture_responses") or {}

    def _engine(self):
        if self.engine is None:
            from vllm import LLM

            from schema_study.production_assets import GENERATIVE

            spec = GENERATIVE[self.specification["model"]]
            gpus = int((self.task.get("resources") or {}).get("gpus") or 1)
            self.engine = LLM(
                model=str(spec["path"]),
                dtype="bfloat16",
                enable_lora=bool(self.specification.get("adapter")),
                tensor_parallel_size=max(1, gpus),
                gpu_memory_utilization=0.9,
            )
            self.loads += 1
        return self.engine

    def _load_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            from schema_study.production_assets import GENERATIVE

            spec = GENERATIVE[self.specification["model"]]
            self.tokenizer = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True, revision=spec["revision"])
        return self.tokenizer

    def _prompts(self, requests: list[dict]) -> list[str]:
        if self.specification.get("model") == "gpt_oss_20b":
            self._load_tokenizer()
            prompts = []
            for request in requests:
                user = request.get("prompt") or ""
                prompts.append(self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                ))
            return prompts
        if not self.specification.get("enable_thinking"):
            return [request.get("rendered") or request.get("prompt") or "" for request in requests]
        self._load_tokenizer()
        prompts = []
        for request in requests:
            user = request.get("prompt") or ""
            prompts.append(self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=True,
            ))
        return prompts

    def generate_batch(self, requests: list[dict], *, batch_size: int = 8) -> list[dict]:
        """Run one loaded backend over bounded batches. The engine count stays at one."""
        if not requests:
            return []
        produced = []
        for start in range(0, len(requests), batch_size):
            produced.extend(self._batch(requests[start:start + batch_size]))
        if not self.fixture and self.loads != 1:
            raise RuntimeError("The production backend constructed more than one engine.")
        return produced

    def _batch(self, requests: list[dict]) -> list[dict]:
        if self.fixture:
            self.loads = 1
            rows = []
            for request in requests:
                found = self.responses.get(request["request_id"])
                if found is None:
                    raise RuntimeError("retryable response missing")
                rows.append(dict(found))
            return rows
        from schema_study.production_inference import vllm_generate

        engine = self._engine()
        sampling = {
            "temperature": self.specification["temperature"],
            "max_tokens": self.specification["max_new_tokens"],
            "n": 5 if self.specification["decoding"] == "self_consistency" else 1,
        }
        if self.specification.get("structured_outputs"):
            sampling["structured_outputs"] = {"json": requests[0].get("schema_json") or {"type": "object"}}
        if self.specification.get("token_logprobs"):
            return self._score_labels(requests, engine, sampling, _generation_schema(self.task))
        rows = vllm_generate(self._prompts(requests), sampling, lora_path=self.specification.get("adapter"), engine=engine)
        if self.specification["decoding"] == "self_consistency":
            for row in rows:
                samples = row.get("samples") or []
                if len(samples) != 5:
                    raise RuntimeError("Self-consistency did not retain five samples from the loaded engine.")
        return rows

    def _score_labels(self, requests: list[dict], engine, sampling: dict, schema) -> list[dict]:
        from schema_study.production_inference import vllm_generate

        pairs = []
        for request in requests:
            legal = list(request.get("legal_labels") or [])
            if not legal:
                legal = _legal_output_labels(self.task, schema)
            if not legal:
                raise RuntimeError("Candidate scoring has no legal output schema.")
            prompt = request.get("rendered") or request.get("prompt") or ""
            for label in legal:
                pairs.append((request["request_id"], label, prompt + str(label)))
        scored = vllm_generate([prompt for _request, _label, prompt in pairs], {**sampling, "max_tokens": 1, "prompt_logprobs": 1, "n": 1}, lora_path=self.specification.get("adapter"), engine=engine)
        grouped: dict[str, dict[str, list]] = {}
        for (request_id, label, _prompt), row in zip(pairs, scored):
            grouped.setdefault(request_id, {})[label] = row.get("prompt_token_logprobs") or []
        return [{"request_id": request["request_id"], "raw": "", "labels": None, "token_logprobs": grouped[request["request_id"]]} for request in requests]


def construct_backend(task: dict) -> GenerationSession:
    """Build one session from the task record. The record itself is not a function."""
    return GenerationSession(task)


def _resume_compatible(task: dict, record: dict) -> bool:
    """A completed ordinary generation is not a thinking trace or a five-sample vote."""
    if task.get("family") == "thinking":
        return record.get("thinking_enabled") is True
    if task.get("family") == "self_consistency":
        return isinstance(record.get("samples"), list) and len(record["samples"]) == 5
    return True


def _legal_output_labels(task: dict, schema) -> list[str]:
    """Positive display names for the schema predicate, plus the empty decision."""
    if schema is None:
        return []
    from schema_study.learning_conditions import HISTORICAL_PREDICATES

    level = task.get("schema_predicate") or HISTORICAL_PREDICATES.get(task.get("predicate"), task.get("predicate"))
    if not level or level not in getattr(schema, "levels", {}):
        return []
    return [*schema.positive_display_names(level), "[]"]


def _generation_schema(task: dict):
    dataset = task.get("dataset")
    if dataset not in {"biored", "bc8", "drugprot"}:
        return None
    from schema_study.paths import get_paths
    from schema_study.schemas import default_schema_path, load_schema

    name = "drugprot" if dataset == "drugprot" else "biored"
    return load_schema(default_schema_path(get_paths().code_root, name))


def _parse_generated(task: dict, request: dict, result: dict, schema) -> dict:
    """Keep a supplied parse. Otherwise parse the original response against the task schema."""
    if result.get("labels") is not None or schema is None or not request.get("candidate_id"):
        return result
    from schema_study.production_runtime import interpret_generation

    from schema_study.learning_conditions import HISTORICAL_PREDICATES

    if result.get("evidence") and not result["evidence"].get("complete"):
        return {**result, "labels": None}
    level = task.get("schema_predicate") or HISTORICAL_PREDICATES.get(task.get("predicate"), task.get("predicate")) or request.get("predicate_level")
    if not level:
        return result
    interpreted = interpret_generation(result.get("raw"), request["candidate_id"], schema, level)
    return {
        **result,
        "raw": result.get("raw"),
        "labels": interpreted["predicted_labels"],
        "illegal_labels": interpreted.get("illegal_labels", []),
        "parsed": interpreted,
    }


def _family_result(family: str, request: dict, result: dict, schema=None, level: str | None = None) -> dict:
    """Apply one named decoding rule to a result that a shared engine already produced."""
    if family == "thinking":
        raw = result.get("raw") or ""
        final = raw.split("</think>")[-1].strip() if "</think>" in raw else raw
        return {**result, "raw": final, "reasoning": raw}
    if family == "evidence":
        from schema_study.production_runtime import evidence_record

        record = evidence_record(result.get("raw") or "", request.get("source_text") or "", result.get("spans") or request.get("spans") or [])
        return {**result, "labels": result.get("labels") if record["complete"] else None, "evidence": record}
    if family == "qualified":
        from schema_study.production_runtime import evidence_record

        record = evidence_record(result.get("raw") or "", request.get("source_text") or "", result.get("spans") or [])
        qualifications = result.get("qualifications") or []
        complete = record["complete"] and bool(qualifications)
        return {**result, "labels": result.get("labels") if complete else None, "qualifications": qualifications, "evidence": record}
    if family == "self_consistency":
        from schema_study.production_runtime import interpret_generation, self_consistency as aggregate

        samples = []
        for sample in result.get("samples") or []:
            raw = sample.get("raw") if isinstance(sample, dict) else str(sample)
            interpreted = interpret_generation(raw, request["candidate_id"], schema, level) if schema is not None and level else {}
            samples.append({
                "raw": raw,
                "predicted_labels": interpreted.get("predicted_labels", sample.get("labels") or []),
                "ledger_status": interpreted.get("ledger_status", "invalid"),
                "illegal_labels": interpreted.get("illegal_labels", []),
            })
        if len(samples) != 5:
            raise RuntimeError("Self-consistency did not retain five samples from the loaded engine.")
        aggregated = aggregate(samples, bool(request.get("multi")))
        return {**result, "raw": result.get("raw") or json.dumps([sample["raw"] for sample in samples]), "labels": aggregated["labels"], "samples": samples, "scores": {"valid_samples": aggregated["valid_samples"], "aggregation": aggregated["aggregation"]}}
    if family == "candidate_scoring":
        from schema_study.production_runtime import require_legal_scores, sequence_logprobs

        legal = list(request.get("legal_labels") or []) or list((result.get("token_logprobs") or {}).keys())
        scored = require_legal_scores(sequence_logprobs(result["token_logprobs"]), legal, bool(request.get("multi")))
        labels = [] if scored["labels"] == ["[]"] else scored["labels"]
        return {"raw": json.dumps({"labels": labels}), "labels": labels, "scores": scored, "token_logprobs": result["token_logprobs"]}
    return result


def _write_predictions(path: Path, records: dict[str, dict]) -> None:
    temporary = path.with_suffix(".jsonl.tmp")
    ordered = [records[key] for key in sorted(records)]
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered), encoding="utf-8")
    temporary.replace(path)


def _request_key(row: dict) -> tuple:
    return (
        row.get("candidate_id"),
        row.get("entity_level") or row.get("entity"),
        row.get("predicate_level") or row.get("predicate"),
    )


def _generation_sources(task: dict, workdir: Path) -> tuple[list[Path], set | None, dict]:
    """One verified directory, or the selected-view population across prepared shards."""
    from schema_study.production_runtime import validate_request_shards

    if task.get("requests") or not task.get("population_filter"):
        shard_dir = Path(task["shard_directory"]) if task.get("shard_directory") else workdir / "shards"
        checked = validate_request_shards(shard_dir)
        return [shard_dir], None, checked
    root = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/runs/formal/formal_matched_e0_e7/generation_inputs/zero_shot") / str(task.get("dataset"))
    directories = sorted({path.parent for path in root.glob("*/*/shard_*.jsonl")})
    if not directories:
        return [], None, {"ready": False, "reason": "population shards are absent"}
    if task.get("population_filter") == "bc8_all":
        return directories, None, {"ready": True}
    from schema_study.production_runtime import select_population

    source = {
        "family": task["population_filter"],
        "dataset": task.get("dataset"),
        "entity": task.get("entity"),
        "predicate": task.get("predicate"),
        "seed": task.get("seed") or 13,
        "exposure": task.get("exposure"),
        "withheld": task.get("withheld"),
        "order": task.get("order"),
    }
    population = select_population(source, _metadata_rows(source))
    allowed = {_request_key(row) for row in population.get("final_evaluation") or []}
    if not allowed:
        return directories, allowed, {"ready": False, "reason": "the selected final population is empty"}
    return directories, allowed, {"ready": True}


def _run_generation(task: dict, workdir: Path) -> dict:
    """Read verified shards, call one backend session, and publish scientific predictions."""
    from schema_study.production_inference import classify_output, pending_requests
    from schema_study.production_runtime import (
        generation_requests,
        prediction_record,
        prediction_usable,
        write_shards,
    )

    if task.get("requests"):
        import hashlib

        prepared = generation_requests(task["requests"])
        for row in prepared:
            row["rendered"] = row.get("rendered") or row["prompt"]
            row["rendered_sha256"] = row.get("rendered_sha256") or hashlib.sha256(row["rendered"].encode("utf-8")).hexdigest()
        write_shards(prepared, workdir / "shards", int(task.get("shard_size") or 32))
        marker = {"requests": len(prepared), "shards": sorted(path.name for path in (workdir / "shards").glob("shard_*.jsonl"))}
        (workdir / "shards" / "COMPLETE").write_text(json.dumps(marker) + "\n", encoding="utf-8")
    directories, allowed, checked = _generation_sources(task, workdir)
    if not checked["ready"]:
        return {"completed": False, "status": "incomplete_inputs", "validation": checked}
    ledger_path = workdir / "predictions.jsonl"
    published: dict[str, dict] = {}
    if ledger_path.is_file():
        retained = []
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if prediction_usable(row):
                    retained.append(row)
        compatible = [row for row in retained if _resume_compatible(task, row)]
        if retained and len(compatible) != len(retained):
            archive = ledger_path.with_name("predictions_previous.jsonl")
            if not archive.exists():
                ledger_path.replace(archive)
            retained = []
        for row in retained:
            published[row["request_id"]] = row
    statuses = {key: row["status"] for key, row in published.items()}
    session = construct_backend(task)
    schema = _generation_schema(task)
    expected = []
    for shard_dir in directories:
        for path in sorted(shard_dir.glob("shard_*.jsonl")):
            prepared = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if allowed is not None:
                prepared = [row for row in prepared if _request_key(row) in allowed]
            expected.extend(row["request_id"] for row in prepared)
            runnable = [row for row in prepared if row["request_id"] in set(pending_requests([row["request_id"] for row in prepared], statuses))]
            if not runnable:
                continue
            for request, result in zip(runnable, session.generate_batch(runnable)):
                level = task.get("schema_predicate")
                shaped = _family_result(task["family"], request, result, schema, level)
                shaped = _parse_generated(task, request, shaped, schema)
                status = classify_output(shaped.get("raw"), shaped.get("labels"), failed=bool(shaped.get("failed")))
                record = prediction_record(request, shaped, task, session.specification, status)
                if task.get("family") == "thinking":
                    record["thinking_enabled"] = True
                if shaped.get("samples"):
                    record["samples"] = shaped["samples"]
                if prediction_usable(record):
                    published[request["request_id"]] = record
                    statuses[request["request_id"]] = status
            _write_predictions(ledger_path, published)
    usable = [prediction_usable(published.get(identifier, {})) for identifier in expected]
    scientific_done = bool(expected) and len(published) == len(expected) and all(usable) and len(set(expected)) == len(expected)
    status_name = "inference_completed" if scientific_done else "valid_continuation_required"
    (workdir / "scientific_status.json").write_text(json.dumps({
        "scientific": status_name,
        "requests": len(expected),
        "predictions_usable": scientific_done,
        "engine_loads": session.loads,
        "configuration_verified": True,
        "thinking_enabled": task.get("family") == "thinking",
        "samples_retained": 5 if task.get("family") == "self_consistency" else None,
    }) + "\n", encoding="utf-8")
    return {
        "completed": scientific_done,
        "status": status_name,
        "requests": len(expected),
        "resumed": 0 if not any(identifier not in statuses for identifier in expected) else len(expected) - len(published),
        "engine_loads": session.loads,
        "backend": session.specification,
    }


def _resume_lora(workdir: Path, model, configuration: dict, fingerprints: dict):
    """Restore a published checkpoint instead of starting the same task again."""
    path = workdir / "adapter_checkpoint.pt"
    if not path.is_file():
        return None, None
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not fingerprints.get("input") or not saved.get("fingerprints", {}).get("input"):
        raise RuntimeError("The input fingerprint is missing. A null fingerprint is not compatibility.")
    if saved["fingerprints"]["input"] != fingerprints["input"]:
        raise RuntimeError("The checkpoint input fingerprint does not match the task.")
    if saved.get("fingerprints", {}).get("configuration") != fingerprints.get("configuration"):
        raise RuntimeError("The checkpoint configuration fingerprint does not match the task.")
    saved_world = int((saved.get("configuration") or {}).get("world_size", 1))
    if saved_world != int(configuration.get("world_size", 1)):
        raise RuntimeError("Changing the distributed world size during continuation is not compatible.")
    return saved, model


def _lora_development_score(task: dict, evaluation: list[dict], tokenizer, prediction_path: Path | None = None):
    """Score the development population with the causal model. Test and BC8 rows are not used."""
    from schema_study.formal_prepare import _fill
    from schema_study.learning_conditions import HISTORICAL_PREDICATES
    from schema_study.production_runtime import interpret_generation, positive_micro_f1
    from schema_study.schemas import default_schema_path, load_schema

    dataset = task["dataset"]
    schema = load_schema(default_schema_path(get_paths().code_root, "drugprot" if dataset == "drugprot" else "biored"))
    needed = {(row["candidate_id"], _view_entity(task, row)) for row in evaluation}
    texts = _context_texts(dataset, needed)

    prompts = []
    for record in evaluation:
        if record.get("split") == "test":
            raise RuntimeError("A test row entered development selection.")
        entity = _view_entity(task, record)
        predicate = record.get("predicate") or task.get("predicate")
        historical = HISTORICAL_PREDICATES.get(predicate, predicate)
        block = "\n".join(f"- {name}" for name in schema.positive_display_names(historical))
        user = _fill(dataset, record["candidate_id"], historical, texts[(record["candidate_id"], entity)], {(dataset, historical): block})
        prompts.append((record, predicate, historical, user))

    def score(model) -> float:
        import os
        from contextlib import nullcontext

        import torch

        rank = int(os.environ.get("RANK", "0"))
        generator = model.module if hasattr(model, "module") else model
        if rank == 0 and not hasattr(generator, "generate"):
            raise RuntimeError("Development selection requires the model's generation interface.")
        device = next(generator.parameters()).device
        if device.type == "cpu" and os.environ.get("SCHEMA_STUDY_REQUIRE_CUDA") == "1":
            raise RuntimeError("Development generation remained on CPU.")
        was_training = model.training
        model.eval()
        rows = []
        context = model.summon_full_params(writeback=False, rank0_only=True) if hasattr(model, "summon_full_params") else nullcontext()

        def _generate():
            for start in range(0, len(prompts), 4):
                chunk = prompts[start:start + 4]
                encoded = [tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=True, add_generation_prompt=True, enable_thinking=False) for _record, _predicate, _historical, user in chunk]
                width = max(len(ids) for ids in encoded)
                tokens = torch.full((len(chunk), width), int(tokenizer.pad_token_id or 0), dtype=torch.long, device=device)
                mask = torch.zeros(len(chunk), width, dtype=torch.long, device=device)
                for index, ids in enumerate(encoded):
                    tokens[index, : len(ids)] = torch.tensor(ids, device=device)
                    mask[index, : len(ids)] = 1
                output = generator.generate(tokens, attention_mask=mask, max_new_tokens=256, do_sample=False)
                for index, (record, predicate, historical, _user) in enumerate(chunk):
                    text = tokenizer.decode(output[index][len(encoded[index]):], skip_special_tokens=True)
                    parsed = interpret_generation(text, record["candidate_id"], schema, historical)
                    gold = (record.get("assistant") or {}).get("labels", [])
                    rows.append({"gold": gold, "predicted": parsed["predicted_labels"], "status": "valid" if parsed["ledger_status"] in {"valid", "valid_empty"} else "invalid", "candidate_id": record["candidate_id"], "raw": text, "illegal_labels": parsed.get("illegal_labels", [])})
            return rows

        def _barrier() -> None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

        from schema_study.production_runtime import rank0_generation

        with context, torch.no_grad():
            rank0_generation(rank, _generate, _barrier)
        metric = float(positive_micro_f1(rows)["f1"]) if rank == 0 else 0.0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            value = torch.tensor([metric], dtype=torch.float64, device=device if device.type == "cuda" else "cpu")
            torch.distributed.broadcast(value, src=0)
            metric = float(value.item())
        if rank == 0 and prediction_path is not None:
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = prediction_path.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
            temporary.replace(prediction_path)
        if was_training:
            model.train()
        return metric

    return score


def _hierarchical_batches(task: dict, rows: list[dict], schema, corpus: str, keep_remainder: bool = False):
    """Tokenise the selected candidates. A sequence longer than the encoder window is rejected, not cut down."""
    import torch

    from schema_study.production_assets import encoder_layout
    from schema_study.production_records import multi_hot_target, single_label_target

    name = "biolinkbert" if task.get("model") == "biolinkbert" else "biomedbert"
    from schema_study.inputs import load_biomedbert_tokenizer

    tokenizer = load_biomedbert_tokenizer(str(encoder_layout(name)["tokenizer"]))
    needed = {(row["candidate_id"], row["entity"]) for row in rows}
    texts = _context_texts(task["dataset"], needed)
    bucket: list[tuple[list[int], dict, str]] = []
    for row in rows:
        encoded = tokenizer(texts[(row["candidate_id"], row["entity"])], add_special_tokens=True, truncation=False)
        if len(encoded["input_ids"]) > 512:
            raise RuntimeError(f"{row['candidate_id']} exceeds the encoder window. The source is not truncated.")
        targets = {}
        for predicate, labels in row["labels_by_predicate"].items():
            if corpus == "drugprot":
                targets[predicate] = multi_hot_target(schema, predicate, labels)
            else:
                targets[predicate] = single_label_target(schema, predicate, labels)
        bucket.append((list(encoded["input_ids"]), targets, row["candidate_id"]))
        if len(bucket) == 16:
            yield _stack_hierarchical(bucket, corpus)
            bucket = []
    if keep_remainder and bucket:
        yield _stack_hierarchical(bucket, corpus)

def _stack_hierarchical(bucket: list[tuple[list[int], dict, str]], corpus: str):
    import torch

    width = max(len(ids) for ids, _targets, _candidate in bucket)
    tokens = torch.zeros(len(bucket), width, dtype=torch.long)
    mask = torch.zeros(len(bucket), width, dtype=torch.long)
    for index, (ids, _targets, _candidate) in enumerate(bucket):
        tokens[index, : len(ids)] = torch.tensor(ids)
        mask[index, : len(ids)] = 1
    targets = {}
    predicates = bucket[0][1]
    for predicate in predicates:
        values = [item[predicate] for _ids, item, _candidate in bucket]
        if corpus == "drugprot":
            targets[predicate] = torch.tensor(values, dtype=torch.float32)
        else:
            targets[predicate] = torch.tensor(values, dtype=torch.long)
    return {"input_ids": tokens, "attention_mask": mask, "targets": targets, "candidate_ids": [candidate for _ids, _targets, candidate in bucket]}


def _hierarchical_development_score(task: dict, evaluation: list[dict], schema, corpus: str):
    from schema_study.learning_conditions import HISTORICAL_PREDICATES
    from schema_study.production_runtime import DRUGPROT_DEVELOPMENT_DECISION, positive_micro_f1

    prepared = list(_hierarchical_batches(task, evaluation, schema, corpus, keep_remainder=True))
    selected_predicate = task.get("predicate") or "P3"

    def score(model) -> float:
        import torch

        rows = []
        was_training = model.training
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            for batch in prepared:
                outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                for index, candidate_id in enumerate(batch["candidate_ids"]):
                    heads = {}
                    for predicate, logits in outputs.items():
                        labels = [label.stable_id for label in schema.level_labels(HISTORICAL_PREDICATES[predicate])]
                        if corpus == "drugprot":
                            chosen = (logits[index].sigmoid() > DRUGPROT_DEVELOPMENT_DECISION["threshold"]).tolist()
                            gold_row = batch["targets"][predicate][index].tolist()
                            heads[predicate] = {
                                "gold": [labels[position] for position, value in enumerate(gold_row) if value],
                                "predicted": [labels[position] for position, value in enumerate(chosen) if value],
                            }
                        else:
                            gold_index = int(batch["targets"][predicate][index])
                            predicted_index = int(logits[index].argmax())
                            heads[predicate] = {
                                "gold": [] if gold_index >= len(labels) else [labels[gold_index]],
                                "predicted": [] if predicted_index >= len(labels) else [labels[predicted_index]],
                            }
                    selected = heads[selected_predicate]
                    rows.append({"candidate_id": candidate_id, "gold": selected["gold"], "predicted": selected["predicted"], "status": "valid", "heads": heads, "decision": DRUGPROT_DEVELOPMENT_DECISION if corpus == "drugprot" else {"rule": "argmax_excluding_negative_class"}})
        if was_training:
            model.train()
        return float(positive_micro_f1(rows)["f1"])

    return score


def _evaluate_hierarchical(task: dict, workdir: Path) -> dict:
    """Score the final populations with the selected hierarchical checkpoint."""
    checkpoint = Path(task["checkpoint_source"])
    if not checkpoint.is_file():
        return {"completed": False, "status": "valid_continuation_required", "reason": "selected checkpoint is absent"}
    import time

    import torch

    from schema_study.formal_train import _build_rows
    from schema_study.learning_conditions import HISTORICAL_PREDICATES, RESEARCH_PREDICATES
    from schema_study.production_encoder import SchemaHierarchicalModel, prepare_marker_encoder
    from schema_study.production_records import head_sizes
    from schema_study.production_runtime import DRUGPROT_DEVELOPMENT_DECISION, publish_scientific_status
    from schema_study.schemas import default_schema_path, load_schema

    corpus = "drugprot" if task.get("dataset") == "drugprot" else "biored"
    schema = load_schema(default_schema_path(get_paths().code_root, corpus))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder_name = "biolinkbert" if task.get("model") == "biolinkbert" else "biomedbert"
    encoder, _tokenizer = prepare_marker_encoder(encoder_name, int(task.get("seed") or 13))
    model = SchemaHierarchicalModel(encoder, int(encoder.config.hidden_size), head_sizes(schema, corpus), corpus)
    model.load_state_dict(saved["state_dict"])
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    targets = [("biored", "test"), ("bc8", "bc8")] if task.get("dataset") == "biored" else [("drugprot", "eval")]
    destination = workdir / "predictions.jsonl"
    temporary = destination.with_suffix(".jsonl.tmp")
    written = 0
    started = time.time()
    with temporary.open("w", encoding="utf-8") as handle, torch.no_grad():
        for target_dataset, split in targets:
            grouped: dict[str, dict] = {}
            for predicate in RESEARCH_PREDICATES:
                rows, _classes, _target_schema = _build_rows(target_dataset, task["entity"], predicate, {split})
                for row in rows:
                    slot = grouped.setdefault(row["candidate_id"], {
                        "candidate_id": row["candidate_id"],
                        "document_id": row.get("document_id", row["candidate_id"]),
                        "entity": task["entity"],
                        "dataset": target_dataset,
                        "split": split,
                        "labels_by_predicate": {},
                    })
                    slot["labels_by_predicate"][predicate] = list(row.get("mapped_gold") or [])
            population = [row for row in grouped.values() if set(RESEARCH_PREDICATES) <= set(row["labels_by_predicate"])]
            if not population:
                return {"completed": False, "status": "valid_continuation_required", "reason": f"{target_dataset} final population is empty"}
            scoring = {**task, "dataset": target_dataset}
            device = next(model.parameters()).device
            for batch in _hierarchical_batches(scoring, population, schema, corpus, keep_remainder=True):
                outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                for index, candidate_id in enumerate(batch["candidate_ids"]):
                    heads = {}
                    for predicate, logits in outputs.items():
                        labels = [label.stable_id for label in schema.level_labels(HISTORICAL_PREDICATES[predicate])]
                        if corpus == "drugprot":
                            chosen = (logits[index].sigmoid() > DRUGPROT_DEVELOPMENT_DECISION["threshold"]).tolist()
                            heads[predicate] = [labels[position] for position, value in enumerate(chosen) if value]
                        else:
                            predicted_index = int(logits[index].argmax())
                            heads[predicate] = [] if predicted_index >= len(labels) else [labels[predicted_index]]
                    handle.write(json.dumps({
                        "candidate_id": candidate_id,
                        "dataset": target_dataset,
                        "split": split,
                        "entity": task.get("entity"),
                        "predicted_labels": heads,
                        "status": "valid",
                        "checkpoint": str(checkpoint),
                        "model": encoder_name,
                    }, sort_keys=True) + "\n")
                    written += 1
    temporary.replace(destination)
    publish_scientific_status(workdir, {
        "scientific": "inference_completed",
        "predictions_usable": written > 0,
        "checkpoint_validated": True,
        "technical_preflight": True,
        "useful_computation_started": started,
        "requests": written,
    })
    return {"completed": written > 0, "status": "inference_completed", "predictions_usable": written > 0, "requests": written}


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print("Usage: production_dispatch resolve|cpu-lora|cpu-hierarchical|cpu-task|run", flush=True)
        return 2
    command = arguments[0]
    if command == "resolve":
        payload = resolve_all()
        print(json.dumps(payload), flush=True)
        return 0 if not payload["missing_executor"] and not payload["marked_executable"] else 1
    if command == "cpu-lora":
        payload = _call("schema_study.production_lora:compare_interrupted_run")
        destination = get_paths().study / "reports" / "experimental_preparation" / "lora_continuation.json"
        get_paths().assert_output(destination)
        publish_file(destination, lambda temporary: temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8"))
        print(json.dumps(payload), flush=True)
        return 0 if payload["weights_match"] and payload["losses_match"] and payload["consumed_match"] else 1
    if command == "cpu-hierarchical":
        from schema_study.production_encoder import train_hierarchical

        payload = train_hierarchical(consistency=True)
        print(json.dumps(payload), flush=True)
        return 0 if payload["all_heads_receive_gradient"] and payload["shared_encoder_gradient"] else 1
    if command == "cpu-task":
        workdir = Path(arguments[1])
        result = execute_task({"model": "cpu_fixture", "family": "cpu_fixture", "task_id": "cpu", "executor": "schema_study.production_lora:fit"}, workdir, gpu_permitted=False)
        print(json.dumps({"completed": result["completed"]}), flush=True)
        return 0 if result["completed"] else 1
    if command == "run":
        if len(arguments) != 3:
            print("Usage: production_dispatch run FAMILY TASK_ID", flush=True)
            return 2
        task = _lookup(arguments[1], arguments[2])
        if task is None:
            print("unknown_task", flush=True)
            return 1
        if not gpu_authorised(get_paths().study, get_paths().code_root):
            print("formal_gpu_submission=blocked", flush=True)
            return 3
        workdir = Path(task["output"])
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            result = execute_task(task, workdir, gpu_permitted=True)
        except PermissionError:
            print("formal_gpu_submission=blocked", flush=True)
            return 3
        print(json.dumps({"completed": result.get("completed", False)}), flush=True)
        return 0 if result.get("completed") else 1
    print(f"Unknown command {command}", flush=True)
    return 2


def _call(dotted: str, *args):
    module_name, function_name = dotted.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)(*args)


if __name__ == "__main__":
    raise SystemExit(main())
