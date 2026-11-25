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
    if not gpu_permitted:
        raise PermissionError("formal_gpu_execution_is_not_authorised")
    if not task.get("executor"):
        raise RuntimeError("The task has no production executor.")
    return _invoke_production(task, workdir)


def _invoke_production(task: dict, workdir: Path) -> dict:
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
        configuration["placement"] = placement(task["model"], requested)["method"]
        configuration["eval_interval"] = horizon["updates_per_epoch"]
        evaluation = population.get("evaluation") or _development_from_candidates(task)
        result = fit(
            model,
            _supervised_batches(task, tokenizer),
            configuration=configuration,
            steps=horizon["total_updates"],
            total_steps=horizon["total_updates"],
            output_dir=workdir,
            fingerprints={"input": (task.get("input") or {}).get("fingerprint"), "configuration": json.dumps(task.get("configuration"), sort_keys=True)},
            score_fn=_lora_development_score(task, evaluation, tokenizer),
        )
        return {"completed": result["completed"], "status": result["status"], "total_updates": horizon["total_updates"], "empirical_qwen_result": False}
    if executor == "schema_study.production_encoder:train_hierarchical":
        from schema_study.production_encoder import SchemaHierarchicalModel, load_pretrained_encoder, train_hierarchical
        from schema_study.production_records import head_sizes
        from schema_study.schemas import default_schema_path, load_schema

        from schema_study.production_runtime import horizon_for_task, select_population

        corpus = task.get("dataset", "biored")
        schema = load_schema(default_schema_path(get_paths().code_root, "drugprot" if corpus == "drugprot" else "biored"))
        population = select_population(task, _metadata_rows(task))
        horizon = horizon_for_task(task, population.get("candidates") or len(population["rows"]), 1, 1)
        encoder_name = "biolinkbert" if task.get("model") == "biolinkbert" else "biomedbert"
        encoder = load_pretrained_encoder(encoder_name)
        model = SchemaHierarchicalModel(encoder, int(encoder.config.hidden_size), head_sizes(schema, corpus), corpus)
        evaluation = population.get("evaluation") or _development_from_candidates(task, schema=schema, hierarchical=True)
        batches = _hierarchical_batches(task, population["rows"], schema, corpus)
        return train_hierarchical(
            model,
            steps=horizon["total_updates"],
            total_steps=horizon["total_updates"],
            consistency=bool(task.get("consistency")),
            corpus=corpus,
            batches=batches,
            score_fn=_hierarchical_development_score(task, evaluation, schema, corpus),
        )
    if executor.startswith("schema_study.production_inference:"):
        from schema_study.production_runtime import constrained_sampling, generation_requests

        requests = generation_requests(task.get("requests") or [])
        if task["family"] == "constrained":
            return {"completed": False, "status": "ready", "sampling": constrained_sampling({"type": "object"}), "requests": len(requests)}
        return {"completed": False, "status": "ready", "requests": len(requests), "reason": "Generation runs only inside an authorised GPU job with these prepared requests."}
    if executor == "schema_study.formal_generate:generate":
        return {"completed": False, "status": "ready", "reason": "The generation shard must contain complete prompts. Supervised identifiers are not prompts."}
    if executor == "schema_study.analysis_plan:analyse_records":
        from schema_study.analysis_plan import analyse_records

        documents = task.get("documents") or []
        if not documents:
            return {"completed": False, "status": "waiting_for_predictions"}
        report = analyse_records(documents)
        return {"completed": True, "status": "training_completed", "report": report}
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


def _supervised_batches(task: dict, tokenizer):
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
        for _repeat in range(_epoch_repeats(task)):
            for row in selected:
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


def _lora_development_score(task: dict, evaluation: list[dict], tokenizer):
    """Score the development population with the causal model. Test and BC8 rows are not used."""
    from schema_study.formal_prepare import _fill
    from schema_study.learning_conditions import HISTORICAL_PREDICATES
    from schema_study.production_runtime import interpret_generation, positive_micro_f1
    from schema_study.schemas import default_schema_path, load_schema

    dataset = task["dataset"]
    schema = load_schema(default_schema_path(get_paths().code_root, "drugprot" if dataset == "drugprot" else "biored"))
    needed = {(row["candidate_id"], _view_entity(task, row)) for row in evaluation}
    texts = _context_texts(dataset, needed)

    def score(model) -> float:
        import torch

        if not hasattr(model, "generate"):
            raise RuntimeError("Development selection requires the model's generation interface.")
        rows = []
        for record in evaluation:
            entity = _view_entity(task, record)
            predicate = record.get("predicate") or task.get("predicate")
            historical = HISTORICAL_PREDICATES.get(predicate, predicate)
            block = "\n".join(f"- {name}" for name in schema.positive_display_names(historical))
            user = _fill(dataset, record["candidate_id"], historical, texts[(record["candidate_id"], entity)], {(dataset, historical): block})
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=True, add_generation_prompt=True, enable_thinking=False,
            )
            output = model.generate(torch.tensor([prompt_ids]), max_new_tokens=64, do_sample=False)
            text = tokenizer.decode(output[0][len(prompt_ids):], skip_special_tokens=True)
            parsed = interpret_generation(text, record["candidate_id"], schema, historical)
            gold = (record.get("assistant") or {}).get("labels", record.get("labels_by_predicate", {}).get(predicate, []))
            rows.append({"gold": gold, "predicted": parsed["predicted_labels"], "status": "valid" if parsed["ledger_status"] in {"valid", "valid_empty"} else "invalid"})
        metric = positive_micro_f1(rows)
        return float(metric["f1"])

    return score


def _hierarchical_batches(task: dict, rows: list[dict], schema, corpus: str):
    """Tokenise the selected candidates. A sequence longer than the encoder window is rejected, not cut down."""
    import torch
    from transformers import AutoTokenizer

    from schema_study.production_assets import encoder_layout
    from schema_study.production_records import multi_hot_target, single_label_target

    name = "biolinkbert" if task.get("model") == "biolinkbert" else "biomedbert"
    tokenizer = AutoTokenizer.from_pretrained(encoder_layout(name)["tokenizer"], local_files_only=True)
    needed = {(row["candidate_id"], row["entity"]) for row in rows}
    texts = _context_texts(task["dataset"], needed)
    bucket: list[tuple[list[int], dict]] = []
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
        bucket.append((list(encoded["input_ids"]), targets))
        if len(bucket) == 16:
            yield _stack_hierarchical(bucket, corpus)
            bucket = []

def _stack_hierarchical(bucket: list[tuple[list[int], dict]], corpus: str):
    import torch

    width = max(len(ids) for ids, _targets in bucket)
    tokens = torch.zeros(len(bucket), width, dtype=torch.long)
    for index, (ids, _targets) in enumerate(bucket):
        tokens[index, : len(ids)] = torch.tensor(ids)
    targets = {}
    predicates = bucket[0][1]
    for predicate in predicates:
        values = [item[predicate] for _ids, item in bucket]
        if corpus == "drugprot":
            targets[predicate] = torch.tensor(values, dtype=torch.float32)
        else:
            targets[predicate] = torch.tensor(values, dtype=torch.long)
    return tokens, targets


def _hierarchical_development_score(task: dict, evaluation: list[dict], schema, corpus: str):
    from schema_study.production_runtime import positive_micro_f1

    prepared = list(_hierarchical_batches(task, evaluation, schema, corpus))

    def score(model) -> float:
        import torch

        rows = []
        model.eval()
        with torch.no_grad():
            for tokens, targets in prepared:
                outputs = model(tokens)
                logits = outputs["P3"]
                gold = targets["P3"]
                if corpus == "drugprot":
                    predicted = (logits.sigmoid() > 0.5).tolist()
                    gold_rows = gold.tolist()
                    for gold_row, predicted_row in zip(gold_rows, predicted):
                        rows.append({"gold": [str(index) for index, value in enumerate(gold_row) if value], "predicted": [str(index) for index, value in enumerate(predicted_row) if value], "status": "valid"})
                else:
                    chosen = logits.argmax(dim=-1).tolist()
                    for gold_index, predicted_index in zip(gold.tolist(), chosen):
                        rows.append({"gold": [str(gold_index)], "predicted": [str(predicted_index)], "status": "valid"})
        model.train()
        return float(positive_micro_f1(rows)["f1"])

    return score


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
