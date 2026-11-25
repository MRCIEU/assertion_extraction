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
    if executor == "schema_study.formal_train:train":
        from schema_study.formal_train import train

        learning_rate = task.get("learning_rate")
        train(
            task["dataset"],
            task["entity"],
            task.get("schema_predicate") or task["predicate"],
            int(task["seed"]),
            model_name=task.get("model") or "biomedbert",
            objective=task.get("loss") or "standard",
            fraction=int(task.get("fraction") or 100),
            negative_strategy=task.get("strategy"),
            learning_rate=learning_rate,
        )
        return {"completed": True}
    if executor == "schema_study.production_lora:fit":
        from schema_study.production_lora import fit, load_causal_lm, production_configuration

        model, _tokenizer = load_causal_lm(task["model"])
        configuration = production_configuration()
        configuration.update({key: task["configuration"][key] for key in task["configuration"] if key in configuration})
        configuration["base_revision"] = task.get("model_revision")
        configuration["tokenizer_revision"] = task.get("tokenizer_revision")
        configuration["micro_batch_size"] = 1
        configuration["world_size"] = int(task["resources"]["gpus"])
        fit(model, _supervised_batches(task, _tokenizer), configuration=configuration, steps=1, total_steps=1, output_dir=workdir)
        return {"completed": True}
    if executor == "schema_study.production_encoder:train_hierarchical":
        from schema_study.production_encoder import load_pretrained_encoder, train_hierarchical

        encoder = load_pretrained_encoder(task.get("model") or "biomedbert")
        hidden = int(encoder.config.hidden_size)
        from schema_study.production_encoder import SchemaHierarchicalModel
        from schema_study.production_records import head_sizes
        from schema_study.schemas import default_schema_path, load_schema

        corpus = task.get("dataset", "biored")
        schema = load_schema(default_schema_path(get_paths().code_root, "drugprot" if corpus == "drugprot" else "biored"))
        model = SchemaHierarchicalModel(encoder, hidden, head_sizes(schema, corpus), corpus)
        return train_hierarchical(model, consistency=bool(task.get("consistency")), corpus=corpus)
    module_name, function_name = executor.split(":")
    function = importlib.import_module(module_name)
    return {"completed": False, "reason": f"{function_name} requires its own prepared arguments", "module": function.__name__}


def _supervised_batches(task: dict, tokenizer):
    """Join source text to the supervised record, then tokenise. An identifier is not a training string."""
    from schema_study.formal_prepare import _fill
    from schema_study.learning_conditions import HISTORICAL_PREDICATES
    from schema_study.production_lora import batches_from_records
    from schema_study.production_records import INPUTS
    from schema_study.schemas import default_schema_path, load_schema

    dataset = task["dataset"]
    entity = task.get("entity") or "native"
    predicate = task.get("predicate")
    historical = HISTORICAL_PREDICATES.get(predicate, predicate)
    schema_name = "drugprot" if dataset == "drugprot" else "biored"
    schema = load_schema(default_schema_path(get_paths().code_root, schema_name))
    block = "\n".join(f"- {name}" for name in schema.positive_display_names(historical))
    texts = {}
    with INPUTS[f"contexts_{dataset}"].open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if entity in row.get("views", {}):
                texts[row["candidate_id"]] = row["views"][entity]

    def records():
        with INPUTS[f"sft_{dataset}"].open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("entity_level") != entity:
                    continue
                if predicate and row.get("predicate_level") != predicate:
                    continue
                source = texts.get(row["candidate_id"])
                if not source:
                    raise RuntimeError(f"{row['candidate_id']} has no source text.")
                yield {
                    "user": _fill(dataset, row["candidate_id"], historical, source, {(dataset, historical): block}),
                    "assistant": json.dumps({"labels": row["assistant"]["labels"]}, ensure_ascii=False),
                }

    yield from batches_from_records(records(), tokenizer, 1)


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
