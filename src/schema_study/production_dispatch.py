"""Dispatch a prepared task to its production executor.

Importing this module does not import PyTorch. GPU commands stay closed
until the resolved manifest is approved for the checkout git reports.
"""

from __future__ import annotations

import importlib
import json
import sys

from schema_study.paths import get_paths
from schema_study.production_plan import gpu_authorised, resolved_tasks
from schema_study.records import publish_file


def resolve_all() -> dict:
    tasks = resolved_tasks()
    missing = [task["task_id"] for task in tasks if not task.get("executor")]
    executable = [task["task_id"] for task in tasks if task["executable"]]
    return {"tasks": len(tasks), "missing_executor": missing, "marked_executable": executable}


def _call(dotted: str, *args):
    module_name, function_name = dotted.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)(*args)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print("Usage: production_dispatch resolve|cpu-lora|cpu-hierarchical|run", flush=True)
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
        return 0 if payload["weights_match"] and payload["losses_match"] else 1
    if command == "cpu-hierarchical":
        from schema_study.production_encoder import ProductionHierarchicalEncoder, train_hierarchical

        payload = train_hierarchical(ProductionHierarchicalEncoder(), consistency=True)
        print(json.dumps(payload), flush=True)
        return 0 if payload["all_heads_receive_gradient"] and payload["shared_encoder_gradient"] else 1
    if command == "run":
        if not gpu_authorised(get_paths().study, get_paths().code_root):
            print("formal_gpu_submission=blocked", flush=True)
            return 3
        print("formal_gpu_submission=authorised", flush=True)
        return 0
    print(f"Unknown command {command}", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
