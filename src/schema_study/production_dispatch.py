"""Slurm entry point. CPU fixtures run. GPU commands stay closed without bound approval."""

from __future__ import annotations

import json
import os
import sys

from schema_study.formal_optim import run_continuation_check, save_and_load_checkpoint
from schema_study.paths import get_paths
from schema_study.production_plan import gpu_authorised
from schema_study.records import publish_file


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print("Usage: production_dispatch cpu-fixture|gpu-train", flush=True)
        return 2
    command = arguments[0]
    if command == "cpu-fixture":
        destination = get_paths().study / "reports" / "experimental_preparation" / "cpu_continuation.json"
        get_paths().assert_output(destination)
        payload = run_continuation_check()
        payload["checkpoint"] = save_and_load_checkpoint(destination.parent / "cpu_fixture_checkpoint")
        publish_file(destination, lambda temporary: temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"))
        print(json.dumps({"weights_match": payload["weights_match"], "all_heads_receive_gradient": payload["all_heads_receive_gradient"]}), flush=True)
        return 0 if payload["weights_match"] and payload["all_heads_receive_gradient"] else 1
    if command == "gpu-train":
        commit = os.environ.get("SCHEMA_STUDY_COMMIT", "")
        if not gpu_authorised(get_paths().study, commit):
            print("formal_gpu_submission=blocked", flush=True)
            return 3
        sys.argv = ["formal_train", *arguments[1:]]
        from schema_study.formal_train import main as train_main

        train_main()
        return 0
    print(f"Unknown command {command}", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
