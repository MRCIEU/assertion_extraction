"""Keep a bounded set of formal Slurm jobs in the queue.

The controller does not change tasks because of scores. It stops submitting
when the allocated-time estimate would pass 1000 GPU hours.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/runs/formal/formal_20261115")
REPORT = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/reports/formal/formal_20261115")
LOG = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/job_logs")
TRAIN = Path("/home/b5ac/freddieyu.b5ac/project_1/scripts/slurm/gpu_formal_train.sbatch")
GENERATE = Path("/home/b5ac/freddieyu.b5ac/project_1/scripts/slurm/gpu_formal_generate.sbatch")
SUMMARY = json.loads((ROOT / "prepare/length_summary.json").read_text(encoding="utf-8"))
SHARDS = {
    (item["dataset"], item["entity_level"], item["predicate_level"], int(item["shard_index"])): item
    for item in SUMMARY["shards"]
}
GPU_LIMIT_SECONDS = 1000 * 3600


def _read_status() -> list[dict]:
    lines = (ROOT / "execution_manifest.tsv").read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        rows.append(dict(zip(header, parts + [""] * (len(header) - len(parts)))))
    return rows


def _write_status(rows: list[dict]) -> None:
    header = ["task_id", "kind", "dataset", "entity_level", "predicate_level", "model", "seed", "shard", "requests", "status", "job_id"]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(row.get(key, "") for key in header))
    text = "\n".join(lines) + "\n"
    (ROOT / "execution_manifest.tsv").write_text(text, encoding="utf-8")
    (REPORT / "execution_status.tsv").write_text(text, encoding="utf-8")


def _complete(row: dict) -> bool:
    if row["kind"] == "train":
        path = ROOT / "train" / row["dataset"] / row["entity_level"] / row["predicate_level"] / f"seed{row['seed']}" / "prediction_complete.json"
        return path.is_file()
    shard = SHARDS[(row["dataset"], row["entity_level"], row["predicate_level"], int(row["shard"]))]
    attempts = ROOT / "generation/outputs" / row["model"] / row["dataset"] / row["entity_level"] / row["predicate_level"] / f"shard_{int(row['shard']):05d}" / "attempts.jsonl"
    if not attempts.is_file():
        return False
    adopted = 0
    for line in attempts.read_text(encoding="utf-8").splitlines():
        if '"adoption": "adopt"' in line or '"adoption":"adopt"' in line:
            adopted += 1
    return adopted >= int(shard["requests"])


def _squeue() -> set[str] | None:
    """Return running or pending job ids. None means the queue could not be read."""
    result = subprocess.run(["squeue", "-u", "freddieyu.b5ac", "-h", "-o", "%i"], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _job_has_left_queue(job_id: str, active: set[str]) -> bool:
    """A missing squeue id is not enough. sacct must show a finished state.

    An empty or failed queue listing must not cause another submission.
    """
    if job_id in active:
        return False
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=JobID,State", "-P", "-n"],
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return False
    state = ""
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) == 2 and parts[0] == job_id:
            state = parts[1]
            break
    if not state:
        return False
    finished = ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "DEADLINE", "PREEMPTED")
    return state.startswith(finished)


def _elapsed_seconds() -> int:
    state_path = ROOT / "launch_jobs.json"
    if not state_path.is_file():
        return 0
    jobs = json.loads(state_path.read_text(encoding="utf-8"))
    if not jobs:
        return 0
    result = subprocess.run(
        ["sacct", "-j", ",".join(jobs), "--format=JobID,Elapsed", "-P", "-n"],
        check=False, capture_output=True, text=True,
    )
    total = 0
    for line in result.stdout.splitlines():
        job_id, elapsed = line.split("|")
        if "." in job_id:
            continue
        days = 0
        if "-" in elapsed:
            day_text, elapsed = elapsed.split("-", 1)
            days = int(day_text)
        parts = elapsed.split(":")
        if len(parts) < 2 or not all(part.isdigit() for part in parts):
            continue
        hours, minutes, seconds = (["0"] + parts)[-3:]
        total += days * 86400 + int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return total


def _inflight_seconds(row: dict) -> int:
    if row["kind"] == "train" and row["dataset"] == "drugprot":
        return 8 * 3600
    if row["kind"] == "train":
        return 4 * 3600
    return 2 * 3600


def _submit(row: dict) -> str:
    if row["kind"] == "train":
        limit = "08:00:00" if row["dataset"] == "drugprot" else "04:00:00"
        command = ["sbatch", "--time", limit, "--job-name", f"p1-tr-{row['dataset'][:2]}", "--output", str(LOG / "formal_%j.out"), "--error", str(LOG / "formal_%j.err"), str(TRAIN), row["dataset"], row["entity_level"], row["predicate_level"], row["seed"]]
    else:
        shard = SHARDS[(row["dataset"], row["entity_level"], row["predicate_level"], int(row["shard"]))]
        memory = "128G" if row["model"] == "qwen32b" else "64G"
        command = ["sbatch", "--mem", memory, "--job-name", f"p1-gen-{row['model'][-2:]}", "--output", str(LOG / "formal_%j.out"), "--error", str(LOG / "formal_%j.err"), str(GENERATE), row["model"], shard["path"]]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split()[-1]
    return job_id


def _priority(row: dict) -> tuple:
    if row["kind"] == "train" and row["dataset"] == "drugprot":
        return (0, row["task_id"])
    if row["kind"] == "generate" and row["model"] == "qwen32b":
        return (1, row["task_id"])
    if row["kind"] == "generate":
        return (2, row["task_id"])
    return (3, row["task_id"])


def _elapsed_seconds(token: str) -> int:
    """Slurm elapsed time. A pending job contributes nothing to the observation window."""
    days = 0
    clock = token
    if "-" in token:
        day_text, clock = token.split("-", 1)
        days = int(day_text)
    parts = [int(item) for item in clock.split(":")]
    if len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    else:
        hours, minutes, seconds = parts
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def _queue_snapshot() -> tuple[dict[str, str] | None, dict]:
    """Map job id to PENDING or RUNNING. A failed query is not an empty queue."""
    result = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %T %M"], capture_output=True, text=True)
    if result.returncode != 0:
        return None, {"running": 0, "pending": 0, "elapsed": {}}
    queued = {}
    elapsed = {}
    running = pending = 0
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        queued[parts[0]] = parts[1]
        if parts[1] == "RUNNING":
            running += 1
            elapsed[parts[0]] = _elapsed_seconds(parts[2]) if len(parts) > 2 else 0
        elif parts[1] == "PENDING":
            pending += 1
            elapsed[parts[0]] = 0
    return queued, {"running": running, "pending": pending, "elapsed": elapsed}


def _sacct_state(job_id: str) -> str | None:
    result = subprocess.run(["sacct", "-j", job_id, "--format=JobID,State", "-P", "-n"], check=False, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) == 2 and parts[0] == job_id:
            return parts[1]
    return None


def _scientific_output(task: dict) -> dict | None:
    """Read the task's own completion record. A Slurm COMPLETED line is not this record."""
    path = Path(task["output"]) / "scientific_status.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def matched_plan() -> dict:
    """Resource groups for the new programme. This function does not call sbatch."""
    from schema_study.production_plan import execution_manifest, submission_allowed

    tasks = execution_manifest()
    groups: dict[str, int] = {}
    for task in tasks:
        if "blocked" in task["statuses"]:
            key = "blocked"
        else:
            resources = task.get("resources") or {}
            key = f"{task['family']}|gpus={resources.get('gpus')}|time={task.get('walltime')}"
        groups[key] = groups.get(key, 0) + 1
    allowed, reason = submission_allowed(0, 0, min(sum(groups.values()), 512))
    return {"groups": groups, "tasks": len(tasks), "submission_probe": {"allowed": allowed, "reason": reason}, "submitted": False}


def requeue_permitted(authorised: bool, finished: bool) -> bool:
    """A controller may requeue itself only after approval and only while work remains."""
    return bool(authorised) and not finished


def submission_commands(limit: int = 256) -> list[list[str]]:
    """Commands for ready, unblocked tasks. Building this list does not call sbatch."""
    from schema_study.production_plan import execution_manifest, submission_allowed
    from schema_study.production_runtime import sbatch_argv

    tasks = [task for task in execution_manifest() if "blocked" not in task["statuses"]]
    allowed, reason = submission_allowed(0, 0, min(len(tasks), limit))
    if not allowed:
        raise RuntimeError(reason)
    return [sbatch_argv(task, "scripts/slurm/matched_task.sbatch") for task in tasks[:limit]]


def matched_main(arguments: list[str]) -> int:
    """Plan the matched programme. sbatch runs only after the approval file matches this checkout."""
    from schema_study.paths import get_paths
    from schema_study.production_plan import gpu_authorised
    from schema_study.production_runtime import controller_requeue

    plan = matched_plan()
    print(json.dumps({"tasks": plan["tasks"], "submitted": False, "groups": len(plan["groups"])}), flush=True)
    if "--submit" not in arguments and "--control" not in arguments:
        return 0
    if not gpu_authorised(get_paths().study, get_paths().code_root):
        print("formal_gpu_submission=blocked", flush=True)
        return 3
    from schema_study.production_plan import executing_revision, execution_manifest
    from schema_study.production_runtime import (
        controller_decisions,
        controller_transaction,
        interpret_slurm,
        load_controller_state,
        merge_group_block,
        probe_readiness,
        programme_remaining,
        reconcile_entry,
        record_submission,
        save_controller_state,
        sbatch_argv,
    )

    tasks = [task for task in execution_manifest() if "blocked" not in task["statuses"]]
    state_path = get_paths().study / "runs" / "formal" / "formal_matched_e0_e7" / "controller_state.json"
    transaction = controller_transaction(state_path)
    try:
        transaction.__enter__()
    except RuntimeError:
        print("formal_gpu_submission=controller_busy", flush=True)
        return 3
    try:
        state = load_controller_state(state_path)
        queued, occupancy = _queue_snapshot()
        if queued is None:
            print("formal_gpu_submission=queue_unreadable", flush=True)
            return 3
        for task in tasks:
            entry = state.get(task["task_id"])
            if not entry or not entry.get("job_id"):
                continue
            observed = None if entry["job_id"] in queued else _sacct_state(entry["job_id"])
            scheduler = interpret_slurm(entry["job_id"], queued, observed)
            output = _scientific_output(task)
            updated = reconcile_entry(
                entry, scheduler, output, task_id=task["task_id"],
                group=f"{task.get('family')}|{task.get('model')}|{(task.get('resources') or {}).get('gpus')}",
                slurm_state=observed or "",
                elapsed_seconds=int(occupancy["elapsed"].get(entry["job_id"], 0)),
            )
            if (
                task.get("family") in {"thinking", "self_consistency", "gpt_oss", "evidence", "qualified", "constrained"}
                and updated.get("state") == "inference_completed"
                and not (output or {}).get("configuration_verified")
            ):
                recovery = state.setdefault("_recovery", {})
                prior = recovery.get(task["task_id"])
                events = prior if isinstance(prior, list) else ([] if prior is None else [prior])
                events.append({
                    "reason": "completed_output_did_not_use_the_corrected_decoding_configuration",
                    "job_id": entry.get("job_id"),
                    "preserved": "predictions_previous.jsonl",
                })
                recovery[task["task_id"]] = events
                updated = {
                    "state": "unsubmitted",
                    "attempts": int(entry.get("attempts", 0)),
                    "scientific": None,
                }
            state[task["task_id"]] = merge_group_block(state, updated)
        evidence = {"approval": {"ready": True}}
        for task in tasks:
            evidence.update(probe_readiness(task))
        running = occupancy["running"]
        pending = occupancy["pending"]
        decisions = controller_decisions(tasks, state, running, pending, evidence)
        by_id = {task["task_id"]: task for task in tasks}
        revision = executing_revision(get_paths().code_root) or ""
        submitted = 0
        for decision in decisions:
            if decision["action"] != "submit":
                continue
            command = sbatch_argv(by_id[decision["task_id"]], "scripts/slurm/matched_task.sbatch")
            command.insert(1, "--parsable")
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            record_submission(state, decision["task_id"], result.stdout.strip().split(";")[0], revision)
            save_controller_state(state_path, state, external_lock=True)
            submitted += 1
        save_controller_state(state_path, state, external_lock=True)
    finally:
        transaction.__exit__(None, None, None)
    active_ids = [entry.get("job_id") for entry in state.values() if isinstance(entry, dict) and entry.get("state") in {"submitted", "pending", "running"}]
    requeue = controller_requeue(os.environ.get("SLURM_JOB_ID", "0"), True, finished=not programme_remaining(tasks, state), active_job_ids=active_ids)
    if requeue is not None:
        subprocess.run(requeue, check=True)
    print(json.dumps({"submitted": submitted, "remaining": programme_remaining(tasks, state)}), flush=True)
    return 0


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "matched":
        raise SystemExit(matched_main(sys.argv[2:]))
    lock = ROOT / "launcher.pid"
    if lock.is_file():
        existing = lock.read_text(encoding="utf-8").strip()
        if existing.isdigit() and Path(f"/proc/{existing}").exists():
            raise SystemExit(f"Launcher already running as pid {existing}")
    lock.write_text(str(Path("/proc/self").resolve().name) + "\n", encoding="utf-8")
    state_path = ROOT / "launch_jobs.json"
    log_path = ROOT / "launcher.log"
    jobs = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else ["8465861", "8465862", "8465863", "8465864"]
    while True:
        try:
            policy = json.loads((ROOT / "launch_policy.json").read_text(encoding="utf-8"))
            rows = _read_status()
            active = _squeue()
            if active is None:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write("queue_unreadable\n")
                time.sleep(60)
                continue
            resumes_path = ROOT / "resume_counts.json"
            resumes = json.loads(resumes_path.read_text(encoding="utf-8")) if resumes_path.is_file() else {}
            for row in rows:
                if _complete(row):
                    row["status"] = "complete"
                    continue
                if row.get("job_id") and row["status"] == "running" and _job_has_left_queue(row["job_id"], active):
                    resumes[row["task_id"]] = int(resumes.get(row["task_id"], 0)) + 1
                    if resumes[row["task_id"]] > 2:
                        row["status"] = "failed"
                    else:
                        row["status"] = "planned"
                        row["job_id"] = ""
            resumes_path.write_text(json.dumps(resumes, indent=2) + "\n", encoding="utf-8")
            used = _elapsed_seconds()
            running = [row for row in rows if row.get("job_id") in active]
            inflight_limit = sum(_inflight_seconds(row) for row in running)
            drugprot_running = sum(row["dataset"] == "drugprot" and row["kind"] == "train" for row in running)
            room = int(policy["max_concurrent"]) - len(running)
            submitted = []
            if used + inflight_limit < GPU_LIMIT_SECONDS:
                for row in sorted((item for item in rows if item["status"] == "planned"), key=_priority):
                    if room <= 0:
                        break
                    if row["kind"] == "train" and row["dataset"] == "drugprot" and drugprot_running >= int(policy["max_drugprot"]):
                        continue
                    job_id = _submit(row)
                    row["status"] = "running"
                    row["job_id"] = job_id
                    jobs.append(job_id)
                    submitted.append(job_id)
                    room -= 1
                    drugprot_running += int(row["kind"] == "train" and row["dataset"] == "drugprot")
                    inflight_limit += _inflight_seconds(row)
                    if used + inflight_limit >= GPU_LIMIT_SECONDS:
                        break
            _write_status(rows)
            state_path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
            counts = {}
            for row in rows:
                counts[row["status"]] = counts.get(row["status"], 0) + 1
            message = json.dumps({"submitted": submitted, "running": len(running), "elapsed_gpu_seconds": used, "counts": counts})
            print(message, flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
            if counts.get("planned", 0) == 0 and not running and counts.get("failed", 0) == 0:
                break
        except Exception as exc:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"controller_error {type(exc).__name__}: {exc}\n")
            print(f"controller_error {type(exc).__name__}: {exc}", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
