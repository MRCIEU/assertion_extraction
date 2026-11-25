"""Keep a bounded set of formal Slurm jobs in the queue.

The controller does not change tasks because of scores. It stops submitting
when the allocated-time estimate would pass 1000 GPU hours.
"""

from __future__ import annotations

import json
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


def matched_main(arguments: list[str]) -> int:
    """Plan the matched programme. Submission stays closed until the approval file matches."""
    from schema_study.paths import get_paths
    from schema_study.production_plan import gpu_authorised

    plan = matched_plan()
    print(json.dumps({"tasks": plan["tasks"], "submitted": False}), flush=True)
    if "--submit" in arguments:
        if not gpu_authorised(get_paths().study, get_paths().code_root):
            print("formal_gpu_submission=blocked", flush=True)
            return 3
        print("formal_gpu_submission=refused_in_this_entry_until_the_operator_confirms", flush=True)
        return 3
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
