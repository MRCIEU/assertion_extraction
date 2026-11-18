"""One directory per execution process. Later processes do not replace it."""

from __future__ import annotations

import json
from pathlib import Path


def process_directory(output_dir: Path, job_id: str, attempt_id: str) -> Path:
    path = output_dir / "processes" / f"{job_id}_{attempt_id}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_process_json(directory: Path, name: str, payload: dict) -> Path:
    path = directory / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def summarize_processes(output_dir: Path) -> dict:
    """Read per-process files. This summary is derived and is not a raw record."""
    root = output_dir / "processes"
    processes = []
    if root.is_dir():
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            resource = directory / "resource_usage.json"
            clock = directory / "run_clock.json"
            processes.append({
                "process_id": directory.name,
                "resource": json.loads(resource.read_text(encoding="utf-8")) if resource.is_file() else None,
                "run_clock": json.loads(clock.read_text(encoding="utf-8")) if clock.is_file() else None,
            })
    return {"process_count": len(processes), "processes": processes}
