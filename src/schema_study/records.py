"""Run identifiers and small tabular writers.

Environment records keep an explicit field list. They never copy the process
environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd
import yaml

ENVIRONMENT_FIELDS = (
    "recorded_at_utc",
    "hostname",
    "platform",
    "machine",
    "python_version",
    "cpu_count",
    "slurm_job_id",
    "slurm_partition",
    "slurm_cpus",
    "slurm_mem_mb",
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def digest_mapping(payload: Mapping) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return digest_text(encoded)


def make_run_id(purpose: str, config: Mapping) -> str:
    """UTC time, purpose, and a short configuration digest."""
    safe_purpose = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in purpose)
    return f"{utc_stamp()}_{safe_purpose}_{digest_mapping(config)[:12]}"


def environment_record() -> dict:
    return {
        "recorded_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_mem_mb": os.environ.get("SLURM_MEM_PER_NODE"),
    }


def publish_file(path: Path, write) -> Path:
    """Write to a temporary file and replace the destination only after success.

    An exception leaves the previous destination in place and removes the
    partial file. A reader therefore does not observe an unfinished result.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        write(temporary)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return destination


def write_json(path: Path, payload: Mapping | Sequence) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    def _write(temporary: Path) -> None:
        temporary.write_text(text, encoding="utf-8")

    publish_file(path, _write)


def write_jsonl(path: Path, rows: Iterable[Mapping]) -> int:
    count = 0

    def _write(temporary: Path) -> None:
        nonlocal count
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
                count += 1

    publish_file(path, _write)
    return count


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_yaml(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_parquet(path: Path, rows: Sequence[Mapping], columns: Sequence[str]) -> None:
    frame = pd.DataFrame(list(rows), columns=list(columns))

    def _write(temporary: Path) -> None:
        frame.to_parquet(temporary, index=False)

    publish_file(path, _write)


def write_tsv(path: Path, rows: Sequence[Mapping], columns: Sequence[str]) -> None:
    frame = pd.DataFrame(list(rows), columns=list(columns))

    def _write(temporary: Path) -> None:
        frame.to_csv(temporary, sep="\t", index=False)

    publish_file(path, _write)


def write_run_skeleton(
    run_dir: Path,
    *,
    run_id: str,
    purpose: str,
    config: Mapping,
    inputs: Mapping,
    status: str,
    message: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    write_yaml(run_dir / "resolved_config.yaml", dict(config))
    write_json(run_dir / "input_sources.json", dict(inputs))
    write_json(run_dir / "environment.json", environment_record())
    write_json(
        run_dir / "run_record.json",
        {
            "run_id": run_id,
            "purpose": purpose,
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config_digest": digest_mapping(config),
        },
    )
    write_json(
        run_dir / "status.json",
        {
            "run_id": run_id,
            "status": status,
            "message": message,
            "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
    write_tsv(run_dir / "output_files.tsv", [], ["path", "role", "bytes", "sha256"])
