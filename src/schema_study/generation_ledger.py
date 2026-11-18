"""Save generation separately from scoring.

A completed answer ledger stays completed when scoring fails. Restarting a job
reads these status files and runs only the missing step.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def completed_ids(rows: list[dict]) -> set[str]:
    return {row["request_id"] for row in rows if row.get("status") == "completed" and not row.get("repeat_check")}


def select_pending(requests: list[dict], done: set[str], stage: str, first_count: int) -> list[dict]:
    if stage == "first":
        return [row for row in requests[:first_count] if row["request_id"] not in done]
    if stage == "resume":
        return [row for row in requests if row["request_id"] not in done]
    if stage == "repeat":
        return requests[:2]
    raise ValueError(stage)


def mark_generation(output_dir: Path, payload: dict) -> None:
    current = read_json(output_dir / "status_generation.json")
    current.update(payload)
    current["step"] = "generation"
    write_json(output_dir / "status_generation.json", current)


def mark_score(output_dir: Path, payload: dict) -> None:
    current = read_json(output_dir / "status_score.json")
    current.update(payload)
    current["step"] = "score"
    write_json(output_dir / "status_score.json", current)


def remaining_steps(output_dir: Path) -> list[str]:
    generation = read_json(output_dir / "status_generation.json")
    score = read_json(output_dir / "status_score.json")
    steps = []
    if not generation.get("complete"):
        steps.append("generate")
    if not score.get("complete"):
        steps.append("score")
    return steps


def resource_path(output_dir: Path, process_id: str) -> Path:
    safe = process_id.replace("/", "_")
    return output_dir / f"resource_usage_{safe}.json"
