"""Immutable generation batches and the resume rule.

An incomplete batch is repeated as a whole. Completed batches are skipped.
The first identity-checked answer for a request stays the adopted answer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def freeze_manifest(requests: list[dict], batch_size: int, prompt_field: str) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batches = []
    for start in range(0, len(requests), batch_size):
        group = requests[start:start + batch_size]
        batches.append({
            "batch_id": f"batch-{start // batch_size:04d}",
            "request_ids": [row["request_id"] for row in group],
            "members": [
                {
                    "request_id": row["request_id"],
                    "candidate_id": row["candidate_id"],
                    "dataset": row["dataset"],
                    "predicate_level": row["predicate_level"],
                    "prompt_sha256": prompt_sha256(row[prompt_field]),
                }
                for row in group
            ],
        })
    return {
        "immutable": True,
        "batch_size": batch_size,
        "prompt_field": prompt_field,
        "request_count": len(requests),
        "batches": batches,
    }


def write_manifest_once(path: Path, manifest: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite batch manifest {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def adopted_answers(attempts: list[dict]) -> dict[str, dict]:
    """Keep the first completed answer whose candidate id matches the request."""
    adopted = {}
    for row in attempts:
        request_id = row.get("request_id")
        if request_id in adopted:
            continue
        if row.get("status") != "completed":
            continue
        if row.get("is_warmup"):
            continue
        if row.get("candidate_id_checked") is False:
            continue
        adopted[request_id] = row
    return adopted


def batches_to_execute(manifest: dict, adopted: dict[str, dict]) -> list[dict]:
    """Return whole batches that are missing at least one adopted answer.

    Member order is the manifest order. Unfinished requests are not regrouped.
    """
    pending = []
    for batch in manifest["batches"]:
        if any(request_id not in adopted for request_id in batch["request_ids"]):
            pending.append(batch)
    return pending


def classify_attempt(batch: dict, row: dict, adopted_before: dict[str, dict]) -> str:
    """Decide whether a newly saved row becomes the adopted answer."""
    request_id = row["request_id"]
    if request_id not in batch["request_ids"]:
        raise ValueError("Attempt is not a member of this batch")
    member = next(item for item in batch["members"] if item["request_id"] == request_id)
    if row.get("candidate_id") != member["candidate_id"] or row.get("prompt_sha256") != member["prompt_sha256"]:
        return "rejected_identity"
    if request_id in adopted_before:
        return "retained_previous"
    return "adopt"
