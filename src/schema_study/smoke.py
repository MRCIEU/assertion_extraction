"""Fixed smoke samples and the only weight-loading permit.

Importing this module does not import PyTorch. Formal experiment configs cannot
pass the smoke checks.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

SMOKE_MODELS = {
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext": "e1354b7a3a09615f6aba48dfad4b7a613eef7062",
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
}
SMOKE_TRAIN_SEED = 13
SMOKE_SAMPLE_SEED = 100
MAX_BIOMEDBERT_UPDATES = 32
MAX_QWEN_GENERATIONS = 50


class SmokeConfigError(ValueError):
    """The requested run is outside the authorized smoke boundary."""


def assert_smoke_config(config: dict) -> None:
    """Allow one limited smoke run. This does not change the formal permit."""
    if config.get("phase") != "smoke":
        raise SmokeConfigError("Only phase=smoke can load weights in this round.")
    if config.get("study_phase") not in (None, "smoke"):
        raise SmokeConfigError("study_phase must be smoke.")
    if config.get("included_in_formal_analysis") not in (None, False):
        raise SmokeConfigError("Smoke output cannot enter the formal analysis.")
    if config.get("formal_execution_authorized") is True:
        raise SmokeConfigError("A smoke config cannot authorize the formal matrix.")
    model_id = config.get("model_id")
    if SMOKE_MODELS.get(model_id) != config.get("revision"):
        raise SmokeConfigError("The model or revision is outside the smoke allow-list.")
    splits = set(config.get("splits") or [])
    if not splits or splits - {"train", "dev"}:
        raise SmokeConfigError("Smoke data must come from train and dev only.")
    output = str(config.get("output_dir") or "")
    if "/runs/smoke/" not in output.replace("\\", "/"):
        raise SmokeConfigError("Smoke output must stay under runs/smoke.")
    updates = int(config.get("max_updates") or 0)
    requests = int(config.get("max_generation_requests") or 0)
    if model_id.startswith("microsoft/") and updates > MAX_BIOMEDBERT_UPDATES:
        raise SmokeConfigError("BiomedBERT smoke updates exceed 32.")
    if model_id.startswith("Qwen/") and requests > MAX_QWEN_GENERATIONS:
        raise SmokeConfigError("Qwen smoke requests exceed 50.")
    from schema_study.models.schedule import ROUND_MODEL_EXECUTION_PERMITTED

    if ROUND_MODEL_EXECUTION_PERMITTED:
        raise SmokeConfigError("The global formal execution permit must stay off.")


def _covers(rows: list[dict], predicate) -> bool:
    return any(predicate(row) for row in rows)


def _take_from_documents(by_document: dict[str, list[dict]], order: list[str], target: int, need_multi: bool) -> tuple[list[str], list[dict]]:
    """Add documents in the fixed order until the pool can fill the target."""
    selected_documents = []
    pool = []
    for document_id in order:
        selected_documents.append(document_id)
        pool.extend(by_document.get(document_id, []))
        enough = len(pool) >= target and _covers(pool, lambda row: row["gold_labels"]) and _covers(pool, lambda row: not row["gold_labels"])
        multi_ready = (not need_multi) or _covers(pool, lambda row: row["gold_status"] == "multi") or not any(
            row["gold_status"] == "multi" for rows in by_document.values() for row in rows
        )
        if enough and multi_ready:
            break
    return selected_documents, pool


def _force_one(chosen: list[dict], pool: list[dict], predicate, reason: str, notes: list[dict]) -> None:
    if _covers(chosen, predicate):
        notes.append({"reason": reason, "forced": False})
        return
    chosen_ids = {row["candidate_id"] for row in chosen}
    donor = next(row for row in pool if predicate(row) and row["candidate_id"] not in chosen_ids)
    protected = set()
    for guard in (lambda row: bool(row["gold_labels"]), lambda row: not row["gold_labels"]):
        match = next((row["candidate_id"] for row in chosen if guard(row)), None)
        if match:
            protected.add(match)
    replace_at = next(index for index, row in enumerate(chosen) if row["candidate_id"] not in protected and not predicate(row))
    notes.append({"reason": reason, "forced": True, "replaced_candidate_id": chosen[replace_at]["candidate_id"], "added_candidate_id": donor["candidate_id"]})
    chosen[replace_at] = donor


def select_split(rows: list[dict], *, target: int, sample_seed: int, need_multi: bool) -> dict:
    """Shuffle documents with the sampling seed, then take a fixed candidate subset."""
    by_document = defaultdict(list)
    for row in rows:
        by_document[row["document_id"]].append(row)
    rng = random.Random(sample_seed)
    order = sorted(by_document)
    rng.shuffle(order)
    documents, pool = _take_from_documents(by_document, order, target, need_multi)
    rng.shuffle(pool)
    chosen = pool[:target]
    notes = []
    _force_one(chosen, pool, lambda row: bool(row["gold_labels"]), "positive", notes)
    _force_one(chosen, pool, lambda row: not row["gold_labels"], "negative", notes)
    if need_multi and any(row["gold_status"] == "multi" for row in pool):
        _force_one(chosen, pool, lambda row: row["gold_status"] == "multi", "multi_label_technical_coverage", notes)
    if len(chosen) < target:
        raise SmokeConfigError(f"Smoke pool has {len(chosen)} candidates, target is {target}.")
    if len({row["candidate_id"] for row in chosen}) != len(chosen):
        raise SmokeConfigError("Smoke candidate ids must be unique.")
    return {"documents": documents, "candidates": chosen, "coverage_notes": notes, "pool_size": len(pool)}
