"""Parse model text into a unified prediction and roll a saved P4 decision up.

Parsing removes code fences mechanically. It does not call another model.
Rollup maps discrete labels. It does not add probabilities or refit a threshold.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from schema_study.schemas import Schema, resolve_output_label, rollup_ids

UNIFIED_COLUMNS = (
    "run_id",
    "dataset",
    "split",
    "document_id",
    "candidate_id",
    "entity_level",
    "predicate_level",
    "route",
    "model_id",
    "seed",
    "input_id",
    "predicted_labels",
    "output_status",
    "probabilities",
    "source_prediction_id",
    "request_id",
)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE.sub("", stripped).strip()
    return stripped


def parse_prediction_text(
    text: str | None,
    *,
    expected_candidate_id: str,
    schema: Schema,
    level: str,
) -> dict:
    """Parse one model answer.

    Status values:
    - `valid`: empty list or an accepted positive label set
    - `missing`: no usable JSON object
    - `rejected`: the whole answer is discarded
    - `partial_illegal`: DrugProt kept the legal labels and counted illegal ones
    """
    multi = schema.label_cardinality() == "multi"
    if text is None or not str(text).strip():
        return _result("missing", [], [], "empty_answer")
    try:
        payload = json.loads(strip_code_fence(str(text)))
    except json.JSONDecodeError:
        return _result("missing", [], [], "unparseable")
    if not isinstance(payload, dict) or "labels" not in payload or "candidate_id" not in payload:
        return _result("missing", [], [], "missing_fields")
    if payload["candidate_id"] != expected_candidate_id:
        return _result("rejected", [], [], "candidate_id_mismatch")
    labels = payload["labels"]
    if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
        return _result("missing", [], [], "labels_not_string_list")

    legal: list[str] = []
    illegal: list[str] = []
    declared_no_relation = False
    for item in labels:
        stable_id, error = resolve_output_label(schema, level, item)
        if error == "no_relation_string":
            declared_no_relation = True
        elif error == "illegal" or stable_id is None:
            illegal.append(item.strip())
        else:
            legal.append(stable_id)
    legal_unique = list(dict.fromkeys(legal))
    illegal_unique = list(dict.fromkeys(illegal))
    has_positive = bool(legal_unique)

    if declared_no_relation and (has_positive or illegal_unique):
        return _result("rejected", [], illegal_unique, "no_relation_mixed_with_labels")
    if not multi:
        if declared_no_relation or illegal_unique or len(legal_unique) != (1 if has_positive else 0):
            if not legal_unique and not illegal_unique and not declared_no_relation:
                return _result("valid", [], [], None)
            return _result("rejected", [], illegal_unique, "not_unique_legal_label")
        return _result("valid", legal_unique, [], None)
    if declared_no_relation and not has_positive and not illegal_unique:
        return _result("partial_illegal", [], [schema.no_relation_name], "no_relation_string")
    if illegal_unique:
        return _result("partial_illegal", legal_unique, illegal_unique, "illegal_labels_retained")
    return _result("valid", legal_unique, [], None)


def _result(status: str, legal: list[str], illegal: list[str], reason: str | None) -> dict:
    return {
        "output_status": status,
        "predicted_labels": legal,
        "illegal_labels": illegal,
        "reason": reason,
    }


def rollup_predicted_labels(
    schema: Schema,
    predicted_labels: Iterable[str],
    *,
    source_level: str,
    target_level: str,
) -> list[str]:
    """Map a saved discrete prediction onto a coarser level and drop duplicates."""
    mapped = rollup_ids(schema, source_level, set(predicted_labels), target_level)
    order = [label.stable_id for label in schema.level_labels(target_level)]
    return [label_id for label_id in order if label_id in mapped]


def unified_row(**fields) -> dict:
    row = {column: None for column in UNIFIED_COLUMNS}
    row.update(fields)
    missing = [column for column in ("run_id", "candidate_id", "output_status", "predicted_labels") if row.get(column) is None]
    if missing:
        raise ValueError(f"Unified prediction is missing {missing}")
    return row
