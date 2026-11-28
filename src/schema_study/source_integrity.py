"""Checks that a derived JSONL file matches an expected identifier population.

File size and a previously written summary are not used as evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


def inspect_jsonl(path: Path, identifier: str, required: dict | None = None, expected_corpus: str | None = None, allowed_splits: set[str] | None = None, extra=None) -> dict:
    """Parse every record and hash the bytes that were actually read.

    A matching identifier set is not, by itself, a completed file. UTF-8,
    JSON, required fields, types, corpus, and split are counted separately.
    """
    digest = hashlib.sha256()
    count = 0
    errors = 0
    structure_errors = 0
    identifiers = set()
    documents = set()
    splits = Counter()
    duplicates = 0
    missing_identifier = 0
    unterminated = False
    last_ok = False
    with Path(path).open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            if not raw.endswith(b"\n"):
                unterminated = True
            try:
                line = raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                errors += 1
                last_ok = False
                continue
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                last_ok = False
                continue
            if not isinstance(row, dict):
                errors += 1
                structure_errors += 1
                last_ok = False
                continue
            problems = _structure_problems(row, required or {}, expected_corpus, allowed_splits)
            if extra is not None:
                problems.extend(extra(row))
            count += 1
            if problems:
                structure_errors += 1
                last_ok = False
            else:
                last_ok = True
            value = row.get(identifier)
            if not isinstance(value, str) or not value:
                missing_identifier += 1
            else:
                if value in identifiers:
                    duplicates += 1
                identifiers.add(value)
            if row.get("document_id") is not None:
                documents.add(str(row["document_id"]))
            if row.get("split") is not None:
                splits[str(row["split"])] += 1
    population_ok = None
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "n_records": count,
        "parse_errors": errors,
        "structure_errors": structure_errors,
        "missing_identifier": missing_identifier,
        "unique_identifiers": len(identifiers),
        "duplicate_identifiers": duplicates,
        "final_record_parses": last_ok and errors == 0 and structure_errors == 0,
        "final_line_terminated": not unterminated,
        "splits": dict(splits),
        "identifiers": identifiers,
        "document_ids": documents,
        "population_agreement": population_ok,
    }


def _structure_problems(row: dict, required: dict, expected_corpus: str | None, allowed_splits: set[str] | None) -> list[str]:
    problems = []
    for field, kind in required.items():
        if field not in row:
            problems.append(f"missing {field}")
            continue
        if not isinstance(row[field], kind):
            problems.append(f"type {field}")
    if expected_corpus is not None and row.get("corpus") != expected_corpus:
        problems.append("corpus")
    if allowed_splits is not None and str(row.get("split")) not in allowed_splits:
        problems.append("split")
    return problems


def collection_complete(inspection: dict, population: dict) -> dict:
    """Identifier agreement does not override a malformed or incomplete file."""
    conditions = {
        "no_parse_errors": inspection["parse_errors"] == 0,
        "no_structure_errors": inspection["structure_errors"] == 0,
        "no_duplicate_identifiers": inspection["duplicate_identifiers"] == 0,
        "no_missing_identifiers": inspection["missing_identifier"] == 0,
        "final_record_parses": inspection["final_record_parses"],
        "final_line_terminated": inspection["final_line_terminated"],
        "population_complete": bool(population.get("complete")),
        "no_unexpected_identifiers": population.get("n_unexpected", 1) == 0,
        "no_missing_population_identifiers": population.get("n_missing", 1) == 0,
        "checksum_recorded": bool(inspection.get("sha256")),
    }
    return {
        "complete": all(conditions.values()),
        "conditions": conditions,
        "sha256": inspection.get("sha256"),
        "n_records": inspection.get("n_records"),
        "population": population,
    }


def population_difference(observed: set, expected: set) -> dict:
    return {
        "n_expected": len(expected),
        "n_observed": len(observed),
        "n_missing": len(expected - observed),
        "n_unexpected": len(observed - expected),
        "complete": observed == expected,
    }
