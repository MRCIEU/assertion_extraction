"""Checks that a derived JSONL file matches an expected identifier population.

File size and a previously written summary are not used as evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


def inspect_jsonl(path: Path, identifier: str) -> dict:
    """Parse every record and hash the bytes that were actually read."""
    digest = hashlib.sha256()
    count = 0
    errors = 0
    identifiers = set()
    documents = set()
    splits = Counter()
    duplicates = 0
    unterminated = False
    last_ok = False
    with Path(path).open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            if not raw.endswith(b"\n"):
                unterminated = True
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                last_ok = False
                continue
            count += 1
            last_ok = True
            value = row.get(identifier)
            if value in identifiers:
                duplicates += 1
            if value is not None:
                identifiers.add(value)
            if row.get("document_id") is not None:
                documents.add(str(row["document_id"]))
            if row.get("split") is not None:
                splits[str(row["split"])] += 1
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "n_records": count,
        "parse_errors": errors,
        "unique_identifiers": len(identifiers),
        "duplicate_identifiers": duplicates,
        "final_record_parses": last_ok and errors == 0,
        "final_line_terminated": not unterminated,
        "splits": dict(splits),
        "identifiers": identifiers,
        "document_ids": documents,
    }


def population_difference(observed: set, expected: set) -> dict:
    return {
        "n_expected": len(expected),
        "n_observed": len(observed),
        "n_missing": len(expected - observed),
        "n_unexpected": len(observed - expected),
        "complete": observed == expected,
    }
