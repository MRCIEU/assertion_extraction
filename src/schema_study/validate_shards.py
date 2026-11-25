"""Checksum generation directories that already have a completion record.

A directory without ``COMPLETE`` is left unmarked.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from schema_study.production_runtime import publish_validation

ROOT = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/runs/formal/formal_matched_e0_e7/generation_inputs")
FINISHED = (
    "zero_shot/biored",
    "few_shot/biored",
    "zero_shot/bc8",
    "few_shot/bc8",
    "zero_shot/drugprot",
    "few_shot/drugprot",
)


def _one(directory: str) -> dict:
    result = publish_validation(Path(directory))
    result["directory"] = directory
    print(json.dumps({key: result.get(key) for key in ("directory", "ready", "requests", "expected", "duplicate_requests", "checksum_failures", "reason")}), flush=True)
    return result


def main() -> int:
    directories = [str(path.parent) for name in FINISHED for path in sorted((ROOT / name).rglob("COMPLETE"))]
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_one, directories))
    manifest = {
        "conditions": len(results),
        "ready": all(row.get("ready") for row in results),
        "requests": sum(int(row.get("requests") or 0) for row in results),
        "checksum_failures": sum(int(row.get("checksum_failures") or 0) for row in results),
        "duplicate_requests": sum(int(row.get("duplicate_requests") or 0) for row in results),
        "directories": [
            {key: row.get(key) for key in ("directory", "ready", "requests", "expected", "duplicate_requests", "checksum_failures", "reason")}
            for row in results
        ],
    }
    temporary = ROOT / ".validation_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    temporary.replace(ROOT / "validation_manifest.json")
    return 0 if manifest["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
