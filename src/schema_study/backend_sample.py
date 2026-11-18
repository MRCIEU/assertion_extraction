"""Freeze the round-4 technical generation sample before any model call."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from schema_study.schemas import load_schema

CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
OUT = STORAGE / "studies/schema_granularity_v1/runs/backend_validation/round4_20261115/samples"
PROMPT = (CODE / "prompts/relation_extraction.txt").read_text(encoding="utf-8")
SOURCES = {
    "biored": {
        "path": STORAGE / "data/inputs/biored/target_anchored_v1__biomedbert_e1354b7a_proposed/contexts.jsonl",
        "view": "E2",
        "entity_level": "E2",
    },
    "drugprot": {
        "path": STORAGE / "data/inputs/drugprot/target_anchored_v1__biomedbert_e1354b7a/contexts.jsonl",
        "view": "native",
        "entity_level": "native",
    },
}


def _dev_rows(path: Path, view: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if '"split": "dev"' not in line:
                continue
            row = json.loads(line)
            if row.get("split") != "dev":
                continue
            text = row["views"][view]
            rows.append({
                "candidate_id": row["candidate_id"],
                "document_id": row["document_id"],
                "length": int(row["encoded_length_by_view"][view]),
                "text": text,
            })
    return rows


def _quartiles(rows: list[dict]) -> list[list[dict]]:
    ordered = sorted(rows, key=lambda row: (row["length"], row["candidate_id"]))
    size = len(ordered) // 4
    groups = [ordered[index * size:(index + 1) * size] for index in range(3)]
    groups.append(ordered[3 * size:])
    return groups


def _draw(groups: list[list[dict]], rng: random.Random, target: int = 16) -> tuple[list[dict], list[dict]]:
    shuffled = []
    notes = []
    for index, group in enumerate(groups):
        pool = group[:]
        rng.shuffle(pool)
        shuffled.append(pool)
        notes.append({"quartile": index + 1, "available": len(group), "assigned": 0})
    chosen = []
    for index, pool in enumerate(shuffled):
        take = pool[:target]
        for row in take:
            row["quartile"] = index + 1
        notes[index]["assigned"] = len(take)
        chosen.extend(take)
        shuffled[index] = pool[target:]
    deficit = 64 - len(chosen)
    cursor = 0
    while deficit > 0:
        pool = shuffled[cursor % 4]
        if pool:
            extra = pool.pop(0)
            extra["quartile"] = (cursor % 4) + 1
            chosen.append(extra)
            notes[cursor % 4]["assigned"] += 1
            deficit -= 1
        cursor += 1
        if cursor > 8 and all(not pool for pool in shuffled):
            raise RuntimeError("A length quartile cannot fill the technical sample.")
    return chosen, notes


def main() -> None:
    rng = random.Random(100)
    prompt_rows = []
    summary = {"seed": 100, "forbidden_splits": [], "quartiles": {}, "documents": {}}
    for dataset, source in SOURCES.items():
        rows = _dev_rows(source["path"], source["view"])
        chosen, notes = _draw(_quartiles(rows), rng)
        if len(chosen) != 64 or len({row["candidate_id"] for row in chosen}) != 64:
            raise RuntimeError(f"{dataset} sample is not 64 unique candidates.")
        schema = load_schema(CODE / "configs/schemas" / f"{dataset}.yaml")
        summary["quartiles"][dataset] = notes
        summary["documents"][dataset] = {
            "unique_documents": len({row["document_id"] for row in chosen}),
            "max_candidates_in_one_document": max(Counter(row["document_id"] for row in chosen).values()),
            "length_min": min(row["length"] for row in chosen),
            "length_max": max(row["length"] for row in chosen),
        }
        for row in chosen:
            for level in ("P2", "P4"):
                block = "\n".join(f"- {name}" for name in schema.positive_display_names(level))
                prompt = (
                    PROMPT.replace("{{candidate_id}}", row["candidate_id"])
                    .replace("{{dataset}}", dataset)
                    .replace("{{label_block}}", block)
                    .replace("{{text}}", row["text"])
                )
                prompt_rows.append({
                    "request_id": f"{dataset}|{level}|{row['candidate_id']}",
                    "candidate_id": row["candidate_id"],
                    "document_id": row["document_id"],
                    "dataset": dataset,
                    "split": "dev",
                    "entity_level": source["entity_level"],
                    "predicate_level": level,
                    "context_length": row["length"],
                    "length_quartile": row["quartile"],
                    "prompt": prompt,
                    "included_in_formal_analysis": False,
                })
    if len(prompt_rows) != 256:
        raise RuntimeError(f"Expected 256 requests, found {len(prompt_rows)}.")
    subset = []
    for dataset in ("biored", "drugprot"):
        per_quartile: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}
        for row in prompt_rows:
            if row["dataset"] != dataset or row["predicate_level"] != "P4":
                continue
            bucket = per_quartile[row["length_quartile"]]
            if len(bucket) < 2:
                bucket.append(row["candidate_id"])
        ids = [candidate_id for quartile in (1, 2, 3, 4) for candidate_id in per_quartile[quartile]]
        for candidate_id in ids:
            subset.extend(row["request_id"] for row in prompt_rows if row["candidate_id"] == candidate_id)
    summary["qwen32b_correspondence_ids"] = subset
    summary["request_count"] = 256
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for row in prompt_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "sampling_record.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"requests": len(prompt_rows), "subset": len(subset), "documents": summary["documents"]}, indent=2))


if __name__ == "__main__":
    main()
