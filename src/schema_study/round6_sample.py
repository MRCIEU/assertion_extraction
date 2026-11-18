"""Freeze the round-6 development check before any new model call.

Documents used by earlier model checks are removed first. The draw uses seed
101 and does not look at previous model answers.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.dataset as ds

from schema_study.schemas import load_schema

CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
OUT = STORAGE / "studies/schema_granularity_v1/runs/task_suitability/round6_20261115"
PROMPT_V0 = (CODE / "prompts/relation_extraction.txt").read_text(encoding="utf-8")
PROMPT_V1 = (CODE / "prompts/relation_extraction_v1.txt").read_text(encoding="utf-8")
CARDINALITY = {
    "biored": "This BioRED candidate allows at most one positive label.",
    "drugprot": "This DrugProt candidate may include more than one positive label.",
}
SOURCES = {
    "biored": {
        "path": STORAGE / "data/inputs/biored/target_anchored_v1__biomedbert_e1354b7a_proposed/contexts.jsonl",
        "view": "E2",
        "entity_level": "E2",
        "candidates": STORAGE / "data/prepared/biored/biored_candidates_v2/candidates.parquet",
    },
    "drugprot": {
        "path": STORAGE / "data/inputs/drugprot/target_anchored_v1__biomedbert_e1354b7a/contexts.jsonl",
        "view": "native",
        "entity_level": "native",
        "candidates": STORAGE / "data/prepared/drugprot/drugprot_candidates_v1/candidates.parquet",
    },
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded_documents() -> dict[str, list[str]]:
    excluded = {"biored": set(), "drugprot": set()}
    round4 = STORAGE / "studies/schema_granularity_v1/runs/backend_validation/round4_20261115/samples/requests.jsonl"
    for line in round4.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            excluded[row["dataset"]].add(str(row["document_id"]))
    smoke = STORAGE / "studies/schema_granularity_v1/runs/smoke/round3_20261115/samples/smoke_documents.tsv"
    for line in smoke.read_text(encoding="utf-8").splitlines()[1:]:
        corpus, split, document_id, _role = line.split("\t")
        if split == "dev" and corpus in excluded:
            excluded[corpus].add(document_id)
    return {name: sorted(values) for name, values in excluded.items()}


def _dev_candidates(dataset: str) -> list[dict]:
    table = ds.dataset(SOURCES[dataset]["candidates"], format="parquet")
    rows = []
    columns = ["candidate_id", "document_id", "split", "gold_labels"]
    for batch in table.to_batches(columns=columns):
        for row in batch.to_pylist():
            if row["split"] != "dev":
                continue
            labels = list(row["gold_labels"] or [])
            rows.append({
                "candidate_id": row["candidate_id"],
                "document_id": str(row["document_id"]),
                "gold_labels": labels,
                "gold_positive": bool(labels),
            })
    return rows


def _draw(rows: list[dict], excluded: set[str], rng: random.Random) -> dict:
    eligible = [row for row in rows if row["document_id"] not in excluded]
    documents = {row["document_id"] for row in eligible}
    positive = [row for row in eligible if row["gold_positive"]]
    negative = [row for row in eligible if not row["gold_positive"]]
    rng.shuffle(positive)
    rng.shuffle(negative)
    chosen_positive = positive[:64]
    chosen_negative = negative[:64]
    chosen = chosen_positive + chosen_negative
    counts = Counter(row["document_id"] for row in chosen)
    label_counts = Counter(label for row in chosen_positive for label in row["gold_labels"])
    return {
        "available_dev_candidates": len(rows),
        "available_dev_documents": len({row["document_id"] for row in rows}),
        "excluded_documents": len(excluded),
        "eligible_candidates": len(eligible),
        "eligible_documents": len(documents),
        "eligible_positive": len(positive),
        "eligible_negative": len(negative),
        "selected_positive": len(chosen_positive),
        "selected_negative": len(chosen_negative),
        "selected_documents": len(counts),
        "max_candidates_in_one_document": max(counts.values()) if counts else 0,
        "gold_label_counts": dict(sorted(label_counts.items())),
        "multilabel_gold": sum(len(row["gold_labels"]) > 1 for row in chosen),
        "chosen": chosen,
    }


def _texts(path: Path, view: str, wanted: set[str]) -> dict[str, str]:
    found = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if '"split": "dev"' not in line:
                continue
            row = json.loads(line)
            if row.get("split") != "dev" or row["candidate_id"] not in wanted:
                continue
            found[row["candidate_id"]] = row["views"][view]
            if len(found) == len(wanted):
                break
    return found


def _fill(template: str, dataset: str, row: dict, schema) -> str:
    return (
        template.replace("{{candidate_id}}", row["candidate_id"])
        .replace("{{dataset}}", dataset)
        .replace("{{cardinality_rule}}", CARDINALITY[dataset])
        .replace("{{label_block}}", "\n".join(f"- {name}" for name in schema.positive_display_names(row["predicate_level"])))
        .replace("{{text}}", row["text"])
    )


def main() -> None:
    if (OUT / "plan/development_check_plan.json").exists():
        raise SystemExit("The development check plan already exists and will not be rewritten.")
    excluded = _excluded_documents()
    rng = random.Random(101)
    draws = {}
    requests = []
    for dataset in ("biored", "drugprot"):
        draw = _draw(_dev_candidates(dataset), set(excluded[dataset]), rng)
        if draw["selected_positive"] + draw["selected_negative"] == 0:
            raise SystemExit(f"{dataset} has no document-isolated development candidates.")
        texts = _texts(SOURCES[dataset]["path"], SOURCES[dataset]["view"], {row["candidate_id"] for row in draw["chosen"]})
        missing = [row["candidate_id"] for row in draw["chosen"] if row["candidate_id"] not in texts]
        if missing:
            raise SystemExit(f"{dataset} is missing context text for {len(missing)} selected candidates.")
        schema = load_schema(CODE / "configs/schemas" / f"{dataset}.yaml")
        for row in draw["chosen"]:
            row["text"] = texts[row["candidate_id"]]
            for level in ("P2", "P4"):
                item = {**row, "predicate_level": level}
                requests.append({
                    "request_id": f"{dataset}|{level}|{row['candidate_id']}",
                    "candidate_id": row["candidate_id"],
                    "document_id": row["document_id"],
                    "dataset": dataset,
                    "split": "dev",
                    "entity_level": SOURCES[dataset]["entity_level"],
                    "predicate_level": level,
                    "gold_positive": row["gold_positive"],
                    "gold_labels": row["gold_labels"],
                    "prompt_v0": _fill(PROMPT_V0, dataset, item, schema),
                    "prompt_v1": _fill(PROMPT_V1, dataset, item, schema),
                    "included_in_formal_analysis": False,
                })
        draw.pop("chosen")
        draws[dataset] = draw
    plan = {
        "study_phase": "development_check",
        "independent_confirmation": False,
        "seed": 101,
        "excluded_documents": excluded,
        "draws": draws,
        "request_count": len(requests),
        "order": "BioRED then DrugProt; within a corpus, shuffled positives then shuffled negatives; P2 then P4 for each candidate",
        "prompt_v0_sha256": hashlib.sha256(PROMPT_V0.encode("utf-8")).hexdigest(),
        "prompt_v1_sha256": hashlib.sha256(PROMPT_V1.encode("utf-8")).hexdigest(),
        "candidate_sources": {name: _sha(source["candidates"]) for name, source in SOURCES.items()},
        "context_sources": {name: str(source["path"]) for name, source in SOURCES.items()},
        "versions": ["v0", "v1"],
        "models": ["qwen8b", "qwen32b"],
        "batch_size": 16,
        "execution_authorized": False,
    }
    (OUT / "plan").mkdir(parents=True, exist_ok=True)
    (OUT / "samples").mkdir(parents=True, exist_ok=True)
    with (OUT / "samples/requests.jsonl").open("w", encoding="utf-8") as handle:
        for row in requests:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    plan["request_ids"] = [row["request_id"] for row in requests]
    plan["batch_manifest"] = "written by round6_tokenize after the rendered strings are frozen and before generation"
    (OUT / "plan/development_check_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"requests": len(requests), "draws": draws}, indent=2))


if __name__ == "__main__":
    main()
