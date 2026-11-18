"""Three-level description of the saved round-4 development answers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.dataset as ds

from schema_study.evaluation import candidate_label_counts, f1_from_counts, summarize_counts
from schema_study.formal_eval import gold_groups
from schema_study.predictions import rollup_predicted_labels
from schema_study.schemas import load_schema

CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
ROUND4 = STORAGE / "studies/schema_granularity_v1/runs/backend_validation/round4_20261115"
OUT = STORAGE / "studies/schema_granularity_v1/runs/task_suitability/round6_20261115/existing"
RUNS = {
    "qwen8b_batch1": ROUND4 / "qwen8b/transformers/batch1_repeat0/attempts.jsonl",
    "qwen8b_batch16": ROUND4 / "qwen8b/transformers/batch16_repeat0/attempts.jsonl",
    "qwen32b_batch16": ROUND4 / "qwen32b/transformers/batch16_repeat0/attempts.jsonl",
    "qwen32b_batch32": ROUND4 / "qwen32b/transformers/batch32_repeat0/attempts.jsonl",
}


def _gold(dataset: str, wanted: set[str]) -> dict[str, dict]:
    version = "biored_candidates_v2" if dataset == "biored" else "drugprot_candidates_v1"
    table = ds.dataset(STORAGE / "data/prepared" / dataset / version / "candidates.parquet", format="parquet")
    found = {}
    for batch in table.to_batches(columns=["candidate_id", "document_id", "split", "gold_labels"]):
        for row in batch.to_pylist():
            if row["candidate_id"] in wanted:
                found[row["candidate_id"]] = {
                    "document_id": row["document_id"],
                    "split": row["split"],
                    "gold_labels": list(row["gold_labels"] or []),
                }
    return found


def _timing(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("included_in_timing"):
            rows[row["request_id"]] = row
    return rows


def _layer(answer: dict, gold_ids: list[str], schema, level: str) -> dict:
    status = answer["output_status"]
    predicted = list(answer["predicted_labels"])
    counted = candidate_label_counts(gold_ids, predicted, status, answer.get("illegal_labels"))
    legal_nonempty = status in {"valid", "partial_illegal"} and bool(predicted)
    return {
        "status": status,
        "predicted": predicted,
        "gold": gold_ids,
        "gold_positive": bool(gold_ids),
        "valid_empty": status == "valid" and not predicted,
        "valid_nonempty": status == "valid" and bool(predicted),
        "legal_nonempty": legal_nonempty,
        "tp": counted["tp"],
        "fp": counted["fp"],
        "fn": counted["fn"],
        "exact": counted["exact"],
    }


def main() -> None:
    requests = [json.loads(line) for line in (ROUND4 / "samples/requests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["request_id"]: row for row in requests}
    candidates = {}
    for row in requests:
        candidates.setdefault(row["candidate_id"], {"dataset": row["dataset"], "document_id": row["document_id"]})
    wanted = {name: {row["candidate_id"] for row in requests if row["dataset"] == name} for name in ("biored", "drugprot")}
    gold = {name: _gold(name, ids) for name, ids in wanted.items()}
    schemas = {name: load_schema(CODE / "configs/schemas" / f"{name}.yaml") for name in ("biored", "drugprot")}
    positive_candidates = []
    for candidate_id, meta in candidates.items():
        labels = gold[meta["dataset"]][candidate_id]["gold_labels"]
        if labels:
            positive_candidates.append(candidate_id)
    runs = {name: _timing(path) for name, path in RUNS.items()}
    cells = []
    for run_name, answers in runs.items():
        for dataset in ("biored", "drugprot"):
            schema = schemas[dataset]
            for level in ("P2", "P4"):
                subset = [row for row in requests if row["dataset"] == dataset and row["predicate_level"] == level]
                detailed = []
                for row in subset:
                    answer = answers[row["request_id"]]
                    gold_ids = gold_groups(schema, gold[dataset][row["candidate_id"]]["gold_labels"], level)
                    detailed.append({**_layer(answer, gold_ids, schema, level), "candidate_id": row["candidate_id"], "document_id": row["document_id"], "illegal_labels": answer.get("illegal_labels") or [], "truncated": bool(answer["truncated"])})
                labels = [label.stable_id for label in schema.level_labels(level)]
                summary = summarize_counts(
                    [{"gold": row["gold"], "predicted": row["predicted"], "status": row["status"], "illegal_labels": row["illegal_labels"]} for row in detailed],
                    labels,
                )
                negatives = [row for row in detailed if not row["gold_positive"]]
                positives = [row for row in detailed if row["gold_positive"]]
                positive_summary = summarize_counts(
                    [{"gold": row["gold"], "predicted": row["predicted"], "status": row["status"], "illegal_labels": row["illegal_labels"]} for row in positives],
                    labels,
                )
                confusion = Counter()
                for row in positives if dataset == "biored" else []:
                    gold_name = row["gold"][0] if len(row["gold"]) == 1 else ("empty" if not row["gold"] else "multi")
                    if row["status"] != "valid":
                        pred_name = row["status"]
                    elif not row["predicted"]:
                        pred_name = "empty"
                    elif len(row["predicted"]) == 1:
                        pred_name = row["predicted"][0]
                    else:
                        pred_name = "multi"
                    confusion[f"{gold_name}->{pred_name}"] += 1
                cells.append({
                    "run": run_name,
                    "dataset": dataset,
                    "predicate_level": level,
                    "requests": len(detailed),
                    "candidates": len({row["candidate_id"] for row in detailed}),
                    "documents": len({row["document_id"] for row in detailed}),
                    "gold_positive": len(positives),
                    "gold_negative": len(negatives),
                    "layer1": {
                        "precision": summary["precision"],
                        "recall": summary["recall"],
                        "f1": summary["f1"],
                        "macro_f1": summary["macro_f1"],
                        "tp": summary["tp"],
                        "fp": summary["fp"],
                        "fn": summary["fn"],
                    },
                    "status": dict(Counter(row["status"] for row in detailed)),
                    "valid_empty": sum(row["valid_empty"] for row in detailed),
                    "truncated": sum(row["truncated"] for row in detailed),
                    "existence": {
                        "negative_denominator": len(negatives),
                        "negative_valid_nonempty": sum(row["valid_nonempty"] for row in negatives),
                        "negative_valid_empty": sum(row["valid_empty"] for row in negatives),
                        "positive_denominator": len(positives),
                        "positive_legal_nonempty": sum(row["legal_nonempty"] for row in positives),
                        "rejected": sum(row["status"] == "rejected" for row in detailed),
                        "partial_illegal": sum(row["status"] == "partial_illegal" for row in detailed),
                        "missing": sum(row["status"] == "missing" for row in detailed),
                    },
                    "layer3_positive_only": {
                        "candidates": len(positives),
                        "precision": positive_summary["precision"],
                        "recall": positive_summary["recall"],
                        "f1": positive_summary["f1"],
                        "tp": positive_summary["tp"],
                        "fp": positive_summary["fp"],
                        "fn": positive_summary["fn"],
                        "per_class": [row for row in positive_summary["per_class"] if row["support"] or row["fp"]],
                        "empty_on_positive": sum(row["valid_empty"] for row in positives),
                        "failed_on_positive": sum(row["status"] in {"rejected", "missing"} for row in positives),
                        "partial_on_positive": sum(row["status"] == "partial_illegal" for row in positives),
                        "confusion": dict(confusion),
                    },
                })
            direct = [row for row in requests if row["dataset"] == dataset and row["predicate_level"] == "P2"]
            rolled_rows = []
            direct_rows = []
            for row in direct:
                source = next(item for item in requests if item["candidate_id"] == row["candidate_id"] and item["predicate_level"] == "P4")
                p4 = answers[source["request_id"]]
                gold_ids = gold_groups(schema, gold[dataset][row["candidate_id"]]["gold_labels"], "P2")
                rolled_labels = rollup_predicted_labels(schema, p4["predicted_labels"], source_level="P4", target_level="P2") if p4["output_status"] in {"valid", "partial_illegal"} else []
                rolled_status = p4["output_status"]
                direct_answer = answers[row["request_id"]]
                rolled_rows.append(_layer({**p4, "predicted_labels": rolled_labels, "output_status": rolled_status}, gold_ids, schema, "P2"))
                direct_rows.append(_layer(direct_answer, gold_ids, schema, "P2"))
            same_empty = sum(left["valid_empty"] == right["valid_empty"] and left["predicted"] == right["predicted"] for left, right in zip(direct_rows, rolled_rows))
            cells.append({
                "run": run_name,
                "dataset": dataset,
                "predicate_level": "P2_vs_P4_rollup",
                "paired_candidates": len(direct),
                "same_label_set": same_empty,
                "direct_valid_empty": sum(row["valid_empty"] for row in direct_rows),
                "rollup_valid_empty": sum(row["valid_empty"] for row in rolled_rows),
                "direct_f1": f1_from_counts(sum(row["tp"] for row in direct_rows), sum(row["fp"] for row in direct_rows), sum(row["fn"] for row in direct_rows)),
                "rollup_f1": f1_from_counts(sum(row["tp"] for row in rolled_rows), sum(row["fp"] for row in rolled_rows), sum(row["fn"] for row in rolled_rows)),
            })
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "unique_candidates": len(candidates),
        "gold_positive_candidates": len(positive_candidates),
        "gold_negative_candidates": len(candidates) - len(positive_candidates),
        "positive_candidate_ids": sorted(positive_candidates),
        "cells": cells,
        "included_in_formal_analysis": False,
        "note": "Exploratory description of the technical sample. Not a preregistered confirmation.",
    }
    (OUT / "three_level.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidates": len(candidates), "positive": len(positive_candidates)}, indent=2))


if __name__ == "__main__":
    main()
