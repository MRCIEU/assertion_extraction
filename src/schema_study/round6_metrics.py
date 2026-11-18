"""Three-level metrics for the round-6 development check.

Gold comes from the candidate tables. Prediction-side gold is not read.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.dataset as ds

from schema_study.batch_plan import adopted_answers
from schema_study.evaluation import f1_from_counts, summarize_counts
from schema_study.formal_eval import gold_groups
from schema_study.generation_ledger import read_jsonl
from schema_study.predictions import rollup_predicted_labels
from schema_study.schemas import load_schema

CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
ROOT = STORAGE / "studies/schema_granularity_v1/runs/task_suitability/round6_20261115"


def _gold() -> dict[str, dict]:
    found = {}
    for dataset, version in (("biored", "biored_candidates_v2"), ("drugprot", "drugprot_candidates_v1")):
        table = ds.dataset(STORAGE / "data/prepared" / dataset / version / "candidates.parquet", format="parquet")
        for batch in table.to_batches(columns=["candidate_id", "document_id", "split", "gold_labels"]):
            for row in batch.to_pylist():
                if row["split"] == "dev":
                    found[row["candidate_id"]] = {
                        "document_id": str(row["document_id"]),
                        "gold_labels": list(row["gold_labels"] or []),
                        "dataset": dataset,
                    }
    return found


def _cell(rows: list[dict], schema, level: str) -> dict:
    labels = [label.stable_id for label in schema.level_labels(level)]
    summary = summarize_counts(rows, labels)
    negatives = [row for row in rows if not row["gold"]]
    positives = [row for row in rows if row["gold"]]
    positive_summary = summarize_counts(positives, labels)
    confusion = Counter()
    for row in positives:
        gold_name = row["gold"][0] if len(row["gold"]) == 1 else "multi"
        if row["status"] != "valid":
            pred_name = row["status"]
        elif not row["predicted"]:
            pred_name = "empty"
        elif len(row["predicted"]) == 1:
            pred_name = row["predicted"][0]
        else:
            pred_name = "multi:" + ",".join(row["predicted"])
        confusion[f"{gold_name}->{pred_name}"] += 1
    documents = Counter(row["document_id"] for row in negatives if row["status"] == "valid" and row["predicted"])
    return {
        "requests": len(rows),
        "candidates": len({row["candidate_id"] for row in rows}),
        "documents": len({row["document_id"] for row in rows}),
        "gold_positive": len(positives),
        "gold_negative": len(negatives),
        "layer1": {key: summary[key] for key in ("precision", "recall", "f1", "macro_f1", "tp", "fp", "fn")},
        "status": dict(Counter(row["status"] for row in rows)),
        "valid_empty": sum(row["status"] == "valid" and not row["predicted"] for row in rows),
        "truncated": sum(row["truncated"] for row in rows),
        "existence": {
            "negative_denominator": len(negatives),
            "negative_valid_nonempty": sum(row["status"] == "valid" and bool(row["predicted"]) for row in negatives),
            "negative_valid_empty": sum(row["status"] == "valid" and not row["predicted"] for row in negatives),
            "positive_denominator": len(positives),
            "positive_legal_nonempty": sum(row["status"] in {"valid", "partial_illegal"} and bool(row["predicted"]) for row in positives),
            "positive_valid_empty": sum(row["status"] == "valid" and not row["predicted"] for row in positives),
            "rejected": sum(row["status"] == "rejected" for row in rows),
            "partial_illegal": sum(row["status"] == "partial_illegal" for row in rows),
            "missing": sum(row["status"] == "missing" for row in rows),
        },
        "layer3_positive_only": {
            "precision": positive_summary["precision"],
            "recall": positive_summary["recall"],
            "f1": positive_summary["f1"],
            "tp": positive_summary["tp"],
            "fp": positive_summary["fp"],
            "fn": positive_summary["fn"],
            "per_class": [row for row in positive_summary["per_class"] if row["support"] or row["fp"]],
            "empty_on_positive": sum(row["status"] == "valid" and not row["predicted"] for row in positives),
            "failed_on_positive": sum(row["status"] in {"rejected", "missing"} for row in positives),
            "confusion": dict(confusion),
        },
        "negative_nonempty_top_documents": documents.most_common(5),
    }


def main() -> None:
    gold = _gold()
    schemas = {name: load_schema(CODE / "configs/schemas" / f"{name}.yaml") for name in ("biored", "drugprot")}
    requests = read_jsonl(ROOT / "samples/requests.jsonl")
    cells = []
    paired = []
    saved = {}
    for version in ("v0", "v1"):
        for model_name in ("qwen8b", "qwen32b"):
            attempts = read_jsonl(ROOT / "generation" / version / model_name / "attempts.jsonl")
            adopted = adopted_answers(attempts)
            saved[(version, model_name)] = adopted
            by_request = {}
            for request in requests:
                answer = adopted.get(request["request_id"])
                table = gold[request["candidate_id"]]
                if table["document_id"] != request["document_id"]:
                    raise SystemExit(f"Document mismatch for {request['candidate_id']}")
                if answer is None:
                    status, predicted, illegal, truncated = "missing", [], [], False
                else:
                    status = answer["output_status"]
                    predicted = list(answer["predicted_labels"])
                    illegal = list(answer["illegal_labels"])
                    truncated = bool(answer["truncated"])
                by_request[request["request_id"]] = {
                    "candidate_id": request["candidate_id"],
                    "document_id": table["document_id"],
                    "dataset": request["dataset"],
                    "level": request["predicate_level"],
                    "gold": gold_groups(schemas[request["dataset"]], table["gold_labels"], request["predicate_level"]),
                    "predicted": predicted if status in {"valid", "partial_illegal"} else [],
                    "status": status,
                    "illegal_labels": illegal,
                    "truncated": truncated,
                    "gold_fine": table["gold_labels"],
                }
            for dataset in ("biored", "drugprot"):
                schema = schemas[dataset]
                for level in ("P2", "P4"):
                    subset = [row for row in by_request.values() if row["dataset"] == dataset and row["level"] == level]
                    cells.append({"version": version, "model": model_name, "dataset": dataset, "predicate_level": level, **_cell(subset, schema, level)})
                direct = [row for row in by_request.values() if row["dataset"] == dataset and row["level"] == "P2"]
                rolled = []
                for row in direct:
                    source = by_request[f"{dataset}|P4|{row['candidate_id']}"]
                    labels = rollup_predicted_labels(schema, source["predicted"], source_level="P4", target_level="P2") if source["status"] in {"valid", "partial_illegal"} else []
                    rolled.append({**source, "predicted": labels, "gold": row["gold"], "level": "P2"})
                def micro(rows):
                    summary = summarize_counts(rows, [label.stable_id for label in schema.level_labels("P2")])
                    return summary["f1"], summary["tp"], summary["fp"], summary["fn"]
                d_f1, d_tp, d_fp, d_fn = micro(direct)
                r_f1, r_tp, r_fp, r_fn = micro(rolled)
                cells.append({
                    "version": version, "model": model_name, "dataset": dataset, "predicate_level": "P2_vs_P4_rollup",
                    "paired_candidates": len(direct),
                    "same_label_set": sum(left["predicted"] == right["predicted"] and left["status"] == right["status"] for left, right in zip(direct, rolled)),
                    "direct_f1": d_f1, "rollup_f1": r_f1,
                    "direct_counts": [d_tp, d_fp, d_fn], "rollup_counts": [r_tp, r_fp, r_fn],
                })
    for model_name in ("qwen8b", "qwen32b"):
        for dataset in ("biored", "drugprot"):
            for level in ("P2", "P4"):
                changes = Counter()
                for request in requests:
                    if request["dataset"] != dataset or request["predicate_level"] != level:
                        continue
                    before = saved[("v0", model_name)].get(request["request_id"])
                    after = saved[("v1", model_name)].get(request["request_id"])
                    def kind(answer, positive):
                        if answer is None:
                            return "missing"
                        if answer["output_status"] != "valid":
                            return answer["output_status"]
                        if answer["predicted_labels"]:
                            return "nonempty"
                        return "empty"
                    positive = bool(gold[request["candidate_id"]]["gold_labels"])
                    changes[f"{'pos' if positive else 'neg'}:{kind(before, positive)}->{kind(after, positive)}"] += 1
                paired.append({"model": model_name, "dataset": dataset, "predicate_level": level, "changes": dict(changes)})
    payload = {
        "included_in_formal_analysis": False,
        "balanced_diagnostic_sample": True,
        "note": "Overall F1 is from a balanced diagnostic sample and is not comparable to the mostly negative technical sample.",
        "cells": cells,
        "paired_changes": paired,
    }
    (ROOT / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("cells", len(cells), "paired", len(paired))


if __name__ == "__main__":
    main()
