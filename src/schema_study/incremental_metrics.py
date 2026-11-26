"""Score completed formal predictions without waiting for the rest of the programme.

Gold comes from the candidate tables. A missing prediction is not entered as a
negative observation, and a condition that has not finished is omitted.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.dataset as ds

from schema_study.evaluation import summarize_counts
from schema_study.formal_eval import score_predictions
from schema_study.formal_train import CANDIDATES, VERSIONS
from schema_study.learning_conditions import HISTORICAL_PREDICATES
from schema_study.predictions import rollup_predicted_labels
from schema_study.production_plan import execution_manifest, task_directory
from schema_study.schemas import default_schema_path, load_schema

STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
STATE = STORAGE / "studies/schema_granularity_v1/runs/formal/formal_matched_e0_e7/controller_state.json"
REPORT = STORAGE / "studies/schema_granularity_v1/reports/incremental_formal"
EVAL_SPLITS = {"biored": "test", "bc8": "bc8", "drugprot": "eval"}
USABLE = {"valid", "partial_illegal", "rejected", "missing"}
COARSE = ("P1", "P2", "P3")


def _population(dataset: str, split: str) -> list[dict]:
    rows = []
    table = ds.dataset(CANDIDATES[dataset], format="parquet")
    columns = ["candidate_id", "document_id", "split", "gold_labels", "argument_1_type", "argument_2_type"]
    for batch in table.to_batches(columns=columns, filter=ds.field("split") == split):
        for row in batch.to_pylist():
            rows.append({
                "candidate_id": row["candidate_id"],
                "document_id": str(row["document_id"]),
                "split": row["split"],
                "gold_labels": list(row["gold_labels"] or []),
                "argument_1_type": row.get("argument_1_type"),
                "argument_2_type": row.get("argument_2_type"),
                "included_in_scoring": True,
            })
    return rows


def _ledger(path: Path) -> tuple[list[dict], str | None]:
    rows = []
    invalid_count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            labels = row.get("predicted_labels")
            if isinstance(labels, dict):
                return [], "hierarchical_heads_are_not_one_predicate_level"
            status = row.get("output_status") or row.get("status") or "rejected"
            if status == "valid_empty":
                status, labels = "valid", []
            elif status == "invalid":
                status, labels = "rejected", []
                invalid_count += 1
            elif status not in USABLE:
                status, labels = "rejected", []
                invalid_count += 1
            rows.append({
                "candidate_id": row.get("candidate_id"),
                "document_id": row.get("document_id"),
                "split": row.get("split"),
                "output_status": status,
                "predicted_labels": list(labels or []),
                "illegal_labels": list(row.get("illegal_labels") or []),
            })
    if invalid_count:
        rows.append({"_invalid_count": invalid_count})
    return rows, None


def _compact(summary: dict) -> dict:
    return {
        "n_candidates": summary.get("n_candidates"),
        "tp": summary.get("tp"),
        "fp": summary.get("fp"),
        "fn": summary.get("fn"),
        "precision": summary.get("precision"),
        "recall": summary.get("recall"),
        "f1": summary.get("f1"),
        "macro_f1": summary.get("macro_f1"),
        "failure_count": summary.get("failure_count"),
        "n_missing_predictions": summary.get("n_missing_predictions"),
        "per_class": summary.get("per_class"),
    }


def _recovered(scored: list[dict]) -> set[str]:
    found = set()
    for row in scored:
        gold = set(row.get("gold") or [])
        if gold and gold <= set(row.get("predicted") or []) and row.get("status") in {"valid", "partial_illegal"}:
            found.add(row["candidate_id"])
    return found


def _positive_ids(scored: list[dict]) -> set[str]:
    return {row["candidate_id"] for row in scored if row.get("gold")}


def main() -> None:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    tasks = execution_manifest()
    by_id = {task["task_id"]: task for task in tasks}
    schemas = {}
    populations: dict[tuple[str, str], list[dict]] = {}
    inventory = []
    comparisons = []
    differential = Counter()
    entity_differential = Counter()
    document_recall = []
    for task in tasks:
        entry = state.get(task["task_id"]) or {}
        if entry.get("scientific") != "inference_completed":
            continue
        directory = Path(task_directory(task))
        ledger = directory / "predictions.jsonl"
        if not ledger.is_file():
            inventory.append({"task_id": task["task_id"], "eligible": False, "reason": "prediction ledger absent"})
            continue
        status_path = directory / "scientific_status.json"
        recorded = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        verified = recorded.get("configuration_verified") is True or recorded.get("checkpoint_validated") is True
        level = task.get("schema_predicate") or HISTORICAL_PREDICATES.get(task.get("predicate"))
        dataset = task.get("dataset")
        if dataset not in EVAL_SPLITS or level not in {"P1", "P2", "P3", "P4"}:
            inventory.append({"task_id": task["task_id"], "eligible": False, "reason": "no single evaluation level", "family": task["family"]})
            continue
        schema_name = "drugprot" if dataset == "drugprot" else "biored"
        schema = schemas.setdefault(schema_name, load_schema(default_schema_path(Path("/home/b5ac/freddieyu.b5ac/project_1"), schema_name)))
        targets = [("biored", "test"), ("bc8", "bc8")] if task["family"] == "final_evaluation" and dataset == "biored" else [(dataset, EVAL_SPLITS[dataset])]
        predictions, problem = _ledger(ledger)
        invalid_count = 0
        if predictions and "_invalid_count" in predictions[-1]:
            invalid_count = predictions.pop()["_invalid_count"]
        if problem:
            inventory.append({"task_id": task["task_id"], "eligible": False, "reason": problem, "family": task["family"]})
            continue
        for target, split in targets:
            population = populations.setdefault((target, split), _population(target, split))
            selected = [row for row in predictions if row.get("split") in (None, "", split) and row.get("dataset", target) == target]
            if task["family"] != "final_evaluation":
                selected = predictions
            try:
                scored = score_predictions(
                    population, selected, schema, level,
                    allowed_splits={split}, data_version=VERSIONS[target], input_version="target_anchored_v1",
                    prediction_version=task["family"], mapping_version=schema.schema_id,
                )
            except Exception as error:  # one bad ledger must not discard the others
                inventory.append({"task_id": task["task_id"], "dataset": target, "eligible": False, "reason": str(error), "family": task["family"]})
                continue
            summary = scored["summary"]
            summary["invalid_predictions"] = invalid_count
            item = {
                "task_id": task["task_id"],
                "eligible": verified,
                "eligibility": "eligible" if verified else "configuration_unverified",
                "family": task["family"],
                "model": task.get("model"),
                "dataset": target,
                "entity": task.get("entity"),
                "predicate": task.get("predicate"),
                "schema_predicate": level,
                "seed": task.get("seed"),
                "backend": (task.get("execution") or {}).get("backend") or recorded.get("inference_backend"),
                "checkpoint": task.get("checkpoint") or task.get("adapter_path"),
                "input_fingerprint": (task.get("input") or {}).get("fingerprint"),
                "metrics": _compact(summary),
            }
            inventory.append(item)
            if not verified:
                continue
            types = {row["candidate_id"]: (row.get("argument_1_type"), row.get("argument_2_type")) for row in population}
            by_document = defaultdict(lambda: [0, 0])
            for row in scored["rows"]:
                for label in row.get("gold") or []:
                    hit = label in set(row.get("predicted") or []) and row.get("status") in {"valid", "partial_illegal"}
                    differential[(target, level, label, "recovered" if hit else "missed")] += 1
                    pair = types.get(row["candidate_id"], (None, None))
                    entity_differential[(target, f"{pair[0]}|{pair[1]}", "recovered" if hit else "missed")] += 1
                    by_document[row["document_id"]][0] += int(hit)
                    by_document[row["document_id"]][1] += 1
            recalls = [found / total for found, total in by_document.values() if total]
            if recalls:
                document_recall.append({"task_id": task["task_id"], "dataset": target, "documents": len(recalls), "mean_document_recall": sum(recalls) / len(recalls)})
            if level == "P4":
                item["recovered_positive_candidates"] = sorted(_recovered(scored["rows"]))
                item["positive_candidates"] = sorted(_positive_ids(scored["rows"]))
                for coarse in COARSE:
                    rolled = []
                    for row in selected:
                        rolled.append({**row, "predicted_labels": rollup_predicted_labels(schema, row["predicted_labels"], source_level="P4", target_level=coarse)})
                    rolled_score = score_predictions(
                        population, rolled, schema, coarse,
                        allowed_splits={split}, data_version=VERSIONS[target], input_version="target_anchored_v1",
                        prediction_version=f"{task['family']}_mapped", mapping_version=schema.schema_id,
                    )
                    comparisons.append({
                        "route": "mapped_from_native",
                        "task_id": task["task_id"],
                        "dataset": target,
                        "entity": task.get("entity"),
                        "model": task.get("model"),
                        "family": task["family"],
                        "seed": task.get("seed"),
                        "level": coarse,
                        "metrics": _compact(rolled_score["summary"]),
                        "recovered": sorted(_recovered(rolled_score["rows"])),
                        "positive": sorted(_positive_ids(rolled_score["rows"])),
                    })
            elif level in COARSE:
                comparisons.append({
                    "route": "direct",
                    "task_id": task["task_id"],
                    "dataset": target,
                    "entity": task.get("entity"),
                    "model": task.get("model"),
                    "family": task["family"],
                    "seed": task.get("seed"),
                    "level": level,
                    "metrics": _compact(summary),
                    "recovered": sorted(_recovered(scored["rows"])),
                    "positive": sorted(_positive_ids(scored["rows"])),
                })
    _write(inventory, comparisons, differential, entity_differential, document_recall, by_id, state)


def _matched(comparisons: list[dict]) -> list[dict]:
    direct = {}
    mapped = {}
    for item in comparisons:
        key = (item["dataset"], item["entity"], item["model"], item["family"], item["seed"], item["level"])
        (direct if item["route"] == "direct" else mapped)[key] = item
    rows = []
    for key, coarse in direct.items():
        fine = mapped.get(key)
        if fine is None:
            continue
        shared = set(coarse["positive"]) & set(fine["positive"])
        both = set(coarse["recovered"]) & set(fine["recovered"]) & shared
        only_direct = (set(coarse["recovered"]) - set(fine["recovered"])) & shared
        only_mapped = (set(fine["recovered"]) - set(coarse["recovered"])) & shared
        rows.append({
            "dataset": key[0], "entity": key[1], "model": key[2], "family": key[3], "seed": key[4], "level": key[5],
            "shared_positive_candidates": len(shared),
            "recovered_by_both": len(both),
            "recovered_only_by_direct": len(only_direct),
            "recovered_only_by_mapped_native": len(only_mapped),
            "missed_by_both": len(shared - both - only_direct - only_mapped),
            "direct_f1": coarse["metrics"]["f1"],
            "mapped_f1": fine["metrics"]["f1"],
        })
    return rows


def _write(inventory, comparisons, differential, entity_differential, document_recall, by_id, state) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    public = []
    for item in inventory:
        public.append({key: value for key, value in item.items() if key not in {"recovered_positive_candidates", "positive_candidates"}})
    matched = _matched(comparisons)
    (REPORT / "condition_metrics.json").write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    (REPORT / "common_level_matches.json").write_text(json.dumps(matched, indent=2) + "\n", encoding="utf-8")
    (REPORT / "document_recall.json").write_text(json.dumps(document_recall, indent=2) + "\n", encoding="utf-8")
    relation_rows = []
    labels = {key[2] for key in differential}
    for dataset in ("biored", "bc8", "drugprot"):
        for level in ("P1", "P2", "P3", "P4"):
            for label in sorted(labels):
                recovered = differential[(dataset, level, label, "recovered")]
                missed = differential[(dataset, level, label, "missed")]
                if recovered or missed:
                    relation_rows.append({"dataset": dataset, "level": level, "label": label, "recovered": recovered, "missed": missed, "denominator": recovered + missed})
    (REPORT / "relation_recovery.json").write_text(json.dumps(relation_rows, indent=2) + "\n", encoding="utf-8")
    entity_rows = []
    pairs = {key[1] for key in entity_differential}
    for dataset in ("biored", "bc8", "drugprot"):
        for pair in sorted(pairs):
            recovered = entity_differential[(dataset, pair, "recovered")]
            missed = entity_differential[(dataset, pair, "missed")]
            if recovered or missed:
                entity_rows.append({"dataset": dataset, "entity_pair": pair, "recovered": recovered, "missed": missed, "denominator": recovered + missed})
    (REPORT / "entity_recovery.json").write_text(json.dumps(entity_rows, indent=2) + "\n", encoding="utf-8")
    eligible = [item for item in public if item.get("eligible")]
    held = [item for item in public if item.get("eligibility") == "configuration_unverified"]
    excluded = [item for item in public if item.get("eligible") is False and item.get("eligibility") != "configuration_unverified"]
    lines = [
        "# Incremental formal results",
        "",
        "These figures cover finished prediction conditions only. They are not the completed programme.",
        "",
        f"Eligible conditions: {len(eligible)}. Separately identified because configuration provenance is unverified: {len(held)}. Excluded: {len(excluded)}.",
        "",
        "| Family | Model | Dataset | Level | Conditions | Micro-F1 range | Macro-F1 range |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    groups = defaultdict(list)
    for item in eligible:
        metrics = item.get("metrics") or {}
        if metrics.get("f1") is None:
            continue
        groups[(item.get("family"), item.get("model"), item.get("dataset"), item.get("schema_predicate"))].append(metrics)
    for key, rows in sorted(groups.items()):
        f1 = [row["f1"] for row in rows]
        macro = [row["macro_f1"] for row in rows]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {len(rows)} | {min(f1):.3f}–{max(f1):.3f} | {min(macro):.3f}–{max(macro):.3f} |")
    lines.extend(["", "| Dataset | Entity | Model | Family | Seed | Level | Shared positives | Both | Direct only | Mapped only | Missed | Direct F1 | Mapped F1 |", "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in matched:
        lines.append(
            f"| {row['dataset']} | {row['entity']} | {row['model']} | {row['family']} | {row['seed']} | {row['level']} | {row['shared_positive_candidates']} | {row['recovered_by_both']} | {row['recovered_only_by_direct']} | {row['recovered_only_by_mapped_native']} | {row['missed_by_both']} | {row['direct_f1']:.3f} | {row['mapped_f1']:.3f} |"
        )
    if not matched:
        lines.append("| — | — | — | — | — | — | 0 | 0 | 0 | 0 | 0 | — | — |")
    lines.extend(["", "Matched common-level rows exist only where a direct condition and a native condition share dataset, entity, model, family and seed. An empty table means that pair has not both finished. Relation and entity counts are label instances across finished conditions, so one candidate can appear once for each finished condition.", ""])
    lines.append("| Dataset | Relation | Level | Recovered | Missed | Denominator |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: |")
    for row in relation_rows[:40]:
        lines.append(f"| {row['dataset']} | {row['label']} | {row['level']} | {row['recovered']} | {row['missed']} | {row['denominator']} |")
    lines.extend(["", "| Dataset | Entity-type pair | Recovered | Missed | Denominator |", "| --- | --- | ---: | ---: | ---: |"])
    for row in entity_rows[:30]:
        lines.append(f"| {row['dataset']} | {row['entity_pair']} | {row['recovered']} | {row['missed']} | {row['denominator']} |")
    lines.extend(["", f"Document-clustered recall summaries: {len(document_recall)}. Certainty measurements and the three SciFact verifier results are unchanged and are not re-estimated here.", ""])
    (REPORT / "incremental_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
