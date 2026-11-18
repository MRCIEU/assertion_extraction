"""Score completed formal outputs from the independent candidate tables.

This module does not call a model and does not change a frozen prediction.
A unit is scored only after its prediction file or every generation shard is
complete. Missing answers stay in the population.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.dataset as ds

from schema_study.batch_plan import adopted_answers
from schema_study.evaluation import information_retention, polarity_summary, summarize_counts
from schema_study.formal_eval import fine_ids_for_names, gold_groups, score_predictions
from schema_study.generation_ledger import read_jsonl
from schema_study.predictions import rollup_predicted_labels
from schema_study.schemas import Schema, load_schema

CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
ROOT = STORAGE / "studies/schema_granularity_v1/runs/formal/formal_20261115"
CANDIDATES = {
    "biored": STORAGE / "data/prepared/biored/biored_candidates_v2/candidates.parquet",
    "bc8": STORAGE / "data/prepared/bc8/bc8_candidates_v2/candidates.parquet",
    "drugprot": STORAGE / "data/prepared/drugprot/drugprot_candidates_v1/candidates.parquet",
}
VERSIONS = {"biored": "biored_candidates_v2", "bc8": "bc8_candidates_v2", "drugprot": "drugprot_candidates_v1"}
SCHEMAS = {
    "biored": CODE / "configs/schemas/biored.yaml",
    "bc8": CODE / "configs/schemas/biored.yaml",
    "drugprot": CODE / "configs/schemas/drugprot.yaml",
}
EVAL_SPLITS = {"biored": "test", "bc8": "bc8", "drugprot": "eval"}


def _population(dataset: str, split: str) -> list[dict]:
    table = ds.dataset(CANDIDATES[dataset], format="parquet")
    rows = []
    columns = ["candidate_id", "document_id", "split", "gold_labels"]
    for batch in table.to_batches(columns=columns, filter=ds.field("split") == split):
        for row in batch.to_pylist():
            rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "document_id": str(row["document_id"]),
                    "split": row["split"],
                    "gold_labels": list(row["gold_labels"] or []),
                    "included_in_scoring": True,
                }
            )
    return rows


def _train_counts(dataset: str, schema: Schema) -> dict[str, int]:
    cache = ROOT / "evaluation" / f"train_fine_counts_{dataset}.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    counts = Counter()
    table = ds.dataset(CANDIDATES[dataset], format="parquet")
    for batch in table.to_batches(columns=["split", "gold_labels"], filter=ds.field("split") == "train"):
        for row in batch.to_pylist():
            for fine_id in fine_ids_for_names(schema, list(row["gold_labels"] or [])):
                counts[fine_id] += 1
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(dict(counts), indent=2) + "\n", encoding="utf-8")
    return dict(counts)


def _three_level(rows: list[dict], schema: Schema, level: str) -> dict:
    labels = [label.stable_id for label in schema.level_labels(level)]
    summary = summarize_counts(rows, labels)
    negatives = [row for row in rows if not row["gold"]]
    positives = [row for row in rows if row["gold"]]
    positive_summary = summarize_counts(positives, labels) if positives else None
    return {
        "layer1_all_candidates": {
            "precision": summary["precision"],
            "recall": summary["recall"],
            "f1": summary["f1"],
            "macro_f1": summary["macro_f1"],
            "exact_match": summary["exact_match"],
            "tp": summary["tp"],
            "fp": summary["fp"],
            "fn": summary["fn"],
            "extra_fp": summary["extra_fp"],
            "n_candidates": summary["n_candidates"],
        },
        "layer2_existence": {
            "negative_denominator": len(negatives),
            "negative_valid_nonempty": sum(row["status"] == "valid" and bool(row["predicted"]) for row in negatives),
            "negative_valid_empty": sum(row["status"] == "valid" and not row["predicted"] for row in negatives),
            "positive_denominator": len(positives),
            "positive_legal_nonempty": sum(row["status"] in {"valid", "partial_illegal"} and bool(row["predicted"]) for row in positives),
            "positive_valid_empty": sum(row["status"] == "valid" and not row["predicted"] for row in positives),
            "rejected": sum(row["status"] == "rejected" for row in rows),
            "partial_illegal": sum(row["status"] == "partial_illegal" for row in rows),
            "missing": sum(row["status"] == "missing" for row in rows),
            "format_failure_counted_as_correct_no_relation": False,
        },
        "layer3_all_gold_positive": None
        if positive_summary is None
        else {
            "precision": positive_summary["precision"],
            "recall": positive_summary["recall"],
            "f1": positive_summary["f1"],
            "tp": positive_summary["tp"],
            "fp": positive_summary["fp"],
            "fn": positive_summary["fn"],
            "empty_on_positive": sum(row["status"] == "valid" and not row["predicted"] for row in positives),
            "failed_on_positive": sum(row["status"] in {"rejected", "missing"} for row in positives),
            "per_class": positive_summary["per_class"],
        },
    }


def _retention_rows(population: list[dict], scored: list[dict], schema: Schema) -> tuple[list[dict], list[dict]]:
    by_id = {row["candidate_id"]: row for row in scored}
    relations = []
    polarity_rows = []
    for candidate in population:
        scored_row = by_id[candidate["candidate_id"]]
        fine_ids = fine_ids_for_names(schema, candidate["gold_labels"])
        if not fine_ids:
            continue
        relations.extend(
            {
                "fine_id": fine_id,
                "status": scored_row["status"],
                "predicted_groups": scored_row["predicted"],
            }
            for fine_id in fine_ids
        )
        polarity_rows.append(
            {"fine_ids": fine_ids, "status": scored_row["status"], "predicted_groups": scored_row["predicted"]}
        )
    return relations, polarity_rows


def score_rows(population: list[dict], predictions: list[dict], schema: Schema, level: str, dataset: str) -> dict:
    scored = score_predictions(
        population,
        predictions,
        schema,
        level,
        allowed_splits={EVAL_SPLITS[dataset]},
        data_version=VERSIONS[dataset],
        input_version="target_anchored_v1",
        prediction_version="formal_20261115",
        mapping_version="accepted_schema",
    )
    relations, polarity_rows = _retention_rows(population, scored["rows"], schema)
    train_dataset = "biored" if dataset == "bc8" else dataset
    retention = information_retention(relations, schema, level, _train_counts(train_dataset, schema))
    return {
        "summary": scored["summary"],
        "three_level": _three_level(scored["rows"], schema, level),
        "information_retention": retention,
        "polarity": polarity_summary(polarity_rows, schema, level),
        "status_counts": dict(Counter(row["status"] for row in scored["rows"])),
        "valid_empty": sum(row["status"] == "valid" and not row["predicted"] for row in scored["rows"]),
    }


def score_biomedbert_file(path: Path, dataset: str, entity: str, predicate: str, seed: int, eval_dataset: str) -> dict:
    schema = load_schema(SCHEMAS[eval_dataset])
    population = _population(eval_dataset, EVAL_SPLITS[eval_dataset])
    wanted = {row["candidate_id"] for row in population}
    predictions = [row for row in read_jsonl(path) if row["candidate_id"] in wanted]
    if len(predictions) != len(wanted):
        raise RuntimeError(f"{path} has {len(predictions)} rows for {eval_dataset}, expected {len(wanted)}")
    result = score_rows(population, predictions, schema, predicate, eval_dataset)
    result.update({"model": "biomedbert", "dataset": eval_dataset, "entity_level": entity, "predicate_level": predicate, "seed": seed, "source": str(path)})
    return result


def write_completed_biomedbert() -> list[str]:
    written = []
    train_root = ROOT / "train"
    for path in sorted(train_root.glob("*/*/*/seed*/prediction_complete.json")):
        run = path.parent
        dataset, entity, predicate, seed_name = run.relative_to(train_root).parts
        seed = int(seed_name.removeprefix("seed"))
        evals = ("test", "bc8") if dataset == "biored" else ("eval",)
        names = {"test": "biored", "bc8": "bc8", "eval": "drugprot"}
        for split_name in evals:
            eval_dataset = names[split_name]
            destination = ROOT / "evaluation" / "direct" / "biomedbert" / eval_dataset / entity / predicate / f"seed{seed}.json"
            if destination.is_file():
                continue
            result = score_biomedbert_file(run / "predictions.jsonl", dataset, entity, predicate, seed, eval_dataset)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            written.append(str(destination))
    return written


def _read_jsonl_finished_lines(path: Path) -> list[dict]:
    """Read a ledger. A trailing partial line is ignored; a torn earlier line is an error."""
    rows = []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise
    return rows


def _adopted_shard(model: str, shard: dict) -> list[dict] | None:
    output = ROOT / "generation" / "outputs" / model / shard["dataset"] / shard["entity_level"] / shard["predicate_level"] / f"shard_{int(shard['shard_index']):05d}" / "attempts.jsonl"
    if not output.is_file():
        return None
    adopted = adopted_answers(_read_jsonl_finished_lines(output))
    if len(adopted) < int(shard["requests"]):
        return None
    return list(adopted.values())


def write_completed_qwen() -> list[str]:
    summary = json.loads((ROOT / "prepare/length_summary.json").read_text(encoding="utf-8"))
    grouped: dict[tuple, list] = {}
    for shard in summary["shards"]:
        grouped.setdefault((shard["dataset"], shard["entity_level"], shard["predicate_level"]), []).append(shard)
    written = []
    for model in ("qwen8b", "qwen32b"):
        for key, shards in grouped.items():
            dataset, entity, predicate = key
            destination = ROOT / "evaluation" / "direct" / model / dataset / entity / predicate / "generation.json"
            if destination.is_file():
                continue
            collected = []
            truncated = 0
            for shard in shards:
                rows = _adopted_shard(model, shard)
                if rows is None:
                    collected = None
                    break
                truncated += sum(bool(row.get("truncated")) for row in rows)
                collected.extend(rows)
            if collected is None:
                continue
            schema = load_schema(SCHEMAS[dataset])
            population = _population(dataset, EVAL_SPLITS[dataset])
            result = score_rows(population, collected, schema, predicate, dataset)
            result.update(
                {
                    "model": model,
                    "dataset": dataset,
                    "entity_level": entity,
                    "predicate_level": predicate,
                    "truncated": truncated,
                    "source_shards": [shard["path"] for shard in shards],
                }
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            written.append(str(destination))
    return written


def rollup_predictions(predictions: list[dict], schema: Schema, target_level: str) -> list[dict]:
    rolled = []
    for row in predictions:
        copied = dict(row)
        if row["output_status"] in {"valid", "partial_illegal"}:
            copied["predicted_labels"] = rollup_predicted_labels(
                schema, row.get("predicted_labels") or [], source_level="P4", target_level=target_level
            )
        else:
            copied["predicted_labels"] = []
        rolled.append(copied)
    return rolled


def _write_score(destination: Path, result: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def write_completed_rollups() -> list[str]:
    """Map saved P4 predictions onto P3, P2, and P1. No model call and no new threshold."""
    written = []
    summary = json.loads((ROOT / "prepare/length_summary.json").read_text(encoding="utf-8"))
    grouped: dict[tuple, list] = {}
    for shard in summary["shards"]:
        if shard["predicate_level"] != "P4":
            continue
        grouped.setdefault((shard["dataset"], shard["entity_level"]), []).append(shard)
    for model in ("qwen8b", "qwen32b"):
        for (dataset, entity), shards in grouped.items():
            collected = []
            for shard in shards:
                rows = _adopted_shard(model, shard)
                if rows is None:
                    collected = None
                    break
                collected.extend(rows)
            if not collected:
                continue
            schema = load_schema(SCHEMAS[dataset])
            population = _population(dataset, EVAL_SPLITS[dataset])
            for target in ("P3", "P2", "P1"):
                destination = ROOT / "evaluation" / "rollup" / model / dataset / entity / f"P4_to_{target}" / "generation.json"
                if destination.is_file():
                    continue
                result = score_rows(population, rollup_predictions(collected, schema, target), schema, target, dataset)
                result.update(
                    {
                        "model": model,
                        "dataset": dataset,
                        "entity_level": entity,
                        "predicate_level": target,
                        "source_level": "P4",
                        "source_kind": "generation",
                        "threshold_refit": False,
                    }
                )
                _write_score(destination, result)
                written.append(str(destination))
    train_root = ROOT / "train"
    for path in sorted(train_root.glob("*/*/*/seed*/prediction_complete.json")):
        run = path.parent
        dataset, entity, predicate, seed_name = run.relative_to(train_root).parts
        if predicate != "P4":
            continue
        seed = int(seed_name.removeprefix("seed"))
        schema_name = "drugprot" if dataset == "drugprot" else "biored"
        schema = load_schema(SCHEMAS[schema_name])
        evals = ("biored", "bc8") if dataset == "biored" else ("drugprot",)
        predictions = read_jsonl(run / "predictions.jsonl")
        for eval_dataset in evals:
            population = _population(eval_dataset, EVAL_SPLITS[eval_dataset])
            wanted = {row["candidate_id"] for row in population}
            selected = [row for row in predictions if row["candidate_id"] in wanted]
            for target in ("P3", "P2", "P1"):
                destination = ROOT / "evaluation" / "rollup" / "biomedbert" / eval_dataset / entity / f"P4_to_{target}" / f"seed{seed}.json"
                if destination.is_file():
                    continue
                result = score_rows(population, rollup_predictions(selected, schema, target), schema, target, eval_dataset)
                result.update(
                    {
                        "model": "biomedbert",
                        "dataset": eval_dataset,
                        "entity_level": entity,
                        "predicate_level": target,
                        "source_level": "P4",
                        "seed": seed,
                        "source_kind": "checkpoint",
                        "threshold_refit": False,
                        "source": str(run / "predictions.jsonl"),
                    }
                )
                _write_score(destination, result)
                written.append(str(destination))
    return written


def main() -> None:
    written = write_completed_biomedbert() + write_completed_qwen() + write_completed_rollups()
    print(json.dumps({"scored_units": written}, indent=2))


if __name__ == "__main__":
    main()
