"""Check the saved Qwen3-8B smoke answers without calling the model."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.dataset as ds

from schema_study.predictions import parse_prediction_text, rollup_predicted_labels
from schema_study.schemas import load_schema

CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
SMOKE = STORAGE / "studies/schema_granularity_v1/runs/smoke/round3_20261115"
OUT = STORAGE / "studies/schema_granularity_v1/runs/validation/round4_cpu/existing_qwen_check.json"


def _gold(dataset: str, version: str, wanted: set[str]) -> dict[str, dict]:
    path = STORAGE / "data/prepared" / dataset / version / "candidates.parquet"
    table = ds.dataset(str(path)).to_table(
        columns=["candidate_id", "split", "document_id", "gold_labels", "argument_1_role", "argument_2_role"],
        filter=ds.field("candidate_id").isin(wanted),
    )
    rows = {}
    for row in table.to_pylist():
        labels = row["gold_labels"] or []
        rows[row["candidate_id"]] = {**row, "gold_labels": list(labels)}
    return rows


def _contexts(dataset: str, path: Path, view: str, wanted: set[str]) -> dict[str, str]:
    found = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not any(candidate_id in line for candidate_id in wanted):
                continue
            row = json.loads(line)
            if row.get("candidate_id") in wanted and row.get("split") == "dev":
                found[row["candidate_id"]] = row["views"][view]
            if len(found) == len(wanted):
                break
    return found


def main() -> None:
    requests = [json.loads(line) for line in (SMOKE / "samples/qwen_requests.jsonl").read_text(encoding="utf-8").splitlines()]
    attempts = [json.loads(line) for line in (SMOKE / "qwen3_8b/attempts.jsonl").read_text(encoding="utf-8").splitlines()]
    primary = [row for row in attempts if not row.get("repeat_check")]
    repeats = [row for row in attempts if row.get("repeat_check")]
    by_request = {row["request_id"]: row for row in requests}
    schemas = {name: load_schema(CODE / "configs/schemas" / f"{name}.yaml") for name in ("biored", "drugprot")}
    wanted = {dataset: {row["candidate_id"] for row in requests if row["dataset"] == dataset} for dataset in schemas}
    gold = {
        "biored": _gold("biored", "biored_candidates_v2", wanted["biored"]),
        "drugprot": _gold("drugprot", "drugprot_candidates_v1", wanted["drugprot"]),
    }
    texts = {
        "biored": _contexts("biored", STORAGE / "data/inputs/biored/target_anchored_v1__biomedbert_e1354b7a_proposed/contexts.jsonl", "E2", wanted["biored"]),
        "drugprot": _contexts("drugprot", STORAGE / "data/inputs/drugprot/target_anchored_v1__biomedbert_e1354b7a/contexts.jsonl", "native", wanted["drugprot"]),
    }
    problems = []
    distributions = {}
    empty = Counter()
    totals = Counter()
    status_counts = Counter()
    illegal = 0
    missing = 0
    rejected = 0
    truncated = 0
    reparse_mismatch = 0
    for row in primary:
        request = by_request.get(row["request_id"])
        if request is None:
            problems.append({"request_id": row["request_id"], "problem": "missing_request"})
            continue
        schema = schemas[request["dataset"]]
        expected_labels = schema.positive_display_names(request["level"])
        block = "\n".join(f"- {name}" for name in expected_labels)
        text = texts[request["dataset"]].get(request["candidate_id"])
        if text is None or f"Text:\n{text}" not in request["prompt"]:
            problems.append({"request_id": row["request_id"], "problem": "prompt_text_differs_from_accepted_context"})
        if block not in request["prompt"]:
            problems.append({"request_id": row["request_id"], "problem": "label_block_differs_from_schema"})
        if request["candidate_id"] not in request["prompt"] or request["candidate_id"] != row["candidate_id"]:
            problems.append({"request_id": row["request_id"], "problem": "candidate_id_not_aligned"})
        if "<<H>>" not in (text or "") or "<<T>>" not in (text or ""):
            problems.append({"request_id": row["request_id"], "problem": "missing_target_role_marker"})
        gold_row = gold[request["dataset"]].get(request["candidate_id"])
        if gold_row is None or gold_row["split"] != "dev":
            problems.append({"request_id": row["request_id"], "problem": "gold_candidate_missing_or_not_dev"})
        parsed = parse_prediction_text(row["raw_answer"], expected_candidate_id=request["candidate_id"], schema=schema, level=request["level"])
        if parsed["predicted_labels"] != row["predicted_labels"] or parsed["output_status"] != row["output_status"]:
            reparse_mismatch += 1
        status_counts[row["output_status"]] += 1
        illegal += int(bool(row.get("illegal_labels")))
        missing += int(row["output_status"] == "missing")
        rejected += int(row["output_status"] == "rejected")
        truncated += int(bool(row.get("truncated")))
        key = (request["dataset"], request["level"])
        totals[key] += 1
        labels = tuple(row["predicted_labels"])
        distributions.setdefault(f"{key[0]}|{key[1]}", Counter())[str(labels)] += 1
        if row["output_status"] == "valid" and not row["predicted_labels"]:
            empty[key] += 1
    repeat_checks = []
    for row in repeats:
        base_id = row["request_id"].replace("#repeat", "")
        base = next(item for item in primary if item["request_id"] == base_id)
        request = by_request[base_id]
        schema = schemas[request["dataset"]]
        again = parse_prediction_text(row["raw_answer"], expected_candidate_id=row["candidate_id"], schema=schema, level=request["level"])
        original = parse_prediction_text(base["raw_answer"], expected_candidate_id=base["candidate_id"], schema=schema, level=request["level"])
        repeat_checks.append({
            "request_id": base_id,
            "same_raw_answer": row["raw_answer"] == base["raw_answer"],
            "same_parsed_labels": again["predicted_labels"] == original["predicted_labels"],
            "parsed_labels": again["predicted_labels"],
        })
    direct_and_rollup = {}
    for dataset, schema in schemas.items():
        for level in ("P4", "P2"):
            population = []
            for row in primary:
                request = by_request[row["request_id"]]
                if request["dataset"] == dataset and request["level"] == level:
                    population.append(row)
            if level == "P4":
                rolled = [tuple(rollup_predicted_labels(schema, row["predicted_labels"], source_level="P4", target_level="P2")) for row in population]
                direct = [tuple(item["predicted_labels"]) for item in primary if by_request[item["request_id"]]["dataset"] == dataset and by_request[item["request_id"]]["level"] == "P2"]
                direct_and_rollup[dataset] = {"direct_p2": [list(item) for item in direct], "rolled_p2": [list(item) for item in rolled], "stored_separately": True}
    report = {
        "primary_requests": len(primary),
        "repeat_checks": len(repeats),
        "unique_primary_ids": len({row["request_id"] for row in primary}),
        "problems": problems,
        "reparse_mismatches": reparse_mismatch,
        "status_counts": dict(status_counts),
        "illegal_answers": illegal,
        "missing_answers": missing,
        "rejected_answers": rejected,
        "truncated_answers": truncated,
        "empty_valid": {f"{dataset}|{level}": empty[(dataset, level)] for dataset, level in totals},
        "request_counts": {f"{dataset}|{level}": totals[(dataset, level)] for dataset, level in totals},
        "predicted_label_distributions": {key: dict(value) for key, value in distributions.items()},
        "repeats": repeat_checks,
        "p2_records": direct_and_rollup,
        "prompt_changed": any(item["problem"] == "prompt_text_differs_from_accepted_context" for item in problems),
        "suitable_for_later_comparison": not problems and reparse_mismatch == 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"problems": len(problems), "reparse_mismatches": reparse_mismatch, "primary": len(primary), "repeats": repeat_checks}, indent=2))


if __name__ == "__main__":
    main()
