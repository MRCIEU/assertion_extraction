"""Audit the frozen round-4 prompts and saved answers without calling a model."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.dataset as ds

from schema_study.schemas import load_schema

CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
ROUND4 = STORAGE / "studies/schema_granularity_v1/runs/backend_validation/round4_20261115"
OUT = STORAGE / "studies/schema_granularity_v1/runs/backend_validation/round5_20261115/audit"

RUNS = {
    "qwen8b_batch1": ROUND4 / "qwen8b/transformers/batch1_repeat0/attempts.jsonl",
    "qwen8b_batch16": ROUND4 / "qwen8b/transformers/batch16_repeat0/attempts.jsonl",
    "qwen32b_batch16": ROUND4 / "qwen32b/transformers/batch16_repeat0/attempts.jsonl",
    "qwen32b_batch32": ROUND4 / "qwen32b/transformers/batch32_repeat0/attempts.jsonl",
}
FORBIDDEN = (
    "must choose",
    "must select",
    "must pick",
    "exactly one",
    "select one",
    "choose one",
    "only one relation",
    "gold_label",
    "gold_status",
    "included_in_scoring",
    "this candidate is positive",
    "this candidate is negative",
)


def _gold(dataset: str, wanted: set[str]) -> dict[str, dict]:
    version = "biored_candidates_v2" if dataset == "biored" else "drugprot_candidates_v1"
    table = ds.dataset(STORAGE / "data/prepared" / dataset / version / "candidates.parquet", format="parquet")
    rows = {}
    for batch in table.to_batches(columns=["candidate_id", "document_id", "split", "gold_labels"]):
        for row in batch.to_pylist():
            if row["candidate_id"] in wanted:
                rows[row["candidate_id"]] = {
                    "document_id": row["document_id"],
                    "split": row["split"],
                    "gold_labels": list(row["gold_labels"] or []),
                }
    return rows


def _timing(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("included_in_timing"):
            rows[row["request_id"]] = row
    return rows


def main() -> None:
    requests = [json.loads(line) for line in (ROUND4 / "samples/requests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(requests) != 256:
        raise RuntimeError(f"Expected 256 frozen requests, found {len(requests)}.")
    by_candidate: dict[str, list[dict]] = defaultdict(list)
    for row in requests:
        by_candidate[row["candidate_id"]].append(row)
    wanted = {dataset: {row["candidate_id"] for row in requests if row["dataset"] == dataset} for dataset in ("biored", "drugprot")}
    gold = {dataset: _gold(dataset, ids) for dataset, ids in wanted.items()}
    schemas = {name: load_schema(CODE / "configs/schemas" / f"{name}.yaml") for name in ("biored", "drugprot")}
    runs = {name: _timing(path) for name, path in RUNS.items()}

    prompt_findings = []
    for row in requests:
        rendered = row.get("rendered_qwen8b") or ""
        rendered_32 = row.get("rendered_qwen32b") or ""
        lower = rendered.lower()
        problems = []
        if "return an empty labels list" not in lower:
            problems.append("missing_empty_list_rule")
        if "at most one positive label" not in lower:
            problems.append("missing_biored_at_most_one")
        if "more than one positive label" not in lower:
            problems.append("missing_drugprot_multiple")
        compact = "".join(rendered.split())
        if '{"candidate_id":"<candidateid>","labels":["<positivelabel>"]}' not in compact:
            problems.append("json_example_missing")
        for phrase in FORBIDDEN:
            if phrase in lower:
                problems.append(f"forbidden:{phrase}")
        if row["candidate_id"] not in rendered or row["dataset"] not in rendered:
            problems.append("identity_missing")
        if "<<H>>" not in row["prompt"] or "<<T>>" not in row["prompt"]:
            problems.append("role_marker_missing")
        block = "\n".join(f"- {name}" for name in schemas[row["dataset"]].positive_display_names(row["predicate_level"]))
        if block not in row["prompt"]:
            problems.append("label_block_mismatch")
        if "<|im_start|>assistant" not in rendered:
            problems.append("generation_prompt_missing")
        if "<think>" in rendered and "</think>" in rendered:
            inside = rendered.split("<think>", 1)[1].split("</think>", 1)[0]
            if inside.strip():
                problems.append("nonempty_think_block")
        elif "<think>" in rendered:
            problems.append("unclosed_think_block")
        if rendered != rendered_32:
            problems.append("8b_32b_render_differs")
        if problems:
            prompt_findings.append({"request_id": row["request_id"], "problems": problems})

    cells = []
    for dataset in ("biored", "drugprot"):
        for level in ("P2", "P4"):
            subset = [row for row in requests if row["dataset"] == dataset and row["predicate_level"] == level]
            candidates = {row["candidate_id"] for row in subset}
            docs = {gold[dataset][item]["document_id"] for item in candidates}
            splits = Counter(gold[dataset][item]["split"] for item in candidates)
            empty_gold = sum(not gold[dataset][item]["gold_labels"] for item in candidates)
            multi = sum(len(gold[dataset][item]["gold_labels"]) > 1 for item in candidates)
            cell = {
                "dataset": dataset,
                "predicate_level": level,
                "requests": len(subset),
                "unique_candidates": len(candidates),
                "documents": len(docs),
                "splits": dict(splits),
                "gold_empty": empty_gold,
                "gold_nonempty": len(candidates) - empty_gold,
                "gold_multilabel": multi,
                "runs": {},
            }
            for name, answers in runs.items():
                valid_empty = 0
                gold_empty_pred_nonempty = 0
                gold_nonempty_pred_empty = 0
                status = Counter()
                truncated = 0
                for row in subset:
                    answer = answers[row["request_id"]]
                    status[answer["output_status"]] += 1
                    truncated += int(bool(answer["truncated"]))
                    predicted = answer["predicted_labels"]
                    is_valid_empty = answer["output_status"] == "valid" and not predicted
                    valid_empty += int(is_valid_empty)
                    gold_labels = gold[dataset][row["candidate_id"]]["gold_labels"]
                    if not gold_labels and predicted:
                        gold_empty_pred_nonempty += 1
                    if gold_labels and is_valid_empty:
                        gold_nonempty_pred_empty += 1
                cell["runs"][name] = {
                    "valid_empty": valid_empty,
                    "gold_empty_predicted_nonempty": gold_empty_pred_nonempty,
                    "gold_nonempty_valid_empty": gold_nonempty_pred_empty,
                    "status": dict(status),
                    "truncated": truncated,
                    "missing": status["missing"],
                    "rejected": status["rejected"],
                    "partial_illegal": status["partial_illegal"],
                }
            cells.append(cell)

    diff_ids = {}
    pairs = {
        "qwen8b": ("qwen8b_batch1", "qwen8b_batch16"),
        "qwen32b": ("qwen32b_batch16", "qwen32b_batch32"),
    }
    for model, (left_name, right_name) in pairs.items():
        left, right = runs[left_name], runs[right_name]
        ids = []
        for request_id in left:
            if left[request_id]["predicted_labels"] != right[request_id]["predicted_labels"] or left[request_id]["output_status"] != right[request_id]["output_status"]:
                ids.append(request_id)
        diff_ids[model] = sorted(ids)

    diagnostic = {}
    for model, ids in diff_ids.items():
        chosen = list(ids)
        keys = [
            (dataset, quartile, empty)
            for dataset in ("biored", "drugprot")
            for quartile in (1, 2, 3, 4)
            for empty in (True, False)
        ]
        pools = {key: [] for key in keys}
        for row in requests:
            if row["predicate_level"] != "P4":
                continue
            empty = not gold[row["dataset"]][row["candidate_id"]]["gold_labels"]
            pools[(row["dataset"], row["length_quartile"], empty)].append(row["candidate_id"])
        for key in pools:
            pools[key] = sorted(set(pools[key]))
        cursor = {key: 0 for key in keys}
        while len(chosen) < 32:
            progressed = False
            for key in keys:
                while cursor[key] < len(pools[key]) and len(chosen) < 32:
                    candidate_id = pools[key][cursor[key]]
                    cursor[key] += 1
                    pair = [item["request_id"] for item in by_candidate[candidate_id]]
                    if any(item in chosen for item in pair):
                        continue
                    chosen.extend(pair[: 32 - len(chosen)])
                    progressed = True
                    break
            if not progressed:
                break
        diagnostic[model] = {
            "rule": "Include every saved label or status disagreement. Fill to 32 with both P2 and P4 of the next candidate, rotating corpus, context-length quartile, and whether the gold set is empty. The list is fixed before the diagnostic generation.",
            "disagreement_ids": ids,
            "request_ids": chosen,
            "n": len(chosen),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "requests": len(requests),
        "unique_candidates": len(by_candidate),
        "levels_per_candidate": Counter(len(rows) for rows in by_candidate.values()),
        "prompt_problem_count": len(prompt_findings),
        "prompt_findings": prompt_findings[:20],
        "cells": cells,
        "diagnostic_ids": diagnostic,
        "included_in_formal_analysis": False,
    }
    payload["levels_per_candidate"] = dict(payload["levels_per_candidate"])
    (OUT / "task_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for model, item in diagnostic.items():
        (OUT / f"diagnostic_ids_{model}.txt").write_text("\n".join(item["request_ids"]) + "\n", encoding="utf-8")
    print(json.dumps({"requests": 256, "prompt_problems": len(prompt_findings), "diffs": {key: len(value) for key, value in diff_ids.items()}}, indent=2))


if __name__ == "__main__":
    main()
