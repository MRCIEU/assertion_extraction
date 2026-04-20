# -*- coding: utf-8 -*-
"""Offline diagnosis for benchmark data gaps (no torch dependency)."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def scan_jsonl_splits(jsonl_path: Path) -> tuple[dict[str, int], int]:
    """Return split name counts and total lines."""
    if not jsonl_path.is_file():
        return {}, 0
    ctr: Counter[str] = Counter()
    n = 0
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ctr[str(d.get("source_split") or "")] += 1
            n += 1
    return dict(ctr), n


def diagnose_drugprot_test_gap(processed_dir: Path) -> dict[str, Any]:
    """
    Explain why ``drugprot_official_test_pairs`` may yield zero rows.

    External evaluation requests ``source_split in {"test"}`` for Layer A DrugProt
    (see ``BENCHMARK_SOURCES`` in ``run_external_evaluation``). If the packaged
    JSONL omits the official test split, the stream is empty before any model runs.
    """
    path = processed_dir / "t1_drugprot.jsonl"
    splits, n_lines = scan_jsonl_splits(path)
    raw_roots: list[str] = []
    # processed_dir = <project_1>/training_data_generation/data/processed
    project_root = processed_dir.parent.parent.parent
    pr = project_root / "data" / "raw" / "drugprot"
    if pr.is_dir():
        raw_roots = [p.name for p in pr.iterdir() if p.is_dir()]

    test_present = splits.get("test", 0) > 0
    blocker = "none"
    if not path.is_file():
        blocker = "missing_jsonl"
    elif not test_present:
        blocker = "no_test_split_in_processed_jsonl"

    return {
        "diagnosis_id": "drugprot_official_test_pairs_empty",
        "evaluated_at_utc": None,
        "processed_jsonl": str(path),
        "jsonl_line_count": n_lines,
        "splits_observed_in_jsonl": splits,
        "raw_drugprot_subdirectories_observed": sorted(raw_roots),
        "external_protocol_requested_splits": ["test"],
        "test_split_present_in_processed": test_present,
        "drop_location": "pair_streams.iter_docs_jsonl_filters_by_source_split_before_doc_to_gold_pair_rows",
        "failure_scope": "global_model_independent_empty_stream",
        "root_cause_category": "training_data_packaging"
        if not test_present
        else "unknown_if_test_present_but_empty_pairs",
        "blocker_code": blocker,
        "fixable_without_new_data_acquisition": test_present,
        "fixable_this_pass_without_new_packaged_test_split": False,
        "recommended_future_work": [
            "Acquire DrugProt official test abstracts/entities/relations (or BigBio test split).",
            "Run the same T1 JSONL + S2 mapping pipeline used for train/dev so ``source_split=test`` lines exist.",
            "Re-run external evaluation GPU job to populate ``drugprot_official_test_pairs`` rows.",
        ],
        "notes": "Do not substitute development split as 'test' without relabeling evidence taxonomy; that would change benchmark definition.",
    }


def write_drugprot_diagnosis_artifacts(
    processed_dir: Path,
    manifests_dir: Path,
    tables_dir: Path,
) -> dict[str, Any]:
    """Write ``drugprot_gap_diagnosis.json`` and ``drugprot_gap_diagnosis_table.csv``."""
    diag = diagnose_drugprot_test_gap(processed_dir)
    from datetime import datetime, timezone

    diag["evaluated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    manifests_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    jpath = manifests_dir / "drugprot_gap_diagnosis.json"
    jpath.write_text(json.dumps(diag, indent=2) + "\n", encoding="utf-8")

    rows = [
        {
            "question": "Is DrugProt test data present in processed JSONL?",
            "answer": "yes" if diag["test_split_present_in_processed"] else "no",
            "evidence": f"splits_observed={diag['splits_observed_in_jsonl']}",
        },
        {
            "question": "Where does the pipeline drop rows?",
            "answer": "split_filter_before_pair_extraction",
            "evidence": diag["drop_location"],
        },
        {
            "question": "Data vs export vs mapping vs eval logic?",
            "answer": "data_packaging_gap" if not diag["test_split_present_in_processed"] else "investigate_mapping",
            "evidence": diag["root_cause_category"],
        },
        {
            "question": "Model-independent?",
            "answer": "yes",
            "evidence": diag["failure_scope"],
        },
        {
            "question": "Fixable in this closure pass without new corpora?",
            "answer": "no" if not diag["fixable_without_new_data_acquisition"] else "unsure",
            "evidence": "see recommended_future_work",
        },
    ]
    tpath = tables_dir / "drugprot_gap_diagnosis_table.csv"
    with tpath.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["question", "answer", "evidence"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    unresolved = {
        "status": "unresolved_official_test_unavailable_in_processed_data",
        "blocker": "processed_t1_drugprot_jsonl_contains_only_train_and_development_splits_no_test",
        "blocker_type": "data_acquisition_and_packaging",
        "excluded_from_benchmark_package": "drugprot_official_test_pairs",
        "why_not_use_development_split_instead": (
            "Layer A benchmark contract is official test pairs; substituting development would change "
            "the benchmark definition and is not an acceptable silent replacement."
        ),
        "drugprot_fix_log_reference": "manifests/drugprot_fix_log.json",
        "closure_without_metric": True,
        "updated_at_utc": diag["evaluated_at_utc"],
    }
    (manifests_dir / "drugprot_unresolved_status.json").write_text(
        json.dumps(unresolved, indent=2) + "\n", encoding="utf-8"
    )

    fix_log = {
        "fix_applied": False,
        "closure_pass_id": "external_eval_drugprot_diagnosis",
        "summary": (
            "No code change alters benchmark definitions. Empty DrugProt rows are explained by absent "
            "test split in t1_drugprot.jsonl; raw DrugProt tree on disk has only training/ and development/."
        ),
        "files_changed": [],
        "behavioral_changes": "none",
        "other_benchmarks_affected": False,
        "prior_tables_must_be_regenerated": False,
        "primary_table_notes_column_updated": True,
        "reason_no_code_fix": (
            "Rigorous DrugProt official test evaluation requires packaging official test split through the "
            "same JSONL + S2 mapping pipeline — out of scope for a silent eval-only patch."
        ),
        "updated_at_utc": diag["evaluated_at_utc"],
    }
    (manifests_dir / "drugprot_fix_log.json").write_text(
        json.dumps(fix_log, indent=2) + "\n", encoding="utf-8"
    )

    return diag
