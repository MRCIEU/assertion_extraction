"""Resume and resource-file tests for incomplete generation batches."""

from __future__ import annotations

from schema_study.batch_plan import (
    adopted_answers,
    batches_to_execute,
    classify_attempt,
    freeze_manifest,
)
from schema_study.process_record import process_directory, summarize_processes, write_process_json


def _requests(n: int) -> list[dict]:
    return [
        {
            "request_id": f"r{index}",
            "candidate_id": f"c{index}",
            "dataset": "biored",
            "predicate_level": "P2",
            "prompt": f"prompt-{index}",
        }
        for index in range(n)
    ]


def _answer(request_id: str, candidate_id: str, prompt: str) -> dict:
    from schema_study.batch_plan import prompt_sha256

    return {
        "request_id": request_id,
        "candidate_id": candidate_id,
        "prompt_sha256": prompt_sha256(prompt),
        "status": "completed",
        "is_warmup": False,
    }


def test_complete_batch_is_skipped_and_partial_batch_is_kept_intact():
    manifest = freeze_manifest(_requests(37), 16, "prompt")
    assert [len(batch["request_ids"]) for batch in manifest["batches"]] == [16, 16, 5]
    attempts = []
    for batch in manifest["batches"][:2]:
        for member in batch["members"]:
            attempts.append(_answer(member["request_id"], member["candidate_id"], f"prompt-{member['request_id'][1:]}"))
    # Fifteen answers in the middle batch, and the short final batch is absent.
    middle = manifest["batches"][1]
    attempts = [row for row in attempts if row["request_id"] != middle["request_ids"][-1]]
    adopted = adopted_answers(attempts)
    pending = batches_to_execute(manifest, adopted)
    assert [batch["batch_id"] for batch in pending] == ["batch-0001", "batch-0002"]
    assert pending[0]["request_ids"] == middle["request_ids"]
    fresh = _answer(middle["request_ids"][-1], middle["members"][-1]["candidate_id"], f"prompt-{middle['request_ids'][-1][1:]}")
    assert classify_attempt(middle, fresh, adopted) == "adopt"
    previous = _answer(middle["request_ids"][0], middle["members"][0]["candidate_id"], f"prompt-{middle['request_ids'][0][1:]}")
    assert classify_attempt(middle, previous, adopted) == "retained_previous"


def test_full_ledger_runs_nothing():
    manifest = freeze_manifest(_requests(16), 16, "prompt")
    attempts = [
        _answer(member["request_id"], member["candidate_id"], f"prompt-{member['request_id'][1:]}")
        for member in manifest["batches"][0]["members"]
    ]
    assert batches_to_execute(manifest, adopted_answers(attempts)) == []


def test_two_process_starts_keep_both_resource_records(tmp_path):
    first = process_directory(tmp_path, "job-a", "attempt-1")
    second = process_directory(tmp_path, "job-b", "attempt-1")
    write_process_json(first, "resource_usage.json", {"peak": 1})
    write_process_json(first, "run_clock.json", {"seconds": 3})
    write_process_json(second, "resource_usage.json", {"peak": 2})
    write_process_json(second, "run_clock.json", {"seconds": 4})
    summary = summarize_processes(tmp_path)
    assert summary["process_count"] == 2
    assert summary["processes"][0]["resource"]["peak"] == 1
    assert summary["processes"][1]["run_clock"]["seconds"] == 4
