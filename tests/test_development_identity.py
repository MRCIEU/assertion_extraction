"""Development ledgers are bound to a checkpoint and a request, not a candidate id."""

import json
from pathlib import Path

from schema_study.production_runtime import (
    development_boundary_complete,
    development_evaluation_identity,
    development_request_key,
    load_development_ledger,
    record_development_boundary,
    scheduled_development_steps,
)


def test_schedule_does_not_add_a_step_when_a_segment_ends_early():
    assert scheduled_development_steps(12, 4) == (4, 8, 12)
    assert 3 not in scheduled_development_steps(12, 4)


def test_checkpoint_ledgers_do_not_reuse_each_other(tmp_path: Path):
    population = [
        development_request_key({"candidate_id": "c1", "entity": "E0"}, "P0"),
        development_request_key({"candidate_id": "c1", "entity": "E2"}, "P0"),
    ]
    first = development_evaluation_identity("task", 4, population)
    second = development_evaluation_identity("task", 8, population)
    assert first != second
    path = tmp_path / f"{first}.jsonl"
    row = {"evaluation_id": first, "request_key": list(population[0]), "predicted": ["A"]}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = load_development_ledger(path, first, set(population))
    assert tuple(loaded) == (population[0],)
    try:
        load_development_ledger(path, second, set(population))
    except RuntimeError as error:
        assert "different checkpoint" in str(error)
    else:
        raise AssertionError("A later checkpoint accepted an earlier ledger.")


def test_same_step_different_weights_differ():
    population = [development_request_key({"candidate_id": "c1", "entity": "E0"}, "P0")]
    first = development_evaluation_identity("task", 4, population, checkpoint_fingerprint="weights-a")
    second = development_evaluation_identity("task", 4, population, checkpoint_fingerprint="weights-b")
    assert first != second


def test_historical_principal_identity_is_unchanged():
    from schema_study.production_plan import _historical_principal

    assert _historical_principal("qwen8b", "biored", "E0", "P0", 13)["task_id"] == "c4057ec5116a397b"


def test_prompt_and_parser_bind_the_ledger():
    population = [development_request_key({"candidate_id": "c1", "entity": "E0"}, "P0")]
    first = development_evaluation_identity("task", 4, population, prompt_fingerprint="a", parser_version="interpret_generation")
    second = development_evaluation_identity("task", 4, population, prompt_fingerprint="b", parser_version="interpret_generation")
    third = development_evaluation_identity("task", 4, population, prompt_fingerprint="a", parser_version="other_parser")
    assert len({first, second, third}) == 3


def test_incomplete_metric_is_not_a_score():
    from schema_study.production_runtime import training_status

    assert training_status(12, 12, False) == "valid_continuation_required"


def test_shared_schema_views_stay_distinct():
    coarse = development_request_key({"candidate_id": "c1", "entity": "E0"}, "P0")
    fine = development_request_key({"candidate_id": "c1", "entity": "E2"}, "P0")
    assert coarse != fine


def test_unfinished_final_boundary_stays_open(tmp_path: Path):
    record_development_boundary(tmp_path, 4, "early", True, 2)
    record_development_boundary(tmp_path, 8, "final", False, 2)
    assert development_boundary_complete(tmp_path, 4)
    assert not development_boundary_complete(tmp_path, 8)


def test_direct_and_mapped_share_an_arm_but_not_a_training_id():
    from schema_study.incremental_metrics import comparison_arm

    direct = {"task_id": "direct-model", "family": "principal_training", "predicate": "P0", "seed": 13, "fraction": None}
    mapped = {"task_id": "mapped-model", "family": "principal_training", "predicate": "P3", "seed": 13, "fraction": None}
    efficiency = {"task_id": "small", "family": "data_efficiency", "predicate": "P3", "seed": 13, "fraction": 10}
    by_id = {item["task_id"]: item for item in (direct, mapped, efficiency)}
    assert comparison_arm({"trains_from": "direct-model"}, by_id)["source_family"] == "principal_training"
    assert comparison_arm({"trains_from": "direct-model"}, by_id) == comparison_arm({"trains_from": "mapped-model"}, by_id)
    assert comparison_arm({"trains_from": "small"}, by_id)["fraction"] == 10
