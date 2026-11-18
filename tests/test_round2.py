"""Round-2 checks for scoring, the execution gate, and accumulation."""

import sys
from pathlib import Path

import pytest

from schema_study.evaluation import (
    BOOTSTRAP_SEED,
    apply_document_indices,
    document_resample_indices,
)
from schema_study.formal_eval import FormalScoreError, score_predictions
from schema_study.models.generation import (
    UnsupportedGenerationSetting,
    check_generation_settings,
    generate_requests,
)
from schema_study.models.runner import dry_run_training, execute_training, run_fake_epoch
from schema_study.models.schedule import (
    ROUND_MODEL_EXECUTION_PERMITTED,
    decay_parameter_names,
    learning_rate_at_step,
)
from schema_study.schemas import load_schema

ROOT = Path(__file__).resolve().parents[1]
BIORED = load_schema(ROOT / "configs" / "schemas" / "biored.yaml")
DRUGPROT = load_schema(ROOT / "configs" / "schemas" / "drugprot.yaml")


def _id(schema, display: str) -> str:
    return {name: stable for stable, name in schema.fine_names.items()}[display]


def _candidate(candidate_id: str, gold: list[str], document_id: str = "1", split: str = "test") -> dict:
    return {
        "candidate_id": candidate_id,
        "document_id": document_id,
        "split": split,
        "gold_labels": gold,
        "included_in_scoring": True,
    }


def _prediction(candidate_id: str, labels: list[str], status: str = "valid", illegal: list[str] | None = None) -> dict:
    return {
        "candidate_id": candidate_id,
        "document_id": "1",
        "split": "test",
        "output_status": status,
        "predicted_labels": labels,
        "illegal_labels": illegal or [],
        "gold_labels": ["this field is ignored"],
    }


def _score(gold: list[str], prediction: dict | None, schema=BIORED, level="P4") -> dict:
    candidates = [_candidate("c1", gold)]
    predictions = [] if prediction is None else [prediction]
    return score_predictions(
        candidates,
        predictions,
        schema,
        level,
        allowed_splits={"train", "dev", "test"},
        data_version="v2",
        input_version="accepted",
        prediction_version="synthetic",
        mapping_version=schema.schema_id,
    )["summary"]


def test_package_import_does_not_load_torch():
    import schema_study.models  # noqa: F401

    assert "torch" not in sys.modules
    assert ROUND_MODEL_EXECUTION_PERMITTED is False


def test_dry_run_checks_configuration_without_execution():
    config = {
        "learning_rate": 2e-5,
        "max_epochs": 10,
        "effective_batch_size": 16,
        "micro_batch_size": 8,
        "warmup_ratio": 0.1,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "classifier_dropout": 0.1,
    }
    result = dry_run_training(config)
    assert result["ok"] is True
    assert result["weights_loaded"] is False
    with pytest.raises(Exception):
        execute_training(config, [], requested=True)


def test_effective_batch_stays_16_when_micro_batch_changes():
    result = run_fake_epoch(32, 8)
    assert result["optimizer_steps"] == 2
    assert result["effective_batch_size"] == 16
    assert result["step_flags"].count(True) == 2


def test_schedule_and_decay_groups():
    warmup = learning_rate_at_step(0, 100, 2e-5, 0.1)
    later = learning_rate_at_step(50, 100, 2e-5, 0.1)
    assert warmup < 2e-5
    assert later < 2e-5
    groups = decay_parameter_names(["encoder.weight", "encoder.bias", "encoder.LayerNorm.weight"])
    assert groups["no_decay"] == ["encoder.bias", "encoder.LayerNorm.weight"]


def test_biored_hand_examples():
    label_a = _id(BIORED, "Association")
    label_b = _id(BIORED, "Positive_Correlation")
    assert _score(["Association"], _prediction("c1", [label_a]))["tp"] == 1
    swapped = _score(["Association"], _prediction("c1", [label_b]))
    assert (swapped["fp"], swapped["fn"]) == (1, 1)
    empty = _score(["Association"], _prediction("c1", []))
    assert empty["fn"] == 1
    missing = _score(["Association"], None)
    assert missing["fn"] == 1
    assert missing["n_missing_predictions"] == 1
    negative = _score([], _prediction("c1", []))
    assert negative["exact_match"] == 1
    missing_negative = _score([], None)
    assert missing_negative["exact_match"] == 0
    illegal = _score(["Association"], _prediction("c1", [], "rejected", ["not-a-label"]))
    assert (illegal["fp"], illegal["fn"]) == (1, 1)
    two = _score(["Association"], _prediction("c1", [label_a, label_b], "rejected"))
    assert (two["fp"], two["fn"]) == (1, 1)


def test_drugprot_hand_examples_and_rollup_dedupe():
    agonist = _id(DRUGPROT, "AGONIST")
    activator = _id(DRUGPROT, "AGONIST-ACTIVATOR")
    both = _score(
        ["AGONIST", "AGONIST-ACTIVATOR"],
        _prediction("c1", [agonist, activator]),
        DRUGPROT,
    )
    assert both["tp"] == 2
    partial = _score(["AGONIST", "AGONIST-ACTIVATOR"], _prediction("c1", [agonist]), DRUGPROT)
    assert (partial["tp"], partial["fn"]) == (1, 1)
    illegal = _score(
        ["AGONIST", "AGONIST-ACTIVATOR"],
        _prediction("c1", [agonist], "partial_illegal", ["X"]),
        DRUGPROT,
    )
    assert (illegal["tp"], illegal["fp"], illegal["fn"]) == (1, 1, 1)
    preview = score_predictions(
        [_candidate("c1", ["AGONIST", "AGONIST-ACTIVATOR"])],
        [],
        DRUGPROT,
        "P3",
        allowed_splits={"test"},
        data_version="v1",
        input_version="accepted",
        prediction_version="synthetic",
        mapping_version="drugprot",
    )
    gold = preview["rows"][0]["gold"]
    assert len(gold) == 1
    corrected = score_predictions(
        [_candidate("c1", ["AGONIST", "AGONIST-ACTIVATOR"])],
        [_prediction("c1", gold)],
        DRUGPROT,
        "P3",
        allowed_splits={"test"},
        data_version="v1",
        input_version="accepted",
        prediction_version="synthetic",
        mapping_version="drugprot",
    )["summary"]
    assert corrected["tp"] == 1


def test_formal_score_rejects_duplicates_extras_and_missing_version():
    common = {
        "allowed_splits": {"test"},
        "data_version": "v2",
        "input_version": "x",
        "prediction_version": "x",
        "mapping_version": "x",
    }
    with pytest.raises(FormalScoreError):
        score_predictions([_candidate("c1", []), _candidate("c1", [])], [], BIORED, "P4", **common)
    with pytest.raises(FormalScoreError):
        score_predictions([_candidate("c1", [])], [_prediction("other", [])], BIORED, "P4", **common)
    with pytest.raises(FormalScoreError):
        score_predictions(
            [],
            [],
            BIORED,
            "P4",
            allowed_splits={"test"},
            data_version="",
            input_version="x",
            prediction_version="x",
            mapping_version="x",
        )


def test_repeated_document_draw_counts_every_candidate_again():
    rows = [
        {"document_id": "d", "candidate_id": "a"},
        {"document_id": "d", "candidate_id": "b"},
    ]
    drawn = apply_document_indices(rows, ["d"], [0, 0])
    assert [row["candidate_id"] for row in drawn] == ["a", "b", "a", "b"]
    assert document_resample_indices(2, n_draws=1) == document_resample_indices(2, n_draws=1, seed=BOOTSTRAP_SEED)
    assert BOOTSTRAP_SEED == 100


def test_generation_rejects_unknown_settings_and_skips_completed_requests():
    with pytest.raises(UnsupportedGenerationSetting):
        check_generation_settings({"temperature": 0, "top_k": 10})

    class Scripted:
        def generate(self, prompt: str) -> str:
            return '{"candidate_id":"c1","labels":[]}'

    first = generate_requests(
        [{"request_id": "r1", "candidate_id": "c1", "prompt": "x"}],
        Scripted(),
        BIORED,
        "P4",
    )
    second = generate_requests(
        [{"request_id": "r1", "candidate_id": "c1", "prompt": "x"}],
        Scripted(),
        BIORED,
        "P4",
        ledger=first["adopted"],
    )
    assert first["adopted"][0]["output_status"] == "valid"
    assert second["skipped_completed"] == 1
    assert second["adopted"] == []
