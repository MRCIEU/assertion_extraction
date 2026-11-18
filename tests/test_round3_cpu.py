"""CPU closeout checks for bootstrap, checkpoints, tail batches, and dependencies."""

import random
from pathlib import Path

from schema_study.evaluation import (
    BOOTSTRAP_SEED,
    bootstrap_paired_mean_difference,
    document_resample_indices,
    formal_bootstrap_from_config,
    load_formal_bootstrap,
)
from schema_study.execution_plan import execution_rows
from schema_study.models.checkpoints import CheckpointBook, threshold_for_epoch
from schema_study.models.schedule import (
    learning_rate_at_step,
    normalized_group_loss,
    plan_optimizer_groups,
    run_accumulated_epoch,
)
from schema_study.review import REVIEW_SEED

ROOT = Path(__file__).resolve().parents[1]


def _rows(document_id: str, correct: bool) -> list[dict]:
    return [
        {
            "document_id": document_id,
            "gold": ["A"] if correct else [],
            "predicted": ["A"] if correct else [],
            "status": "valid",
            "illegal_labels": [],
        }
    ]


def test_formal_bootstrap_reads_explicit_config_seed():
    config = {"bootstrap": {"seed": 100, "draws": 2000, "interval": "percentile_95"}}
    settings = formal_bootstrap_from_config(config)
    loaded = load_formal_bootstrap(ROOT / "configs" / "evaluation.yaml")
    documents = ["d1", "d2"]
    bundled = {
        "13": {"direct": _rows("d1", True) + _rows("d2", False), "other": _rows("d1", False) + _rows("d2", True)},
        "42": {"direct": _rows("d1", True) + _rows("d2", True), "other": _rows("d1", False) + _rows("d2", False)},
    }
    result = bootstrap_paired_mean_difference(
        bundled,
        documents,
        ["A"],
        condition_a="direct",
        condition_b="other",
        n_draws=settings["n_draws"],
        seed=settings["seed"],
    )
    assert settings == loaded
    assert result["seed"] == config["bootstrap"]["seed"] == 100
    assert result["n_draws"] == 2000
    other = formal_bootstrap_from_config({"bootstrap": {"seed": 101, "draws": 2000, "interval": "percentile_95"}})
    assert other["seed"] == 101
    assert other["seed"] != BOOTSTRAP_SEED
    assert document_resample_indices(3, 2, REVIEW_SEED) != document_resample_indices(3, 2, loaded["seed"])


def test_one_document_sample_is_shared_inside_a_comparison(monkeypatch):
    calls = []
    original = document_resample_indices

    def wrapped(n_documents, n_draws=2000, seed=BOOTSTRAP_SEED):
        calls.append(seed)
        return original(n_documents, n_draws, seed)

    monkeypatch.setattr("schema_study.evaluation.document_resample_indices", wrapped)
    documents = ["d1"]
    bundled = {"13": {"a": _rows("d1", True), "b": _rows("d1", False)}}
    bootstrap_paired_mean_difference(bundled, documents, ["A"], condition_a="a", condition_b="b", n_draws=2, seed=100)
    assert calls == [100]


def test_best_is_independent_of_later_parameter_changes():
    book = CheckpointBook()
    parameters = {"weight": [1.0]}
    for epoch, score, weight in ((1, 0.2, 1.0), (2, 0.9, 2.0), (3, 0.4, 3.0)):
        parameters = {"weight": [weight]}
        book.observe(
            epoch=epoch,
            f1=score,
            parameters=parameters,
            optimizer_updates=epoch * 2,
            scheduler_state={"step": epoch * 2},
            optimizer_state={"step": epoch * 2},
            rng_state={"token": epoch},
            threshold=0.5,
            data_order_seed=13,
            amp_state={"enabled": False},
            training_config={"phase": "cpu_closeout"},
            data_version="v",
        )
    parameters["weight"][0] = 99.0
    assert book.best["completed_epochs"] == 2
    assert book.best["model_parameters"]["weight"] == [2.0]
    assert book.last["completed_epochs"] == 3
    assert book.last["model_parameters"]["weight"] == [3.0]
    assert book.best["best_score"] == 0.9
    assert book.best["threshold"] == 0.5


def test_restore_last_keeps_optimizer_schedule_rng_and_selection():
    book = CheckpointBook()
    rng = random.Random(13)
    rng.random()
    book.observe(
        epoch=2,
        f1=0.4,
        parameters={"weight": [3.0]},
        optimizer_updates=16,
        scheduler_state={"last_step": 16, "learning_rate": 1.5e-5},
        optimizer_state={"step": 16},
        rng_state=rng.getstate(),
        threshold=0.7,
        data_order_seed=13,
        amp_state=None,
        training_config={"seed": 13},
        data_version="drugprot_candidates_v1",
    )
    restored = book.restore_last()
    restored["optimizer_state"]["step"] = 0
    assert book.last["optimizer_state"]["step"] == 16
    assert restored["scheduler_state"]["last_step"] == 16
    assert restored["rng_state"] == rng.getstate()
    assert restored["best_epoch"] == 2
    assert restored["data_order_seed"] == 13


def test_ties_keep_earlier_epoch_and_higher_threshold():
    book = CheckpointBook()
    for epoch, score in ((1, 0.5), (2, 0.5)):
        book.observe(
            epoch=epoch,
            f1=score,
            parameters={"weight": [epoch]},
            optimizer_updates=epoch,
            scheduler_state={"step": epoch},
            optimizer_state={"step": epoch},
            rng_state=epoch,
            threshold=0.4 if epoch == 1 else 0.8,
            data_order_seed=13,
            training_config={},
            data_version="v",
        )
    assert book.best["completed_epochs"] == 1
    chosen = threshold_for_epoch(
        [{"A"}],
        [{"A": 0.9, "B": 0.1}],
        ["A", "B"],
    )
    tied = threshold_for_epoch(
        [set(), {"A"}],
        [{"A": 0.0}, {"A": 1.0}],
        ["A"],
    )
    assert tied["threshold"] == 0.95


def test_tail_group_is_updated_and_normalized_by_its_own_count():
    groups = plan_optimizer_groups(25, 8, 16)
    assert [group["n_examples"] for group in groups] == [16, 9]
    flags = []

    def step(do_step: bool, count: int) -> None:
        flags.append((do_step, count))

    result = run_accumulated_epoch(25, 8, step)
    assert result["optimizer_steps"] == 2
    assert result["example_indices"] == list(range(25))
    assert result["pending_examples"] == 0
    assert flags[-1] == (True, 1)
    tail_totals = [8 * 2.0, 1 * 11.0]
    assert normalized_group_loss(tail_totals, [8, 1]) == (16.0 + 11.0) / 9
    assert normalized_group_loss(tail_totals, [8, 1]) != (2.0 + 11.0) / 2
    rates = [learning_rate_at_step(step_index, 2, 2e-5, 0.1) for step_index in range(result["optimizer_steps"])]
    assert rates[0] != rates[1] or result["optimizer_steps"] == 2


def test_execution_dependencies_cover_84_predictions_without_authorizing_them():
    rows = execution_rows([])
    predictions = [row for row in rows if row["stage"] == "predict"]
    biored = [row for row in predictions if row["dataset"] == "biored"]
    drugprot = [row for row in predictions if row["dataset"] == "drugprot"]
    rollups = [row for row in rows if row["stage"] == "rollup"]
    assert len(biored) == 72
    assert len(drugprot) == 12
    assert all(row["execution_authorized"] is False for row in rows)
    assert all(row["study_phase"] == "formal" for row in rows)
    assert all(row["source_stage"] == "prediction" for row in rollups)
    assert all(row["source_id"].startswith("predict|") for row in rollups)
    assert all(row["requires_model_execution"] is False for row in rollups)
    assert all(row["requires_model_execution"] is True for row in predictions)


def test_smoke_permit_rejects_formal_and_held_out_data(tmp_path):
    from schema_study.models.schedule import ROUND_MODEL_EXECUTION_PERMITTED
    from schema_study.smoke import SmokeConfigError, assert_smoke_config

    output = str(tmp_path / "studies" / "schema_granularity_v1" / "runs" / "smoke" / "round3")
    good = {
        "phase": "smoke",
        "study_phase": "smoke",
        "included_in_formal_analysis": False,
        "model_id": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "revision": "e1354b7a3a09615f6aba48dfad4b7a613eef7062",
        "splits": ["train", "dev"],
        "output_dir": output,
        "max_updates": 32,
    }
    assert_smoke_config(good)
    assert ROUND_MODEL_EXECUTION_PERMITTED is False
    for broken in (
        {**good, "phase": "formal"},
        {**good, "splits": ["train", "test"]},
        {**good, "max_updates": 33},
        {**good, "model_id": "Qwen/Qwen3-32B", "revision": "9216db5781bf21249d130ec9da846c4624c16137", "max_generation_requests": 1},
    ):
        try:
            assert_smoke_config(broken)
        except SmokeConfigError:
            continue
        raise AssertionError(broken)
