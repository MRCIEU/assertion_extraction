"""Acceptance tests for the consolidated release gates."""

from pathlib import Path

from schema_study.adapter_numeric import mapped_keys
from schema_study.checkpoint_roles import release_evidence_complete, role_weights, selected_checkpoint
from schema_study.input_acceptance import (
    assess_context_record,
    assess_population,
    classify_equivalence,
    contexts_accepted,
    reuse_decision,
    type_marker_reasons,
)
from schema_study.scientific_coverage import canonical_counts, completion_allowed


def _row(candidate: str, levels: tuple[str, ...] = tuple(f"E{i}" for i in range(9))) -> dict:
    views = {level: f"<<T>>alpha<</T>> and <<T>>beta<</T>> <<TYPE:{level}>>" for level in levels}
    views = {level: text.replace(f"<<TYPE:{level}>>", "") if level == "E0" else text for level, text in views.items()}
    bare = "source"
    views = {level: bare + ("" if level == "E0" else f"<<TYPE:{level}>>") + "<<T>>a<</T>><<T>>b<</T>>" for level in levels}
    return {
        "candidate_id": candidate,
        "document_id": "1",
        "split": "train",
        "views": views,
        "fragments": [{"char_start": 0, "char_end": 5}],
        "marked_mention_ids": ["m1", "m2"],
        "encoded_length_by_view": {level: 12 for level in levels},
    }


def test_incomplete_context_and_missing_e8_are_rejected():
    row = _row("c1", tuple(f"E{i}" for i in range(8)))
    assert any(reason.startswith("missing_view") and "E8" in reason for reason in assess_context_record(row))


def test_duplicate_candidates_are_rejected():
    rows = [_row("c1"), _row("c1")]
    outcome = assess_population(rows, {"c1"})
    assert outcome["passed"] is False
    assert outcome["duplicate_candidates"] == 1


def test_source_window_and_type_mismatch_are_rejected():
    row = _row("c1")
    row["views"]["E8"] = "different source<<T>>a<</T>><<T>>b<</T>>"
    assert "source_text_differs_across_levels" in assess_context_record(row)
    truncated = _row("c1")
    truncated["views"]["E3"] = "source<<T>>a<</T>>"
    assert "target_markers_missing" in assess_context_record(truncated)
    assert type_marker_reasons(_row("c1"), {"m1": {"E4": "Disease"}})


def test_file_existence_is_not_acceptance_or_release():
    assert contexts_accepted({"contexts_ready": True, "concepts_sha256": "77fa5b41a6e18c277f987dedd74ea28833b8afa4c823b1396c935f5d7ee850a5"}) is False
    assert release_evidence_complete({"verified": True, "checks": ["checkpoint_resumed", "adapter_export"]}) is False
    assert release_evidence_complete({"verified": True, "checks": {"adapter_config_json": True}}) is False


def test_loose_and_supplementary_predecessors_are_rejected(tmp_path: Path):
    artifact = tmp_path / "best_adapter"
    artifact.mkdir()
    (artifact / "adapter_config.json").write_text("{}", encoding="utf-8")
    task = {"replaces": "principal-1", "model": "biomedbert"}
    record = {
        "predecessor": "principal-1",
        "source_family": "lora_sensitivity",
        "model": "biomedbert",
        "training_tokens_match": True,
        "development_tokens_match": True,
        "artifact": str(artifact),
        "selection_protocol": "encoder_loss",
    }
    assert reuse_decision(task, record) is None
    assert reuse_decision({"model": "biomedbert"}, record) is None
    record["source_family"] = "principal_training"
    record["predecessor"] = "other"
    assert reuse_decision(task, record) is None


def test_reused_model_releases_changed_evaluation(tmp_path: Path):
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_text("weights", encoding="utf-8")
    decision = reuse_decision(
        {"replaces": "principal-1", "model": "biomedbert"},
        {
            "predecessor": "principal-1",
            "source_family": "principal_training",
            "model": "biomedbert",
            "training_tokens_match": True,
            "development_tokens_match": True,
            "evaluation_tokens_match": False,
            "selection_protocol": "encoder_loss",
            "artifact": str(artifact),
        },
    )
    assert decision["artifact"] == str(artifact)
    assert decision["release_evaluation"] is True
    assert decision["scientific"] == "reused_historical_equivalent"


def test_best_and_last_weights_stay_distinct():
    rolling = {"adapter": {"w": "B"}, "best_adapter": {"w": "A"}, "step": 10}
    best = {"adapter": {"w": "A"}, "best_adapter": {"w": "A"}, "role": "best", "step": 4}
    assert role_weights(best, "best")["w"] == "A"
    assert role_weights(rolling, "last")["w"] == "B"
    assert role_weights(rolling, "best")["w"] == "A"
    assert (rolling.get("best_adapter") or rolling.get("adapter"))["w"] == "A"


def test_incomplete_score_selects_nothing():
    assert selected_checkpoint({"best": 0.4, "last": -1}) is None
    assert selected_checkpoint({"best": 0.2, "last": 0.4}) == "last"
    assert classify_equivalence(False, "qwen8b", "unrecoverable") == "retrain"
    assert classify_equivalence(False, "biomedbert", "encoder_loss") == "reuse"


def test_missing_or_superseded_cells_cannot_complete():
    assert completion_allowed([]) is False
    assert completion_allowed([{"status": "training_completed", "artifact_present": False}]) is False
    assert completion_allowed([{"status": "superseded_stored_representation", "artifact_present": True, "superseded_without_replacement": True}]) is False
    assert completion_allowed([{"status": "training_completed", "artifact_present": True}]) is True


def test_parameter_name_collision_is_rejected_before_export():
    try:
        mapped_keys(["lora_A", "lora_B"], lambda name: "same")
    except ValueError as error:
        assert "collision" in str(error)
    else:
        raise AssertionError("A colliding export key was accepted.")


def test_canonical_matrix_counts():
    counts = canonical_counts()
    assert counts["principal_training"] == 360
    assert counts["principal_prompts"] == 304
    assert counts["principal_evaluations"] == 684
