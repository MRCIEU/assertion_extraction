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
from schema_study.programme_closure import assess_prompt_requests, merge_acceptance, resolve_equivalence
from schema_study.scientific_coverage import canonical_counts, completion_allowed, coverage_report


def _row(candidate: str, levels: tuple[str, ...] = tuple(f"E{i}" for i in range(9))) -> dict:
    views = {level: f"<<T>>alpha<</T>> and <<T>>beta<</T>> <<TYPE:{level}>>" for level in levels}
    views = {level: text.replace(f"<<TYPE:{level}>>", "") if level == "E0" else text for level, text in views.items()}
    bare = "source"
    views = {level: bare + ("" if level == "E0" else f"<<TYPE:{level}>>") + "<<H>>a<</H>><<T>>b<</T>>" for level in levels}
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
        "rendered_input_match": True,
        "equivalence_method": "encoder_rendered_view",
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
            "rendered_input_match": True,
            "equivalence_method": "encoder_rendered_view",
            "evaluation_rendered_match": False,
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
    assert classify_equivalence(False, "qwen8b", "left_pad_online") == "retrain"
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


def test_snapshot_marker_is_the_execution_revision(tmp_path: Path):
    from schema_study.production_plan import executing_revision

    snapshot = tmp_path / "execution_snapshots" / "abc"
    snapshot.mkdir(parents=True)
    commit = "b07c8000ee5d310e9af17863a54cd0d85d2f8224"
    (snapshot / ".snapshot_complete").write_text(commit + "\n", encoding="utf-8")
    assert executing_revision(snapshot) == commit
    assert executing_revision(tmp_path) is None


def test_execution_snapshot_is_not_a_storage_collision():
    from schema_study.paths import get_paths, overlaps

    storage = Path("/lus/lfs1aip2/projects/b5ac/project_1")
    snapshot = storage / "execution_snapshots" / "4e29f048bcd72427ec097c080521971d9f91a269"
    assert overlaps(storage, snapshot) is False
    assert overlaps(storage, storage / "data") is True
    if snapshot.is_dir():
        paths = get_paths(storage, snapshot)
        assert paths.storage_root == storage


def test_canonical_matrix_counts():
    counts = canonical_counts()
    assert counts["principal_training"] == 360
    assert counts["principal_prompts"] == 304
    assert counts["principal_evaluations"] == 684
    assert counts["biored_evaluations"] == 324
    assert counts["bc8_evaluations"] == 324
    assert counts["drugprot_evaluations"] == 36
    report = coverage_report([], {})
    assert report["passed"] is False
    assert report["cells"] == 1348
    assert report["missing"] == 1348


def test_missing_encoded_length_and_empty_type_map_are_rejected():
    row = _row("c1")
    del row["encoded_length_by_view"]["E8"]
    assert any(reason.startswith("encoded_length_missing") for reason in assess_context_record(row))
    assert type_marker_reasons(row, {}) == ["mention_map_missing"]
    row["split"] = "bc8"
    assert "bad_split" not in assess_context_record(row)


def test_prompt_acceptance_is_independent_and_checks_request_identity():
    kept = merge_acceptance({"acceptance": {"contexts_passed": True, "prompts_passed": True, "validator": "input_acceptance"}}, contexts={"passed": True, "concepts_sha256": "abc"})
    assert kept["acceptance"]["prompts_passed"] is True
    cleared = merge_acceptance(kept, prompts={"passed": False})
    assert cleared["acceptance"]["contexts_passed"] is True
    assert cleared["acceptance"]["prompts_passed"] is False
    row = {
        "candidate_id": "a", "dataset": "biored", "entity_level": "E0", "predicate_level": "P0",
        "rendered": "prompt", "rendered_sha256": __import__("schema_study.batch_plan", fromlist=["prompt_sha256"]).prompt_sha256("prompt"),
        "request_id": "biored:a:E0:P0",
    }
    assert assess_prompt_requests([row, dict(row)], {"a"}, "E0", "P0")["passed"] is False
    assert assess_prompt_requests([row], {"a", "b"}, "E0", "P0")["passed"] is False
    assert assess_prompt_requests([row], {"a"}, "E0", "P0")["passed"] is True


def test_controller_binds_a_predecessor_that_finishes_later(tmp_path: Path):
    from schema_study.input_acceptance import FROZEN_CONCEPT_SHA256
    from schema_study.production_dispatch import bind_resolved_artifact
    from schema_study.production_runtime import resolve_task_equivalence

    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"weights")
    task = {
        "task_id": "corrected", "family": "principal_training", "representation_version": "mesh_ancestor_v1",
        "entity": "E0", "model": "biomedbert", "replaces": "historical",
    }
    info = {
        "predecessor": "historical", "source_family": "principal_training", "model": "biomedbert",
        "rendered_input_match": True, "equivalence_method": "encoder_rendered_view",
        "evaluation_rendered_match": True, "selection_protocol": "encoder_loss", "artifact": str(artifact),
    }
    manifest = {
        "concepts_sha256": FROZEN_CONCEPT_SHA256,
        "acceptance": {"contexts_passed": True, "validator": "input_acceptance"},
        "cell_disposition": {"corrected": "reuse"},
        "equivalence": {"corrected": info},
    }
    state = {"historical": {"state": "running"}, "corrected": {"state": "unsubmitted"}}
    resolve_task_equivalence(state, task, manifest)
    assert state["corrected"]["equivalence_resolution"] == "wait"
    state["historical"]["state"] = "training_completed"
    resolve_task_equivalence(state, task, manifest)
    assert state["corrected"]["scientific"] == "reused_historical_equivalent"
    assert state["corrected"]["artifact"] == str(artifact)
    failed = {"historical": {"state": "terminal_failure"}, "corrected": {"state": "unsubmitted"}}
    resolve_task_equivalence(failed, task, manifest)
    assert failed["corrected"]["equivalence_resolution"] == "retrain"
    downstream = {"family": "final_evaluation", "trains_from": "corrected", "checkpoint_source": str(tmp_path / "empty")}
    bind_resolved_artifact(downstream, state["corrected"])
    assert downstream["checkpoint_source"] == str(artifact)


def test_predecessor_completion_resolves_without_another_validation_run():
    assert resolve_equivalence("reuse", "running", True, True) == "wait"
    assert resolve_equivalence("reuse", "training_completed", True, True) == "reuse"
    assert resolve_equivalence("reuse", "terminal_failure", True, True) == "retrain"
    assert resolve_equivalence("reuse", "training_completed", False, True) == "retrain"
    assert resolve_equivalence("reuse_when_complete", "training_completed", True, False) == "retrain"
    assert resolve_equivalence("retrain", "training_completed", True, True) == "retrain"


def test_validated_checkpoint_exit_is_continuation_not_terminal():
    from schema_study.production_runtime import reconcile_entry

    continued = reconcile_entry(
        {"state": "running", "attempts": 7},
        "terminal",
        {"scientific": "valid_continuation_required", "checkpoint_validated": True},
        task_id="task",
        slurm_state="FAILED",
    )
    assert continued["state"] == "valid_continuation_required"
    assert continued["attempts"] == 7
    failed = reconcile_entry({"state": "running", "attempts": 3}, "terminal", {"scientific": "failed"}, task_id="task", slurm_state="FAILED")
    assert failed["state"] == "terminal_failure"
    preflight = reconcile_entry(
        {"state": "submitted", "attempts": 1},
        "terminal",
        {"checkpoint_validated": True, "technical_preflight": True, "optimiser_steps": 1},
        task_id="representative",
        slurm_state="FAILED",
    )
    assert preflight["state"] == "valid_continuation_required"


def test_rolling_checkpoint_and_population_count_do_not_accept_a_cell(tmp_path: Path):
    from schema_study.programme_closure import assess_prediction_rows, continuation_is_verified, publish_ledger_acceptance
    from schema_study.scientific_coverage import _cell_accepted

    output = tmp_path / "train"
    output.mkdir()
    (output / "adapter_checkpoint.pt").write_bytes(b"rolling")
    (output / "scientific_status.json").write_text(
        '{"selection_recorded": true, "checkpoint_validated": true}\n', encoding="utf-8",
    )
    cell = {"model": "qwen8b", "dataset": "drugprot", "entity": "native", "predicate": "P0", "seed": 13}
    task = {"output": str(output), "model": "qwen8b", "family": "principal_training", "source_family": "principal_training"}
    assert _cell_accepted("training", cell, task, {"scientific": "training_completed"}) is False
    rows = [{"candidate_id": "a", "dataset": "biored"}, {"candidate_id": "b", "dataset": "bc8"}]
    assert assess_prediction_rows(rows, {"a"}, "biored")["passed"] is True
    assert assess_prediction_rows(rows, {"b", "c"}, "bc8")["passed"] is False
    ledger = tmp_path / "eval"
    ledger.mkdir()
    (ledger / "predictions.jsonl").write_text('{"candidate_id": "a", "dataset": "biored"}\n', encoding="utf-8")
    publish_ledger_acceptance(ledger, "biored", [{"candidate_id": "a", "dataset": "biored"}], {"a"})
    (ledger / "predictions.jsonl").write_text('{"candidate_id": "changed", "dataset": "biored"}\n', encoding="utf-8")
    from schema_study.programme_closure import ledger_acceptance_valid

    assert ledger_acceptance_valid(ledger, "biored") is False
    supplementary = {**task, "output": str(ledger), "source_family": "schema_conditioned_lora", "family": "adapter_inference"}
    assert _cell_accepted("evaluations", cell, supplementary, {"scientific": "inference_completed"}) is False
    assert continuation_is_verified(10, 10, False, 10, 4) is False
    assert continuation_is_verified(10, 10, True, 10, 0) is False
