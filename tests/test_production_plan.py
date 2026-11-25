import json
from pathlib import Path

import pytest

from schema_study.production_inference import pending_requests, release_dependents
from schema_study.production_plan import (
    audit_drugprot_encoder,
    curriculum_sequence,
    curriculum_tasks,
    drugprot_checkpoint_reusable,
    family_counts,
    gpu_authorised,
    membership_rows,
    resolved_tasks,
    runnable_slots,
    submission_allowed,
    unseen_training_views,
    validate_dependencies,
)
from schema_study.analysis_plan import synthetic_report
from schema_study.schemas import default_schema_path, load_schema


def test_counts_follow_the_agreed_conditions():
    counts = family_counts(True)
    assert counts["principal_designed"] == 324
    assert counts["principal_new_training"] == 312
    assert counts["principal_reused_drugprot_encoder"] == 12
    assert counts["bc8_training"] == 0
    assert counts["biolinkbert"] == 48
    assert counts["hierarchical_multitask"] == 12
    assert counts["hierarchical_consistency"] == 12
    assert counts["schema_conditioned_lora"] == 24
    assert counts["curriculum"] == 27
    assert len(curriculum_tasks()) == 27


def test_curriculum_orders_share_the_final_native_block():
    examples = {name: [f"{name}-{index}"] for index, name in enumerate(("P0", "P1", "P2", "P3"))}
    sequences = {order: curriculum_sequence(order, examples) for order in ("coarse_to_fine", "fine_to_coarse", "mixed")}
    finals = {tuple(row for row in sequence if row[2] == "final_native") for sequence in sequences.values()}
    assert len(finals) == 1
    exposures = [sum(1 for row in sequence if row[2] == "exposure") for sequence in sequences.values()]
    assert exposures == [4, 4, 4]


def test_dependencies_cycles_and_queue_limits():
    tasks = [
        {"task_id": "train", "depends_on": ["inputs_prepared", "approval"]},
        {"task_id": "predict", "depends_on": ["train"]},
    ]
    assert validate_dependencies(tasks) == []
    assert validate_dependencies([{"task_id": "predict", "depends_on": ["missing"]}])
    cycle = [
        {"task_id": "a", "depends_on": ["b"]},
        {"task_id": "b", "depends_on": ["a"]},
    ]
    assert validate_dependencies(cycle) == ["dependency cycle"]
    assert submission_allowed(0, 0, 200) == (True, "accepted")
    assert submission_allowed(200, 0, 100) == (True, "accepted")
    assert submission_allowed(0, 400, 200)[0] is False
    assert submission_allowed(0, 0, 1001)[0] is False


def test_approval_must_name_the_resolved_manifest(tmp_path: Path):
    assert gpu_authorised(tmp_path, tmp_path) is False
    (tmp_path / "formal_execution_approval.json").write_text(json.dumps({
        "approved": True,
        "statement": "approved",
        "manifest_sha256": "different",
        "commit": "9dce264",
        "account": "brics.b5ac",
        "scope": "principal_and_agreed_supplementary",
    }), encoding="utf-8")
    assert gpu_authorised(tmp_path, tmp_path) is False


def test_a_checkpoint_is_reusable_only_with_a_completion_record(tmp_path: Path):
    assert drugprot_checkpoint_reusable(tmp_path) is False
    checkpoint = tmp_path / "checkpoints" / "best"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.pt").write_bytes(b"weights")
    (tmp_path / "prediction_complete.json").write_text(json.dumps({"rows_written": True, "checkpoint": "best"}), encoding="utf-8")
    assert drugprot_checkpoint_reusable(tmp_path) is True
    root = tmp_path / "train"
    for predicate in ("P1", "P2", "P3", "P4"):
        for seed in (13, 42, 2026):
            directory = root / "drugprot" / "native" / predicate / f"seed{seed}"
            model = directory / "checkpoints" / "best"
            model.mkdir(parents=True)
            (model / "checkpoint.pt").write_bytes(b"weights")
            (directory / "prediction_complete.json").write_text(json.dumps({"rows_written": True, "checkpoint": "best"}), encoding="utf-8")
    assert audit_drugprot_encoder(root)["reuse_count"] == 12


def test_resolved_tasks_are_not_marked_executable_without_preflight():
    tasks = resolved_tasks()
    assert tasks
    assert all(task["executable"] is False for task in tasks)
    assert all(task["executor"] for task in tasks)
    views = unseen_training_views("E3")
    assert "E3" not in views and "E2" in views and len(views) > 1
    examples = {name: [f"{name}-a", f"{name}-b"] for name in ("P0", "P1", "P2", "P3")}
    mixed = [row for row in curriculum_sequence("mixed", examples) if row[2] == "exposure"]
    blocked = [row for row in curriculum_sequence("coarse_to_fine", examples) if row[2] == "exposure"]
    assert mixed != blocked
    assert [row[0] for row in mixed].count("P0") == [row[0] for row in blocked].count("P0")
    schema = load_schema(default_schema_path(Path("/home/b5ac/freddieyu.b5ac/project_1"), "biored"))
    rows = membership_rows(schema, "P1", "P4")
    assert all(sum(column) == 1 for column in zip(*rows))
    assert pending_requests(["a", "b", "c"], {"a": "valid_empty", "b": "retryable"}) == ["b", "c"]
    released = release_dependents(
        [{"task_id": "predict", "depends_on": ["train"]}, {"task_id": "score", "depends_on": ["predict"]}],
        {"train"},
    )
    assert released == ["predict"]
    report = synthetic_report([
        {"system_a": {"tp": 2, "fp": 0, "fn": 1}, "system_b": {"tp": 1, "fp": 1, "fn": 1}},
        {"system_a": {"tp": 1, "fp": 0, "fn": 0}, "system_b": {"tp": 0, "fp": 1, "fn": 1}},
    ])
    assert report["synthetic"] is True
    assert report["document_f1_averaged"] is False
    assert runnable_slots(200) == 56


def test_torch_fixture_is_not_collected_without_torch():
    torch = pytest.importorskip("torch")
    from schema_study.formal_optim import assistant_supervision, parent_child_penalty
    from schema_study.production_encoder import ProductionHierarchicalEncoder, train_encoder, train_hierarchical
    from schema_study.production_lora import compare_interrupted_run

    labels = assistant_supervision(torch.tensor([[1, 2, 3, 0, 0]]), torch.tensor([[1, 1, 1, 1, 0]]), [2])
    assert labels.tolist() == [[-100, -100, 3, 0, -100]]
    parent = torch.tensor([0.2], requires_grad=True)
    child = torch.tensor([0.8], requires_grad=True)
    parent_child_penalty(parent, child).backward()
    assert parent.grad is not None and float(parent.grad) < 0
    assert child.grad is not None and float(child.grad) > 0
    continued = compare_interrupted_run()
    assert continued["weights_match"] is True
    assert continued["losses_match"] is True
    assert continued["steps"] == 4
    assert continued["empirical_qwen_result"] is False
    hierarchical = train_hierarchical(ProductionHierarchicalEncoder(), consistency=True)
    assert hierarchical["all_heads_receive_gradient"] is True
    assert hierarchical["shared_encoder_gradient"] is True
    weighted = train_encoder(ProductionHierarchicalEncoder(), loss_name="class_weighted")
    focal = train_encoder(ProductionHierarchicalEncoder(), loss_name="focal")
    assert weighted["finite"] is True and focal["finite"] is True
