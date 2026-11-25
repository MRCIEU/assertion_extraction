import json
from pathlib import Path

import pytest

from schema_study.production_plan import (
    audit_drugprot_encoder,
    curriculum_sequence,
    curriculum_tasks,
    drugprot_checkpoint_reusable,
    family_counts,
    gpu_authorised,
    submission_allowed,
    validate_dependencies,
)


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


def test_dependencies_and_queue_limits():
    tasks = [
        {"task_id": "train", "depends_on": ["inputs_prepared", "approval"]},
        {"task_id": "predict", "depends_on": ["train"]},
    ]
    assert validate_dependencies(tasks) == []
    assert validate_dependencies([{"task_id": "predict", "depends_on": ["missing"]}])
    assert submission_allowed(0, 0, 200) == (True, "accepted")
    assert submission_allowed(0, 400, 200)[0] is False
    assert submission_allowed(0, 0, 1001)[0] is False


def test_approval_must_name_the_manifest(tmp_path: Path):
    assert gpu_authorised(tmp_path, "9dce264") is False
    (tmp_path / "formal_execution_approval.json").write_text(json.dumps({
        "approved": True,
        "statement": "approved",
        "manifest_sha256": "different",
        "commit": "9dce264",
        "account": "brics.b5ac",
        "scope": "principal_and_agreed_supplementary",
    }), encoding="utf-8")
    assert gpu_authorised(tmp_path, "9dce264") is False


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


def test_torch_fixture_is_not_collected_without_torch():
    torch = pytest.importorskip("torch")
    from schema_study.formal_optim import assistant_supervision, lora_parameter_names, parent_child_penalty, run_continuation_check
    from torch import nn

    labels = assistant_supervision(torch.tensor([[1, 2, 3, 0, 0]]), torch.tensor([[1, 1, 1, 1, 0]]), [2])
    assert labels.tolist() == [[-100, -100, 3, 0, -100]]
    parent = torch.tensor([0.2], requires_grad=True)
    child = torch.tensor([0.8], requires_grad=True)
    parent_child_penalty(parent, child).backward()
    assert parent.grad is not None and float(parent.grad) < 0
    assert child.grad is not None and float(child.grad) > 0

    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4)
            self.other = nn.Linear(4, 4)

    names = lora_parameter_names(list(Projection().named_parameters()))
    assert any("q_proj" in name for name in names)
    assert all("other" not in name for name in names)
    result = run_continuation_check()
    assert result["weights_match"] is True
    assert result["all_heads_receive_gradient"] is True
    assert result["empirical_qwen_result"] is False
