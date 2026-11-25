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
    from schema_study.formal_launch import requeue_permitted
    from schema_study.production_dispatch import main as dispatch_main
    from schema_study.production_plan import approval_digest, retry_decision
    from schema_study.production_records import (
        all_view_records,
        class_weight_from_counts,
        curriculum_records,
        negative_population,
        one_view_records,
        unseen_split,
    )

    tasks = resolved_tasks()
    digest = approval_digest(tasks, "abc")
    changed = [dict(item) for item in tasks]
    changed[0] = dict(changed[0])
    changed[0]["configuration"] = dict(changed[0]["configuration"], learning_rate=9)
    assert approval_digest(changed, "abc") != digest
    walled = [dict(item, walltime="00:10:00") for item in tasks]
    assert approval_digest(walled, "abc") == digest
    assert dispatch_main(["run", "principal_training", "missing"]) == 1
    assert dispatch_main(["run", tasks[0]["family"], tasks[0]["task_id"]]) == 3
    assert pending_requests(["kept"], {"kept": "invalid"}) == []
    assert release_dependents([{"task_id": "needs-model", "depends_on": ["model_assets"]}], set(), set()) == []
    assert release_dependents([{"task_id": "needs-model", "depends_on": ["model_assets"]}], set(), {"model_assets"}) == ["needs-model"]
    assert requeue_permitted(False, False) is False
    assert requeue_permitted(True, False) is True
    assert retry_decision("filesystem", 2) == "stop"
    assert retry_decision("nan_loss", 0) == "stop"
    examples = [
        {"candidate_id": f"{entity}-{predicate}", "entity": entity, "predicate": predicate}
        for entity in tuple(f"E{level}" for level in range(8))
        for predicate in ("P0", "P1", "P2", "P3")
    ]
    held = unseen_split("E3", examples)
    assert all(row["entity"] != "E3" for row in held["train"])
    assert "E2" in {row["entity"] for row in held["train"]}
    assert held["evaluation"]
    views = ["E0/P0", "E2/P3"]
    assert len(one_view_records([{"candidate_id": "c"}], views, 0, 13)) == 1
    assert len(all_view_records([{"candidate_id": "c"}], views)) == 2
    mixed_rows = curriculum_records("mixed", {name: [f"{name}-a", f"{name}-b"] for name in ("P0", "P1", "P2", "P3")})
    assert mixed_rows[-1]["predicate"] == "P3" and mixed_rows[-1]["phase"] == "final_native"
    assert class_weight_from_counts([1, 3]) == [2.0, 2.0 / 3.0]
    sampled = negative_population(
        [
            {"candidate_id": "p", "positive": True, "document_id": "d"},
            {"candidate_id": "h", "positive": False, "document_id": "d"},
            {"candidate_id": "r", "positive": False, "document_id": "other"},
        ],
        "hard_negative",
        13,
    )
    assert sampled["negative_ids"] == ["h"]


def test_torch_fixture_is_not_collected_without_torch(tmp_path: Path):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from schema_study.formal_optim import IGNORE_INDEX, assistant_supervision, parent_child_penalty
    from schema_study.production_assets import GENERATIVE
    from schema_study.production_dispatch import execute_task
    from schema_study.production_encoder import ProductionHierarchicalEncoder, train_encoder, train_hierarchical
    from schema_study.production_lora import causal_next_token_loss, compare_interrupted_run, production_configuration, render_conversation
    from schema_study.production_records import head_sizes
    from schema_study.schemas import default_schema_path, load_schema

    labels = assistant_supervision(torch.tensor([[1, 2, 3, 0, 0]]), torch.tensor([[1, 1, 1, 1, 0]]), [2])
    assert labels.tolist() == [[-100, -100, 3, 0, -100]]
    logits = torch.full((1, 4, 8), -20.0)
    logits[0, 1, 5] = 20.0
    logits[0, 2, 6] = 20.0
    aligned = torch.tensor([[-100, -100, 5, 6]])
    shifted = causal_next_token_loss(logits, aligned)
    unshifted = F.cross_entropy(logits.reshape(-1, 8), aligned.reshape(-1), ignore_index=IGNORE_INDEX)
    assert float(shifted) < 0.1
    assert float(unshifted) > 1.0
    assert production_configuration()["rank"] == 16
    assert production_configuration()["learning_rate"] == 1e-4
    assert production_configuration()["dropout"] == 0.05
    parent = torch.tensor([0.2], requires_grad=True)
    child = torch.tensor([0.8], requires_grad=True)
    parent_child_penalty(parent, child).backward()
    assert parent.grad is not None and float(parent.grad) < 0
    assert child.grad is not None and float(child.grad) > 0
    continued = compare_interrupted_run()
    assert continued["weights_match"] is True
    assert continued["losses_match"] is True
    assert continued["consumed_match"] is True
    assert continued["fresh_model"] is True
    assert continued["full_base_weights_saved"] is False
    assert continued["steps"] == 4
    assert continued["empirical_qwen_result"] is False
    published = execute_task({"model": "cpu_fixture", "family": "cpu_fixture", "task_id": "cpu", "executor": "schema_study.production_lora:fit"}, tmp_path, gpu_permitted=False)
    assert published["completed"] is True
    assert (tmp_path / "predictions.jsonl").is_file()
    hierarchical = train_hierarchical(consistency=True)
    biored_sizes = head_sizes(load_schema(default_schema_path(Path("/home/b5ac/freddieyu.b5ac/project_1"), "biored")), "biored")
    assert hierarchical["head_sizes"] == biored_sizes
    assert hierarchical["head_sizes"]["P3"] != 5
    assert hierarchical["detail"]["consistency"] == "biored_detached_mapped_native_kl"
    assert hierarchical["all_heads_receive_gradient"] is True
    assert hierarchical["shared_encoder_gradient"] is True
    drugprot = train_hierarchical(ProductionHierarchicalEncoder("drugprot"), consistency=True, corpus="drugprot")
    assert drugprot["detail"]["consistency"] == "drugprot_squared_hinge"
    assert drugprot["all_heads_receive_gradient"] is True
    tokenizer_path = GENERATIVE["qwen8b"]["path"]
    if tokenizer_path.is_dir():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        rendered = render_conversation(tokenizer, "Name the relation.", '{"labels": ["Association"]}')
        rendered_labels = assistant_supervision(torch.tensor([rendered["input_ids"]]), torch.ones(1, len(rendered["input_ids"])), [rendered["prompt_tokens"]])
        assert int((rendered_labels[0, rendered["prompt_tokens"]:] != -100).sum()) > 0
    weighted = train_encoder(ProductionHierarchicalEncoder(), loss_name="class_weighted")
    focal = train_encoder(ProductionHierarchicalEncoder(), loss_name="focal")
    assert weighted["finite"] is True and focal["finite"] is True
