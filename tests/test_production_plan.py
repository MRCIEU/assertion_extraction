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


def test_operational_evaluation_covers_agreed_adapters():
    from schema_study.production_plan import execution_manifest

    tasks = execution_manifest()
    training_families = {
        "principal_training", "biolinkbert", "hierarchical_multitask", "hierarchical_consistency",
        "schema_conditioned_lora", "curriculum", "class_imbalance", "negative_sampling",
        "data_efficiency", "unseen_schema", "encoder_sensitivity", "lora_sensitivity",
    }
    training = [task for task in tasks if task["family"] in training_families]
    assert len(training) == 576
    assert not any(task.get("dataset") == "bc8" and task["family"] in training_families for task in tasks)
    biored = [task for task in training if task["family"] == "principal_training" and task["dataset"] == "biored" and task["adaptation"] == "lora_sft"]
    drug = [task for task in training if task["family"] == "principal_training" and task["dataset"] == "drugprot" and task["adaptation"] == "lora_sft"]
    assert len(biored) == 192 and len(drug) == 24
    by_source: dict[str, list] = {}
    for task in tasks:
        if task["family"] == "adapter_inference":
            by_source.setdefault(task["trains_from"], []).append(task)
    for source in biored:
        roles = {item.get("evaluation_role") for item in by_source[source["task_id"]]}
        assert roles == {"internal", "external"}
        assert {item["dataset"] for item in by_source[source["task_id"]]} == {"biored", "bc8"}
    for source in drug:
        assert [item["dataset"] for item in by_source[source["task_id"]]] == ["drugprot"]
    encoder_families = {
        "principal_training", "biolinkbert", "hierarchical_multitask", "hierarchical_consistency",
        "class_imbalance", "negative_sampling", "data_efficiency", "encoder_sensitivity",
    }
    encoder_ids = {task["task_id"] for task in training if task["family"] in encoder_families and not str(task.get("model") or "").startswith("qwen")}
    final_ids = {task["trains_from"] for task in tasks if task["family"] == "final_evaluation"}
    assert encoder_ids <= final_ids
    prompts = [task for task in tasks if task["family"] == "zero_shot" and "blocked" not in task["statuses"]]
    assert prompts and all(task.get("shard_directory") for task in prompts)


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
        prompt = list(tokenizer.apply_chat_template(
            [{"role": "user", "content": "Name the relation."}], tokenize=True, add_generation_prompt=True, enable_thinking=False,
        ))
        rendered_labels = assistant_supervision(torch.tensor([rendered["input_ids"]]), torch.ones(1, len(rendered["input_ids"])), [rendered["prompt_tokens"]])
        assert int((rendered_labels[0, rendered["prompt_tokens"]:] != -100).sum()) > 0
        if rendered["boundary"] == "prefix":
            assert rendered["input_ids"][: len(prompt)] == prompt
        else:
            assert rendered["boundary"] == "assistant_mask"
    weighted = train_encoder(ProductionHierarchicalEncoder(), loss_name="class_weighted")
    focal = train_encoder(ProductionHierarchicalEncoder(), loss_name="focal")
    assert weighted["finite"] is True and focal["finite"] is True
    from schema_study.production_lora import FixtureCausalModel, _fixture_batches, fit, prepare_process

    configuration = production_configuration(micro_batch_size=2, world_size=1)
    configuration["effective_batch_size"] = 16
    configuration["eval_interval"] = 1
    batches = _fixture_batches(32)
    stopped = fit(FixtureCausalModel(), batches, configuration=configuration, steps=4, total_steps=4, stop_after=2)
    assert stopped["steps"] == 2
    assert stopped["completed"] is False
    assert stopped["status"] == "valid_continuation_required"
    scores = iter((0.9, 0.1))
    selected = fit(FixtureCausalModel(), batches, configuration=configuration, steps=2, total_steps=2, score_fn=lambda _model: next(scores), output_dir=tmp_path / "selection")
    assert selected["completed"] is True
    assert selected["best"]["development_micro_f1"] == 0.9
    assert selected["best"]["step"] == 1
    assert (tmp_path / "selection" / "best" / "adapter_checkpoint.pt").is_file()
    assert (tmp_path / "selection" / "best_adapter" / "adapter_config.json").is_file()
    saved_selection = torch.load(tmp_path / "selection" / "adapter_checkpoint.pt", map_location="cpu", weights_only=False)
    restored = FixtureCausalModel()
    resumed = fit(restored, batches, configuration=configuration, steps=2, total_steps=2, resume=saved_selection, fresh=restored, score_fn=lambda _model: 0.0)
    assert resumed["best"]["development_micro_f1"] == 0.9
    assert resumed["steps"] == 2
    from schema_study.production_encoder import TinyEncoder

    encoder = TinyEncoder(8, 4)
    tokens = torch.tensor([[1, 2, 3, 0]])
    masked = encoder(tokens, torch.tensor([[1, 1, 1, 0]]))
    unmasked = encoder(tokens, torch.ones(1, 4))
    assert not torch.allclose(masked, unmasked)
    verifier = execute_task({"executor": "schema_study.measurement_cpu:adapt_verifier", "family": "verifier_adaptation", "task_id": "verifier"}, tmp_path / "verifier", gpu_permitted=False)
    assert verifier["started"] is False and verifier["completed"] is False
    partial = fit(
        FixtureCausalModel(), batches, configuration=configuration, steps=1, total_steps=2,
        fingerprints={"input": "population-a"}, output_dir=tmp_path / "finger",
    )
    saved = torch.load(tmp_path / "finger" / "adapter_checkpoint.pt", map_location="cpu", weights_only=False)
    fresh = FixtureCausalModel()
    with pytest.raises(RuntimeError, match="fingerprint"):
        fit(fresh, batches, configuration=configuration, steps=2, total_steps=2, resume=saved, fresh=fresh, fingerprints={"input": "population-b"})
    import os
    os.environ.pop("WORLD_SIZE", None)
    with pytest.raises(RuntimeError, match="world size"):
        prepare_process(2)
    status = json.loads((tmp_path / "selection" / "scientific_status.json").read_text(encoding="utf-8"))
    assert status["scientific"] == "training_completed"
    assert status["monitoring_guarantees_completion"] is False
    continued = json.loads((tmp_path / "finger" / "scientific_status.json").read_text(encoding="utf-8"))
    assert continued["scientific"] == "valid_continuation_required"
    exported = json.loads((tmp_path / "selection" / "best_adapter" / "adapter_config.json").read_text(encoding="utf-8"))
    assert exported["peft_type"] == "LORA"
    assert not (tmp_path / "selection" / "best_adapter" / "adapter_checkpoint.pt").exists()
    from schema_study.production_lora import cpu_collective_probe

    collective = cpu_collective_probe(tmp_path / "gloo")
    assert collective["gh200"] is False
    assert collective["ranks"][0]["generated"] == 1
    assert collective["ranks"][1]["generated"] == 0
    assert collective["ranks"][0]["broadcast"] == 0 and collective["ranks"][1]["broadcast"] == 0


def test_population_horizon_and_scheduler_are_distinct():
    from schema_study.formal_launch import matched_main
    from schema_study.production_inference import candidate_scores, constrained_generate, evidence_generate, self_consistency
    from schema_study.production_runtime import (
        assert_runtime_world,
        constrained_sampling,
        context_sources,
        evaluation_released,
        external_model_convention,
        generation_requests,
        horizon_for_task,
        interpret_generation,
        optimiser_horizon,
        placement,
        sbatch_argv,
        select_population,
        sensitivity_changes,
        training_status,
        write_shards,
    )
    from schema_study.schemas import default_schema_path, load_schema

    horizon = optimiser_horizon(20, 3, 1, 1)
    assert horizon["remainder_dropped"] == 4
    assert horizon["total_updates"] == 3
    with pytest.raises(RuntimeError, match="world size"):
        assert_runtime_world(2, 1)
    qwen32 = placement("qwen32b", 2)
    assert qwen32["method"] == "fsdp_full_shard" and qwen32["parameter_sharding"] is True
    qwen8 = placement("qwen8b", 2)
    assert qwen8["replicates_full_parameters"] is True and qwen8["parameter_sharding"] is False
    rows = []
    for split in ("train", "dev"):
        for entity in ("E0", "E2"):
            for predicate in ("P0", "P3"):
                rows.append({
                    "candidate_id": f"{split}-{entity}",
                    "entity": entity,
                    "predicate": predicate,
                    "split": split,
                    "document_id": "d1" if entity == "E0" else "d2",
                    "assistant": {"labels": ["Association"] if predicate == "P3" and entity == "E0" else []},
                })
    independent = select_population({"family": "principal_training", "entity": "E2", "predicate": "P3"}, rows)
    other = select_population({"family": "principal_training", "entity": "E0", "predicate": "P0"}, rows)
    assert {row["candidate_id"] for row in independent["rows"]} == {"train-E2"}
    assert independent["rows"] != other["rows"]
    drugprot_rows = []
    for predicate in ("P0", "P1", "P2", "P3"):
        drugprot_rows.append({"candidate_id": "c", "entity": "native", "predicate": predicate, "split": "train", "assistant": {"labels": []}, "document_id": "d"})
    one = select_population({"family": "schema_conditioned_lora", "dataset": "drugprot", "exposure": "resource_efficient_one_view", "seed": 13}, drugprot_rows)
    every = select_population({"family": "schema_conditioned_lora", "dataset": "drugprot", "exposure": "exposure_controlled_all_views", "seed": 13}, drugprot_rows)
    assert len(one["rows"]) == 3
    assert len(every["rows"]) == 4
    assert all(row["entity"] == "native" for row in one["rows"])
    biored_rows = [
        {"candidate_id": "c", "entity": entity, "predicate": predicate, "split": "train", "assistant": {"labels": []}, "document_id": "d"}
        for entity in tuple(f"E{level}" for level in range(8))
        for predicate in ("P0", "P1", "P2", "P3")
    ]
    shared = select_population({"family": "schema_conditioned_lora", "dataset": "biored", "exposure": "resource_efficient_one_view", "seed": 13}, biored_rows)
    assert shared["rows"][0]["entity"] != "native"
    curriculum_rows = [
        {"candidate_id": f"{entity}-{predicate}-{copy}", "entity": entity, "predicate": predicate, "split": "train", "assistant": {"labels": []}, "document_id": "d"}
        for entity in ("E2", "E7")
        for predicate in ("P0", "P1", "P2", "P3")
        for copy in (1, 2)
    ]
    coarse = select_population({"family": "curriculum", "entity": "E2", "order": "coarse_to_fine"}, curriculum_rows)
    mixed = select_population({"family": "curriculum", "entity": "E2", "order": "mixed"}, curriculum_rows)
    assert [row["predicate"] for row in coarse["rows"] if row["phase"] == "exposure"] != [row["predicate"] for row in mixed["rows"] if row["phase"] == "exposure"]
    assert coarse["rows"][-1]["predicate"] == "P3"
    assert all(row["entity"] == "E2" for row in coarse["rows"])
    unseen_rows = [
        {"candidate_id": f"{entity}", "entity": entity, "predicate": "P3", "split": "train", "assistant": {"labels": []}, "document_id": entity}
        for entity in tuple(f"E{level}" for level in range(8))
    ]
    unseen_rows.append({"candidate_id": "E3-dev", "entity": "E3", "predicate": "P3", "split": "dev", "assistant": {"labels": []}, "document_id": "dev"})
    held = select_population({"family": "unseen_schema", "withheld": "E3"}, unseen_rows)
    assert all(row["entity"] != "E3" for row in held["rows"])
    assert held["evaluation"]
    documents = []
    for index in range(10):
        documents.append({"candidate_id": f"c{index}", "entity": "E2", "predicate": "P3", "split": "train", "document_id": f"doc{index}", "assistant": {"labels": []}})
    documents.append({"candidate_id": "dev", "entity": "E2", "predicate": "P3", "split": "dev", "document_id": "held", "assistant": {"labels": []}})
    subset = select_population({"family": "data_efficiency", "entity": "E2", "predicate": "P3", "fraction": 10, "seed": 13}, documents)
    allowed = set(subset["documents"])
    assert subset["rows"] and all(row["document_id"] in allowed for row in subset["rows"])
    assert all(row["split"] == "dev" for row in subset["evaluation"])
    negatives = select_population(
        {"family": "negative_sampling", "entity": "native", "predicate": "P3", "strategy": "hard_negative", "seed": 13},
        [
            {"candidate_id": "p", "entity": "native", "predicate": "P3", "split": "train", "document_id": "d", "assistant": {"labels": ["Activation"]}},
            {"candidate_id": "h", "entity": "native", "predicate": "P3", "split": "train", "document_id": "d", "assistant": {"labels": []}},
            {"candidate_id": "r", "entity": "native", "predicate": "P3", "split": "train", "document_id": "other", "assistant": {"labels": []}},
            {"candidate_id": "eval", "entity": "native", "predicate": "P3", "split": "dev", "document_id": "other", "assistant": {"labels": []}},
        ],
    )
    assert negatives["negative_ids"] == ["h"]
    assert [row["candidate_id"] for row in negatives["evaluation"]] == ["eval"]
    with pytest.raises(ValueError, match="identical"):
        sensitivity_changes({"learning_rate": 1e-4, "rank": 16}, {"learning_rate": 1e-4, "rank": 16})
    assert sensitivity_changes({"learning_rate": 1e-4, "rank": 16}, {"learning_rate": 2e-4, "rank": 16})["learning_rate"] == 2e-4
    assert training_status(1, 10, False) == "valid_continuation_required"
    assert training_status(10, 10, False) == "valid_continuation_required"
    assert training_status(10, 10, True) == "training_completed"
    schema = load_schema(default_schema_path(Path("/home/b5ac/freddieyu.b5ac/project_1"), "biored"))
    empty = interpret_generation('{"labels": []}', "c1", schema, "P4")
    assert empty["ledger_status"] == "valid_empty" and empty["retry"] is False
    invalid = interpret_generation("not json", "c1", schema, "P4")
    assert invalid["ledger_status"] == "invalid" and invalid["retry"] is False
    with pytest.raises(ValueError, match="model input"):
        generation_requests([{"request_id": "r", "candidate_id": "c", "prompt": "c", "model": "qwen8b"}])
    requests = generation_requests([{"request_id": "r", "candidate_id": "c", "prompt": "Read this passage.", "model": "qwen8b"}])
    shards = write_shards(requests, Path("/tmp/schema-study-shards-test"), 1)
    assert "Read this passage." in shards[0].read_text(encoding="utf-8")
    assert "structured_outputs" in constrained_sampling({"type": "object"})
    constrained = constrained_generate([{"request_id": "r", "prompt": "Read this passage."}])
    assert constrained["submitted"] is False and constrained["requests"][0]["decoding"] == "constrained"
    calls = []

    def _sample(request):
        calls.append(request["sample_index"])
        return {"raw": "{}", "labels": ["A"]}

    aggregated = self_consistency([{"request_id": "r", "multi": False}], {}, backend=_sample)
    assert calls == [0, 1, 2, 3, 4]
    assert aggregated["executed"] is True
    scored = candidate_scores(
        [{"request_id": "r", "multi": False, "legal_labels": ["A", "B"]}],
        {},
        backend=lambda _request: {"token_logprobs": {"A": [-0.4, -0.2], "B": [-0.1]}},
    )
    assert scored["executed"] is True and scored["ledger"]["r"] == "valid"
    with pytest.raises(RuntimeError, match="token log"):
        candidate_scores([{"request_id": "r", "legal_labels": ["A", "B"]}], {}, backend=lambda _request: {"scores": {"A": 1.0}})
    evidence = evidence_generate([{"request_id": "r", "source_text": "No span here."}], {}, backend=lambda _request: {"raw": '{"labels": ["A"]}', "labels": ["A"], "spans": []})
    assert evidence["executed"] is True
    command = sbatch_argv({"family": "principal_training", "task_id": "t", "resources": {"gpus": 1}, "walltime": "04:00:00"}, "scripts/slurm/matched_task.sbatch")
    assert command[0] == "sbatch" and "--account=brics.b5ac" in command and command[-3].endswith("matched_task.sbatch")
    assert evaluation_released("valid_continuation_required") is False
    assert evaluation_released("training_completed") is True
    assert matched_main(["--submit"]) == 3
    assert external_model_convention("qwen36")["executable"] is False
    assert external_model_convention("gpt_oss_20b")["uses_qwen_chat_template"] is False
    curriculum = horizon_for_task({"family": "curriculum"}, 32, 1, 1, sequence_examples=32)
    assert curriculum["epochs"] == 1 and curriculum["total_updates"] == 2
    thirty_two = [task for task in resolved_tasks() if task.get("model") == "qwen32b" and task["family"] == "principal_training"]
    assert thirty_two and thirty_two[0]["resources"]["strategy"] == "fsdp_full_shard"
    curriculum_tasks = [task for task in resolved_tasks() if task["family"] == "curriculum"]
    assert curriculum_tasks[0]["configuration"]["max_epochs"] == 1
    assert curriculum_tasks[0]["configuration"]["data_schedule"] == "single_matched_curriculum_pass"
    assert context_sources("bc8") == ("bc8", "biored")
    assert context_sources("drugprot") == ("drugprot",)


def test_controller_requeue_preflight_and_backend(tmp_path: Path):
    import threading

    from schema_study.production_dispatch import _run_generation, backend_specification
    from schema_study.production_runtime import (
        controller_decisions,
        controller_requeue,
        controller_transaction,
        coordinate_rank_evaluation,
        interpret_slurm,
        merge_group_block,
        probe_readiness,
        programme_remaining,
        publish_validation,
        reconcile_entry,
        record_submission,
        validate_request_shards,
        verify_peft_adapter,
    )

    tasks = [
        {"task_id": "rep", "family": "principal_training", "model": "qwen8b", "resources": {"gpus": 1}, "depends_on": ["approval"], "statuses": [], "representative": True, "representative_group": "g"},
        {"task_id": "later", "family": "principal_training", "model": "qwen8b", "resources": {"gpus": 1}, "depends_on": ["approval"], "statuses": [], "representative": False, "representative_group": "g"},
        {"task_id": "adapt", "family": "adapter_inference", "depends_on": ["rep"], "statuses": [], "trains_from": "rep", "representative": True, "representative_group": "adapt"},
    ]
    evidence = {"approval": {"ready": True}}
    first = controller_decisions(tasks, {}, 0, 0, evidence)
    assert [row["action"] for row in first] == ["submit", "skip", "skip"]
    state = record_submission({}, "rep", "501", "rev")
    assert programme_remaining(tasks, state) is True
    command = controller_requeue("90", True, finished=False, active_job_ids=["501"])
    assert "--begin=now+10minutes" in command and "afterany:90" in command[2]
    assert "501" not in command[2]
    state["rep"] = reconcile_entry(state["rep"], interpret_slurm("501", {"501": "RUNNING"}, None), None, task_id="rep", group="g")
    second = controller_decisions(tasks, state, 1, 0, evidence)
    assert [row["action"] for row in second] == ["skip", "skip", "skip"]
    state["rep"] = reconcile_entry(state["rep"], "running", {"technical_preflight": True}, task_id="rep", group="g", elapsed_seconds=0)
    waiting = controller_decisions(tasks, state, 1, 0, evidence)
    assert waiting[1]["reason"] == "representative_pending"
    state["rep"] = reconcile_entry(state["rep"], "running", {"technical_preflight": True}, task_id="rep", group="g", elapsed_seconds=600)
    assert state["rep"].get("scientific") != "preflight_passed"
    state["rep"] = reconcile_entry(
        state["rep"], "running", {"technical_preflight": True, "useful_computation_started": 1},
        task_id="rep", group="g", observation_seconds=600,
    )
    released = controller_decisions(tasks, state, 1, 0, evidence)
    assert released[1]["action"] == "submit"
    assert released[2]["action"] == "skip"
    state = record_submission(state, "later", "502", "rev")
    repeated = controller_decisions(tasks, state, 2, 0, evidence)
    assert repeated[1]["reason"] == "submitted"
    continued = reconcile_entry({"job_id": "501", "state": "running"}, "slurm_completed", {"scientific": "valid_continuation_required"}, task_id="rep", slurm_state="COMPLETED")
    assert continued["state"] == "valid_continuation_required"
    retried = reconcile_entry({"job_id": "9", "state": "running"}, "retryable", None, task_id="rep", slurm_state="TIMEOUT")
    assert retried["state"] == "retryable_failure"
    closed = reconcile_entry({"job_id": "9", "state": "running", "attempts": 2}, "retryable", None, task_id="rep", slurm_state="NODE_FAIL")
    assert closed["state"] == "terminal_failure"
    assert interpret_slurm("9", {}, None) == "unknown"
    bare = reconcile_entry({"job_id": "9", "state": "running"}, "slurm_completed", None, task_id="rep", slurm_state="COMPLETED")
    assert bare["reason"] == "slurm_completed_without_scientific_record"
    unfinished = reconcile_entry({}, "slurm_completed", {"scientific": "training_completed"}, task_id="rep", slurm_state="COMPLETED")
    assert unfinished["reason"] == "training_completion_missing_selection_or_checkpoint"
    thin = reconcile_entry({}, "slurm_completed", {"scientific": "inference_completed"}, task_id="rep", slurm_state="COMPLETED")
    assert thin["reason"] == "inference_completion_missing_prediction_content"
    oom = reconcile_entry({"job_id": "9", "state": "running"}, "terminal", None, task_id="rep", group="g", slurm_state="OUT_OF_MEMORY")
    grouped = {}
    merge_group_block(grouped, oom)
    assert grouped["_groups"]["g"]["blocked"] == "OUT_OF_MEMORY"
    finished = {"rep": {"state": "training_completed"}, "later": {"state": "training_completed"}, "adapt": {"state": "inference_completed"}}
    assert programme_remaining(tasks, finished) is False
    held = [{"task_id": "verifier", "family": "verifier_adaptation", "depends_on": [], "statuses": [], "representative": True, "representative_group": "verifier"}]
    assert controller_decisions(held, {}, 0, 0, {})[0]["reason"] == "held_pending_independent_evidence"
    assert programme_remaining(held, {}) is False
    assert controller_requeue("90", True, finished=True) is None
    held = controller_transaction(tmp_path / "state.json")
    with held:
        with pytest.raises(RuntimeError, match="Another controller"):
            with controller_transaction(tmp_path / "state.json"):
                pass
    present = tmp_path / "input.jsonl"
    present.write_text("{}\n", encoding="utf-8")
    biored = probe_readiness({"dataset": "biored", "family": "analysis", "input": {"path": str(present), "fingerprint": "abc", "kind": "contexts"}})
    drugprot = probe_readiness({"dataset": "drugprot", "family": "analysis", "input": {"path": str(tmp_path / "missing"), "fingerprint": "abc", "kind": "contexts"}})
    assert biored["inputs:biored:contexts"]["ready"] is True
    assert drugprot["inputs:drugprot:contexts"]["ready"] is False
    specific = [
        {"task_id": "bio", "family": "analysis", "depends_on": ["inputs:biored:contexts"], "statuses": [], "representative": True, "representative_group": "bio"},
        {"task_id": "drug", "family": "analysis", "depends_on": ["inputs:drugprot:contexts"], "statuses": [], "representative": True, "representative_group": "drug"},
    ]
    chosen = controller_decisions(specific, {}, 0, 0, {**biored, **drugprot})
    assert chosen[0]["action"] == "submit"
    assert chosen[1]["reason"] == "prerequisite"
    assert validate_request_shards(tmp_path / "shards")["ready"] is False
    box = {}
    barrier = threading.Barrier(2)
    scores = []

    def _rank(rank: int) -> None:
        scores.append(coordinate_rank_evaluation(rank, lambda: 0.5, barrier.wait, box))

    threads = [threading.Thread(target=_rank, args=(rank,)) for rank in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert scores == [0.5, 0.5]
    assert backend_specification({"family": "self_consistency", "model": "qwen8b"})["samples"] == 5
    assert backend_specification({"family": "constrained", "model": "qwen8b"})["structured_outputs"] is True
    resumed = _run_generation({
        "family": "zero_shot",
        "requests": [
            {"request_id": "keep", "candidate_id": "c", "prompt": "Read the passage.", "model": "qwen8b"},
            {"request_id": "again", "candidate_id": "d", "prompt": "Read the other passage.", "model": "qwen8b"},
        ],
        "backend_name": "fixture",
        "fixture_responses": {"keep": {"raw": '{"labels": []}', "labels": []}, "again": {"raw": "not json", "labels": None}},
    }, tmp_path / "resume")
    assert resumed["completed"] is True and resumed["engine_loads"] == 1
    stored = {row["request_id"]: row for row in (json.loads(line) for line in (tmp_path / "resume" / "predictions.jsonl").read_text(encoding="utf-8").splitlines())}
    assert stored["keep"]["raw"] == '{"labels": []}' and stored["keep"]["candidate_id"] == "c"
    assert "predicted_labels" in stored["keep"]
    second_pass = _run_generation({
        "family": "zero_shot",
        "shard_directory": str(tmp_path / "resume" / "shards"),
        "backend_name": "fixture",
        "fixture_responses": {},
    }, tmp_path / "resume")
    assert second_pass["completed"] is True and second_pass.get("resumed", 0) == 0
    retained = {row["request_id"]: row for row in (json.loads(line) for line in (tmp_path / "resume" / "predictions.jsonl").read_text(encoding="utf-8").splitlines())}
    assert retained["keep"]["raw"] == '{"labels": []}' and "predicted_labels" in retained["again"]
    checked = publish_validation(tmp_path / "resume" / "shards")
    assert checked["ready"] is True
    assert probe_readiness({
        "family": "zero_shot", "dataset": "biored", "entity": "E0", "predicate": "P9",
        "input": {"path": str(present), "fingerprint": "abc", "kind": "contexts"},
    })["shards:zero_shot:biored:E0:P9"]["ready"] is False
    adapter = tmp_path / "best_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": "rev"}) + "\n", encoding="utf-8")
    (adapter / "adapter_model.bin").write_bytes(b"weights")
    assert verify_peft_adapter(adapter, "rev")["ready"] is True
    assert verify_peft_adapter(adapter, "other")["reason"] == "base_model_mismatch"


def _rank_worker(rank: int, world: int, path: str, identifiers: list[str]) -> None:
    from schema_study.production_runtime import rank_partitions, resume_rank_examples

    partition = rank_partitions(identifiers, world=world, micro_batch=1, seed=13, epoch=1, shuffle=True)
    assigned = partition["ranks"][rank]
    Path(path).write_text(json.dumps({"assigned": assigned, "resumed": resume_rank_examples(assigned, 3), "dropped": partition["dropped"]}), encoding="utf-8")


def test_device_partition_selection_and_controller(tmp_path: Path):
    from schema_study.production_dispatch import _run_generation
    from schema_study.production_runtime import (
        controller_decisions,
        load_controller_state,
        positive_micro_f1,
        rank_partitions,
        record_submission,
        require_production_device,
        resume_rank_examples,
        save_controller_state,
        select_population,
    )

    with pytest.raises(RuntimeError, match="CPU"):
        require_production_device("single_gpu", False, 1)
    assert require_production_device("fsdp", True, 2)["moved_unsharded_before_wrap"] is False
    assert require_production_device("ddp", True, 2)["replicates_full_parameters"] is True
    identifiers = [f"example-{index}" for index in range(36)]
    curriculum = rank_partitions(identifiers, world=2, micro_batch=1, seed=13, epoch=0, shuffle=False)
    assert curriculum["kept"] == identifiers[:32]
    assert set(curriculum["ranks"][0]).isdisjoint(curriculum["ranks"][1])
    assert len(curriculum["ranks"][0]) + len(curriculum["ranks"][1]) == 32
    rebuilt = []
    for start in range(0, 32, 16):
        window = curriculum["kept"][start:start + 16]
        assert len(window) == len(set(window)) == 16
        rebuilt.extend(window)
    assert rebuilt == identifiers[:32]
    continued = resume_rank_examples(curriculum["ranks"][0], 4)
    assert continued == curriculum["ranks"][0][4:]
    assert positive_micro_f1([{"gold": ["NO_RELATION", "Association"], "predicted": ["NO_RELATION"], "status": "valid"}])["f1"] == 0.0
    rows = [
        {"candidate_id": "train", "entity": "E2", "predicate": "P3", "split": "train", "document_id": "d", "assistant": {"labels": []}},
        {"candidate_id": "dev", "entity": "E2", "predicate": "P3", "split": "dev", "document_id": "d", "assistant": {"labels": ["Association"]}},
        {"candidate_id": "test", "entity": "E2", "predicate": "P3", "split": "test", "document_id": "d", "assistant": {"labels": ["Association"]}},
    ]
    for index in range(16):
        rows.append({"candidate_id": f"more-{index}", "entity": "E2", "predicate": "P3", "split": "train", "document_id": f"doc{index}", "assistant": {"labels": []}})
    selected = select_population({"family": "data_efficiency", "entity": "E2", "predicate": "P3", "fraction": 100, "seed": 13}, rows)
    assert [row["candidate_id"] for row in selected["development"]] == ["dev"]
    assert [row["candidate_id"] for row in selected["final_evaluation"]] == ["test"]
    tasks = [
        {"task_id": "rep", "family": "principal_training", "model": "qwen8b", "resources": {"gpus": 1}, "depends_on": ["approval"], "statuses": []},
        {"task_id": "later", "family": "principal_training", "model": "qwen8b", "resources": {"gpus": 1}, "depends_on": ["approval"], "statuses": []},
        {"task_id": "score", "family": "analysis", "depends_on": ["rep"], "statuses": []},
    ]
    first = controller_decisions(tasks, {}, 0, 0, {"approval"})
    assert [row["action"] for row in first] == ["submit", "skip", "skip"]
    state = record_submission({}, "rep", "100", "abc")
    with pytest.raises(RuntimeError, match="already"):
        record_submission(state, "rep", "101", "abc")
    state["rep"]["state"] = "valid_continuation_required"
    waiting = controller_decisions(tasks, state, 0, 0, {"approval"})
    assert waiting[2]["action"] == "skip"
    state["rep"]["state"] = "completed"
    released = controller_decisions(tasks, state, 0, 0, {"approval"})
    assert released[1]["action"] == "submit"
    assert released[2]["action"] == "submit"
    deferred = controller_decisions(tasks, {"rep": {"state": "completed"}}, 512, 0, {"approval"})
    assert deferred[1]["action"] == "defer"
    save_controller_state(tmp_path / "state.json", state)
    assert load_controller_state(tmp_path / "state.json")["rep"]["job_id"] == "100"
    generated = _run_generation({
        "family": "constrained",
        "requests": [{"request_id": "r", "candidate_id": "c", "prompt": "Read the passage.", "model": "qwen8b"}],
        "backend_name": "fixture",
        "fixture_responses": {"r": {"raw": '{"labels": []}', "labels": []}},
    }, tmp_path / "generate")
    assert generated["completed"] is True
    assert generated["status"] == "inference_completed"
    assert "Read the passage." in (tmp_path / "generate" / "predictions.jsonl").read_text(encoding="utf-8")
    assert generated["backend"]["structured_outputs"] is True
    missing = _run_generation({"family": "zero_shot", "requests": []}, tmp_path / "empty")
    assert missing["completed"] is False
    from schema_study.production_dispatch import execute_task

    waiting = execute_task({"executor": "schema_study.analysis_plan:analyse_records", "family": "analysis", "task_id": "analysis", "documents": []}, tmp_path / "analysis", gpu_permitted=False)
    assert waiting["completed"] is False and waiting["status"] == "waiting_for_predictions"
    import multiprocessing

    identifiers = [f"rank-example-{index}" for index in range(40)]
    context = multiprocessing.get_context("fork")
    processes = []
    for rank in range(2):
        process = context.Process(target=_rank_worker, args=(rank, 2, str(tmp_path / f"rank-{rank}.json"), identifiers))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        assert process.exitcode == 0
    left = json.loads((tmp_path / "rank-0.json").read_text(encoding="utf-8"))
    right = json.loads((tmp_path / "rank-1.json").read_text(encoding="utf-8"))
    assert set(left["assigned"]).isdisjoint(right["assigned"])
    assert len(left["assigned"]) + len(right["assigned"]) == 32
    assert left["resumed"] == left["assigned"][3:]
    from schema_study.production_runtime import record_failure

    failed = record_failure({}, "later", "filesystem", group="principal_training|qwen8b|1")
    assert failed["later"]["state"] == "retryable_failure"
    closed = record_failure(failed, "later", "filesystem", systematic=True, group="principal_training|qwen8b|1")
    tasks.append({"task_id": "sibling", "family": "principal_training", "model": "qwen8b", "resources": {"gpus": 1}, "depends_on": ["approval"], "statuses": [], "representative": False, "representative_group": "principal_training|qwen8b|1"})
    blocked = {row["task_id"]: row for row in controller_decisions(tasks, closed, 0, 0, {"approval"})}
    assert blocked["sibling"]["reason"] == "systematic_group_failure"
