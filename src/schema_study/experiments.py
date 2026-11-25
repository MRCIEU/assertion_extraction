"""Experiment inventory, seed streams, and blocked model entry points.

The command-line entry points call these functions with model execution
disabled. Nothing in this module imports a training framework.
"""

from __future__ import annotations

import random
from hashlib import sha256
from typing import Mapping

from schema_study.records import write_tsv

TRAIN_SEEDS = (13, 42, 2026)
BIORED_ENTITY_LEVELS = ("E0", "E1", "E2")
PREDICATE_LEVELS = ("P1", "P2", "P3", "P4")
BIOMEDBERT_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
BIOMEDBERT_REVISION = "e1354b7a3a09615f6aba48dfad4b7a613eef7062"
QWEN8_ID = "Qwen/Qwen3-8B"
QWEN8_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
QWEN32_ID = "Qwen/Qwen3-32B"
QWEN32_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"

TRAINING_PROCEDURE = (
    "load_frozen_tokenizer_without_encoder_weights",
    "initialize_encoder_from_the_recorded_revision",
    "initialize_marker_embeddings_from_the_encoder_seed",
    "initialize_the_classification_head_from_a_separate_seed",
    "shuffle_training_candidates_with_the_data_order_seed",
    "optimize_with_the_recorded_AdamW_schedule",
    "score_each_dev_epoch_without_updating_on_dev",
    "keep_best_and_last_checkpoints_only",
)

MODEL_EXECUTION_ENABLED = False

# Modules that may import a model library. They are not part of CPU validation.
# The default command line does not import them, and its train and generate
# commands stay blocked while MODEL_EXECUTION_ENABLED is false.
AUTHORISED_MODEL_EXECUTION_MODULES = (
    "formal_train.py",
    "formal_generate.py",
    "round6_generate.py",
    "backend_generate.py",
    "round5_followup.py",
    "round5_diagnose.py",
    "weight_verify.py",
    "smoke_qwen.py",
    "smoke_biomedbert.py",
    "models/torch_backend.py",
    "models/qwen_backend.py",
    "measurement_cpu.py",
)


class ModelExecutionBlocked(RuntimeError):
    """Raised because this delivery does not run training or model inference."""


def require_model_execution(action: str) -> None:
    if not MODEL_EXECUTION_ENABLED:
        raise ModelExecutionBlocked(
            f"{action} is not run in this delivery. Training, pretrained-weight "
            "loading, GPU jobs, and model API calls remain disabled."
        )


def derive_seed(base_seed: int, stream: str, *parts: str) -> int:
    text = "|".join((str(base_seed), stream, *parts))
    return int(sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def seed_streams(base_seed: int, entity_level: str, predicate_level: str) -> dict:
    """Encoder and example-order streams depend only on the base seed.

    The classification head also depends on the entity and predicate level
    because those heads do not have the same width.
    """
    return {
        "base_seed": base_seed,
        "encoder_init": derive_seed(base_seed, "encoder_init"),
        "marker_embedding_init": derive_seed(base_seed, "encoder_init"),
        "data_order": derive_seed(base_seed, "data_order"),
        "classifier_head": derive_seed(base_seed, "classifier_head", entity_level, predicate_level),
    }


def training_hyperparameters() -> dict:
    return {
        "learning_rate": 2e-5,
        "max_epochs": 10,
        "effective_batch_size": 16,
        "warmup_ratio": 0.1,
        "schedule": "linear_decay_after_warmup",
        "optimizer": "AdamW",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.01,
        "weight_decay_excluded": ["bias", "normalization"],
        "gradient_clip_norm": 1.0,
        "encoder_dropout": "pretrained_config",
        "classifier_dropout": 0.1,
        "class_weights": False,
        "extra_negative_sampling": False,
        "novelty_head": False,
        "biored_loss": "softmax_cross_entropy",
        "drugprot_loss": "bce_with_logits_mean_over_examples_and_labels",
        "selection_biored": "dev_positive_micro_f1_then_earlier_epoch",
        "selection_drugprot": "common_threshold_0.05_to_0.95_step_0.05_then_earlier_epoch_then_higher_threshold",
        "training_precision": "bf16",
        "biomedbert_prediction_precision": "fp32",
        "qwen_generation_precision": "bf16",
        "qwen_quantization": "none",
        "precision_status": "candidate_pending_gpu_smoke_test",
        "checkpoints": ["best", "last"],
    }


def generation_settings() -> dict:
    return {
        "mode": "non_thinking",
        "enable_thinking": False,
        "temperature": 0,
        "top_p": 1,
        "max_new_tokens": 256,
        "do_not_silently_replace_unsupported_parameters": True,
        "published_generation_config_is_not_the_study_setting": True,
        "quantization": "unresolved",
        "output_object": {"candidate_id": "...", "labels": ["positive labels only"]},
        "compatibility_tested": False,
    }


def plan_experiments(controls: list[dict] | None = None) -> list[dict]:
    """Return the main training runs, prompt groups, controls, and P4 rollups."""
    rows = []
    for entity in BIORED_ENTITY_LEVELS:
        for predicate in PREDICATE_LEVELS:
            for seed in TRAIN_SEEDS:
                rows.append(_row("biored", entity, predicate, "direct_train", BIOMEDBERT_ID, seed))
            rows.append(_row("biored", entity, predicate, "direct_prompt", QWEN8_ID, None))
            rows.append(_row("biored", entity, predicate, "direct_prompt", QWEN32_ID, None))
    for predicate in PREDICATE_LEVELS:
        for seed in TRAIN_SEEDS:
            rows.append(_row("drugprot", "native", predicate, "direct_train", BIOMEDBERT_ID, seed))
        rows.append(_row("drugprot", "native", predicate, "direct_prompt", QWEN8_ID, None))
        rows.append(_row("drugprot", "native", predicate, "direct_prompt", QWEN32_ID, None))
    for control in controls or []:
        rows.append(
            _row(
                "biored",
                "E2",
                control["predicate_level"],
                "control_prompt",
                QWEN32_ID,
                None,
                control_id=control["control_id"],
                control_status="draft_pending_confirmation",
            )
        )
    rollups = []
    for row in rows:
        if row["predicate_level"] == "P4" and row["route"] in {"direct_train", "direct_prompt"}:
            for target in ("P3", "P2", "P1"):
                rolled = dict(row)
                rolled["route"] = "rollup"
                rolled["predicate_level"] = target
                rolled["source_predicate_level"] = "P4"
                rolled["source_plan_id"] = row["plan_id"]
                rolled["plan_id"] = f"{row['plan_id']}__rollup_{target}"
                rolled["requires_model_execution"] = False
                rolled["execution_status"] = "planned"
                rolled["execution_authorized"] = False
                rolled["executes_model"] = False
                rollups.append(rolled)
    return rows + rollups


def _row(dataset, entity, predicate, route, model_id, seed, control_id="", control_status="") -> dict:
    plan_id = f"{dataset}|{entity}|{predicate}|{route}|{model_id}|{seed or 'noseed'}|{control_id}"
    return {
        "plan_id": plan_id,
        "dataset": dataset,
        "entity_level": entity,
        "predicate_level": predicate,
        "route": route,
        "model_id": model_id,
        "model_revision_status": (
            "accepted_for_input_tokenization"
            if "BiomedBERT" in model_id
            else "proposed_not_frozen"
        ),
        "seed": seed if seed is not None else "",
        "control_id": control_id,
        "control_status": control_status,
        "source_predicate_level": "",
        "source_plan_id": "",
        "requires_model_execution": route in {"direct_train", "direct_prompt", "control_prompt"},
        "execution_status": "planned",
        "execution_authorized": False,
        "executes_model": route in {"direct_train", "direct_prompt", "control_prompt"},
        "status": "planned",
    }


def write_experiment_table(path, rows: list[dict]) -> None:
    columns = [
        "plan_id",
        "dataset",
        "entity_level",
        "predicate_level",
        "route",
        "model_id",
        "model_revision_status",
        "seed",
        "control_id",
        "control_status",
        "source_predicate_level",
        "source_plan_id",
        "requires_model_execution",
        "execution_status",
        "execution_authorized",
        "executes_model",
        "status",
    ]
    write_tsv(path, rows, columns)


def plan_counts(rows: list[Mapping]) -> dict:
    def count(**wanted) -> int:
        return sum(1 for row in rows if all(row.get(key) == value for key, value in wanted.items()))
    return {
        "biored_biomedbert_train": count(dataset="biored", route="direct_train", model_id=BIOMEDBERT_ID),
        "drugprot_biomedbert_train": count(dataset="drugprot", route="direct_train", model_id=BIOMEDBERT_ID),
        "main_finetunes": count(route="direct_train"),
        "qwen8_biored": count(dataset="biored", route="direct_prompt", model_id=QWEN8_ID),
        "qwen32_biored": count(dataset="biored", route="direct_prompt", model_id=QWEN32_ID),
        "qwen8_drugprot": count(dataset="drugprot", route="direct_prompt", model_id=QWEN8_ID),
        "qwen32_drugprot": count(dataset="drugprot", route="direct_prompt", model_id=QWEN32_ID),
        "controls": count(route="control_prompt"),
        "rollups": count(route="rollup"),
    }


def blocked_train(config: Mapping) -> dict:
    require_model_execution("training")
    return {"status": "not_run", "config": config}


def blocked_predict(config: Mapping) -> dict:
    require_model_execution("encoder inference")
    return {"status": "not_run", "config": config}


def blocked_generate(config: Mapping) -> dict:
    require_model_execution("prompted generation")
    return {"status": "not_run", "config": config}


def example_order(candidate_ids: list[str], base_seed: int) -> list[str]:
    """The training order for one seed. It does not depend on the predicate level."""
    ordered = sorted(candidate_ids)
    random.Random(derive_seed(base_seed, "data_order")).shuffle(ordered)
    return ordered


def downsampling_plan(
    fine_counts: Mapping[str, int],
    relations: list[dict],
    *,
    coarse_of,
    seed: int = 42,
) -> dict:
    """Reserve three fixed training lists. This does not launch an experiment.

    `coarse_level` is left undecided: the cap is the floor of the mean fine-class
    count, and each positive coarse class is limited to that cap once a level
    is supplied by the caller.
    """
    if len(fine_counts) != 8:
        raise ValueError("The reserved cap is defined from BioRED's eight fine classes.")
    cap = int(sum(fine_counts.values()) // len(fine_counts))
    by_coarse = {}
    for relation in relations:
        by_coarse.setdefault(coarse_of(relation["fine_id"]), []).append(relation["relation_id"])
    rng = random.Random(seed)
    capped = []
    for coarse, ids in sorted(by_coarse.items()):
        ordered = sorted(ids)
        rng.shuffle(ordered)
        capped.extend(ordered[:cap])
    target_n = len(capped)
    proportional = _proportional_ids(by_coarse, target_n, random.Random(seed + 1))
    return {
        "status": "reserved_not_launched",
        "coarse_level": "undecided_until_after_main_results",
        "cap": cap,
        "full_positive_ids": sorted(relation["relation_id"] for relation in relations),
        "capped_positive_ids": sorted(capped),
        "proportional_positive_ids": proportional,
        "no_relation": "unchanged",
        "update_budget": "matched_to_full_data",
        "causal_claim": False,
    }


def _proportional_ids(by_coarse: dict[str, list[str]], target_n: int, rng: random.Random) -> list[str]:
    population = [relation_id for _coarse, ids in sorted(by_coarse.items()) for relation_id in sorted(ids)]
    if target_n >= len(population):
        return population
    total = sum(len(ids) for ids in by_coarse.values())
    quotas = {coarse: int(target_n * len(ids) / total) for coarse, ids in by_coarse.items()}
    remainder = target_n - sum(quotas.values())
    for coarse, _ids in sorted(by_coarse.items(), key=lambda item: (-len(item[1]), item[0]))[:remainder]:
        quotas[coarse] += 1
    chosen = []
    for coarse, ids in sorted(by_coarse.items()):
        ordered = ids[:]
        rng.shuffle(ordered)
        chosen.extend(ordered[:quotas[coarse]])
    return sorted(chosen)
