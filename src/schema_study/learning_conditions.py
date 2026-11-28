"""Corpus-specific learning conditions. This module does not train or generate.

BioRED crosses the proposed entity levels with P0–P3. BC8 reuses the BioRED
models and demonstration pool. DrugProt stays at its native entity representation.
A training run, an evaluation, and a deterministic abstraction are different units.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from schema_study.experiments import (
    BIOMEDBERT_ID,
    BIOMEDBERT_REVISION,
    QWEN32_ID,
    QWEN32_REVISION,
    QWEN8_ID,
    QWEN8_REVISION,
    TRAIN_SEEDS,
)

RESEARCH_PREDICATES = ("P0", "P1", "P2", "P3")
HISTORICAL_PREDICATES = {"P0": "P1", "P1": "P2", "P2": "P3", "P3": "P4"}
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

SYSTEMS = (
    {"system": "M1", "model": "biomedbert", "model_id": BIOMEDBERT_ID, "revision": BIOMEDBERT_REVISION, "paradigm": "supervised", "adaptation": "full_finetune", "trains": True},
    {"system": "M2", "model": "qwen8b", "model_id": QWEN8_ID, "revision": QWEN8_REVISION, "paradigm": "zero_shot", "adaptation": "none", "trains": False},
    {"system": "M3", "model": "qwen8b", "model_id": QWEN8_ID, "revision": QWEN8_REVISION, "paradigm": "few_shot", "adaptation": "retrieved_demonstrations", "trains": False},
    {"system": "M4", "model": "qwen8b", "model_id": QWEN8_ID, "revision": QWEN8_REVISION, "paradigm": "supervised", "adaptation": "lora_sft", "trains": True},
    {"system": "M5", "model": "qwen32b", "model_id": QWEN32_ID, "revision": QWEN32_REVISION, "paradigm": "zero_shot", "adaptation": "none", "trains": False},
    {"system": "M6", "model": "qwen32b", "model_id": QWEN32_ID, "revision": QWEN32_REVISION, "paradigm": "few_shot", "adaptation": "retrieved_demonstrations", "trains": False},
    {"system": "M7", "model": "qwen32b", "model_id": QWEN32_ID, "revision": QWEN32_REVISION, "paradigm": "supervised", "adaptation": "lora_sft", "trains": True},
)


def lora_configuration() -> dict:
    """Fixed starting settings. They are not a claim that the values are optimal."""
    return {
        "precision": "bf16",
        "quantization": "none",
        "same_adaptation_for_8b_and_32b": True,
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": list(LORA_TARGET_MODULES),
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "schedule": "linear_decay_after_warmup",
        "warmup_ratio": 0.1,
        "max_epochs": 3,
        "seeds": list(TRAIN_SEEDS),
        "selection": "development_positive_micro_f1",
        "loss": "assistant_completion_tokens_only",
        "assistant_includes_candidate_identifier": False,
        "parser_conversion": "The stored candidate identifier is attached before the existing parser. It is not part of the loss.",
        "per_device_batch_size": 1,
        "gradient_accumulation": 16,
        "effective_batch_size": 16,
        "checkpoints": ["best", "last"],
        "trained_here": False,
    }


def schema_conditioned_configuration(training_candidates: int, entity_views: int = 8, predicate_views: int = 4) -> dict:
    """One Qwen3-8B adapter across schema views. It is not part of the principal matrix."""
    views = entity_views * predicate_views
    independent = training_candidates * views
    return {
        "model": "qwen8b",
        "in_principal_matrix": False,
        "adapters": 1,
        "sampling": "Within a document, one entity and predicate view of a candidate is drawn each epoch. The views are not additional findings.",
        "training_candidates": training_candidates,
        "views_per_candidate": views,
        "independent_sft_examples_per_epoch": independent,
        "schema_conditioned_examples_per_epoch": training_candidates,
        "exposure_ratio_independent_over_conditioned": views,
        "token_budget_ratio_per_epoch": views,
        "purpose": "Whether one model can answer under several requested representations.",
        "trained_here": False,
    }


def principal_conditions(entity_levels: list[str]) -> list[dict]:
    """Training runs, direct inference, BC8 evaluation, and DrugProt at its native type."""
    rows = []
    for entity in entity_levels:
        for predicate in RESEARCH_PREDICATES:
            for system in SYSTEMS:
                rows.extend(_biored_family(entity, predicate, system))
    for predicate in RESEARCH_PREDICATES:
        for system in SYSTEMS:
            rows.extend(_drugprot(predicate, system))
    return rows


def _biored_family(entity: str, predicate: str, system: dict) -> list[dict]:
    rows = []
    seeds = list(TRAIN_SEEDS) if system["trains"] else [None]
    for seed in seeds:
        if system["trains"]:
            rows.append(_unit("training_run", "biored", entity, predicate, system, seed, executes="new_training"))
        rows.append(_unit("direct_inference", "biored", entity, predicate, system, seed, executes="model_inference"))
        rows.append(_unit("evaluation", "bc8", entity, predicate, system, seed, executes="model_inference", trained_on="biored"))
        if predicate != "P3":
            rows.append(_unit("deterministic_abstraction", "biored", entity, predicate, system, seed, executes="no_model"))
            rows.append(_unit("deterministic_abstraction", "bc8", entity, predicate, system, seed, executes="no_model", trained_on="biored"))
    return rows


def _drugprot(predicate: str, system: dict) -> list[dict]:
    rows = []
    seeds = list(TRAIN_SEEDS) if system["trains"] else [None]
    for seed in seeds:
        if system["trains"]:
            rows.append(_unit("training_run", "drugprot", "native", predicate, system, seed, executes="new_training"))
        rows.append(_unit("direct_inference", "drugprot", "native", predicate, system, seed, executes="model_inference"))
        rows.append(_unit("evaluation", "drugprot", "native", predicate, system, seed, executes="model_inference"))
        if predicate != "P3":
            rows.append(_unit("deterministic_abstraction", "drugprot", "native", predicate, system, seed, executes="no_model"))
    return rows


def _unit(unit, dataset, entity, predicate, system, seed, executes, trained_on=None) -> dict:
    return {
        "unit": unit,
        "dataset": dataset,
        "entity_level": entity,
        "predicate_level": predicate,
        "historical_predicate_level": HISTORICAL_PREDICATES[predicate],
        "system": system["system"],
        "model": system["model"],
        "model_id": system["model_id"],
        "revision": system["revision"],
        "paradigm": system["paradigm"],
        "adaptation": system["adaptation"],
        "seed": "" if seed is None else str(seed),
        "trained_on": trained_on or dataset,
        "planned_execution": executes,
        "launched": False,
    }


def condition_counts(rows: list[dict]) -> dict:
    counts = Counter(row["unit"] for row in rows)
    training = [row for row in rows if row["unit"] == "training_run"]
    by_model = Counter(row["model"] for row in training)
    return {
        "units": dict(counts),
        "training_runs": len(training),
        "training_runs_by_model": dict(by_model),
        "drugprot_entity_levels": sorted({row["entity_level"] for row in rows if row["dataset"] == "drugprot"}),
        "bc8_training_runs": sum(1 for row in training if row["dataset"] == "bc8"),
    }


def recover_arguments(candidate: dict, mentions: dict) -> dict:
    """Mention text from the candidate record. A candidate identifier is not a name."""
    left = _surface(candidate.get("mention_ids_1"), mentions)
    right = _surface(candidate.get("mention_ids_2"), mentions)
    if left is None or right is None:
        return {"status": "unresolved_entity", "argument_1": None, "argument_2": None}
    if left["text"] == candidate.get("candidate_id") or right["text"] == candidate.get("candidate_id"):
        return {"status": "unresolved_entity", "argument_1": None, "argument_2": None}
    return {
        "status": "recovered",
        "argument_1": left["text"],
        "argument_2": right["text"],
        "argument_1_mention_id": left["mention_id"],
        "argument_2_mention_id": right["mention_id"],
        "argument_1_concept_id": left.get("concept_id"),
        "argument_2_concept_id": right.get("concept_id"),
    }


def _surface(mention_ids, mentions: dict) -> dict | None:
    rows = []
    for mention_id in list(mention_ids or []):
        row = mentions.get(mention_id)
        text = str((row or {}).get("text") or "").strip()
        if row is None or not text:
            continue
        rows.append(row)
    if not rows:
        return None
    rows.sort(key=lambda row: (int(row.get("char_start") or 0), str(row.get("mention_id"))))
    return rows[0]


def sft_example(candidate_id: str, labels: list[str], entity_level: str, predicate_level: str) -> dict:
    """The loss is on the label object. The candidate identifier stays outside that object."""
    return {
        "candidate_id": candidate_id,
        "entity_level": entity_level,
        "predicate_level": predicate_level,
        "loss_tokens": "assistant_only",
        "assistant": {"labels": list(labels)},
        "prompt_includes_source_and_instruction": True,
        "assistant_includes_candidate_identifier": False,
    }


def bm25_index(documents: list[dict]) -> dict:
    """documents: id, document_id, tokens, negative."""
    postings = defaultdict(list)
    lengths = []
    for index, document in enumerate(documents):
        counts = Counter(document["tokens"])
        lengths.append(sum(counts.values()) or 1)
        for term, frequency in counts.items():
            postings[term].append((index, frequency))
    return {
        "documents": documents,
        "postings": postings,
        "lengths": lengths,
        "average_length": sum(lengths) / len(lengths) if lengths else 1.0,
        "n_documents": len(documents),
        "document_frequency": {term: len(rows) for term, rows in postings.items()},
    }


def select_demonstrations(query_tokens: list[str], document_id: str, index: dict, n: int = 5) -> list[str]:
    """Five training candidates. The same query text is used at every granularity."""
    scores = _bm25_scores(query_tokens, index)
    ranked = []
    for doc_index, score in scores.items():
        document = index["documents"][doc_index]
        if document["document_id"] == document_id:
            continue
        ranked.append((score, document["id"], doc_index))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    chosen = ranked[:n]
    if chosen and not any(index["documents"][item[2]]["negative"] for item in chosen):
        negatives = [item for item in ranked if index["documents"][item[2]]["negative"]]
        if negatives:
            chosen = chosen[: n - 1] + [negatives[0]]
    return [item[1] for item in chosen]


def _bm25_scores(query_tokens: list[str], index: dict, k1: float = 1.2, b: float = 0.75) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    n_documents = index["n_documents"] or 1
    average = index["average_length"] or 1.0
    for term in set(query_tokens):
        rows = index["postings"].get(term)
        if not rows:
            continue
        df = index["document_frequency"][term]
        idf = math.log(1 + (n_documents - df + 0.5) / (df + 0.5))
        for doc_index, frequency in rows:
            length = index["lengths"][doc_index]
            denominator = frequency + k1 * (1 - b + b * length / average)
            scores[doc_index] += idf * frequency * (k1 + 1) / denominator
    return scores


def random_demonstrations(document_id: str, pool: list[dict], n: int, seed: int) -> list[str]:
    """A limited sensitivity set. It is not part of the principal factorial."""
    import random

    eligible = sorted((row for row in pool if row["document_id"] != document_id), key=lambda row: row["id"])
    generator = random.Random(seed)
    generator.shuffle(eligible)
    chosen = eligible[:n]
    if chosen and not any(row["negative"] for row in chosen):
        negatives = [row for row in eligible if row["negative"]]
        if negatives:
            chosen = chosen[: n - 1] + [negatives[0]]
    return [row["id"] for row in chosen]


def verify_lora_modules(weight_names: list[str]) -> dict:
    """Confirm the proposed modules occur in a model index. This does not load weights."""
    present = {module: any(module in name for name in weight_names) for module in LORA_TARGET_MODULES}
    return {"present": present, "complete": all(present.values())}
