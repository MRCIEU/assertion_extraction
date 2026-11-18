"""Review sampling and empty annotation tables.

The sampler records the frame, the random order, and the inclusion probability.
It does not fill a semantic judgement, and it does not look at model output.
"""

from __future__ import annotations

import random

from schema_study.records import write_json, write_tsv

REVIEW_SEED = 20261115
MAIN_QUOTAS = {"association": 100, "other_positive": 60, "no_relation": 40}
FALLBACK_QUOTAS = {"association": 60, "other_positive": 36, "no_relation": 24}
DEVELOPMENT_QUOTAS = {"association": 25, "other_positive": 15, "no_relation": 10}
STRATA = ("association", "other_positive", "no_relation")
ANSWER_COLUMNS = (
    "package_id",
    "reviewer_role",
    "candidate_id",
    "document_id",
    "existence",
    "specific_relation",
    "polarity",
    "conditionality",
    "uncertainty",
    "unable_to_judge",
    "notes",
)


def stratum_of(gold_labels: list[str], association_name: str = "Association") -> str:
    if not gold_labels:
        return "no_relation"
    if gold_labels == [association_name] or set(gold_labels) == {association_name}:
        return "association"
    return "other_positive"


def stratified_permutation(records: list[dict], *, seed: int = REVIEW_SEED) -> dict[str, list[dict]]:
    """Shuffle each stratum independently after a stable candidate-id sort."""
    rng = random.Random(seed)
    grouped = {name: [] for name in STRATA}
    for record in records:
        grouped[record["stratum"]].append(record)
    ordered = {}
    for name in STRATA:
        items = sorted(grouped[name], key=lambda item: item["candidate_id"])
        rng.shuffle(items)
        ordered[name] = items
    return ordered


def select_prefix(ordered: dict[str, list[dict]], quotas: dict[str, int]) -> dict:
    selected = []
    shortfalls = {}
    probabilities = []
    for name in STRATA:
        items = ordered[name]
        requested = quotas[name]
        available = len(items)
        taken = min(requested, available)
        if available < requested:
            shortfalls[name] = {"requested": requested, "available": available}
        probability = (taken / available) if available else None
        for index, item in enumerate(items):
            probabilities.append(
                {
                    "candidate_id": item["candidate_id"],
                    "document_id": item["document_id"],
                    "stratum": name,
                    "order": index,
                    "included": index < taken,
                    "inclusion_probability": probability,
                }
            )
            if index < taken:
                selected.append(item)
    return {"selected": selected, "shortfalls": shortfalls, "probabilities": probabilities}


def nested_samples(records: list[dict], *, seed: int = REVIEW_SEED) -> dict:
    """The fallback sample is a prefix of the same stratified order."""
    ordered = stratified_permutation(records, seed=seed)
    main = select_prefix(ordered, MAIN_QUOTAS)
    fallback = select_prefix(ordered, FALLBACK_QUOTAS)
    main_ids = {item["candidate_id"] for item in main["selected"]}
    fallback_ids = {item["candidate_id"] for item in fallback["selected"]}
    if not fallback_ids <= main_ids:
        raise RuntimeError("Fallback sample is not nested in the main sample")
    double = double_review_ids(main["selected"], seed)
    return {
        "seed": seed,
        "main": main,
        "fallback": fallback,
        "double_review_ids": double,
        "double_review_fraction": 0.25,
        "decision_time": "before_answers_are_seen",
    }


def development_and_check(records: list[dict], *, seed: int = REVIEW_SEED) -> dict:
    """First quota block develops the rules; the next disjoint block checks them."""
    ordered = stratified_permutation(records, seed=seed)
    first = select_prefix(ordered, DEVELOPMENT_QUOTAS)
    remaining = {}
    for name in STRATA:
        used = DEVELOPMENT_QUOTAS[name]
        remaining[name] = ordered[name][used:]
    second = select_prefix(remaining, DEVELOPMENT_QUOTAS)
    first_ids = {item["candidate_id"] for item in first["selected"]}
    second_ids = {item["candidate_id"] for item in second["selected"]}
    if first_ids & second_ids:
        raise RuntimeError("Development and check samples overlap")
    return {"development": first, "check": second, "seed": seed}


def double_review_ids(selected: list[dict], seed: int) -> list[str]:
    ordered = sorted(selected, key=lambda item: item["candidate_id"])
    random.Random(seed).shuffle(ordered)
    width = int(round(len(ordered) * 0.25))
    return [item["candidate_id"] for item in ordered[:width]]


def write_empty_answers(path, package_id: str, records: list[dict], reviewer_role: str) -> None:
    rows = [
        {
            "package_id": package_id,
            "reviewer_role": reviewer_role,
            "candidate_id": record["candidate_id"],
            "document_id": record["document_id"],
            "existence": "",
            "specific_relation": "",
            "polarity": "",
            "conditionality": "",
            "uncertainty": "",
            "unable_to_judge": "",
            "notes": "",
        }
        for record in records
    ]
    write_tsv(path, rows, ANSWER_COLUMNS)


def write_sampling_bundle(directory, dataset: str, records: list[dict]) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    result = nested_samples(records)
    write_json(
        directory / f"{dataset}_sampling_frame.json",
        {
            "dataset": dataset,
            "seed": REVIEW_SEED,
            "n_records": len(records),
            "strata": {name: sum(record["stratum"] == name for record in records) for name in STRATA},
            "shortfalls_main": result["main"]["shortfalls"],
            "shortfalls_fallback": result["fallback"]["shortfalls"],
            "answers_filled": False,
        },
    )
    write_tsv(
        directory / f"{dataset}_inclusion_probabilities.tsv",
        result["main"]["probabilities"],
        ["candidate_id", "document_id", "stratum", "order", "included", "inclusion_probability"],
    )
    write_tsv(
        directory / f"{dataset}_main_sample.tsv",
        [_public_row(item) for item in result["main"]["selected"]],
        ["candidate_id", "document_id", "stratum"],
    )
    write_tsv(
        directory / f"{dataset}_fallback_prefix.tsv",
        [_public_row(item) for item in result["fallback"]["selected"]],
        ["candidate_id", "document_id", "stratum"],
    )
    write_json(directory / f"{dataset}_double_review_ids.json", {"candidate_ids": result["double_review_ids"], "fraction": 0.25})
    write_empty_answers(directory / f"{dataset}_answers_reviewer_a_empty.tsv", f"{dataset}_main", result["main"]["selected"], "reviewer_a")
    return {
        "n_main": len(result["main"]["selected"]),
        "n_fallback": len(result["fallback"]["selected"]),
        "n_double": len(result["double_review_ids"]),
        "shortfalls": result["main"]["shortfalls"],
    }


def _public_row(item: dict) -> dict:
    return {"candidate_id": item["candidate_id"], "document_id": item["document_id"], "stratum": item["stratum"]}


def blind_package_row(record: dict, text: str, target_1: str, target_2: str) -> dict:
    """First-pass material shows the text and the two targets, not the label or a prediction."""
    return {
        "candidate_id": record["candidate_id"],
        "document_id": record["document_id"],
        "text": text,
        "target_1": target_1,
        "target_2": target_2,
    }
