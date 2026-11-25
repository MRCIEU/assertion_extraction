"""Definitions for linguistic certainty, source support, and semantic fidelity.

These functions construct and score records. They do not call a model and they
do not treat an entailment probability as the probability that a biological
claim is true.
"""

from __future__ import annotations

from schema_study.records import digest_mapping

MEASUREMENT_VERSION = "semantic_measures_v1"
CERTAINTY_ESTIMATOR = {
    "id": "pei_jurgens_sentence_level",
    "model_id": "pedropei/sentence-level-certainty",
    "construct": "sentence-level scientific certainty",
    "scale": "raw_score_1_to_6",
    "higher_means": "more_certain",
    "not_the_aspect_level_model": "pedropei/aspect-level-certainty",
    "status": "checkpoint_identified_not_executed",
    "reason": "The analysis environment has transformers and does not have torch, which this checkpoint requires.",
}
GENERAL_NLI = {
    "model_id": "MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-wanli",
    "labels": ["entailment", "neutral", "contradiction"],
    "max_length": 512,
    "status": "checkpoint_selected_not_executed",
    "context_policy": "do_not_truncate_relevant_evidence",
}
SCIFACT_ADAPTATION = {
    "training_authorised": False,
    "base_checkpoint": GENERAL_NLI["model_id"],
    "dataset": "allenai/scifact official release",
    "status": "configuration_only",
}
CERTAINTY_CANDIDATES = (
    {
        "id": "pei_jurgens_2021_overall",
        "construct": "sentence-level scientific certainty",
        "model_id": CERTAINTY_ESTIMATOR["model_id"],
        "scale": CERTAINTY_ESTIMATOR["scale"],
        "status": CERTAINTY_ESTIMATOR["status"],
        "selection_role": "candidate",
    },
    {
        "id": "qwen3_8b_assertion_conditioned",
        "construct": "finding-level epistemic commitment",
        "checkpoint": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "status": "specified_not_executed",
        "selection_role": "candidate",
    },
)
VERIFICATION_CANDIDATES = (
    {
        "id": "scifact_adaptation",
        "base_checkpoint": "MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-wanli",
        "status": "configuration_only",
        "requirement": "No training is authorised. The official SciFact release is the adaptation data.",
    },
    {
        "id": "qwen3_8b_prompted_verifier",
        "checkpoint": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "status": "specified_not_executed",
    },
)


def certainty_record(finding_id: str, value, scale: str, method: str) -> dict:
    """Linguistic certainty belongs to the source finding and stays fixed across extraction conditions."""
    return {
        "construct": "linguistic_certainty",
        "unit": "source_finding",
        "finding_id": finding_id,
        "value": value,
        "scale": scale,
        "method": method,
        "fixed_across_extraction_conditions": True,
        "is_probability_of_biological_truth": False,
    }


_COMPONENT_STATUSES = {"preserved", "validly_abstracted", "omitted", "distorted", "not_assessable"}


def _finite_unit(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")} or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1].")
    return number


def support_distribution(entailment: float | None, contradiction: float | None, neutral: float | None) -> dict:
    """Forward inference from source text to an assertion. Neutral is not contradiction."""
    if entailment is None and contradiction is None and neutral is None:
        return {"status": "unassessable", "entailment": None, "contradiction": None, "neutral": None}
    values = (
        _finite_unit(entailment, "entailment"),
        _finite_unit(contradiction, "contradiction"),
        _finite_unit(neutral, "neutral"),
    )
    if any(value is None for value in values):
        raise ValueError("An assessable distribution needs all three probabilities.")
    total = sum(values)
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Support probabilities must sum to one when the case is assessable.")
    return {
        "status": "assessable",
        "direction": "forward",
        "entailment": values[0],
        "contradiction": values[1],
        "neutral": values[2],
        "is_probability_of_biological_truth": False,
    }


def aggregate_passages(passages: list[dict]) -> dict:
    """Do not treat one supporting passage as cancelling a contradiction elsewhere."""
    if any(row.get("status") == "unassessable" for row in passages):
        return {"status": "unassessable", "reason": "one_or_more_passages_unassessable"}
    entails = [row for row in passages if row.get("entailment", 0) >= row.get("contradiction", 0) and row.get("entailment", 0) >= row.get("neutral", 0)]
    contradicts = [row for row in passages if row.get("contradiction", 0) > row.get("entailment", 0) and row.get("contradiction", 0) >= row.get("neutral", 0)]
    if entails and contradicts:
        return {"status": "conflicting_passages", "n_entailing": len(entails), "n_contradicting": len(contradicts)}
    if contradicts:
        return {"status": "contradicted", "n_passages": len(passages)}
    if entails:
        return {"status": "supported", "n_passages": len(passages)}
    return {"status": "unsupported", "n_passages": len(passages)}


def context_decision(character_length: int, token_length: int | None, limit: int) -> dict:
    """A source that does not fit is unassessable. It is not silently truncated."""
    if token_length is not None and token_length > limit:
        return {"status": "unassessable_context", "action": "not_truncated", "token_length": token_length, "limit": limit}
    return {"status": "within_limit", "action": "score_complete_text", "characters": character_length, "limit": limit}


def reverse_entailment(probability: float | None) -> dict:
    """A candidate proxy for completeness. It is not a proportion of scientific information."""
    return {
        "direction": "reverse",
        "entailment": probability,
        "status": "unassessable" if probability is None else "assessable",
        "literal_information_proportion": False,
    }


def atomic_coverage(components: list[dict]) -> dict:
    """Coverage among assessable components. A valid abstraction is not full preservation."""
    unknown = sorted({row["status"] for row in components if row["status"] not in _COMPONENT_STATUSES})
    if unknown:
        raise ValueError("Unknown component status: " + ", ".join(unknown))
    assessable = [row for row in components if row["status"] != "not_assessable"]
    counts = {name: sum(1 for row in assessable if row["status"] == name) for name in ("preserved", "validly_abstracted", "omitted", "distorted")}
    unassessable = len(components) - len(assessable)
    coverage = None if not assessable else counts["preserved"] / len(assessable)
    return {
        "n_applicable": len(components),
        "n_assessable": len(assessable),
        "n_unassessable": unassessable,
        "counts": counts,
        "preserved_fraction_of_assessable": coverage,
        "unassessable_fraction": None if not components else unassessable / len(components),
        "abstraction_counts_as_full_preservation": False,
        "by_component": preservation_by_component(components),
    }


def preservation_by_component(components: list[dict]) -> dict:
    """Relation and qualifier preservation stay separate from fixed entity identities."""
    grouped: dict[str, list] = {}
    for row in components:
        grouped.setdefault(row.get("component") or "unspecified", []).append(row)
    return {name: atomic_counts(rows) for name, rows in sorted(grouped.items())}


def atomic_counts(components: list[dict]) -> dict:
    assessable = [row for row in components if row["status"] != "not_assessable"]
    preserved = sum(1 for row in assessable if row["status"] == "preserved")
    return {
        "n_applicable": len(components),
        "n_assessable": len(assessable),
        "preserved_fraction_of_assessable": None if not assessable else preserved / len(assessable),
    }


def nli_input(source_text: str, assertion: str, context: str) -> dict:
    """The verifier sees the source and the verbalised assertion, not a dataset label."""
    if context not in {"full_document", "original_model_input", "source_finding"}:
        raise ValueError(context)
    return {
        "model_id": GENERAL_NLI["model_id"],
        "context": context,
        "source_text": source_text,
        "assertion": assertion,
        "truncation": False,
    }


def certainty_input(source_text: str, proposition: str | None, method: str) -> dict:
    """Assertion-conditioned certainty names the proposition. The published estimator scores the sentence."""
    if method == "pei_jurgens_sentence_level":
        return {"method": method, "model_id": CERTAINTY_ESTIMATOR["model_id"], "text": source_text, "scale": "raw_score_1_to_6"}
    if method == "qwen3_8b_assertion_conditioned":
        if not proposition:
            raise ValueError("Assertion-conditioned certainty requires the proposition.")
        return {
            "method": method,
            "model_id": "Qwen/Qwen3-8B",
            "source_text": source_text,
            "proposition": proposition,
            "categories": ["asserted", "hedged", "conditional", "negated", "uncommitted", "abstain"],
            "ordered_axis": False,
            "executed": False,
        }
    raise ValueError(method)


def fidelity(support: dict, completeness: dict) -> dict:
    """Support and completeness stay separate. They are not averaged."""
    return {"support": support, "completeness": completeness, "combined_score": None}


def match_findings(references: list[dict], predictions: list[dict]) -> dict:
    """Match on a specified key. One prediction is used once, in finding-identifier order."""
    used = set()
    pairs = []
    for reference in sorted(references, key=lambda row: row["finding_id"]):
        match = None
        for prediction in predictions:
            if prediction["prediction_id"] in used:
                continue
            if prediction.get("match_key") == reference.get("match_key"):
                match = prediction
                break
        if match is None:
            pairs.append({"finding_id": reference["finding_id"], "prediction_id": None, "status": "missing"})
            continue
        used.add(match["prediction_id"])
        pairs.append({"finding_id": reference["finding_id"], "prediction_id": match["prediction_id"], "status": "matched"})
    return {
        "pairs": pairs,
        "unmatched_predictions": [row["prediction_id"] for row in predictions if row["prediction_id"] not in used],
        "denominator": len(references),
        "recovered": sum(1 for row in pairs if row["status"] == "matched"),
        "rule": "stable_finding_order_same_match_key_one_prediction_once",
        "highest_score_selection": False,
    }


def match_multilabel(gold_labels: list[str], predicted_labels: list[str]) -> dict:
    """Each gold label stays in the denominator. One predicted set does not multiply a finding."""
    gold = list(dict.fromkeys(gold_labels))
    predicted = list(dict.fromkeys(predicted_labels))
    gold_set = set(gold)
    predicted_set = set(predicted)
    return {
        "recovered": [label for label in gold if label in predicted_set],
        "missed": [label for label in gold if label not in predicted_set],
        "spurious": [label for label in predicted if label not in gold_set],
        "denominator": len(gold),
        "one_prediction_recovers_only_matching_labels": True,
    }


def measurement_identifier(record: dict) -> str:
    """Identical scientific inputs share one measurement. Distinct findings do not."""
    fields = (
        "source_version",
        "finding_id",
        "evaluation_text",
        "assertion",
        "predicate_interpretation",
        "verbalisation",
        "direction",
        "method",
        "model_revision",
        "prompt_version",
        "calibration_version",
    )
    missing = [field for field in fields if field not in record]
    if missing:
        raise KeyError(", ".join(missing))
    return digest_mapping({field: record[field] for field in fields})


def loss_decomposition(native_gold, coarse_gold, predicted, *, validated_proportion: bool) -> dict:
    """Separate annotation limits, abstraction, and extraction when the measure is a validated proportion."""
    gaps = {
        "native_gap": None if native_gold is None else 1 - native_gold,
        "schema_gap": None if None in (native_gold, coarse_gold) else native_gold - coarse_gold,
        "model_gap": None if None in (coarse_gold, predicted) else coarse_gold - predicted,
    }
    if validated_proportion and all(value is not None for value in gaps.values()):
        gaps["total"] = gaps["native_gap"] + gaps["schema_gap"] + gaps["model_gap"]
        gaps["interpretation"] = "validated_proportion"
    else:
        gaps["total"] = None
        gaps["interpretation"] = "unvalidated_proxy_or_incomplete"
    gaps["non_negative_required"] = False
    return gaps
