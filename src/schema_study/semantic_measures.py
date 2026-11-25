"""Definitions for linguistic certainty, source support, and semantic fidelity.

These functions construct and score records. They do not call a model and they
do not treat an entailment probability as the probability that a biological
claim is true.
"""

from __future__ import annotations

from schema_study.records import digest_mapping

MEASUREMENT_VERSION = "semantic_measures_v1"
CERTAINTY_CANDIDATES = (
    {
        "id": "pei_jurgens_2021_overall",
        "construct": "sentence-level scientific certainty",
        "status": "resource_not_yet_acquired",
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
        "id": "scifact_verifier",
        "status": "checkpoint_not_selected",
        "requirement": "An official SciFact-trained verifier checkpoint and its label map.",
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


def support_distribution(entailment: float | None, contradiction: float | None, neutral: float | None) -> dict:
    """Forward inference from source text to an assertion. Neutral is not contradiction."""
    if entailment is None:
        return {"status": "unassessable", "entailment": None, "contradiction": None, "neutral": None}
    total = entailment + contradiction + neutral
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Support probabilities must sum to one when the case is assessable.")
    return {
        "status": "assessable",
        "direction": "forward",
        "entailment": entailment,
        "contradiction": contradiction,
        "neutral": neutral,
        "is_probability_of_biological_truth": False,
    }


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
        "abstraction_counts_as_full_preservation": False,
    }


def fidelity(support: dict, completeness: dict) -> dict:
    """Support and completeness stay separate. They are not averaged."""
    return {"support": support, "completeness": completeness, "combined_score": None}


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
