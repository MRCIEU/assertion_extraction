"""Paired contrasts for the formal analysis. These functions do not read study results."""

from __future__ import annotations


def entity_predicate_interaction(values: dict[tuple[str, str], float], entity_fine: str, entity_coarse: str, predicate_fine: str, predicate_coarse: str) -> float:
    """Extra entity information at the fine predicate, minus the same contrast at the coarse predicate."""
    fine = values[(entity_fine, predicate_fine)] - values[(entity_coarse, predicate_fine)]
    coarse = values[(entity_fine, predicate_coarse)] - values[(entity_coarse, predicate_coarse)]
    return fine - coarse


def population_roles() -> dict:
    return {
        "eligible_candidates": "complete-population extraction and recovery",
        "annotated_positive_relations": "positive-relation recovery",
        "identified_findings": "source-level semantic outcomes, once findings exist",
        "predicted_positive_assertions": "support and contradiction among produced assertions",
        "changed_representation": "effect where the displayed representation changes",
        "unchanged_representation": "effect where the displayed representation does not change",
        "unassessable": "kept separate from insufficient support and from contradiction",
        "invalid_or_missing_output": "kept in the extraction denominator and not given a semantic score",
    }
