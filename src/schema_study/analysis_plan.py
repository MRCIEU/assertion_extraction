"""Paired contrasts for the formal analysis. These functions do not read study results."""

from __future__ import annotations


def entity_predicate_interaction(values: dict[tuple[str, str], float], entity_fine: str, entity_coarse: str, predicate_fine: str, predicate_coarse: str) -> float:
    """Extra entity information at the fine predicate, minus the same contrast at the coarse predicate."""
    fine = values[(entity_fine, predicate_fine)] - values[(entity_coarse, predicate_fine)]
    coarse = values[(entity_fine, predicate_coarse)] - values[(entity_coarse, predicate_coarse)]
    return fine - coarse


def paired_bootstrap(left: list[float], right: list[float], draws: int = 1000, seed: int = 20261125) -> dict:
    """Document-level paired difference. One draw resamples documents, not training seeds."""
    if len(left) != len(right) or not left:
        raise ValueError("The two series must be paired and non-empty.")
    import random

    generator = random.Random(seed)
    size = len(left)
    observed = sum(a - b for a, b in zip(left, right)) / size
    samples = []
    for _ in range(draws):
        total = 0.0
        for _index in range(size):
            choice = generator.randrange(size)
            total += left[choice] - right[choice]
        samples.append(total / size)
    samples.sort()
    low = samples[int(0.025 * (draws - 1))]
    high = samples[int(0.975 * (draws - 1))]
    return {"estimate": observed, "interval": [low, high], "unit": "document", "seeds_pooled": False}


def primary_contrasts() -> dict:
    """Contrasts fixed before the expanded experiment is run."""
    return {
        "entity": "E2 against E0 at native predicates, on the full population and on the matched changed population",
        "predicate": "P3 against P0 at the native entity representation",
        "interaction": "the entity contrast at P3 minus the entity contrast at P0",
        "direct_versus_abstraction": "a prediction made at the target level against a native prediction mapped to that level",
        "support": "entailment among produced assertions, with contradiction and unassessable kept separate",
        "completeness": "preserved fraction among assessable components, with the intended denominator reported",
        "certainty": "the same source-finding certainty across extraction conditions; it is a moderator, not an outcome of extraction",
        "denominator_quality": "conditional on produced assertions",
        "denominator_recovery": "all eligible findings, once findings exist",
    }


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
