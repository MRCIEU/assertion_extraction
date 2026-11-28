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


def micro_f1_bootstrap(
    documents: list[dict],
    *,
    left: str,
    right: str,
    draws: int = 2000,
    seed: int = 100,
) -> dict:
    """Resample documents and recompute corpus micro-F1. Do not average document F1."""

    def score(rows: list[dict], key: str) -> float:
        true_positive = sum(row[key]["tp"] for row in rows)
        false_positive = sum(row[key]["fp"] for row in rows)
        false_negative = sum(row[key]["fn"] for row in rows)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    if not documents:
        raise ValueError("The document table is empty.")
    import random

    observed = score(documents, left) - score(documents, right)
    generator = random.Random(seed)
    size = len(documents)
    samples = []
    for _ in range(draws):
        chosen = [documents[generator.randrange(size)] for _index in range(size)]
        samples.append(score(chosen, left) - score(chosen, right))
    samples.sort()
    return {
        "estimate": observed,
        "interval": [samples[int(0.025 * (draws - 1))], samples[int(0.975 * (draws - 1))]],
        "unit": "document",
        "statistic": "corpus_micro_f1_difference",
        "draws": draws,
        "seed": seed,
        "seeds_pooled": False,
        "document_f1_averaged": False,
    }


def analyse_records(documents: list[dict], *, left: str = "system_a", right: str = "system_b") -> dict:
    """Read prediction counts and recompute corpus micro-F1. This is the formal analysis path."""
    contrast = micro_f1_bootstrap(documents, left=left, right=right, draws=50, seed=100)
    return {
        "synthetic": all(row.get("synthetic") for row in documents),
        "contrast": contrast,
        "document_f1_averaged": False,
        "rows": len(documents),
        "catalogue": [item["id"] for item in analysis_catalogue()],
    }


def synthetic_report(documents: list[dict]) -> dict:
    """Controlled records used to test `analyse_records`. They are not study results."""
    marked = [{**row, "synthetic": True} for row in documents]
    report = analyse_records(marked)
    report["synthetic"] = True
    return report


def primary_contrasts() -> dict:
    """Contrasts fixed before the expanded experiment is run."""
    return {
        "entity": "Each entity step from E0 through the proposed maximum, on the full population and on the matched population whose displayed type changes at that step",
        "predicate": "P3 against P0 at the native entity representation",
        "interaction": "the entity contrast at P3 minus the entity contrast at P0",
        "direct_versus_abstraction": "a prediction made at the target level against a native prediction mapped to that level",
        "support": "entailment among produced assertions, with contradiction and unassessable kept separate",
        "completeness": "preserved fraction among assessable components, with the intended denominator reported",
        "certainty": "the same source-finding certainty across extraction conditions; it is a moderator, not an outcome of extraction",
        "denominator_quality": "conditional on produced assertions",
        "denominator_recovery": "all eligible findings, once findings exist",
    }


def analysis_catalogue() -> list[dict]:
    """The comparisons the formal analysis will report. No empirical result is stored here."""
    shared = "Resample documents and recompute corpus micro-F1 from pooled counts. Do not average document-level F1."
    rows = [
        ("predicate_granularity", "P3 against P0 at the same entity representation", shared),
        ("requested_entity_granularity", "Each requested step from E0 through the proposed maximum", shared),
        ("realised_entity_granularity", "The same steps using the realised level where a finer class is unavailable", shared),
        ("matched_changed_representation", "Candidates whose displayed type changes at that step", shared),
        ("unchanged_representation", "Candidates whose displayed type does not change at that step", shared),
        ("entity_predicate_interaction", "Entity contrast at P3 minus the entity contrast at P0", shared),
        ("direct_versus_abstraction", "A direct prediction against a native prediction mapped to the same level", shared),
        ("model_capacity", "Qwen3-32B against Qwen3-8B within one learning paradigm", shared),
        ("zero_shot_versus_few_shot", "Five retrieved demonstrations against the definition-only prompt", shared),
        ("zero_shot_versus_supervised", "Supervised adaptation against the definition-only prompt", shared),
        ("predicate_by_paradigm", "The predicate contrast within each learning paradigm, then their difference", shared),
        ("certainty_moderation", "The same extraction contrast inside source-finding certainty strata", "Certainty stays fixed for a finding and is not an extraction outcome."),
        ("source_support", "Support, contradiction, insufficient evidence, and unassessable kept separate", "No averaged semantic score."),
        ("completeness", "Preserved, abstracted, omitted, distorted, and unassessable components", "A valid abstraction is not full preservation."),
        ("supported_finding_recovery", "Findings recovered with support, over all eligible findings", "Missing findings stay in the denominator."),
        ("information_loss", "Representation-induced gap and extraction-induced gap, when the measure is a proportion", "Reverse-entailment differences are inference-score differences."),
        ("alternative_partitions", "The same contrast under each retained alternative predicate partition", shared),
        ("verbalisation_sensitivity", "The same assertions under an alternative documented wording", shared),
        ("computational_cost", "Wall time and completed outputs per condition", "Cost is not an accuracy outcome."),
    ]
    return [{"id": name, "contrast": contrast, "statistic": statistic} for name, contrast, statistic in rows]


def stratum_difference(high: list[dict], low: list[dict], left: str, right: str, draws: int = 2000, seed: int = 100) -> dict:
    """Difference of two corpus micro-F1 contrasts. The strata are not pooled."""
    upper = micro_f1_bootstrap(high, left=left, right=right, draws=draws, seed=seed)
    lower = micro_f1_bootstrap(low, left=left, right=right, draws=draws, seed=seed)
    return {
        "high_estimate": upper["estimate"],
        "low_estimate": lower["estimate"],
        "difference": upper["estimate"] - lower["estimate"],
        "document_f1_averaged": False,
        "seeds_pooled": False,
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
