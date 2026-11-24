"""Information in the frozen entity and predicate levels.

The calculations describe the existing E0–E2 and P1–P4 representations.
They do not add a category, change a split, or read a model weight.
Entropy is a property of the label distribution. It is not a measure of
scientific meaning in the source text.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping, Sequence

from schema_study.evaluation import information_content, information_retention, smoothed_fine_probabilities
from schema_study.inputs import mapped_type
from schema_study.schemas import LEVELS, Schema, group_direction, map_fine_ids, partition_errors, rollup_ids

# Human readings of the frozen group names against the BioRED guideline and the
# ChemProt guideline. They do not alter membership.
SEMANTIC_GROUP_NOTES = {
    "biored": {
        "BR_P1_ANY": "Any_Relation collects every positive native label and states no particular mechanism.",
        "BR_P2_OTHER": "Other_Relation places Association, Bind, and four chemical relations in one bin. In the BioRED guideline, Bind is a physical interaction and Association is the residual unspecified relation.",
        "BR_P3_REMAINING": "Remaining_Relation collects Comparison, Cotreatment, Drug_Interaction, and Conversion. The guideline defines these as different chemical relations.",
        "BR_P3_ASSOCIATION": "Association keeps its released name. In the guideline it is also the residual bin for relations that are not given a more specific type.",
        "BR_P3_BIND": "Bind keeps the guideline's physical-interaction category. A binding that is stated to cause a correlation is annotated as that correlation instead.",
    },
    "drugprot": {
        "DP_P1_ANY": "Any_Relation collects every positive DrugProt label and states no particular mechanism.",
        "DP_P2_OTHER": "Other_Relation collects agonist labels, antagonism, unspecified direct regulation, part-of, and the metabolic roles. These are not one biological mechanism.",
        "DP_P3_UP": "ACTIVATOR and INDIRECT-UPREGULATOR are both explicit increases in the ChemProt guideline. They differ in whether the chemical acts directly on the gene product.",
        "DP_P3_DOWN": "INHIBITOR and INDIRECT-DOWNREGULATOR are both explicit decreases in the guideline. ANTAGONIST is left outside this group.",
        "DP_P3_AGONISM": "The ChemProt guideline defines AGONIST-ACTIVATOR as increasing the receptor response and AGONIST-INHIBITOR as decreasing it. The shared Agonism label therefore mixes opposite guideline directions. The schema stores all three member directions as nonpolar.",
        "DP_P3_METABOLIC": "SUBSTRATE, PRODUCT-OF, and SUBSTRATE_PRODUCT-OF are distinct enzymatic roles in the guideline. Metabolic_relation names that family.",
        "DP_P3_DIRECT": "DIRECT-REGULATOR remains one released label: a direct binder whose effect on activity is unspecified.",
        "DP_P3_ANTAGONISM": "ANTAGONIST remains one released label. The guideline defines it as reducing the action of another chemical, commonly at the same receptor.",
        "DP_P3_PART": "PART-OF remains one released label for a chemical that forms part of a gene product.",
    },
}

ORIGINAL_BIORED_TYPES = (
    "GeneOrGeneProduct",
    "DiseaseOrPhenotypicFeature",
    "ChemicalEntity",
    "SequenceVariant",
    "OrganismTaxon",
    "CellLine",
)
E1_MERGE = {
    "GeneOrGeneProduct": "Molecular_Entity",
    "ChemicalEntity": "Molecular_Entity",
    "SequenceVariant": "Molecular_Entity",
}


def nats_to_bits(value: float) -> float:
    """Convert a natural-log information quantity into bits."""
    return value / math.log(2)


def structural_hierarchy_errors(schema: Schema) -> list[str]:
    """Automatic checks that the positive levels are nested partitions.

    A semantic objection to a group name is not an error here.
    """
    errors = partition_errors(schema)
    for fine_id in schema.fine_ids:
        finer_members: frozenset[str] | None = None
        for index, level in enumerate(LEVELS):
            direct = map_fine_ids(schema, {fine_id}, level)
            if len(direct) != 1:
                errors.append(f"{fine_id} has {len(direct)} groups at {level}")
                continue
            members = frozenset(schema.by_id(level)[next(iter(direct))].members)
            if finer_members is not None and not finer_members <= members:
                errors.append(f"{fine_id} at {level} does not contain the finer member set")
            if schema.no_relation_id in members:
                errors.append(f"{level} includes No_Relation")
            finer_members = members
            current = {fine_id}
            for source in LEVELS[: index + 1]:
                if source == "P4":
                    current = map_fine_ids(schema, current, source)
                else:
                    current = rollup_ids(schema, LEVELS[LEVELS.index(source) - 1], current, source)
            if current != direct:
                errors.append(f"{fine_id}: path to {level} is {sorted(current)}, direct mapping is {sorted(direct)}")
    return errors


def mixed_schema_direction_groups(schema: Schema) -> list[dict]:
    """Groups whose stored member directions are not all the same.

    This uses the direction field in the schema. It does not read a guideline.
    """
    mixed = []
    for level in LEVELS:
        for label in schema.level_labels(level):
            directions = sorted({schema.directions[member] for member in label.members})
            if len(directions) > 1:
                mixed.append(
                    {
                        "level": level,
                        "group_id": label.stable_id,
                        "display_name": label.display_name,
                        "member_directions": directions,
                        "assigned_group_direction": group_direction(schema, level, label.stable_id),
                    }
                )
    return mixed


def _positive_counts(schema: Schema, counts: Mapping[str, int]) -> dict[str, int]:
    unknown = sorted(set(counts) - set(schema.fine_ids))
    if unknown:
        raise KeyError(f"Counts contain labels outside {schema.schema_id}: {unknown}")
    return {fine_id: int(counts.get(fine_id, 0)) for fine_id in schema.fine_ids}


def shannon_entropy_nats(probabilities: Iterable[float]) -> float:
    """Shannon entropy from the study's natural-log information content."""
    total = 0.0
    mass = 0.0
    for probability in probabilities:
        if probability < 0.0 or probability > 1.0 + 1e-9:
            raise ValueError(f"Probability outside [0, 1]: {probability}")
        usable = min(float(probability), 1.0)
        mass += usable
        if usable > 0.0:
            total += usable * information_content(usable)
    if abs(mass - 1.0) > 1e-8:
        raise ValueError(f"Probabilities sum to {mass}, not 1")
    return total


def _group_for(schema: Schema, fine_id: str, level: str):
    mapped = map_fine_ids(schema, {fine_id}, level)
    if len(mapped) != 1:
        raise ValueError(f"{fine_id} does not have one group at {level}")
    return schema.by_id(level)[next(iter(mapped))]


def predicate_information(
    schema: Schema,
    counts: Mapping[str, int],
    observation_counts: Mapping[str, int] | None = None,
) -> dict:
    """Entropy of the native labels and of each deterministic grouping.

    Native probabilities use add-one smoothing on `counts`. A broader category
    takes the sum of its member probabilities. Transition counts use
    `observation_counts` when a second corpus is being described with the same
    reference distribution. No_Relation is outside this distribution.
    """
    positive_counts = _positive_counts(schema, counts)
    observed_counts = positive_counts if observation_counts is None else _positive_counts(schema, observation_counts)
    probabilities = smoothed_fine_probabilities(positive_counts, schema.fine_ids)
    native_entropy = shannon_entropy_nats(probabilities.values())
    n_reference = sum(positive_counts.values())
    n_observed = sum(observed_counts.values())
    levels = {}
    label_rows = []
    for level in LEVELS:
        labels = schema.level_labels(level)
        group_probabilities = [sum(probabilities[member] for member in label.members) for label in labels]
        entropy = shannon_entropy_nats(group_probabilities)
        levels[level] = {
            "n_categories": len(labels),
            "entropy_nats": entropy,
            "entropy_bits": nats_to_bits(entropy),
            "retained_fraction": None if native_entropy == 0.0 else entropy / native_entropy,
            "categories": [
                {
                    "group_id": label.stable_id,
                    "display_name": label.display_name,
                    "n_native_members": len(label.members),
                    "native_ids": list(label.members),
                    "native_names": [schema.fine_names[member] for member in label.members],
                    "reference_count": sum(positive_counts[member] for member in label.members),
                    "observation_count": sum(observed_counts[member] for member in label.members),
                    "probability": probability,
                    "information_content_nats": information_content(probability),
                    "information_content_bits": nats_to_bits(information_content(probability)),
                }
                for label, probability in zip(labels, group_probabilities)
            ],
        }
    for fine_id in schema.fine_ids:
        row = {
            "fine_id": fine_id,
            "display_name": schema.fine_names[fine_id],
            "reference_count": positive_counts[fine_id],
            "observation_count": observed_counts[fine_id],
            "probability": probabilities[fine_id],
            "information_content_nats": information_content(probabilities[fine_id]),
            "information_content_bits": nats_to_bits(information_content(probabilities[fine_id])),
            "levels": {},
        }
        for level in LEVELS:
            group = _group_for(schema, fine_id, level)
            probability = sum(probabilities[member] for member in group.members)
            row["levels"][level] = {
                "group_id": group.stable_id,
                "display_name": group.display_name,
                "n_native_members": len(set(group.members)),
                "information_content_nats": information_content(probability),
            }
        label_rows.append(row)
    transitions = []
    for coarse, fine in (("P1", "P2"), ("P2", "P3"), ("P3", "P4")):
        changed = 0
        unchanged = 0
        by_label = []
        for row in label_rows:
            coarse_members = frozenset(_group_for(schema, row["fine_id"], coarse).members)
            fine_members = frozenset(_group_for(schema, row["fine_id"], fine).members)
            if fine_members == coarse_members:
                kind = "unchanged"
                unchanged += row["observation_count"]
            elif fine_members < coarse_members:
                kind = "more_specific"
                changed += row["observation_count"]
            else:
                raise ValueError(f"{row['fine_id']} is not nested from {coarse} to {fine}")
            delta = row["levels"][fine]["information_content_nats"] - row["levels"][coarse]["information_content_nats"]
            by_label.append(
                {
                    "fine_id": row["fine_id"],
                    "display_name": row["display_name"],
                    "observation_count": row["observation_count"],
                    "change": kind,
                    "information_content_change_nats": delta,
                    "information_content_change_bits": nats_to_bits(delta),
                }
            )
        transitions.append(
            {
                "from_level": coarse,
                "to_level": fine,
                "n_more_specific": changed,
                "n_unchanged": unchanged,
                "proportion_more_specific": None if n_observed == 0 else changed / n_observed,
                "proportion_unchanged": None if n_observed == 0 else unchanged / n_observed,
                "by_label": by_label,
            }
        )
    increments = {}
    ordered = ("P1", "P2", "P3", "P4")
    for coarse, fine in zip(ordered, ordered[1:]):
        delta = levels[fine]["entropy_nats"] - levels[coarse]["entropy_nats"]
        increments[f"{coarse}_to_{fine}"] = {"nats": delta, "bits": nats_to_bits(delta)}
    ceilings = _schema_ceilings(schema, positive_counts, observed_counts)
    return {
        "smoothing": "add_one_on_native_labels",
        "logarithm": "natural",
        "bit_conversion": "nats / ln(2)",
        "denominator": "positive gold relation labels in the reference counts; No_Relation excluded",
        "n_reference_relations": n_reference,
        "n_observed_relations": n_observed,
        "native_entropy_nats": native_entropy,
        "native_entropy_bits": nats_to_bits(native_entropy),
        "levels": levels,
        "entropy_introduced_nats": increments,
        "native_labels": label_rows,
        "transitions": transitions,
        "schema_ceiling": ceilings,
    }


def _schema_ceilings(schema: Schema, reference_counts: Mapping[str, int], observation_counts: Mapping[str, int]) -> dict:
    """Schema-induced ceiling of the existing retention score.

    Smoothing uses the reference counts. The average is over the observed
    relations. The prediction-induced term is omitted because this summary
    does not use model output. The definition of the ceiling is unchanged.
    """
    rows = []
    for fine_id, count in observation_counts.items():
        rows.extend({"fine_id": fine_id, "status": "valid", "predicted_groups": []} for _ in range(int(count)))
    ceilings = {}
    for level in LEVELS:
        result = information_retention(rows, schema, level, reference_counts)
        ceilings[level] = {
            "n_gold_relations": result["n_gold_relations"],
            "frequency_ceiling_C": result["frequency"]["ceiling_C"],
            "uniform_ceiling_C": result["uniform"]["ceiling_C"],
            "schema_loss_frequency": result["frequency"]["schema_loss_1_minus_C"],
            "schema_loss_uniform": result["uniform"]["schema_loss_1_minus_C"],
        }
    return ceilings


def displayed_entity_type(entity_type: str, level: str, merge: Mapping[str, str] | None = None) -> str | None:
    """Type token supplied at one frozen entity level. E0 supplies none."""
    if merge is None and level == "E1":
        merge = E1_MERGE
    return mapped_type(entity_type, level, dict(merge or {}))


def entity_type_information(
    counts: Mapping[str, int],
    *,
    native_types: Sequence[str] = ORIGINAL_BIORED_TYPES,
    merge: Mapping[str, str] | None = None,
    unit: str,
) -> dict:
    """Retained entity-type entropy of E1 relative to the native types.

    E0 is recorded as the absence of an explicit type and has no entropy.
    Broader E1 probabilities are sums of the add-one native probabilities.
    Counts outside the native type list are reported and left out of the entropy.
    """
    merge = dict(E1_MERGE if merge is None else merge)
    native = tuple(native_types)
    outside = {entity_type: int(count) for entity_type, count in counts.items() if entity_type not in native and count}
    inside = {entity_type: int(counts.get(entity_type, 0)) for entity_type in native}
    probabilities = smoothed_fine_probabilities(inside, native)
    e1_members: dict[str, list[str]] = {}
    for entity_type in native:
        shown = displayed_entity_type(entity_type, "E1", merge)
        if shown is None:
            raise ValueError(f"E1 produced no type for {entity_type}")
        e1_members.setdefault(shown, []).append(entity_type)
    e1_probabilities = {
        name: sum(probabilities[member] for member in members) for name, members in e1_members.items()
    }
    native_entropy = shannon_entropy_nats(probabilities.values())
    e1_entropy = shannon_entropy_nats(e1_probabilities.values())
    n_inside = sum(inside.values())
    more_specific = sum(inside[entity_type] for entity_type in native if entity_type in merge)
    return {
        "unit": unit,
        "smoothing": "add_one_on_the_six_native_types",
        "logarithm": "natural",
        "e0": "no_explicit_type",
        "n_observations_in_native_types": n_inside,
        "n_observations_outside_native_types": sum(outside.values()),
        "outside_native_types": outside,
        "native_counts": inside,
        "native_probability": probabilities,
        "native_information_content_nats": {
            entity_type: information_content(probability) for entity_type, probability in probabilities.items()
        },
        "e1_probability": e1_probabilities,
        "e1_members": e1_members,
        "entropy_e2_nats": native_entropy,
        "entropy_e2_bits": nats_to_bits(native_entropy),
        "entropy_e1_nats": e1_entropy,
        "entropy_e1_bits": nats_to_bits(e1_entropy),
        "retained_fraction_e1": None if native_entropy == 0.0 else e1_entropy / native_entropy,
        "n_more_specific_at_e2": more_specific,
        "proportion_more_specific_at_e2": None if n_inside == 0 else more_specific / n_inside,
        "merged_at_e1": sorted(merge),
        "unchanged_at_e1": [entity_type for entity_type in native if entity_type not in merge],
    }


def candidate_type_changes(
    pairs: Iterable[tuple[str, str]],
    *,
    merge: Mapping[str, str] | None = None,
    native_types: Sequence[str] = ORIGINAL_BIORED_TYPES,
) -> dict:
    """Count candidates by whether each target type is more specific at E2 than at E1."""
    merge = dict(E1_MERGE if merge is None else merge)
    native = set(native_types)
    summary = Counter()
    argument_specific = [0, 0]
    n_pairs = 0
    outside = 0
    for left, right in pairs:
        n_pairs += 1
        if left not in native or right not in native:
            outside += 1
            summary["outside_native_types"] += 1
            continue
        flags = [left in merge, right in merge]
        argument_specific[0] += int(flags[0])
        argument_specific[1] += int(flags[1])
        summary[{0: "neither", 1: "one", 2: "both"}[sum(flags)]] += 1
    considered = n_pairs - outside
    return {
        "n_candidates": n_pairs,
        "n_outside_native_types": outside,
        "n_considered": considered,
        "argument_1_more_specific": argument_specific[0],
        "argument_2_more_specific": argument_specific[1],
        "neither": summary["neither"],
        "one": summary["one"],
        "both": summary["both"],
        "proportion_argument_1": None if considered == 0 else argument_specific[0] / considered,
        "proportion_argument_2": None if considered == 0 else argument_specific[1] / considered,
        "proportion_neither": None if considered == 0 else summary["neither"] / considered,
        "proportion_one": None if considered == 0 else summary["one"] / considered,
        "proportion_both": None if considered == 0 else summary["both"] / considered,
    }


def compare_identifiers(expected: set[str], observed: Sequence[str]) -> dict:
    """Compare expected identifiers with observed rows, including duplicates."""
    observed_rows = list(observed)
    counts = Counter(observed_rows)
    observed_ids = set(counts)
    duplicate_ids = sorted(identifier for identifier, count in counts.items() if count > 1)
    missing = expected - observed_ids
    unexpected = observed_ids - expected
    return {
        "n_expected": len(expected),
        "n_observed_rows": len(observed_rows),
        "n_unique_observed": len(observed_ids),
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
        "n_duplicate_ids": len(duplicate_ids),
        "missing_examples": sorted(missing)[:5],
        "unexpected_examples": sorted(unexpected)[:5],
        "duplicate_examples": duplicate_ids[:5],
        "identity_complete": not missing and not unexpected and not duplicate_ids,
    }


def label_record_problem(
    schema: Schema,
    level: str,
    status: str,
    predicted: Sequence[str],
    illegal: Sequence[str],
) -> str | None:
    """Return a rule violation, or None when the stored label set is interpretable.

    A rejected or missing answer is interpretable. It remains in the population.
    """
    from schema_study.schemas import resolve_output_label

    if status not in {"valid", "partial_illegal", "rejected", "missing"}:
        return "unknown_status"
    multi = schema.label_cardinality() == "multi"
    resolved = []
    for label in predicted:
        stable_id, error = resolve_output_label(schema, level, label)
        if error is not None or stable_id is None:
            return "predicted_label_not_in_schema"
        resolved.append(stable_id)
    if status == "valid":
        if illegal:
            return "valid_answer_has_illegal_labels"
        if not multi and len(resolved) > 1:
            return "single_label_answer_has_several_labels"
        if len(resolved) != len(set(resolved)):
            return "repeated_predicted_label"
    elif status == "partial_illegal":
        if not multi:
            return "partial_illegal_on_single_label_schema"
        if not illegal:
            return "partial_illegal_without_illegal_label"
    elif predicted:
        return "failed_answer_keeps_predicted_labels"
    return None
