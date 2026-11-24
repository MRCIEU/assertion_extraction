"""Checks for predicate entropy, entity-type changes, and identifier coverage."""

from __future__ import annotations

from pathlib import Path

from schema_study.granularity import (
    candidate_type_changes,
    compare_identifiers,
    displayed_entity_type,
    entity_type_information,
    label_record_problem,
    predicate_information,
    structural_hierarchy_errors,
)
from schema_study.schemas import load_schema

ROOT = Path(__file__).resolve().parents[1]


def biored():
    return load_schema(ROOT / "configs" / "schemas" / "biored.yaml")


def drugprot():
    return load_schema(ROOT / "configs" / "schemas" / "drugprot.yaml")


def test_existing_hierarchies_are_nested_partitions():
    assert structural_hierarchy_errors(biored()) == []
    assert structural_hierarchy_errors(drugprot()) == []


def test_native_level_keeps_entropy_and_positive_level_keeps_none():
    counts = {fine_id: index + 1 for index, fine_id in enumerate(biored().fine_ids)}
    measured = predicate_information(biored(), counts)
    assert measured["levels"]["P4"]["retained_fraction"] == 1.0
    assert measured["levels"]["P1"]["entropy_nats"] == 0.0
    assert measured["levels"]["P1"]["retained_fraction"] == 0.0
    assert measured["levels"]["P1"]["n_categories"] == 1


def test_broader_probability_is_the_sum_of_add_one_members():
    schema = drugprot()
    counts = {fine_id: 0 for fine_id in schema.fine_ids}
    counts["DP01_ACTIVATOR"] = 3
    counts["DP08_INDIRECT_UPREGULATOR"] = 1
    measured = predicate_information(schema, counts)
    native = {row["fine_id"]: row["probability"] for row in measured["native_labels"]}
    up = next(row for row in measured["levels"]["P3"]["categories"] if row["group_id"] == "DP_P3_UP")
    assert up["probability"] == native["DP01_ACTIVATOR"] + native["DP08_INDIRECT_UPREGULATOR"]
    assert up["n_native_members"] == 2


def test_specificity_uses_member_sets_rather_than_names():
    schema = biored()
    counts = {fine_id: 2 for fine_id in schema.fine_ids}
    measured = predicate_information(schema, counts)
    p3_to_p4 = {row["fine_id"]: row for row in measured["transitions"][2]["by_label"]}
    assert p3_to_p4["BR01_POSITIVE_CORRELATION"]["change"] == "unchanged"
    assert p3_to_p4["BR01_POSITIVE_CORRELATION"]["information_content_change_nats"] == 0.0
    assert p3_to_p4["BR05_COMPARISON"]["change"] == "more_specific"
    assert p3_to_p4["BR05_COMPARISON"]["information_content_change_nats"] > 0.0


def test_observation_counts_do_not_change_the_reference_probability():
    schema = biored()
    reference = {fine_id: 4 for fine_id in schema.fine_ids}
    observed = {fine_id: 0 for fine_id in schema.fine_ids}
    observed["BR04_BIND"] = 5
    measured = predicate_information(schema, reference, observed)
    assert measured["n_reference_relations"] == 32
    assert measured["n_observed_relations"] == 5
    assert measured["transitions"][0]["n_more_specific"] == 5


def test_zero_native_entropy_has_an_undefined_retained_fraction():
    schema = drugprot()
    # A one-label schema is not constructed. The guard is the P1 result above
    # and the explicit None when native entropy is zero, exercised by equal
    # probabilities only through the public ratio. A single observed label still
    # has positive entropy after add-one smoothing across every native label.
    measured = predicate_information(schema, {schema.fine_ids[0]: 10})
    assert measured["native_entropy_nats"] > 0.0
    assert measured["levels"]["P4"]["retained_fraction"] == 1.0


def test_e0_is_absence_and_e1_merges_only_three_types():
    assert displayed_entity_type("GeneOrGeneProduct", "E0") is None
    assert displayed_entity_type("DiseaseOrPhenotypicFeature", "E1") == "DiseaseOrPhenotypicFeature"
    assert displayed_entity_type("GeneOrGeneProduct", "E1") == "Molecular_Entity"
    counts = {entity_type: 10 for entity_type in ("GeneOrGeneProduct", "ChemicalEntity", "SequenceVariant", "DiseaseOrPhenotypicFeature", "OrganismTaxon", "CellLine")}
    measured = entity_type_information(counts, unit="mention")
    assert measured["e0"] == "no_explicit_type"
    assert measured["proportion_more_specific_at_e2"] == 0.5
    assert measured["retained_fraction_e1"] < 1.0
    pairs = candidate_type_changes([("GeneOrGeneProduct", "DiseaseOrPhenotypicFeature"), ("DiseaseOrPhenotypicFeature", "CellLine")])
    assert pairs["one"] == 1
    assert pairs["neither"] == 1
    assert pairs["argument_1_more_specific"] == 1
    assert pairs["argument_2_more_specific"] == 0


def test_identifier_coverage_rejects_a_matching_count_with_the_wrong_ids():
    result = compare_identifiers({"a", "b"}, ["a", "a"])
    assert result["n_observed_rows"] == 2
    assert result["n_missing"] == 1
    assert result["n_duplicate_ids"] == 1
    assert result["identity_complete"] is False


def test_rejected_answer_stays_interpretable_and_a_second_biored_label_does_not():
    schema = biored()
    assert label_record_problem(schema, "P4", "rejected", [], ["not-a-label"]) is None
    assert label_record_problem(schema, "P4", "valid", ["BR01_POSITIVE_CORRELATION", "BR04_BIND"], []) == "single_label_answer_has_several_labels"
    assert label_record_problem(drugprot(), "P4", "partial_illegal", ["DP01_ACTIVATOR"], ["made-up"]) is None
