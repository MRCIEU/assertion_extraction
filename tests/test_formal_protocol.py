"""Checks for classification, verbalisation, integrity, and analysis definitions."""

from __future__ import annotations

from pathlib import Path

import pytest

from schema_study.analysis_plan import entity_predicate_interaction, paired_bootstrap
from schema_study.classification_resources import _ancestor_chain
from schema_study.entity_hierarchy import (
    classify_cell_line,
    classify_concept,
    classify_mesh_identifier,
    classify_variant,
    displayed_type,
    mesh_chain,
    proposed_global_depth,
    realised_classification,
)
from schema_study.predicate_notation import historical_level, load_notation, positive_group_count
from schema_study.records import publish_collection, publish_file
from schema_study.semantic_measures import atomic_coverage, loss_decomposition, measurement_identifier, support_distribution
from schema_study.source_identification import recover_quotation, validate_model_response
from schema_study.source_integrity import collection_complete, inspect_jsonl, population_difference
from schema_study.verbalisation import prompt_label, verbalise

ROOT = Path(__file__).resolve().parents[1]


def test_unique_temporary_files_do_not_replace_a_finished_result(tmp_path: Path):
    destination = tmp_path / "result.json"
    destination.write_text("finished\n", encoding="utf-8")
    seen = []

    def fail(temporary: Path) -> None:
        seen.append(temporary.name)
        temporary.write_text("partial\n", encoding="utf-8")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        publish_file(destination, fail)
    assert destination.read_text(encoding="utf-8") == "finished\n"
    assert seen and seen[0] != "result.json.partial"
    assert list(tmp_path.iterdir()) == [destination]


def test_a_failed_collection_does_not_leave_a_new_completion_record(tmp_path: Path):
    (tmp_path / "completion.json").write_text('{"status": "previous"}\n', encoding="utf-8")

    def fail(temporary: Path) -> None:
        temporary.write_text("x\n", encoding="utf-8")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        publish_collection(tmp_path, {"rows.json": fail}, {"status": "new"})
    assert not (tmp_path / "rows.json").exists()
    assert (tmp_path / "completion.json").read_text(encoding="utf-8") == '{"status": "previous"}\n'


def test_composite_mesh_mapping_requires_every_component():
    descriptors = {
        "D000001": {"name": "A", "tree_numbers": ["D02.1"], "n_pharmacological_actions": 0},
    }
    labels = {"D02": "Organic chemicals"}
    complete = classify_mesh_identifier("D000001", "ChemicalEntity", descriptors, labels)
    assert complete["status"] == "classified"
    assert complete["chain"][0]["label"] == "Organic chemicals"
    incomplete = classify_mesh_identifier("D000001|D000002", "ChemicalEntity", descriptors, labels)
    assert incomplete["status"] == "incomplete_composite"
    assert incomplete["eligible"] is False


def test_terminal_classification_is_not_a_new_distinction():
    chain = [{"node": "D02", "label": "Organic chemicals", "relation": "mesh_tree_ancestor"}]
    carried = realised_classification(chain, 4)
    assert carried["carry_forward"] is True
    assert carried["new_distinction"] is False
    assert carried["label"] == "Organic chemicals"
    assert proposed_global_depth({3: 100, 4: 3}) == 3
    assert proposed_global_depth({4: 200}) == 2


def test_pharmacological_wording_and_non_is_a_links_stay_out_of_the_type():
    blocked = mesh_chain(["D02.1"], "ChemicalEntity", {"D02": "Enzyme inhibitors"}, 2)
    assert blocked["status"] == "excluded_relation_leakage"
    assert blocked["pharmacological_actions_excluded"] == 2
    functional = mesh_chain(["D27.1"], "ChemicalEntity", {"D27": "Dopamine Agents"}, 0)
    assert functional["status"] == "excluded_functional_branch"
    assert functional["chain"] == []
    cell = classify_cell_line(["Cancer_cell_line", "Female"], ["NCIt:C123", "BTO:0001"])
    assert cell["chain"] == [{"node": "Cancer_cell_line", "label": "Cancer_cell_line", "relation": "cellosaurus_category"}]
    assert cell["disease_of_origin_excluded"] == ["NCIt:C123"]


def test_variant_spelling_and_gene_identity_do_not_invent_a_finer_type():
    assert classify_variant("rs74805019", {"SO:1000002"})["reason"] == "rs_identifier_does_not_determine_variant_class"
    mapped = classify_variant("p|SUB|S|138|G", {"SO:1000002"})
    assert mapped["chain"][0]["node"] == "SO:1000002"
    gene = classify_concept({"concept_id": "3175", "entity_type": "GeneOrGeneProduct", "type_conflict": False}, {})
    assert gene["reason"] == "gene_identifier_does_not_establish_molecular_form"
    recorded = classify_concept(
        {"concept_id": "3175", "entity_type": "GeneOrGeneProduct", "type_conflict": False},
        {"gene_types": {"3175": {"type_of_gene": "protein-coding", "tax_id": "9606"}}},
    )
    assert recorded["chain"] == []
    assert recorded["database_type"] == "protein-coding"
    assert displayed_type("ChemicalEntity", [], 4) == "ChemicalEntity"
    conflict = classify_concept({"concept_id": "1", "entity_type": None, "type_conflict": True}, {})
    assert conflict["status"] == "type_conflict"


def test_research_p0_reuses_historical_p1_without_a_new_partition():
    notation = load_notation(ROOT / "configs" / "schemas" / "predicate_notation.yaml")
    assert historical_level(notation, "P0") == "P1"
    assert historical_level(notation, "P3") == "P4"
    assert positive_group_count("P0", 8) == 1
    assert notation["negative_outcome"]["inside_positive_hierarchy"] is False


def test_verbalisation_does_not_turn_a_residual_or_correlation_into_a_mechanism():
    association = verbalise("Association", "aspirin", "asthma")
    assert "caus" not in association["text"].lower()
    assert "annotated" not in association["text"].lower()
    residual = verbalise("Other_Relation", "a", "b", members=["Association", "Bind"])
    assert residual["residual_is_a_disjunction"] is True
    assert "not one mechanism" in residual["text"]
    reporting = verbalise("Association", "aspirin", "asthma", condition="reporting")
    assert reporting["text"].startswith("The source reports that")
    empty = verbalise(None, "a", "b")
    assert empty["status"] == "not_produced"
    absent = verbalise("NO_RELATION", "a", "b")
    assert absent["text"] is None
    assert absent["status"] == "no_positive_assertion"
    assert prompt_label("RELATED_TO") == "Any_Relation"


def test_semantic_measures_keep_support_and_completeness_separate():
    coverage = atomic_coverage(
        [
            {"status": "preserved"},
            {"status": "validly_abstracted"},
            {"status": "not_assessable"},
        ]
    )
    assert coverage["preserved_fraction_of_assessable"] == 0.5
    assert coverage["abstraction_counts_as_full_preservation"] is False
    proxy = loss_decomposition(0.8, 0.6, 0.4, validated_proportion=False)
    assert proxy["total"] is None
    validated = loss_decomposition(0.8, 0.6, 0.4, validated_proportion=True)
    assert validated["total"] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        support_distribution(0.2, 0.2, 0.2)
    with pytest.raises(ValueError):
        support_distribution(float("nan"), 0.0, 1.0)
    with pytest.raises(ValueError):
        atomic_coverage([{"status": "invented", "component": "relation"}])
    base = {
        "source_version": "s",
        "finding_id": "f1",
        "evaluation_text": "text",
        "assertion": "claim",
        "predicate_interpretation": "v1",
        "verbalisation": "proposition",
        "direction": "forward",
        "method": "m",
        "model_revision": "r",
        "prompt_version": "p",
        "calibration_version": "none",
    }
    other = dict(base, finding_id="f2")
    assert measurement_identifier(base) != measurement_identifier(other)


def test_interaction_uses_the_matched_four_cells():
    values = {("E2", "P3"): 0.5, ("E0", "P3"): 0.4, ("E2", "P0"): 0.3, ("E0", "P0"): 0.3}
    assert entity_predicate_interaction(values, "E2", "E0", "P3", "P0") == pytest.approx(0.1)
    estimate = paired_bootstrap([0.2, 0.4, 0.5], [0.1, 0.1, 0.2], draws=40)
    assert estimate["seeds_pooled"] is False
    assert estimate["unit"] == "document"


def test_a_branching_ontology_keeps_the_shared_prefix():
    names = {"CHEBI:1": "organic", "CHEBI:2": "left", "CHEBI:3": "right"}
    parents = {"CHEBI:9": ["CHEBI:1"], "CHEBI:1": ["CHEBI:2", "CHEBI:3"]}
    chain, alternatives = _ancestor_chain("CHEBI:9", names, parents, "chebi_is_a")
    assert [node["node"] for node in chain] == ["CHEBI:1"]
    assert {node["node"] for node in alternatives} == {"CHEBI:2", "CHEBI:3"}


def test_a_shared_ancestor_is_kept_when_the_components_diverge():
    descriptors = {
        "D000001": {"name": "A", "tree_numbers": ["D02.1.1"], "n_pharmacological_actions": 0},
        "D000002": {"name": "B", "tree_numbers": ["D02.2.1"], "n_pharmacological_actions": 0},
    }
    labels = {"D02": "Organic chemicals", "D02.1": "Left", "D02.2": "Right"}
    shared = classify_mesh_identifier("D000001|D000002", "ChemicalEntity", descriptors, labels)
    assert shared["status"] == "partial_common_ancestor"
    assert shared["chain"] == [{"node": "D02", "label": "Organic chemicals", "relation": "mesh_tree_ancestor"}]


def test_a_repeated_quotation_is_not_resolved_by_its_first_copy():
    document = "reduced activity. Later, reduced activity."
    ambiguous = recover_quotation(document, "reduced activity")
    assert ambiguous["status"] == "ambiguous_repeated_quotation"
    second = recover_quotation(document, "reduced activity", occurrence=1)
    assert second["status"] == "recovered"
    assert second["char_start"] == document.rfind("reduced activity")
    invalid = validate_model_response(document, {"status": "guess"})
    assert invalid["accepted"] is False


def test_a_quotation_must_occur_in_the_document_and_a_bad_record_is_not_complete(tmp_path: Path):
    assert recover_quotation("Imatinib reduced activity.", "reduced activity")["status"] == "recovered"
    assert recover_quotation("Imatinib reduced activity.", "caused remission")["status"] == "rejected_not_in_source"
    path = tmp_path / "rows.jsonl"
    path.write_text('{"candidate_id": "a"}\n', encoding="utf-8")
    inspected = inspect_jsonl(path, "candidate_id", {"candidate_id": str, "document_id": str})
    population = population_difference(inspected["identifiers"], {"a"})
    completed = collection_complete(inspected, population)
    assert population["complete"] is True
    assert completed["complete"] is False
