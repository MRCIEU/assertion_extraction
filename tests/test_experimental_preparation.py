"""Checks for the corrected experimental preparation."""

from __future__ import annotations

from schema_study.analysis_plan import analysis_catalogue, micro_f1_bootstrap, stratum_difference
from schema_study.experimental_preparation import _draw, _level_detail
from schema_study.learning_conditions import (
    bm25_index,
    condition_counts,
    principal_conditions,
    random_demonstrations,
    recover_arguments,
    schema_conditioned_configuration,
    select_demonstrations,
    sft_example,
)
from schema_study.semantic_measures import match_findings, match_multilabel
from schema_study.verbalisation import verbalise


def test_displayed_levels_follow_the_proposed_inventory_and_keep_the_native_type():
    chain = [{"node": "SO:1000002", "label": "substitution", "relation": "sequence_ontology_nomenclature"}]
    levels = [f"E{level}" for level in range(8)]
    displayed, detail = _level_detail("SequenceVariant", chain, levels)
    assert set(displayed) == set(levels)
    assert displayed["E2"] == "SequenceVariant"
    assert displayed["E3"] == "substitution"
    assert displayed["E7"] == "substitution"
    assert detail["E7"]["previous_type_retained"] is True
    assert detail["E7"]["changed_from_previous"] is False
    empty, empty_detail = _level_detail("GeneOrGeneProduct", [], levels)
    assert empty["E7"] == "GeneOrGeneProduct"
    assert empty_detail["E7"]["requested_level"] == 7


def test_the_matrix_is_corpus_specific_and_counts_training_separately():
    rows = principal_conditions([f"E{level}" for level in range(8)])
    counts = condition_counts(rows)
    assert counts["training_runs"] == 324
    assert counts["training_runs_by_model"]["biomedbert"] == 108
    assert counts["bc8_training_runs"] == 0
    assert counts["drugprot_entity_levels"] == ["native"]
    assert any(row["dataset"] == "bc8" and row["unit"] == "evaluation" for row in rows)


def test_a_missing_mention_does_not_become_the_candidate_identifier():
    recovered = recover_arguments(
        {"candidate_id": "biored:1:a:b", "mention_ids_1": ["m1"], "mention_ids_2": []},
        {"m1": {"mention_id": "m1", "text": "aspirin", "char_start": 0, "concept_id": "D1"}},
    )
    assert recovered["status"] == "unresolved_entity"
    assert recovered["argument_1"] is None
    present = recover_arguments(
        {"candidate_id": "biored:1:a:b", "mention_ids_1": ["m1"], "mention_ids_2": ["m2"]},
        {
            "m1": {"mention_id": "m1", "text": "aspirin", "char_start": 4, "concept_id": "D1"},
            "m2": {"mention_id": "m2", "text": "asthma", "char_start": 1, "concept_id": "D2"},
        },
    )
    assert present["argument_1"] == "aspirin"
    assert present["argument_2"] == "asthma"
    assert verbalise("Association", "", "asthma")["status"] == "unresolved_entity"


def test_demonstrations_stay_fixed_and_include_a_negative_when_one_exists():
    documents = [
        {"id": "t", "document_id": "9", "tokens": ["aspirin", "asthma"], "negative": False},
        {"id": "a", "document_id": "1", "tokens": ["aspirin", "asthma"], "negative": False},
        {"id": "b", "document_id": "2", "tokens": ["aspirin", "kinase"], "negative": False},
        {"id": "c", "document_id": "3", "tokens": ["aspirin", "trial"], "negative": True},
        {"id": "d", "document_id": "4", "tokens": ["unrelated"], "negative": False},
        {"id": "e", "document_id": "5", "tokens": ["aspirin"], "negative": False},
    ]
    index = bm25_index(documents)
    first = select_demonstrations(["aspirin", "asthma"], "9", index, 5)
    second = select_demonstrations(["aspirin", "asthma"], "9", index, 5)
    assert first == second
    assert "t" not in first
    assert "c" in first


def test_supervised_targets_do_not_put_the_candidate_identifier_in_the_loss():
    example = sft_example("biored:1:a:b", [], "E3", "P0")
    assert example["assistant"] == {"labels": []}
    assert example["assistant_includes_candidate_identifier"] is False
    assert "biored:1:a:b" not in str(example["assistant"])


def test_inclusion_probability_follows_the_realised_sample():
    import random

    draw = _draw(["0", "1", "2"], lambda _unit: "only", random.Random(1), fraction=0.2)
    stratum = draw["strata"]["only"]
    assert stratum["drawn"] == 1
    assert stratum["inclusion_probability"] == 1 / 3


def test_matching_does_not_select_the_highest_score_and_keeps_missing_findings():
    references = [
        {"finding_id": "a", "match_key": "pair-1"},
        {"finding_id": "b", "match_key": "pair-2"},
    ]
    predictions = [{"prediction_id": "p", "match_key": "pair-1", "score": 0.1}]
    matched = match_findings(references, predictions)
    assert matched["highest_score_selection"] is False
    assert matched["denominator"] == 2
    assert matched["recovered"] == 1
    assert matched["pairs"][1]["status"] == "missing"
    labels = match_multilabel(["agonist", "inhibitor"], ["agonist"])
    assert labels["denominator"] == 2
    assert labels["missed"] == ["inhibitor"]


def test_random_demonstrations_exclude_the_target_document_and_stay_fixed():
    pool = [
        {"id": "a", "document_id": "1", "negative": False},
        {"id": "b", "document_id": "2", "negative": True},
        {"id": "c", "document_id": "9", "negative": False},
    ]
    first = random_demonstrations("9", pool, 2, 5)
    assert first == random_demonstrations("9", pool, 2, 5)
    assert "c" not in first
    assert "b" in first


def test_schema_conditioned_exposure_is_one_view_per_candidate():
    configuration = schema_conditioned_configuration(10, 8, 4)
    assert configuration["independent_sft_examples_per_epoch"] == 320
    assert configuration["schema_conditioned_examples_per_epoch"] == 10
    assert configuration["in_principal_matrix"] is False


def test_the_analysis_catalogue_covers_the_principal_contrasts_without_averaging_document_f1():
    names = {row["id"] for row in analysis_catalogue()}
    assert "predicate_by_paradigm" in names
    assert "certainty_moderation" in names
    documents = [
        {"left": {"tp": 2, "fp": 0, "fn": 0}, "right": {"tp": 1, "fp": 0, "fn": 1}},
        {"left": {"tp": 0, "fp": 1, "fn": 1}, "right": {"tp": 0, "fp": 0, "fn": 1}},
    ]
    result = stratum_difference(documents, documents, "left", "right", draws=10, seed=100)
    assert result["document_f1_averaged"] is False
    assert result["difference"] == 0


def test_micro_f1_is_recomputed_from_pooled_counts():
    documents = [
        {"left": {"tp": 1, "fp": 0, "fn": 0}, "right": {"tp": 0, "fp": 1, "fn": 1}},
        {"left": {"tp": 1, "fp": 1, "fn": 0}, "right": {"tp": 1, "fp": 0, "fn": 0}},
    ]
    result = micro_f1_bootstrap(documents, left="left", right="right", draws=20, seed=100)
    assert result["document_f1_averaged"] is False
    assert result["statistic"] == "corpus_micro_f1_difference"
    assert result["seed"] == 100
