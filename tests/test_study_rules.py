"""CPU checks for the rules that change a study conclusion."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
import yaml

from schema_study.data import (
    _drugprot_document,
    build_biored_document,
    choose_fixed_separator,
    drugprot_candidate_id,
    parse_pubtator,
    prepare_pubtator_corpus,
    split_drugprot_training,
)
from schema_study.evaluation import (
    average_seed_metrics,
    binary_cross_entropy_with_logits,
    bootstrap_paired_mean_difference,
    candidate_label_counts,
    document_resample_indices,
    f1_from_counts,
    information_retention,
    polarity_summary,
    score_pooled_predictions,
    select_common_threshold,
    select_epoch,
    smoothed_fine_probabilities,
    summarize_counts,
)
from schema_study.experiments import (
    ModelExecutionBlocked,
    blocked_generate,
    blocked_train,
    downsampling_plan,
    example_order,
    plan_counts,
    plan_experiments,
    seed_streams,
)
from schema_study.inputs import (
    GAP_MARKER,
    MentionView,
    WhitespaceEncoder,
    build_target_anchored_context,
    render_marked_text,
)
from schema_study.predictions import parse_prediction_text, rollup_predicted_labels
from schema_study.reporting import plot_paired_differences
from schema_study.review import development_and_check, nested_samples, stratum_of
from schema_study.schemas import (
    composed_equals_direct,
    load_schema,
    map_fine_ids,
    partition_errors,
    random_merge_versions,
    rollup_ids,
)
from schema_study.validation import ValidationReport

ROOT = Path(__file__).resolve().parents[1]


def biored():
    return load_schema(ROOT / "configs" / "schemas" / "biored.yaml")


def drugprot():
    return load_schema(ROOT / "configs" / "schemas" / "drugprot.yaml")


def test_partitions_are_complete_and_composition_is_stable():
    for schema in (biored(), drugprot()):
        assert partition_errors(schema) == []
        assert composed_equals_direct(schema)
        for fine_id in schema.fine_ids:
            direct = map_fine_ids(schema, {fine_id}, "P2")
            via = rollup_ids(schema, "P3", map_fine_ids(schema, {fine_id}, "P3"), "P2")
            assert direct == via
    assert "AGONIST" not in drugprot().by_id("P2")["DP_P2_UP"].members
    assert "ANTAGONIST" not in drugprot().by_id("P2")["DP_P2_DOWN"].members


def test_random_merge_matches_the_draft_configuration():
    schema = biored()
    versions = random_merge_versions(list(schema.fine_ids), seed=42)
    with (ROOT / "configs" / "schemas" / "biored_controls.yaml").open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    assert payload["status"] == "draft_pending_confirmation"
    stored = payload["random_merge"]["versions"]
    assert len(versions) == 3
    for generated, saved in zip(versions, stored):
        assert {frozenset(group) for group in generated} == {frozenset(group["members"]) for group in saved["groups"]}
    assert len({frozenset(frozenset(group) for group in version) for version in versions}) == 3


def test_failure_is_not_no_relation_and_illegal_counts_are_separate():
    positive = biored().positive_display_names("P4")
    assert positive
    rows = [
        {"gold": ["BR01_POSITIVE_CORRELATION"], "predicted": ["BR01_POSITIVE_CORRELATION"], "status": "valid"},
        {"gold": ["BR01_POSITIVE_CORRELATION"], "predicted": ["BR02_NEGATIVE_CORRELATION"], "status": "valid"},
        {"gold": ["BR01_POSITIVE_CORRELATION"], "predicted": [], "status": "valid"},
        {"gold": [], "predicted": [], "status": "valid"},
        {"gold": [], "predicted": [], "status": "missing"},
        {"gold": ["BR01_POSITIVE_CORRELATION"], "predicted": [], "status": "missing"},
        {"gold": ["BR01_POSITIVE_CORRELATION"], "predicted": ["BR01_POSITIVE_CORRELATION", "BR02_NEGATIVE_CORRELATION"], "status": "rejected"},
        {"gold": ["DP09_INHIBITOR"], "predicted": ["DP09_INHIBITOR"], "status": "partial_illegal", "illegal_labels": ["NOT_A_LABEL"]},
    ]
    wrong = candidate_label_counts(rows[1]["gold"], rows[1]["predicted"], "valid")
    assert (wrong["tp"], wrong["fp"], wrong["fn"]) == (0, 1, 1)
    missing_negative = candidate_label_counts([], [], "missing")
    valid_negative = candidate_label_counts([], [], "valid")
    assert missing_negative["exact"] is False
    assert valid_negative["exact"] is True
    assert missing_negative["fp"] == 0
    rejected = candidate_label_counts(["BR01_POSITIVE_CORRELATION"], [], "rejected")
    assert rejected["fp"] == 1 and rejected["fn"] == 1 and rejected["tp"] == 0
    partial = candidate_label_counts(["DP09_INHIBITOR"], ["DP09_INHIBITOR"], "partial_illegal", ["NOT_A_LABEL"])
    assert partial["tp"] == 1 and partial["extra_fp"] == 1 and partial["exact"] is False
    summary = summarize_counts(rows[7:8], ["DP09_INHIBITOR"])
    assert summary["fp"] == 1
    assert summary["class_fp_sum"] == 0
    empty = summarize_counts([{"gold": [], "predicted": [], "status": "valid"}], ["BR01_POSITIVE_CORRELATION"])
    assert empty["precision"] is None and empty["recall"] is None
    assert empty["f1"] == 0.0
    assert empty["per_class"][0]["support"] == 0
    assert empty["macro_f1"] == 0.0


def test_parser_examples():
    schema = biored()
    drug = drugprot()
    assert parse_prediction_text('```json\n{"candidate_id":"c","labels":[]}\n```', expected_candidate_id="c", schema=schema, level="P4")["output_status"] == "valid"
    repeated = parse_prediction_text('{"candidate_id":"c","labels":["Positive_Correlation","Positive_Correlation"]}', expected_candidate_id="c", schema=schema, level="P4")
    assert repeated["predicted_labels"] == ["BR01_POSITIVE_CORRELATION"]
    conflict = parse_prediction_text('{"candidate_id":"c","labels":["Positive_Correlation","Bind"]}', expected_candidate_id="c", schema=schema, level="P4")
    assert conflict["output_status"] == "rejected"
    mixed = parse_prediction_text('{"candidate_id":"c","labels":["No_Relation","INHIBITOR"]}', expected_candidate_id="c", schema=drug, level="P4")
    assert mixed["output_status"] == "rejected"
    illegal = parse_prediction_text('{"candidate_id":"c","labels":["INHIBITOR","NOT_A_LABEL"]}', expected_candidate_id="c", schema=drug, level="P4")
    assert illegal["output_status"] == "partial_illegal"
    assert illegal["predicted_labels"] == ["DP09_INHIBITOR"]
    assert parse_prediction_text("not json", expected_candidate_id="c", schema=schema, level="P4")["output_status"] == "missing"
    rolled = rollup_predicted_labels(schema, ["BR05_COMPARISON", "BR08_CONVERSION"], source_level="P4", target_level="P2")
    assert rolled == ["BR_P2_OTHER"]


def test_information_formula_and_p1_distinction():
    schema = biored()
    counts = {fine_id: 0 for fine_id in schema.fine_ids}
    counts["BR01_POSITIVE_CORRELATION"] = 1
    probabilities = smoothed_fine_probabilities(counts, schema.fine_ids)
    assert math.isclose(sum(probabilities.values()), 1.0)
    relations = [
        {"fine_id": "BR01_POSITIVE_CORRELATION", "status": "valid", "predicted_groups": ["BR01_POSITIVE_CORRELATION"]},
        {"fine_id": "BR02_NEGATIVE_CORRELATION", "status": "missing", "predicted_groups": []},
    ]
    scored = information_retention(relations, schema, "P4", counts)
    assert scored["frequency"]["retained_R"] > 0
    assert scored["frequency"]["ceiling_C"] > scored["frequency"]["retained_R"]
    reconstructed = (
        scored["frequency"]["schema_loss_1_minus_C"]
        + scored["frequency"]["prediction_loss_C_minus_R"]
    )
    assert math.isclose(reconstructed, scored["frequency"]["total_loss_1_minus_R"])
    p1 = information_retention(relations, schema, "P1", counts)
    assert p1["frequency"]["ceiling_C"] == 0.0
    assert p1["frequency"]["retained_R"] == 0.0


def test_polarity_outcomes_including_bidirectional_gold():
    schema = drugprot()
    rows = [
        {"fine_ids": ["DP01_ACTIVATOR"], "status": "valid", "predicted_groups": ["DP_P2_UP"]},
        {"fine_ids": ["DP01_ACTIVATOR"], "status": "valid", "predicted_groups": ["DP_P2_DOWN"]},
        {"fine_ids": ["DP01_ACTIVATOR"], "status": "valid", "predicted_groups": ["DP_P2_OTHER"]},
        {"fine_ids": ["DP01_ACTIVATOR"], "status": "missing", "predicted_groups": []},
        {"fine_ids": ["DP01_ACTIVATOR"], "status": "rejected", "predicted_groups": ["DP_P2_UP"]},
        {"fine_ids": ["DP01_ACTIVATOR"], "status": "valid", "predicted_groups": ["DP_P2_UP", "DP_P2_DOWN"]},
        {"fine_ids": ["DP01_ACTIVATOR", "DP09_INHIBITOR"], "status": "valid", "predicted_groups": ["DP_P2_UP", "DP_P2_DOWN"]},
    ]
    counts = polarity_summary(rows, schema, "P2")
    assert counts["retained"] == 1
    assert counts["reversed"] == 1
    assert counts["coarsened_nonpolar"] == 1
    assert counts["missed"] == 1
    assert counts["conflict"] == 2
    assert counts["bidirectional_set_match"] == 1


def test_seed_mean_is_not_the_pooled_score_and_bootstrap_is_paired():
    labels = ["A"]
    seed_one = [
        {"document_id": "d1", "gold": ["A"], "predicted": ["A"], "status": "valid"},
        {"document_id": "d2", "gold": ["A"], "predicted": ["A"], "status": "valid"},
    ]
    seed_two = [
        {"document_id": "d1", "gold": ["A"], "predicted": [], "status": "valid"},
        {"document_id": "d2", "gold": ["A"], "predicted": [], "status": "valid"},
    ]
    per_seed = [summarize_counts(seed_one, labels), summarize_counts(seed_two, labels)]
    averaged = average_seed_metrics(per_seed)
    pooled = score_pooled_predictions(seed_one + seed_two, labels)
    assert per_seed[0]["f1"] == 1.0
    assert per_seed[1]["f1"] == 0.0
    assert averaged["f1"] == 0.5
    assert pooled["f1"] == f1_from_counts(2, 0, 2)
    assert averaged["f1"] != pooled["f1"]
    first = document_resample_indices(4, 5, 20261115)
    second = document_resample_indices(4, 5, 20261115)
    assert first == second
    rows = {
        "13": {"direct": seed_one, "rollup": seed_two},
        "42": {"direct": seed_two, "rollup": seed_two},
    }
    result = bootstrap_paired_mean_difference(
        rows,
        ["d1", "d2"],
        labels,
        condition_a="direct",
        condition_b="rollup",
        n_draws=20,
        seed=20261115,
    )
    assert result["observed_mean_difference"] == 0.5
    assert result["interval_low"] <= result["observed_mean_difference"] <= result["interval_high"]


def test_threshold_and_epoch_ties():
    labels = ["A", "B"]
    gold = [{"A"}, set()]
    probs = [{"A": 0.2, "B": 0.0}, {"A": 0.0, "B": 0.0}]
    chosen = select_common_threshold(gold, probs, labels)
    assert chosen["threshold"] >= 0.05
    epoch = select_epoch([{"epoch": 2, "f1": 0.4}, {"epoch": 1, "f1": 0.4}, {"epoch": 3, "f1": 0.3}])
    assert epoch["epoch"] == 1
    assert binary_cross_entropy_with_logits([[0.0]], [[0.0]]) > 0


def test_context_overlap_budget_and_dual_fragment():
    text = "0123456789"
    mentions = [
        MentionView("h", 0, 6, "H", "GeneOrGeneProduct"),
        MentionView("t", 4, 10, "T", "ChemicalEntity"),
    ]
    rendered = render_marked_text(text, mentions, level="E0")
    assert "".join(character for character in rendered if character.isdigit()) == text
    assert rendered.count("0123") == 1
    words = " ".join(f"w{index}" for index in range(12))
    head = MentionView("h", 0, 2, "H", "GeneOrGeneProduct")
    tail = MentionView("t", words.rfind("w11"), words.rfind("w11") + 3, "T", "GeneOrGeneProduct")
    encoder = WhitespaceEncoder()
    views = {"E0": None, "E2": {}}
    tight = build_target_anchored_context(words, [head, tail], encoder, views, budget=12)
    assert tight["mode"] == "dual"
    assert GAP_MARKER in tight["views"]["E0"]
    assert "w5" not in tight["views"]["E0"]
    short = "alpha beta"
    short_mentions = [MentionView("h", 0, 5, "H", "GeneOrGeneProduct"), MentionView("t", 6, 10, "T", "ChemicalEntity")]
    full = build_target_anchored_context(short, short_mentions, encoder, views, budget=50)
    assert full["mode"] == "full"
    heavy = WhitespaceEncoder({"<<TYPE:GeneOrGeneProduct>>": 3})
    nearby = [head, MentionView("t2", 3, 5, "T", "GeneOrGeneProduct")]
    limited = build_target_anchored_context(words, nearby, heavy, views, budget=20)
    assert limited["mode"] == "single"
    assert limited["encoded_length_by_view"]["E2"] > limited["encoded_length_by_view"]["E0"]
    assert limited["max_encoded_length"] <= 20


def test_candidates_conflicts_and_drugprot_multilabel():
    schema = biored()
    title = "Alpha meets beta."
    abstract = "Alpha binds beta."
    text = title + " " + abstract
    alpha_title = (0, 5)
    beta_title = (text.index("beta"), text.index("beta") + 4)
    alpha_abs = (text.index("Alpha binds"), text.index("Alpha binds") + 5)
    beta_abs = (text.rindex("beta"), text.rindex("beta") + 4)
    cell = (text.index("meets"), text.index("meets") + 5)
    pubtator = "\n".join(
        [
            "7|t|" + title,
            "7|a|" + abstract,
            f"7\t{alpha_title[0]}\t{alpha_title[1]}\tAlpha\tGeneOrGeneProduct\tG1",
            f"7\t{beta_title[0]}\t{beta_title[1]}\tbeta\tGeneOrGeneProduct\tG2",
            f"7\t{alpha_abs[0]}\t{alpha_abs[1]}\tAlpha\tGeneOrGeneProduct\tG1",
            f"7\t{beta_abs[0]}\t{beta_abs[1]}\tbeta\tGeneOrGeneProduct\tG2",
            f"7\t{cell[0]}\t{cell[1]}\tmeets\tCellLine\tC1",
            "7\tPositive_Correlation\tG1\tG2\tNovel",
            "7\tPositive_Correlation\tG1\tG2\tNo",
            "7\tAssociation\tG1\tG2\tNovel",
            "7\tBind\tG1\tC1\tNovel",
            "",
        ]
    )
    document = parse_pubtator(pubtator)[0]
    separator = choose_fixed_separator([document])
    document["text"] = document["title"] + separator["separator"] + document["abstract"]
    document["title_char_end"] = len(document["title"])
    document["abstract_char_start"] = len(document["title"]) + 1
    document["separator"] = separator["separator"]
    built = build_biored_document(document, schema, "train", ValidationReport("synthetic"))
    by_id = {row["candidate_id"]: row for row in built["candidates"]}
    pair = by_id[next(iter(by_id))]
    assert any(row["gold_status"] == "conflict" for row in built["candidates"])
    assert any(row["reason"] == "unsupported_type_pair" for row in built["exclusions"])
    assert pair["candidate_id"].startswith("biored:7:")
    mentions = [
        {"term": "T1", "entity_type": "CHEMICAL", "start": 0, "end": 4, "text": "drug"},
        {"term": "T2", "entity_type": "GENE-Y", "start": 5, "end": 9, "text": "gene"},
        {"term": "T3", "entity_type": "GENE-N", "start": 10, "end": 14, "text": "prot"},
    ]
    relations = [
        {"label": "INHIBITOR", "chemical_term": "T1", "gene_term": "T2"},
        {"label": "ACTIVATOR", "chemical_term": "T1", "gene_term": "T2"},
        {"label": "INHIBITOR", "chemical_term": "T1", "gene_term": "T2"},
    ]
    _entities, candidates, _exclusions, _relations = _drugprot_document(
        "9", "drug gene prot", mentions, relations, "train", drugprot(), ValidationReport("drug")
    )
    multi = next(row for row in candidates if row["candidate_id"] == drugprot_candidate_id("9", "T1", "T2"))
    assert multi["gold_status"] == "multi"
    assert multi["gold_labels"] == ["ACTIVATOR", "INHIBITOR"]
    assert len(candidates) == 2


def test_composite_identifier_keeps_each_official_concept_link():
    schema = biored()
    title = "Alpha and beta."
    document = {
        "document_id": "8",
        "text": title,
        "title": title,
        "abstract": "",
        "title_char_end": len(title),
        "abstract_char_start": len(title),
        "separator": " ",
        "entities": [
            {"start": 0, "end": 14, "text": "Alpha and beta", "entity_type": "DiseaseOrPhenotypicFeature", "concept_id": "D1,D2"},
            {"start": 0, "end": 5, "text": "Alpha", "entity_type": "GeneOrGeneProduct", "concept_id": "7157"},
            {"start": 10, "end": 14, "text": "beta", "entity_type": "ChemicalEntity", "concept_id": "-"},
        ],
        "relations": [
            {"concept_id_1": "D1", "concept_id_2": "7157", "label": "Association", "novelty": "Novel"},
        ],
        "direction_notes": [],
        "unknown_lines": [],
    }
    built = build_biored_document(document, schema, "train", ValidationReport("composite"))
    concepts = {row["concept_id"] for row in built["entities"] if row["concept_id"]}
    assert concepts == {"D1", "D2", "7157"}
    assert any(row["reason"] == "no_concept_identifier" for row in built["exclusions"])
    by_pair = {tuple(sorted((row["argument_1_id"], row["argument_2_id"]))): row for row in built["candidates"]}
    assert by_pair[("7157", "D1")]["gold_labels"] == ["Association"]
    assert by_pair[("7157", "D2")]["gold_labels"] == []
    assert ("D1", "D2") not in by_pair


def test_drugprot_split_and_sampling_and_plan(tmp_path: Path):
    train, dev = split_drugprot_training([str(number) for number in range(3500)])
    assert len(train) == 2800 and len(dev) == 700 and set(train).isdisjoint(dev)
    again, _dev_again = split_drugprot_training([str(number) for number in range(3500)])
    assert train == again
    with pytest.raises(ValueError):
        split_drugprot_training([str(number) for number in range(10)])
    frame = []
    for index in range(300):
        labels = ["Association"] if index < 120 else (["Bind"] if index < 200 else [])
        frame.append({"candidate_id": f"c{index:03d}", "document_id": f"d{index // 3}", "stratum": stratum_of(labels)})
    sampled = nested_samples(frame)
    assert {item["candidate_id"] for item in sampled["fallback"]["selected"]} <= {item["candidate_id"] for item in sampled["main"]["selected"]}
    association = [row for row in sampled["main"]["probabilities"] if row["stratum"] == "association"]
    assert association[0]["inclusion_probability"] == 100 / 120
    held = development_and_check(frame)
    assert {item["candidate_id"] for item in held["development"]["selected"]}.isdisjoint(
        {item["candidate_id"] for item in held["check"]["selected"]}
    )
    controls = [
        {"control_id": "rename_p4", "predicate_level": "P4"},
        {"control_id": "rename_p2", "predicate_level": "P2"},
        {"control_id": "random_merge_1", "predicate_level": "P_control"},
        {"control_id": "random_merge_2", "predicate_level": "P_control"},
        {"control_id": "random_merge_3", "predicate_level": "P_control"},
    ]
    counts = plan_counts(plan_experiments(controls))
    assert counts["biored_biomedbert_train"] == 36
    assert counts["drugprot_biomedbert_train"] == 12
    assert counts["main_finetunes"] == 48
    assert counts["qwen8_biored"] == 12 and counts["qwen32_biored"] == 12
    assert counts["qwen8_drugprot"] == 4 and counts["qwen32_drugprot"] == 4
    assert counts["controls"] == 5
    streams_p2 = seed_streams(13, "E0", "P2")
    streams_p4 = seed_streams(13, "E0", "P4")
    assert streams_p2["encoder_init"] == streams_p4["encoder_init"]
    assert streams_p2["data_order"] == streams_p4["data_order"]
    assert streams_p2["classifier_head"] != streams_p4["classifier_head"]
    assert example_order(["b", "a"], 13) == example_order(["b", "a"], 13)
    plan = downsampling_plan(
        {f"L{index}": index for index in range(8)},
        [{"relation_id": f"r{index}", "fine_id": "L1"} for index in range(6)],
        coarse_of=lambda fine_id: "coarse",
    )
    assert plan["status"] == "reserved_not_launched"
    assert plan["causal_claim"] is False
    with pytest.raises(ModelExecutionBlocked):
        blocked_train({})
    with pytest.raises(ModelExecutionBlocked):
        blocked_generate({})
    plot_paired_differences(["direct-rollup"], [0.1], [0.0], [0.2], tmp_path / "diff", synthetic=True)
    with pytest.raises(ValueError):
        plot_paired_differences(["x"], [0.1], [0.0], [0.2], tmp_path / "evaluation" / "diff", synthetic=True)


def test_no_weight_loading_or_api_call_is_present():
    source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src" / "schema_study").glob("*.py"))
    lowered = source.lower()
    for banned in ("import torch", "automodel", "openai", "vllm", "api.openai.com"):
        assert banned not in lowered
    from schema_study.cli import main

    config = ROOT / "configs" / "experiments" / "pilot.yaml"
    if config.exists():
        assert main(["train", "--config", str(config)]) == 3


def test_resume_ledger_does_not_duplicate_completed_runs(tmp_path: Path):
    ledger = [{"run_id": "r1", "status": "complete"}, {"run_id": "r2", "status": "failed"}]
    (tmp_path / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    loaded = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    pending = [row["run_id"] for row in loaded if row["status"] != "complete"]
    assert pending == ["r2"]


def test_prepare_synthetic_corpus_keeps_splits_disjoint(tmp_path: Path):
    schema = biored()
    documents = []
    for split, document_id in (("train", "1"), ("dev", "2")):
        text = f"{document_id}|t|Alpha one.\n{document_id}|a|Beta two.\n"
        text += f"{document_id}\t0\t5\tAlpha\tGeneOrGeneProduct\tG1\n"
        text += f"{document_id}\t11\t15\tBeta\tChemicalEntity\tC1\n\n"
        documents.extend(parse_pubtator(text))
    summary = prepare_pubtator_corpus(
        documents,
        schema,
        corpus="synthetic",
        release_id="synthetic",
        source_files=[],
        output_dir=tmp_path / "prepared",
        split_of={"1": "train", "2": "dev"}.get,
    )
    assert summary["n_documents_by_split"] == {"train": 1, "dev": 1}
    assert (tmp_path / "prepared" / "candidates.parquet").exists()
