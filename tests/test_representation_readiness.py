"""Boundary checks for publication, mappings, partitions, and source spans."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from schema_study.entity_classification import identifier_family, load_mesh_descriptors, summarise_mappings
from schema_study.predicate_alternatives import load_alternatives, partition_errors, partition_information
from schema_study.records import publish_file
from schema_study.schemas import load_schema
from schema_study.sentences import segment_document, terminal_punctuation_sentences
from schema_study.source_integrity import inspect_jsonl, population_difference
from schema_study.source_references import source_spans

ROOT = Path(__file__).resolve().parents[1]


def test_interrupted_publication_does_not_replace_a_finished_file(tmp_path: Path):
    destination = tmp_path / "result.json"
    destination.write_text("finished\n", encoding="utf-8")

    def fail(temporary: Path) -> None:
        temporary.write_text("partial\n", encoding="utf-8")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        publish_file(destination, fail)
    assert destination.read_text(encoding="utf-8") == "finished\n"
    assert not (tmp_path / "result.json.partial").exists()


def test_identifier_families_do_not_invent_a_variant_type():
    assert identifier_family("D003924") == "mesh_descriptor"
    assert identifier_family("C567134") == "mesh_supplemental"
    assert identifier_family("D007859|D008569") == "composite_mesh"
    assert identifier_family("rs74805019") == "dbsnp"
    assert identifier_family("p|SUB|S|138|G") == "variant_notation"
    assert identifier_family("") == "missing"
    assert identifier_family("T1") == "other"


def test_mesh_descriptor_keeps_its_own_identifier_and_every_tree(tmp_path: Path):
    path = tmp_path / "desc.xml.gz"
    xml = """<DescriptorRecordSet>
<DescriptorRecord>
  <DescriptorUI>D000001</DescriptorUI>
  <TreeNumberList><TreeNumber>D03.1</TreeNumber><TreeNumber>C01.2</TreeNumber></TreeNumberList>
  <PharmacologicalActionList><PharmacologicalAction><DescriptorUI>D999999</DescriptorUI></PharmacologicalAction></PharmacologicalActionList>
</DescriptorRecord>
</DescriptorRecordSet>"""
    with gzip.open(path, "wb") as handle:
        handle.write(xml.encode("utf-8"))
    index = load_mesh_descriptors(path)
    assert set(index) == {"D000001"}
    assert index["D000001"]["tree_numbers"] == ["D03.1", "C01.2"]
    assert index["D000001"]["n_pharmacological_actions"] == 1
    summary = summarise_mappings(
        [{"concept_id": "D000001", "entity_type": "ChemicalEntity"}, {"concept_id": "3175", "entity_type": "GeneOrGeneProduct"}],
        index,
    )
    assert summary["multiple_tree_numbers"] == 1
    assert summary["multiple_top_level_categories"] == 1
    assert summary["top_level_outside_the_native_expectation"] == 1
    assert summary["descriptors_with_a_pharmacological_action"] == 1


def test_alternative_partitions_cover_each_native_label_once():
    for corpus in ("biored", "drugprot"):
        schema = load_schema(ROOT / "configs" / "schemas" / f"{corpus}.yaml")
        item = next(row for row in load_alternatives(ROOT / "configs" / "schemas" / "predicate_alternatives.yaml") if row["corpus"] == corpus)
        assert partition_errors(schema, item["groups"]) == []
    schema = load_schema(ROOT / "configs" / "schemas" / "biored.yaml")
    measured = partition_information(schema, {fine_id: 1 for fine_id in schema.fine_ids}, [{"display_name": "All", "members": list(schema.fine_ids)}])
    assert measured["n_categories"] == 1
    assert measured["retained_fraction"] == 0.0


def test_noncontiguous_sentences_remain_separate_spans():
    rows = [
        {"sentence_id": "a", "char_start": 0, "char_end": 5},
        {"sentence_id": "b", "char_start": 6, "char_end": 11},
        {"sentence_id": "c", "char_start": 12, "char_end": 18},
    ]
    by_id = {row["sentence_id"]: row for row in rows}
    assert source_spans(by_id, ["a", "c"]) == [[0, 5], [12, 18]]
    assert source_spans(by_id, ["a", "b"]) == [[0, 11]]


def test_biomedical_boundary_examples_keep_a_human_reference_empty():
    abbreviation = _abstract("The interval was 120 min. After treatment the level fell.")
    recorded = segment_document(abbreviation)
    terminal = terminal_punctuation_sentences(abbreviation)
    assert len(recorded) == 1
    assert len(terminal) == 2
    variant = _abstract("The mutation is p.R229G in exon 8.")
    assert len(segment_document(variant)) == 1
    assert len(terminal_punctuation_sentences(variant)) == 2


def test_jsonl_inspection_reports_duplicates_and_a_missing_identifier(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"candidate_id": "a"}\n{"candidate_id": "a"}\nnot-json\n', encoding="utf-8")
    inspected = inspect_jsonl(path, "candidate_id")
    assert inspected["n_records"] == 2
    assert inspected["parse_errors"] == 1
    assert inspected["duplicate_identifiers"] == 1
    assert inspected["final_record_parses"] is False
    assert population_difference(inspected["identifiers"], {"a", "b"})["n_missing"] == 1


def _abstract(text: str) -> dict:
    return {
        "document_id": "example",
        "split": "example",
        "title_char_end": 0,
        "abstract_char_start": 0,
        "separator": "",
        "text": text,
    }
