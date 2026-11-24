"""Sentence offsets, argument alignment, passage selection, and finding records."""

from __future__ import annotations

import inspect
from pathlib import Path

from schema_study.representation_meaning import (
    REQUIRED_FIELDS,
    entity_extension_requirements,
    predicate_interpretations,
    unresolved_questions,
)
from schema_study.schemas import LEVELS, load_schema
from schema_study.sentences import mention_sentences, segment_document
from schema_study.source_references import (
    align_arguments,
    build_passages,
    finding_reference,
    finding_references_for_labels,
    select_examples,
)

ROOT = Path(__file__).resolve().parents[1]


def _document() -> dict:
    title = "Dr. Smith measured 0.5 mg."
    abstract = "BRCA1 binds TP53. Later BRCA1 and TP53 were measured again."
    return {
        "document_id": "1",
        "split": "dev",
        "title_char_end": len(title),
        "abstract_char_start": len(title) + 1,
        "separator": " ",
        "text": title + " " + abstract,
    }


def _sentences():
    document = _document()
    rows = segment_document(document)
    for row in rows:
        assert document["text"][row["char_start"]:row["char_end"]] == row["text"]
    return document, rows


def test_sentence_offsets_keep_abbreviations_decimals_and_the_separator():
    document, rows = _sentences()
    assert [row["text"] for row in rows] == [
        "Dr. Smith measured 0.5 mg.",
        "BRCA1 binds TP53.",
        "Later BRCA1 and TP53 were measured again.",
    ]
    assert rows[0]["char_end"] == document["title_char_end"]
    assert rows[1]["char_start"] == document["abstract_char_start"]
    assert all(
        row["char_end"] <= document["title_char_end"] or row["char_start"] >= document["abstract_char_start"]
        for row in rows
    )


def test_mentions_align_to_sentences_and_a_crossing_mention_is_recorded():
    document, rows = _sentences()
    first = document["text"].index("BRCA1")
    second = document["text"].index("TP53")
    brca = {"mention_id": "a", "char_start": first, "char_end": first + 5, "text": "BRCA1"}
    tp53 = {"mention_id": "b", "char_start": second, "char_end": second + 4, "text": "TP53"}
    assert mention_sentences(brca, rows) == [rows[1]["sentence_id"]]
    crossing_start = document["text"].index("TP53. Later")
    crossing = {
        "mention_id": "c",
        "char_start": crossing_start,
        "char_end": crossing_start + len("TP53. Later"),
        "text": "TP53. Later",
    }
    aligned = align_arguments([crossing], rows)
    assert len(aligned["crossing"]) == 1
    assert aligned["crossing"][0]["sentence_ids"] == [rows[1]["sentence_id"], rows[2]["sentence_id"]]


def test_passage_selection_keeps_alternatives_and_ignores_labels():
    document, rows = _sentences()
    text = document["text"]
    first_brca = text.index("BRCA1")
    second_brca = text.index("BRCA1", first_brca + 1)
    first_tp = text.index("TP53")
    second_tp = text.index("TP53", first_tp + 1)
    first = align_arguments(
        [
            {"mention_id": "a1", "char_start": first_brca, "char_end": first_brca + 5, "text": "BRCA1"},
            {"mention_id": "a2", "char_start": second_brca, "char_end": second_brca + 5, "text": "BRCA1"},
        ],
        rows,
    )
    second = align_arguments(
        [
            {"mention_id": "b1", "char_start": first_tp, "char_end": first_tp + 4, "text": "TP53"},
            {"mention_id": "b2", "char_start": second_tp, "char_end": second_tp + 4, "text": "TP53"},
        ],
        rows,
    )
    assert "label" not in inspect.signature(build_passages).parameters
    built = build_passages(rows, first, second, len(text), [{"char_start": rows[1]["char_start"], "char_end": rows[1]["char_end"]}])
    assert built["n_same_sentence_locations"] == 2
    assert built["min_sentence_distance"] == 0
    assert built["n_shortest_intervals"] >= 2
    both = built["passages"]["both_targets"]["alternatives"]
    assert len(both) == 2
    assert {alternative["model_input"] for alternative in both} == {"all", "none"}
    shortest = built["passages"]["shortest_interval"]["alternatives"]
    assert all(alternative["n_sentences"] == 1 for alternative in shortest)
    neighbour = built["passages"]["neighbouring_sentences"]["alternatives"]
    assert neighbour
    partial = build_passages(rows, first, second, len(text), [{"char_start": rows[1]["char_start"], "char_end": rows[1]["char_start"] + 4}])
    assert "part" in {alternative["model_input"] for alternative in partial["passages"]["both_targets"]["alternatives"]}
    assert built["passages"]["document"]["alternatives"][0]["spans"] == [[0, len(text)]]


def test_neighbour_context_stops_at_the_document_edge():
    document, rows = _sentences()
    text = document["text"]
    smith = text.index("Smith")
    brca = text.index("BRCA1")
    first = align_arguments([{"mention_id": "s", "char_start": smith, "char_end": smith + 5, "text": "Smith"}], rows)
    second = align_arguments([{"mention_id": "b", "char_start": brca, "char_end": brca + 5, "text": "BRCA1"}], rows)
    built = build_passages(rows, first, second, len(text), [])
    assert built["min_sentence_distance"] == 1
    assert built["n_same_sentence_locations"] == 0
    assert built["passages"]["both_targets"]["alternatives"] == []
    shortest = built["passages"]["shortest_interval"]["alternatives"][0]
    assert shortest["model_input"] == "none"
    assert shortest["spans"] == [[rows[0]["char_start"], rows[1]["char_end"]]]
    neighbour_ids = built["passages"]["neighbouring_sentences"]["alternatives"][0]["sentence_ids"]
    assert neighbour_ids[0] == rows[0]["sentence_id"]
    assert neighbour_ids[-1] == rows[-1]["sentence_id"]


def test_finding_records_can_remain_unidentified_or_hold_several_findings():
    empty = finding_reference("candidate", "INHIBITOR")
    assert empty["status"] == "unidentified"
    assert empty["findings"] == []
    empty["status"] = "ambiguous"
    empty["findings"] = [
        {"finding_id": "one", "sentence_ids": ["s1"], "spans": [[0, 8]], "alternatives": []},
        {
            "finding_id": "two",
            "sentence_ids": ["s2", "s3"],
            "spans": [[12, 20], [30, 40]],
            "alternatives": [{"sentence_ids": ["s4"], "spans": [[50, 60]]}],
        },
    ]
    assert len(empty["findings"]) == 2
    assert len(empty["findings"][1]["sentence_ids"]) == 2
    separate = finding_references_for_labels("candidate", ["INHIBITOR", "SUBSTRATE"])
    assert [item["native_label"] for item in separate] == ["INHIBITOR", "SUBSTRATE"]
    assert finding_references_for_labels("candidate", [])[0]["native_label"] is None


def test_sampling_is_deterministic_and_document_disjoint():
    rows = []
    for index in range(30):
        rows.append(
            {
                "candidate_id": f"p{index:02d}",
                "document_id": f"d{index:02d}",
                "positive": True,
                "primary_label": ["ACTIVATOR", "INHIBITOR", "AGONIST"][index % 3],
                "gold_labels": ["ACTIVATOR"] if index % 5 else ["INHIBITOR", "SUBSTRATE"],
                "min_sentence_distance": 0 if index % 2 == 0 else 2,
                "repeated_mentions": index % 4 == 0,
                "model_input_mode": "single" if index % 6 == 0 else "full",
                "multi_label": index % 5 == 0,
            }
        )
    rows.append(
        {
            "candidate_id": "n1",
            "document_id": "n-doc",
            "positive": False,
            "primary_label": "",
            "gold_labels": [],
            "min_sentence_distance": 0,
            "repeated_mentions": False,
            "model_input_mode": "full",
            "multi_label": False,
        }
    )
    first = select_examples(rows, n_positive=8, n_negative=1, seed=5)
    second = select_examples(rows, n_positive=8, n_negative=1, seed=5)
    assert [row["candidate_id"] for row in first["positive"]] == [row["candidate_id"] for row in second["positive"]]
    chosen = first["positive"] + first["negative"]
    assert len({row["document_id"] for row in chosen}) == len(chosen)
    held_out = select_examples(rows, n_positive=4, n_negative=1, seed=6, excluded_documents=set(first["document_ids"]))
    assert set(held_out["document_ids"]).isdisjoint(first["document_ids"])


def test_every_predicate_group_has_a_semantic_record():
    interpretations = predicate_interpretations()
    for corpus in ("biored", "drugprot"):
        schema = load_schema(ROOT / "configs" / "schemas" / f"{corpus}.yaml")
        for level in LEVELS:
            for label in schema.level_labels(level):
                row = interpretations[corpus][level][label.display_name]
                expected = sorted(schema.fine_names[member] for member in label.members)
                assert sorted(row["native_categories"]) == expected
                assert set(REQUIRED_FIELDS) <= set(row)
    questions = " ".join(unresolved_questions())
    assert "Conversion" in questions
    assert "Agonism" in questions
    requirements = entity_extension_requirements()
    assert requirements["current_conditions_retained"] == ["E0", "E1", "E2"]
    assert requirements["resources_not_acquired"] == ["UMLS", "MeSH"]
