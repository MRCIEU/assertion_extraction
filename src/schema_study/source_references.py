"""Candidate source passages from original text and mention offsets.

Passage selection uses sentence boundaries and entity locations. It does not
read the native relation, a model prediction, a novelty flag, or a certainty
judgement. A passage is a candidate location. It is not a finding, and it is
not validated evidence.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.dataset as ds

from schema_study.contexts import INPUT_VERSION
from schema_study.paths import get_paths
from schema_study.records import write_json
from schema_study.schemas import LEVELS, Schema, default_schema_path, load_schema
from schema_study.sentences import (
    ABBREVIATIONS,
    NEIGHBOUR_SENTENCES,
    mention_sentences,
    segment_document,
    segmentation_diagnostics,
)

SOURCE_SAMPLE_SEED = 20261124
CURRENT_VERSIONS = {
    "biored": "biored_candidates_v2",
    "bc8": "bc8_candidates_v2",
    "drugprot": "drugprot_candidates_v1",
}
TARGET_SPLITS = {
    "biored": ("dev", "test"),
    "bc8": ("bc8",),
    "drugprot": ("dev", "eval"),
}
DEVELOPMENT_SPLITS = {"biored": "dev", "drugprot": "dev"}
HUMAN_SPLITS = {"biored": "test", "bc8": "bc8", "drugprot": "eval"}
SAMPLE_SIZE = {
    "development": {"positive": 20, "negative": 5},
    "human": {"positive": 20, "negative": 5},
}
RESERVED_LIMITS = {
    "cross_sentence": 4,
    "repeated_mentions": 3,
    "partial_model_input": 3,
    "multi_label": 2,
}
FRAGMENT_RE = re.compile(r'"fragments": (\[.*?\])')
MODE_RE = re.compile(r'"mode": "([^"]*)"')
CANDIDATE_RE = re.compile(r'"candidate_id": "((?:\\.|[^"\\])*)"')
SPLIT_RE = re.compile(r'"split": "([^"]*)"')


def finding_reference(candidate_id: str, native_label: str | None = None) -> dict:
    """An unfilled finding reference.

    The object can later hold one sentence, several sentences, several
    findings, and alternative locations. This function does not choose them.
    """
    return {
        "candidate_id": candidate_id,
        "native_label": native_label,
        "status": "unidentified",
        "findings": [],
    }


def finding_references_for_labels(candidate_id: str, labels: list[str]) -> list[dict]:
    """One empty reference per positive label, or one empty reference if there is none."""
    if not labels:
        return [finding_reference(candidate_id, None)]
    return [finding_reference(candidate_id, label) for label in labels]


def align_arguments(mention_rows: list[dict], sentences: list[dict]) -> dict:
    """Locate one argument's mentions in the document sentences."""
    located = []
    missing = []
    crossing = []
    for mention in mention_rows:
        sentence_ids = mention_sentences(mention, sentences)
        record = {
            "mention_id": mention["mention_id"],
            "char_start": int(mention["char_start"]),
            "char_end": int(mention["char_end"]),
            "text": mention.get("text", ""),
            "sentence_ids": sentence_ids,
        }
        if not sentence_ids:
            missing.append(record)
        else:
            located.append(record)
            if len(sentence_ids) > 1:
                crossing.append(record)
    return {"mentions": located, "missing": missing, "crossing": crossing}


def build_passages(
    sentences: list[dict],
    first: dict,
    second: dict,
    document_length: int,
    fragments: list[dict] | None = None,
) -> dict:
    """Deterministic passages for two already aligned arguments.

    The relation label is not an input.
    """
    by_id = {row["sentence_id"]: row for row in sentences}
    order = [row["sentence_id"] for row in sentences]
    index = {sentence_id: position for position, sentence_id in enumerate(order)}
    first_ids = _sentence_set(first)
    second_ids = _sentence_set(second)
    shared = [sentence_id for sentence_id in order if sentence_id in first_ids and sentence_id in second_ids]
    intervals = _shortest_intervals(first_ids, second_ids, index)
    supplied = fragments or []
    passages = {
        "document": _passage("document", [order] if order else [[]], by_id, supplied, document_length),
        "both_targets": _passage("both_targets", [[sentence_id] for sentence_id in shared], by_id, supplied, document_length),
        "shortest_interval": _passage(
            "shortest_interval",
            [order[start:end + 1] for start, end in intervals],
            by_id,
            supplied,
            document_length,
        ),
        "neighbouring_sentences": _passage(
            "neighbouring_sentences",
            [_expand_ids(start, end, order) for start, end in intervals],
            by_id,
            supplied,
            document_length,
        ),
    }
    distance = None
    if first_ids and second_ids:
        distance = min(abs(index[left] - index[right]) for left in first_ids for right in second_ids)
    return {
        "sentence_ids_1": sorted(first_ids, key=index.get),
        "sentence_ids_2": sorted(second_ids, key=index.get),
        "both_target_sentence_ids": shared,
        "n_same_sentence_locations": len(shared),
        "min_sentence_distance": distance,
        "n_shortest_intervals": len(intervals),
        "passages": passages,
    }


def select_examples(
    rows: list[dict],
    *,
    n_positive: int,
    n_negative: int,
    seed: int,
    excluded_documents: set[str] | None = None,
    multi_label_limit: int = RESERVED_LIMITS["multi_label"],
) -> dict:
    """Draw a document-disjoint sample after sorting and a seeded shuffle."""
    excluded = set(excluded_documents or ())
    usable = [row for row in rows if row["document_id"] not in excluded]
    positives = sorted((row for row in usable if row["positive"]), key=lambda row: row["candidate_id"])
    negatives = sorted((row for row in usable if not row["positive"]), key=lambda row: row["candidate_id"])
    generator = random.Random(seed)
    generator.shuffle(positives)
    generator.shuffle(negatives)
    selected: list[dict] = []
    used_documents: set[str] = set()
    selected_ids: set[str] = set()
    reservations = [
        ("cross_sentence", RESERVED_LIMITS["cross_sentence"], lambda row: (row.get("min_sentence_distance") or 0) >= 1),
        ("repeated_mentions", RESERVED_LIMITS["repeated_mentions"], lambda row: bool(row.get("repeated_mentions"))),
        ("partial_model_input", RESERVED_LIMITS["partial_model_input"], lambda row: row.get("model_input_mode") in {"single", "dual"}),
        ("multi_label", multi_label_limit, lambda row: bool(row.get("multi_label"))),
    ]
    reservation_counts = {}
    for name, limit, predicate in reservations:
        reservation_counts[name] = _take(positives, predicate, limit, selected, used_documents, selected_ids, n_positive)
    _fill_by_label(positives, selected, used_documents, selected_ids, n_positive)
    document_reuse = False
    if len(selected) < n_positive:
        document_reuse = True
        for row in positives:
            if row["candidate_id"] in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row["candidate_id"])
            if len(selected) >= n_positive:
                break
    negative_rows = []
    for row in negatives:
        if row["document_id"] in used_documents:
            continue
        negative_rows.append(row)
        used_documents.add(row["document_id"])
        if len(negative_rows) >= n_negative:
            break
    return {
        "positive": selected,
        "negative": negative_rows,
        "reservation_counts": reservation_counts,
        "document_reuse": document_reuse,
        "n_positive_shortfall": max(0, n_positive - len(selected)),
        "n_negative_shortfall": max(0, n_negative - len(negative_rows)),
        "document_ids": sorted({row["document_id"] for row in selected + negative_rows}),
    }


def prepare() -> int:
    """Write sentence, alignment, passage, and sample records for the target splits."""
    paths = get_paths()
    destination = paths.assert_output(paths.study / "reports" / "source_references")
    destination.mkdir(parents=True, exist_ok=True)
    schemas = {
        "biored": load_schema(default_schema_path(paths.code_root, "biored")),
        "drugprot": load_schema(default_schema_path(paths.code_root, "drugprot")),
    }
    schemas["bc8"] = schemas["biored"]
    summary = {
        "input_version": INPUT_VERSION,
        "data_versions": CURRENT_VERSIONS,
        "target_splits": {corpus: list(splits) for corpus, splits in TARGET_SPLITS.items()},
        "sentence_segmentation": {
            "method": "deterministic abbreviation and decimal rules, applied separately to the title and the abstract",
            "neighbour_sentences": NEIGHBOUR_SENTENCES,
            "original_text_rewritten": False,
            "abbreviations": sorted(ABBREVIATIONS),
        },
        "sampling": {
            "seed": SOURCE_SAMPLE_SEED,
            "development_splits": DEVELOPMENT_SPLITS,
            "human_splits": HUMAN_SPLITS,
            "per_corpus": SAMPLE_SIZE,
            "reserved_limits": RESERVED_LIMITS,
            "procedure": (
                "Identifiers are sorted, then shuffled with random.Random. Reserved positive "
                "slots are drawn first. The remaining positive slots rotate through native labels. "
                "A document is used once unless the positive quota cannot otherwise be met. "
                "Sample identifiers are assigned after a final sort. The human draw excludes every "
                "development document identifier."
            ),
            "independence": (
                "The human documents are disjoint from the documents used to inspect this "
                "source-identification procedure. They are not independent of earlier scoring: "
                "BioRED test, BC8, and DrugProt eval were already evaluation collections. "
                "DrugProt prior tuning on its official development collection is unconfirmed."
            ),
        },
        "corpora": {},
    }
    feature_rows: dict[str, list[dict]] = {}
    stored: dict[str, dict] = {}
    for corpus in ("biored", "bc8", "drugprot"):
        print(f"Preparing {corpus}.", flush=True)
        corpus_summary, features, corpus_store = _prepare_corpus(paths, destination, corpus, schemas[corpus])
        summary["corpora"][corpus] = corpus_summary
        feature_rows[corpus] = features
        stored[corpus] = corpus_store
    development_documents: set[str] = set()
    samples = {"development": {}, "human": {}}
    for corpus, split in DEVELOPMENT_SPLITS.items():
        draw = select_examples(
            [row for row in feature_rows[corpus] if row["split"] == split],
            n_positive=SAMPLE_SIZE["development"]["positive"],
            n_negative=SAMPLE_SIZE["development"]["negative"],
            seed=SOURCE_SAMPLE_SEED + _seed_offset("development", corpus),
            excluded_documents=development_documents,
            multi_label_limit=RESERVED_LIMITS["multi_label"] if corpus == "drugprot" else 0,
        )
        development_documents.update(draw["document_ids"])
        samples["development"][corpus] = draw
    for corpus, split in HUMAN_SPLITS.items():
        samples["human"][corpus] = select_examples(
            [row for row in feature_rows[corpus] if row["split"] == split],
            n_positive=SAMPLE_SIZE["human"]["positive"],
            n_negative=SAMPLE_SIZE["human"]["negative"],
            seed=SOURCE_SAMPLE_SEED + _seed_offset("human", corpus),
            excluded_documents=development_documents,
            multi_label_limit=RESERVED_LIMITS["multi_label"] if corpus == "drugprot" else 0,
        )
    _write_samples(destination, samples, stored)
    summary["samples"] = {
        name: {corpus: _sample_summary(draw) for corpus, draw in corpus_draws.items()}
        for name, corpus_draws in samples.items()
    }
    summary["illustrations"] = _illustrations(samples["development"], stored)
    summary["finding_references_populated"] = False
    write_json(destination / "summary.json", summary)
    write_json(destination / "finding_reference_schema.json", _finding_schema())
    print(f"Wrote {destination}", flush=True)
    return 0


def _prepare_corpus(paths, destination: Path, corpus: str, schema: Schema):
    splits = set(TARGET_SPLITS[corpus])
    documents = _load_documents(paths, corpus, splits)
    sentences_by_document = {}
    diagnostics = Counter()
    crossing_examples = []
    sentence_path = destination / "sentences" / f"{corpus}.jsonl"
    sentence_path.parent.mkdir(parents=True, exist_ok=True)
    with sentence_path.open("w", encoding="utf-8") as handle:
        for document_id, document in documents.items():
            rows = segment_document(document)
            text = document["text"]
            title_end = int(document["title_char_end"])
            abstract_start = int(document["abstract_char_start"])
            for section, origin, end in (("title", 0, title_end), ("abstract", abstract_start, len(text))):
                diagnostics.update(segmentation_diagnostics(text[origin:end]))
                section_rows = [row for row in rows if row["section"] == section]
                for left, right in zip(section_rows, section_rows[1:]):
                    gap = text[left["char_end"]:right["char_start"]]
                    if gap.strip():
                        diagnostics["nonwhitespace_gap"] += 1
            for row in rows:
                if text[row["char_start"]:row["char_end"]] != row["text"]:
                    raise ValueError(f"Unrecoverable sentence {row['sentence_id']}")
                if row["char_start"] < title_end < row["char_end"] or row["char_start"] < abstract_start < row["char_end"]:
                    diagnostics["separator_inside_sentence"] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            sentences_by_document[document_id] = rows
    mentions = _load_mentions(paths, corpus, documents)
    print(f"  {corpus}: reading model-input spans.", flush=True)
    contexts = _load_contexts(paths, corpus, splits)
    features = []
    population = _empty_distribution()
    positive_candidates = _empty_distribution()
    positive_labels = _empty_distribution()
    label_change = Counter()
    alignment_path = destination / "alignments" / f"{corpus}.jsonl"
    passage_path = destination / "passages" / f"{corpus}.jsonl"
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    passage_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = ds.dataset(paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "candidates.parquet", format="parquet")
    columns = [
        "candidate_id", "corpus", "split", "document_id", "gold_labels", "gold_status",
        "argument_1_id", "argument_2_id", "argument_1_type", "argument_2_type",
        "mention_ids_1", "mention_ids_2", "included_in_scoring",
    ]
    n_written = 0
    with alignment_path.open("w", encoding="utf-8") as alignment_handle, passage_path.open("w", encoding="utf-8") as passage_handle:
        for batch in dataset.to_batches(columns=columns, filter=ds.field("split").isin(sorted(splits))):
            for row in batch.to_pylist():
                document_id = str(row["document_id"])
                document = documents.get(document_id)
                if document is None:
                    diagnostics["candidate_without_document"] += 1
                    continue
                sentences = sentences_by_document[document_id]
                first_mentions, first_absent = _resolve_mentions(row["mention_ids_1"], mentions, document)
                second_mentions, second_absent = _resolve_mentions(row["mention_ids_2"], mentions, document)
                diagnostics["mention_text_mismatch"] += sum(1 for item in first_mentions + second_mentions if not item["text_matches"])
                first = align_arguments(first_mentions, sentences)
                second = align_arguments(second_mentions, sentences)
                for item in first["crossing"] + second["crossing"]:
                    if len(crossing_examples) < 12:
                        crossing_examples.append(
                            {"document_id": document_id, "mention_id": item["mention_id"], "sentence_ids": item["sentence_ids"], "text": item["text"]}
                        )
                built = build_passages(sentences, first, second, len(document["text"]), _fragments(contexts, row["candidate_id"]))
                labels = [label for label in list(row["gold_labels"] or []) if label != schema.no_relation_name]
                alignment = {
                    "candidate_id": row["candidate_id"],
                    "corpus": corpus,
                    "split": row["split"],
                    "document_id": document_id,
                    "identity": "mention" if corpus == "drugprot" else "normalised_concept",
                    "argument_1_id": row["argument_1_id"],
                    "argument_2_id": row["argument_2_id"],
                    "argument_1": first,
                    "argument_2": second,
                    "absent_mention_ids": first_absent + second_absent,
                    "sentence_ids_1": built["sentence_ids_1"],
                    "sentence_ids_2": built["sentence_ids_2"],
                    "both_target_sentence_ids": built["both_target_sentence_ids"],
                    "min_sentence_distance": built["min_sentence_distance"],
                    "n_same_sentence_locations": built["n_same_sentence_locations"],
                    "n_shortest_intervals": built["n_shortest_intervals"],
                    "repeated_mentions": len(first_mentions) > 1 or len(second_mentions) > 1,
                    "missing_alignment": bool(first["missing"] or second["missing"] or first_absent or second_absent),
                    "mention_crosses_sentence": bool(first["crossing"] or second["crossing"]),
                }
                passage_row = {
                    "candidate_id": row["candidate_id"],
                    "corpus": corpus,
                    "split": row["split"],
                    "document_id": document_id,
                    "passages": built["passages"],
                    "distinct_from_model_input": True,
                }
                alignment_handle.write(json.dumps(alignment, ensure_ascii=False) + "\n")
                passage_handle.write(json.dumps(passage_row, ensure_ascii=False) + "\n")
                feature = _feature(row, alignment, built, labels, contexts.get(row["candidate_id"], {}), schema)
                features.append(feature)
                _update_distribution(population, feature)
                if feature["positive"]:
                    _update_distribution(positive_candidates, feature)
                    for change in feature["label_changes"]:
                        _update_distribution(positive_labels, feature)
                        label_change["n"] += 1
                        label_change["unchanged_through_p2"] += int(change["unchanged_through_p2"])
                        label_change["changes_p3_to_p4"] += int(change["changes_p3_to_p4"])
                        label_change["changes_p2_to_p3"] += int(change["changes_p2_to_p3"])
                        label_change[f"distance_{_distance_key(feature)}_unchanged_through_p2"] += int(change["unchanged_through_p2"])
                        label_change[f"distance_{_distance_key(feature)}_changes_p3_to_p4"] += int(change["changes_p3_to_p4"])
                n_written += 1
                if n_written % 50000 == 0:
                    print(f"  {corpus}: {n_written}", flush=True)
    corpus_summary = {
        "n_documents": len(documents),
        "n_sentences": sum(len(rows) for rows in sentences_by_document.values()),
        "n_candidates": n_written,
        "segmentation": dict(diagnostics),
        "mention_crossing_examples": crossing_examples,
        "all_candidates": _finish_distribution(population),
        "positive_candidates": _finish_distribution(positive_candidates),
        "positive_relations": _finish_distribution(positive_labels),
        "relation_representation": dict(label_change),
        "context_records": len(contexts),
    }
    store = {"documents": documents, "sentences": sentences_by_document, "mentions": mentions, "contexts": contexts}
    return corpus_summary, features, store


def _feature(row, alignment, built, labels, context, schema: Schema) -> dict:
    return {
        "candidate_id": row["candidate_id"],
        "document_id": str(row["document_id"]),
        "corpus": row["corpus"],
        "split": row["split"],
        "positive": bool(labels),
        "gold_labels": labels,
        "gold_status": row.get("gold_status"),
        "primary_label": sorted(labels)[0] if labels else "",
        "multi_label": len(labels) > 1,
        "min_sentence_distance": alignment["min_sentence_distance"],
        "n_same_sentence_locations": alignment["n_same_sentence_locations"],
        "n_shortest_intervals": alignment["n_shortest_intervals"],
        "repeated_mentions": alignment["repeated_mentions"],
        "missing_alignment": alignment["missing_alignment"],
        "mention_crosses_sentence": alignment["mention_crosses_sentence"],
        "model_input_mode": context.get("mode", "missing"),
        "shortest_coverage": _single_coverage(built["passages"]["shortest_interval"]),
        "shortest_sentences": _minimum_field(built["passages"]["shortest_interval"], "n_sentences"),
        "shortest_characters": _minimum_field(built["passages"]["shortest_interval"], "n_characters"),
        "neighbour_coverage": _single_coverage(built["passages"]["neighbouring_sentences"]),
        "both_coverage": _single_coverage(built["passages"]["both_targets"]),
        "document_coverage": _single_coverage(built["passages"]["document"]),
        "label_changes": [_label_change(schema, label) for label in labels],
        "argument_1_id": row["argument_1_id"],
        "argument_2_id": row["argument_2_id"],
        "argument_1_type": row["argument_1_type"],
        "argument_2_type": row["argument_2_type"],
        "mention_ids_1": list(row["mention_ids_1"] or []),
        "mention_ids_2": list(row["mention_ids_2"] or []),
    }


def _label_change(schema: Schema, display_name: str) -> dict:
    names = {name: fine_id for fine_id, name in schema.fine_names.items()}
    fine_id = names[display_name]
    members = {}
    for level in LEVELS:
        found = [group for group in schema.level_labels(level) if fine_id in group.members]
        if len(found) != 1:
            raise KeyError(f"{display_name} is not in exactly one {level} group")
        members[level] = frozenset(found[0].members)
    return {
        "label": display_name,
        "changes_p3_to_p4": members["P4"] < members["P3"],
        "changes_p2_to_p3": members["P3"] < members["P2"],
        "unchanged_through_p2": members["P4"] == members["P2"],
    }


def _write_samples(destination: Path, samples: dict, stored: dict) -> None:
    for sample_name, corpus_draws in samples.items():
        review_rows = []
        label_rows = []
        number = 1
        for corpus, draw in corpus_draws.items():
            ordered = sorted(draw["positive"] + draw["negative"], key=lambda row: (row["split"], row["candidate_id"]))
            for feature in ordered:
                packet, label_row = _packet(sample_name, corpus, number, feature, stored[corpus], include_labels=sample_name != "human")
                review_rows.append(packet)
                label_rows.append(label_row)
                number += 1
        review_dir = destination / ("development_sample" if sample_name == "development" else "human_review")
        review_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(review_dir / "examples.jsonl", review_rows)
        label_dir = destination / "label_record"
        label_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(label_dir / f"{sample_name}.jsonl", label_rows)


def _packet(sample_name, corpus, number, feature, store, *, include_labels: bool) -> tuple[dict, dict]:
    document = store["documents"][feature["document_id"]]
    sentences = store["sentences"][feature["document_id"]]
    by_mention = store["mentions"]
    first_rows = [by_mention[mention_id] for mention_id in feature["mention_ids_1"]]
    second_rows = [by_mention[mention_id] for mention_id in feature["mention_ids_2"]]
    first = align_arguments(first_rows, sentences)
    second = align_arguments(second_rows, sentences)
    built = build_passages(sentences, first, second, len(document["text"]), _fragments(store["contexts"], feature["candidate_id"]))
    packet = {
        "sample_id": f"{sample_name}:{corpus}:{number:03d}",
        "corpus": corpus,
        "split": feature["split"],
        "document_id": feature["document_id"],
        "candidate_id": feature["candidate_id"],
        "argument_1_type": feature["argument_1_type"],
        "argument_2_type": feature["argument_2_type"],
        "argument_1_mentions": first["mentions"] + first["missing"],
        "argument_2_mentions": second["mentions"] + second["missing"],
        "sentences": [
            {
                "sentence_id": row["sentence_id"],
                "char_start": row["char_start"],
                "char_end": row["char_end"],
                "text": row["text"],
                "marked_text": _mark_sentence(row, first_rows, second_rows),
            }
            for row in sentences
        ],
        "document_text": document["text"],
        "passages": _passages_with_text(document["text"], built["passages"]),
    }
    if include_labels:
        packet["gold_labels"] = feature["gold_labels"]
        packet["gold_status"] = feature["gold_status"]
    label_row = {
        "sample_id": packet["sample_id"],
        "candidate_id": feature["candidate_id"],
        "corpus": corpus,
        "split": feature["split"],
        "document_id": feature["document_id"],
        "gold_labels": feature["gold_labels"],
        "gold_status": feature["gold_status"],
        "model_predictions_included": False,
        "finding_references": finding_references_for_labels(feature["candidate_id"], feature["gold_labels"]),
    }
    return packet, label_row


def _load_documents(paths, corpus: str, splits: set[str]) -> dict:
    documents = {}
    path = paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "documents.jsonl"
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") in splits:
            documents[str(row["document_id"])] = row
    return documents


def _load_mentions(paths, corpus: str, documents: dict) -> dict:
    mentions = {}
    path = paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "entities.jsonl"
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        document = documents.get(str(row["document_id"]))
        if document is None:
            continue
        observed = document["text"][int(row["char_start"]):int(row["char_end"])]
        row["text_matches"] = observed == row.get("text")
        mentions[row["mention_id"]] = row
    return mentions


def _load_contexts(paths, corpus: str, splits: set[str]) -> dict:
    directory = "target_anchored_v1__biomedbert_e1354b7a"
    if corpus != "drugprot":
        directory += "_proposed"
    path = paths.inputs / corpus / directory / "contexts.jsonl"
    contexts = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            tail = line[-400:]
            if '"split": "train"' in tail or '"split": "unlabeled"' in tail:
                continue
            found_splits = SPLIT_RE.findall(line)
            if not found_splits or found_splits[-1] not in splits:
                continue
            found_ids = CANDIDATE_RE.findall(line)
            found_modes = MODE_RE.findall(line)
            found_fragments = FRAGMENT_RE.search(line)
            if not found_ids or not found_modes or found_fragments is None:
                continue
            contexts[found_ids[-1]] = {
                "mode": found_modes[0],
                "fragments": json.loads(found_fragments.group(1)),
            }
    return contexts


def _resolve_mentions(mention_ids, mentions: dict, document: dict) -> tuple[list[dict], list[str]]:
    found = []
    absent = []
    for mention_id in mention_ids or []:
        row = mentions.get(mention_id)
        if row is None:
            absent.append(mention_id)
        else:
            found.append(row)
    return found, absent


def _fragments(contexts: dict, candidate_id: str) -> list[dict]:
    return list(contexts.get(candidate_id, {}).get("fragments") or [])


def _sentence_set(argument: dict) -> set[str]:
    return {sentence_id for mention in argument["mentions"] for sentence_id in mention["sentence_ids"]}


def _shortest_intervals(first_ids: set[str], second_ids: set[str], index: dict[str, int]) -> list[tuple[int, int]]:
    if not first_ids or not second_ids:
        return []
    pairs = []
    best = None
    for left in first_ids:
        for right in second_ids:
            start, end = sorted((index[left], index[right]))
            length = end - start
            if best is None or length < best:
                best = length
                pairs = [(start, end)]
            elif length == best:
                pairs.append((start, end))
    return sorted(set(pairs))


def _passage(kind: str, selections: list[list[str]], by_id: dict, fragments: list[dict], document_length: int) -> dict:
    alternatives = []
    seen = set()
    for sentence_ids in selections:
        if kind == "document":
            spans = [[0, document_length]]
            key = tuple(map(tuple, spans))
        else:
            if not sentence_ids:
                continue
            rows = [by_id[sentence_id] for sentence_id in sentence_ids]
            spans = [[rows[0]["char_start"], rows[-1]["char_end"]]]
            key = (tuple(sentence_ids), tuple(map(tuple, spans)))
        if key in seen:
            continue
        seen.add(key)
        alternatives.append(
            {
                "sentence_ids": sentence_ids if kind != "document" else list(by_id),
                "spans": spans,
                "n_sentences": len(sentence_ids) if kind != "document" else len(by_id),
                "n_characters": sum(end - start for start, end in spans),
                "model_input": _coverage(spans, fragments),
            }
        )
    return {"kind": kind, "alternatives": alternatives}


def _expand_ids(start: int, end: int, order: list[str]) -> list[str]:
    left = max(0, start - NEIGHBOUR_SENTENCES)
    right = min(len(order) - 1, end + NEIGHBOUR_SENTENCES)
    return order[left:right + 1]


def _coverage(spans: list[list[int]], fragments: list[dict]) -> str:
    if not spans:
        return "none"
    intervals = sorted((int(item["char_start"]), int(item["char_end"])) for item in fragments)
    if not intervals:
        return "none"

    def covered(start: int, end: int) -> bool:
        cursor = start
        for left, right in intervals:
            if right <= cursor:
                continue
            if left > cursor:
                return False
            cursor = max(cursor, right)
            if cursor >= end:
                return True
        return cursor >= end

    def overlaps(start: int, end: int) -> bool:
        return any(start < right and end > left for left, right in intervals)

    if all(covered(start, end) for start, end in spans):
        return "all"
    if any(overlaps(start, end) for start, end in spans):
        return "part"
    return "none"


def _single_coverage(passage: dict) -> str:
    alternatives = passage["alternatives"]
    if not alternatives:
        return "no_passage"
    values = {item["model_input"] for item in alternatives}
    if len(values) == 1:
        return values.pop()
    return "mixed"


def _minimum_field(passage: dict, field: str) -> int | None:
    if not passage["alternatives"]:
        return None
    return min(item[field] for item in passage["alternatives"])


def _empty_distribution() -> dict:
    return {
        "n": 0,
        "missing_alignment": 0,
        "distance": Counter(),
        "cooccurrence": Counter(),
        "several_shortest": 0,
        "crossing": 0,
        "repeated": 0,
        "shortest_characters": [],
        "shortest_sentences": [],
        "shortest_coverage": Counter(),
        "neighbour_coverage": Counter(),
        "both_coverage": Counter(),
        "document_coverage": Counter(),
        "model_input_mode": Counter(),
    }


def _distance_key(feature: dict) -> str:
    if feature["missing_alignment"] or feature["min_sentence_distance"] is None:
        return "missing"
    distance = feature["min_sentence_distance"]
    if distance >= 2:
        return "2plus"
    return str(distance)


def _update_distribution(bucket: dict, feature: dict) -> None:
    bucket["n"] += 1
    bucket["missing_alignment"] += int(feature["missing_alignment"])
    bucket["distance"][_distance_key(feature)] += 1
    if feature["missing_alignment"]:
        cooccurrence = "missing"
    elif feature["n_same_sentence_locations"] == 0:
        cooccurrence = "none"
    elif feature["n_same_sentence_locations"] == 1:
        cooccurrence = "one"
    else:
        cooccurrence = "several"
    bucket["cooccurrence"][cooccurrence] += 1
    bucket["several_shortest"] += int(feature["n_shortest_intervals"] > 1)
    bucket["crossing"] += int(feature["mention_crosses_sentence"])
    bucket["repeated"] += int(feature["repeated_mentions"])
    if feature["shortest_characters"] is not None:
        bucket["shortest_characters"].append(feature["shortest_characters"])
        bucket["shortest_sentences"].append(feature["shortest_sentences"])
    bucket["shortest_coverage"][feature["shortest_coverage"]] += 1
    bucket["neighbour_coverage"][feature["neighbour_coverage"]] += 1
    bucket["both_coverage"][feature["both_coverage"]] += 1
    bucket["document_coverage"][feature["document_coverage"]] += 1
    bucket["model_input_mode"][feature["model_input_mode"]] += 1


def _finish_distribution(bucket: dict) -> dict:
    finished = {}
    for key, value in bucket.items():
        if isinstance(value, Counter):
            finished[key] = dict(value)
        elif key in {"shortest_characters", "shortest_sentences"}:
            finished[key] = _length_summary(value)
        else:
            finished[key] = value
    return finished


def _length_summary(values: list[int]) -> dict:
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "n": len(values),
        "min": ordered[0],
        "median": median,
        "mean": sum(values) / len(values),
        "max": ordered[-1],
    }


def _take(pool, predicate, limit, selected, used, selected_ids, cap) -> int:
    taken = 0
    for row in pool:
        if taken >= limit or len(selected) >= cap:
            break
        if row["candidate_id"] in selected_ids or row["document_id"] in used:
            continue
        if predicate(row):
            selected.append(row)
            selected_ids.add(row["candidate_id"])
            used.add(row["document_id"])
            taken += 1
    return taken


def _fill_by_label(positives, selected, used, selected_ids, n_positive) -> None:
    remaining = n_positive - len(selected)
    buckets = defaultdict(list)
    for row in positives:
        if row["candidate_id"] not in selected_ids:
            buckets[row["primary_label"]].append(row)
    labels = sorted(buckets)
    while remaining > 0 and any(buckets.values()):
        progress = False
        for label in labels:
            if remaining <= 0:
                break
            while buckets[label]:
                row = buckets[label].pop(0)
                if row["candidate_id"] in selected_ids or row["document_id"] in used:
                    continue
                selected.append(row)
                selected_ids.add(row["candidate_id"])
                used.add(row["document_id"])
                remaining -= 1
                progress = True
                break
        if not progress:
            break


def _seed_offset(sample: str, corpus: str) -> int:
    return {"development": 0, "human": 10}[sample] + {"biored": 0, "bc8": 1, "drugprot": 2}[corpus]


def _sample_summary(draw: dict) -> dict:
    rows = draw["positive"] + draw["negative"]
    return {
        "n_positive": len(draw["positive"]),
        "n_negative": len(draw["negative"]),
        "reservation_counts": draw["reservation_counts"],
        "document_reuse": draw["document_reuse"],
        "n_positive_shortfall": draw["n_positive_shortfall"],
        "n_negative_shortfall": draw["n_negative_shortfall"],
        "n_documents": len(draw["document_ids"]),
        "labels": dict(Counter(label for row in draw["positive"] for label in row["gold_labels"])),
        "distance": dict(Counter(_distance_key(row) for row in rows)),
        "repeated_mentions": sum(1 for row in rows if row["repeated_mentions"]),
        "multi_label": sum(1 for row in draw["positive"] if row["multi_label"]),
        "partial_model_input": sum(1 for row in rows if row["model_input_mode"] in {"single", "dual"}),
        "candidate_ids": [row["candidate_id"] for row in sorted(rows, key=lambda row: row["candidate_id"])],
    }


def _illustrations(development: dict, stored: dict) -> list[dict]:
    rows = []
    for draw in development.values():
        rows.extend(draw["positive"])
        rows.extend(draw["negative"])
    specs = (
        (
            "straightforward",
            lambda row: row["min_sentence_distance"] == 0
            and row["n_same_sentence_locations"] == 1
            and not row["repeated_mentions"]
            and not row["mention_crosses_sentence"],
        ),
        (
            "several_locations",
            lambda row: row["n_same_sentence_locations"] >= 2 or row["n_shortest_intervals"] >= 2,
        ),
        ("uncertain_scope", lambda row: (row["shortest_sentences"] or 0) >= 4),
        (
            "cross_sentence",
            lambda row: row["min_sentence_distance"] is not None and row["min_sentence_distance"] >= 1,
        ),
    )
    chosen = []
    used = set()
    for name, predicate in specs:
        match = next((row for row in rows if row["candidate_id"] not in used and predicate(row)), None)
        if match is None:
            chosen.append({"kind": name, "found": False})
            continue
        used.add(match["candidate_id"])
        chosen.append(_illustration(name, match, stored[match["corpus"]]))
    return chosen


def _illustration(kind: str, feature: dict, store: dict) -> dict:
    document = store["documents"][feature["document_id"]]
    sentences = store["sentences"][feature["document_id"]]
    by_mention = store["mentions"]
    first = align_arguments([by_mention[mention_id] for mention_id in feature["mention_ids_1"]], sentences)
    second = align_arguments([by_mention[mention_id] for mention_id in feature["mention_ids_2"]], sentences)
    built = build_passages(sentences, first, second, len(document["text"]), _fragments(store["contexts"], feature["candidate_id"]))
    rendered = _passages_with_text(document["text"], built["passages"])
    shortest = rendered["shortest_interval"]["alternatives"]
    return {
        "kind": kind,
        "found": True,
        "corpus": feature["corpus"],
        "split": feature["split"],
        "document_id": feature["document_id"],
        "candidate_id": feature["candidate_id"],
        "gold_labels": feature["gold_labels"],
        "gold_used_for_passage_selection": False,
        "min_sentence_distance": feature["min_sentence_distance"],
        "n_same_sentence_locations": feature["n_same_sentence_locations"],
        "n_shortest_intervals": feature["n_shortest_intervals"],
        "repeated_mentions": feature["repeated_mentions"],
        "model_input_mode": feature["model_input_mode"],
        "shortest_coverage": feature["shortest_coverage"],
        "shortest_excerpts": [alternative["text_spans"] for alternative in shortest[:3]],
    }


def _passages_with_text(text: str, passages: dict) -> dict:
    rendered = {}
    for kind, passage in passages.items():
        alternatives = []
        for alternative in passage["alternatives"]:
            copied = dict(alternative)
            copied["text_spans"] = [text[start:end] for start, end in alternative["spans"]]
            alternatives.append(copied)
        rendered[kind] = {"kind": kind, "alternatives": alternatives}
    return rendered


def _mark_sentence(sentence: dict, first_rows: list[dict], second_rows: list[dict]) -> str:
    text = sentence["text"]
    origin = int(sentence["char_start"])
    inserts = []
    for mention, opening, closing in (
        *[(mention, "[1]", "[/1]") for mention in first_rows],
        *[(mention, "[2]", "[/2]") for mention in second_rows],
    ):
        start = max(int(mention["char_start"]), origin)
        end = min(int(mention["char_end"]), int(sentence["char_end"]))
        if start < end:
            inserts.append((start - origin, end - origin, opening, closing))
    inserts.sort()
    if any(right > next_left for (_, right, _, _), (next_left, _, _, _) in zip(inserts, inserts[1:])):
        return text
    pieces = []
    cursor = 0
    for start, end, opening, closing in inserts:
        pieces.append(text[cursor:start])
        pieces.append(opening + text[start:end] + closing)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _finding_schema() -> dict:
    return {
        "status_values": ["unidentified", "ambiguous", "located"],
        "finding_fields": ["finding_id", "sentence_ids", "spans", "alternatives"],
        "can_represent": {
            "one_sentence": {"sentence_ids": ["document:abstract:0000"], "spans": [[0, 10]]},
            "several_sentences": {"sentence_ids": ["document:abstract:0000", "document:abstract:0001"]},
            "several_findings": "a list of finding objects on one reference",
            "alternative_locations": "a list of alternative sentence-id sets on one finding",
            "ambiguous_or_unidentified": "status unidentified or ambiguous, with an empty finding list when no location has been judged",
            "drugprot_multi_label": "one finding reference for each native label",
        },
        "populated": False,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
