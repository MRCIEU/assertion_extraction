"""Readers, document splits, and candidate inventories.

Official files stay read-only. A candidate id does not contain a model, an
entity level, a predicate level, or a prediction.
"""

from __future__ import annotations

import random
import urllib.parse
from collections import defaultdict
from pathlib import Path

from schema_study.records import digest_file, write_json, write_jsonl, write_parquet, write_tsv
from schema_study.schemas import Schema
from schema_study.validation import ValidationReport

BIORED_LABELS = {
    "Positive_Correlation",
    "Negative_Correlation",
    "Association",
    "Bind",
    "Comparison",
    "Cotreatment",
    "Drug_Interaction",
    "Conversion",
}
SEPARATOR_CANDIDATES = (" ", "\n")
AFFECTING = {
    "label_conflict",
    "missing_concept",
    "type_conflict",
    "unsupported_type_pair",
    "mention_text_mismatch",
}


def split_concept_ids(raw: str) -> list[str]:
    """Split an official composite identifier and keep every component.

    BioRED writes several concept ids in one mention field, separated by commas.
    Relations cite those components individually. ``-`` means the mention has no
    concept id and is not merged with any other unlinked mention.
    """
    text = (raw or "").strip()
    if text in {"", "-"}:
        return []
    return [part.strip() for part in text.split(",") if part.strip() and part.strip() != "-"]

CANDIDATE_COLUMNS = (
    "candidate_id",
    "corpus",
    "split",
    "document_id",
    "argument_1_id",
    "argument_2_id",
    "argument_1_type",
    "argument_2_type",
    "argument_1_role",
    "argument_2_role",
    "gold_labels",
    "gold_status",
    "included_in_scoring",
    "mention_ids_1",
    "mention_ids_2",
)


def quote_id(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def biored_candidate_id(document_id: str, left: str, right: str) -> str:
    first, second = sorted((left, right))
    return f"biored:{document_id}:{quote_id(first)}:{quote_id(second)}"


def drugprot_candidate_id(document_id: str, chemical_term: str, gene_term: str) -> str:
    return f"drugprot:{document_id}:{quote_id(chemical_term)}:{quote_id(gene_term)}"


def parse_pubtator(text: str) -> list[dict]:
    documents = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        title = abstract = None
        document_id = None
        entities = []
        relations = []
        direction_notes = []
        unknown_lines = []
        for line in block.splitlines():
            if "|t|" in line and line.find("|t|") > 0:
                document_id, title = line.split("|t|", 1)
                continue
            if "|a|" in line and line.find("|a|") > 0:
                document_id, abstract = line.split("|a|", 1)
                continue
            parts = line.split("\t")
            if len(parts) >= 6 and parts[1].isdigit():
                entities.append(
                    {
                        "document_id": parts[0],
                        "start": int(parts[1]),
                        "end": int(parts[2]),
                        "text": parts[3],
                        "entity_type": parts[4],
                        "concept_id": parts[5],
                    }
                )
            elif len(parts) >= 4 and parts[1] in BIORED_LABELS:
                relations.append(
                    {
                        "document_id": parts[0],
                        "label": parts[1],
                        "concept_id_1": parts[2],
                        "concept_id_2": parts[3],
                        "novelty": parts[4] if len(parts) > 4 else None,
                    }
                )
            elif len(parts) >= 4 and parts[-1].startswith("Subject:"):
                direction_notes.append(
                    {
                        "document_id": parts[0],
                        "concept_id_1": parts[1],
                        "concept_id_2": parts[2],
                        "subject": parts[-1].split(":", 1)[1],
                    }
                )
            elif line.strip():
                unknown_lines.append(line)
        documents.append(
            {
                "document_id": document_id,
                "title": title or "",
                "abstract": abstract or "",
                "entities": entities,
                "relations": relations,
                "direction_notes": direction_notes,
                "unknown_lines": unknown_lines,
            }
        )
    return documents


def choose_fixed_separator(documents: list[dict]) -> dict:
    """Pick one title/abstract join from the release offsets.

    The choice is a property of the release, not of an entity or predicate level.
    A single space is used when every length-1 separator matches.
    """
    scores = {separator: 0 for separator in SEPARATOR_CANDIDATES}
    mismatches = {separator: 0 for separator in SEPARATOR_CANDIDATES}
    for document in documents:
        for separator in SEPARATOR_CANDIDATES:
            text = document["title"] + separator + document["abstract"]
            for entity in document["entities"]:
                if text[entity["start"]:entity["end"]] == entity["text"]:
                    scores[separator] += 1
                else:
                    mismatches[separator] += 1
    perfect = [separator for separator in SEPARATOR_CANDIDATES if mismatches[separator] == 0]
    if " " in perfect:
        chosen = " "
        reason = "space_matches_all_mentions"
    elif perfect:
        chosen = perfect[0]
        reason = "only_nonspace_separator_matches"
    else:
        chosen = " "
        reason = "no_separator_matches_all_mentions"
    return {
        "separator": chosen,
        "reason": reason,
        "mismatch_counts": mismatches,
        "match_counts": scores,
        "ambiguous_length": mismatches.get(" ", 1) == 0 and mismatches.get("\n", 1) == 0,
    }


def apply_separator(documents: list[dict], separator: str) -> list[dict]:
    prepared = []
    for document in documents:
        text = document["title"] + separator + document["abstract"]
        copied = dict(document)
        copied["text"] = text
        copied["title_char_end"] = len(document["title"])
        copied["abstract_char_start"] = len(document["title"]) + len(separator)
        copied["separator"] = separator
        prepared.append(copied)
    return prepared


def legal_pair(schema: Schema, left_type: str, right_type: str) -> bool:
    allowed = {frozenset(pair) for pair in schema.raw.get("legal_type_pairs", [])}
    return frozenset((left_type, right_type)) in allowed


def build_biored_document(document: dict, schema: Schema, split: str, report: ValidationReport) -> dict:
    concepts: dict[str, dict] = {}
    entities = []
    exclusions = []
    for index, mention in enumerate(document["entities"]):
        observed = document["text"][mention["start"]:mention["end"]]
        mention_id = f"{document['document_id']}:{mention['start']}:{mention['end']}:{quote_id(mention['concept_id'])}"
        if observed != mention["text"]:
            _exclude(exclusions, document, split, "mention_text_mismatch", detail=f"{mention['start']}:{mention['end']}")
            report.add("error", "mention_text_mismatch", "Mention text does not match the fixed document join.")
            continue
        concept_ids = split_concept_ids(mention["concept_id"])
        if not concept_ids:
            _exclude(exclusions, document, split, "no_concept_identifier", detail=mention["text"])
            report.add("warning", "no_concept_identifier", "Mention has no concept id and is not used as a normalized concept.")
            entities.append(
                {
                    "document_id": document["document_id"],
                    "mention_id": f"{document['document_id']}:{mention['start']}:{mention['end']}:none",
                    "char_start": mention["start"],
                    "char_end": mention["end"],
                    "text": mention["text"],
                    "entity_type": mention["entity_type"],
                    "concept_id": "",
                    "raw_identifier": mention["concept_id"],
                    "text_matches": True,
                }
            )
            continue
        for concept_id in concept_ids:
            mention_id = f"{document['document_id']}:{mention['start']}:{mention['end']}:{quote_id(concept_id)}"
            entities.append(
                {
                    "document_id": document["document_id"],
                    "mention_id": mention_id,
                    "char_start": mention["start"],
                    "char_end": mention["end"],
                    "text": mention["text"],
                    "entity_type": mention["entity_type"],
                    "concept_id": concept_id,
                    "raw_identifier": mention["concept_id"],
                    "text_matches": True,
                }
            )
            bucket = concepts.setdefault(
                concept_id,
                {"concept_id": concept_id, "types": set(), "mention_ids": []},
            )
            bucket["types"].add(mention["entity_type"])
            bucket["mention_ids"].append(mention_id)
    usable = {}
    for concept_id, concept in concepts.items():
        if len(concept["types"]) != 1:
            _exclude(
                exclusions,
                document,
                split,
                "type_conflict",
                concept_id_1=concept_id,
                detail=",".join(sorted(concept["types"])),
            )
            report.add("error", "type_conflict", "One concept id has more than one entity type.")
            continue
        usable[concept_id] = {
            "concept_id": concept_id,
            "entity_type": next(iter(concept["types"])),
            "mention_ids": concept["mention_ids"],
        }
    grouped_relations = defaultdict(list)
    for relation in document["relations"]:
        left, right = relation["concept_id_1"], relation["concept_id_2"]
        if left not in usable or right not in usable:
            _exclude(
                exclusions,
                document,
                split,
                "missing_concept",
                concept_id_1=left,
                concept_id_2=right,
                labels=relation["label"],
            )
            report.add("error", "missing_concept", "Relation endpoint is not a usable concept.")
            continue
        if left == right:
            _exclude(exclusions, document, split, "self_relation", concept_id_1=left, labels=relation["label"])
            report.add("warning", "self_relation", "Self relation was not turned into a candidate.")
            continue
        if not legal_pair(schema, usable[left]["entity_type"], usable[right]["entity_type"]):
            _exclude(
                exclusions,
                document,
                split,
                "unsupported_type_pair",
                concept_id_1=left,
                concept_id_2=right,
                labels=relation["label"],
                detail=f"{usable[left]['entity_type']}|{usable[right]['entity_type']}",
            )
            report.add("error", "unsupported_type_pair", "Gold relation uses a type combination outside the contract.")
            continue
        key = tuple(sorted((left, right)))
        grouped_relations[key].append(relation)
    candidates = []
    concept_ids = sorted(usable)
    for left_index, left in enumerate(concept_ids):
        for right in concept_ids[left_index + 1:]:
            if not legal_pair(schema, usable[left]["entity_type"], usable[right]["entity_type"]):
                continue
            key = (left, right)
            relations = grouped_relations.get(key, [])
            labels = sorted({item["label"] for item in relations})
            if len(labels) > 1:
                status = "conflict"
                included = False
                _exclude(
                    exclusions,
                    document,
                    split,
                    "label_conflict",
                    concept_id_1=left,
                    concept_id_2=right,
                    labels="|".join(labels),
                )
                report.add("error", "label_conflict", "One concept pair has more than one positive label.")
            elif len(labels) == 1:
                status = "unique"
                included = True
            else:
                status = "no_relation"
                included = True
            first, second = key
            candidates.append(
                {
                    "candidate_id": biored_candidate_id(document["document_id"], first, second),
                    "corpus": schema.corpus,
                    "split": split,
                    "document_id": document["document_id"],
                    "argument_1_id": first,
                    "argument_2_id": second,
                    "argument_1_type": usable[first]["entity_type"],
                    "argument_2_type": usable[second]["entity_type"],
                    "argument_1_role": "H",
                    "argument_2_role": "T",
                    "gold_labels": labels,
                    "gold_status": status,
                    "included_in_scoring": included,
                    "mention_ids_1": usable[first]["mention_ids"],
                    "mention_ids_2": usable[second]["mention_ids"],
                }
            )
    return {
        "document": {
            "document_id": document["document_id"],
            "corpus": schema.corpus,
            "split": split,
            "title": document["title"],
            "abstract": document["abstract"],
            "text": document["text"],
            "title_char_end": document["title_char_end"],
            "abstract_char_start": document["abstract_char_start"],
            "separator": document["separator"],
        },
        "entities": entities,
        "relations": document["relations"],
        "direction_notes": document["direction_notes"],
        "candidates": candidates,
        "exclusions": exclusions,
        "unknown_lines": document["unknown_lines"],
    }


def _exclude(rows: list[dict], document: dict, split: str, reason: str, **fields) -> None:
    rows.append(
        {
            "document_id": document["document_id"],
            "split": split,
            "reason": reason,
            "concept_id_1": fields.get("concept_id_1", ""),
            "concept_id_2": fields.get("concept_id_2", ""),
            "labels": fields.get("labels", ""),
            "detail": fields.get("detail", ""),
        }
    )


def prepare_pubtator_corpus(
    documents: list[dict],
    schema: Schema,
    *,
    corpus: str,
    release_id: str,
    source_files: list[Path],
    output_dir: Path,
    split_of,
) -> dict:
    report = ValidationReport(name=f"{corpus}_preparation")
    separator_info = choose_fixed_separator(documents)
    if separator_info["reason"] == "no_separator_matches_all_mentions":
        report.add("error", "separator_unresolved", "No fixed join reproduces every mention offset.")
    joined = apply_separator(documents, separator_info["separator"])
    prepared_docs = []
    entities = []
    relations = []
    candidates = []
    exclusions = []
    direction_notes = []
    unknown = 0
    seen_ids = defaultdict(list)
    for document in joined:
        split = split_of(document["document_id"])
        seen_ids[split].append(document["document_id"])
        built = build_biored_document(document, schema, split, report)
        prepared_docs.append(built["document"])
        entities.extend(built["entities"])
        for relation in built["relations"]:
            relations.append({**relation, "split": split, "corpus": corpus})
        candidates.extend(built["candidates"])
        exclusions.extend(built["exclusions"])
        direction_notes.extend(built["direction_notes"])
        unknown += len(built["unknown_lines"])
    if unknown:
        report.add("warning", "unknown_pubtator_line", "A PubTator line was neither an entity, a known relation, nor a Subject note.", count=unknown)
    overlap = _split_overlap(seen_ids)
    if overlap:
        report.add("error", "split_overlap", f"Document ids occur in more than one split: {overlap[:10]}")
    affecting = [row for row in exclusions if row["reason"] in AFFECTING and row["split"] == "train"]
    formally_trainable = report.error_count == 0 and not affecting
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "documents.jsonl", prepared_docs)
    write_jsonl(output_dir / "entities.jsonl", entities)
    write_jsonl(output_dir / "relations.jsonl", relations)
    write_jsonl(output_dir / "direction_notes.jsonl", direction_notes)
    write_parquet(output_dir / "candidates.parquet", candidates, CANDIDATE_COLUMNS)
    write_tsv(
        output_dir / "splits.tsv",
        [{"split": split, "document_id": document_id} for split, ids in sorted(seen_ids.items()) for document_id in ids],
        ["split", "document_id"],
    )
    write_tsv(
        output_dir / "exclusions.tsv",
        exclusions,
        ["document_id", "split", "reason", "concept_id_1", "concept_id_2", "labels", "detail"],
    )
    summary = {
        "corpus": corpus,
        "release_id": release_id,
        "data_version": f"{corpus}_candidates_v2",
        "formally_trainable": formally_trainable,
        "held_out_first_contact": False,
        "separator": separator_info,
        "n_documents": len(prepared_docs),
        "n_documents_by_split": {split: len(ids) for split, ids in seen_ids.items()},
        "n_entities": len(entities),
        "n_relations": len(relations),
        "n_candidates": len(candidates),
        "n_scoring_candidates": sum(1 for row in candidates if row["included_in_scoring"]),
        "n_exclusions": len(exclusions),
        "exclusion_counts": _count_reasons(exclusions),
        "label_counts": _count_labels(relations),
    }
    write_json(output_dir / "data_summary.json", summary)
    write_json(output_dir / "validation_report.json", report.as_dict())
    write_json(
        output_dir / "provenance.json",
        {
            "corpus": corpus,
            "release_id": release_id,
            "sources": [{"path": str(path), "sha256": digest_file(path)} for path in source_files],
            "separator": separator_info,
            "candidate_id_rule": "biored:{document}:{quoted_sorted_concept_id}:{quoted_sorted_concept_id}",
            "composite_identifier_rule": "comma-separated official ids are kept as separate concept links on the same mention",
            "direction_notes_are_relations": False,
        },
    )
    summary["validation_status"] = report.as_dict()["status"]
    return summary


def _split_overlap(seen: dict[str, list[str]]) -> list[str]:
    owners = defaultdict(set)
    for split, ids in seen.items():
        for document_id in ids:
            owners[document_id].add(split)
    return sorted(document_id for document_id, splits in owners.items() if len(splits) > 1)


def _count_reasons(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["reason"]] = counts.get(row["reason"], 0) + 1
    return counts


def _count_labels(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    return counts


def read_tsv(path: Path) -> list[list[str]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(line.rstrip("\n").split("\t"))
    return rows


def split_drugprot_training(document_ids: list[str], *, seed: int = 42, n_train: int = 2800) -> tuple[list[str], list[str]]:
    """Document-level split of the official training collection.

    Identifiers are sorted numerically before `random.Random(seed)` shuffles them.
    """
    ordered = sorted(document_ids, key=lambda value: (int(value), value))
    if len(ordered) != len(set(ordered)):
        raise ValueError("Duplicate DrugProt training document ids")
    if len(ordered) != 3500:
        raise ValueError(f"Official DrugProt training count is {len(ordered)}, not 3500. No alternative split was created.")
    shuffled = ordered[:]
    random.Random(seed).shuffle(shuffled)
    return shuffled[:n_train], shuffled[n_train:]


def prepare_drugprot(
    training: dict,
    evaluation: dict,
    schema: Schema,
    *,
    release_id: str,
    source_files: list[Path],
    output_dir: Path,
    unlabeled: dict | None = None,
) -> dict:
    report = ValidationReport(name="drugprot_preparation")
    try:
        train_ids, dev_ids = split_drugprot_training(list(training["documents"]))
    except ValueError as exc:
        report.add("error", "drugprot_split", str(exc))
        train_ids, dev_ids = [], []
    split_of = {}
    for document_id in train_ids:
        split_of[document_id] = "train"
    for document_id in dev_ids:
        split_of[document_id] = "dev"
    for document_id in evaluation["documents"]:
        split_of[document_id] = "eval"
    if unlabeled:
        for document_id in unlabeled["documents"]:
            split_of[document_id] = "unlabeled"
    overlap = [document_id for document_id, split in split_of.items() if split == "eval" and document_id in set(train_ids) | set(dev_ids)]
    if overlap:
        report.add("error", "split_overlap", "Official development documents overlap the training collection.")
    bundles = [("train_source", training), ("eval_source", evaluation)]
    if unlabeled:
        bundles.append(("unlabeled_source", unlabeled))
    separator_info = choose_fixed_separator(
        [
            {"title": bundle[1]["documents"][document_id]["title"], "abstract": bundle[1]["documents"][document_id]["abstract"], "entities": bundle[1]["entities"].get(document_id, [])}
            for bundle in bundles
            for document_id in bundle[1]["documents"]
        ]
    )
    documents = []
    entities = []
    relations = []
    exclusions = []
    n_candidates = 0
    n_scoring_candidates = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_writer = None
    candidate_batch: list[dict] = []

    def flush_candidates() -> None:
        nonlocal candidate_writer
        if not candidate_batch:
            return
        candidate_writer = _append_parquet(output_dir / "candidates.parquet", candidate_batch, CANDIDATE_COLUMNS, candidate_writer)
        candidate_batch.clear()

    for _name, bundle in bundles:
        for document_id, document in bundle["documents"].items():
            split = split_of.get(document_id, "unassigned")
            text = document["title"] + separator_info["separator"] + document["abstract"]
            documents.append(
                {
                    "document_id": document_id,
                    "corpus": "drugprot",
                    "split": split,
                    "title": document["title"],
                    "abstract": document["abstract"],
                    "text": text,
                    "title_char_end": len(document["title"]),
                    "abstract_char_start": len(document["title"]) + len(separator_info["separator"]),
                    "separator": separator_info["separator"],
                }
            )
            mention_rows, candidate_rows, exclusion_rows, relation_rows = _drugprot_document(
                document_id,
                text,
                bundle["entities"].get(document_id, []),
                bundle["relations"].get(document_id, []),
                split,
                schema,
                report,
            )
            entities.extend(mention_rows)
            exclusions.extend(exclusion_rows)
            relations.extend(relation_rows)
            n_candidates += len(candidate_rows)
            n_scoring_candidates += sum(1 for row in candidate_rows if row["included_in_scoring"])
            candidate_batch.extend(candidate_rows)
            if len(candidate_batch) >= 20000:
                flush_candidates()
    flush_candidates()
    if candidate_writer is not None:
        candidate_writer.close()
    affecting = [row for row in exclusions if row["reason"] in AFFECTING and row["split"] == "train"]
    formally_trainable = report.error_count == 0 and not affecting and len(train_ids) == 2800
    write_jsonl(output_dir / "documents.jsonl", documents)
    write_jsonl(output_dir / "entities.jsonl", entities)
    write_jsonl(output_dir / "relations.jsonl", relations)
    write_tsv(
        output_dir / "splits.tsv",
        [{"split": split_of[document_id], "document_id": document_id} for document_id in sorted(split_of, key=lambda value: (split_of[value], int(value)))],
        ["split", "document_id"],
    )
    write_tsv(
        output_dir / "exclusions.tsv",
        exclusions,
        ["document_id", "split", "reason", "concept_id_1", "concept_id_2", "labels", "detail"],
    )
    summary = {
        "corpus": "drugprot",
        "release_id": release_id,
        "data_version": "drugprot_candidates_v1",
        "formally_trainable": formally_trainable,
        "held_out_first_contact": None,
        "prior_tuning_use_confirmed": False,
        "separator": separator_info,
        "n_documents": len(documents),
        "n_train": len(train_ids),
        "n_dev": len(dev_ids),
        "n_eval": sum(1 for split in split_of.values() if split == "eval"),
        "n_unlabeled": sum(1 for split in split_of.values() if split == "unlabeled"),
        "n_relations": len(relations),
        "n_candidates": n_candidates,
        "n_scoring_candidates": n_scoring_candidates,
        "exclusion_counts": _count_reasons(exclusions),
        "label_counts": _count_labels(relations),
        "unlabeled_used_for_scoring": False,
    }
    write_json(output_dir / "data_summary.json", summary)
    write_json(output_dir / "validation_report.json", report.as_dict())
    write_json(
        output_dir / "provenance.json",
        {
            "corpus": "drugprot",
            "release_id": release_id,
            "sources": [{"path": str(path), "sha256": digest_file(path)} for path in source_files],
            "split_seed": 42,
            "split_rule": "numeric PMID sort, random.Random(42).shuffle, first 2800 train, remaining official-training documents dev",
            "evaluation_split": "official development set of 750 documents",
            "candidate_id_rule": "drugprot:{document}:{quoted_chemical_term}:{quoted_gene_term}",
        },
    )
    summary["validation_status"] = report.as_dict()["status"]
    return summary


def _drugprot_document(document_id, text, mentions, relations, split, schema, report):
    entity_rows = []
    by_term = {}
    exclusions = []
    for mention in mentions:
        observed = text[mention["start"]:mention["end"]] if 0 <= mention["start"] <= mention["end"] <= len(text) else None
        original_type = mention["entity_type"]
        unified = "GENE" if original_type in {"GENE-Y", "GENE-N", "GENE"} else original_type
        mention_id = f"{document_id}:{mention['term']}"
        entity_rows.append(
            {
                "document_id": document_id,
                "mention_id": mention_id,
                "term": mention["term"],
                "char_start": mention["start"],
                "char_end": mention["end"],
                "text": mention["text"],
                "entity_type_original": original_type,
                "entity_type": unified,
                "text_matches": observed == mention["text"],
            }
        )
        if observed != mention["text"]:
            exclusions.append(_drugprot_exclusion(document_id, split, "mention_text_mismatch", mention["term"], "", "", mention_id))
            report.add("error", "mention_text_mismatch", "DrugProt mention text does not match the fixed join.")
            continue
        by_term[mention["term"]] = {**mention, "unified_type": unified, "mention_id": mention_id}
    chemicals = [row for row in by_term.values() if row["unified_type"] == "CHEMICAL"]
    genes = [row for row in by_term.values() if row["unified_type"] == "GENE"]
    gold = defaultdict(list)
    relation_rows = []
    for relation in relations:
        chemical = relation["chemical_term"]
        gene = relation["gene_term"]
        if chemical not in by_term or gene not in by_term:
            exclusions.append(_drugprot_exclusion(document_id, split, "missing_concept", chemical, gene, relation["label"], ""))
            report.add("error", "missing_concept", "DrugProt relation endpoint is missing.")
            continue
        if by_term[chemical]["unified_type"] != "CHEMICAL" or by_term[gene]["unified_type"] != "GENE":
            exclusions.append(_drugprot_exclusion(document_id, split, "unsupported_type_pair", chemical, gene, relation["label"], ""))
            report.add("error", "unsupported_type_pair", "DrugProt relation is not a chemical-gene mention pair.")
            continue
        gold[(chemical, gene)].append(relation["label"])
        relation_rows.append(
            {
                "document_id": document_id,
                "corpus": "drugprot",
                "split": split,
                "label": relation["label"],
                "concept_id_1": chemical,
                "concept_id_2": gene,
                "novelty": None,
            }
        )
    candidates = []
    scoring = split != "unlabeled"
    for chemical in chemicals:
        for gene in genes:
            labels = sorted(set(gold.get((chemical["term"], gene["term"]), [])))
            known = {item["display_name"] for item in schema.raw["fine_labels"]}
            unexpected = [label for label in labels if label not in known]
            if unexpected:
                exclusions.append(_drugprot_exclusion(document_id, split, "unknown_label", chemical["term"], gene["term"], "|".join(unexpected), ""))
                report.add("error", "unknown_label", "DrugProt relation label is outside the 13 official classes.")
                status = "conflict"
                included = False
            elif not labels:
                status = "no_relation"
                included = scoring
            elif len(labels) == 1:
                status = "unique"
                included = scoring
            else:
                status = "multi"
                included = scoring
            if not scoring:
                status = "unlabeled"
                included = False
            candidates.append(
                {
                    "candidate_id": drugprot_candidate_id(document_id, chemical["term"], gene["term"]),
                    "corpus": "drugprot",
                    "split": split,
                    "document_id": document_id,
                    "argument_1_id": chemical["term"],
                    "argument_2_id": gene["term"],
                    "argument_1_type": "CHEMICAL",
                    "argument_2_type": "GENE",
                    "argument_1_role": "H",
                    "argument_2_role": "T",
                    "gold_labels": labels,
                    "gold_status": status,
                    "included_in_scoring": included,
                    "mention_ids_1": [chemical["mention_id"]],
                    "mention_ids_2": [gene["mention_id"]],
                }
            )
    return entity_rows, candidates, exclusions, relation_rows


def _candidate_arrow_schema():
    import pyarrow as pa

    fields = []
    for name in CANDIDATE_COLUMNS:
        if name in {"gold_labels", "mention_ids_1", "mention_ids_2"}:
            fields.append(pa.field(name, pa.list_(pa.string())))
        elif name == "included_in_scoring":
            fields.append(pa.field(name, pa.bool_()))
        else:
            fields.append(pa.field(name, pa.string()))
    return pa.schema(fields)


def _append_parquet(path: Path, rows: list[dict], columns, writer):
    """Append one batch. The caller closes the returned writer."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _candidate_arrow_schema()
    table = pa.Table.from_pylist([{name: row.get(name) for name in columns} for row in rows], schema=schema)
    if writer is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        writer = pq.ParquetWriter(path, schema)
    writer.write_table(table)
    return writer


def _drugprot_exclusion(document_id, split, reason, left, right, labels, detail) -> dict:
    return {
        "document_id": document_id,
        "split": split,
        "reason": reason,
        "concept_id_1": left,
        "concept_id_2": right,
        "labels": labels,
        "detail": detail,
    }


def load_drugprot_directory(directory: Path) -> dict:
    """Load one DrugProt folder of abstracts, entities, and optional relations."""
    files = [path for path in Path(directory).rglob("*.tsv")]
    abstracts = _one_matching(files, "abstrac")
    entities = _one_matching(files, "entit")
    relations = _one_matching(files, "relation", required=False)
    documents = {}
    for row in read_tsv(abstracts):
        if len(row) < 3:
            continue
        if not row[0].isdigit():
            continue
        documents[row[0]] = {"title": row[1], "abstract": row[2]}
    entity_map = defaultdict(list)
    for row in read_tsv(entities):
        if len(row) < 6:
            continue
        if not row[0].isdigit() or not row[3].isdigit() or not row[4].isdigit():
            continue
        entity_map[row[0]].append(
            {
                "document_id": row[0],
                "term": row[1],
                "entity_type": row[2],
                "start": int(row[3]),
                "end": int(row[4]),
                "text": row[5],
            }
        )
    relation_map = defaultdict(list)
    if relations is not None:
        for row in read_tsv(relations):
            if len(row) < 4 or not row[0].isdigit():
                continue
            relation_map[row[0]].append(
                {
                    "label": row[1].strip(),
                    "chemical_term": row[2].split(":", 1)[-1].strip(),
                    "gene_term": row[3].split(":", 1)[-1].strip(),
                }
            )
    return {"documents": documents, "entities": entity_map, "relations": relation_map, "files": [abstracts, entities] + ([relations] if relations else [])}


def _one_matching(files: list[Path], token: str, required: bool = True) -> Path | None:
    matches = [path for path in files if token in path.name.lower()]
    if len(matches) != 1:
        if required:
            raise FileNotFoundError(f"Expected one DrugProt file containing {token!r}, found {[path.name for path in matches]}")
        return None
    return matches[0]
