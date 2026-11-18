"""Trace each raw relation line to one primary destination.

The order is identity, self-relation, type scope, exact duplicates, then
label conflict or inclusion. Additional reasons are stored separately and are
not added into the exclusion total.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from schema_study.data import (
    BIORED_LABELS,
    biored_candidate_id,
    drugprot_candidate_id,
    legal_pair,
    load_drugprot_directory,
    split_concept_ids,
    split_drugprot_training,
)
from schema_study.schemas import Schema

LINEAGE_COLUMNS = (
    "source_file",
    "line_number",
    "document_id",
    "endpoint_1",
    "endpoint_2",
    "label",
    "primary_destination",
    "primary_reason",
    "additional_reasons",
    "candidate_id",
)


def _concepts(document: dict) -> tuple[dict, dict]:
    """Return usable concepts and ids dropped for a type conflict."""
    grouped: dict[str, set[str]] = defaultdict(set)
    for mention in document["entities"]:
        for concept_id in split_concept_ids(mention["concept_id"]):
            grouped[concept_id].add(mention["entity_type"])
    usable = {}
    conflicts = {}
    for concept_id, types in grouped.items():
        if len(types) == 1:
            usable[concept_id] = next(iter(types))
        else:
            conflicts[concept_id] = types
    return usable, conflicts


def _pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def trace_biored_file(path: Path, schema: Schema, split: str) -> list[dict]:
    documents: dict[str, dict] = {}
    current = None
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                current = None
                continue
            if "|t|" in stripped and stripped.find("|t|") > 0:
                document_id, title = stripped.split("|t|", 1)
                current = documents.setdefault(
                    document_id,
                    {"document_id": document_id, "entities": [], "title": title, "abstract": ""},
                )
                current["title"] = title
                continue
            if "|a|" in stripped and stripped.find("|a|") > 0:
                document_id, abstract = stripped.split("|a|", 1)
                current = documents.setdefault(
                    document_id,
                    {"document_id": document_id, "entities": [], "title": "", "abstract": abstract},
                )
                current["abstract"] = abstract
                continue
            parts = stripped.split("\t")
            if current is None and parts:
                current = documents.setdefault(parts[0], {"document_id": parts[0], "entities": [], "title": "", "abstract": ""})
            if len(parts) >= 6 and parts[1].isdigit():
                current["entities"].append(
                    {"entity_type": parts[4], "concept_id": parts[5], "start": int(parts[1]), "end": int(parts[2]), "text": parts[3]}
                )
                continue
            if len(parts) >= 4 and parts[1] in BIORED_LABELS:
                rows.append(
                    {
                        "source_file": str(path),
                        "line_number": line_number,
                        "document_id": parts[0],
                        "endpoint_1": parts[2],
                        "endpoint_2": parts[3],
                        "label": parts[1],
                        "split": split,
                    }
                )
    usable_by_doc = {}
    conflict_by_doc = {}
    for document_id, document in documents.items():
        usable, conflicts = _concepts(document)
        usable_by_doc[document_id] = usable
        conflict_by_doc[document_id] = conflicts
    pending = []
    traced = []
    for row in rows:
        usable = usable_by_doc.get(row["document_id"], {})
        conflicts = conflict_by_doc.get(row["document_id"], {})
        left, right = row["endpoint_1"], row["endpoint_2"]
        additional = []
        if left in conflicts:
            additional.append("type_conflict")
        if right in conflicts:
            additional.append("type_conflict")
        left_ok = left in usable
        right_ok = right in usable
        if not left_ok or not right_ok:
            reason = "type_conflict" if (left in conflicts or right in conflicts) and left not in {"", "-"} and right not in {"", "-"} else "missing_identity"
            if left in {"", "-"} or right in {"", "-"}:
                reason = "missing_identity"
            traced.append(_done(row, "excluded", reason, additional, ""))
            continue
        if left == right:
            traced.append(_done(row, "excluded", "self_relation", additional, ""))
            continue
        if not legal_pair(schema, usable[left], usable[right]):
            traced.append(_done(row, "excluded", "out_of_scope", additional, ""))
            continue
        pending.append(row)
    seen = set()
    by_pair = defaultdict(list)
    for row in pending:
        key = (row["document_id"], _pair(row["endpoint_1"], row["endpoint_2"]), row["label"])
        if key in seen:
            candidate = biored_candidate_id(row["document_id"], row["endpoint_1"], row["endpoint_2"])
            traced.append(_done(row, "duplicate", "duplicate_relation", [], candidate))
            continue
        seen.add(key)
        by_pair[(row["document_id"], _pair(row["endpoint_1"], row["endpoint_2"]))].append(row)
    for (_document_id, pair), members in by_pair.items():
        labels = {row["label"] for row in members}
        candidate = biored_candidate_id(members[0]["document_id"], pair[0], pair[1])
        if len(labels) > 1:
            for row in members:
                traced.append(_done(row, "unresolved", "label_conflict", [], ""))
            continue
        for row in members:
            traced.append(_done(row, "included", "included", [], candidate))
    traced.sort(key=lambda item: (item["source_file"], item["line_number"]))
    return traced


def _done(row, destination, reason, additional, candidate_id) -> dict:
    return {
        "source_file": row["source_file"],
        "line_number": row["line_number"],
        "document_id": row["document_id"],
        "endpoint_1": row["endpoint_1"],
        "endpoint_2": row["endpoint_2"],
        "label": row["label"],
        "split": row.get("split", ""),
        "primary_destination": destination,
        "primary_reason": reason,
        "additional_reasons": "|".join(additional),
        "candidate_id": candidate_id,
    }


def summarize_lineage(rows: list[dict], candidates: list[dict] | None = None) -> dict:
    primary = defaultdict(int)
    by_split = defaultdict(lambda: defaultdict(int))
    triples = set()
    for row in rows:
        primary[row["primary_reason"]] += 1
        by_split[row.get("split", "")][row["primary_reason"]] += 1
        triples.add((row["document_id"], tuple(sorted((row["endpoint_1"], row["endpoint_2"]))), row["label"]))
    summary = {
        "raw_relation_rows": len(rows),
        "parsed_relation_rows": len(rows),
        "duplicate_relation_rows": primary["duplicate_relation"],
        "unique_document_pair_label": len(triples),
        "primary_reason_counts": dict(primary),
        "by_split": {split: dict(counts) for split, counts in by_split.items()},
        "unresolved_rows": primary["label_conflict"] + primary.get("unresolved", 0),
    }
    if candidates is not None:
        positive = [row for row in candidates if row.get("gold_labels")]
        summary["candidate_rows"] = len(candidates)
        summary["positive_candidates"] = len(positive)
        summary["negative_candidates"] = len(candidates) - len(positive)
        summary["multi_label_candidates"] = sum(1 for row in candidates if row.get("gold_status") == "multi")
    return summary


def trace_drugprot(training_dir: Path, development_dir: Path) -> tuple[list[dict], dict]:
    training = load_drugprot_directory(training_dir)
    development = load_drugprot_directory(development_dir)
    train_ids, dev_ids = split_drugprot_training(list(training["documents"]))
    split_of = {document_id: "train" for document_id in train_ids}
    split_of.update({document_id: "dev" for document_id in dev_ids})
    rows = []
    files = []
    for bundle, split_name in ((training, None), (development, "eval")):
        relation_path = next(path for path in bundle["files"] if "relation" in path.name.lower())
        files.append(relation_path)
        entities = bundle["entities"]
        with relation_path.open(encoding="utf-8") as handle:
            seen = set()
            parsed_rows = []
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4 or not parts[0].isdigit():
                    parsed_rows.append(
                        {
                            "source_file": str(relation_path),
                            "line_number": line_number,
                            "document_id": parts[0] if parts else "",
                            "endpoint_1": "",
                            "endpoint_2": "",
                            "label": parts[1] if len(parts) > 1 else "",
                            "split": split_name or "",
                            "primary_destination": "unresolved",
                            "primary_reason": "unparsed",
                            "additional_reasons": "",
                            "candidate_id": "",
                        }
                    )
                    continue
                document_id = parts[0]
                label = parts[1].strip()
                chemical = parts[2].split(":", 1)[-1].strip()
                gene = parts[3].split(":", 1)[-1].strip()
                assigned = split_name or split_of.get(document_id, "unassigned")
                mentions = {row["term"]: row for row in entities.get(document_id, [])}
                chemical_ok = chemical in mentions
                gene_ok = gene in mentions
                if not chemical_ok or not gene_ok:
                    parsed_rows.append(_done(
                        {"source_file": str(relation_path), "line_number": line_number, "document_id": document_id,
                         "endpoint_1": chemical, "endpoint_2": gene, "label": label, "split": assigned},
                        "excluded", "missing_identity", [], "",
                    ))
                    continue
                chemical_type = mentions[chemical]["entity_type"]
                gene_type = mentions[gene]["entity_type"]
                if chemical == gene:
                    parsed_rows.append(_done(
                        {"source_file": str(relation_path), "line_number": line_number, "document_id": document_id,
                         "endpoint_1": chemical, "endpoint_2": gene, "label": label, "split": assigned},
                        "excluded", "self_relation", [], "",
                    ))
                    continue
                if chemical_type != "CHEMICAL" or gene_type not in {"GENE", "GENE-Y", "GENE-N"}:
                    parsed_rows.append(_done(
                        {"source_file": str(relation_path), "line_number": line_number, "document_id": document_id,
                         "endpoint_1": chemical, "endpoint_2": gene, "label": label, "split": assigned},
                        "excluded", "out_of_scope", [], "",
                    ))
                    continue
                key = (document_id, chemical, gene, label)
                candidate = drugprot_candidate_id(document_id, chemical, gene)
                if key in seen:
                    parsed_rows.append(_done(
                        {"source_file": str(relation_path), "line_number": line_number, "document_id": document_id,
                         "endpoint_1": chemical, "endpoint_2": gene, "label": label, "split": assigned},
                        "duplicate", "duplicate_relation", [], candidate,
                    ))
                    continue
                seen.add(key)
                parsed_rows.append(_done(
                    {"source_file": str(relation_path), "line_number": line_number, "document_id": document_id,
                     "endpoint_1": chemical, "endpoint_2": gene, "label": label, "split": assigned},
                    "included", "included", [], candidate,
                ))
            rows.extend(parsed_rows)
    summary = summarize_lineage(rows)
    summary["train_documents"] = len(train_ids)
    summary["dev_documents"] = len(dev_ids)
    summary["eval_documents"] = len(development["documents"])
    summary["split_overlap_train_dev"] = sorted(set(train_ids) & set(dev_ids))
    summary["split_overlap_eval"] = sorted((set(train_ids) | set(dev_ids)) & set(development["documents"]))
    summary["source_files"] = [str(path) for path in files]
    return rows, summary
