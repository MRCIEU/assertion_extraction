"""Build the corrected classification, verbalisation, and experiment records.

The previous formal-protocol report counted MeSH tree nodes. This module
does not overwrite it. Primary additional types are restricted to relations
that the source resource states as classification.
"""

from __future__ import annotations

import csv
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.dataset as ds

from schema_study.classification_resources import (
    chebi_for_mesh,
    load_chebi_by_cas,
    load_gene_types,
    load_mesh_cas,
    load_mondo_by_mesh,
    load_supplemental_mappings,
)
from schema_study.entity_classification import concept_inventory, load_mesh_descriptors, tree_labels
from schema_study.entity_hierarchy import (
    GLOBAL_LEVEL_MIN_TRAINING_CONCEPTS,
    SELECTION_RULE,
    classify_concept,
    displayed_type,
    information_content,
    proposed_global_depth,
    realised_classification,
)
from schema_study.experiments import TRAIN_SEEDS
from schema_study.formal_protocol import (
    CURRENT_VERSIONS,
    _ancestor_ids,
    _cell_lines,
    _drugprot_identifiers,
    _pair_changes,
    _so,
    _splits,
    _taxonomy_names,
    _taxonomy_nodes,
)
from schema_study.paths import get_paths, load_settings
from schema_study.predicate_notation import historical_level, load_notation
from schema_study.records import publish_collection, repository_provenance
from schema_study.schemas import default_schema_path, load_schema
from schema_study.verbalisation import VERBALISATION_VERSION, verbalise

REPORT = "entity_classification"
E1_MERGE = {
    "GeneOrGeneProduct": "Molecular_Entity",
    "ChemicalEntity": "Molecular_Entity",
    "SequenceVariant": "Molecular_Entity",
}


def prepare() -> int:
    paths = get_paths()
    destination = paths.assert_output(paths.study / "reports" / REPORT)
    destination.mkdir(parents=True, exist_ok=True)
    print("Loading classification resources.", flush=True)
    resources, resource_note = _resources(paths)
    print("Classifying concepts.", flush=True)
    concepts, summary = _classify(paths, resources)
    summary["resources"] = resource_note
    summary["selection_rule"] = SELECTION_RULE
    summary["verbalisation_version"] = VERBALISATION_VERSION
    summary["drugprot"] = _drugprot_identifiers(paths)
    summary["context_budget"] = _context_budget(paths, concepts)
    summary["experiment_matrix"] = _matrix(paths, summary["proposed_entity_levels"])
    summary["scifact"] = _scifact(paths)
    summary["civic"] = _civic(paths, destination)
    summary["human_sample"] = _human_sample(paths, destination)
    summary["predictions"] = _verbalise_predictions(paths, destination)
    summary["provenance"] = repository_provenance(paths.code_root)
    summary["previous_protocol"] = (
        "reports/formal_protocol remains the MeSH-tree-node result. "
        "This directory is the corrected classification."
    )

    def write_concepts(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in concepts:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def write_summary(temporary: Path) -> None:
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    publish_collection(
        destination,
        {"concepts.jsonl": write_concepts, "summary.json": write_summary},
        {"status": "complete", "report": REPORT},
    )
    print(json.dumps({
        "proposed_entity_levels": summary["proposed_entity_levels"],
        "biored_training_new_distinctions": summary["biored_training_new_distinctions"],
        "pair_changes": summary["biored_training_pair_changes"],
    }, indent=2))
    return 0


def _resources(paths) -> tuple[dict, dict]:
    mesh_path = paths.raw / "mesh" / "nlm_2026" / "desc2026.gz"
    descriptors = load_mesh_descriptors(mesh_path)
    labels = tree_labels(descriptors)
    wanted = _wanted_identifiers(paths)
    cas = load_mesh_cas(mesh_path)
    chebi_path = paths.raw / "chebi" / "chebi.obo.gz"
    chebi_by_cas = load_chebi_by_cas(chebi_path, set(cas.values())) if chebi_path.is_file() else {}
    mondo_path = paths.raw / "mondo" / "mondo.obo"
    mondo = load_mondo_by_mesh(mondo_path, wanted["mesh"]) if mondo_path.is_file() else {}
    supp_path = paths.raw / "mesh" / "nlm_2026" / "supp2026.gz"
    supplemental = load_supplemental_mappings(supp_path, wanted["supplemental"]) if supp_path.is_file() else {}
    gene_types = load_gene_types(paths.raw / "gene")
    so_version, so_ids = _so(paths.raw / "sequence_ontology" / "so.obo")
    parents, ranks = _taxonomy_nodes(paths.raw / "taxonomy" / "nodes.dmp")
    names = _taxonomy_names(paths.raw / "taxonomy" / "names.dmp", _ancestor_ids(wanted["taxa"], parents))
    cell_lines = _cell_lines(paths.raw / "cellosaurus" / "cellosaurus.obo", wanted["cells"])
    resources = {
        "descriptors": descriptors,
        "labels": labels,
        "so_ids": so_ids,
        "parents": parents,
        "ranks": ranks,
        "names": names,
        "cell_lines": cell_lines,
        "gene_types": gene_types,
        "chebi_by_mesh": chebi_for_mesh(cas, chebi_by_cas),
        "mondo_by_mesh": mondo,
        "supplemental": supplemental,
        "variant_classes": {},
    }
    note = {
        "mesh_descriptors": len(descriptors),
        "mesh_cas_numbers": len(cas),
        "chebi_cas_hits": len(chebi_by_cas),
        "chebi_mesh_joins": len(resources["chebi_by_mesh"]),
        "chebi_unique_joins": sum(1 for row in resources["chebi_by_mesh"].values() if row["n_matches"] == 1),
        "mondo_mesh_joins": len(mondo),
        "mondo_unique_joins": sum(1 for row in mondo.values() if row["n_matches"] == 1),
        "supplemental_records": len(supplemental),
        "gene_info_rows": len(gene_types),
        "gene_info_files": sorted(path.name for path in (paths.raw / "gene").glob("*.gene_info.gz")),
        "sequence_ontology": so_version,
        "cell_lines": len(cell_lines),
        "umls": {"available": False, "used": False},
        "dbsnp": {
            "used": False,
            "reason": "An rs identifier is not a variant class. No official class table was joined, so rs concepts stay unresolved.",
        },
        "chebi_role_relation": "has_role was not read",
        "mesh_functional_tops_excluded": ["D06", "D26", "D27"],
    }
    return resources, note


def _wanted_identifiers(paths) -> dict:
    wanted = {"mesh": set(), "supplemental": set(), "cells": set(), "taxa": set()}
    for corpus in ("biored", "bc8"):
        for line in (paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "entities.jsonl").open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            concept_id = str(row.get("concept_id") or "")
            entity_type = str(row.get("entity_type") or "")
            if concept_id[:1] == "D" and concept_id[1:2].isdigit():
                wanted["mesh"].add(concept_id)
            if concept_id[:1] == "C" and len(concept_id) > 1 and concept_id[1].isdigit():
                wanted["supplemental"].add(concept_id)
            if entity_type == "CellLine" and concept_id.startswith("CVCL"):
                wanted["cells"].add(concept_id)
            if entity_type in {"OrganismTaxon", "Organism"} and concept_id.isdigit():
                wanted["taxa"].add(concept_id)
    return wanted


def _classify(paths, resources: dict) -> tuple[list[dict], dict]:
    records = []
    by_corpus = {}
    biored_new = Counter()
    biored_chains = []
    biored_by_id = {}
    for corpus in ("biored", "bc8"):
        document_splits = _splits(paths, corpus)
        rows = []
        concept_splits = defaultdict(set)
        for line in (paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "entities.jsonl").open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            concept_splits[str(row.get("concept_id") or "")].add(document_splits.get(str(row["document_id"])))
            rows.append(row)
        inventory = concept_inventory(rows)
        status_counts = Counter()
        by_type = defaultdict(Counter)
        by_split = defaultdict(Counter)
        for concept in inventory["concepts"]:
            classified = classify_concept(concept, resources)
            chain = classified.get("chain") or []
            native = concept.get("entity_type")
            record = {
                "corpus": corpus,
                "concept_id": concept["concept_id"],
                "entity_type": native,
                "splits": sorted(item for item in concept_splits.get(concept["concept_id"], set()) if item),
                "status": classified.get("status"),
                "reason": classified.get("reason"),
                "eligible": bool(classified.get("eligible")),
                "chain": [{"node": node["node"], "label": node["label"], "relation": node["relation"]} for node in chain],
                "mesh_chain": [
                    {"node": node["node"], "label": node["label"]}
                    for node in (classified.get("mesh_chain") or [])
                ],
                "database_type": classified.get("database_type"),
                "displayed": {},
            }
            records.append(record)
            status_counts[record["status"]] += 1
            by_type[native or "conflict"][record["status"]] += 1
            for split in record["splits"]:
                by_split[split][record["status"]] += 1
            if corpus == "biored" and "train" in record["splits"] and chain:
                biored_chains.append([node["node"] for node in chain])
                biored_by_id[concept["concept_id"]] = chain
                for level in range(3, 12):
                    if realised_classification(chain, level)["new_distinction"]:
                        biored_new[level] += 1
        by_corpus[corpus] = {
            "n_concepts": len(inventory["concepts"]),
            "status_counts": dict(status_counts),
            "by_native_type": {key: dict(value) for key, value in sorted(by_type.items())},
            "by_split": {key: dict(value) for key, value in sorted(by_split.items())},
            "examples": _examples(records, corpus),
        }
    depth = proposed_global_depth(dict(biored_new))
    proposed = ["E0", "E1", "E2"] + [f"E{level}" for level in range(3, depth + 1)]
    for record in records:
        chain = record["chain"]
        record["displayed"] = {
            level_name: displayed_type(record["entity_type"], chain, int(level_name[1:]), E1_MERGE)
            for level_name in proposed
        }
    ic = information_content(biored_chains)
    top_ic = sorted(ic.items(), key=lambda item: -item[1])[:15]
    return records, {
        "global_level_minimum_training_concepts": GLOBAL_LEVEL_MIN_TRAINING_CONCEPTS,
        "biored_training_new_distinctions": {str(key): value for key, value in sorted(biored_new.items())},
        "alternative_thresholds": {
            str(threshold): proposed_global_depth(dict(biored_new), threshold)
            for threshold in (25, 50, 100, 250)
        },
        "proposed_global_depth": depth,
        "proposed_entity_levels": proposed,
        "recommendation": _recommendation(depth, biored_new),
        "biored_training_pair_changes": _pair_changes(paths, biored_by_id),
        "information_content_top": [{"node": node, "nats": value} for node, value in top_ic],
        "corpora": by_corpus,
    }


def _examples(records: list[dict], corpus: str) -> list[dict]:
    chosen = []
    seen = set()
    for row in records:
        if row["corpus"] != corpus or not row["chain"]:
            continue
        key = (row["entity_type"], row["chain"][0]["relation"], row["status"])
        if key in seen:
            continue
        seen.add(key)
        chosen.append({
            "concept_id": row["concept_id"],
            "entity_type": row["entity_type"],
            "status": row["status"],
            "path": [node["label"] for node in row["chain"]],
            "relation": row["chain"][0]["relation"],
        })
        if len(chosen) >= 12:
            break
    return chosen


def _recommendation(depth: int, counts: Counter) -> str:
    if depth <= 2:
        return (
            "No additional level changes 100 BioRED training concepts under the corrected relations. "
            "E0–E2 remain the global conditions. Concept-specific classes are retained in the records."
        )
    return (
        f"E0–E{depth} are the proposed global conditions. "
        f"E3 changes {counts.get(3, 0)} BioRED training concepts. "
        "The threshold is coverage, not extraction performance. "
        "Concepts without a further class keep their terminal label."
    )


def _context_budget(paths, concepts: list[dict]) -> dict:
    settings = load_settings(paths.code_root)
    tokenizer_dir = Path(str(settings.get("biomedbert_tokenizer_dir") or ""))
    labels = sorted({
        node["label"]
        for row in concepts
        for node in row["chain"]
        if node.get("label")
    })
    measured = _token_lengths(tokenizer_dir, labels[:200] if len(labels) > 200 else labels)
    near_limit = _near_budget(paths)
    return {
        "budget": 512,
        "new_special_tokens": False,
        "historical_e0_e2_inputs": "unchanged and reusable where the prompt, label strings, and model revision match",
        "additional_type_encoding": "ordinary tokenizer text, not a new vocabulary entry",
        "n_distinct_additional_labels": len(labels),
        "token_lengths": measured,
        "biored_inputs_within_16_tokens_of_512": near_limit,
        "implication": (
            "A condition that inserts an additional class string is a different model input. "
            "Its historical prediction is not reusable. E0–E2 files are unchanged."
        ),
    }


def _token_lengths(tokenizer_dir: Path, labels: list[str]) -> dict:
    if not tokenizer_dir.is_dir():
        return {"status": "tokenizer_directory_absent", "path": str(tokenizer_dir)}
    try:
        from transformers import BertTokenizerFast
    except ImportError:
        return {"status": "transformers_absent"}
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir, local_files_only=True)
    lengths = [len(tokenizer.encode(label, add_special_tokens=False)) for label in labels]
    if not lengths:
        return {"status": "no_additional_labels", "n": 0}
    return {
        "status": "measured",
        "n": len(lengths),
        "max_tokens": max(lengths),
        "median_tokens": sorted(lengths)[len(lengths) // 2],
    }


def _near_budget(paths) -> dict:
    path = paths.inputs / "biored" / "target_anchored_v1__biomedbert_e1354b7a_proposed" / "contexts.jsonl"
    if not path.is_file():
        return {"status": "absent"}
    counts = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            length = json.loads(line).get("max_encoded_length") or 0
            counts["n"] += 1
            if length > 496:
                counts["within_16"] += 1
            if length > 512:
                counts["over"] += 1
    return dict(counts)


def _matrix(paths, entity_levels: list[str]) -> dict:
    accepted = _accepted(paths)
    notation = load_notation(paths.code_root / "configs" / "schemas" / "predicate_notation.yaml")
    predicates = ["P0", "P1", "P2", "P3"]
    models = ("biomedbert", "qwen8b", "qwen32b")
    counts = Counter()
    for dataset in ("biored", "bc8", "drugprot"):
        for entity in entity_levels:
            for predicate in predicates:
                for model in models:
                    if model == "biomedbert" and dataset == "bc8":
                        continue
                    seeds = [str(seed) for seed in TRAIN_SEEDS] if model == "biomedbert" else [""]
                    for seed in seeds:
                        decision = _reuse(dataset, entity, predicate, model, seed, accepted, notation)
                        counts[decision] += 1
    return {
        "entity_levels": entity_levels,
        "predicate_levels": predicates,
        "decisions": dict(counts),
        "launched": False,
        "gpu_time": "not estimated; this preparation did not remeasure runtimes",
    }


def _accepted(paths) -> set[tuple]:
    path = paths.study / "reports" / "representation_granularity" / "granularity_measurements.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    found = set()
    for row in payload.get("predictions", {}).get("accepted", []):
        found.add((row["dataset"], row["entity_level"], row["predicate_level"], row["model"], str(row.get("seed") or "")))
    return found


def _reuse(dataset, entity, predicate, model, seed, accepted, notation) -> str:
    historical = historical_level(notation, predicate)
    key = (dataset, entity, historical, model, seed)
    if entity in {"E0", "E1", "E2"} and key in accepted:
        return "previous_prediction_reusable"
    native_key = (dataset, entity, "P4", model, seed)
    if entity in {"E0", "E1", "E2"} and predicate != "P3" and native_key in accepted:
        return "deterministic_abstraction"
    if model == "biomedbert":
        return "new_training"
    return "new_inference"


def _scifact(paths) -> dict:
    archive = paths.raw / "scifact" / "data.tar.gz"
    if not archive.is_file():
        return {"status": "absent"}
    counts = Counter()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".jsonl") or "claims_" not in member.name:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            split = Path(member.name).name
            for line in handle:
                row = json.loads(line)
                counts[f"{split}:n"] += 1
                for evidence in (row.get("evidence") or {}).values():
                    for item in evidence:
                        counts[f"{split}:{item.get('label')}"] += 1
    return {"archive": str(archive), "status": "acquired_not_scored", "counts": dict(counts)}


def _civic(paths, destination: Path) -> dict:
    path = paths.raw / "civic" / "nightly-ClinicalEvidenceSummaries.tsv"
    if not path.is_file():
        return {"status": "absent"}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    disease_key = "disease" if rows and "disease" in rows[0] else ""
    doid_key = "doid" if rows and "doid" in rows[0] else ""
    if not disease_key or not doid_key:
        return {"status": "unexpected_columns", "columns": sorted(rows[0]) if rows else []}
    lung_doids = sorted({
        row[doid_key].strip()
        for row in rows
        if row.get(doid_key, "").strip() and "lung" in row.get(disease_key, "").lower()
    })
    doid_set = set(lung_doids)
    selected = [row for row in rows if row.get(doid_key, "").strip() in doid_set]
    unresolved = [
        row for row in rows
        if "lung" in row.get(disease_key, "").lower() and not row.get(doid_key, "").strip()
    ]
    out = destination / "civic_lung_evidence.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps({
                "molecular_profile": row.get("molecular_profile"),
                "disease": row.get(disease_key),
                "doid": row.get(doid_key),
                "therapies": row.get("therapies"),
                "evidence_type": row.get("evidence_type"),
                "evidence_direction": row.get("evidence_direction"),
                "evidence_level": row.get("evidence_level"),
                "significance": row.get("significance"),
                "citation_id": row.get("citation_id"),
                "evidence_status": row.get("evidence_status"),
                "curated_statement_used_as_model_input": False,
            }, ensure_ascii=False) + "\n")
    abstracts = _pubmed_abstracts(
        [row.get("citation_id") for row in selected],
        destination / "civic_pubmed_abstracts.jsonl",
    )
    return {
        "n_snapshot_rows": len(rows),
        "lung_doids": lung_doids,
        "n_rows_with_those_doids": len(selected),
        "n_lung_name_without_doid": len(unresolved),
        "absence_is_not_a_negative_label": True,
        "historical_knowledge_base": False,
        "output": str(out),
        "pubmed": abstracts,
    }


def _pubmed_abstracts(citation_ids: list, destination: Path) -> dict:
    """Titles and abstracts for the linked citations. A failure is recorded, not invented."""
    import urllib.request
    import xml.etree.ElementTree as ET

    identifiers = sorted({str(value).strip() for value in citation_ids if str(value or "").strip().isdigit()})
    found = {}
    errors = []
    for start in range(0, len(identifiers), 100):
        batch = identifiers[start:start + 100]
        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=pubmed&retmode=xml&id={','.join(batch)}"
        )
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
        except Exception as exc:
            errors.append(str(exc))
            break
        root = ET.fromstring(payload)
        for article in root.iter("PubmedArticle"):
            pmid = ""
            for node in article.iter("PMID"):
                pmid = (node.text or "").strip()
                break
            title = " ".join((node.text or "") for node in article.iter("ArticleTitle"))
            abstract = " ".join((node.text or "") for node in article.iter("AbstractText"))
            if pmid:
                found[pmid] = {"title": title, "abstract": abstract}
    with destination.open("w", encoding="utf-8") as handle:
        for pmid, row in sorted(found.items()):
            handle.write(json.dumps({"pmid": pmid, **row}, ensure_ascii=False) + "\n")
    return {
        "n_requested": len(identifiers),
        "n_retrieved": len(found),
        "errors": errors,
        "output": str(destination),
    }


def _human_sample(paths, destination: Path) -> dict:
    excluded = _sample_documents(paths)
    frame = []
    positive_docs = set()
    candidate_path = paths.prepared / "biored" / CURRENT_VERSIONS["biored"] / "candidates.parquet"
    dataset = ds.dataset(candidate_path, format="parquet")
    for batch in dataset.to_batches(columns=["document_id", "split", "gold_status"], filter=ds.field("split") == "test"):
        for row in batch.to_pylist():
            if row["gold_status"] == "unique":
                positive_docs.add(str(row["document_id"]))
    documents = paths.prepared / "biored" / CURRENT_VERSIONS["biored"] / "documents.jsonl"
    for line in documents.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != "test":
            continue
        document_id = str(row["document_id"])
        if document_id in excluded:
            continue
        frame.append({"document_id": document_id, "stratum": "positive" if document_id in positive_docs else "unannotated"})
    import random
    generator = random.Random(20261125)
    drawn = []
    for stratum in ("positive", "unannotated"):
        members = sorted(row["document_id"] for row in frame if row["stratum"] == stratum)
        generator.shuffle(members)
        take = max(1, round(0.2 * len(members))) if members else 0
        for document_id in members[:take]:
            drawn.append({
                "document_id": document_id,
                "stratum": stratum,
                "inclusion_probability": 0.2,
                "analysis_weight": 5.0,
                "judgement": None,
            })
    out = destination / "human_probability_sample.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in drawn:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "population": "biored_test_excluding_development_and_diagnostic_documents",
        "seed": 20261125,
        "inclusion_probability": 0.2,
        "n_frame": len(frame),
        "n_drawn": len(drawn),
        "answers_populated": False,
        "output": str(out),
    }


def _sample_documents(paths) -> set[str]:
    path = paths.study / "reports" / "source_references" / "summary.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    found = set()
    for block in (payload.get("samples") or {}).values():
        for corpus in block.values():
            for candidate_id in corpus.get("candidate_ids") or []:
                parts = str(candidate_id).split(":")
                if len(parts) > 1:
                    found.add(parts[1])
    return found


def _verbalise_predictions(paths, destination: Path) -> dict:
    schemas = {
        "biored": load_schema(default_schema_path(paths.code_root, "biored")),
        "drugprot": load_schema(default_schema_path(paths.code_root, "drugprot")),
    }
    schemas["bc8"] = schemas["biored"]
    root = paths.study / "runs" / "formal" / "formal_20261115"
    files = sorted(root.rglob("predictions.jsonl")) if root.is_dir() else []
    out = destination / "verbalised_predictions.jsonl"
    counts = Counter()
    with out.open("w", encoding="utf-8") as handle:
        for path in files:
            parts = path.parts
            try:
                dataset = parts[parts.index("formal_20261115") + 2]
                entity = parts[parts.index("formal_20261115") + 3]
                predicate = parts[parts.index("formal_20261115") + 4]
            except (ValueError, IndexError):
                continue
            schema = schemas.get(dataset)
            if schema is None or predicate not in schema.levels:
                counts["skipped_files"] += 1
                continue
            by_id = schema.by_id(predicate)
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                row = json.loads(line)
                counts["records"] += 1
                labels = row.get("predicted_labels") or []
                status = row.get("output_status") or "missing"
                if status != "valid":
                    rendered = verbalise(None, "", "", prediction_status="invalid" if status == "invalid" else "missing")
                    counts[rendered["status"]] += 1
                    continue
                if not labels or labels == [schema.no_relation_id]:
                    rendered = verbalise("NO_RELATION", row.get("argument_1_id") or row["candidate_id"], row.get("argument_2_id") or "")
                    counts[rendered["status"]] += 1
                    continue
                for label_id in labels:
                    label = by_id.get(label_id)
                    if label is None:
                        counts["unknown_label"] += 1
                        continue
                    members = [schema.fine_names.get(member, member) for member in label.members]
                    rendered = verbalise(
                        label.display_name,
                        row.get("argument_1_id") or row["candidate_id"],
                        row.get("argument_2_id") or "",
                        members=members,
                        corpus=schema.corpus,
                    )
                    counts["verbalised"] += 1
                    handle.write(json.dumps({
                            "candidate_id": row["candidate_id"],
                            "dataset": dataset,
                            "entity_level": entity,
                            "predicate_level": predicate,
                            "label": label.display_name,
                            "text": rendered["text"],
                            "version": rendered["version"],
                        }, ensure_ascii=False) + "\n")
    counts["files"] = len(files)
    return dict(counts)
