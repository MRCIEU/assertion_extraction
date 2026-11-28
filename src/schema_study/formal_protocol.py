"""Assemble the pre-training protocol from prepared data and public resources.

The command does not train, generate, or score with a neural model.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.dataset as ds

from schema_study.entity_classification import concept_inventory, load_mesh_descriptors, tree_labels
from schema_study.entity_hierarchy import (
    GLOBAL_LEVEL_MIN_TRAINING_CONCEPTS,
    classify_concept,
    information_content,
    proposed_global_depth,
    realised_classification,
)
from schema_study.experiments import TRAIN_SEEDS, training_hyperparameters, generation_settings
from schema_study.paths import get_paths
from schema_study.predicate_alternatives import load_alternatives, partition_information
from schema_study.predicate_notation import historical_level, load_notation
from schema_study.records import digest_file, publish_collection, repository_provenance, write_json
from schema_study.schemas import default_schema_path, load_schema
from schema_study.semantic_measures import CERTAINTY_CANDIDATES, MEASUREMENT_VERSION, VERIFICATION_CANDIDATES
from schema_study.sentences import BIOMEDICAL_SEGMENTER_CANDIDATE, SENTENCE_SEGMENTATION_VERSION
from schema_study.source_identification import LOCALISATION_MODEL
from schema_study.source_integrity import collection_complete, inspect_jsonl, population_difference
from schema_study.source_references import CURRENT_VERSIONS, TARGET_SPLITS
from schema_study.verbalisation import VERBALISATION_VERSION

RESEARCH_PREDICATES = ("P0", "P1", "P2", "P3")
NATIVE_ENTITY_LEVELS = ("E0", "E1", "E2")
MODELS = (
    {"model": "biomedbert", "kind": "fine_tuning", "seeds": [str(seed) for seed in TRAIN_SEEDS], "trains": True},
    {"model": "qwen8b", "kind": "prompted_generation", "seeds": [""], "trains": False},
    {"model": "qwen32b", "kind": "prompted_generation", "seeds": [""], "trains": False},
)


def prepare() -> int:
    paths = get_paths()
    destination = paths.assert_output(paths.study / "reports" / "formal_protocol")
    print("Checking source-reference structure.", flush=True)
    integrity = _source_integrity(paths)
    print("Classifying entities.", flush=True)
    entities = _entities(paths)
    print("Reading CIViC evidence summaries.", flush=True)
    civic = _civic(paths.raw / "civic" / "nightly-ClinicalEvidenceSummaries.tsv")
    notation = load_notation(paths.code_root / "configs" / "schemas" / "predicate_notation.yaml")
    matrix = _matrix(paths, notation, entities["proposed_entity_levels"])
    alternatives = _alternatives(paths)
    payload = {
        "provenance": repository_provenance(paths.code_root),
        "verbalisation_version": VERBALISATION_VERSION,
        "measurement_version": MEASUREMENT_VERSION,
        "sentence_segmentation_version": SENTENCE_SEGMENTATION_VERSION,
        "biomedical_segmenter_candidate": BIOMEDICAL_SEGMENTER_CANDIDATE,
        "source_localisation": LOCALISATION_MODEL,
        "source_integrity": integrity,
        "entities": {key: value for key, value in entities.items() if key != "review_rows"},
        "predicate_notation": notation,
        "alternative_partitions": alternatives,
        "experiment_matrix": matrix,
        "training_hyperparameters": training_hyperparameters(),
        "generation_settings": generation_settings(),
        "certainty_candidates": list(CERTAINTY_CANDIDATES),
        "verification_candidates": list(VERIFICATION_CANDIDATES),
        "civic": civic,
        "gpu_executed": False,
    }
    files = {
        "protocol.json": lambda temporary: write_json(temporary, payload),
        "human_review_examples.jsonl": lambda temporary: _write_jsonl(temporary, entities["review_rows"]),
    }
    manifest = {
        "status": "complete" if integrity["all_split_populations_complete"] else "source_integrity_failed",
        "proposed_entity_levels": entities["proposed_entity_levels"],
        "proposed_research_predicates": list(RESEARCH_PREDICATES),
        "gpu_executed": False,
        "repository_revision": payload["provenance"]["revision"],
        "working_tree_dirty": payload["provenance"]["working_tree_dirty"],
    }
    publish_collection(destination, files, manifest)
    print(f"Wrote {destination}", flush=True)
    return 0 if integrity["all_split_populations_complete"] else 1


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _source_integrity(paths) -> dict:
    report = {}
    complete = True
    root = paths.study / "reports" / "source_references"
    for corpus, splits in TARGET_SPLITS.items():
        split_ids, scoring_ids, excluded = _candidate_ids(paths, corpus, set(splits))
        files = {}
        for kind, identifier, required in (
            ("sentences", "sentence_id", {"sentence_id": str, "document_id": str, "split": str, "char_start": int, "char_end": int, "text": str}),
            ("alignments", "candidate_id", {"candidate_id": str, "corpus": str, "split": str, "document_id": str, "argument_1": dict, "argument_2": dict}),
            ("passages", "candidate_id", {"candidate_id": str, "corpus": str, "split": str, "document_id": str, "passages": dict}),
        ):
            path = root / kind / f"{corpus}.jsonl"
            expected_corpus = None if kind == "sentences" else corpus
            extra = _passage_problems if kind == "passages" else None
            inspected = inspect_jsonl(path, identifier, required, expected_corpus, set(splits), extra)
            if kind == "sentences":
                population = {"complete": True, "n_expected": inspected["n_records"], "n_observed": inspected["n_records"], "n_missing": 0, "n_unexpected": 0}
                checked = collection_complete(inspected, population)
                checked["population_definition"] = "sentence records are checked for structure, not against candidate identifiers"
            else:
                population = population_difference(inspected["identifiers"], split_ids)
                checked = collection_complete(inspected, population)
                checked["scoring_eligible_all_present"] = scoring_ids <= inspected["identifiers"]
                checked["n_excluded_candidates_present"] = excluded
                checked["scoring_population_only"] = excluded == 0
                checked["complete"] = checked["complete"] and checked["scoring_eligible_all_present"]
            inspected.pop("identifiers", None)
            inspected.pop("document_ids", None)
            files[kind] = {"inspection": {key: value for key, value in inspected.items() if key != "identifiers"}, "completion": checked}
            complete = complete and checked["complete"]
        report[corpus] = files
    report["all_split_populations_complete"] = complete
    return report


def _passage_problems(row: dict) -> list[str]:
    passages = row.get("passages")
    if not isinstance(passages, dict):
        return ["passages"]
    problems = []
    for name in ("document", "both_targets", "shortest_interval", "neighbouring_sentences"):
        item = passages.get(name)
        if not isinstance(item, dict) or not isinstance(item.get("alternatives"), list):
            problems.append(name)
            continue
        if name == "document" and not item["alternatives"]:
            problems.append("document_empty")
        for alternative in item["alternatives"]:
            spans = alternative.get("spans") if isinstance(alternative, dict) else None
            if not isinstance(spans, list) or not spans:
                problems.append("spans")
                break
            if any(not (isinstance(span, list) and len(span) == 2 and all(isinstance(bound, int) for bound in span)) for span in spans):
                problems.append("span_bounds")
                break
            if alternative.get("model_input") not in {"all", "part", "none"}:
                problems.append("model_input")
                break
    return problems


def _candidate_ids(paths, corpus: str, splits: set[str]) -> tuple[set[str], set[str], int]:
    dataset = ds.dataset(paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "candidates.parquet", format="parquet")
    split_ids = set()
    scoring = set()
    excluded = 0
    for batch in dataset.to_batches(columns=["candidate_id", "included_in_scoring"], filter=ds.field("split").isin(sorted(splits))):
        for row in batch.to_pylist():
            split_ids.add(row["candidate_id"])
            if row["included_in_scoring"]:
                scoring.add(row["candidate_id"])
            else:
                excluded += 1
    return split_ids, scoring, excluded


def _entities(paths) -> dict:
    mesh_path = paths.raw / "mesh" / "nlm_2026" / "desc2026.gz"
    descriptors = load_mesh_descriptors(mesh_path)
    labels = tree_labels(descriptors)
    so_version, so_ids = _so(paths.raw / "sequence_ontology" / "so.obo")
    parents, ranks = _taxonomy_nodes(paths.raw / "taxonomy" / "nodes.dmp")
    inventories = {}
    wanted_cells = set()
    wanted_taxa = set()
    for corpus in ("biored", "bc8"):
        document_splits = _splits(paths, corpus)
        rows = []
        concept_splits = defaultdict(set)
        for line in (paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "entities.jsonl").open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            concept_id = str(row.get("concept_id") or "")
            concept_splits[concept_id].add(document_splits.get(str(row["document_id"])))
            rows.append(row)
            if str(row.get("entity_type")) == "CellLine" and concept_id.startswith("CVCL"):
                wanted_cells.add(concept_id)
            if str(row.get("entity_type")) in {"OrganismTaxon", "Organism"} and concept_id.isdigit():
                wanted_taxa.add(concept_id)
        inventories[corpus] = {"concept_splits": concept_splits, "inventory": concept_inventory(rows)}
    names = _taxonomy_names(paths.raw / "taxonomy" / "names.dmp", _ancestor_ids(wanted_taxa, parents))
    cell_lines = _cell_lines(paths.raw / "cellosaurus" / "cellosaurus.obo", wanted_cells)
    resources = {
        "descriptors": descriptors,
        "labels": labels,
        "so_ids": so_ids,
        "parents": parents,
        "ranks": ranks,
        "names": names,
        "cell_lines": cell_lines,
    }
    summaries = {}
    review = []
    biored_train_chains = []
    biored_train_by_id = {}
    biored_new = Counter()
    for corpus, block in inventories.items():
        status_counts = Counter()
        depth_counts = Counter()
        by_type = defaultdict(Counter)
        examples = []
        for concept in block["inventory"]["concepts"]:
            splits_for_concept = block["concept_splits"].get(concept["concept_id"], set())
            classified = classify_concept(concept, resources)
            status_counts[classified["status"]] += 1
            entity_type = concept["entity_type"] or "conflict"
            by_type[entity_type][classified["status"]] += 1
            depth = len(classified["chain"]) if classified.get("eligible") else 0
            depth_counts[depth] += 1
            if corpus == "biored" and "train" in splits_for_concept and classified.get("eligible"):
                biored_train_chains.append([node["node"] for node in classified["chain"]])
                biored_train_by_id[concept["concept_id"]] = classified["chain"]
                for level in range(3, 12):
                    if realised_classification(classified["chain"], level)["new_distinction"]:
                        biored_new[level] += 1
            if len(examples) < 40 and classified["status"] in {
                "ambiguous_paths",
                "excluded_relation_leakage",
                "incomplete_composite",
                "type_conflict",
                "unresolved_variant_code",
                "classified",
                "no_additional_distinction",
            }:
                examples.append(
                    {
                        "corpus": corpus,
                        "concept_id": concept["concept_id"],
                        "entity_type": concept["entity_type"],
                        "status": classified["status"],
                        "chain": [node["label"] for node in classified["chain"]],
                        "human_judgement": None,
                    }
                )
        review.extend(row for row in examples if row["status"] != "no_additional_distinction")
        summaries[corpus] = {
            "inventory": {key: value for key, value in block["inventory"].items() if key != "concepts"},
            "status_counts": dict(status_counts),
            "additional_depth_counts": {str(key): value for key, value in sorted(depth_counts.items())},
            "by_native_type": {key: dict(value) for key, value in sorted(by_type.items())},
            "examples": examples[:12],
        }
    proposed = ["E0", "E1", "E2"]
    depth = proposed_global_depth(dict(biored_new))
    for level in range(3, depth + 1):
        proposed.append(f"E{level}")
    pair_changes = _pair_changes(paths, biored_train_by_id)
    drugprot = _drugprot_identifiers(paths)
    return {
        "resources": _resource_record(paths, so_version, len(so_ids), len(cell_lines)),
        "selection_rule": "mesh_native_top_then_lexicographic_v1",
        "global_level_minimum_training_concepts": GLOBAL_LEVEL_MIN_TRAINING_CONCEPTS,
        "biored_training_new_distinctions": {str(key): value for key, value in sorted(biored_new.items())},
        "proposed_entity_levels": proposed,
        "proposed_global_depth": depth,
        "information_content_nats": _top_information(biored_train_chains),
        "biored_training_pair_changes": pair_changes,
        "drugprot": drugprot,
        "corpora": summaries,
        "review_rows": review[:80],
        "umls": {"available": False, "used": False},
    }


def _splits(paths, corpus: str) -> dict[str, str]:
    splits = {}
    for line in (paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "documents.jsonl").open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            splits[str(row["document_id"])] = row["split"]
    return splits


def _so(path: Path) -> tuple[str | None, set[str]]:
    version = None
    identifiers = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("data-version:"):
                version = line.split(":", 1)[1].strip()
            elif line.startswith("id: SO:"):
                identifiers.add(line.split()[1])
    return version, identifiers


def _taxonomy_nodes(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    parents = {}
    ranks = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split("\t|\t")
            if len(parts) < 3:
                continue
            parents[parts[0]] = parts[1]
            ranks[parts[0]] = parts[2]
    return parents, ranks


def _ancestor_ids(taxids: set[str], parents: dict[str, str]) -> set[str]:
    wanted = set(taxids)
    for taxid in list(taxids):
        current = taxid
        seen = set()
        while current and current not in seen:
            seen.add(current)
            wanted.add(current)
            parent = parents.get(current)
            if not parent or parent == current:
                break
            current = parent
    return wanted


def _taxonomy_names(path: Path, wanted: set[str]) -> dict[str, str]:
    names = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split("\t|\t")
            if len(parts) < 4 or parts[0] not in wanted:
                continue
            if parts[3].startswith("scientific name"):
                names[parts[0]] = parts[1]
    return names


def _cell_lines(path: Path, wanted: set[str]) -> dict[str, dict]:
    found = {}
    current = None
    keep = False
    subsets: list[str] = []
    xrefs: list[str] = []

    def save() -> None:
        if keep and current:
            found[current] = {"subsets": list(subsets), "xrefs": list(xrefs)}

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("[Term]"):
                save()
                current = None
                keep = False
                subsets = []
                xrefs = []
            elif line.startswith("id: "):
                current = line[4:].strip()
                keep = current in wanted
            elif keep and line.startswith("subset: "):
                subsets.append(line[8:].strip())
            elif keep and line.startswith("xref: "):
                xrefs.append(line.split(" ", 1)[1].strip())
        save()
    return found


def _pair_changes(paths, chains: dict[str, list]) -> dict:
    dataset = ds.dataset(paths.prepared / "biored" / CURRENT_VERSIONS["biored"] / "candidates.parquet", format="parquet")
    counts = Counter()
    for batch in dataset.to_batches(
        columns=["argument_1_id", "argument_2_id", "included_in_scoring"],
        filter=ds.field("split") == "train",
    ):
        for row in batch.to_pylist():
            if not row["included_in_scoring"]:
                continue
            changed = [len(chains.get(str(row["argument_1_id"])) or []) > 0, len(chains.get(str(row["argument_2_id"])) or []) > 0]
            counts["n"] += 1
            counts[str(sum(changed))] += 1
    return {
        "population": "biored_train_scoring_eligible",
        "comparison": "E2_to_E3",
        "n_candidates": counts["n"],
        "neither_argument_changes": counts["0"],
        "one_argument_changes": counts["1"],
        "both_arguments_change": counts["2"],
        "matched_changed_population": counts["1"] + counts["2"],
    }


def _drugprot_identifiers(paths) -> dict:
    local = 0
    other = 0
    path = paths.prepared / "drugprot" / CURRENT_VERSIONS["drugprot"] / "entities.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            term = str(json.loads(line).get("term") or "")
            if len(term) > 1 and term[0] == "T" and term[1:].isdigit():
                local += 1
            else:
                other += 1
    return {
        "entity_level_experiment": "not_proposed",
        "n_local_mention_identifiers": local,
        "n_other_identifiers": other,
        "normalised_concept_identifier": False,
        "reason": "DrugProt arguments are local mention terms such as T1. No external linking resource was applied.",
    }


def _resource_record(paths, so_version: str | None, n_so: int, n_cell_lines: int) -> dict:
    files = {
        "mesh_desc2026": paths.raw / "mesh" / "nlm_2026" / "desc2026.gz",
        "sequence_ontology": paths.raw / "sequence_ontology" / "so.obo",
        "ncbi_taxonomy_dump": paths.raw / "taxonomy" / "taxdump.tar.gz",
        "cellosaurus": paths.raw / "cellosaurus" / "cellosaurus.obo",
    }
    return {
        "mesh": {"release": "desc2026", "sha256": digest_file(files["mesh_desc2026"]), "role": "disease and chemical ancestor classes"},
        "sequence_ontology": {"release": so_version, "n_terms_read": n_so, "sha256": digest_file(files["sequence_ontology"]), "role": "variant class for a documented PubTator type code"},
        "ncbi_taxonomy": {"source": "ftp://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz", "sha256": digest_file(files["ncbi_taxonomy_dump"]), "role": "organism ancestor ranks"},
        "cellosaurus": {"release": "57.0", "n_accessions_used": n_cell_lines, "sha256": digest_file(files["cellosaurus"]), "role": "cell-line category, not disease of origin"},
        "umls": "not available",
        "mondo": "not acquired; MeSH identifiers already supply the disease class path, and a second hierarchy was not merged into it",
        "chebi": "not acquired; prepared chemical identifiers are MeSH descriptors, and pharmacological roles are excluded",
        "not_committed": True,
    }


def _top_information(chains: list[list[str]]) -> dict:
    scores = information_content(chains)
    ranked = sorted(scores.items(), key=lambda item: item[1])[:15]
    return {
        "denominator": "BioRED training concepts with an eligible classification, including those with an empty additional chain",
        "unit": "nats",
        "multiple_inheritance": "each concept is counted once along its single selected path",
        "lowest_information_examples": [{"node": node, "nats": value} for node, value in ranked],
    }


def _civic(path: Path) -> dict:
    total = 0
    lung = Counter()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            total += 1
            if "lung" in (row.get("disease") or "").lower():
                lung[row.get("significance") or ""] += 1
    return {
        "source": "https://civicdb.org/downloads/nightly/nightly-ClinicalEvidenceSummaries.tsv",
        "sha256": digest_file(path),
        "n_evidence_rows": total,
        "n_lung_disease_rows": sum(lung.values()),
        "lung_significance": dict(lung),
        "historical_release_compared": False,
        "literature_extracted": False,
    }


def _alternatives(paths) -> dict:
    schemas = {
        "biored": load_schema(default_schema_path(paths.code_root, "biored")),
        "drugprot": load_schema(default_schema_path(paths.code_root, "drugprot")),
    }
    counts = {name: _gold_counts(paths, name, schema) for name, schema in schemas.items()}
    recorded = []
    for item in load_alternatives(paths.code_root / "configs" / "schemas" / "predicate_alternatives.yaml"):
        schema = schemas[item["corpus"]]
        measured = partition_information(schema, counts[item["corpus"]], item["groups"])
        recorded.append(
            {
                "id": item["id"],
                "display_names": [group["display_name"] for group in item["groups"]],
                "historical_display_names": [group.get("historical_display_name") for group in item["groups"]],
                "members_unchanged_from_v1": True,
                "entropy_bits": measured["entropy_bits"],
                "retained_fraction": measured["retained_fraction"],
                "n_relations_whose_group_is_broader": measured["n_relations_whose_group_is_broader"],
                "n_relations_unchanged": measured["n_relations_unchanged"],
            }
        )
    return {"interpretation_version": "predicate_alternatives_interpretation_v2", "partitions": recorded}


def _gold_counts(paths, corpus: str, schema) -> dict[str, int]:
    names = {name: fine_id for fine_id, name in schema.fine_names.items()}
    counts = Counter()
    dataset = ds.dataset(paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "candidates.parquet", format="parquet")
    for batch in dataset.to_batches(columns=["gold_labels"], filter=ds.field("split") == "train"):
        for row in batch.to_pylist():
            for label in row["gold_labels"] or []:
                fine_id = names.get(label)
                if fine_id:
                    counts[fine_id] += 1
    return dict(counts)


def _matrix(paths, notation: dict, entity_levels: list[str]) -> dict:
    accepted_path = paths.study / "reports" / "representation_granularity" / "granularity_measurements.json"
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))["predictions"]["accepted"]
    reusable = {
        (row["dataset"], row["entity_level"], row["predicate_level"], row["model"], row["kind"], str(row.get("seed") or ""))
        for row in accepted
    }
    cells = []
    new_training = 0
    new_inference = 0
    reuse_direct = 0
    abstraction_available = 0
    requirement_counts = Counter()
    datasets = ("biored", "bc8", "drugprot")
    for dataset in datasets:
        entities = ["native"] if dataset == "drugprot" else list(entity_levels)
        for entity in entities:
            for research in RESEARCH_PREDICATES:
                historical = historical_level(notation, research)
                for model in MODELS:
                    if dataset == "bc8" and model["trains"]:
                        continue
                    for seed in model["seeds"]:
                        key = (dataset, entity if dataset != "drugprot" else "native", historical, model["model"], model["kind"], seed)
                        known_entity = entity in set(NATIVE_ENTITY_LEVELS) | {"native"}
                        direct = key in reusable and known_entity
                        native_key = (dataset, entity if dataset != "drugprot" else "native", "P4", model["model"], model["kind"], seed)
                        abstraction = research != "P3" and native_key in reusable and known_entity
                        if abstraction:
                            abstraction_available += 1
                        if direct:
                            reuse_direct += 1
                            requirement = "reuse_direct_prediction"
                        elif abstraction:
                            requirement = "reuse_native_prediction_by_abstraction"
                        elif model["trains"]:
                            new_training += 1
                            requirement = "new_training"
                        else:
                            new_inference += 1
                            requirement = "new_inference"
                        requirement_counts[requirement] += 1
                        cells.append(
                            {
                                "dataset": dataset,
                                "entity_level": entity,
                                "research_predicate": research,
                                "historical_predicate": historical,
                                "model": model["model"],
                                "kind": model["kind"],
                                "seed": seed,
                                "requirement": requirement,
                            }
                        )
    return {
        "n_cells": len(cells),
        "reuse_direct_prediction": reuse_direct,
        "abstraction_available_from_native_predictions": abstraction_available,
        "new_training": new_training,
        "new_inference": new_inference,
        "requirement_counts": dict(requirement_counts),
        "accepted_prediction_conditions": len(accepted),
        "cells": cells,
        "wording_caveat": "Historical P1 prompts use Any_Relation. A prompt that instead says RELATED_TO is not treated as the same generation.",
    }
