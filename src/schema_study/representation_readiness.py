"""Measurements for the representation and source-reference protocol.

The command reads prepared data, the public MeSH descriptor file, and the
existing source-reference files. It does not train a model or replace those
files.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.dataset as ds

from schema_study.entity_classification import (
    MESH_RELEASE,
    MESH_SOURCE,
    identifier_family,
    load_mesh_descriptors,
    summarise_mappings,
)
from schema_study.paths import get_paths
from schema_study.predicate_alternatives import load_alternatives, partition_information, random_control_description
from schema_study.records import digest_file, write_json
from schema_study.schemas import default_schema_path, load_schema
from schema_study.sentences import (
    SENTENCE_SEGMENTATION_VERSION,
    terminal_punctuation_sentences,
)
from schema_study.source_integrity import inspect_jsonl, population_difference
from schema_study.source_references import CURRENT_VERSIONS, TARGET_SPLITS

MESH_RELATIVE = Path("data/raw/mesh/nlm_2026/desc2026.gz")


def prepare() -> int:
    paths = get_paths()
    destination = paths.assert_output(paths.study / "reports" / "representation_readiness")
    destination.mkdir(parents=True, exist_ok=True)
    print("Checking source-reference files.", flush=True)
    source_report = _source_report(paths)
    print("Reading MeSH descriptors.", flush=True)
    mesh_path = paths.storage_root / MESH_RELATIVE
    descriptors = load_mesh_descriptors(mesh_path)
    print("Measuring identifier coverage.", flush=True)
    entities = _entity_report(paths, descriptors)
    print("Measuring alternative predicate partitions.", flush=True)
    predicates = _predicate_report(paths)
    print("Comparing sentence boundaries.", flush=True)
    sentences = _sentence_report(paths)
    samples = _sample_report(paths)
    payload = {
        "repository_revision_expected": "831ad5e0de8d37df5b1c18568e729f49cd095768",
        "sentence_segmentation_version": SENTENCE_SEGMENTATION_VERSION,
        "mesh": {
            "release": MESH_RELEASE,
            "source": MESH_SOURCE,
            "path": str(mesh_path),
            "sha256": digest_file(mesh_path),
            "n_descriptors": len(descriptors),
            "umls": "not_available",
            "umls_limitation": "No UMLS distribution or licence credential is present. MeSH descriptors are not a substitute for UMLS semantic types.",
        },
        "source_references": source_report,
        "entities": entities,
        "predicates": predicates,
        "sentences": sentences,
        "samples": samples,
        "proposed_entity_conditions": ["E0", "E1", "E2"],
        "additional_entity_level": "not_supported",
        "proposed_predicate_conditions": ["P1", "P2", "P3", "P4"],
        "alternative_partitions_replace_primary": False,
    }
    write_json(destination / "readiness.json", payload)
    write_json(destination / "completion.json", _completion(payload))
    print(f"Wrote {destination}", flush=True)
    return 0


def _source_report(paths) -> dict:
    root = paths.study / "reports" / "source_references"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    report = {"original_files_replaced": False, "corpora": {}}
    for corpus, splits in TARGET_SPLITS.items():
        expected = _candidate_ids(paths, corpus, set(splits))
        corpus_report = {"summary_n_candidates": summary["corpora"][corpus]["n_candidates"], "files": {}}
        observed = {}
        for kind, identifier in (("sentences", "sentence_id"), ("alignments", "candidate_id"), ("passages", "candidate_id")):
            inspected = inspect_jsonl(root / kind / f"{corpus}.jsonl", identifier)
            identifiers = inspected.pop("identifiers")
            documents = inspected.pop("document_ids")
            inspected["n_documents"] = len(documents)
            if kind != "sentences":
                difference = population_difference(identifiers, expected)
                inspected["population"] = difference
                observed[kind] = identifiers
            corpus_report["files"][kind] = inspected
        corpus_report["alignment_and_passage_populations_agree"] = observed["alignments"] == observed["passages"]
        corpus_report["summary_matches_alignment_count"] = (
            summary["corpora"][corpus]["n_candidates"] == corpus_report["files"]["alignments"]["n_records"]
        )
        report["corpora"][corpus] = corpus_report
    return report


def _candidate_ids(paths, corpus: str, splits: set[str]) -> set[str]:
    dataset = ds.dataset(paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "candidates.parquet", format="parquet")
    identifiers = set()
    for batch in dataset.to_batches(columns=["candidate_id"], filter=ds.field("split").isin(sorted(splits))):
        identifiers.update(batch.column("candidate_id").to_pylist())
    return identifiers


def _entity_report(paths, descriptors: dict) -> dict:
    report = {}
    for corpus in ("biored", "bc8"):
        concepts = {}
        mentions = []
        path = paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "entities.jsonl"
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            mentions.append({"concept_id": row.get("concept_id", ""), "entity_type": row.get("entity_type", "")})
            concept_id = str(row.get("concept_id") or "")
            slot = concepts.setdefault(
                concept_id,
                {"concept_id": concept_id, "entity_type": row.get("entity_type", ""), "entity_types": set()},
            )
            slot["entity_types"].add(row.get("entity_type", ""))
            if len(slot["entity_types"]) > 1:
                slot["entity_type"] = "type_conflict"
                slot["type_conflict"] = True
        report[corpus] = {
            "mentions": summarise_mappings(mentions, descriptors),
            "concepts": summarise_mappings(list(concepts.values()), descriptors),
        }
    families = Counter()
    path = paths.prepared / "drugprot" / CURRENT_VERSIONS["drugprot"] / "entities.jsonl"
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        families[identifier_family(row.get("term") or row.get("mention_id") or "")] += 1
    report["drugprot"] = {
        "normalised_concept_identifier": False,
        "mention_identifier_families": dict(families),
        "entity_level_experiment": "not_proposed",
        "reason": "DrugProt arguments are local mention terms. External linking would be a separate problem from classification.",
    }
    report["proposal"] = {
        "additional_experimental_level": False,
        "reason": (
            "MeSH supplies tree paths for some descriptor-identified concepts, often more than one path. "
            "Genes, variants, cell lines, supplemental concepts, and DrugProt mentions do not receive a "
            "defensible finer type from this resource. No additional entity condition is added."
        ),
    }
    return report


def _predicate_report(paths) -> dict:
    alternatives = load_alternatives(paths.code_root / "configs" / "schemas" / "predicate_alternatives.yaml")
    report = {"primary_hierarchy_replaced": False, "alternatives": [], "random_control": {}}
    for item in alternatives:
        schema = load_schema(default_schema_path(paths.code_root, item["corpus"]))
        counts = _training_counts(paths, item["corpus"], schema)
        measured = partition_information(schema, counts, item["groups"])
        report["alternatives"].append(
            {
                "id": item["id"],
                "corpus": item["corpus"],
                "question": item["question"],
                "semantic_status": item["semantic_status"],
                "preserves": item["preserves"],
                "omits": item["omits"],
                "direction": item["direction"],
                "measurement": measured,
                "direct_model_prediction": "not_run",
                "saved_native_predictions_can_be_mapped": True,
                "mapped_prediction_establishes_direct_performance": False,
            }
        )
        if item["corpus"] not in report["random_control"]:
            report["random_control"][item["corpus"]] = random_control_description(schema)
    return report


def _training_counts(paths, corpus: str, schema) -> dict[str, int]:
    names = {name: fine_id for fine_id, name in schema.fine_names.items()}
    counts = {fine_id: 0 for fine_id in schema.fine_ids}
    dataset = ds.dataset(paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "candidates.parquet", format="parquet")
    for batch in dataset.to_batches(columns=["gold_labels", "split"], filter=ds.field("split") == "train"):
        for labels in batch.column("gold_labels").to_pylist():
            for label in labels or []:
                if label in names:
                    counts[names[label]] += 1
    return counts


def _sentence_report(paths) -> dict:
    disagreements = Counter()
    compared = 0
    documents = paths.prepared / "biored" / CURRENT_VERSIONS["biored"] / "documents.jsonl"
    from schema_study.sentences import segment_document

    for line in documents.open(encoding="utf-8"):
        document = json.loads(line)
        if document.get("split") != "dev":
            continue
        compared += 1
        recorded = {row["char_end"] for row in segment_document(document)}
        terminal = {row["char_end"] for row in terminal_punctuation_sentences(document)}
        disagreements["recorded_only"] += len(recorded - terminal)
        disagreements["terminal_only"] += len(terminal - recorded)
        disagreements["shared"] += len(recorded & terminal)
    examples = []
    for name, text in _boundary_examples():
        document = {
            "document_id": name,
            "split": "example",
            "title_char_end": 0,
            "abstract_char_start": 0,
            "separator": "",
            "text": text,
        }
        # The examples are abstract fragments. Put the whole string in the abstract.
        document["title_char_end"] = 0
        document["abstract_char_start"] = 0
        document["separator"] = ""
        document["text"] = text
        recorded = [row["text"] for row in segment_document(document)]
        terminal = [row["text"] for row in terminal_punctuation_sentences(document)]
        examples.append(
            {
                "name": name,
                "recorded": recorded,
                "terminal_punctuation": terminal,
                "human_reference": None,
            }
        )
    return {
        "recorded_version": SENTENCE_SEGMENTATION_VERSION,
        "comparison_corpus": "biored development documents",
        "n_documents": compared,
        "boundary_end_offsets": dict(disagreements),
        "reference_boundary_f1": None,
        "reason_f1_unavailable": "No adjudicated sentence-boundary reference is available for these documents.",
        "agreement_is_not_correctness": True,
        "examples": examples,
        "selected_method": "unresolved",
    }


def _boundary_examples() -> list[tuple[str, str]]:
    return [
        ("variant_notation", "The mutation is p.R229G in exon 8."),
        ("abbreviation_before_capital", "The interval was 120 min. After treatment the level fell."),
        ("genomic_coordinate", "The allele is ECA3g.79544206A>T in this breed."),
        ("numbered_chemical", "Compound 7. 8-Epi-PGF2 alpha showed weak binding."),
        ("decimal", "The dose was 0.5 mg. The gene rose."),
    ]


def _sample_report(paths) -> dict:
    root = paths.study / "reports" / "source_references"
    development = _documents(root / "development_sample" / "examples.jsonl")
    human = _documents(root / "human_review" / "examples.jsonl")
    return {
        "development_documents": len(development),
        "human_documents": len(human),
        "shared_documents": sorted(development & human),
        "document_disjoint": development.isdisjoint(human),
        "roles": {
            "current_development_sample": "method_development",
            "current_human_sample": "diagnostic_enrichment",
            "representative_evaluation": "not_drawn",
            "independent_confirmation": "not_available",
        },
        "confirmation_limitation": (
            "BioRED test, BC8, and DrugProt evaluation have already been used for scoring. "
            "They are not untouched confirmation data."
        ),
    }


def _documents(path: Path) -> set[str]:
    documents = set()
    for line in path.open(encoding="utf-8"):
        documents.add(str(json.loads(line)["document_id"]))
    return documents


def _completion(payload: dict) -> dict:
    complete = all(
        item["files"][kind]["population"]["complete"]
        for item in payload["source_references"]["corpora"].values()
        for kind in ("alignments", "passages")
    )
    return {
        "status": "complete" if complete else "incomplete",
        "source_reference_populations_complete": complete,
        "mesh_release": MESH_RELEASE,
        "additional_entity_level": payload["additional_entity_level"],
        "sentence_method_selected": payload["sentences"]["selected_method"],
    }
