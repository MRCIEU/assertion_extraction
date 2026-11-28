"""Measure the frozen representation levels and audit saved predictions.

The command reads prepared tables and formal outputs. It does not train, call
a model, or rewrite those outputs.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.dataset as ds

from schema_study.batch_plan import adopted_answers
from schema_study.contexts import INPUT_VERSION
from schema_study.granularity import (
    SEMANTIC_GROUP_NOTES,
    candidate_type_changes,
    compare_identifiers,
    entity_type_information,
    label_record_problem,
    mixed_schema_direction_groups,
    predicate_information,
    structural_hierarchy_errors,
)
from schema_study.paths import get_paths
from schema_study.predictions import parse_prediction_text
from schema_study.records import write_json, write_tsv
from schema_study.reporting import plot_level_series, plot_named_bars
from schema_study.schemas import default_schema_path, load_schema

CURRENT_VERSIONS = {
    "biored": "biored_candidates_v2",
    "bc8": "bc8_candidates_v2",
    "drugprot": "drugprot_candidates_v1",
}
REFERENCE_SPLIT = {"biored": "train", "drugprot": "train"}
EVALUATION_SPLITS = {"biored": "test", "bc8": "bc8", "drugprot": "eval"}
FORMAL_RUN = "formal_20261115"


def measure() -> int:
    paths = get_paths()
    destination = paths.assert_output(paths.study / "reports" / "representation_granularity")
    destination.mkdir(parents=True, exist_ok=True)
    schemas = {
        "biored": load_schema(default_schema_path(paths.code_root, "biored")),
        "drugprot": load_schema(default_schema_path(paths.code_root, "drugprot")),
    }
    schemas["bc8"] = schemas["biored"]
    structure = {
        name: {
            "structural_errors": structural_hierarchy_errors(schema),
            "mixed_schema_directions": mixed_schema_direction_groups(schema),
            "semantic_notes_require_human_judgement": SEMANTIC_GROUP_NOTES[name],
        }
        for name, schema in schemas.items()
        if name != "bc8"
    }
    if any(item["structural_errors"] for item in structure.values()):
        write_json(destination / "structural_checks.json", structure)
        print("Structural hierarchy checks failed.")
        return 1
    print("Reading candidate tables.", flush=True)
    populations = {name: _population(paths, name) for name in CURRENT_VERSIONS}
    documents = {name: _documents(paths, name) for name in CURRENT_VERSIONS}
    predicates = _predicate_measurements(schemas, populations)
    entities = _entity_measurements(paths, documents, populations)
    shared = _shared_documents(documents)
    sources = _source_availability(paths, populations)
    print("Checking saved predictions.", flush=True)
    predictions = _audit_predictions(paths, schemas, populations, sources)
    payload = {
        "input_version": INPUT_VERSION,
        "data_versions": CURRENT_VERSIONS,
        "schema_sha256": {
            name: hashlib.sha256(default_schema_path(paths.code_root, name).read_bytes()).hexdigest()
            for name in ("biored", "drugprot")
        },
        "formal_run": str(paths.study / "runs" / "formal" / FORMAL_RUN),
        "output_directory": str(destination),
        "structure": structure,
        "predicates": predicates,
        "entities": entities,
        "shared_documents": shared,
        "source_availability": sources,
        "predictions": predictions,
    }
    write_json(destination / "granularity_measurements.json", payload)
    _write_tables(destination, predicates, entities, predictions, shared)
    _write_figures(destination, predicates, entities)
    print(json.dumps(_brief(predicates, entities, predictions), indent=2))
    return 0


def _prepared(paths, corpus: str) -> Path:
    directory = paths.prepared / corpus / CURRENT_VERSIONS[corpus]
    summary = json.loads((directory / "data_summary.json").read_text(encoding="utf-8"))
    if summary["data_version"] != CURRENT_VERSIONS[corpus]:
        raise ValueError(f"{corpus} summary version is {summary['data_version']}")
    return directory


def _population(paths, corpus: str) -> dict:
    directory = _prepared(paths, corpus)
    schema = load_schema(default_schema_path(paths.code_root, "biored" if corpus == "bc8" else corpus))
    names = {name: fine_id for fine_id, name in schema.fine_names.items()}
    by_split = {}
    dataset = ds.dataset(directory / "candidates.parquet", format="parquet")
    columns = [
        "candidate_id",
        "split",
        "document_id",
        "gold_labels",
        "argument_1_type",
        "argument_2_type",
        "mention_ids_1",
        "mention_ids_2",
        "included_in_scoring",
    ]
    for batch in dataset.to_batches(columns=columns):
        for row in batch.to_pylist():
            if not row.get("included_in_scoring", True):
                continue
            slot = by_split.setdefault(
                row["split"],
                {
                    "gold": Counter(),
                    "n_candidates": 0,
                    "n_positive_candidates": 0,
                    "n_unannotated_candidates": 0,
                    "pairs": Counter(),
                    "rows": {},
                },
            )
            slot["n_candidates"] += 1
            labels = list(row["gold_labels"] or [])
            if labels:
                slot["n_positive_candidates"] += 1
            else:
                slot["n_unannotated_candidates"] += 1
            for label in labels:
                if label not in names:
                    raise KeyError(f"{corpus} gold label is not in the schema: {label}")
                slot["gold"][names[label]] += 1
            slot["pairs"][(row["argument_1_type"], row["argument_2_type"])] += 1
            if row["split"] == EVALUATION_SPLITS[corpus]:
                slot["rows"][row["candidate_id"]] = {
                    "document_id": row["document_id"],
                    "split": row["split"],
                    "mentions": list(row["mention_ids_1"] or []) + list(row["mention_ids_2"] or []),
                }
    return {"directory": str(directory), "by_split": by_split}


def _documents(paths, corpus: str) -> dict:
    splits = {}
    missing_text = 0
    for line in (_prepared(paths, corpus) / "documents.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        splits[str(row["document_id"])] = row["split"]
        if not row.get("text"):
            missing_text += 1
    return {"splits": splits, "n_missing_text": missing_text}


def _predicate_measurements(schemas, populations) -> dict:
    biored_train = populations["biored"]["by_split"]["train"]["gold"]
    drugprot_train = populations["drugprot"]["by_split"]["train"]["gold"]
    result = {
        "biored": _wrap_predicate(
            "biored_candidates_v2",
            "train",
            populations["biored"]["by_split"]["train"],
            predicate_information(schemas["biored"], biored_train),
        ),
        "bc8": _wrap_predicate(
            "bc8_candidates_v2",
            "bc8",
            populations["bc8"]["by_split"]["bc8"],
            predicate_information(schemas["bc8"], biored_train, populations["bc8"]["by_split"]["bc8"]["gold"]),
        ),
        "drugprot": _wrap_predicate(
            "drugprot_candidates_v1",
            "train",
            populations["drugprot"]["by_split"]["train"],
            predicate_information(schemas["drugprot"], drugprot_train),
        ),
    }
    result["bc8"]["reference_distribution"] = "BioRED training positive relation labels"
    result["biored"]["reference_distribution"] = "BioRED training positive relation labels"
    result["drugprot"]["reference_distribution"] = "DrugProt training positive relation labels"
    result["evaluation_transitions"] = {
        "biored_test": predicate_information(schemas["biored"], biored_train, populations["biored"]["by_split"]["test"]["gold"])["transitions"],
        "drugprot_eval": predicate_information(schemas["drugprot"], drugprot_train, populations["drugprot"]["by_split"]["eval"]["gold"])["transitions"],
    }
    return result


def _wrap_predicate(data_version: str, split: str, population: dict, measured: dict) -> dict:
    measured["data_version"] = data_version
    measured["observation_split"] = split
    measured["n_candidates"] = population["n_candidates"]
    measured["n_positive_candidates"] = population["n_positive_candidates"]
    measured["n_unannotated_candidates"] = population["n_unannotated_candidates"]
    measured["unannotated_pairs_excluded_from_entropy"] = True
    return measured


def _entity_measurements(paths, documents, populations) -> dict:
    result = {}
    for corpus in ("biored", "bc8"):
        split = REFERENCE_SPLIT["biored"] if corpus == "biored" else "bc8"
        mentions, concepts, conflicts = _entity_counts(paths, corpus, documents[corpus]["splits"], split)
        result[corpus] = {
            "mentions": entity_type_information(mentions, unit="mention"),
            "concepts": entity_type_information(concepts, unit="normalised_concept"),
            "n_concept_type_conflicts": conflicts,
            "candidates": candidate_type_changes(populations[corpus]["by_split"][split]["pairs"].elements()),
            "reference_split": split,
        }
    original, unified = _drugprot_types(paths, documents["drugprot"]["splits"], "train")
    result["drugprot"] = {
        "representation": "native",
        "hierarchy": None,
        "reference_split": "train",
        "unified_input_type": dict(unified),
        "original_annotation_type": dict(original),
        "normalised_concept_identifiers": False,
        "note": "DrugProt has no E0–E2 comparison. Local mention identifiers are not biological concept identifiers.",
    }
    return result


def _entity_counts(paths, corpus: str, splits: dict[str, str], split: str):
    mentions = Counter()
    concepts: dict[tuple[str, str], str] = {}
    conflicts = 0
    for line in (_prepared(paths, corpus) / "entities.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if splits.get(str(row["document_id"])) != split:
            continue
        mentions[row["entity_type"]] += 1
        concept_id = str(row.get("concept_id") or "")
        if not concept_id or concept_id == "-":
            continue
        key = (str(row["document_id"]), concept_id)
        previous = concepts.get(key)
        if previous is None:
            concepts[key] = row["entity_type"]
        elif previous != row["entity_type"]:
            conflicts += 1
            concepts.pop(key, None)
    concept_counts = Counter(concepts.values())
    return mentions, concept_counts, conflicts


def _drugprot_types(paths, splits: dict[str, str], split: str):
    original = Counter()
    unified = Counter()
    for line in (_prepared(paths, "drugprot") / "entities.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if splits.get(str(row["document_id"])) != split:
            continue
        unified[row["entity_type"]] += 1
        original[row.get("entity_type_original") or row["entity_type"]] += 1
    return original, unified


def _shared_documents(documents) -> dict:
    membership = {}
    for corpus, payload in documents.items():
        for document_id, split in payload["splits"].items():
            membership.setdefault(document_id, []).append({"corpus": corpus, "split": split})
    shared = [
        {"document_id": document_id, "membership": items}
        for document_id, items in sorted(membership.items())
        if len({item["corpus"] for item in items}) > 1
    ]
    return {
        "rule": "A future cross-corpus sensitivity population excludes every document identifier that occurs in more than one corpus. The stored splits are unchanged.",
        "n_shared_documents": len(shared),
        "documents": shared,
        "within_corpus_duplicate_documents": {
            corpus: len(payload["splits"]) != len(set(payload["splits"]))
            for corpus, payload in documents.items()
        },
        "documents_missing_text": {corpus: payload["n_missing_text"] for corpus, payload in documents.items()},
    }


def _source_availability(paths, populations) -> dict:
    availability = {}
    for corpus, population in populations.items():
        split = EVALUATION_SPLITS[corpus]
        rows = population["by_split"][split]["rows"]
        context_ids = _context_ids(paths, corpus)
        mention_ids = _mention_ids(paths, corpus)
        missing_context = 0
        missing_mentions = 0
        for candidate_id, row in rows.items():
            if candidate_id not in context_ids:
                missing_context += 1
            if any(mention_id not in mention_ids for mention_id in row["mentions"]):
                missing_mentions += 1
        availability[corpus] = {
            "split": split,
            "n_candidates": len(rows),
            "n_missing_context": missing_context,
            "n_missing_entity_mentions": missing_mentions,
            "input_version": INPUT_VERSION,
        }
    return availability


def _context_ids(paths, corpus: str) -> set[str]:
    matches = sorted((paths.inputs / corpus).glob("*/contexts.jsonl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{corpus} context files: {matches}")
    found = set()
    marker = '"candidate_id": "'
    for line in matches[0].open(encoding="utf-8"):
        start = line.find(marker)
        if start < 0:
            continue
        start += len(marker)
        found.add(line[start:line.find('"', start)])
    return found


def _mention_ids(paths, corpus: str) -> set[str]:
    found = set()
    marker = '"mention_id": "'
    for line in (_prepared(paths, corpus) / "entities.jsonl").open(encoding="utf-8"):
        start = line.find(marker)
        if start < 0:
            continue
        start += len(marker)
        found.add(line[start:line.find('"', start)])
    return found


def _audit_predictions(paths, schemas, populations, sources) -> dict:
    formal = paths.study / "runs" / "formal" / FORMAL_RUN
    conditions = []
    conditions.extend(_audit_training(formal, schemas, populations, sources))
    conditions.extend(_audit_generation(formal, schemas, populations, sources))
    accepted = [row for row in conditions if row["accepted_for_source_level_evaluation"]]
    return {
        "n_conditions_examined": len(conditions),
        "n_accepted": len(accepted),
        "accepted": [
            {key: row[key] for key in ("kind", "dataset", "entity_level", "predicate_level", "model", "seed", "n_records")}
            for row in accepted
        ],
        "conditions": conditions,
    }


def _audit_training(formal: Path, schemas, populations, sources) -> list[dict]:
    rows = []
    root = formal / "train"
    if not root.is_dir():
        return rows
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for entity_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            for predicate_dir in sorted(path for path in entity_dir.iterdir() if path.is_dir()):
                if "invalid" in predicate_dir.name:
                    rows.append(
                        {
                            "kind": "fine_tuning",
                            "dataset": dataset_dir.name,
                            "entity_level": entity_dir.name,
                            "predicate_level": predicate_dir.name,
                            "model": "biomedbert",
                            "seed": "",
                            "accepted_for_source_level_evaluation": False,
                            "reason": "invalid_duplicate_writer_directory",
                            "n_records": 0,
                        }
                    )
                    continue
                for seed_dir in sorted(predicate_dir.glob("seed*")):
                    if not seed_dir.name[4:].isdigit():
                        continue
                    rows.append(
                        _audit_prediction_file(
                            seed_dir / "predictions.jsonl",
                            complete=(seed_dir / "prediction_complete.json").is_file(),
                            dataset=dataset_dir.name,
                            entity_level=entity_dir.name,
                            predicate_level=predicate_dir.name,
                            seed=seed_dir.name[4:],
                            schemas=schemas,
                            populations=populations,
                            sources=sources,
                        )
                    )
    return rows


def _audit_prediction_file(path: Path, *, complete: bool, dataset: str, entity_level: str, predicate_level: str, seed: str, schemas, populations, sources) -> dict:
    expected = {}
    targets = ["biored", "bc8"] if dataset == "biored" else [dataset]
    for name in targets:
        split = EVALUATION_SPLITS[name]
        for candidate_id, row in populations[name]["by_split"][split]["rows"].items():
            expected[candidate_id] = {"dataset": name, **row}
    observed = []
    problems = Counter()
    statuses = Counter()
    if path.is_file():
        schema_for = {"biored": schemas["biored"], "bc8": schemas["biored"], "drugprot": schemas["drugprot"]}
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_id = row.get("candidate_id")
            observed.append(candidate_id)
            statuses[row.get("output_status")] += 1
            if row.get("dataset") not in targets or row.get("entity_level") != entity_level or row.get("predicate_level") != predicate_level or str(row.get("seed")) != seed:
                problems["condition_field_mismatch"] += 1
            reference = expected.get(candidate_id)
            schema = schema_for.get(row.get("dataset"), schema_for[dataset])
            problem = label_record_problem(schema, predicate_level, row.get("output_status"), row.get("predicted_labels") or [], row.get("illegal_labels") or [])
            if problem:
                problems[problem] += 1
            if reference is None:
                continue
            if row.get("document_id") != reference["document_id"] or row.get("split") != reference["split"] or row.get("dataset") != reference["dataset"]:
                problems["identifier_disagreement"] += 1
    coverage = compare_identifiers(set(expected), observed)
    source_gap = any(sources[name]["n_missing_context"] or sources[name]["n_missing_entity_mentions"] for name in targets)
    accepted = complete and coverage["identity_complete"] and not problems and not source_gap
    return {
        "kind": "fine_tuning",
        "dataset": dataset,
        "entity_level": entity_level,
        "predicate_level": predicate_level,
        "model": "biomedbert",
        "seed": seed,
        "prediction_file_present": path.is_file(),
        "prediction_complete_record": complete,
        "n_records": coverage["n_observed_rows"],
        "coverage": coverage,
        "output_status_counts": dict(statuses),
        "rule_problems": dict(problems),
        "source_records_available": not source_gap,
        "accepted_for_source_level_evaluation": accepted,
        "reason": "complete" if accepted else "incomplete_or_inconsistent",
    }


def _audit_generation(formal: Path, schemas, populations, sources) -> list[dict]:
    summary_path = formal / "prepare" / "length_summary.json"
    if not summary_path.is_file():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    grouped = {}
    for shard in summary["shards"]:
        key = (shard["dataset"], shard["entity_level"], shard["predicate_level"])
        grouped.setdefault(key, []).append(shard)
    rows = []
    for model in ("qwen32b", "qwen8b"):
        for key, shards in sorted(grouped.items()):
            print(f"Checking {model} {key[0]} {key[1]} {key[2]}.", flush=True)
            rows.append(_audit_generation_condition(formal, model, shards, schemas, populations, sources))
    return rows


def _audit_generation_condition(formal, model, shards, schemas, populations, sources) -> dict:
    dataset = shards[0]["dataset"]
    entity_level = shards[0]["entity_level"]
    predicate_level = shards[0]["predicate_level"]
    schema = schemas["biored"] if dataset in {"biored", "bc8"} else schemas["drugprot"]
    split = EVALUATION_SPLITS[dataset]
    expected = populations[dataset]["by_split"][split]["rows"]
    observed = []
    problems = Counter()
    statuses = Counter()
    ledger_disagreements = 0
    for shard in shards:
        requests = {}
        for line in Path(shard["path"]).open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                requests[row["request_id"]] = row
        attempt_path = (
            formal / "generation" / "outputs" / model / dataset / entity_level / predicate_level / f"shard_{int(shard['shard_index']):05d}" / "attempts.jsonl"
        )
        selected = {}
        field_selected = {}
        if attempt_path.is_file():
            attempts = [json.loads(line) for line in attempt_path.open(encoding="utf-8") if line.strip()]
            selected = adopted_answers(attempts)
            for row in attempts:
                if row.get("adoption") != "adopt":
                    continue
                if row["request_id"] in field_selected:
                    problems["duplicate_adoption_field"] += 1
                else:
                    field_selected[row["request_id"]] = row
        if set(selected) != set(field_selected):
            ledger_disagreements += len(set(selected) ^ set(field_selected))
        for request_id, request in requests.items():
            answer = field_selected.get(request_id)
            if answer is None:
                problems["missing_adopted_answer"] += 1
                continue
            if answer.get("candidate_id") != request.get("candidate_id") or answer.get("document_id") != request.get("document_id") or answer.get("split") != request.get("split"):
                problems["request_disagreement"] += 1
            reference = expected.get(answer.get("candidate_id"))
            if reference is None or answer.get("document_id") != reference["document_id"] or answer.get("split") != reference["split"]:
                problems["identifier_disagreement"] += 1
            observed.append(answer.get("candidate_id"))
            statuses[answer.get("output_status")] += 1
            problem = label_record_problem(schema, predicate_level, answer.get("output_status"), answer.get("predicted_labels") or [], answer.get("illegal_labels") or [])
            if problem:
                problems[problem] += 1
            parsed = parse_prediction_text(
                answer.get("raw_answer"),
                expected_candidate_id=request.get("candidate_id"),
                schema=schema,
                level=predicate_level,
            )
            if parsed["output_status"] != answer.get("output_status") or parsed["predicted_labels"] != answer.get("predicted_labels") or parsed["illegal_labels"] != answer.get("illegal_labels"):
                problems["stored_parse_disagrees"] += 1
    coverage = compare_identifiers(set(expected), observed)
    source_gap = bool(sources[dataset]["n_missing_context"] or sources[dataset]["n_missing_entity_mentions"])
    accepted = coverage["identity_complete"] and not problems and ledger_disagreements == 0 and not source_gap
    return {
        "kind": "prompted_generation",
        "dataset": dataset,
        "entity_level": entity_level,
        "predicate_level": predicate_level,
        "model": model,
        "seed": "",
        "n_shards": len(shards),
        "n_records": coverage["n_observed_rows"],
        "coverage": coverage,
        "output_status_counts": dict(statuses),
        "rule_problems": dict(problems),
        "ledger_disagreements": ledger_disagreements,
        "source_records_available": not source_gap,
        "accepted_for_source_level_evaluation": accepted,
        "reason": "complete" if accepted else "incomplete_or_inconsistent",
    }


def _write_tables(destination: Path, predicates, entities, predictions, shared) -> None:
    predicate_rows = []
    transition_rows = []
    for corpus, measured in predicates.items():
        if corpus == "evaluation_transitions":
            continue
        for level, payload in measured["levels"].items():
            predicate_rows.append(
                {
                    "corpus": corpus,
                    "level": level,
                    "n_categories": payload["n_categories"],
                    "entropy_bits": payload["entropy_bits"],
                    "retained_fraction": payload["retained_fraction"],
                    "n_reference_relations": measured["n_reference_relations"],
                    "n_observed_relations": measured["n_observed_relations"],
                }
            )
        for transition in measured["transitions"]:
            transition_rows.append({"corpus": corpus, **{key: value for key, value in transition.items() if key != "by_label"}})
    write_tsv(
        destination / "predicate_entropy.tsv",
        predicate_rows,
        ["corpus", "level", "n_categories", "entropy_bits", "retained_fraction", "n_reference_relations", "n_observed_relations"],
    )
    write_tsv(
        destination / "predicate_transitions.tsv",
        transition_rows,
        ["corpus", "from_level", "to_level", "n_more_specific", "n_unchanged", "proportion_more_specific", "proportion_unchanged"],
    )
    entity_rows = []
    for corpus in ("biored", "bc8"):
        for unit in ("mentions", "concepts"):
            measured = entities[corpus][unit]
            entity_rows.append(
                {
                    "corpus": corpus,
                    "unit": unit,
                    "n_observations": measured["n_observations_in_native_types"],
                    "entropy_e2_bits": measured["entropy_e2_bits"],
                    "entropy_e1_bits": measured["entropy_e1_bits"],
                    "retained_fraction_e1": measured["retained_fraction_e1"],
                    "proportion_more_specific_at_e2": measured["proportion_more_specific_at_e2"],
                }
            )
    write_tsv(
        destination / "entity_entropy.tsv",
        entity_rows,
        ["corpus", "unit", "n_observations", "entropy_e2_bits", "entropy_e1_bits", "retained_fraction_e1", "proportion_more_specific_at_e2"],
    )
    write_tsv(
        destination / "prediction_conditions.tsv",
        [
            {
                "kind": row["kind"],
                "dataset": row["dataset"],
                "entity_level": row["entity_level"],
                "predicate_level": row["predicate_level"],
                "model": row["model"],
                "seed": row.get("seed", ""),
                "n_records": row.get("n_records", 0),
                "accepted": row["accepted_for_source_level_evaluation"],
                "reason": row.get("reason", ""),
            }
            for row in predictions["conditions"]
        ],
        ["kind", "dataset", "entity_level", "predicate_level", "model", "seed", "n_records", "accepted", "reason"],
    )
    write_json(destination / "shared_documents.json", shared)


def _write_figures(destination: Path, predicates, entities) -> None:
    levels = ["P1", "P2", "P3", "P4"]
    plot_level_series(
        levels,
        {
            "BioRED reference": [predicates["biored"]["levels"][level]["entropy_bits"] for level in levels],
            "DrugProt training": [predicates["drugprot"]["levels"][level]["entropy_bits"] for level in levels],
        },
        destination / "predicate_entropy",
        "Label entropy (bits)",
        synthetic=False,
    )
    transitions = ["P1 to P2", "P2 to P3", "P3 to P4"]
    keys = ["P1_to_P2", "P2_to_P3", "P3_to_P4"]
    plot_level_series(
        transitions,
        {
            "BioRED reference": [predicates["biored"]["entropy_introduced_nats"][key]["bits"] for key in keys],
            "DrugProt training": [predicates["drugprot"]["entropy_introduced_nats"][key]["bits"] for key in keys],
        },
        destination / "predicate_entropy_introduced",
        "Additional label entropy (bits)",
        synthetic=False,
    )
    plot_named_bars(
        ["BioRED mentions", "BioRED concepts", "BC8 mentions", "BC8 concepts"],
        [
            entities["biored"]["mentions"]["retained_fraction_e1"],
            entities["biored"]["concepts"]["retained_fraction_e1"],
            entities["bc8"]["mentions"]["retained_fraction_e1"],
            entities["bc8"]["concepts"]["retained_fraction_e1"],
        ],
        destination / "entity_entropy_retained",
        "E1 entropy / native entity-type entropy",
        synthetic=False,
    )


def _brief(predicates, entities, predictions) -> dict:
    return {
        "biored_retained_fraction": {level: predicates["biored"]["levels"][level]["retained_fraction"] for level in ("P1", "P2", "P3", "P4")},
        "drugprot_retained_fraction": {level: predicates["drugprot"]["levels"][level]["retained_fraction"] for level in ("P1", "P2", "P3", "P4")},
        "biored_transitions": [
            (row["from_level"], row["to_level"], row["proportion_more_specific"]) for row in predicates["biored"]["transitions"]
        ],
        "entity_proportion_more_specific": {
            corpus: entities[corpus]["mentions"]["proportion_more_specific_at_e2"] for corpus in ("biored", "bc8")
        },
        "predictions_accepted": predictions["n_accepted"],
        "predictions_examined": predictions["n_conditions_examined"],
    }


if __name__ == "__main__":
    raise SystemExit(measure())
