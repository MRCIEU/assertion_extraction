"""Prepare the corrected representations, inputs, and learning inventory.

Previous entity-classification and verbalisation files are left in place.
This command writes a separate collection.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.dataset as ds

from schema_study.contexts import build_context_tables, load_entity_configuration
from schema_study.entity_hierarchy import displayed_type, information_content, realised_classification

E1_MERGE = {
    "GeneOrGeneProduct": "Molecular_Entity",
    "ChemicalEntity": "Molecular_Entity",
    "SequenceVariant": "Molecular_Entity",
}
from schema_study.experiments import TRAIN_SEEDS
from schema_study.formal_protocol import CURRENT_VERSIONS
from schema_study.learning_conditions import (
    RESEARCH_PREDICATES,
    bm25_index,
    condition_counts,
    lora_configuration,
    principal_conditions,
    random_demonstrations,
    recover_arguments,
    schema_conditioned_configuration,
    select_demonstrations,
    sft_example,
    verify_lora_modules,
)
from schema_study.paths import get_paths, load_settings
from schema_study.records import publish_file, repository_provenance
from schema_study.schemas import default_schema_path, load_schema, map_fine_ids
from schema_study.source_identification import build_request
from schema_study.verbalisation import VERBALISATION_VERSION, verbalise

REPORT = "experimental_preparation"
INPUT_VERSION = "target_anchored_v2__biomedbert_e1354b7a_matched_e0_e7"
MODEL_IDS = {
    "biomedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "qwen8b": "Qwen/Qwen3-8B",
    "qwen32b": "Qwen/Qwen3-32B",
}


def prepare() -> int:
    paths = get_paths()
    destination = paths.assert_output(paths.study / "reports" / REPORT)
    destination.mkdir(parents=True, exist_ok=True)
    print("Extending displayed representations.", flush=True)
    concepts, levels = _extend_concepts(paths, destination)
    print("Measuring every entity transition.", flush=True)
    transitions = _transitions(paths, concepts, levels)
    print("Verbalising predictions from mention text.", flush=True)
    verbalisation = _verbalise(paths, destination)
    print("Writing the learning inventory.", flush=True)
    inventory = _inventory(paths, levels)
    print("Assigning few-shot demonstrations.", flush=True)
    demonstrations = _few_shot(paths, destination)
    print("Writing supervised example records.", flush=True)
    supervision = _supervision(paths, destination, levels)
    print("Building matched BiomedBERT inputs.", flush=True)
    inputs = _inputs(paths, concepts, levels)
    summary = {
        "proposed_entity_levels": levels,
        "transitions": transitions,
        "verbalisation": verbalisation,
        "inventory": inventory,
        "demonstrations": demonstrations,
        "supervision": supervision,
        "inputs": inputs,
        "lora": lora_configuration(),
        "lora_modules": _lora_modules(paths),
        "schema_conditioned": schema_conditioned_configuration(supervision.get("biored_training_candidates", 0)),
        "civic": _civic(paths),
        "human_sample": _human_sample(paths, destination),
        "sentence_segmenter": _segmenter_status(),
        "measurement_models": _measurement_status(),
        "provenance": repository_provenance(paths.code_root),
        "historical_verbalisation_retained": str(paths.study / "reports" / "entity_classification" / "verbalised_predictions.jsonl"),
        "launched": False,
    }
    publish_file(destination / "summary.json", lambda temporary: temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"))
    print(json.dumps({
        "levels": levels,
        "training_runs": inventory["counts"]["training_runs"],
        "verbalised": verbalisation.get("verbalised"),
        "unresolved_entity": verbalisation.get("unresolved_entity"),
        "inputs": {key: inputs[key] for key in inputs if key in {"biored", "bc8"}},
    }, indent=2, default=str))
    return 0


def _extend_concepts(paths, destination: Path) -> tuple[dict, list[str]]:
    source = paths.study / "reports" / "entity_classification" / "concepts.jsonl"
    payload = json.loads((paths.study / "reports" / "entity_classification" / "summary.json").read_text(encoding="utf-8"))
    levels = list(payload["proposed_entity_levels"])
    concepts = {}
    out = destination / "concepts.jsonl"

    def write(temporary: Path) -> None:
        with source.open(encoding="utf-8") as incoming, temporary.open("w", encoding="utf-8") as handle:
            for line in incoming:
                if not line.strip():
                    continue
                row = json.loads(line)
                displayed, detail = _level_detail(row.get("entity_type"), row.get("chain") or [], levels)
                row["displayed"] = displayed
                row["levels"] = detail
                missing = [level for level in levels if level != "E0" and not displayed.get(level)]
                row["representation_complete"] = not missing
                concepts[(row["corpus"], row["concept_id"])] = row
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    publish_file(out, write)
    return concepts, levels


def _level_detail(native: str | None, chain: list[dict], levels: list[str]) -> tuple[dict, dict]:
    displayed = {}
    detail = {}
    previous = None
    for name in levels:
        level = int(name[1:])
        label = displayed_type(native, chain, level, E1_MERGE)
        if level >= 3:
            realised = realised_classification(chain, level)
            realised_level = realised["realised_level"]
            reason = realised.get("reason") or ("new_distinction" if realised["new_distinction"] else "previous_type_retained")
            carry = bool(realised["carry_forward"])
        elif level == 0:
            realised_level = 0
            reason = "type_withheld"
            carry = False
        else:
            realised_level = level
            reason = "native_representation"
            carry = False
        changed = previous is not None and label != previous
        detail[name] = {
            "requested_level": level,
            "realised_level": realised_level,
            "displayed_type": label,
            "changed_from_previous": changed,
            "previous_type_retained": carry or (level > 0 and not changed),
            "reason": reason,
        }
        displayed[name] = label
        previous = label
    return displayed, detail


def _transitions(paths, concepts: dict, levels: list[str]) -> dict:
    result = {}
    for corpus in ("biored", "bc8"):
        result[corpus] = {}
        splits = ("train", "dev", "test") if corpus == "biored" else (None,)
        for split in splits:
            rows = _candidate_rows(paths, corpus, split)
            if not rows:
                continue
            population = split if split is not None else "bc8"
            result[corpus][population] = {}
            for left, right in zip(levels, levels[1:]):
                result[corpus][population][f"{left}_to_{right}"] = _one_transition(rows, concepts, corpus, left, right)
    training = [row for (corpus, _concept), row in concepts.items() if corpus == "biored" and "train" in row.get("splits", [])]
    complete = []
    conditional = []
    for row in training:
        nodes = [node["node"] for node in row.get("chain") or []]
        if nodes:
            conditional.append(nodes)
            complete.append(nodes)
        else:
            complete.append([f"native:{row.get('entity_type')}"])
    result["information_content"] = {
        "complete_training_concepts": len(complete),
        "conditional_mapped_concepts": len(conditional),
        "complete_definition": "Every BioRED training concept, with the native category retained when no additional class exists.",
        "conditional_definition": "Only BioRED training concepts with a non-empty primary class path.",
    }
    result["information_content_complete_top"] = _top_information(complete)
    result["information_content_conditional_top"] = _top_information(conditional)
    return result


def _one_transition(rows: list[dict], concepts: dict, corpus: str, left: str, right: str) -> dict:
    counts = Counter()
    combinations = Counter()
    for row in rows:
        if not row.get("included_in_scoring", True):
            continue
        counts["full_population"] += 1
        first = concepts.get((corpus, str(row["argument_1_id"])))
        second = concepts.get((corpus, str(row["argument_2_id"])))
        if first is None or second is None:
            counts["missing_concept"] += 1
            continue
        changed_first = first["displayed"].get(left) != first["displayed"].get(right)
        changed_second = second["displayed"].get(left) != second["displayed"].get(right)
        if changed_first or changed_second:
            counts["matched_changed"] += 1
            combinations[f"{first.get('entity_type')}|{second.get('entity_type')}"] += 1
        else:
            counts["unchanged"] += 1
    return {
        "full_population": counts["full_population"],
        "matched_changed": counts["matched_changed"],
        "unchanged": counts["unchanged"],
        "missing_concept": counts["missing_concept"],
        "changed_type_combinations": dict(combinations.most_common(12)),
    }


def _top_information(chains: list[list[str]]) -> list[dict]:
    values = information_content(chains)
    ordered = sorted(values.items(), key=lambda item: -item[1])[:8]
    return [{"node": node, "nats": value} for node, value in ordered]


def _candidate_rows(paths, corpus: str, split: str | None) -> list[dict]:
    path = paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "candidates.parquet"
    if not path.is_file():
        return []
    dataset = ds.dataset(path, format="parquet")
    columns = ["candidate_id", "argument_1_id", "argument_2_id", "included_in_scoring", "split", "gold_status"]
    expression = ds.field("split") == split if split else None
    table = dataset.to_table(columns=columns, filter=expression)
    return table.to_pylist()


def _verbalise(paths, destination: Path) -> dict:
    mentions, candidates = _mention_index(paths, _wanted_candidates(paths))
    schemas = {
        "biored": load_schema(default_schema_path(paths.code_root, "biored")),
        "drugprot": load_schema(default_schema_path(paths.code_root, "drugprot")),
    }
    schemas["bc8"] = schemas["biored"]
    counts = Counter()
    unique = set()
    out = destination / "verbalised_predictions.jsonl"

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8") as handle:
            root = paths.study / "runs" / "formal" / "formal_20261115"
            for path in sorted(root.rglob("predictions.jsonl")) if root.is_dir() else []:
                if "invalid" in str(path):
                    counts["excluded_invalid_directory"] += 1
                    continue
                _write_prediction_file(path, "fine_tuning", "biomedbert", schemas, mentions, candidates, handle, counts, unique)
            output_root = root / "generation" / "outputs"
            for path in sorted(output_root.rglob("attempts.jsonl")) if output_root.is_dir() else []:
                if "invalid" in str(path):
                    counts["excluded_invalid_directory"] += 1
                    continue
                _write_attempts(path, schemas, mentions, candidates, handle, counts, unique)

    publish_file(out, write)
    counts["files_written"] = 1
    counts["deduplicated_positive_propositions"] = len(unique)
    counts["version"] = VERBALISATION_VERSION
    return dict(counts)


def _write_prediction_file(path: Path, paradigm: str, model: str, schemas, mentions, candidates, handle, counts, unique) -> None:
    parts = path.parts
    try:
        anchor = parts.index("formal_20261115")
        dataset = parts[anchor + 2]
        entity = parts[anchor + 3]
        predicate = parts[anchor + 4]
        seed = next((part[4:] for part in parts if part.startswith("seed") and part[4:].isdigit()), "")
    except (ValueError, IndexError):
        counts["skipped_files"] += 1
        return
    schema = schemas.get(dataset)
    if schema is None or predicate not in schema.levels:
        counts["skipped_files"] += 1
        return
    for line in path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            _emit(row, dataset, entity, predicate, model, seed, paradigm, "direct_prediction", schema, mentions, candidates, handle, counts, unique)


def _write_attempts(path: Path, schemas, mentions, candidates, handle, counts, unique) -> None:
    parts = path.parts
    try:
        anchor = parts.index("outputs")
        model = parts[anchor + 1]
        dataset = parts[anchor + 2]
        entity = parts[anchor + 3]
        predicate = parts[anchor + 4]
    except (ValueError, IndexError):
        counts["skipped_files"] += 1
        return
    schema = schemas.get(dataset)
    if schema is None or predicate not in schema.levels:
        counts["skipped_files"] += 1
        return
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("is_warmup") or row.get("adoption") != "adopt":
            continue
        _emit(row, dataset, entity, predicate, model, "", "zero_shot", "direct_prediction", schema, mentions, candidates, handle, counts, unique, revision=row.get("model_revision"))


def _emit(row, dataset, entity, predicate, model, seed, paradigm, route, schema, mentions, candidates, handle, counts, unique, revision=None) -> None:
    counts["records"] += 1
    status = row.get("output_status") or "missing"
    labels = row.get("predicted_labels") or []
    candidate = candidates.get((dataset if dataset != "bc8" else "bc8", row.get("candidate_id")))
    if dataset == "bc8":
        candidate = candidates.get(("bc8", row.get("candidate_id")))
    recovered = recover_arguments(candidate, mentions) if candidate else {"status": "unresolved_entity", "argument_1": None, "argument_2": None}
    base = {
        "corpus": dataset,
        "document_id": row.get("document_id") or (candidate or {}).get("document_id"),
        "candidate_id": row.get("candidate_id"),
        "split": row.get("split") or (candidate or {}).get("split"),
        "model": model,
        "model_id": MODEL_IDS.get(model, model),
        "revision": revision,
        "paradigm": paradigm,
        "seed": seed,
        "run": "formal_20261115",
        "entity_level": entity,
        "predicate_level": predicate,
        "route": route,
        "prediction_status": status,
        "verbalisation_version": VERBALISATION_VERSION,
        "argument_1": recovered.get("argument_1"),
        "argument_2": recovered.get("argument_2"),
    }
    if status != "valid":
        counts[status if status in {"missing", "invalid", "rejected"} else "other_status"] += 1
        return
    if not labels or labels == [schema.no_relation_id]:
        counts["no_positive_assertion"] += 1
        return
    if recovered["status"] != "recovered":
        counts["unresolved_entity"] += 1
        return
    by_id = schema.by_id(predicate)
    for label_id in labels:
        label = by_id.get(label_id)
        if label is None:
            counts["unknown_label"] += 1
            continue
        members = [schema.fine_names.get(member, member) for member in label.members]
        rendered = verbalise(label.display_name, recovered["argument_1"], recovered["argument_2"], members=members, corpus=schema.corpus)
        counts[rendered["status"]] += 1
        if rendered["status"] != "verbalised" or not rendered.get("text"):
            continue
        if recovered["argument_1"] not in rendered["text"] or recovered["argument_2"] not in rendered["text"]:
            counts["assertion_missing_argument"] += 1
            continue
        proposition_id = hashlib.sha256(rendered["text"].encode("utf-8")).hexdigest()[:16]
        unique.add(proposition_id)
        handle.write(json.dumps({**base, "label": label.display_name, "text": rendered["text"], "proposition_id": proposition_id, "definition_status": rendered.get("definition_status")}, ensure_ascii=False) + "\n")


def _wanted_candidates(paths) -> dict[str, set[str]]:
    wanted = {"biored": set(), "bc8": set(), "drugprot": set()}
    root = paths.study / "runs" / "formal" / "formal_20261115"
    if not root.is_dir():
        return wanted
    for path in list(root.rglob("predictions.jsonl")) + list((root / "generation" / "outputs").rglob("attempts.jsonl")):
        if "invalid" in str(path):
            continue
        parts = path.parts
        dataset = "biored"
        if "drugprot" in parts:
            dataset = "drugprot"
        elif "bc8" in parts:
            dataset = "bc8"
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("candidate_id"):
                wanted[dataset].add(row["candidate_id"])
    return wanted


def _mention_index(paths, wanted: dict[str, set[str]]):
    mentions = {}
    candidates = {}
    for corpus, identifiers in wanted.items():
        if not identifiers:
            continue
        entity_path = paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "entities.jsonl"
        for line in entity_path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                mentions[row["mention_id"]] = row
        dataset = ds.dataset(paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "candidates.parquet", format="parquet")
        columns = ["candidate_id", "document_id", "split", "mention_ids_1", "mention_ids_2", "gold_labels", "gold_status", "argument_1_id", "argument_2_id"]
        for batch in dataset.to_batches(columns=columns):
            for row in batch.to_pylist():
                if row["candidate_id"] in identifiers:
                    candidates[(corpus, row["candidate_id"])] = row
    return mentions, candidates


def _inventory(paths, levels: list[str]) -> dict:
    rows = principal_conditions(levels)
    accepted = _accepted(paths)
    decisions = Counter()
    for row in rows:
        if row["unit"] != "direct_inference":
            decisions[row["unit"]] += 1
            continue
        key = (row["dataset"], row["entity_level"], row["historical_predicate_level"], row["model"], row["seed"])
        if row["entity_level"] in {"E0", "E1", "E2", "native"} and key in accepted and row["paradigm"] == "zero_shot":
            decisions["historical_zero_shot_reusable_if_window_matches"] += 1
        elif row["entity_level"] in {"E0", "E1", "E2", "native"} and key in accepted and row["adaptation"] == "full_finetune":
            decisions["historical_finetuned_prediction_reusable_if_window_matches"] += 1
        elif row["paradigm"] == "few_shot" or row["adaptation"] == "lora_sft":
            decisions["new_execution"] += 1
        else:
            decisions["new_execution"] += 1
    manifest = paths.study / "reports" / REPORT / "learning_conditions.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    counts = condition_counts(rows)
    counts["reuse_decisions_before_window_check"] = dict(decisions)
    return {"counts": counts, "n_condition_rows": len(rows), "manifest": str(manifest)}


def _accepted(paths) -> set[tuple]:
    path = paths.study / "reports" / "representation_granularity" / "granularity_measurements.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (row["dataset"], row["entity_level"], row["predicate_level"], row["model"], str(row.get("seed") or ""))
        for row in payload.get("predictions", {}).get("accepted", [])
    }


def _few_shot(paths, destination: Path) -> dict:
    summary = {}
    for corpus, target_split, pool_split in (
        ("biored", "test", "train"),
        ("bc8", None, None),
        ("drugprot", "eval", "train"),
    ):
        pool_corpus = "biored" if corpus == "bc8" else corpus
        pool_rows = _retrieval_rows(paths, pool_corpus, "train")
        targets = _retrieval_rows(paths, corpus, target_split)
        index = bm25_index(pool_rows)
        out = destination / f"few_shot_{corpus}.jsonl"
        n = 0
        with out.open("w", encoding="utf-8") as handle:
            for target in targets:
                chosen = select_demonstrations(target["tokens"], target["document_id"], index, 5)
                handle.write(json.dumps({"candidate_id": target["id"], "document_id": target["document_id"], "demonstrations": chosen}, sort_keys=True) + "\n")
                n += 1
        summary[corpus] = {"targets": n, "pool": len(pool_rows), "demonstrations": 5, "retrieval": "bm25_k1_1.2_b_0.75_tie_break_candidate_id", "output": str(out)}
    return summary


def _retrieval_rows(paths, corpus: str, split: str | None) -> list[dict]:
    mentions = {}
    entity_path = paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "entities.jsonl"
    for line in entity_path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            mentions[row["mention_id"]] = row.get("text") or ""
    dataset = ds.dataset(paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "candidates.parquet", format="parquet")
    expression = ds.field("split") == split if split else None
    rows = []
    for batch in dataset.to_batches(columns=["candidate_id", "document_id", "split", "mention_ids_1", "mention_ids_2", "gold_status"], filter=expression):
        for row in batch.to_pylist():
            if corpus != "drugprot" and split and row["split"] != split:
                continue
            texts = [mentions.get(mention_id, "") for mention_id in list(row["mention_ids_1"] or []) + list(row["mention_ids_2"] or [])]
            tokens = [token.lower() for text in texts for token in str(text).split() if token]
            rows.append({
                "id": row["candidate_id"],
                "document_id": str(row["document_id"]),
                "tokens": tokens or ["empty"],
                "negative": row.get("gold_status") != "unique",
            })
    return rows


def _supervision(paths, destination: Path, levels: list[str]) -> dict:
    schema = load_schema(default_schema_path(paths.code_root, "biored"))
    drugprot = load_schema(default_schema_path(paths.code_root, "drugprot"))
    biored_n = _write_sft(paths, destination / "sft_biored.jsonl", "biored", schema, levels, "train")
    drugprot_n = _write_sft(paths, destination / "sft_drugprot.jsonl", "drugprot", drugprot, ["native"], "train")
    return {
        "biored_example_rows": biored_n,
        "drugprot_example_rows": drugprot_n,
        "biored_training_candidates": biored_n // max(1, len(levels) * len(RESEARCH_PREDICATES)),
        "loss": "assistant_completion_tokens_only",
        "trained_here": False,
    }


def _write_sft(paths, output: Path, corpus: str, schema, levels: list[str], split: str) -> int:
    names = {name: stable for stable, name in schema.fine_names.items()}
    dataset = ds.dataset(paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "candidates.parquet", format="parquet")
    expression = ds.field("split") == split
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for batch in dataset.to_batches(columns=["candidate_id", "gold_labels", "split"], filter=expression):
            for row in batch.to_pylist():
                fine = []
                for label in row["gold_labels"] or []:
                    stable = names.get(label)
                    if stable:
                        fine.append(stable)
                for entity in levels:
                    for predicate in RESEARCH_PREDICATES:
                        historical = {"P0": "P1", "P1": "P2", "P2": "P3", "P3": "P4"}[predicate]
                        grouped = sorted(map_fine_ids(schema, fine, historical)) if fine else []
                        example = sft_example(row["candidate_id"], grouped, entity, predicate)
                        example["corpus"] = corpus
                        example["split"] = split
                        handle.write(json.dumps(example, sort_keys=True) + "\n")
                        written += 1
    return written


def build_matched_shard(corpus: str, shard_index: int, shard_count: int, workers: int) -> int:
    """Build one document shard of the matched E0–E7 inputs.

    The concepts file already written by the preparation command supplies the
    displayed types. This entry point does not repeat verbalisation or retrieval.
    """
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be inside shard_count")
    paths = get_paths()
    concepts, levels = _load_extended_concepts(paths)
    settings = load_settings(paths.code_root)
    tokenizer_dir = Path(str(settings.get("biomedbert_tokenizer_dir") or ""))
    output = paths.inputs / corpus / INPUT_VERSION / "shards" / f"shard_{shard_index:02d}"
    summary = build_context_tables(
        paths.prepared / corpus / CURRENT_VERSIONS[corpus],
        output,
        corpus=corpus,
        tokenizer_dir=tokenizer_dir,
        code_root=paths.code_root,
        workers=workers,
        shard_index=shard_index,
        shard_count=shard_count,
        input_version=INPUT_VERSION,
        views_override=_matched_views(paths, levels),
        display_by_mention=_display_by_mention(paths, corpus, concepts),
    )
    summary["output"] = str(output)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def merge_matched_inputs(corpus: str, shard_count: int = 8) -> dict:
    """Join completed shards and compare their source windows with the frozen inputs."""
    paths = get_paths()
    root = paths.inputs / corpus / INPUT_VERSION
    parts = []
    for index in range(shard_count):
        path = root / "shards" / f"shard_{index:02d}" / "contexts.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Matched input shard is missing: {path}")
        parts.append(path)
    destination = root / "contexts.jsonl"

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8") as outgoing:
            for path in parts:
                with path.open(encoding="utf-8") as incoming:
                    for line in incoming:
                        if line.strip():
                            outgoing.write(line if line.endswith("\n") else line + "\n")

    publish_file(destination, write)
    comparison = _window_comparison(paths, corpus, destination)
    report = {"corpus": corpus, "shards": shard_count, "contexts": str(destination), "window_comparison": comparison}
    publish_file(root / "merge_report.json", lambda temporary: temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"))
    return report


def _load_extended_concepts(paths) -> tuple[dict, list[str]]:
    source = paths.study / "reports" / REPORT / "concepts.jsonl"
    payload = json.loads((paths.study / "reports" / "entity_classification" / "summary.json").read_text(encoding="utf-8"))
    levels = list(payload["proposed_entity_levels"])
    concepts = {}
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            concepts[(row["corpus"], row["concept_id"])] = row
    if not concepts:
        raise FileNotFoundError(f"Extended concepts are missing: {source}")
    return concepts, levels


def _matched_views(paths, levels: list[str]) -> dict:
    raw = load_entity_configuration(paths.code_root)
    views = {"E0": None, "E1": dict(raw["entity_levels"]["E1"]["merge"]), "E2": {}}
    for level in levels:
        if level not in views:
            views[level] = {}
    return views


def _display_by_mention(paths, corpus: str, concepts: dict) -> dict:
    display = {}
    entity_path = paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "entities.jsonl"
    with entity_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            concept = concepts.get((corpus, str(row.get("concept_id") or "")))
            if concept:
                display[row["mention_id"]] = concept["displayed"]
    return display


def _inputs(paths, concepts: dict, levels: list[str]) -> dict:
    settings = load_settings(paths.code_root)
    tokenizer_dir = Path(str(settings.get("biomedbert_tokenizer_dir") or ""))
    raw = load_entity_configuration(paths.code_root)
    views = {"E0": None, "E1": dict(raw["entity_levels"]["E1"]["merge"]), "E2": {}}
    for level in levels:
        if level not in views:
            views[level] = {}
    summaries = {}
    for corpus in ("biored", "bc8"):
        display = {}
        entity_path = paths.prepared / corpus / CURRENT_VERSIONS[corpus] / "entities.jsonl"
        for line in entity_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            concept = concepts.get((corpus, str(row.get("concept_id") or "")))
            if concept:
                display[row["mention_id"]] = concept["displayed"]
        output = paths.inputs / corpus / INPUT_VERSION
        summaries[corpus] = build_context_tables(
            paths.prepared / corpus / CURRENT_VERSIONS[corpus],
            output,
            corpus=corpus,
            tokenizer_dir=tokenizer_dir,
            code_root=paths.code_root,
            workers=4,
            input_version=INPUT_VERSION,
            views_override=views,
            display_by_mention=display,
        )
        summaries[corpus]["window_comparison"] = _window_comparison(paths, corpus, output / "contexts.jsonl")
        summaries[corpus]["output"] = str(output)
    return summaries


def _window_comparison(paths, corpus: str, new_path: Path) -> dict:
    historical = paths.inputs / corpus / "target_anchored_v1__biomedbert_e1354b7a_proposed" / "contexts.jsonl"
    if not historical.is_file() or not new_path.is_file():
        return {"status": "historical_file_absent"}
    old = {}
    with historical.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                old[row["candidate_id"]] = json.dumps(row.get("fragments"), sort_keys=True)
    same = changed = missing = 0
    lengths = []
    over = 0
    with new_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            signature = json.dumps(row.get("fragments"), sort_keys=True)
            previous = old.get(row["candidate_id"])
            if previous is None:
                missing += 1
            elif previous == signature:
                same += 1
            else:
                changed += 1
            length = row.get("max_encoded_length")
            if isinstance(length, int):
                lengths.append(length)
                if length > 512:
                    over += 1
    lengths.sort()
    return {
        "same_source_window": same,
        "changed_source_window": changed,
        "absent_from_historical": missing,
        "over_budget": over,
        "n_lengths": len(lengths),
        "median_encoded_length": lengths[len(lengths) // 2] if lengths else None,
        "max_encoded_length": lengths[-1] if lengths else None,
        "historical_predictions_comparable": changed == 0,
    }


def _lora_modules(paths) -> dict:
    found = {}
    for name, revision in (("qwen8b", "b968826d9c46dd6066d109eabc6255188de91218"), ("qwen32b", "9216db5781bf21249d130ec9da846c4624c16137")):
        directory = paths.storage_root / "models" / {"qwen8b": "Qwen__Qwen3-8B", "qwen32b": "Qwen__Qwen3-32B"}[name] / revision
        index = directory / "model.safetensors.index.json"
        if not index.is_file():
            found[name] = {"status": "index_absent"}
            continue
        names = list(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}))
        found[name] = verify_lora_modules(names)
    return found


def _civic(paths) -> dict:
    path = paths.study / "reports" / "entity_classification" / "civic_lung_evidence.jsonl"
    if not path.is_file():
        return {"status": "absent"}
    counts = Counter()
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        counts[f"type:{row.get('evidence_type')}"] += 1
        counts[f"direction:{row.get('evidence_direction')}"] += 1
        counts[f"significance:{row.get('significance')}"] += 1
        counts["rows"] += 1
    predictive = counts.get("type:Predictive", 0)
    return {
        "rows": counts["rows"],
        "predictive": predictive,
        "by_type": {key.split(":", 1)[1]: value for key, value in counts.items() if key.startswith("type:")},
        "by_direction": {key.split(":", 1)[1]: value for key, value in counts.items() if key.startswith("direction:")},
        "by_significance": {key.split(":", 1)[1]: value for key, value in counts.items() if key.startswith("significance:")},
        "curated_statement_is_model_input": False,
        "extracted_here": False,
    }


def _human_sample(paths, destination: Path) -> dict:
    excluded = set()
    summary_path = paths.study / "reports" / "source_references" / "summary.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for block in (payload.get("samples") or {}).values():
            for corpus in block.values():
                for candidate_id in corpus.get("candidate_ids") or []:
                    parts = str(candidate_id).split(":")
                    if len(parts) > 1:
                        excluded.add(parts[1])
    documents = {}
    for line in (paths.prepared / "biored" / CURRENT_VERSIONS["biored"] / "documents.jsonl").open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            if row.get("split") == "test" and str(row["document_id"]) not in excluded:
                documents[str(row["document_id"])] = row
    positive_docs = set()
    candidates = []
    dataset = ds.dataset(paths.prepared / "biored" / CURRENT_VERSIONS["biored"] / "candidates.parquet", format="parquet")
    for batch in dataset.to_batches(columns=["candidate_id", "document_id", "split", "gold_status", "mention_ids_1", "mention_ids_2"], filter=ds.field("split") == "test"):
        for row in batch.to_pylist():
            if str(row["document_id"]) not in documents:
                continue
            if row["gold_status"] == "unique":
                positive_docs.add(str(row["document_id"]))
            candidates.append(row)
    import random
    generator = random.Random(20261125)
    doc_draw = _draw(sorted(documents), lambda document_id: "positive" if document_id in positive_docs else "unannotated", generator)
    status_by_candidate = {row["candidate_id"]: row["gold_status"] for row in candidates}
    candidate_draw = _draw(
        list(status_by_candidate),
        lambda candidate_id: "annotated" if status_by_candidate[candidate_id] == "unique" else "unannotated",
        generator,
        fraction=0.02,
    )
    by_id = {row["candidate_id"]: row for row in candidates}
    mentions = {}
    entity_path = paths.prepared / "biored" / CURRENT_VERSIONS["biored"] / "entities.jsonl"
    with entity_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("document_id")) in documents:
                mentions[row["mention_id"]] = row
    packet = destination / "annotation_packets.jsonl"
    with packet.open("w", encoding="utf-8") as handle:
        for item in doc_draw["drawn"]:
            document = documents[item["unit_id"]]
            handle.write(json.dumps({
                "packet": "source_reading",
                "estimand": "BioRED test document, after excluding documents already used in the diagnostic source review",
                "document_id": item["unit_id"],
                "text": document.get("text"),
                "gold_label_visible": False,
                "model_visible": False,
                "judgement_options": ["finding identified", "no finding identified", "unassessable"],
                "judgement": None,
                "reviewer_a": None,
                "reviewer_b": None,
                "adjudication": None,
                "inclusion_probability": item["inclusion_probability"],
                "analysis_weight": item["analysis_weight"],
            }, ensure_ascii=False) + "\n")
        for item in candidate_draw["drawn"]:
            row = by_id[item["unit_id"]]
            mention_ids = list(row["mention_ids_1"] or []) + list(row["mention_ids_2"] or [])
            marked = []
            for mention_id in mention_ids:
                mention = mentions.get(mention_id)
                if mention:
                    marked.append({
                        "mention_id": mention_id,
                        "text": mention.get("text"),
                        "char_start": mention.get("char_start"),
                        "char_end": mention.get("char_end"),
                        "entity_type": mention.get("entity_type"),
                    })
            handle.write(json.dumps({
                "packet": "candidate_context",
                "estimand": "BioRED test candidate, including unannotated pairs inside the sampled documents",
                "document_id": row["document_id"],
                "candidate_id": row["candidate_id"],
                "text": documents[str(row["document_id"])].get("text"),
                "mentions": marked,
                "gold_label_visible": False,
                "model_visible": False,
                "judgement_options": ["source states a relation", "source does not state a relation", "unassessable"],
                "judgement": None,
                "reviewer_a": None,
                "reviewer_b": None,
                "adjudication": None,
                "inclusion_probability": item["inclusion_probability"],
                "analysis_weight": item["analysis_weight"],
            }, ensure_ascii=False) + "\n")
    return {"documents": doc_draw, "candidates": candidate_draw, "answers_populated": False, "output": str(packet)}


def _draw(units: list[str], stratum_of, generator, fraction: float = 0.2) -> dict:
    grouped = defaultdict(list)
    for unit in units:
        grouped[stratum_of(unit)].append(unit)
    drawn = []
    strata = {}
    for name, members in sorted(grouped.items()):
        members = sorted(set(members))
        generator.shuffle(members)
        take = int(round(fraction * len(members)))
        probability = (take / len(members)) if members else None
        strata[name] = {"population": len(members), "drawn": take, "inclusion_probability": probability, "analysis_weight": (1 / probability) if probability else None}
        for unit in members[:take]:
            drawn.append({"unit_id": unit, "stratum": name, "inclusion_probability": probability, "analysis_weight": (1 / probability) if probability else None, "judgement": None})
    return {"fraction_requested": fraction, "strata": strata, "n_drawn": len(drawn), "drawn": drawn}


def finalise() -> int:
    """Write the preparation summary from records already on disk. Does not rebuild inputs."""
    paths = get_paths()
    destination = paths.assert_output(paths.study / "reports" / REPORT)
    destination.mkdir(parents=True, exist_ok=True)
    print("Recomputing entity transitions.", flush=True)
    concepts, levels = _load_extended_concepts(paths)
    transitions = _transitions(paths, concepts, levels)
    print("Recounting verbalised predictions.", flush=True)
    verbalisation = _verbalise(paths, destination)
    print("Indexing requests that can be built now.", flush=True)
    requests = _index_requests(destination)
    print("Listing changed source windows.", flush=True)
    windows = {corpus: _changed_windows(paths, corpus, destination) for corpus in ("biored", "bc8")}
    print("Writing annotation packets.", flush=True)
    human = _human_sample(paths, destination)
    for key in ("documents", "candidates"):
        human[key] = {name: value for name, value in human[key].items() if name != "drawn"}
    print("Writing the random-demonstration sensitivity set.", flush=True)
    sensitivity = _random_sensitivity(paths, destination)
    inventory = _inventory(paths, levels)
    supervision = _existing_supervision_summary(destination, levels)
    qwen_path = destination / "qwen_lengths.json"
    summary = {
        "proposed_entity_levels": levels,
        "transitions": transitions,
        "verbalisation": verbalisation,
        "semantic_requests": requests,
        "changed_windows": windows,
        "inputs": _merge_reports(paths),
        "inventory": inventory,
        "demonstrations": _existing_demonstration_summary(destination),
        "random_demonstration_sensitivity": sensitivity,
        "supervision": supervision,
        "lora": lora_configuration(),
        "lora_modules": _lora_modules(paths),
        "schema_conditioned": schema_conditioned_configuration(supervision["biored_training_candidates"], len(levels), len(RESEARCH_PREDICATES)),
        "civic": _civic(paths),
        "human_sample": human,
        "sentence_segmenter": _segmenter_status(),
        "measurement_models": _measurement_status(),
        "qwen_lengths": json.loads(qwen_path.read_text(encoding="utf-8")) if qwen_path.is_file() else {"status": "not_yet_written"},
        "execution": _execution_plan(inventory, windows),
        "external_model": {
            "id": "GPT-OSS-20B",
            "role": "limited check outside the Qwen family",
            "in_principal_matrix": False,
            "weights_acquired": False,
            "executed": False,
        },
        "provenance": repository_provenance(paths.code_root),
        "historical_verbalisation_retained": str(paths.study / "reports" / "entity_classification" / "verbalised_predictions.jsonl"),
        "launched": False,
        "gpu_jobs_submitted": False,
    }
    publish_file(destination / "summary.json", lambda temporary: temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"))
    print(json.dumps({
        "levels": levels,
        "training_runs": inventory["counts"]["training_runs"],
        "verbalised": verbalisation.get("verbalised"),
        "changed_windows": {key: value.get("changed") for key, value in windows.items()},
        "source_requests": requests.get("source_requests"),
        "unique_forward_requests": requests.get("unique_forward_requests"),
    }, indent=2), flush=True)
    return 0


def _merge_reports(paths) -> dict:
    reports = {}
    for corpus in ("biored", "bc8"):
        path = paths.inputs / corpus / INPUT_VERSION / "merge_report.json"
        reports[corpus] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"status": "absent"}
    return reports


def _changed_windows(paths, corpus: str, destination: Path) -> dict:
    historical_name = "target_anchored_v1__biomedbert_e1354b7a_proposed"
    historical = paths.inputs / corpus / historical_name / "contexts.jsonl"
    current = paths.inputs / corpus / INPUT_VERSION / "contexts.jsonl"
    old = {}
    with historical.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                old[row["candidate_id"]] = json.dumps(row.get("fragments"), sort_keys=True)
    out = destination / f"changed_windows_{corpus}.jsonl"
    changed = same = 0
    by_split = Counter()
    with current.open(encoding="utf-8") as incoming, out.open("w", encoding="utf-8") as handle:
        for line in incoming:
            if not line.strip():
                continue
            row = json.loads(line)
            signature = json.dumps(row.get("fragments"), sort_keys=True)
            if old.get(row["candidate_id"]) == signature:
                same += 1
                continue
            changed += 1
            by_split[str(row.get("split"))] += 1
            handle.write(json.dumps({"candidate_id": row["candidate_id"], "document_id": row.get("document_id"), "split": row.get("split")}, sort_keys=True) + "\n")
    return {"changed": changed, "unchanged": same, "changed_by_split": dict(by_split), "output": str(out)}


def _index_requests(destination: Path) -> dict:
    from schema_study.source_identification import INSTRUCTION_VERSION, build_request

    source = destination / "verbalised_predictions.jsonl"
    forward = Counter()
    source_keys = set()
    source_path = destination / "source_localisation_requests.jsonl"
    semantic_path = destination / "semantic_requests.jsonl"
    with source.open(encoding="utf-8") as incoming, source_path.open("w", encoding="utf-8") as located:
        for line in incoming:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("text") or not row.get("argument_1") or not row.get("argument_2"):
                continue
            forward[(row.get("corpus"), str(row.get("document_id")), row["proposition_id"])] += 1
            key = (row.get("corpus"), str(row.get("document_id")), row.get("candidate_id"), row["proposition_id"])
            if key in source_keys:
                continue
            source_keys.add(key)
            request = build_request(
                document_id=str(row.get("document_id")),
                document_text=row["text"],
                candidate_id=str(row.get("candidate_id")),
                argument_1=row["argument_1"],
                argument_2=row["argument_2"],
                proposition=row["text"],
                mode="relation_conditioned",
            )
            request["document_characters"] = None
            request["document_text_included"] = False
            request["proposition_id"] = row["proposition_id"]
            request["corpus"] = row.get("corpus")
            request["verbalisation_version"] = row.get("verbalisation_version")
            request["instruction_version"] = INSTRUCTION_VERSION
            request["executed"] = False
            located.write(json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n")
    with semantic_path.open("w", encoding="utf-8") as handle:
        for (corpus, document_id, proposition_id), count in forward.items():
            handle.write(json.dumps({
                "corpus": corpus,
                "document_id": document_id,
                "proposition_id": proposition_id,
                "n_observations": count,
                "direction": "forward",
                "context": "full_document",
                "verbalisation_version": VERBALISATION_VERSION,
                "executed": False,
            }, sort_keys=True) + "\n")
    unique_propositions = len({key[2] for key in forward})
    return {
        "verbalised_lines_indexed": sum(forward.values()),
        "unique_forward_requests": len(forward),
        "unique_proposition_texts": unique_propositions,
        "source_requests": len(source_keys),
        "reverse_entailment_requests_exact": None,
        "reverse_entailment_upper_bound": unique_propositions,
        "finding_level_certainty_requests_exact": None,
        "finding_level_certainty_upper_bound": unique_propositions,
        "future_counts_not_claimed": True,
        "outputs": {"source": str(source_path), "forward": str(semantic_path)},
    }


def _random_sensitivity(paths, destination: Path) -> dict:
    import hashlib

    pool = _retrieval_rows(paths, "biored", "train")
    targets = _retrieval_rows(paths, "biored", "dev")
    out = destination / "random_demonstrations_biored_dev.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for target in targets:
            seed = int(hashlib.sha256(f"20261125:{target['id']}".encode("utf-8")).hexdigest()[:8], 16)
            chosen = random_demonstrations(target["document_id"], pool, 5, seed)
            handle.write(json.dumps({"candidate_id": target["id"], "demonstrations": chosen, "seed_scope": "biored_dev_20261125"}, sort_keys=True) + "\n")
    return {
        "corpus": "biored",
        "split": "dev",
        "targets": len(targets),
        "demonstrations": 5,
        "in_principal_matrix": False,
        "retrieval": "random_without_replacement_seeded_by_candidate_exclude_target_document_force_negative",
        "output": str(out),
    }


def _existing_demonstration_summary(destination: Path) -> dict:
    summary = {}
    for corpus in ("biored", "bc8", "drugprot"):
        path = destination / f"few_shot_{corpus}.jsonl"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                summary[corpus] = sum(1 for line in handle if line.strip())
    return summary


def _existing_supervision_summary(destination: Path, levels: list[str]) -> dict:
    def lines(name: str) -> int:
        path = destination / name
        if not path.is_file():
            return 0
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    biored = lines("sft_biored.jsonl")
    drugprot = lines("sft_drugprot.jsonl")
    return {
        "biored_example_rows": biored,
        "drugprot_example_rows": drugprot,
        "biored_training_candidates": biored // max(1, len(levels) * len(RESEARCH_PREDICATES)),
        "loss": "assistant_completion_tokens_only",
        "trained_here": False,
    }


def _execution_plan(inventory: dict, windows: dict) -> dict:
    counts = inventory["counts"]
    return {
        "training_runs": counts.get("training_runs"),
        "training_runs_by_model": counts.get("training_runs_by_model"),
        "bc8_training_runs": counts.get("bc8_training_runs"),
        "units": counts.get("units"),
        "reuse": "Historical E0–E2 predictions remain valid for the frozen input version. The matched E0–E7 collection uses a new source window for the candidates listed in the changed-window files, so those historical predictions are not the matched comparison.",
        "changed_windows": {corpus: row.get("changed") for corpus, row in windows.items()},
        "new_training": "All 324 training runs require new execution. E3–E7 were not trained, and a changed training window means an E0–E2 run is not the matched condition.",
        "dependencies": [
            "Matched inputs before training and direct inference",
            "Training checkpoints before supervised evaluation",
            "Direct predictions before deterministic abstraction",
            "Verbalised assertions before forward support measurement",
            "Source findings before finding-level certainty and reverse entailment",
            "No GPU job starts while an input or measurement definition has failed",
        ],
        "historical_gpu_seconds_accounted": 392433,
        "historical_gpu_time_is_not_a_forecast": True,
        "gpu_submitted_in_this_preparation": False,
        "duplicate_writer_safeguard": "A shard is complete only when its ledger and output agree. A second writer does not append to an existing shard.",
        "resume": "A Slurm array resumes from completed shard indices and does not rewrite them.",
    }


def _segmenter_status() -> dict:
    comparison = get_paths().study / "reports" / REPORT / "sentence_comparison.json"
    if comparison.is_file():
        payload = json.loads(comparison.read_text(encoding="utf-8"))
        payload["status"] = "compared_on_biored_development"
        payload["hf_hpc_has_spacy"] = False
        return payload
    return {
        "status": "comparison_not_written",
        "baseline_retained": "abbreviation_decimal_v1",
        "agreement_is_not_accuracy": True,
        "hf_hpc_has_spacy": False,
    }


def _measurement_status() -> dict:
    """Record hf-hpc without importing PyTorch in the analysis interpreter."""
    return {
        "environment": "hf-hpc",
        "activation": "source /home/b5ac/freddieyu.b5ac/miniforge3/etc/profile.d/conda.sh && conda activate hf-hpc",
        "prefix": "/home/b5ac/freddieyu.b5ac/miniforge3/envs/hf-hpc",
        "python": "3.10.19",
        "torch": "2.9.1",
        "transformers": "4.57.5",
        "cuda_available_on_login_node": False,
        "shared_environment_modified": False,
        "certainty_checkpoint": "pedropei/sentence-level-certainty",
        "verifier_checkpoint": "MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-wanli",
        "executed": False,
    }
