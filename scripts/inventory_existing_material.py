"""Read-only inventory for the stage-00 granularity assessment.

The script counts prepared records and checks formal outputs. It does not
train, generate, or rewrite datasets.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.dataset as ds

STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
PREPARED = {
    "biored": STORAGE / "data/prepared/biored/biored_candidates_v2",
    "bc8": STORAGE / "data/prepared/bc8/bc8_candidates_v2",
    "drugprot": STORAGE / "data/prepared/drugprot/drugprot_candidates_v1",
}
FORMAL = STORAGE / "studies/schema_granularity_v1/runs/formal/formal_20261115"


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _dataset_inventory(name: str, root: Path) -> dict:
    summary = json.loads((root / "data_summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    documents = defaultdict(int)
    document_ids = defaultdict(set)
    for row in _jsonl(root / "documents.jsonl"):
        documents[row["split"]] += 1
        document_ids[row["split"]].add(str(row["document_id"]))
    split_of = {}
    for split, ids in document_ids.items():
        for document_id in ids:
            split_of[document_id] = split
    entities = Counter()
    concepts = defaultdict(set)
    types = Counter()
    offset_missing = 0
    identifier_missing = 0
    for row in _jsonl(root / "entities.jsonl"):
        split = split_of.get(str(row["document_id"]), "unassigned")
        entities[split] += 1
        concepts[split].add((str(row["document_id"]), str(row.get("concept_id") or "")))
        types[row.get("entity_type") or "missing"] += 1
        if row.get("char_start") is None or row.get("char_end") is None:
            offset_missing += 1
        if not row.get("concept_id") and not row.get("raw_identifier"):
            identifier_missing += 1
    relations = Counter()
    relation_labels = Counter()
    relation_fields = None
    for row in _jsonl(root / "relations.jsonl"):
        relations[row["split"]] += 1
        relation_labels[row["label"]] += 1
        if relation_fields is None:
            relation_fields = sorted(row)
    exclusions = Counter()
    if (root / "exclusions.tsv").is_file():
        lines = (root / "exclusions.tsv").read_text(encoding="utf-8").splitlines()
        header = lines[0].split("\t")
        reason_at = header.index("reason")
        for line in lines[1:]:
            if line.strip():
                exclusions[line.split("\t")[reason_at]] += 1
    candidates = Counter()
    positive_candidates = Counter()
    positive_labels = Counter()
    table = ds.dataset(root / "candidates.parquet", format="parquet")
    for batch in table.to_batches(columns=["split", "gold_labels"]):
        for row in batch.to_pylist():
            candidates[row["split"]] += 1
            labels = list(row["gold_labels"] or [])
            if labels:
                positive_candidates[row["split"]] += 1
                positive_labels[row["split"]] += len(labels)
    return {
        "data_version": summary["data_version"],
        "release_id": summary.get("release_id"),
        "sources": provenance.get("sources", []),
        "source_url": provenance.get("source_url") or provenance.get("url"),
        "n_documents_by_split": dict(documents),
        "document_ids": {split: sorted(ids) for split, ids in document_ids.items()},
        "n_entity_mentions_by_split": dict(entities),
        "n_concepts_by_split": {split: len(ids) for split, ids in concepts.items()},
        "entity_types": dict(types),
        "offset_missing": offset_missing,
        "identifier_missing": identifier_missing,
        "n_annotated_relations_by_split": dict(relations),
        "relation_fields": relation_fields,
        "n_candidates_by_split": dict(candidates),
        "n_positive_candidates_by_split": dict(positive_candidates),
        "n_positive_gold_labels_by_split": dict(positive_labels),
        "exclusions": dict(exclusions),
        "summary_n_documents": summary.get("n_documents"),
        "summary_n_candidates": summary.get("n_candidates"),
        "summary_n_relations": summary.get("n_relations"),
        "summary_n_entities": summary.get("n_entities"),
        "formally_trainable": summary.get("formally_trainable"),
        "held_out_first_contact": summary.get("held_out_first_contact"),
        "prior_tuning_use_confirmed": summary.get("prior_tuning_use_confirmed"),
    }


def _overlap(inventories: dict) -> dict:
    sets = {}
    for name, item in inventories.items():
        ids = set()
        for split_ids in item["document_ids"].values():
            ids.update(split_ids)
        sets[name] = ids
    names = list(sets)
    pairs = {}
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            shared = sets[names[left]] & sets[names[right]]
            pairs[f"{names[left]}__{names[right]}"] = len(shared)
    return pairs


def _formal_outputs() -> dict:
    manifest_path = FORMAL / "execution_manifest.tsv"
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    rows = [dict(zip(header, line.split("\t"))) for line in lines[1:]]
    summary = json.loads((FORMAL / "prepare/length_summary.json").read_text(encoding="utf-8"))
    shards = {
        (item["dataset"], item["entity_level"], item["predicate_level"], int(item["shard_index"])): item
        for item in summary["shards"]
    }
    checked = []
    for row in rows:
        if row["kind"] == "train":
            path = FORMAL / "train" / row["dataset"] / row["entity_level"] / row["predicate_level"] / f"seed{row['seed']}" / "prediction_complete.json"
            predictions = path.parent / "predictions.jsonl"
            artefact = path.is_file() and predictions.is_file() and predictions.stat().st_size > 0
            checked.append({**row, "artefact_complete": artefact})
            continue
        shard = shards[(row["dataset"], row["entity_level"], row["predicate_level"], int(row["shard"]))]
        attempts = FORMAL / "generation/outputs" / row["model"] / row["dataset"] / row["entity_level"] / row["predicate_level"] / f"shard_{int(row['shard']):05d}" / "attempts.jsonl"
        adopted = 0
        if attempts.is_file():
            with attempts.open(encoding="utf-8") as handle:
                for line in handle:
                    if '"adoption": "adopt"' in line or '"adoption":"adopt"' in line:
                        adopted += 1
        checked.append({**row, "artefact_complete": adopted >= int(shard["requests"]), "adopted": adopted, "requests": int(shard["requests"])})
    def key(row):
        return (row["kind"], row["dataset"], row["entity_level"], row["predicate_level"], row.get("model") or "biomedbert", row.get("seed") or "", row.get("shard") or "")
    return {
        "manifest_status": dict(Counter(row["status"] for row in rows)),
        "verified": [
            {
                "kind": row["kind"],
                "dataset": row["dataset"],
                "entity_level": row["entity_level"],
                "predicate_level": row["predicate_level"],
                "model": row.get("model") or "biomedbert",
                "seed": row.get("seed") or "",
                "shard": row.get("shard") or "",
                "manifest_status": row["status"],
                "artefact_complete": row["artefact_complete"],
                "adopted": row.get("adopted"),
                "requests": row.get("requests"),
            }
            for row in sorted(checked, key=key)
        ],
    }


def main() -> None:
    inventories = {name: _dataset_inventory(name, root) for name, root in PREPARED.items()}
    payload = {
        "datasets": {
            name: {key: value for key, value in item.items() if key != "document_ids"}
            for name, item in inventories.items()
        },
        "document_overlap": _overlap(inventories),
        "formal": _formal_outputs(),
        "notes": {
            "drugprot_identifier_missing": "DrugProt entity rows have mention identifiers, character offsets, and mention text. They do not have concept_id or raw_identifier fields, so identifier_missing equals the mention count.",
            "drugprot_concept_count": "n_concepts_by_split collapses empty concept identifiers to one value per document and is not a normalised concept count.",
            "generation_completion": "A generation shard is artefact_complete only when the number of adopted attempt lines is at least the frozen request count.",
        },
    }
    destination = Path(__file__).resolve().parents[1] / "docs/existing_material_assessment/inventory.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
