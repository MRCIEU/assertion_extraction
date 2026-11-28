"""Training-view records for schema, curriculum and held-out comparisons.

These functions build the records the production trainer consumes. They do not
load a model and they do not tokenise a corpus.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from schema_study.formal_methods import nested_document_subsets, schema_view
from schema_study.learning_conditions import HISTORICAL_PREDICATES, RESEARCH_PREDICATES

STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
PREPARATION = STORAGE / "studies/schema_granularity_v1/reports/experimental_preparation"
INPUTS = {
    "sft_biored": PREPARATION / "sft_biored.jsonl",
    "sft_drugprot": PREPARATION / "sft_drugprot.jsonl",
    "few_shot_biored": PREPARATION / "few_shot_biored.jsonl",
    "few_shot_bc8": PREPARATION / "few_shot_bc8.jsonl",
    "few_shot_drugprot": PREPARATION / "few_shot_drugprot.jsonl",
    "contexts_biored": STORAGE / "data/inputs/biored/target_anchored_v2__biomedbert_e1354b7a_matched_e0_e7/contexts.jsonl",
    "contexts_bc8": STORAGE / "data/inputs/bc8/target_anchored_v2__biomedbert_e1354b7a_matched_e0_e7/contexts.jsonl",
    "contexts_drugprot": STORAGE / "data/inputs/drugprot/target_anchored_v1__biomedbert_e1354b7a/contexts.jsonl",
    "workload_biored": PREPARATION / "qwen_workload_biored.json",
}
_FINGERPRINTS: dict[str, str | None] = {}


def content_fingerprint(path: Path) -> str | None:
    """SHA-256 of the file bytes. Missing files stay missing rather than being invented."""
    key = str(path)
    if key in _FINGERPRINTS:
        return _FINGERPRINTS[key]
    if not path.is_file():
        _FINGERPRINTS[key] = None
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    _FINGERPRINTS[key] = digest.hexdigest()
    return _FINGERPRINTS[key]


def input_reference(dataset: str, kind: str) -> dict:
    if kind == "sft":
        key = f"sft_{dataset}"
    elif kind == "few_shot":
        key = f"few_shot_{dataset}"
    elif kind == "contexts":
        key = f"contexts_{dataset}"
    else:
        raise KeyError(kind)
    path = INPUTS[key]
    return {"path": str(path), "fingerprint": content_fingerprint(path), "kind": kind, "dataset": dataset}


def view_names(dataset: str) -> list[str]:
    entities = ["native"] if dataset == "drugprot" else [f"E{level}" for level in range(8)]
    return [f"{entity}/{predicate}" for entity in entities for predicate in RESEARCH_PREDICATES]


def one_view_records(candidates: list[dict], views: list[str], epoch: int, seed: int) -> list[dict]:
    """One deterministic view of each candidate. A second view is not an extra finding."""
    records = []
    for candidate in candidates:
        view = schema_view(candidate["candidate_id"], epoch, views, seed)
        records.append({**candidate, "view": view, "epoch": epoch, "exposure": "resource_efficient_one_view"})
    return records


def all_view_records(candidates: list[dict], views: list[str]) -> list[dict]:
    """Every requested representation of each candidate, under the exposure-controlled definition."""
    records = []
    for candidate in candidates:
        for view in views:
            records.append({**candidate, "view": view, "epoch": 0, "exposure": "exposure_controlled_all_views"})
    return records


def curriculum_records(order: str, examples_by_predicate: dict[str, list]) -> list[dict]:
    """Materialise the curriculum. The mixed order is an interleave, not another block permutation."""
    from schema_study.production_plan import curriculum_sequence

    rows = []
    for position, (predicate, example, phase) in enumerate(curriculum_sequence(order, examples_by_predicate)):
        record = {"position": position, "predicate": predicate, "phase": phase, "order": order}
        if isinstance(example, dict):
            record.update(example)
        else:
            record["example_id"] = example
        rows.append(record)
    finals = [row for row in rows if row["phase"] == "final_native"]
    if not finals or any(row["predicate"] != "P3" for row in finals):
        raise ValueError("The curriculum must finish on the native P3 block.")
    return rows


def unseen_split(withheld: str, examples: list[dict]) -> dict:
    """Hold out one representation. The training rows must still contain the other agreed views."""
    from schema_study.production_plan import unseen_training_views

    views = unseen_training_views(withheld)
    if withheld.startswith("E"):
        train = [row for row in examples if row["entity"] != withheld]
        evaluation = [row for row in examples if row["entity"] == withheld]
        present = {row["entity"] for row in train}
    else:
        train = [row for row in examples if row["predicate"] != withheld]
        evaluation = [row for row in examples if row["predicate"] == withheld]
        present = {row["predicate"] for row in train}
    if any(_withheld_in_row(withheld, row) for row in train):
        raise ValueError("A withheld view entered a supervised record.")
    if not set(views).issubset(present):
        raise ValueError("The training inventory does not contain the remaining views.")
    if not evaluation:
        raise ValueError("The withheld representation has no evaluation request.")
    return {
        "withheld": withheld,
        "training_views": views,
        "train": train,
        "evaluation": evaluation,
        "withheld_absent_from_training": True,
    }


def _withheld_in_row(withheld: str, row: dict) -> bool:
    return row.get("entity") == withheld or row.get("predicate") == withheld or row.get("view") == withheld


def subset_document_ids(document_ids: list[str], fraction: int, seed: int) -> list[str]:
    """The nested document subset. Ten percent remains inside the larger fractions."""
    table = nested_document_subsets(document_ids, seed)
    if str(fraction) not in table:
        raise KeyError(fraction)
    return table[str(fraction)]


def negative_population(rows: list[dict], strategy: str, seed: int) -> dict:
    """Keep every positive row. Negatives are either a seeded sample or same-document hard negatives."""
    if strategy not in {"random_negative", "hard_negative"}:
        raise ValueError(strategy)
    positives = [row for row in rows if row.get("positive")]
    negatives = [row for row in rows if not row.get("positive")]
    if strategy == "random_negative":
        generator = random.Random(seed)
        chosen = list(negatives)
        generator.shuffle(chosen)
        chosen = chosen[: len(positives)]
    else:
        positive_documents = {row["document_id"] for row in positives}
        hard = [row for row in negatives if row.get("document_id") in positive_documents]
        generator = random.Random(seed)
        if len(hard) > len(positives):
            generator.shuffle(hard)
            hard = hard[: len(positives)]
        chosen = hard
        if len(chosen) < len(positives):
            rest = [row for row in negatives if row not in chosen]
            generator.shuffle(rest)
            chosen.extend(rest[: len(positives) - len(chosen)])
    selected = positives + chosen
    return {
        "strategy": strategy,
        "seed": seed,
        "positive_ids": [row["candidate_id"] for row in positives],
        "negative_ids": [row["candidate_id"] for row in chosen],
        "rows": selected,
    }


def class_weight_from_counts(counts: list[int]) -> list[float]:
    """Inverse-frequency weights, normalised so that they average to one."""
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("Class weights require a positive count for every class in the training population.")
    total = float(sum(counts))
    scale = total / len(counts)
    return [scale / count for count in counts]


def head_sizes(schema, corpus: str) -> dict[str, int]:
    """Head width comes from the schema, not from the predicate index."""
    sizes = {}
    multi = schema.label_cardinality() == "multi" or corpus == "drugprot"
    for research, historical in HISTORICAL_PREDICATES.items():
        positive = len(schema.level_labels(historical))
        sizes[research] = positive if multi else positive + 1
    if list(sizes) != list(RESEARCH_PREDICATES):
        raise ValueError("Head sizes must cover P0, P1, P2 and P3.")
    return sizes


def single_label_target(schema, research_predicate: str, native_ids: list[str]) -> int:
    """Map a native BioRED annotation onto one class, with the final index reserved for no positive label."""
    historical = HISTORICAL_PREDICATES[research_predicate]
    labels = schema.level_labels(historical)
    if not native_ids:
        return len(labels)
    native = native_ids[0]
    for position, label in enumerate(labels):
        if native == label.stable_id or native in label.members:
            return position
    return len(labels)


def multi_hot_target(schema, research_predicate: str, native_ids: list[str]) -> list[int]:
    """DrugProt keeps a multi-hot vector. An empty vector is the no-positive outcome, not an extra class."""
    historical = HISTORICAL_PREDICATES[research_predicate]
    labels = schema.level_labels(historical)
    vector = [0] * len(labels)
    index = {}
    for position, label in enumerate(labels):
        index[label.stable_id] = position
        for member in label.members:
            index.setdefault(member, position)
    for native in native_ids:
        if native not in index:
            raise KeyError(native)
        vector[index[native]] = 1
    return vector


def parent_child_edges(schema) -> list[dict]:
    """Positive DrugProt parent labels and the native labels they contain."""
    native = [label.stable_id for label in schema.level_labels("P4")]
    index = {identifier: position for position, identifier in enumerate(native)}
    edges = []
    for research, historical in (("P0", "P1"), ("P1", "P2"), ("P2", "P3")):
        for parent_index, label in enumerate(schema.level_labels(historical)):
            for member in label.members:
                if member in index:
                    edges.append({
                        "parent_level": research,
                        "parent_index": parent_index,
                        "child_level": "P3",
                        "child_index": index[member],
                        "parent_id": label.stable_id,
                        "child_id": member,
                    })
    return edges


def exposure_summary(records: list[dict]) -> dict:
    views = {}
    candidates = set()
    for record in records:
        views[record["view"]] = views.get(record["view"], 0) + 1
        candidates.add(record["candidate_id"])
    return {"records": len(records), "candidates": len(candidates), "views": views}


def record_fingerprint(records: list[dict]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
