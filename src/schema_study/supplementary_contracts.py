"""Corrected evidence, qualified, and legal-label contracts.

These prompts replace the label-only instruction for the named supplementary
conditions. A quotation locates source text. It is not semantic support.
"""

from __future__ import annotations

import json
from pathlib import Path

from schema_study.learning_conditions import HISTORICAL_PREDICATES
from schema_study.paths import get_paths
from schema_study.qwen_workload import CONTEXTS
from schema_study.schemas import default_schema_path, load_schema

STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
SENTENCES = STORAGE / "studies/schema_granularity_v1/reports/source_references/sentences"
EVALUATION_SPLIT = {"biored": "test", "bc8": "bc8", "drugprot": "eval"}
QUALIFIER_FIELDS = (
    "polarity",
    "epistemic_commitment",
    "conditionality",
    "disease_context",
    "experimental_context",
    "assessability",
)
UNKNOWN = "unknown"


def legal_names(task: dict) -> list[str]:
    dataset = task.get("dataset")
    schema = load_schema(default_schema_path(get_paths().code_root, "drugprot" if dataset == "drugprot" else "biored"))
    level = task.get("schema_predicate") or HISTORICAL_PREDICATES.get(task.get("predicate"), task.get("predicate"))
    return list(schema.positive_display_names(level))


def legal_label_schema(task: dict) -> dict:
    """Constrain the object to the predicate inventory. An empty list remains legal."""
    names = legal_names(task)
    # vLLM 0.33 guidance rejects uniqueItems. Uniqueness is checked after decoding.
    labels = {"type": "array", "items": {"type": "string", "enum": names}}
    if task.get("dataset") != "drugprot":
        labels["maxItems"] = 1
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"candidate_id": {"type": "string"}, "labels": labels},
        "required": ["candidate_id", "labels"],
    }


def _sentence_index(dataset: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    with (SENTENCES / f"{dataset}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped.setdefault(row["document_id"], []).append(row)
    return grouped


def _fragments(dataset: str) -> dict[str, list[dict]]:
    found = {}
    with CONTEXTS[dataset].open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != EVALUATION_SPLIT[dataset]:
                continue
            found[row["candidate_id"]] = row.get("fragments") or []
    return found


def _overlapping(sentences: list[dict], fragments: list[dict]) -> list[dict]:
    chosen = []
    for sentence in sentences:
        start = int(sentence["char_start"])
        end = int(sentence["char_end"])
        if any(end > int(fragment["char_start"]) and start < int(fragment["char_end"]) for fragment in fragments):
            chosen.append(sentence)
    chosen.sort(key=lambda row: (int(row["char_start"]), row["sentence_id"]))
    return chosen


def _shown_sentences(sentences: list[dict]) -> str:
    return "\n".join(f"{row['sentence_id']}\t{row['text']}" for row in sentences)


def evidence_schema(task: dict) -> dict:
    schema = legal_label_schema(task)
    schema["properties"]["evidence"] = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"sentence_id": {"type": "string"}, "quotation": {"type": "string"}},
            "required": ["sentence_id", "quotation"],
        },
    }
    schema["required"] = ["candidate_id", "labels", "evidence"]
    return schema


def qualified_schema(task: dict) -> dict:
    schema = evidence_schema(task)
    for field in QUALIFIER_FIELDS:
        schema["properties"][field] = {"type": "string"}
    schema["properties"]["unresolved"] = {"type": "array", "items": {"type": "string"}}
    schema["required"] = ["candidate_id", "labels", "evidence", *QUALIFIER_FIELDS, "unresolved"]
    return schema


def _base_instruction(task: dict, request: dict, sentences: list[dict]) -> str:
    names = "\n".join(f"- {name}" for name in legal_names(task))
    cardinality = (
        "DrugProt may contain more than one positive label."
        if task.get("dataset") == "drugprot"
        else "BioRED allows at most one positive label, or an empty list."
    )
    return (
        "You extract one biomedical relation candidate from the source sentences.\n"
        f"Candidate id: {request['candidate_id']}\n"
        f"Dataset: {task.get('dataset')}\n"
        f"{cardinality}\n"
        "Positive labels:\n"
        f"{names}\n"
        "Use only those labels. An empty labels list means no positive relation is stated.\n"
        "Do not write No_Relation. Do not use a relation known only from outside these sentences.\n"
        "Source sentences, with their authoritative identifiers:\n"
        f"{_shown_sentences(sentences)}\n"
    )


_SOURCE_CACHE: dict[str, tuple] = {}


def apply_contract(task: dict, requests: list[dict]) -> list[dict]:
    """Attach the corrected prompt and schema. The original zero-shot shard is not rewritten."""
    family = task.get("family")
    if family not in {"evidence", "qualified"}:
        return requests
    dataset = task.get("dataset")
    if dataset not in _SOURCE_CACHE:
        _SOURCE_CACHE[dataset] = (_sentence_index(dataset), _fragments(dataset))
    sentences, fragments = _SOURCE_CACHE[dataset]
    schema = evidence_schema(task) if family == "evidence" else qualified_schema(task)
    prepared = []
    for request in requests:
        rows = _overlapping(sentences.get(request.get("document_id"), []), fragments.get(request["candidate_id"], []))
        if not rows:
            prepared.append({**request, "contract_ready": False, "contract_reason": "no_overlapping_source_sentence"})
            continue
        instruction = _base_instruction(task, request, rows)
        if family == "evidence":
            instruction += (
                "Return one JSON object with candidate_id, labels, and evidence.\n"
                "Each evidence item has a sentence_id from the list above and a quotation copied exactly from that sentence.\n"
                "Use more than one item when the support is repeated or discontinuous.\n"
                "A positive label requires at least one evidence item. An empty label list uses an empty evidence list.\n"
                "A quotation only locates the source. It does not by itself establish the relation.\n"
            )
        else:
            instruction += (
                "Return one JSON object with candidate_id, labels, evidence, polarity, epistemic_commitment, "
                "conditionality, disease_context, experimental_context, assessability, and unresolved.\n"
                "Evidence items use sentence_id and an exact quotation, as in an evidence-grounded extraction.\n"
                "polarity is positive, negative, or unknown. It is not a certainty score.\n"
                "epistemic_commitment is asserted, hedged, or unknown.\n"
                "conditionality is unconditional, conditional, or unknown.\n"
                "Use unknown or an unresolved entry when the source does not state a field.\n"
                "Do not invent a qualifier that the sentences do not support.\n"
                "A labels array alone is not a complete qualified assertion.\n"
            )
        prepared.append({
            **request,
            "prompt": instruction,
            "rendered": instruction,
            "schema_json": schema,
            "source_sentences": [
                {"sentence_id": row["sentence_id"], "text": row["text"], "char_start": row["char_start"], "char_end": row["char_end"]}
                for row in rows
            ],
            "contract_ready": True,
            "prompt_contract": "evidence_grounded" if family == "evidence" else "qualified_assertion",
        })
    return prepared


def recover_spans(evidence: list[dict], sentences: list[dict]) -> dict:
    """Map an exact quotation back to document character offsets. An unmatched quotation stays unresolved."""
    by_id = {row["sentence_id"]: row for row in sentences}
    resolved = []
    unresolved = []
    for item in evidence:
        sentence = by_id.get(item.get("sentence_id"))
        quotation = item.get("quotation") or ""
        if sentence is None or not quotation or quotation not in sentence["text"]:
            unresolved.append(item)
            continue
        offset = sentence["text"].index(quotation)
        start = int(sentence["char_start"]) + offset
        resolved.append({
            "sentence_id": sentence["sentence_id"],
            "quotation": quotation,
            "char_start": start,
            "char_end": start + len(quotation),
            "quotation_is_semantic_support": False,
        })
    repeated = len(resolved) != len({(row["sentence_id"], row["quotation"]) for row in resolved})
    return {"resolved": resolved, "unresolved": unresolved, "repeated_quotation": repeated, "complete": not unresolved}


def interpret_contract(task: dict, request: dict, raw: str | None) -> dict:
    """A label-only object is not evidence-grounded or a qualified assertion."""
    family = task.get("family")
    try:
        payload = json.loads(raw or "")
    except json.JSONDecodeError:
        return {"complete": False, "reason": "invalid_json", "labels": None}
    if not isinstance(payload, dict):
        return {"complete": False, "reason": "invalid_json", "labels": None}
    labels = payload.get("labels")
    if not isinstance(labels, list):
        return {"complete": False, "reason": "missing_labels", "labels": None}
    legal = set(legal_names(task))
    illegal = [label for label in labels if label not in legal]
    if len(labels) != len(set(labels)):
        return {"complete": False, "reason": "duplicate_label", "labels": None, "illegal_labels": illegal}
    if family == "constrained":
        return {"complete": not illegal, "labels": [] if illegal else labels, "illegal_labels": illegal, "reason": "illegal_label" if illegal else "legal_label_schema"}
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        return {"complete": False, "reason": "label_only_response", "labels": None, "illegal_labels": illegal}
    spans = recover_spans(evidence, request.get("source_sentences") or [])
    if illegal or spans["unresolved"] or (labels and not spans["resolved"]):
        return {"complete": False, "reason": "unresolved_or_illegal_evidence", "labels": None, "illegal_labels": illegal, "evidence": spans}
    if family == "qualified":
        missing = [field for field in QUALIFIER_FIELDS if field not in payload]
        if missing:
            return {"complete": False, "reason": "label_only_response", "labels": None, "evidence": spans}
    return {
        "complete": True,
        "labels": labels,
        "illegal_labels": [],
        "evidence": spans,
        "qualifiers": {field: payload.get(field, UNKNOWN) for field in QUALIFIER_FIELDS} if family == "qualified" else {},
        "unresolved": payload.get("unresolved") or [],
        "quotation_is_not_semantic_support": True,
        "reason": "evidence_located" if family == "evidence" else "qualified_assertion",
    }
