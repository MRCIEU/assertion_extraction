"""Abstract lookup and entity matching / span localisation."""

from __future__ import annotations

import json
import re
from typing import Any

from .config import EVIDENCE_JSON as CIVIC_EVIDENCE_JSON


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def surface_variants(term: str) -> list[str]:
    base = term.strip()
    if not base:
        return []
    candidates = {base, base.lower(), base.upper(), base.title()}
    if "-" in base:
        candidates.add(base.replace("-", " "))
        candidates.add(base.replace("-", ""))
    if " " in base:
        candidates.add(base.replace(" ", "-"))
        candidates.add(base.replace(" ", ""))
    for alias in re.findall(r"\(([^)]+)\)", base):
        candidates.add(alias.strip())
    return sorted(candidates, key=len, reverse=True)


def build_abstract_lookup() -> dict[str, str]:
    records = json.loads(CIVIC_EVIDENCE_JSON.read_text(encoding="utf-8"))
    lookup: dict[str, str] = {}
    for item in records:
        source = item.get("source") or {}
        pmid = str(source.get("citationId") or "")
        abstract = source.get("abstract") or ""
        if pmid and abstract and pmid not in lookup:
            lookup[pmid] = abstract
    return lookup


def entity_in_text(text: str, entity: str, entity_type: str) -> bool:
    if not text or not entity:
        return False
    norm = normalize_text(text)
    for variant in surface_variants(entity):
        if len(variant) >= 2 and variant in norm:
            return True
    if entity_type == "variant" and " " in entity:
        tokens = [t for t in re.split(r"[\s\-/]+", entity) if len(t) >= 3]
        if sum(1 for t in tokens if normalize_text(t) in norm) >= min(2, len(tokens)):
            return True
    return False


def locate_span(text: str, entity: str) -> dict[str, Any]:
    """Return first character span for entity using case-insensitive search."""
    if not text or not entity:
        return {"start": None, "end": None, "matched_text": None, "status": "missing_entity"}

    for variant in surface_variants(entity):
        if len(variant) < 2:
            continue
        pattern = re.compile(re.escape(variant), re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return {
                "start": match.start(),
                "end": match.end(),
                "matched_text": text[match.start() : match.end()],
                "status": "found",
            }

    return {"start": None, "end": None, "matched_text": None, "status": "not_found"}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def sentence_index_for_span(text: str, start: int, end: int) -> int | None:
    if start is None or end is None:
        return None
    cursor = 0
    for idx, sentence in enumerate(split_sentences(text)):
        sent_start = text.find(sentence, cursor)
        if sent_start == -1:
            continue
        sent_end = sent_start + len(sentence)
        cursor = sent_end
        if start >= sent_start and end <= sent_end:
            return idx
    return None


def difficulty_features(text: str, head_span: dict, tail_span: dict) -> dict[str, Any]:
    h_idx = sentence_index_for_span(text, head_span["start"], head_span["end"])
    t_idx = sentence_index_for_span(text, tail_span["start"], tail_span["end"])
    if h_idx is None or t_idx is None:
        return {
            "head_sentence_idx": h_idx,
            "tail_sentence_idx": t_idx,
            "sentence_distance": None,
            "co_sentence": None,
            "difficulty_status": "unresolved",
        }
    return {
        "head_sentence_idx": h_idx,
        "tail_sentence_idx": t_idx,
        "sentence_distance": abs(h_idx - t_idx),
        "co_sentence": h_idx == t_idx,
        "difficulty_status": "ok",
    }
