"""Text matching between CIViC entity strings and PubTator3 annotations."""

from __future__ import annotations

import re


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


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def entities_match(civic_entity: str, pubtator_text: str, civic_type: str = "") -> bool:
    """Return True if CIViC entity string matches a PubTator3 annotation surface form."""
    if not civic_entity or not pubtator_text:
        return False
    civic_norm = normalize_text(civic_entity)
    pt_norm = normalize_text(pubtator_text)
    if civic_norm == pt_norm:
        return True
    if len(civic_norm) >= 3 and civic_norm in pt_norm:
        return True
    if len(pt_norm) >= 3 and pt_norm in civic_norm:
        return True
    for variant in surface_variants(civic_entity):
        v = variant.lower()
        if len(v) >= 2 and v in pt_norm:
            return True
    if civic_type == "gene" and " " in civic_entity:
        tokens = [t for t in re.split(r"[\s\-/]+", civic_entity) if len(t) >= 3]
        if tokens and sum(1 for t in tokens if normalize_text(t) in pt_norm) >= min(2, len(tokens)):
            return True
    if civic_type == "gene" and " " in pubtator_text:
        tokens = [t for t in re.split(r"[\s\-/]+", pubtator_text) if len(t) >= 3]
        if tokens and sum(1 for t in tokens if normalize_text(t) in civic_norm) >= min(2, len(tokens)):
            return True
    return False
