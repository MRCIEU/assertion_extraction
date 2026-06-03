"""Entity mention matching for abstract–assertion alignment checks."""

from __future__ import annotations

import re
from typing import Iterable


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _variants_for_term(term: str) -> list[str]:
    """Generate simple surface-form variants for transparent matching."""
    base = term.strip()
    if not base:
        return []

    candidates = {base, base.lower(), base.upper()}
    candidates.add(base.title())

    # Hyphen/space swaps common in drug and gene names.
    if "-" in base:
        candidates.add(base.replace("-", " "))
        candidates.add(base.replace("-", ""))
    if " " in base:
        candidates.add(base.replace(" ", "-"))
        candidates.add(base.replace(" ", ""))

    # Parenthetical aliases, e.g. "Non-Small Cell Lung Cancer (NSCLC)".
    paren = re.findall(r"\(([^)]+)\)", base)
    for alias in paren:
        candidates.add(alias.strip())

    return sorted({_normalize(c) for c in candidates if c}, key=len, reverse=True)


def entity_mentioned(abstract: str, entity_name: str, entity_type: str) -> bool:
    """Return True if entity_name (or simple variants) appears in abstract."""
    if not abstract or not entity_name:
        return False

    abstract_norm = _normalize(abstract)
    for variant in _variants_for_term(entity_name):
        if len(variant) < 2:
            continue
        if variant in abstract_norm:
            return True

    # For variants, also try gene token + variant token co-occurrence when both short.
    if entity_type == "variant" and " " in entity_name:
        tokens = [t for t in re.split(r"[\s\-/]+", entity_name) if len(t) >= 3]
        if len(tokens) >= 2:
            hits = sum(1 for token in tokens if _normalize(token) in abstract_norm)
            if hits >= min(2, len(tokens)):
                return True

    return False


def check_alignment(
    abstract: str,
    head_entity: str,
    head_type: str,
    tail_entity: str,
    tail_type: str,
) -> dict[str, bool | str]:
    head_hit = entity_mentioned(abstract, head_entity, head_type)
    tail_hit = entity_mentioned(abstract, tail_entity, tail_type)

    if head_hit and tail_hit:
        status = "both_present"
    elif not head_hit and not tail_hit:
        status = "both_absent"
    elif not head_hit:
        status = "head_absent"
    else:
        status = "tail_absent"

    return {
        "head_mentioned": head_hit,
        "tail_mentioned": tail_hit,
        "both_mentioned": head_hit and tail_hit,
        "alignment_status": status,
    }


def summarize_alignment(rows: Iterable[dict]) -> dict:
    records = list(rows)
    total = len(records)
    if total == 0:
        return {"n": 0}

    both = sum(1 for r in records if r["both_mentioned"])
    any_missing = total - both
    head_missing = sum(1 for r in records if not r["head_mentioned"])
    tail_missing = sum(1 for r in records if not r["tail_mentioned"])

    return {
        "n": total,
        "both_mentioned": both,
        "both_mentioned_rate": both / total,
        "any_entity_missing": any_missing,
        "any_entity_missing_rate": any_missing / total,
        "head_missing": head_missing,
        "head_missing_rate": head_missing / total,
        "tail_missing": tail_missing,
        "tail_missing_rate": tail_missing / total,
    }
