"""Parse PubTator3 biocjson documents into normalised entity records."""

from __future__ import annotations

from typing import Any

from .config import PUBTATOR_TYPE_MAP


def parse_entities(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract entity annotations with CIViC-aligned types."""
    entities: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int | None]] = set()

    for passage in doc.get("passages") or []:
        for ann in passage.get("annotations") or []:
            infons = ann.get("infons") or {}
            pt_type = infons.get("type") or ""
            civic_type = PUBTATOR_TYPE_MAP.get(pt_type)
            if civic_type is None:
                continue

            text = (ann.get("text") or "").strip()
            if not text:
                continue

            loc = (ann.get("locations") or [{}])[0]
            offset = loc.get("offset")
            length = loc.get("length")
            norm_id = infons.get("normalized_id") or infons.get("identifier")
            if isinstance(norm_id, list):
                norm_id = norm_id[0] if norm_id else None

            entity_key = str(norm_id) if norm_id not in (None, "", "None") else text.lower()
            dedupe_key = (civic_type, entity_key, offset)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            entities.append(
                {
                    "text": text,
                    "civic_type": civic_type,
                    "pubtator_type": pt_type,
                    "normalized_id": norm_id,
                    "entity_key": entity_key,
                    "offset": offset,
                    "length": length,
                    "database": infons.get("database"),
                }
            )
    return entities


def entities_by_type(entities: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group unique entities by CIViC type; retain all surface forms for matching."""
    buckets: dict[str, dict[str, dict[str, Any]]] = {}
    for ent in entities:
        ctype = ent["civic_type"]
        key = ent["entity_key"]
        buckets.setdefault(ctype, {})
        if key not in buckets[ctype]:
            buckets[ctype][key] = {**ent, "all_texts": {ent["text"]}}
        else:
            buckets[ctype][key]["all_texts"].add(ent["text"])
            # Prefer shortest surface form as display text (usually gene symbol).
            if len(ent["text"]) < len(buckets[ctype][key]["text"]):
                buckets[ctype][key]["text"] = ent["text"]

    out: dict[str, list[dict[str, Any]]] = {}
    for ctype, vals in buckets.items():
        records = []
        for rec in vals.values():
            rec = dict(rec)
            rec["all_texts"] = sorted(rec["all_texts"])
            records.append(rec)
        out[ctype] = records
    return out
