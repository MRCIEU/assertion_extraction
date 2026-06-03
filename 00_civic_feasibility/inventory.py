"""Build evaluable-target inventory from cached CIViC evidence items."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, OUTPUT_DIR


def _first_gene(record: dict) -> tuple[str | None, str | None]:
    mp = record.get("molecularProfile") or {}
    for variant in mp.get("variants") or []:
        feature = variant.get("feature") or {}
        if feature.get("featureType") == "GENE" and feature.get("name"):
            return feature["name"], "gene"
    return None, None


def _first_variant(record: dict) -> tuple[str | None, str | None]:
    mp = record.get("molecularProfile") or {}
    variants = mp.get("variants") or []
    if variants and variants[0].get("name"):
        return variants[0]["name"], "variant"
    if mp.get("name"):
        return mp["name"], "variant"
    return None, None


def _resolve_head(record: dict) -> tuple[str | None, str | None]:
    gene_name, _ = _first_gene(record)
    if gene_name:
        return gene_name, "gene"
    return _first_variant(record)


def _resolve_tail(record: dict) -> tuple[str | None, str | None, list[str]]:
    therapies = record.get("therapies") or []
    therapy_names = [t["name"] for t in therapies if t.get("name")]
    if therapy_names:
        return therapy_names[0], "drug", therapy_names
    disease = record.get("disease") or {}
    if disease.get("name"):
        return disease["name"], "disease", []
    return None, None, therapy_names


def _pair_type(head_type: str | None, tail_type: str | None) -> str | None:
    if not head_type or not tail_type:
        return None
    return f"{head_type}–{tail_type}"


def _is_pubmed_source(source: dict | None) -> bool:
    return bool(source and source.get("sourceType") == "PUBMED" and source.get("citationId"))


def build_inventory(
    evidence_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    evidence_path = evidence_path or DATA_DIR / "evidence_items.json"
    output_path = output_path or OUTPUT_DIR / "evaluable_inventory.csv"

    records = json.loads(evidence_path.read_text(encoding="utf-8"))
    rows = []

    for item in records:
        source = item.get("source") or {}
        head_name, head_type = _resolve_head(item)
        tail_name, tail_type, all_therapies = _resolve_tail(item)
        assertions = item.get("assertions") or []
        assertion = assertions[0] if assertions else {}

        is_pubmed = _is_pubmed_source(source)
        has_two_entities = bool(head_name and tail_name and head_type != tail_type)
        is_evaluable = is_pubmed and has_two_entities

        rows.append(
            {
                "evidence_id": item.get("id"),
                "evidence_name": item.get("name"),
                "status": item.get("status"),
                "source_id": source.get("id"),
                "source_type": source.get("sourceType"),
                "pmid": source.get("citationId"),
                "source_title": source.get("title"),
                "has_civic_abstract": bool(source.get("abstract")),
                "head_entity": head_name,
                "head_type": head_type,
                "tail_entity": tail_name,
                "tail_type": tail_type,
                "entity_pair_type": _pair_type(head_type, tail_type),
                "all_therapies": "; ".join(all_therapies),
                "disease": (item.get("disease") or {}).get("name"),
                "molecular_profile": (item.get("molecularProfile") or {}).get("name"),
                "evidence_type": item.get("evidenceType"),
                "evidence_direction": item.get("evidenceDirection"),
                "clinical_significance": item.get("significance"),
                "evidence_level": item.get("evidenceLevel"),
                "assertion_id": assertion.get("id"),
                "assertion_direction": assertion.get("assertionDirection"),
                "assertion_type": assertion.get("assertionType"),
                "assertion_significance": assertion.get("significance"),
                "is_pubmed_source": is_pubmed,
                "has_two_entities": has_two_entities,
                "is_evaluable_target": is_evaluable,
            }
        )

    inventory = pd.DataFrame(rows)
    inventory.to_csv(output_path, index=False)

    print("\n=== Inventory summary ===")
    print(f"  total accepted evidence items: {len(inventory)}")
    print(f"  pubmed sources: {inventory['is_pubmed_source'].sum()}")
    print(f"  evaluable (pubmed + two-entity): {inventory['is_evaluable_target'].sum()}")

    return inventory


if __name__ == "__main__":
    build_inventory()
