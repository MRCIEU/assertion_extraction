"""Map corpus entity-type labels onto the CIViC evaluation entity space."""

from __future__ import annotations

import re

ENTITY_TYPE_MAP: dict[str, str] = {
    "GeneOrGeneProduct": "gene",
    "ChemicalEntity": "drug",
    "DiseaseOrPhenotypicFeature": "disease",
    "SequenceVariant": "variant",
    "GENE": "gene",
    "GENE-Y": "gene",
    "GENE-N": "gene",
    "CHEMICAL": "drug",
    "Chemical": "drug",
    "Disease": "disease",
}

CIVIC_TYPE_ORDER = {"gene": 0, "variant": 1, "drug": 2, "disease": 3}


def normalize_entity_type(raw_type: str) -> str | None:
    if raw_type in ENTITY_TYPE_MAP:
        return ENTITY_TYPE_MAP[raw_type]
    for key, value in ENTITY_TYPE_MAP.items():
        if key.lower() == raw_type.lower():
            return value
    return None


def civic_pair_type(type_a: str, type_b: str) -> str | None:
    a = normalize_entity_type(type_a)
    b = normalize_entity_type(type_b)
    if not a or not b or a == b:
        return None
    ordered = sorted([a, b], key=lambda t: CIVIC_TYPE_ORDER.get(t, 99))
    return "-".join(ordered)


def normalize_entity_text(text: str) -> str:
    """Lowercase surface form for cross-corpus entity matching (conflict detection)."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def entity_surface(entity: dict) -> str:
    raw = entity.get("text") or [""]
    if isinstance(raw, list):
        return str(raw[0]) if raw else ""
    return str(raw)


def ordered_pair_key(type_a: str, text_a: str, type_b: str, text_b: str) -> tuple[str, str, str] | None:
    """Canonical undirected pair key: (pair_type, norm_ent_low_type, norm_ent_high_type)."""
    pt = civic_pair_type(type_a, type_b)
    if not pt:
        return None
    a_norm = normalize_entity_text(text_a)
    b_norm = normalize_entity_text(text_b)
    ta = normalize_entity_type(type_a)
    tb = normalize_entity_type(type_b)
    if not ta or not tb:
        return None
    if CIVIC_TYPE_ORDER.get(ta, 99) <= CIVIC_TYPE_ORDER.get(tb, 99):
        return (pt, a_norm, b_norm)
    return (pt, b_norm, a_norm)


def normalization_notes() -> list[str]:
    return [
        "BigBio `GeneOrGeneProduct` and DrugProt `GENE`/`GENE-Y` map to CIViC `gene`.",
        "BigBio `ChemicalEntity`, DrugProt `CHEMICAL`, and BC5CDR `Chemical` map to CIViC `drug`.",
        "BigBio `DiseaseOrPhenotypicFeature` and BC5CDR `Disease` map to CIViC `disease`.",
        "BigBio `SequenceVariant` maps to CIViC `variant`.",
        "Pairs are undirected when comparing to CIViC (gene–drug equals drug–gene).",
        "Conflict detection matches entity pairs on normalised surface text (lowercase, collapsed whitespace) plus CIViC entity type and pair type; database IDs are not used.",
        "BC5CDR chemical–disease pairs map to `drug-disease`, which is not a CIViC evaluation pair.",
    ]
