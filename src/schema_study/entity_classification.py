"""Identifier families and public MeSH descriptor paths.

UMLS is not read. A MeSH tree path is provenance for a descriptor. It is not
a new entity identity, and a pharmacological action is not used as a type.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

MESH_RELEASE = "desc2026"
MESH_SOURCE = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.gz"
MESH_TOP = {
    "A": "Anatomy",
    "B": "Organisms",
    "C": "Diseases",
    "D": "Chemicals_and_Drugs",
    "E": "Analytical_Diagnostic_and_Therapeutic_Techniques",
    "F": "Psychiatry_and_Psychology",
    "G": "Phenomena_and_Processes",
    "H": "Disciplines_and_Occupations",
    "I": "Anthropology_Education_Sociology",
    "J": "Technology_Industry_Agriculture",
    "K": "Humanities",
    "L": "Information_Science",
    "M": "Named_Groups",
    "N": "Health_Care",
    "V": "Publication_Characteristics",
    "Z": "Geographicals",
}
NATIVE_EXPECTED_TOP = {
    "DiseaseOrPhenotypicFeature": {"C"},
    "ChemicalEntity": {"D"},
    "OrganismTaxon": {"B"},
}


def identifier_family(concept_id: str) -> str:
    """Classify a recorded identifier without inferring a biological type."""
    value = str(concept_id or "").strip()
    if not value:
        return "missing"
    parts = [part.strip() for part in value.split("|") if part.strip()]
    if len(parts) > 1 and all(_single_family(part) in {"mesh_descriptor", "mesh_supplemental"} for part in parts):
        return "composite_mesh"
    return _single_family(value)


def _single_family(value: str) -> str:
    if value.startswith("OMIM:"):
        return "omim"
    if value.startswith("CVCL"):
        return "cellosaurus"
    if len(value) > 1 and value[0] == "D" and value[1].isdigit():
        return "mesh_descriptor"
    if len(value) > 1 and value[0] == "C" and value[1].isdigit():
        return "mesh_supplemental"
    if value.isdigit():
        return "numeric"
    if value.startswith("rs") and value[2:].isdigit():
        return "dbsnp"
    if "|" in value or value.startswith(("p.", "c.", "g.")):
        return "variant_notation"
    return "other"


def load_mesh_descriptors(path: Path) -> dict[str, dict]:
    """Descriptor UI to tree numbers. Nested descriptor references are ignored."""
    index = {}
    with gzip.open(path, "rb") as handle:
        current = None
        trees: list[str] = []
        actions = 0
        seen_ui = False
        for event, element in ET.iterparse(handle, events=("start", "end")):
            if event == "start" and element.tag == "DescriptorRecord":
                current = None
                trees = []
                actions = 0
                seen_ui = False
            elif event == "end" and element.tag == "DescriptorUI" and not seen_ui:
                current = (element.text or "").strip()
                seen_ui = True
            elif event == "end" and element.tag == "TreeNumber" and seen_ui:
                if element.text:
                    trees.append(element.text.strip())
            elif event == "end" and element.tag == "PharmacologicalAction":
                actions += 1
            elif event == "end" and element.tag == "DescriptorRecord":
                if current:
                    index[current] = {"tree_numbers": list(trees), "n_pharmacological_actions": actions}
                element.clear()
    return index


def top_levels(tree_numbers: list[str]) -> list[str]:
    return sorted({number.split(".", 1)[0][:1] for number in tree_numbers if number})


def summarise_mappings(rows: list[dict], descriptors: dict[str, dict]) -> dict:
    """Coverage of one corpus. `rows` are mention or concept records."""
    by_type = {}
    multiple_paths = 0
    multiple_tops = 0
    outside_expected = 0
    pharmacological = 0
    mapped = 0
    for row in rows:
        concept_id = str(row.get("concept_id") or "")
        family = identifier_family(concept_id)
        entity_type = row.get("entity_type", "")
        slot = by_type.setdefault(
            entity_type,
            {"n": 0, "families": Counter(), "mesh_descriptor_found": 0, "mesh_descriptor_absent": 0},
        )
        slot["n"] += 1
        slot["families"][family] += 1
        descriptor_ids = [
            part
            for part in concept_id.split("|")
            if _single_family(part.strip()) == "mesh_descriptor"
        ]
        if not descriptor_ids:
            continue
        trees = []
        actions = 0
        absent = False
        for descriptor_id in descriptor_ids:
            record = descriptors.get(descriptor_id)
            if record is None:
                absent = True
                continue
            trees.extend(record["tree_numbers"])
            actions += int(record["n_pharmacological_actions"])
        if absent or not trees:
            slot["mesh_descriptor_absent"] += 1
            continue
        mapped += 1
        slot["mesh_descriptor_found"] += 1
        if len(trees) > 1 or len(descriptor_ids) > 1:
            multiple_paths += 1
        tops = top_levels(trees)
        if len(tops) > 1:
            multiple_tops += 1
        expected = NATIVE_EXPECTED_TOP.get(entity_type)
        if expected is not None and set(tops) - expected:
            outside_expected += 1
        if actions:
            pharmacological += 1
    return {
        "n": len(rows),
        "by_type": {
            entity_type: {**slot, "families": dict(slot["families"])}
            for entity_type, slot in sorted(by_type.items())
        },
        "mesh_descriptors_with_a_tree": mapped,
        "multiple_tree_numbers": multiple_paths,
        "multiple_top_level_categories": multiple_tops,
        "top_level_outside_the_native_expectation": outside_expected,
        "descriptors_with_a_pharmacological_action": pharmacological,
    }
