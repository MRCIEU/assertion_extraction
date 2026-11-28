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
        name = ""
        in_action = 0
        for event, element in ET.iterparse(handle, events=("start", "end")):
            if event == "start" and element.tag == "DescriptorRecord":
                current = None
                trees = []
                actions = 0
                seen_ui = False
                name = ""
                in_action = 0
            elif event == "start" and element.tag == "PharmacologicalAction":
                in_action += 1
            elif event == "end" and element.tag == "DescriptorUI" and not seen_ui:
                current = (element.text or "").strip()
                seen_ui = True
            elif event == "end" and element.tag == "String" and seen_ui and not name and in_action == 0:
                name = (element.text or "").strip()
            elif event == "end" and element.tag == "TreeNumber" and seen_ui and in_action == 0:
                if element.text:
                    trees.append(element.text.strip())
            elif event == "end" and element.tag == "PharmacologicalAction":
                actions += 1
                in_action = max(0, in_action - 1)
            elif event == "end" and element.tag == "DescriptorRecord":
                if current:
                    index[current] = {
                        "name": name,
                        "tree_numbers": list(trees),
                        "n_pharmacological_actions": actions,
                    }
                element.clear()
    return index


def tree_labels(descriptors: dict[str, dict]) -> dict[str, str]:
    """Map a MeSH tree number to the descriptor that owns that exact number."""
    labels = {}
    for record in descriptors.values():
        for number in record.get("tree_numbers") or []:
            labels.setdefault(number, record.get("name") or "")
    return labels


def concept_inventory(rows: list[dict]) -> dict:
    """Group mentions by identifier without taking the first entity type."""
    mentions_without_identifier = 0
    grouped: dict[str, dict] = {}
    for row in rows:
        concept_id = str(row.get("concept_id") or "").strip()
        if not concept_id or concept_id == "-":
            mentions_without_identifier += 1
            continue
        slot = grouped.setdefault(concept_id, {"n_mentions": 0, "entity_types": set()})
        slot["n_mentions"] += 1
        slot["entity_types"].add(str(row.get("entity_type") or ""))
    concepts = []
    composites = 0
    conflicts = 0
    for concept_id, slot in grouped.items():
        types = sorted(slot["entity_types"])
        conflict = len(types) != 1
        family = identifier_family(concept_id)
        conflicts += int(conflict)
        composites += int(family == "composite_mesh")
        concepts.append(
            {
                "concept_id": concept_id,
                "n_mentions": slot["n_mentions"],
                "entity_types": types,
                "entity_type": None if conflict else types[0],
                "type_conflict": conflict,
                "identifier_family": family,
            }
        )
    return {
        "n_mentions": len(rows),
        "n_mentions_without_identifier": mentions_without_identifier,
        "n_unique_nonempty_identifiers": len(concepts),
        "n_composite_identifiers": composites,
        "n_type_conflicts": conflicts,
        "concepts": concepts,
    }


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
