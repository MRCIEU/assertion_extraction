"""Loaders for classification resources that are not MeSH descriptor trees.

A cross-reference is recorded only when the file states it. A functional
relationship is not stored as a subclass.
"""

from __future__ import annotations

import gzip
import re
import xml.etree.ElementTree as ET
from pathlib import Path

CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")


def load_gene_types(directory: Path) -> dict[str, dict]:
    """NCBI type_of_gene for the species files present in the directory."""
    found = {}
    for path in sorted(directory.glob("*.gene_info.gz")):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            header = handle.readline().lstrip("#").rstrip("\n").split("\t")
            tax_at = header.index("tax_id")
            gene_at = header.index("GeneID")
            type_at = header.index("type_of_gene")
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= type_at:
                    continue
                found[parts[gene_at]] = {
                    "tax_id": parts[tax_at],
                    "type_of_gene": parts[type_at],
                    "source_file": path.name,
                }
    return found


def load_mesh_cas(path: Path) -> dict[str, str]:
    """Preferred-concept CAS registry numbers. A zero or an enzyme number is not a CAS."""
    found = {}
    with gzip.open(path, "rb") as handle:
        descriptor = ""
        preferred = False
        seen_ui = False
        for event, element in ET.iterparse(handle, events=("start", "end")):
            tag = element.tag.rsplit("}", 1)[-1]
            if event == "start" and tag == "DescriptorRecord":
                descriptor = ""
                seen_ui = False
                preferred = False
            elif event == "end" and tag == "DescriptorUI" and not seen_ui:
                descriptor = (element.text or "").strip()
                seen_ui = True
            elif event == "start" and tag == "Concept":
                preferred = element.attrib.get("PreferredConceptYN") == "Y"
            elif event == "end" and tag == "RegistryNumber" and preferred and descriptor and descriptor not in found:
                value = (element.text or "").strip()
                if CAS_PATTERN.match(value):
                    found[descriptor] = value
            elif event == "end" and tag == "DescriptorRecord":
                element.clear()
    return found


def load_chebi_by_cas(path: Path, wanted_cas: set[str]) -> dict[str, dict]:
    """ChEBI is_a ancestors for CAS numbers. has_role is not read as a class."""
    names = {}
    parents: dict[str, list[str]] = {}
    cas_hits: dict[str, list[str]] = {}
    current = ""
    current_parents: list[str] = []
    current_name = ""

    def save() -> None:
        if not current:
            return
        names[current] = current_name
        parents[current] = list(current_parents)

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("[Term]"):
                save()
                current = ""
                current_parents = []
                current_name = ""
            elif line.startswith("id: CHEBI:"):
                current = line[4:].strip()
            elif line.startswith("name: ") and not current_name:
                current_name = line[6:].strip()
            elif line.startswith("is_a: CHEBI:"):
                current_parents.append(line.split()[1])
            elif line.startswith("xref: cas:") and wanted_cas:
                cas = line.split(":", 2)[2].split()[0].strip()
                if cas in wanted_cas and current:
                    cas_hits.setdefault(cas, []).append(current)
        save()
    by_cas = {}
    for cas, identifiers in cas_hits.items():
        unique = list(dict.fromkeys(identifiers))
        chain = []
        alternatives = []
        if len(unique) == 1:
            chain, alternatives = _ancestor_chain(unique[0], names, parents, "chebi_is_a")
        by_cas[cas] = {
            "n_matches": len(unique),
            "chebi_ids": unique,
            "chain": chain,
            "alternative_parents": alternatives,
            "relation": "chebi_is_a",
        }
    return by_cas


def load_mondo_by_mesh(path: Path, wanted_mesh: set[str]) -> dict[str, dict]:
    """MONDO is_a ancestors for a unique equivalentTo MeSH cross-reference."""
    names = {}
    parents: dict[str, list[str]] = {}
    hits: dict[str, list[str]] = {}
    current = ""
    current_parents: list[str] = []
    current_name = ""

    def save() -> None:
        if current:
            names[current] = current_name
            parents[current] = list(current_parents)

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("[Term]"):
                save()
                current = ""
                current_parents = []
                current_name = ""
            elif line.startswith("id: MONDO:"):
                current = line[4:].strip()
            elif line.startswith("name: ") and not current_name:
                current_name = line[6:].strip()
            elif line.startswith("is_a: MONDO:"):
                current_parents.append(line.split()[1])
            elif line.startswith("xref: MESH:") and "equivalentTo" in line and current:
                mesh = line.split(":", 2)[2].split()[0].strip()
                if mesh in wanted_mesh:
                    hits.setdefault(mesh, []).append(current)
        save()
    found = {}
    for mesh, identifiers in hits.items():
        unique = list(dict.fromkeys(identifiers))
        chain, alternatives = _ancestor_chain(unique[0], names, parents, "mondo_is_a") if len(unique) == 1 else ([], [])
        found[mesh] = {
            "n_matches": len(unique),
            "mondo_ids": unique,
            "chain": chain,
            "alternative_parents": alternatives,
            "relation": "mondo_is_a",
            "reason": "unique_equivalentTo_mesh_cross_reference" if len(unique) == 1 else "several_equivalent_mondo_records",
        }
    return found


def load_supplemental_mappings(path: Path, wanted: set[str]) -> dict[str, dict]:
    """HeadingMappedTo descriptors. The mapped descriptor is not treated as the concept."""
    found = {}
    if not wanted:
        return found
    with gzip.open(path, "rb") as handle:
        current = ""
        descriptors = []
        actions = 0
        in_heading = 0
        in_action = 0
        for event, element in ET.iterparse(handle, events=("start", "end")):
            tag = element.tag.rsplit("}", 1)[-1]
            if event == "start" and tag == "SupplementalRecord":
                current = ""
                descriptors = []
                actions = 0
                in_heading = 0
                in_action = 0
            elif event == "start" and tag == "HeadingMappedTo":
                in_heading += 1
            elif event == "end" and tag == "HeadingMappedTo":
                in_heading = max(0, in_heading - 1)
            elif event == "start" and tag == "PharmacologicalAction":
                in_action += 1
            elif event == "end" and tag == "PharmacologicalAction":
                if in_action:
                    actions += 1
                in_action = max(0, in_action - 1)
            elif event == "end" and tag == "SupplementalRecordUI":
                current = (element.text or "").strip()
            elif event == "end" and tag == "DescriptorUI" and current in wanted and in_heading and not in_action:
                value = (element.text or "").strip().lstrip("*")
                if value.startswith("D"):
                    descriptors.append(value)
            elif event == "end" and tag == "SupplementalRecord":
                if current in wanted:
                    found[current] = {"descriptors": list(dict.fromkeys(descriptors)), "n_pharmacological_actions": actions}
                element.clear()
    return found


def chebi_for_mesh(cas_by_mesh: dict[str, str], chebi_by_cas: dict[str, dict]) -> dict[str, dict]:
    """Join a descriptor to ChEBI only through its preferred CAS."""
    joined = {}
    for mesh, cas in cas_by_mesh.items():
        hit = chebi_by_cas.get(cas)
        if not hit:
            continue
        joined[mesh] = {
            "n_matches": hit["n_matches"],
            "chain": hit["chain"] if hit["n_matches"] == 1 else [],
            "alternative_parents": hit.get("alternative_parents") or [],
            "relation": "chebi_is_a",
            "cas": cas,
            "chebi_ids": hit["chebi_ids"],
            "reason": "unique_cas_cross_reference" if hit["n_matches"] == 1 else "several_chebi_records_for_one_cas",
        }
    return joined


def _ancestor_chain(identifier: str, names: dict[str, str], parents: dict[str, list[str]], relation: str) -> tuple[list[dict], list[dict]]:
    """Single-parent ancestors. At a branch, keep the shared prefix and the alternative parents."""
    chain = []
    alternatives = []
    seen = {identifier}
    current = identifier
    while len(chain) <= 12:
        incoming = [parent for parent in parents.get(current, []) if parent not in seen]
        if len(incoming) != 1:
            alternatives = [
                {"node": parent, "label": names.get(parent) or parent, "relation": relation}
                for parent in incoming
                if names.get(parent)
            ]
            break
        parent = incoming[0]
        label = names.get(parent) or ""
        if not label:
            break
        chain.append({"node": parent, "label": label, "relation": relation})
        seen.add(parent)
        current = parent
    chain.reverse()
    return chain, alternatives
