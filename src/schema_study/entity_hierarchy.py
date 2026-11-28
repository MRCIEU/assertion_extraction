"""Variable-depth entity classes from selected public resources.

E0–E2 stay the native representation conditions. An additional level is a
taxonomic or nomenclature class, not the entity's own identifier and not a
pharmacological role. A branch that stops at E2 is a terminal classification.
"""

from __future__ import annotations

import math
from collections import Counter

from schema_study.entity_classification import (
    NATIVE_EXPECTED_TOP,
    _single_family,
    identifier_family,
)

SELECTION_RULE = "admissible_branch_then_common_ancestor_v1"
GLOBAL_LEVEL_MIN_TRAINING_CONCEPTS = 100
LEAKAGE_TERMS = ("inhibitor", "antagonist", "agonist", "activator", "blocker")
STRUCTURAL_CHEMICAL_TOPS = frozenset({
    "D01", "D02", "D03", "D04", "D05", "D08", "D09", "D10", "D12", "D13", "D20", "D23", "D25",
})
FUNCTIONAL_CHEMICAL_TOPS = frozenset({"D06", "D26", "D27"})
TAXON_DISPLAY_RANKS = ("superkingdom", "kingdom", "phylum", "class", "order", "family", "genus")
CELL_CLASS_SUBSETS = {
    "Cancer_cell_line",
    "Conditionally_immortalized_cell_line",
    "Embryonic_stem_cell",
    "Factor-dependent_cell_line",
    "Finite_cell_line",
    "Hybrid_cell_line",
    "Hybridoma",
    "Induced_pluripotent_stem_cell",
    "Somatic_stem_cell",
    "Spontaneously_immortalized_cell_line",
    "Stromal_cell_line",
    "Telomerase_immortalized_cell_line",
    "Transformed_cell_line",
    "Undefined_cell_line_type",
}
VARIANT_CODES = {
    "SUB": ("SO:1000002", "substitution"),
    "DEL": ("SO:0000159", "deletion"),
    "INS": ("SO:0000667", "insertion"),
    "DELINS": ("SO:1000032", "delins"),
    "INDEL": ("SO:1000032", "delins"),
    "DUP": ("SO:1000035", "duplication"),
}
DISEASE_XREF_PREFIXES = ("NCIt:", "MONDO:", "Orphanet:", "ORDO:")


def _leakage(label: str) -> bool:
    lowered = label.lower()
    return any(term in lowered for term in LEAKAGE_TERMS)


def _tree_top(tree: str) -> str:
    return tree.split(".", 1)[0]


def _ancestor_nodes(tree: str, labels: dict[str, str]) -> tuple[list[dict], str]:
    """Ancestors of a descriptor. The final tree node is the concept identity."""
    pieces = tree.split(".")
    nodes = []
    for index in range(1, len(pieces)):
        node = ".".join(pieces[:index])
        label = labels.get(node) or ""
        if not label:
            return [], "missing_ancestor_label"
        if _leakage(label):
            return [], "supplementary_keyword"
        nodes.append({"node": node, "label": label, "relation": "mesh_tree_ancestor"})
    return nodes, ""


def mesh_chain(tree_numbers: list[str], entity_type: str, labels: dict[str, str], pharmacological_actions: int) -> dict:
    """Keep scientifically admissible branches. Lexicographic order is only the remaining tie-break."""
    expected = NATIVE_EXPECTED_TOP.get(entity_type, set())
    functional = []
    keyword_excluded = []
    admissible = []
    for tree in tree_numbers:
        top = _tree_top(tree)
        if entity_type == "ChemicalEntity" and top in FUNCTIONAL_CHEMICAL_TOPS:
            functional.append(tree)
            continue
        if tree[:1] not in expected:
            continue
        if entity_type == "ChemicalEntity" and top not in STRUCTURAL_CHEMICAL_TOPS:
            functional.append(tree)
            continue
        nodes, problem = _ancestor_nodes(tree, labels)
        if problem == "supplementary_keyword":
            keyword_excluded.append(tree)
            continue
        if problem == "missing_ancestor_label":
            continue
        if nodes:
            admissible.append((tree, nodes))
    base = {
        "pharmacological_actions_excluded": pharmacological_actions,
        "functional_trees_excluded": functional,
        "keyword_excluded_trees": keyword_excluded,
        "selection_rule": SELECTION_RULE,
        "relation_semantics": "mesh_tree_not_a_formal_subclass",
    }
    if not admissible:
        if functional and not keyword_excluded:
            status = "excluded_functional_branch"
        elif keyword_excluded:
            status = "excluded_relation_leakage"
        else:
            status = "no_admissible_tree"
        return {"status": status, "chain": [], "eligible": False, **base}
    admissible.sort(key=lambda item: item[0])
    selected, chain = admissible[0]
    return {
        "status": "classified",
        "chain": chain,
        "eligible": True,
        "selected_tree": selected,
        "alternative_chains": [nodes for _, nodes in admissible[1:]],
        "alternative_trees": [tree for tree, _ in admissible[1:]],
        "n_admissible_trees": len(admissible),
        **base,
    }


def classify_mesh_identifier(concept_id: str, entity_type: str, descriptors: dict[str, dict], labels: dict[str, str]) -> dict:
    """A composite is complete only when every descriptor part meets the same chain."""
    parts = [part.strip() for part in str(concept_id).split("|") if part.strip()]
    if not parts or any(_single_family(part) != "mesh_descriptor" for part in parts):
        return {"status": "incomplete_composite" if len(parts) > 1 else "identifier_not_a_descriptor", "chain": [], "eligible": False}
    results = []
    for part in parts:
        record = descriptors.get(part)
        if record is None or not record.get("tree_numbers"):
            return {"status": "incomplete_composite", "chain": [], "eligible": False, "missing_descriptor": part}
        results.append(mesh_chain(record["tree_numbers"], entity_type, labels, int(record.get("n_pharmacological_actions") or 0)))
    if len(parts) == 1:
        return results[0]
    if any(not item["eligible"] for item in results):
        failed = dict(next(item for item in results if not item["eligible"]))
        failed["status"] = "incomplete_composite"
        failed["eligible"] = False
        return failed
    chains = [item["chain"] for item in results]
    prefix = _common_nodes(chains)
    identities = {tuple(node["node"] for node in chain) for chain in chains}
    if len(identities) == 1:
        chosen = dict(results[0])
        chosen["n_components"] = len(parts)
        chosen["composite_complete"] = True
        return chosen
    if prefix:
        return {
            "status": "partial_common_ancestor",
            "chain": prefix,
            "eligible": True,
            "n_components": len(parts),
            "alternative_chains": chains,
            "reason": "components_share_a_class_and_differ_below_it",
        }
    return {
        "status": "ambiguous_paths",
        "chain": [],
        "eligible": False,
        "alternative_chains": chains,
        "reason": "no_shared_class_identity",
    }


def _common_nodes(chains: list[list[dict]]) -> list[dict]:
    """Shared prefix by ontology node identity, not by display label."""
    if not chains:
        return []
    prefix = []
    for nodes in zip(*chains):
        if len({node["node"] for node in nodes}) != 1:
            break
        prefix.append(dict(nodes[0]))
    return prefix


def classify_variant(concept_id: str, so_ids: set[str]) -> dict:
    """Documented PubTator type codes only. An rs identifier does not determine a class."""
    family = identifier_family(concept_id)
    if family == "dbsnp":
        return {"status": "no_additional_distinction", "chain": [], "eligible": True, "reason": "rs_identifier_does_not_determine_variant_class"}
    if family != "variant_notation":
        return {"status": "unresolved_variant", "chain": [], "eligible": False}
    parts = str(concept_id).split("|")
    code = parts[1] if len(parts) > 1 else ""
    mapped = VARIANT_CODES.get(code)
    if mapped is None:
        return {"status": "unresolved_variant_code", "chain": [], "eligible": False, "code": code}
    so_id, name = mapped
    if so_id not in so_ids:
        return {"status": "sequence_ontology_id_absent", "chain": [], "eligible": False, "so_id": so_id}
    return {
        "status": "classified",
        "chain": [{"node": so_id, "label": name, "relation": "sequence_ontology_nomenclature"}],
        "eligible": True,
        "reason": "pubtator_type_code",
    }


def taxonomy_chain(taxid: str, parents: dict[str, str], ranks: dict[str, str], names: dict[str, str]) -> dict:
    """Ranks between the organism and the taxonomy root. The taxon itself is not a new type."""
    if taxid not in parents and taxid not in ranks:
        return {"status": "unresolved_taxon", "chain": [], "eligible": False}
    current = taxid
    seen = set()
    collected = []
    while current and current not in seen:
        seen.add(current)
        if current != taxid and ranks.get(current) in TAXON_DISPLAY_RANKS:
            label = names.get(current) or ""
            if not label:
                return {"status": "missing_taxon_name", "chain": [], "eligible": False}
            collected.append({"node": f"NCBITaxon:{current}", "label": label, "rank": ranks.get(current), "relation": "taxonomic_parent"})
        parent = parents.get(current)
        if not parent or parent == current:
            break
        current = parent
    collected.reverse()
    status = "classified" if collected else "no_additional_distinction"
    return {"status": status, "chain": collected, "eligible": True}


def classify_cell_line(subsets: list[str], xrefs: list[str]) -> dict:
    """A Cellosaurus category is one class. Disease of origin is not the next level."""
    classes = sorted({item for item in subsets if item in CELL_CLASS_SUBSETS})
    disease = [item for item in xrefs if item.startswith(DISEASE_XREF_PREFIXES)]
    if len(classes) > 1:
        return {
            "status": "ambiguous_paths",
            "chain": [],
            "eligible": False,
            "alternative_labels": classes,
            "disease_of_origin_excluded": disease,
        }
    chain = []
    if len(classes) == 1:
        chain = [{"node": classes[0], "label": classes[0], "relation": "cellosaurus_category"}]
    return {
        "status": "classified" if chain else "no_additional_distinction",
        "chain": chain,
        "eligible": True,
        "disease_of_origin_excluded": disease,
    }


def classify_gene(concept_id: str, gene_types: dict[str, dict] | None) -> dict:
    """A database gene type is not the molecular form named by a mention."""
    record = (gene_types or {}).get(str(concept_id))
    if not record:
        return {
            "status": "no_additional_distinction",
            "chain": [],
            "eligible": True,
            "database_type": None,
            "reason": "gene_identifier_does_not_establish_molecular_form",
        }
    return {
        "status": "database_gene_type_recorded",
        "chain": [],
        "eligible": True,
        "database_type": record.get("type_of_gene"),
        "tax_id": record.get("tax_id"),
        "reason": "type_of_gene_does_not_establish_the_molecular_form_in_the_mention",
    }


def classify_concept(concept: dict, resources: dict) -> dict:
    """Choose a resource from the native type. Do not take the first mention's type when they conflict."""
    if concept.get("type_conflict"):
        return {"status": "type_conflict", "chain": [], "eligible": False, "reason": "several_native_categories"}
    entity_type = concept.get("entity_type") or ""
    concept_id = str(concept.get("concept_id") or "")
    if entity_type == "GeneOrGeneProduct":
        return classify_gene(concept_id, resources.get("gene_types"))
    if entity_type == "SequenceVariant":
        classified = classify_variant(concept_id, resources.get("so_ids") or set())
        annotated = (resources.get("variant_classes") or {}).get(concept_id)
        if annotated:
            classified = dict(classified)
            classified["official_variant_class"] = annotated
            if classified.get("chain"):
                return classified
            if annotated.get("n_classes", 0) == 1 and annotated.get("so_id") in (resources.get("so_ids") or set()):
                return {
                    "status": "classified",
                    "chain": [{
                        "node": annotated["so_id"],
                        "label": annotated["label"],
                        "relation": "official_variant_annotation",
                    }],
                    "eligible": True,
                    "reason": "official_variant_record",
                }
            classified["reason"] = "variant_record_does_not_give_one_class"
        return classified
    if entity_type == "CellLine":
        record = (resources.get("cell_lines") or {}).get(concept_id)
        if record is None:
            return {"status": "unresolved_cell_line", "chain": [], "eligible": False}
        return classify_cell_line(record.get("subsets") or [], record.get("xrefs") or [])
    if entity_type in {"OrganismTaxon", "Organism"}:
        return taxonomy_chain(concept_id, resources.get("parents") or {}, resources.get("ranks") or {}, resources.get("names") or {})
    if entity_type not in NATIVE_EXPECTED_TOP:
        return {"status": "no_rule", "chain": [], "eligible": False}
    supplemental = (resources.get("supplemental") or {}).get(concept_id)
    if supplemental is not None and _single_family(concept_id) == "mesh_supplemental":
        return {
            "status": "associated_descriptor_not_a_subclass",
            "chain": [],
            "eligible": True,
            "mapped_descriptors": supplemental.get("descriptors") or [],
            "reason": "heading_mapped_to_is_an_indexing_relation",
        }
    mesh = classify_mesh_identifier(concept_id, entity_type, resources.get("descriptors") or {}, resources.get("labels") or {})
    return _prefer_formal_relation(concept_id, entity_type, mesh, resources)


def _prefer_formal_relation(concept_id: str, entity_type: str, mesh: dict, resources: dict) -> dict:
    """Use an is_a mapping when it is unique. A MeSH tree remains provenance, not the primary type."""
    parts = [part.strip() for part in concept_id.split("|") if part.strip()]
    overlay = None
    if len(parts) == 1:
        table = resources.get("chebi_by_mesh") if entity_type == "ChemicalEntity" else resources.get("mondo_by_mesh")
        overlay = (table or {}).get(parts[0])
    if overlay and overlay.get("n_matches") == 1 and overlay.get("chain"):
        descriptor_name = ((resources.get("descriptors") or {}).get(concept_id) or {}).get("name") or ""
        chain = [
            node for node in overlay["chain"]
            if not descriptor_name or node["label"].casefold() != descriptor_name.casefold()
        ]
        if chain:
            return {
                "status": "classified",
                "chain": chain,
                "eligible": True,
                "relation": overlay.get("relation"),
                "alternative_parents": overlay.get("alternative_parents") or [],
                "mesh_chain": mesh.get("chain") or [],
                "mesh_status": mesh.get("status"),
                "reason": overlay.get("reason"),
            }
    if overlay and overlay.get("n_matches", 0) > 1:
        mesh = dict(mesh)
        mesh["mapping_ambiguity"] = overlay.get("relation")
        mesh["n_matches"] = overlay["n_matches"]
    if mesh.get("eligible") and mesh.get("chain"):
        recorded = dict(mesh)
        recorded["mesh_chain"] = mesh["chain"]
        recorded["chain"] = []
        recorded["status"] = "mesh_tree_recorded_not_used_as_type"
        recorded["eligible"] = True
        recorded["reason"] = "mesh_tree_edge_is_not_a_formal_subclass"
        return recorded
    return mesh


def displayed_type(native_type: str | None, chain: list[dict], requested_level: int, merges: dict[str, str] | None = None) -> str | None:
    """A missing deeper class keeps the native type. It does not erase it."""
    if requested_level <= 0:
        return None
    if requested_level == 1:
        return (merges or {}).get(native_type or "", native_type)
    if requested_level <= 2 or not chain:
        return native_type
    realised = realised_classification(chain, requested_level)
    return realised["label"] or native_type


def realised_classification(chain: list[dict], requested_level: int) -> dict:
    """Carry the terminal class forward. A repeated class is not a new distinction."""
    if requested_level < 3:
        raise ValueError("E0–E2 are the native representation conditions, not positions on this chain.")
    index = requested_level - 3
    if index < len(chain):
        return {
            "requested_level": requested_level,
            "realised_level": requested_level,
            "label": chain[index]["label"],
            "new_distinction": True,
            "carry_forward": False,
        }
    if chain:
        return {
            "requested_level": requested_level,
            "realised_level": 2 + len(chain),
            "label": chain[-1]["label"],
            "new_distinction": False,
            "carry_forward": True,
            "reason": "terminal_classification",
        }
    return {
        "requested_level": requested_level,
        "realised_level": 2,
        "label": None,
        "new_distinction": False,
        "carry_forward": True,
        "reason": "no_additional_distinction",
    }


def information_content(chain_nodes: list[list[str]]) -> dict[str, float]:
    """Natural-log information content. Each concept contributes once along its selected path."""
    total = len(chain_nodes)
    if total == 0:
        return {}
    counts: Counter[str] = Counter()
    for nodes in chain_nodes:
        for node in dict.fromkeys(nodes):
            counts[node] += 1
    return {node: -math.log(count / total) for node, count in counts.items()}


def proposed_global_depth(new_distinction_counts: dict[int, int], minimum: int = GLOBAL_LEVEL_MIN_TRAINING_CONCEPTS) -> int:
    """Deepest contiguous level at which each step changes at least `minimum` training concepts."""
    deepest = 2
    level = 3
    while new_distinction_counts.get(level, 0) >= minimum:
        deepest = level
        level += 1
    return deepest
