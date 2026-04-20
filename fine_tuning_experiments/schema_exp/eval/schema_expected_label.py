#!/usr/bin/env python3.11
"""
schema_expected_label — maps a CIViC audit target to the label (or set of
labels) that *should* be predicted under a given schema.

This module is the basis for three correctness-aware KB metrics used in the
Phase A re-analysis and Phase B evaluation:

    Method A : schema_expected_label_hit  = 1 if argmax(P_schema) ∈ expected_set
    Method B : mean_P_expected            = sum_{L in expected_set} P(L)
    Method C : AUC of abstention-recall   curve (uses Method A's hit definition)

Design choices (peer-reviewable; see schema_expected_label_mapping_rationale.md):

1. Mapping is defined at the **family** level (not label level) so that a schema
   with more fine-grained heads (e.g. S_mech's five DGR_* mechanisms) is
   credited when any member of the family is the argmax, rather than being
   penalised for distributing probability across sub-heads.

2. For gene_drug targets we use the **primary mapping** (trust expected_pairing_family,
   ignore the occasional `heuristic_gold_s2_label == ASSOCIATION_GENERAL`
   annotation) because CIViC gene_drug evidence is definitionally drug-gene
   regulation; the heuristic AG calls (64/154) reflect label uncertainty
   rather than a genuine "general association, not regulation" judgement.
   A sensitivity mapping that respects heuristic AG calls is also provided.

3. For variant_disease targets, heuristic_gold_s2_label == VARIANT_GENE (3/11)
   is unmapped across all schemas because no schema has a VARIANT_GENE head;
   these targets are excluded from the metric denominator.

4. For variant_disease targets with heuristic_gold_s2_label == ASSOCIATION_GENERAL,
   we project to the schema's *dedicated* VARIANT_DISEASE head when available
   (S_pair / S_mech) and fall back to ASSOCIATION_GENERAL in S_flat (which has
   no VD head).  Rationale: when a schema can express the pair type directly,
   that is the semantically correct target.

The function returns a set of acceptable labels (not a single string) so that
Method A/B can accumulate over the set in a schema-appropriate way.
"""
from __future__ import annotations

from typing import Literal, Mapping

# ───────────────────────────────────────────────────────────────────────
# Schema label vocabularies (must match the trained classifiers' label2id)
# ───────────────────────────────────────────────────────────────────────

SCHEMA_LABELS: Mapping[str, frozenset[str]] = {
    "S_flat": frozenset({
        "ASSOCIATION_GENERAL", "DRUG_DISEASE", "DRUG_GENE_REGULATION",
        "__NEGATIVE__",
    }),
    "S_pair": frozenset({
        "ASSOCIATION_GENERAL", "DRUG_DISEASE", "DRUG_GENE_REGULATION",
        "DRUG_VARIANT_ASSOC", "GENE_DISEASE", "GENE_GENE_ASSOC",
        "VARIANT_DISEASE", "__NEGATIVE__",
    }),
    "S_mech": frozenset({
        "ASSOCIATION_GENERAL", "DRUG_DISEASE", "DRUG_GENE_REGULATION",
        "DRUG_VARIANT_ASSOC", "GENE_DISEASE", "GENE_GENE_ASSOC",
        "VARIANT_DISEASE",
        "DGR_ACTIVATE", "DGR_INHIBIT", "DGR_METABOLIC",
        "DGR_REGULATE", "DGR_STRUCTURAL",
        "__NEGATIVE__",
    }),
}

# Abstract families used for cross-schema mapping. Each family specifies which
# schema labels are accepted as a "correct" argmax for that family.

FAMILY_TO_SCHEMA_LABELS: Mapping[str, Mapping[str, frozenset[str]]] = {
    "DGR_FAMILY": {
        "S_flat": frozenset({"DRUG_GENE_REGULATION"}),
        "S_pair": frozenset({"DRUG_GENE_REGULATION"}),
        "S_mech": frozenset({
            "DRUG_GENE_REGULATION",
            "DGR_ACTIVATE", "DGR_INHIBIT", "DGR_METABOLIC",
            "DGR_REGULATE", "DGR_STRUCTURAL",
        }),
    },
    "AG_FAMILY": {
        "S_flat": frozenset({"ASSOCIATION_GENERAL"}),
        "S_pair": frozenset({"ASSOCIATION_GENERAL"}),
        "S_mech": frozenset({"ASSOCIATION_GENERAL"}),
    },
    "VD_FAMILY": {
        # S_flat has no VARIANT_DISEASE head; AG is the schema's catch-all
        # for any relation it cannot express specifically.
        "S_flat": frozenset({"ASSOCIATION_GENERAL"}),
        "S_pair": frozenset({"VARIANT_DISEASE"}),
        "S_mech": frozenset({"VARIANT_DISEASE"}),
    },
    # VG_FAMILY intentionally absent — no schema has a VARIANT_GENE head.
}


MappingStrategy = Literal["primary", "sensitivity_trust_heuristic"]


def resolve_family(
    target: Mapping[str, str],
    strategy: MappingStrategy = "primary",
) -> tuple[str | None, str]:
    """Return (family, confidence) for a CIViC audit target.

    Parameters
    ----------
    target
        A mapping with at least the keys `expected_pairing_family` and
        `heuristic_gold_s2_label`.  Extra keys are ignored.
    strategy
        `"primary"` (default): family is determined from
        `expected_pairing_family`; `heuristic_gold_s2_label` is used only to
        flag VARIANT_GENE as unmapped.
        `"sensitivity_trust_heuristic"`: gene_drug targets whose
        `heuristic_gold_s2_label == ASSOCIATION_GENERAL` are assigned to
        AG_FAMILY rather than DGR_FAMILY.

    Returns
    -------
    (family, confidence)
        `family` is one of {"DGR_FAMILY", "AG_FAMILY", "VD_FAMILY", None}.
        `None` indicates the target is unmapped (excluded from KB metrics).
        `confidence` is one of {"high", "medium", "unmapped"}.
    """
    pf = target.get("expected_pairing_family", "") or ""
    gold = target.get("heuristic_gold_s2_label", "") or ""

    if pf == "gene_drug":
        if gold == "DRUG_GENE_REGULATION":
            return "DGR_FAMILY", "high"
        if gold == "ASSOCIATION_GENERAL":
            if strategy == "sensitivity_trust_heuristic":
                return "AG_FAMILY", "medium"
            return "DGR_FAMILY", "medium"
        # Unexpected gold value under gene_drug — flag but default to DGR
        return "DGR_FAMILY", "medium"

    if pf == "variant_disease":
        if gold == "VARIANT_GENE":
            return None, "unmapped"
        if gold == "ASSOCIATION_GENERAL":
            return "VD_FAMILY", "high"
        # Unexpected gold value under variant_disease — flag but default to VD
        return "VD_FAMILY", "medium"

    return None, "unmapped"


def schema_expected_label_set(
    target: Mapping[str, str],
    schema: str,
    strategy: MappingStrategy = "primary",
) -> tuple[frozenset[str], str]:
    """Return (set_of_acceptable_schema_labels, confidence) for a target.

    An argmax prediction whose label is in the returned set counts as a hit
    for Method A.  For Method B, the probability mass is summed over the set.

    Returns an empty frozenset with confidence `"unmapped"` if the target is
    not expressible in the given schema (excluded from the metric denominator).
    """
    if schema not in SCHEMA_LABELS:
        raise ValueError(
            f"Unknown schema {schema!r}; expected one of {sorted(SCHEMA_LABELS)}"
        )
    family, confidence = resolve_family(target, strategy=strategy)
    if family is None:
        return frozenset(), confidence
    family_labels = FAMILY_TO_SCHEMA_LABELS[family][schema]
    return family_labels, confidence


def is_hit(
    pred_label: str,
    target: Mapping[str, str],
    schema: str,
    strategy: MappingStrategy = "primary",
) -> bool:
    """Method A: argmax hit test."""
    expected_set, confidence = schema_expected_label_set(target, schema, strategy)
    if confidence == "unmapped":
        return False
    return pred_label in expected_set


__all__ = [
    "SCHEMA_LABELS",
    "FAMILY_TO_SCHEMA_LABELS",
    "MappingStrategy",
    "resolve_family",
    "schema_expected_label_set",
    "is_hit",
]
