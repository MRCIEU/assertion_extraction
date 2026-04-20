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

1. Mapping is defined at the **family** level (not label level). Two projection
   modes are exposed:
     - `set_valued` (default, used as primary): when a schema has multiple
       labels covering one CIViC-level family (e.g. S_mech's DGR head + 5
       mechanism sub-heads all covering DGR_FAMILY), the expected set includes
       all of them, and any of them as argmax counts as a hit.
     - `single_label` (used as supplementary): the expected set is forced to a
       single canonical label per family (the semantic centre; for DGR_FAMILY
       this is the catch-all `DRUG_GENE_REGULATION` head in all schemas,
       not the sub-mechanism heads).
   `set_valued` is structurally favourable to S_mech (six hit paths vs one);
   `single_label` is structurally unfavourable to S_mech (it has to hit the
   dead catch-all head). Both are reported so that schema rankings on Method A
   and Method C can be shown to be robust to the projection choice
   (§8.5 of the rationale document).

2. For gene_drug targets we use the **primary strategy** (trust
   `expected_pairing_family`, ignore the occasional
   `heuristic_gold_s2_label == ASSOCIATION_GENERAL` annotation) because CIViC
   gene_drug evidence is definitionally drug-gene regulation; the heuristic
   AG calls (64/154) reflect label uncertainty rather than a genuine "general
   association, not regulation" judgement. A `sensitivity_trust_heuristic`
   strategy that respects heuristic AG calls is also provided.

3. For variant_disease targets, heuristic_gold_s2_label == VARIANT_GENE (3/11)
   is unmapped across all schemas because no schema has a VARIANT_GENE head;
   these targets are excluded from the metric denominator.

4. For variant_disease targets with heuristic_gold_s2_label == ASSOCIATION_GENERAL,
   we project to the schema's *dedicated* VARIANT_DISEASE head when available
   (S_pair / S_mech) and fall back to ASSOCIATION_GENERAL in S_flat (which has
   no VD head).

The function returns a set of acceptable labels so that Method A/B can
accumulate over the set in a schema-appropriate way. Under `single_label`
the set always has cardinality 0 or 1.
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

# Set-valued family → schema label projection. This is the "permissive" mapping
# that credits a schema's fine-grained heads when they cover a CIViC family.

FAMILY_TO_SCHEMA_LABELS_SET_VALUED: Mapping[str, Mapping[str, frozenset[str]]] = {
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
        "S_flat": frozenset({"ASSOCIATION_GENERAL"}),
        "S_pair": frozenset({"VARIANT_DISEASE"}),
        "S_mech": frozenset({"VARIANT_DISEASE"}),
    },
    # VG_FAMILY intentionally absent — no schema has a VARIANT_GENE head.
}

# Single-label family → schema label projection. The "strict" mapping that
# forces a single canonical head per family, regardless of how many heads a
# schema has in that family. Under this mapping S_mech gets only the DGR
# catch-all head for DGR_FAMILY (the 5 mechanism sub-heads are excluded).

FAMILY_TO_SCHEMA_LABELS_SINGLE_LABEL: Mapping[str, Mapping[str, frozenset[str]]] = {
    "DGR_FAMILY": {
        "S_flat": frozenset({"DRUG_GENE_REGULATION"}),
        "S_pair": frozenset({"DRUG_GENE_REGULATION"}),
        "S_mech": frozenset({"DRUG_GENE_REGULATION"}),  # catch-all only
    },
    "AG_FAMILY": {
        "S_flat": frozenset({"ASSOCIATION_GENERAL"}),
        "S_pair": frozenset({"ASSOCIATION_GENERAL"}),
        "S_mech": frozenset({"ASSOCIATION_GENERAL"}),
    },
    "VD_FAMILY": {
        "S_flat": frozenset({"ASSOCIATION_GENERAL"}),
        "S_pair": frozenset({"VARIANT_DISEASE"}),
        "S_mech": frozenset({"VARIANT_DISEASE"}),
    },
}

MappingStrategy = Literal["primary", "sensitivity_trust_heuristic"]
ProjectionMode = Literal["set_valued", "single_label"]


def _projection_table(mode: ProjectionMode) -> Mapping[str, Mapping[str, frozenset[str]]]:
    if mode == "set_valued":
        return FAMILY_TO_SCHEMA_LABELS_SET_VALUED
    if mode == "single_label":
        return FAMILY_TO_SCHEMA_LABELS_SINGLE_LABEL
    raise ValueError(f"Unknown projection_mode {mode!r}")


def resolve_family(
    target: Mapping[str, str],
    strategy: MappingStrategy = "primary",
) -> tuple[str | None, str]:
    """Return (family, confidence) for a CIViC audit target.

    Parameters
    ----------
    target
        A mapping with at least the keys `expected_pairing_family` and
        `heuristic_gold_s2_label`. Extra keys are ignored.
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
        return "DGR_FAMILY", "medium"

    if pf == "variant_disease":
        if gold == "VARIANT_GENE":
            return None, "unmapped"
        if gold == "ASSOCIATION_GENERAL":
            return "VD_FAMILY", "high"
        return "VD_FAMILY", "medium"

    return None, "unmapped"


def schema_expected_label_set(
    target: Mapping[str, str],
    schema: str,
    strategy: MappingStrategy = "primary",
    projection_mode: ProjectionMode = "set_valued",
) -> tuple[frozenset[str], str]:
    """Return (set_of_acceptable_schema_labels, confidence) for a target.

    An argmax prediction whose label is in the returned set counts as a hit
    for Method A. For Method B, the probability mass is summed over the set.

    Under `projection_mode == "single_label"` the returned set has cardinality
    at most 1; under `"set_valued"` it may have cardinality up to 6 (the
    DGR_FAMILY in S_mech case).
    """
    if schema not in SCHEMA_LABELS:
        raise ValueError(
            f"Unknown schema {schema!r}; expected one of {sorted(SCHEMA_LABELS)}"
        )
    family, confidence = resolve_family(target, strategy=strategy)
    if family is None:
        return frozenset(), confidence
    table = _projection_table(projection_mode)
    return table[family][schema], confidence


def is_hit(
    pred_label: str,
    target: Mapping[str, str],
    schema: str,
    strategy: MappingStrategy = "primary",
    projection_mode: ProjectionMode = "set_valued",
) -> bool:
    """Method A: argmax hit test.

    Returns False for unmapped targets.
    """
    expected_set, confidence = schema_expected_label_set(
        target, schema, strategy=strategy, projection_mode=projection_mode,
    )
    if confidence == "unmapped":
        return False
    return pred_label in expected_set


# Backwards-compatible alias: older callers may use the combined set-valued
# projection table under its original name.
FAMILY_TO_SCHEMA_LABELS = FAMILY_TO_SCHEMA_LABELS_SET_VALUED


__all__ = [
    "SCHEMA_LABELS",
    "FAMILY_TO_SCHEMA_LABELS",
    "FAMILY_TO_SCHEMA_LABELS_SET_VALUED",
    "FAMILY_TO_SCHEMA_LABELS_SINGLE_LABEL",
    "MappingStrategy",
    "ProjectionMode",
    "resolve_family",
    "schema_expected_label_set",
    "is_hit",
]
