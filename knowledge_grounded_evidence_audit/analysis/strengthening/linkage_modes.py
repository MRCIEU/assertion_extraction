"""
KB linkage and audit-outcome variants (L1 strict, L2 relaxed, L3 grouped).

Relaxations are explicit and logged in linkage_variant_rules.json — not uncontrolled fuzzy matching.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# Import shared patterns from pipeline
import sys
from pathlib import Path

_KG = Path(__file__).resolve().parent.parent.parent
if str(_KG) not in sys.path:
    sys.path.insert(0, str(_KG))

from run_pipeline import LUNG_PAT, RawAssertion, norm_token

# Therapy families: token overlap with harmonized drug_therapy strings
THERAPY_FAMILIES: Dict[str, frozenset[str]] = {
    "egfr_tki_family": frozenset(
        {"gefitinib", "erlotinib", "icotinib", "afatinib", "dacomitinib", "osimertinib", "tarceva", "iressa"}
    ),
    "alk_inhibitor_family": frozenset(
        {"crizotinib", "alectinib", "brigatinib", "lorlatinib", "ceritinib", "ensartinib"}
    ),
    "braf_mek_family": frozenset({"dabrafenib", "trametinib", "vemurafenib", "encorafenib", "binimetinib"}),
}


def _drug_tokens(s: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", norm_token(s)) if len(t) >= 4]


def _family_match(drug_assert: str, drug_ledger: str) -> bool:
    da = set(_drug_tokens(drug_assert))
    dl = set(_drug_tokens(drug_ledger))
    if not da or not dl:
        return False
    if da & dl:
        return True
    for fam in THERAPY_FAMILIES.values():
        if da & fam and dl & fam:
            return True
    return False


def link_to_kb_variant(assertion: RawAssertion, harm_rows: List[Dict[str, str]], linkage_mode: str) -> Tuple[str, str, str]:
    """
    L1_strict: same scoring as production link_to_kb (gene + literal/partial drug rules).
    L2_relaxed: promote score-2 drug matches when therapy families align; allow L1 at score>=2 for drug_gene.
    L3_grouped: if gene matches any harmonized row and lung context, assign best L2 with first gene-matched row
                when drug match fails (documented weak grouping — audit only).
    """
    if assertion.relation_family == "negative":
        return "L3", "", "model_predicted_negative_relation_class"

    g = assertion.entity_a.get("normalized") or assertion.entity_a.get("text")
    db = assertion.entity_b.get("normalized") or assertion.entity_b.get("text")
    candidates: List[Tuple[int, Dict[str, str]]] = []

    for h in harm_rows:
        if h["gene"] != g:
            continue
        drug_h = h.get("drug_therapy", "")
        score = 0
        if assertion.relation_family == "drug_gene":
            if db and norm_token(db) in norm_token(drug_h):
                score = 3
            elif db and any(t in norm_token(drug_h) for t in norm_token(db).split()):
                score = 2
            elif linkage_mode != "L1_strict" and db and _family_match(str(db), drug_h):
                score = 2
        elif assertion.relation_family == "drug_disease":
            if db and norm_token(db) in norm_token(drug_h):
                score = 3
            elif db and any(t in norm_token(drug_h) for t in norm_token(db).split()):
                score = 2
            elif linkage_mode != "L1_strict" and db and _family_match(str(db), drug_h):
                score = 2
            elif LUNG_PAT.search(assertion.sentence):
                score = 1
        elif assertion.relation_family == "variant_disease":
            vn = (h.get("variant_civic") or "") + " " + (h.get("variant_oncokb") or "")
            et = (assertion.entity_b.get("text") or "").upper()
            if et and et[:16] in vn.upper():
                score = 2
            elif LUNG_PAT.search(assertion.sentence):
                score = 1
        elif assertion.relation_family == "gene_disease":
            score = 1 if LUNG_PAT.search(assertion.sentence) else 0
        if score > 0:
            candidates.append((score, h))

    if not candidates and linkage_mode == "L3_grouped" and LUNG_PAT.search(assertion.sentence or ""):
        for h in harm_rows:
            if h["gene"] == g:
                candidates.append((1, h))
                break

    if not candidates:
        return "L3", "", "no_kb_row_met_gene_drug_disease_overlap_rules"

    candidates.sort(key=lambda x: -x[0])
    best_s, best_h = candidates[0]

    if linkage_mode == "L1_strict":
        if best_s >= 3:
            return "L1", best_h["harmonized_key"], "gene_drug_literal_overlap_with_harmonized_anchor"
        if best_s == 2:
            return "L2", best_h["harmonized_key"], "partial_drug_token_overlap"
        return "L2", best_h["harmonized_key"], "gene_and_lung_context_without_exact_therapy_match"

    if linkage_mode == "L2_relaxed":
        if best_s >= 3:
            return "L1", best_h["harmonized_key"], "strict_or_family_aligned_therapy_match"
        if best_s == 2:
            return "L1", best_h["harmonized_key"], "relaxed_family_or_partial_drug_promoted_to_L1_for_audit"
        return "L2", best_h["harmonized_key"], "lung_context_only_or_weak_variant_overlap"

    # L3_grouped
    if best_s >= 3:
        return "L1", best_h["harmonized_key"], "grouped_gene_anchor_with_strong_overlap"
    if best_s >= 2:
        return "L2", best_h["harmonized_key"], "grouped_partial_or_family_overlap"
    return "L2", best_h["harmonized_key"], "grouped_gene_plus_lung_context_fallback"


def audit_outcome_variant(
    assertion: RawAssertion,
    link_level: str,
    in_ledger_gene: bool,
    *,
    l1_support: float = 0.55,
    l1_weak: float = 0.55,
    l2_conflict: float = 0.52,
    l3_gap: float = 0.48,
) -> str:
    conf = assertion.confidence
    if link_level == "L1" and conf >= l1_support:
        return "kb_supported_aligned"
    if link_level == "L1" and conf < l1_weak:
        return "kb_known_but_weak_current_support"
    if link_level == "L2" and conf >= l2_conflict:
        return "conflict_or_ambiguity"
    if link_level == "L2":
        return "kb_known_but_weak_current_support"
    if link_level == "L3" and in_ledger_gene and conf >= l3_gap:
        return "literature_supported_kb_absent_candidate"
    return "unsupported_or_low_trust"


def audit_outcome_permissive_gap(
    assertion: RawAssertion,
    link_level: str,
    in_ledger_gene: bool,
) -> str:
    """Slightly lower bar for gap bucket under linkage sensitivity (documented)."""
    return audit_outcome_variant(
        assertion,
        link_level,
        in_ledger_gene,
        l1_support=0.52,
        l1_weak=0.52,
        l2_conflict=0.50,
        l3_gap=0.45,
    )
