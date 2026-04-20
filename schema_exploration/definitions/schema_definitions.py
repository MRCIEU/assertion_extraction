"""
Schema definitions for SC0, SC1, SC3.

Three data-driven schema candidates organised along two axes:
  Axis 1 (Entity-pair discrimination): E0 vs E1
  Axis 2 (Mechanism discrimination within DRUG_GENE): M0 vs M1
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# Axis 2 — DrugProt mechanism grouping (M1)
# Follows DrugProt annotation guidelines (Miranda-Escalada et al. 2021)
# ---------------------------------------------------------------------------
DRUGPROT_TO_M1: dict[str, str] = {
    # Inhibitory group (drug reduces protein function — direct or indirect)
    "INHIBITOR":              "DGR_INHIBIT",
    "ANTAGONIST":             "DGR_INHIBIT",
    "INDIRECT-DOWNREGULATOR": "DGR_INHIBIT",
    # Activating group (drug increases protein function or mimics ligand)
    "ACTIVATOR":              "DGR_ACTIVATE",
    "INDIRECT-UPREGULATOR":   "DGR_ACTIVATE",
    "AGONIST":                "DGR_ACTIVATE",
    "AGONIST-ACTIVATOR":      "DGR_ACTIVATE",
    # Enzymatic/metabolic group (protein metabolises or processes drug)
    "SUBSTRATE":              "DGR_METABOLIC",
    "PRODUCT-OF":             "DGR_METABOLIC",
    "SUBSTRATE_PRODUCT-OF":   "DGR_METABOLIC",
    "AGONIST-INHIBITOR":      "DGR_METABOLIC",
    # Structural/compositional group (drug is component of protein complex)
    "PART-OF":                "DGR_STRUCTURAL",
    # Regulatory (ambiguous direction per DrugProt guidelines)
    "DIRECT-REGULATOR":       "DGR_REGULATE",
}

# BioRED biochemical drug-gene interactions (explicit binding types)
BIORED_DGR_TYPES = {"Bind", "Cotreatment", "Drug_Interaction"}


# ---------------------------------------------------------------------------
# Mapping functions
# ---------------------------------------------------------------------------

def sc0_label(head_type: str, tail_type: str, source_label: str,
              source_dataset: str) -> str:
    """
    SC0: Faithful reproduction of the original S2_current operational mapping.

    Mapping rules (from relation_mapping_matrix.csv, schema_id=S2):
      BioRED Bind / Cotreatment / Drug_Interaction → DRUG_GENE_REGULATION
      ALL other BioRED relations (including DRUG-DISEASE Association/Pos/Neg_Corr)
        → ASSOCIATION_GENERAL  ← DRUG-DISEASE pairs stay in ASSOC_GENERAL
      DrugProt (all 13 types) → DRUG_GENE_REGULATION
      BC5CDR CID → DRUG_DISEASE  ← ONLY BC5CDR contributes DRUG_DISEASE head

    VARIANT_GENE (BioRED Conversion, 4 instances) → ASSOCIATION_GENERAL
    (below N_min=50 threshold; folded into ASSOC_GENERAL for trainability)

    SC0 active heads: ASSOCIATION_GENERAL, DRUG_GENE_REGULATION, DRUG_DISEASE
    """
    if source_dataset == "drugprot":
        return "DRUG_GENE_REGULATION"
    if source_dataset == "bc5cdr":
        return "DRUG_DISEASE"
    # BioRED: ONLY explicit biochemical-interaction labels → DGR; ALL others → ASSOC_GENERAL
    if source_label in BIORED_DGR_TYPES:
        return "DRUG_GENE_REGULATION"
    # Includes Association, Pos_Corr, Neg_Corr, Comparison, Conversion for ALL pair types
    # This means BioRED (DRUG, DISEASE) pairs → ASSOC_GENERAL (not DRUG_DISEASE)
    return "ASSOCIATION_GENERAL"


def sc1_label(head_type: str, tail_type: str, source_label: str,
              source_dataset: str) -> str:
    """
    SC1: Fully entity-pair-type-aware mapping (E1 × M0).
    All four core oncology entity-pair types get specific heads.
    DrugProt and BioRED DRUG-GENE → single DRUG_GENE_REGULATION.
    """
    pair = frozenset([head_type, tail_type])
    if pair == frozenset(["GENE",    "DISEASE"]): return "GENE_DISEASE"
    if pair == frozenset(["VARIANT", "DISEASE"]): return "VARIANT_DISEASE"
    if pair == frozenset(["DRUG",    "DISEASE"]): return "DRUG_DISEASE"
    if pair == frozenset(["DRUG",    "GENE"]):    return "DRUG_GENE_REGULATION"
    if pair == frozenset(["GENE",    "GENE"]):    return "GENE_GENE_ASSOC"
    if pair == frozenset(["DRUG",    "VARIANT"]): return "DRUG_VARIANT_ASSOC"
    return "ASSOCIATION_GENERAL"


def sc3_label(head_type: str, tail_type: str, source_label: str,
              source_dataset: str) -> str:
    """
    SC3: Entity-pair-type-aware + coarse mechanism split (E1 × M1).
    DrugProt DRUG-GENE → 5 functional mechanism groups (following DrugProt guidelines).
    BioRED DRUG-GENE (Bind/Cotreat/Drug_Int and assoc-type) → DRUG_GENE_REGULATION
      (BioRED has insufficient instances for mechanism-specific heads; only DrugProt gets split).
    All other entity-pair pairs same as SC1.

    Note: DGR_INTERACT (BioRED Bind/Cotreat = ~23 instances) is below N_min=50.
    Merged into DRUG_GENE_REGULATION to maintain trainability.
    """
    pair = frozenset([head_type, tail_type])
    if pair == frozenset(["GENE",    "DISEASE"]): return "GENE_DISEASE"
    if pair == frozenset(["VARIANT", "DISEASE"]): return "VARIANT_DISEASE"
    if pair == frozenset(["DRUG",    "DISEASE"]): return "DRUG_DISEASE"
    if pair == frozenset(["GENE",    "GENE"]):    return "GENE_GENE_ASSOC"
    if pair == frozenset(["DRUG",    "VARIANT"]): return "DRUG_VARIANT_ASSOC"
    if pair == frozenset(["DRUG",    "GENE"]):
        if source_dataset == "drugprot":
            return DRUGPROT_TO_M1.get(source_label, "DGR_REGULATE")
        # BioRED drug-gene: all → DRUG_GENE_REGULATION
        # (BioRED Bind/Cotreat = ~23 instances; insufficient for separate DGR_INTERACT head)
        return "DRUG_GENE_REGULATION"
    return "ASSOCIATION_GENERAL"


# ---------------------------------------------------------------------------
# Schema metadata
# ---------------------------------------------------------------------------
SCHEMAS: dict[str, dict] = {
    "S_flat": {
        "name": "S_flat",
        "axes": "E0×M0",
        "description": "Original S2_current mapping (partially entity-type-aware baseline)",
        "label_fn": sc0_label,
        "n_heads_approx": 3,
        "data_suffix": "",          # no suffix — uses existing trn packages
        "notes": (
            "SC0 is NOT 'corpus membership only': BioRED Bind/Cotreatment/Drug_Interaction "
            "are already mapped to DRUG_GENE_REGULATION (entity-type-aware), while "
            "BioRED association-type relations collapse to ASSOCIATION_GENERAL regardless "
            "of entity pair type. This inconsistency is documented."
        ),
    },
    "S_pair": {
        "name": "S_pair",
        "axes": "E1×M0",
        "description": "Fully entity-pair-type-aware; core oncology pairs distinguished",
        "label_fn": sc1_label,
        "n_heads_approx": 6,
        "data_suffix": "_Spair",
        "notes": (
            "SC1 collapses relation polarity within entity-pair types "
            "(Positive/Negative Correlation both → GENE_DISEASE) to maximise "
            "between-type discriminability. No public span-supervised corpus provides "
            "directional labels for gene-disease in oncology."
        ),
    },
    "S_mech": {
        "name": "S_mech",
        "axes": "E1×M1",
        "description": "Entity-pair-type-aware + DrugProt coarse mechanism split (5 groups)",
        "label_fn": sc3_label,
        "n_heads_approx": 10,
        "data_suffix": "_Smech",
        "notes": (
            "Mechanism grouping follows DrugProt annotation guidelines "
            "(Miranda-Escalada et al. 2021): inhibitory, activating, enzymatic/metabolic, "
            "structural/compositional, regulatory. PART-OF is DGR_STRUCTURAL "
            "(compositional), not DGR_METABOLIC (enzymatic)."
        ),
    },
}
