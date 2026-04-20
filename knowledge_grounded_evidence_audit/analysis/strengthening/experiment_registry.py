"""Formal enhancement experiment registry (CSV + JSON + family definitions)."""

from __future__ import annotations

import csv
import json
from typing import Any, Dict, List

from .paths import CODE_DESIGN, DESIGN, ensure_dirs

MODELS = ["M015", "M021", "M003", "S002"]
SCOPE = "NSCLC_panel_kb_anchored_audit"

FAMILIES = [
    ("retrieval_ablation", "Retrieval variant comparison on gold-lite (R1–R4)."),
    ("context_ablation", "Context window comparison for same oracle pairs (C1–C5)."),
    ("proposal_expansion", "Proposal space P1–P5 density and recall metrics."),
    ("oracle_upper_bound", "Oracle pair / sentence / joint conditions O1–O4."),
    ("linkage_sensitivity", "Linkage strictness L1–L3 outcome shifts."),
    ("model_operating_profile", "Model × setting matrix for operating profiles."),
]


def _tier(m: str) -> str:
    return {"oracle_upper_bound": "P0", "model_operating_profile": "P0", "context_ablation": "P1"}.get(m, "P2")


def build_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    eid = 0

    def add(
        family: str,
        rid: str,
        ctx: str,
        prop: str,
        link: str,
        oracle: str,
        model: str,
        compute: str,
        tier: str,
        notes: str,
    ) -> None:
        nonlocal eid
        eid += 1
        rows.append(
            {
                "experiment_id": f"EXP_{eid:04d}_{rid}",
                "family": family,
                "retrieval_variant": rid,
                "context_variant": ctx,
                "proposal_variant": prop,
                "linkage_variant": link,
                "oracle_condition": oracle,
                "model_id": model,
                "scope": SCOPE,
                "expected_compute_level": compute,
                "priority_tier": tier,
                "notes": notes,
            }
        )

    for fam, desc in FAMILIES:
        if fam == "retrieval_ablation":
            for rv in ["R1_current", "R2_expanded_lexical", "R3_annotation_assisted", "R4_ternary_query"]:
                add(
                    fam,
                    rv,
                    "C1_abstract_full",
                    "P1_baseline_inventory",
                    "L1_strict",
                    "none",
                    "ALL_SHORTLIST",
                    "low",
                    "P1",
                    desc + f" Variant {rv}.",
                )
        elif fam == "context_ablation":
            for cv in ["C1_abstract", "C2_sentence", "C3_pm1", "C4_window", "C5_pmc_optional_skip"]:
                for mid in MODELS:
                    add(
                        fam,
                        "R1_current",
                        cv,
                        "P1_baseline",
                        "L1_strict",
                        "none",
                        mid,
                        "medium",
                        _tier(fam),
                        "Same oracle head/tail; context string only changes.",
                    )
        elif fam == "proposal_expansion":
            for pv in ["P1_gene_drug", "P2_expanded_binary", "P3_kb_constrained", "P4_sentence_conditioned", "P5_oracle_pair"]:
                add(
                    fam,
                    "R1_current",
                    "C1_abstract",
                    pv,
                    "L1_strict",
                    "O1" if pv == "P5_oracle_pair" else "none",
                    "ALL_SHORTLIST",
                    "medium",
                    "P1",
                    "Proposal recall / volume / density on gold-lite.",
                )
        elif fam == "oracle_upper_bound":
            for oc in ["O1_oracle_pair", "O2_oracle_sentence", "O3_oracle_pair_sentence", "O4_oracle_pair_rich_excerpt"]:
                for mid in MODELS:
                    add(
                        fam,
                        "R1_current",
                        "C2_sentence" if "sentence" in oc else "C1_abstract",
                        "P5_oracle_pair",
                        "L1_strict",
                        oc,
                        mid,
                        "high",
                        "P0",
                        "Upper-bound classification on gold-lite heuristic labels.",
                    )
        elif fam == "linkage_sensitivity":
            for lv in ["L1_strict", "L2_relaxed_semantic", "L3_clinical_grouped"]:
                add(
                    fam,
                    "R1_current",
                    "C1_abstract",
                    "P1_baseline",
                    lv,
                    "none",
                    "ALL_SHORTLIST",
                    "low",
                    "P1",
                    "Re-score assertions with controlled linkage relaxation.",
                )
        else:  # model_operating_profile
            settings = [
                ("baseline", "R1_current", "C1_abstract", "P1_baseline", "L1_strict", "none"),
                ("best_retrieval_proxy", "R2_expanded_lexical", "C1_abstract", "P1_baseline", "L1_strict", "none"),
                ("best_proposal_proxy", "R1_current", "C1_abstract", "P2_expanded_binary", "L1_strict", "none"),
                ("best_context_proxy", "R1_current", "C2_sentence", "P1_baseline", "L1_strict", "none"),
                ("oracle_pair", "R1_current", "C1_abstract", "P5_oracle_pair", "L1_strict", "O1_oracle_pair"),
                ("oracle_joint", "R1_current", "C2_sentence", "P5_oracle_pair", "L1_strict", "O3_oracle_pair_sentence"),
            ]
            for label, rv, cv, pv, lk, oc in settings:
                for mid in MODELS:
                    add(
                        fam,
                        rv,
                        cv,
                        pv,
                        lk,
                        oc,
                        mid,
                        "high",
                        "P0",
                        f"Matrix row: {label}.",
                    )

    return rows


def write_experiment_registry() -> None:
    ensure_dirs()
    DESIGN.mkdir(parents=True, exist_ok=True)
    CODE_DESIGN.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    fields = list(rows[0].keys()) if rows else []
    for dest in (DESIGN, CODE_DESIGN):
        with open(dest / "enhancement_experiment_registry.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    payload = {"experiments": rows, "model_shortlist": MODELS, "scope": SCOPE}
    for dest in (DESIGN, CODE_DESIGN):
        with open(dest / "enhancement_experiment_registry.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    fam_rows = [
        {
            "family_id": fid,
            "description": fdesc,
            "primary_RQ": "RQ-KA1" if "retrieval" in fid else "RQ-KA2" if "oracle" in fid else "RQ-KA3",
        }
        for fid, fdesc in FAMILIES
    ]
    for dest in (DESIGN, CODE_DESIGN):
        with open(dest / "enhancement_family_definition.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fam_rows[0].keys()))
            w.writeheader()
            w.writerows(fam_rows)
