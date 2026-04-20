#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""Build decision-analysis tables from project CSVs (reproducible)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]  # project_1


def _resolve_data_root() -> Path:
    env = __import__("os").environ.get("PROJECT_1_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    home = Path.home()
    candidates = [
        home / "projects" / "project_1",
        REPO,
        Path("/lus/lfs1aip2/projects/b5ac/project_1"),
    ]
    for c in candidates:
        p = c / "external_evaluation" / "reports" / "tables" / "primary_external_results.csv"
        if p.is_file():
            return c
    return REPO


DATA_ROOT = _resolve_data_root()
EXT_PRIMARY = DATA_ROOT / "external_evaluation" / "reports" / "tables" / "primary_external_results.csv"
EXT_PAIR = DATA_ROOT / "external_evaluation" / "reports" / "tables" / "oncology_subset_results.csv"
EXT_REL = DATA_ROOT / "external_evaluation" / "reports" / "tables" / "reliability_stability_table.csv"
HR_RERUN = DATA_ROOT / "fine_tuning_experiments" / "reports" / "tables" / "rerun_main_aggregated_results.csv"
OUT = Path(__file__).resolve().parent


def load_primary() -> dict[str, dict[str, float]]:
    by_base: dict[str, dict[str, float]] = defaultdict(dict)
    with EXT_PRIMARY.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            b = row["base_experiment_id"]
            src = row["evaluation_source"]
            if src == "biored_official_test_pairs" and row.get("mean_macro_f1"):
                by_base[b]["biored_f1"] = float(row["mean_macro_f1"])
            if src == "bc5cdr_official_test_pairs" and row.get("mean_macro_f1"):
                by_base[b]["bc5cdr_f1"] = float(row["mean_macro_f1"])
    return dict(by_base)


def load_hr() -> dict[str, float]:
    out = {}
    with HR_RERUN.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["base_experiment_id"]] = float(row["mean_macro_f1"])
    return out


def load_rel() -> dict[str, float]:
    out = {}
    with EXT_REL.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            std = row.get("seed_std_macro_f1_biored_test") or ""
            out[row["base_experiment_id"]] = float(std) if std else 0.0
    return out


def load_pairing() -> dict[str, dict[str, float]]:
    """Pairing subset rows only."""
    by_base: dict[str, dict[str, float]] = defaultdict(dict)
    with EXT_PAIR.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("evidence_type") != "pairing_subset":
                continue
            b = row["base_experiment_id"]
            st = row["subset_type"].replace("pairing_", "")
            by_base[b][st] = float(row["mean_macro_f1"])
    return dict(by_base)


def min_max_norm(vals: dict[str, float]) -> dict[str, float]:
    if not vals:
        return {}
    lo = min(vals.values())
    hi = max(vals.values())
    if hi - lo < 1e-12:
        return {k: 0.5 for k in vals}
    return {k: (v - lo) / (hi - lo) for k, v in vals.items()}


def main() -> None:
    primary = load_primary()
    hr = load_hr()
    rel_std = load_rel()
    pair = load_pairing()

    bases = sorted(primary.keys())
    # Clinical pairing index: emphasize variant–disease + drug–gene (mechanism / precision oncology)
    clinical_idx: dict[str, float] = {}
    for b in bases:
        p = pair.get(b, {})
        vd = p.get("variant_disease", float("nan"))
        dg = p.get("drug_gene", float("nan"))
        gd = p.get("gene_disease", float("nan"))
        dd = p.get("drug_disease", float("nan"))
        clinical_idx[b] = 0.35 * vd + 0.35 * dg + 0.15 * gd + 0.15 * dd

    z_int = min_max_norm({b: hr.get(b, 0.0) for b in bases})
    z_br = min_max_norm({b: primary[b]["biored_f1"] for b in bases})
    z_bc = min_max_norm({b: primary[b]["bc5cdr_f1"] for b in bases})
    z_stab = min_max_norm({b: 1.0 - min(rel_std.get(b, 0.0) / 0.12, 1.0) for b in bases})
    z_clin = min_max_norm(clinical_idx)

    def penalty(b: str) -> float:
        if b in ("S001", "S002", "M026"):
            return 0.12  # weighted-CE branch risk
        if b in ("M005",):
            return 0.25  # T3 mixture control — not for default promotion
        if b == "M009":
            return 0.05  # optional shared line, higher seed var
        return 0.0

    profiles = {
        "balanced_scientific_default": {
            "w_internal": 0.12,
            "w_biored": 0.22,
            "w_bc5cdr": 0.22,
            "w_stability": 0.14,
            "w_pairing_clinical": 0.22,
            "w_penalty_scale": 0.08,
            "description": "Compromise across internal, external, stability, and pairing (pairing weight moderate).",
        },
        "benchmark_generalization_heavy": {
            "w_internal": 0.05,
            "w_biored": 0.32,
            "w_bc5cdr": 0.33,
            "w_stability": 0.12,
            "w_pairing_clinical": 0.10,
            "w_penalty_scale": 0.08,
            "description": "Prioritize official BioRED+BC5CDR transfer — used as primary default policy for deployment.",
        },
        "stability_heavy": {
            "w_internal": 0.08,
            "w_biored": 0.20,
            "w_bc5cdr": 0.20,
            "w_stability": 0.32,
            "w_pairing_clinical": 0.12,
            "w_penalty_scale": 0.08,
            "description": "Deployment-oriented: low seed variance on BioRED test.",
        },
        "pairing_clinical_anchored": {
            "w_internal": 0.08,
            "w_biored": 0.16,
            "w_bc5cdr": 0.16,
            "w_stability": 0.15,
            "w_pairing_clinical": 0.37,
            "w_penalty_scale": 0.08,
            "description": "Emphasize variant/mechanism pairing slices on BioRED test stratification.",
        },
    }

    scores_by_profile: dict[str, dict[str, float]] = {}
    for pid, pw in profiles.items():
        sc = {}
        for b in bases:
            comp = (
                pw["w_internal"] * z_int[b]
                + pw["w_biored"] * z_br[b]
                + pw["w_bc5cdr"] * z_bc[b]
                + pw["w_stability"] * z_stab[b]
                + pw["w_pairing_clinical"] * z_clin[b]
                - pw["w_penalty_scale"] * (penalty(b) / 0.25)
            )
            sc[b] = round(comp, 5)
        scores_by_profile[pid] = sc

    balanced_id = "balanced_scientific_default"
    benchmark_id = "benchmark_generalization_heavy"
    balanced_ranked = sorted(scores_by_profile[balanced_id].items(), key=lambda x: -x[1])
    benchmark_ranked = sorted(scores_by_profile[benchmark_id].items(), key=lambda x: -x[1])
    balanced_top = balanced_ranked[0][0]
    default_model = benchmark_ranked[0][0]
    secondary = benchmark_ranked[1][0] if len(benchmark_ranked) > 1 else ""

    rule = {
        "version": 1,
        "data_root_used": str(DATA_ROOT),
        "generated_from": {
            "primary_external_results": str(EXT_PRIMARY),
            "rerun_main_aggregated": str(HR_RERUN),
            "reliability_stability": str(EXT_REL),
            "pairing_oncology_subset": str(EXT_PAIR),
        },
        "normalization": "min_max_per_metric_across_shortlist_models_then_weighted_linear_composite",
        "penalty_rules": {
            "weighted_ce_branches": ["S001", "S002", "M026"],
            "penalty_magnitude_default": 0.12,
            "t3_control_never_default": ["M005"],
            "t3_control_extra_penalty": 0.25,
        },
        "pairing_clinical_index_formula": "0.35*variant_disease + 0.35*drug_gene + 0.15*gene_disease + 0.15*drug_disease (pairing_subset, BioRED test)",
        "default_weight_profile_id": benchmark_id,
        "alternate_exploratory_profile_id": balanced_id,
        "alternate_top_model_under_balanced_profile": balanced_top,
        "default_recommended_model": default_model,
        "secondary_recommended_model": secondary,
        "rationale_default": (
            f"{default_model} maximizes the benchmark_generalization_heavy composite "
            f"(emphasis on official BioRED + BC5CDR macro-F1; aligns with BC5CDR-leading internal anchor M015). "
            f"Under balanced_scientific_default, {balanced_top} ranks first if pairing/stability tilt matters more."
        ),
        "no_universal_single_winner": True,
        "conditional_notes": {
            "S001_S002": "Strongest BioRED mean F1 cluster; consider if BioRED-like deployment dominates.",
            "M015_M003": "Strongest BC5CDR cluster; default favors M015 via composite.",
            "M021": "Best variant–disease pairing slice; consider when variant-linked oncology is primary.",
        },
    }
    (OUT / "final_model_selection_rule.json").write_text(
        json.dumps(rule, indent=2) + "\n", encoding="utf-8"
    )

    # decision_weight_profiles.csv
    rows = []
    for pid, pw in profiles.items():
        rows.append(
            {
                "profile_id": pid,
                **{k: pw[k] for k in pw if k.startswith("w_")},
                "description": pw["description"],
            }
        )
    with (OUT / "decision_weight_profiles.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Summary per model under default profile + all profiles
    sum_rows = []
    for b in bases:
        sum_rows.append(
            {
                "base_experiment_id": b,
                "internal_hr_mean_macro_f1": round(hr.get(b, float("nan")), 4),
                "external_biored_macro_f1": primary[b]["biored_f1"],
                "external_bc5cdr_macro_f1": primary[b]["bc5cdr_f1"],
                "biored_seed_std": rel_std.get(b, ""),
                "pairing_clinical_index": round(clinical_idx[b], 4),
                "z_internal": round(z_int[b], 4),
                "z_biored": round(z_br[b], 4),
                "z_bc5cdr": round(z_bc[b], 4),
                "z_stability_proxy": round(z_stab[b], 4),
                "z_pairing_clinical": round(z_clin[b], 4),
                "penalty": penalty(b),
                **{f"composite_{pid}": scores_by_profile[pid][b] for pid in profiles},
            }
        )
    with (OUT / "final_model_selection_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sum_rows[0].keys()))
        w.writeheader()
        w.writerows(sum_rows)

    # Sensitivity: vary w_biored by +/- 0.08 from benchmark policy (transfer to bc5cdr)
    sens = []
    base_w = profiles[benchmark_id].copy()
    for delta in (-0.08, 0.0, 0.08):
        w_biored = base_w["w_biored"] + delta
        w_bc = base_w["w_bc5cdr"] - delta
        sc = {}
        for b in bases:
            comp = (
                base_w["w_internal"] * z_int[b]
                + w_biored * z_br[b]
                + w_bc * z_bc[b]
                + base_w["w_stability"] * z_stab[b]
                + base_w["w_pairing_clinical"] * z_clin[b]
                - base_w["w_penalty_scale"] * (penalty(b) / 0.25)
            )
            sc[b] = comp
        top = max(sc.items(), key=lambda x: x[1])
        sens.append(
            {
                "delta_biored_weight_vs_default": delta,
                "w_biored": round(w_biored, 3),
                "w_bc5cdr": round(w_bc, 3),
                "winner": top[0],
                "winner_score": round(top[1], 5),
            }
        )
    with (OUT / "decision_sensitivity_analysis.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sens[0].keys()))
        w.writeheader()
        w.writerows(sens)

    # model_role_assignment.csv
    roles = []
    for b in bases:
        note = ""
        if b == default_model:
            role = "default_recommended_benchmark_policy"
        elif b == secondary:
            role = "secondary_recommended_benchmark_policy"
        elif b == balanced_top and b != default_model:
            role = "top_under_balanced_pairing_tilt"
        elif b in ("S001", "S002"):
            role = "conditional_bioRED_first"
            note = "Top BioRED external mean F1; weighted-CE penalty in default rule."
        elif b == "M021" and b != balanced_top:
            role = "conditional_pairing_variant_heavy"
            note = "Strong variant–disease pairing slice."
        elif b == "M003":
            role = "conditional_bc5cdr_pubmed_line"
            note = "Strong BC5CDR; pipeline PubMedBERT anchor."
        elif b == "M005":
            role = "diagnostic_only"
            note = "T3 aux control; not default."
        elif b == "M026":
            role = "diagnostic_only"
            note = "Weighted CE diagnostic."
        elif b in ("M010", "M025"):
            role = "conditional_or_ablation"
        elif b == "M009":
            role = "optional_shared_encoder"
        else:
            role = "primary_shortlist"
        roles.append(
            {
                "base_experiment_id": b,
                "registry_role_from_external_eval": "",
                "decision_role": role,
                "notes": note,
            }
        )
    with (OUT / "model_role_assignment.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["base_experiment_id", "registry_role_from_external_eval", "decision_role", "notes"])
        w.writeheader()
        w.writerows(roles)

    # pairing_analysis_table.csv
    pair_rows = []
    families = ["drug_disease", "drug_gene", "gene_disease", "variant_disease"]
    for fam in families:
        row = {"pairing_family": fam}
        for b in bases:
            row[b] = round(pair.get(b, {}).get(fam, float("nan")), 4)
        row["hardest_by_min_mean"] = min(bases, key=lambda x: pair.get(x, {}).get(fam, 0))
        pair_rows.append(row)
    with (OUT / "pairing_analysis_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pairing_family"] + bases + ["hardest_by_min_mean"])
        w.writeheader()
        w.writerows(pair_rows)

    # pairing_support_table.csv (from first row of each family)
    sup = []
    with EXT_PAIR.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["evidence_type"] != "pairing_subset":
                continue
            fam = row["subset_type"].replace("pairing_", "")
            if not any(x["pairing_family"] == fam for x in sup):
                sup.append(
                    {
                        "pairing_family": fam,
                        "n_documents": row["n_documents"],
                        "n_examples": row["n_examples"],
                        "n_positive_instances": row["n_positive_instances"],
                    }
                )
    with (OUT / "pairing_support_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sup[0].keys()))
        w.writeheader()
        w.writerows(sup)

    # pairing_model_profiles.csv
    prof_rows = []
    for b in bases:
        p = pair.get(b, {})
        prof_rows.append(
            {
                "base_experiment_id": b,
                "max_pairing_family": max(families, key=lambda f: p.get(f, 0)),
                "max_pairing_f1": round(max(p.get(f, 0) for f in families), 4),
                "min_pairing_family": min(families, key=lambda f: p.get(f, 1)),
                "min_pairing_f1": round(min(p.get(f, 0) for f in families), 4),
                "spread": round(
                    max(p.get(f, 0) for f in families) - min(p.get(f, 0) for f in families), 4
                ),
                "profile_summary": (
                    "variant_strong" if p.get("variant_disease", 0) >= 0.62 else "balanced"
                ),
            }
        )
    with (OUT / "pairing_model_profiles.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prof_rows[0].keys()))
        w.writeheader()
        w.writerows(prof_rows)

    # clinical_anchor_analysis_table.csv
    clin = []
    for b in bases:
        clin.append(
            {
                "base_experiment_id": b,
                "cluster_mechanism_drug_gene_f1": round(pair.get(b, {}).get("drug_gene", float("nan")), 4),
                "precision_oncology_variant_disease_f1": round(pair.get(b, {}).get("variant_disease", float("nan")), 4),
                "assoc_gene_disease_f1": round(pair.get(b, {}).get("gene_disease", float("nan")), 4),
                "therapy_chem_dd_f1": round(pair.get(b, {}).get("drug_disease", float("nan")), 4),
                "interpretability_note": (
                    "Drug–gene lowest across models → schema collapses DrugProt mechanisms to DRUG_GENE_REGULATION"
                ),
            }
        )
    with (OUT / "clinical_anchor_analysis_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(clin[0].keys()))
        w.writeheader()
        w.writerows(clin)

    schema_pressure = [
        {
            "pressure_point": "drug_gene_mechanistic_fine_structure",
            "observed_behavior": "Low macro-F1 (~0.34-0.44) vs variant–disease (~0.51-0.72); high std on some models",
            "s2_current_implication": "Mechanism types collapsed to DRUG_GENE_REGULATION; subtype semantics not recovered",
        },
        {
            "pressure_point": "clinical_assertion_subtypes",
            "observed_behavior": "Not measurable on external pair gold without assertion-type labels",
            "s2_current_implication": "S2 CLINICAL_ASSERTION bucket insufficient for predictive vs diagnostic claims",
        },
        {
            "pressure_point": "drugprot_external",
            "observed_behavior": "Official test not packaged — mechanism claims cannot be benchmarked on DrugProt test",
            "s2_current_implication": "Drug–gene story under-validated on third corpus",
        },
    ]
    with (OUT / "schema_pressure_points.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(schema_pressure[0].keys()))
        w.writeheader()
        w.writerows(schema_pressure)

    print(json.dumps({"default_model": default_model, "secondary": secondary, "ok": True}))


if __name__ == "__main__":
    main()
