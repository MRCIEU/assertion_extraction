"""
CPU-only: join Tier-1/2, external eval, rerun, decision policy; emit final analysis tables and JSON rules.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .paths import FT_RUNS, MANIFESTS, PROC, PROJECT_ROOT, REPORTS, TABLES, ensure_dirs
from .training_metrics_loader import load_run_metrics


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _ext_map(path: Path) -> Dict[str, Dict[str, str]]:
    """base_id -> {biored_mean, bc5cdr_mean, biored_std, bc5cdr_std}"""
    rows = _read_csv(path)
    out: Dict[str, Dict[str, str]] = {}
    for r in rows:
        bid = r.get("base_experiment_id", "")
        src = r.get("evaluation_source", "")
        if not bid:
            continue
        out.setdefault(bid, {})
        if "biored" in src:
            out[bid]["external_biored_macro_f1_mean"] = r.get("mean_macro_f1", "")
            out[bid]["external_biored_macro_f1_std"] = r.get("std_macro_f1", "")
        if "bc5cdr" in src:
            out[bid]["external_bc5cdr_macro_f1_mean"] = r.get("mean_macro_f1", "")
            out[bid]["external_bc5cdr_macro_f1_std"] = r.get("std_macro_f1", "")
    return out


def run_final_consolidation() -> Dict[str, Any]:
    ensure_dirs()
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    ext_path = PROJECT_ROOT / "external_evaluation" / "reports" / "tables" / "primary_external_results.csv"
    ext = _ext_map(ext_path)

    rerun_path = PROJECT_ROOT / "fine_tuning_experiments" / "reports" / "tables" / "rerun_main_aggregated_results.csv"
    rerun = {r["base_experiment_id"]: r for r in _read_csv(rerun_path)}

    dec_path = PROJECT_ROOT / "report" / "decision_analysis" / "final_model_selection_summary.csv"
    decision = {r["base_experiment_id"]: r for r in _read_csv(dec_path)}
    roles_path = PROJECT_ROOT / "report" / "decision_analysis" / "model_role_assignment.csv"
    model_roles = {r["base_experiment_id"]: r.get("decision_role", "") for r in _read_csv(roles_path)}

    tier1 = _read_csv(TABLES / "training_to_downstream_master_table.csv")
    tier2 = _read_csv(TABLES / "tier2_multiseed_results.csv")
    tier2_by: Dict[Tuple[str, str], Dict[str, str]] = {}
    for r in tier2:
        tier2_by[(r["model_base_id"], r["downstream_setting_id"])] = r

    families_tier2 = sorted({r["model_base_id"] for r in tier2}) if tier2 else []

    joined: List[Dict[str, str]] = []
    wide: List[Dict[str, str]] = []

    bases = sorted({r["model_base_id"] for r in tier1} | set(ext.keys()) | set(rerun.keys()))
    for base in bases:
        t1 = next((x for x in tier1 if x["model_base_id"] == base), {})
        rr = rerun.get(base, {})
        dd = decision.get(base, {})
        ex = ext.get(base, {})
        run_dir = FT_RUNS / f"HR_{base}_s01"
        tm = load_run_metrics(run_dir)

        t2_s1 = tier2_by.get((base, "S1_current_realistic"), {})
        t2_s2 = tier2_by.get((base, "S2_improved_realistic"), {})
        t2_s3 = tier2_by.get((base, "S3_oracle_like"), {})

        row = {
            "base_experiment_id": base,
            "encoder": tm.get("encoder", "") or rr.get("encoder", ""),
            "architecture": tm.get("architecture", "") or rr.get("architecture", ""),
            "schedule": tm.get("schedule_resolved", "") or rr.get("schedule", ""),
            "update_regime": rr.get("update_regime", ""),
            "loss_mode": tm.get("loss_mode", "") or rr.get("loss_mode", ""),
            "internal_hr_mean_macro_f1_rerun": rr.get("mean_macro_f1", ""),
            "internal_hr_std_macro_f1_rerun": rr.get("std_macro_f1", ""),
            "external_biored_macro_f1_mean": ex.get("external_biored_macro_f1_mean", ""),
            "external_biored_macro_f1_std": ex.get("external_biored_macro_f1_std", ""),
            "external_bc5cdr_macro_f1_mean": ex.get("external_bc5cdr_macro_f1_mean", ""),
            "external_bc5cdr_macro_f1_std": ex.get("external_bc5cdr_macro_f1_std", ""),
            "decision_policy_composite_benchmark_heavy": dd.get("composite_benchmark_generalization_heavy", ""),
            "decision_policy_role_before_downstream": model_roles.get(base, ""),
            "tier1_downstream_nn_R1C1_s01": t1.get("downstream_nn_yield_R1C1", ""),
            "tier1_downstream_macro_f1_R1C1": t1.get("downstream_macro_f1_R1C1", ""),
            "tier1_oracle_O3_macro_f1_s01": t1.get("downstream_oracle_O3_macro_f1", ""),
            "tier2_pred_nn_mean_S1": t2_s1.get("pred_nonnegative_mean", ""),
            "tier2_pred_nn_std_S1": t2_s1.get("pred_nonnegative_std", ""),
            "tier2_pred_nn_mean_S2": t2_s2.get("pred_nonnegative_mean", ""),
            "tier2_pred_nn_std_S2": t2_s2.get("pred_nonnegative_std", ""),
            "tier2_oracle_o3_mean_S3": t2_s3.get("macro_f1_heuristic_mean", ""),
            "tier2_oracle_o3_std_S3": t2_s3.get("macro_f1_heuristic_std", ""),
            "final_recommended_downstream_role": "",
            "main_caveat": "",
        }

        # Role + caveat heuristics (evidence-linked, not a single winner)
        caveat = []
        if base == "M015":
            row["final_recommended_downstream_role"] = "conservative_support_finding_default"
            caveat.append("Tier-1 R1/C1 pred_nonnegative=0; maximally conservative under heuristic gold-lite.")
        elif base == "M025":
            row["final_recommended_downstream_role"] = "candidate_surfacing_high_volume"
            caveat.append("High pred_nonnegative under R1/C1; not clinical validation; external BC5CDR below M015 cluster.")
        elif base == "M003":
            row["final_recommended_downstream_role"] = "candidate_surfacing_alternative"
            caveat.append("Second-highest Tier-1 nn yield; internal/external middling vs M015 on benchmarks.")
        elif base == "M026":
            row["final_recommended_downstream_role"] = "diagnostic_weighted_ce_oracle_interest"
            caveat.append("Only nonzero Tier-1 O3 ALL macro-F1 among families; weighted-CE branch; oracle still near-zero.")
        else:
            row["final_recommended_downstream_role"] = "not_in_tier2_shortlist"
            caveat.append("Not selected for Tier-2 multi-seed; see Tier-1 table for yield.")

        row["main_caveat"] = " ".join(caveat)
        joined.append(row)

        w = dict(row)
        w["wide_tier2_S1"] = json.dumps(t2_s1, ensure_ascii=False) if t2_s1 else ""
        w["wide_tier2_S2"] = json.dumps(t2_s2, ensure_ascii=False) if t2_s2 else ""
        w["wide_tier2_S3"] = json.dumps(t2_s3, ensure_ascii=False) if t2_s3 else ""
        wide.append(w)

    if joined:
        with open(TABLES / "final_project_joined_model_table.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(joined[0].keys()))
            w.writeheader()
            w.writerows(joined)
        with open(TABLES / "final_project_joined_model_table_wide.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(wide[0].keys()))
            w.writeheader()
            w.writerows(wide)

    (REPORTS / "final_joined_model_table_note.md").write_text(
        f"# Final joined model table\n\n"
        f"Generated {utc}.\n\n"
        f"**Primary:** `reports/tables/final_project_joined_model_table.csv`\n\n"
        f"Joins: rerun aggregates, external BioRED/BC5CDR (primary_external_results), decision summary, "
        f"Tier-1 master (s01), Tier-2 aggregates where present.\n\n"
        f"Heuristic gold-lite; downstream columns are audit-proxies, not clinical endpoints.\n",
        encoding="utf-8",
    )

    # 5.1 metric transfer
    mrows = []
    for base in bases:
        t1 = next((x for x in tier1 if x["model_base_id"] == base), {})
        rr = rerun.get(base, {})
        exb = _f(ext.get(base, {}).get("external_biored_macro_f1_mean"))
        exc = _f(ext.get(base, {}).get("external_bc5cdr_macro_f1_mean"))
        hr = _f(rr.get("mean_macro_f1"))
        nn = _f(t1.get("downstream_nn_yield_R1C1"))
        mrows.append(
            {
                "base_experiment_id": base,
                "internal_hr_mean_macro_f1": rr.get("mean_macro_f1", ""),
                "internal_hr_seed_std": rr.get("std_macro_f1", ""),
                "external_biored_mean": ext.get(base, {}).get("external_biored_macro_f1_mean", ""),
                "external_bc5cdr_mean": ext.get(base, {}).get("external_bc5cdr_macro_f1_mean", ""),
                "tier1_pred_nonnegative_R1C1": t1.get("downstream_nn_yield_R1C1", ""),
                "spearman_proxy_note": "visual_decoupling: high HR can pair with zero nn",
                "predicts_downstream_nn": "no_single_metric",
            }
        )
    if mrows:
        with open(TABLES / "final_metric_to_downstream_transfer.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(mrows[0].keys()))
            w.writeheader()
            w.writerows(mrows)

    (REPORTS / "final_metric_to_downstream_transfer.md").write_text(
        "# Metric-to-downstream transfer\n\n"
        "**Fails to predict downstream audit yield (Tier-1):** internal HR mean macro-F1 — several families with HR≈0.99 "
        "have **zero** R1/C1 pred_nonnegative.\n\n"
        "**Not sufficient alone:** external BioRED/BC5CDR means — benchmark-favored lines (e.g. M015) can still show "
        "**zero** heuristic downstream nn under conservative surfacing.\n\n"
        "**Table:** `reports/tables/final_metric_to_downstream_transfer.csv`\n",
        encoding="utf-8",
    )

    # 5.2 factor effects — stub from joined
    fact = []
    for j in joined:
        fact.append(
            {
                "encoder": j["encoder"],
                "loss_mode": j["loss_mode"],
                "schedule": j["schedule"],
                "update_regime": j["update_regime"],
                "base_experiment_id": j["base_experiment_id"],
                "tier1_nn_R1C1": j["tier1_downstream_nn_R1C1_s01"],
                "note": "interpret_with_n_equals_families",
            }
        )
    if fact:
        with open(TABLES / "final_factor_effects_downstream.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fact[0].keys()))
            w.writeheader()
            w.writerows(fact)

    (REPORTS / "final_factor_effects_downstream.md").write_text(
        "# Training-strategy effects on downstream\n\n"
        "PubMedBERT vs BioLinkBERT and multitask vs pipeline do **not** align with downstream nn yield on this gold-lite "
        "slice — **same encoder can be high-surfacing or zero-yield** depending on family (e.g. M025 vs M015).\n\n"
        "See `reports/tables/final_factor_effects_downstream.csv`.\n",
        encoding="utf-8",
    )

    # 5.3 setting interaction
    sit = []
    for j in joined:
        if j["base_experiment_id"] not in families_tier2:
            continue
        sit.append(
            {
                "base_experiment_id": j["base_experiment_id"],
                "S1_pred_nn_mean": j["tier2_pred_nn_mean_S1"],
                "S2_pred_nn_mean": j["tier2_pred_nn_mean_S2"],
                "S3_oracle_macro_mean": j["tier2_oracle_o3_mean_S3"],
                "pattern_note": "M025/M003 remain non-zero under S1; oracle O3 remains tiny for all",
            }
        )
    if sit:
        with open(TABLES / "final_setting_interaction_table.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sit[0].keys()))
            w.writeheader()
            w.writerows(sit)

    (REPORTS / "final_setting_interaction_analysis.md").write_text(
        "# Setting interaction\n\n"
        "**S1 (realistic R1/C1)** vs **S2 (R2×C4)** vs **S3 (oracle O3)** — surfacing families retain higher pred_nonnegative "
        "under S1/S2; **oracle macro-F1 stays near-zero** for all under heuristic gold — **do not rank families by O3 alone**.\n\n"
        "`reports/tables/final_setting_interaction_table.csv`\n",
        encoding="utf-8",
    )

    profiles = {
        "version": "v2",
        "generated_utc": utc,
        "profiles": {
            "conservative_support_finding": {
                "primary_family": "M015",
                "rationale": "Zero pred_nonnegative on R1/C1 — minimizes false surfacing under heuristic audit.",
            },
            "candidate_surfacing": {
                "primary_families": ["M025", "M003"],
                "rationale": "Highest Tier-1 nn yields; Tier-2 confirms seed stability band (see tier2 tables).",
            },
            "benchmark_balanced_default": {
                "primary_family": "M015",
                "rationale": "Matches prior decision-analysis default for external composites; downstream audit is a different objective.",
            },
            "diagnostic_weighted_loss_oracle": {
                "primary_family": "M026",
                "rationale": "Weighted CE; only non-zero Tier-1 O3 ALL row among shortlist — still near-zero magnitude.",
            },
        },
        "no_universal_winner": True,
    }
    (PROC / "final_downstream_operating_profiles_v2.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")

    (REPORTS / "final_downstream_operating_profiles_v2.md").write_text(
        "# Downstream operating profiles (v2)\n\n"
        "Split policy: **no single model** maximizes benchmark composites, downstream surfacing, and conservative support "
        "simultaneously. See `data/processed/final_downstream_operating_profiles_v2.json`.\n",
        encoding="utf-8",
    )

    rule = {
        "version": "final_downstream_transfer",
        "generated_utc": utc,
        "objectives": {
            "benchmark_balanced_deployment": {"preferred_family": "M015", "caveat": "Best external composites in policy table; downstream nn=0 on this gold-lite slice."},
            "downstream_audit_conservative_support": {"preferred_family": "M015", "caveat": "Deliberately suppresses candidate assertions."},
            "candidate_surfacing_gap_review": {"preferred_family": "M025", "secondary": "M003", "caveat": "Higher false-surfacing risk; not benchmark-first."},
            "variant_centric_oncology_audit": {"preferred_family": "M026", "caveat": "Diagnostic branch; oracle signal weak."},
        },
        "do_not_use_as_universal_default": ["M005", "M009", "M027", "S003"],
        "reason_do_not_use": "Tier-1: high internal HR but zero pred_nonnegative R1/C1 — misleading if selected by HR alone.",
        "split_policy_not_single_default": True,
        "evidence_artifacts": [
            "reports/tables/final_project_joined_model_table.csv",
            "manifests/tier1_key_findings.json",
            "reports/tables/tier2_multiseed_results.csv",
        ],
    }
    (PROC / "final_downstream_transfer_selection_rule.json").write_text(json.dumps(rule, indent=2), encoding="utf-8")

    roles = []
    for k, v in rule["objectives"].items():
        roles.append({"objective_key": k, "preferred_family_or_policy": json.dumps(v, ensure_ascii=False)})
    roles.append({"objective_key": "avoid_hr_only_selection", "preferred_family_or_policy": "true"})
    roles.append(
        {
            "objective_key": "do_not_default_without_profile",
            "preferred_family_or_policy": ",".join(rule["do_not_use_as_universal_default"]),
        }
    )
    with open(TABLES / "final_downstream_transfer_roles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["objective_key", "preferred_family_or_policy"])
        w.writeheader()
        w.writerows(roles)

    (REPORTS / "final_downstream_transfer_selection_report.md").write_text(
        "# Final downstream transfer selection\n\n"
        "Answers benchmark vs audit vs surfacing objectives explicitly — **split policy**. "
        "Rule file: `data/processed/final_downstream_transfer_selection_rule.json`.\n",
        encoding="utf-8",
    )

    # External baseline bridge
    bridge = []
    for j in joined:
        bridge.append(
            {
                "base_experiment_id": j["base_experiment_id"],
                "benchmark_favored_by_policy": j["decision_policy_composite_benchmark_heavy"],
                "tier1_downstream_nn": j["tier1_downstream_nn_R1C1_s01"],
                "policy_shift_note": "benchmark-first M015 remains valid for composites; downstream audit requires profile split",
            }
        )
    if bridge:
        with open(TABLES / "external_to_downstream_policy_shift.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(bridge[0].keys()))
            w.writeheader()
            w.writerows(bridge)

    (REPORTS / "external_baseline_downstream_bridge.md").write_text(
        "# External baseline → downstream bridge\n\n"
        "**Benchmark-favored lines:** M015, M021, S002 score high on policy composites yet **M015 shows zero** Tier-1 "
        "pred_nonnegative — benchmark-first selection is **not** downstream-audit-optimal.\n\n"
        "**Downstream-useful surfacing:** M025/M003 were **not** benchmark defaults — weighting logic must **explicitly** "
        "allow non-default families for surfacing workflows.\n\n"
        "Table: `reports/tables/external_to_downstream_policy_shift.csv`\n",
        encoding="utf-8",
    )

    _write_tier2_multiseed_report(utc, tier2, families_tier2)
    _write_tier2_verification(utc, tier2)

    return {"joined_rows": len(joined)}


def _write_tier2_multiseed_report(utc: str, tier2: List[Dict[str, str]], families: List[str]) -> None:
    lines = [
        "# Tier-2 multi-seed downstream report",
        "",
        f"*Generated {utc}*",
        "",
        "## Summary",
        "",
        f"Families: **{', '.join(families)}** — five seeds per family × three settings (S1 realistic, S2 improved, S3 oracle-like).",
        "",
        "## Per-setting aggregates",
        "",
        "See `reports/tables/tier2_multiseed_results.csv` and `tier2_seed_stability_table.csv`.",
        "",
        "## Interpretation",
        "",
        "- **S1** proxies operational R1/C1 surfacing.",
        "- **S2** tests lexical-expanded + window context (linkage L2_relaxed).",
        "- **S3** oracle O3 remains a **weak scalar** under heuristic gold — use for diagnostics, not ranking alone.",
        "",
    ]
    (REPORTS / "tier2_multiseed_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_tier2_verification(utc: str, tier2: List[Dict[str, str]]) -> None:
    payload = {
        "verification_utc": utc,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "status": "completed" if tier2 else "pending_gpu_run",
        "tier2_multiseed_results_rows": len(tier2),
        "artifacts": [
            "reports/tables/tier2_multiseed_raw.csv",
            "reports/tables/tier2_multiseed_results.csv",
            "reports/tables/tier2_seed_stability_table.csv",
        ],
    }
    (MANIFESTS / "tier2_run_verification.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

