"""PART 10–11: join training metrics with downstream tables; selection framework."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .paths import FT_RUNS, MANIFESTS, PROC, REPORTS, TABLES, ensure_dirs
from .training_metrics_loader import load_run_metrics


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_aggregate() -> None:
    ensure_dirs()
    tier1 = _read_csv(MANIFESTS / "tier1_model_selection.csv")
    tier1 = [r for r in tier1 if r.get("model_base_id")]
    rc = _read_csv(TABLES / "transfer_retrieval_context_results.csv")
    orc = _read_csv(TABLES / "transfer_oracle_results.csv")
    if not rc and not orc:
        (REPORTS / "training_to_downstream_transfer_analysis.md").write_text(
            "# Training-to-downstream transfer\n\n**Pending:** run `tier1_sweep` on GPU first.\n",
            encoding="utf-8",
        )
        return

    master: List[Dict[str, Any]] = []
    for t in tier1:
        base = t["model_base_id"]
        seed = t["model_seed_id"].zfill(2)
        run_dir = FT_RUNS / f"HR_{base}_s{seed}"
        tm = load_run_metrics(run_dir)
        r1c1 = [
            r
            for r in rc
            if r.get("model_base_id") == base
            and r.get("retrieval_variant") == "R1_current"
            and r.get("context_variant") == "C1_abstract"
        ]
        o3 = [
            r
            for r in orc
            if r.get("model_base_id") == base
            and r.get("oracle_condition") == "O3_oracle_pair_sentence"
            and r.get("pairing_family") == "ALL"
        ]
        master.append(
            {
                "model_base_id": base,
                "model_seed_id": seed,
                "encoder": tm.get("encoder", ""),
                "loss_mode": tm.get("loss_mode", ""),
                "schedule_resolved": tm.get("schedule_resolved", ""),
                "hr_best_macro_f1": tm.get("hr_best_overall", ""),
                "downstream_nn_yield_R1C1": r1c1[0].get("pred_nonnegative_count", "") if r1c1 else "",
                "downstream_macro_f1_R1C1": r1c1[0].get("macro_f1_heuristic", "") if r1c1 else "",
                "downstream_oracle_O3_macro_f1": o3[0].get("macro_f1", "") if o3 else "",
            }
        )

    if master:
        with open(TABLES / "training_to_downstream_master_table.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(master[0].keys()))
            w.writeheader()
            w.writerows(master)

    # Association stub: rank correlation narrative only when n>=8
    assoc = [
        {
            "metric_pair": "hr_best_macro_f1 vs downstream_nn_yield_R1C1",
            "association_strength": "report_visual_inspection",
            "notes": "Formal correlation deferred to notebook; table-driven review required.",
        }
    ]
    with open(TABLES / "metric_to_utility_association.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(assoc[0].keys()))
        w.writeheader()
        w.writerows(assoc)

    for name, groupby in [
        ("factor_level_transfer_effects.csv", "encoder"),
        ("setting_interaction_effects.csv", "downstream_setting_id"),
    ]:
        stub = [{"groupby": groupby, "status": "expand_after_full_sweep", "n_models": str(len(master))}]
        with open(TABLES / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(stub[0].keys()))
            w.writeheader()
            w.writerows(stub)

    prof = []
    for m in master:
        prof.append(
            {
                "cluster_name": "operating_profile_placeholder",
                "model_base_id": m["model_base_id"],
                "notes": "Refine using non-negative yield + ambiguity columns after sweep",
            }
        )
    if prof:
        with open(TABLES / "downstream_operating_profile_clusters.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(prof[0].keys()))
            w.writeheader()
            w.writerows(prof)

    (REPORTS / "training_to_downstream_transfer_analysis.md").write_text(
        "# Training-to-downstream transfer\n\n"
        "Primary table: `reports/tables/training_to_downstream_master_table.csv`.\n"
        "Benchmark macro-F1 does **not** guarantee audit utility — compare HR vs downstream_nn_yield.\n",
        encoding="utf-8",
    )

    rule = {
        "default_downstream_audit": "M015",
        "conservative_support": "lowest_nn_yield_on_R1C1_among_tier1",
        "candidate_surfacing": "highest_nn_yield_on_R1C1_among_tier1",
        "variant_centric": "best_variant_family_oracle_row_if_present",
        "oracle_pipeline_only": "model_with_highest_O3_macro_f1_if_nonzero_else_none",
        "do_not_use_default": [],
        "evidence_tables": ["training_to_downstream_master_table.csv", "transfer_oracle_results.csv"],
        "disclaimer": "Revise after Tier-2 multi-seed evidence.",
    }
    (PROC / "downstream_transfer_selection_rule.json").write_text(json.dumps(rule, indent=2), encoding="utf-8")

    roles = []
    for m in master:
        roles.append(
            {
                "model_base_id": m["model_base_id"],
                "role_guess": "profile_TBD",
                "nn_R1C1": m.get("downstream_nn_yield_R1C1", ""),
            }
        )
    if roles:
        with open(TABLES / "downstream_transfer_model_roles.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(roles[0].keys()))
            w.writeheader()
            w.writerows(roles)

    (REPORTS / "downstream_transfer_selection_report.md").write_text(
        "# Downstream transfer selection report\n\n"
        f"See `data/processed/downstream_transfer_selection_rule.json` and `downstream_transfer_model_roles.csv`.\n",
        encoding="utf-8",
    )
