"""PART 1: baseline freeze + model universe + tiering + registry skeleton + settings."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from .paths import CODE_DESIGN, DESIGN, FT_RUNS, MANIFESTS, OUT_ROOT, PROC, PROJECT_ROOT, REPORTS, ensure_dirs
from .training_metrics_loader import load_run_metrics

BASE_FAMILIES = [
    "M003",
    "M004",
    "M005",
    "M009",
    "M010",
    "M015",
    "M021",
    "M025",
    "M026",
    "M027",
    "S001",
    "S002",
    "S003",
]


def _read_json(p: Path) -> Any:
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_downstream_transfer_baseline_freeze() -> None:
    ensure_dirs()
    b1 = _read_json(MANIFESTS / "current_baseline_freeze.json")
    b2 = _read_json(MANIFESTS / "strengthening_run_verification.json")
    bot = _read_json(MANIFESTS / "bottleneck_summary.json")
    freeze = {
        "reference": "strengthening_pass_completed_job_3713429",
        "prior_baseline_freeze": b1,
        "strengthening_verification": b2,
        "bottleneck_attribution_summary": bot,
        "operating_profiles_prior": _read_json(PROC / "final_downstream_operating_profiles.json"),
        "known_anomalies": [
            "M015: zero pred_nonnegative on gold-lite oracle strings under several context variants; conservative classifier behavior.",
            "Oracle aggregated macro-F1 can read 0.0 with heuristic gold + dominant __NEGATIVE__ — interpret with per-model context tables.",
            "Gold-lite: heuristic labels only; human_confirmed_fraction=0.",
        ],
        "scope": "NSCLC panel knowledge-grounded audit (unchanged)",
    }
    (MANIFESTS / "downstream_transfer_baseline_freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    (REPORTS / "downstream_transfer_baseline_freeze.md").write_text(
        "# Downstream transfer — baseline freeze\n\n"
        "This captures the post-strengthening state before the transfer sweep.\n\n"
        "```json\n"
        + json.dumps(freeze, indent=2)[:12000]
        + "\n```\n",
        encoding="utf-8",
    )


def _checkpoint_exists(base: str, seed: int) -> bool:
    sd = f"{seed:02d}"
    p = FT_RUNS / f"HR_{base}_s{sd}" / "checkpoints" / "best.pt"
    return p.is_file()


def write_model_universe() -> None:
    ensure_dirs()
    rows: List[Dict[str, str]] = []
    for base in BASE_FAMILIES:
        run_dir = FT_RUNS / f"HR_{base}_s01"
        m = load_run_metrics(run_dir)
        rows.append(
            {
                "model_base_id": base,
                "checkpoint_tier1_seed": "01",
                "best_pt_present": "yes" if _checkpoint_exists(base, 1) else "no",
                "encoder": str(m.get("encoder", "")),
                "schedule": str(m.get("schedule_resolved", "")),
                "loss_mode": str(m.get("loss_mode", "")),
                "hr_best_macro_f1_proxy": str(m.get("hr_best_overall", "")),
                "external_biored_macro_f1": str(m.get("external_biored_macro_f1", "")),
                "external_bc5cdr_macro_f1": str(m.get("external_bc5cdr_macro_f1", "")),
                "stability_notes": "multi-seed available s01–s05" if _checkpoint_exists(base, 5) else "partial seeds",
                "project_role_guess": "default"
                if base == "M015"
                else ("conditional_weighted_ce" if base == "S002" else "family_variant"),
                "sweep_tier_recommendation": "Tier1",
            }
        )
    with open(MANIFESTS / "downstream_candidate_model_universe.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (REPORTS / "downstream_candidate_model_universe.md").write_text(
        "# Downstream candidate model universe\n\n"
        "One row per **base family** with **s01** metrics proxy from `run_manifest.json` / `metrics_best_checkpoint.json`.\n"
        "External BioRED/BC5CDR macro-F1 must be joined from the external evaluation bundle when available — not duplicated here.\n\n"
        "See `manifests/downstream_candidate_model_universe.csv`.\n",
        encoding="utf-8",
    )


def write_tiering_policy() -> None:
    pol = {
        "tier1_rule": "For each base family in BASE_FAMILIES, run downstream sweep on HR_{base}_s01 if best.pt exists.",
        "tier1_exceptions": "None — all 13 families attempted; families without checkpoint are skipped at runtime with logged reason.",
        "tier2_rule": "After Tier1, select 4–6 families by downstream dispersion × training informativeness × policy relevance; then run s02–s05 for chosen families only.",
        "tier2_status": "pending_tier1_completion",
    }
    (DESIGN / "downstream_model_tiering_policy.json").write_text(json.dumps(pol, indent=2), encoding="utf-8")
    shutil.copy(DESIGN / "downstream_model_tiering_policy.json", CODE_DESIGN / "downstream_model_tiering_policy.json")
    (REPORTS / "downstream_model_tiering_policy.md").write_text(
        "# Downstream model tiering policy\n\n"
        + "\n".join(f"- **{k}:** {v}" for k, v in pol.items()),
        encoding="utf-8",
    )


def write_tier_selection_csvs() -> None:
    ensure_dirs()
    tier1: List[Dict[str, str]] = []
    for base in BASE_FAMILIES:
        if not _checkpoint_exists(base, 1):
            continue
        tier1.append(
            {
                "model_base_id": base,
                "model_seed_id": "01",
                "run_directory": str(FT_RUNS / f"HR_{base}_s01"),
                "tier": "1",
                "selection_rule": "canonical_seed_s01_per_family",
            }
        )
    with open(MANIFESTS / "tier1_model_selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tier1[0].keys()))
        w.writeheader()
        w.writerows(tier1)

    with open(MANIFESTS / "tier2_model_selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model_base_id", "model_seed_id", "tier", "selection_rule", "status"],
        )
        w.writeheader()
        w.writerow(
            {
                "model_base_id": "",
                "model_seed_id": "",
                "tier": "2",
                "selection_rule": "populate_after_tier1_aggregate",
                "status": "pending",
            }
        )


def write_setting_definitions() -> None:
    ensure_dirs()
    defs = {
        "S1_current_realistic": {
            "retrieval": "R1_current_manifest",
            "context": "C1_abstract_full",
            "proposal": "P1_gene_drug_inventory",
            "linkage": "L1_strict",
            "oracle": "none",
            "question": "RQ-DT3 current operational audit utility",
        },
        "S2_improved_realistic": {
            "retrieval": "R2_expanded_lexical_proxy",
            "context": "C4_richer_excerpt_window",
            "proposal": "P2_expanded_binary_families",
            "linkage": "L2_relaxed_semantic",
            "oracle": "none",
            "question": "RQ-DT2 transfer under improved but non-oracle formulation",
        },
        "S3_oracle_like": {
            "retrieval": "R1_cached_pmids",
            "context": "C2_evidence_sentence",
            "proposal": "P5_oracle_pair",
            "linkage": "L1_strict",
            "oracle": "O3_pair_plus_sentence",
            "question": "RQ-DT3 upper-bound utility vs formulation",
        },
    }
    (DESIGN / "downstream_setting_definitions.json").write_text(json.dumps(defs, indent=2), encoding="utf-8")
    shutil.copy(DESIGN / "downstream_setting_definitions.json", CODE_DESIGN / "downstream_setting_definitions.json")

    sm_rows = []
    for sid, d in defs.items():
        sm_rows.append(
            {
                "downstream_setting_id": sid,
                "retrieval": d["retrieval"],
                "context": d["context"],
                "proposal": d["proposal"],
                "linkage": d["linkage"],
                "oracle": d["oracle"],
                "question": d["question"],
            }
        )
    with open(REPORTS / "tables" / "downstream_setting_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sm_rows[0].keys()))
        w.writeheader()
        w.writerows(sm_rows)

    (REPORTS / "downstream_setting_definitions.md").write_text(
        "# Downstream setting definitions\n\n"
        + "\n".join(f"## {k}\n\n```json\n{json.dumps(v, indent=2)}\n```\n" for k, v in defs.items()),
        encoding="utf-8",
    )


def build_sweep_registry_rows() -> List[Dict[str, str]]:
    """Compressed registry: one row per (tier1 model × downstream_setting × sweep_block)."""
    rows: List[Dict[str, str]] = []
    p = MANIFESTS / "tier1_model_selection.csv"
    if not p.is_file():
        return []
    tier1 = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    settings = ["S1_current_realistic", "S2_improved_realistic", "S3_oracle_like"]
    blocks = ["A_retrieval_context", "B_proposal", "C_oracle", "D_linkage"]
    for t in tier1:
        base = t["model_base_id"]
        seed = t["model_seed_id"]
        for sid in settings:
            for blk in blocks:
                rows.append(
                    {
                        "sweep_experiment_id": f"DT_{base}_s{seed}_{sid}_{blk}",
                        "model_base_id": base,
                        "model_seed_id": seed,
                        "model_tier": "1",
                        "encoder": "",
                        "architecture": "bert_sequence_classification",
                        "schedule": "",
                        "update_regime": "adamw_scientific_trainer",
                        "loss_mode": "",
                        "downstream_setting_id": sid,
                        "retrieval_variant": "see_setting",
                        "context_variant": "see_setting",
                        "proposal_variant": "see_setting",
                        "linkage_variant": "see_setting",
                        "oracle_condition": "see_setting",
                        "priority_tier": "P1",
                        "compute_budget_class": "gpu_medium",
                        "sbatch_group": "transfer_tier1",
                        "notes": "Registry driver row; encoder/schedule/loss filled from run_manifest at aggregate time.",
                    }
                )
    return rows


def write_sweep_registry() -> None:
    ensure_dirs()
    write_tier_selection_csvs()
    rows = build_sweep_registry_rows()
    if not rows:
        return
    fields = list(rows[0].keys())
    for dest in (DESIGN, CODE_DESIGN):
        dest.mkdir(parents=True, exist_ok=True)
        with open(dest / "downstream_transfer_sweep_registry.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        with open(dest / "downstream_transfer_sweep_registry.json", "w", encoding="utf-8") as f:
            json.dump({"experiments": rows}, f, indent=2)

    fam = [
        {
            "family_id": "encoder_family",
            "description": "BioLinkBERT vs PubMedBERT vs other encoders in HR runs",
        },
        {
            "family_id": "loss_family",
            "description": "re_ce vs weighted_ce branch (S*)",
        },
        {
            "family_id": "schedule_family",
            "description": "T1_to_T2 vs schedules with T3/T4 exposure",
        },
    ]
    for dest in (DESIGN, CODE_DESIGN):
        with open(dest / "downstream_transfer_family_definitions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fam[0].keys()))
            w.writeheader()
            w.writerows(fam)


def run_metadata_phase() -> None:
    write_downstream_transfer_baseline_freeze()
    write_model_universe()
    write_tiering_policy()
    write_setting_definitions()
    write_sweep_registry()
