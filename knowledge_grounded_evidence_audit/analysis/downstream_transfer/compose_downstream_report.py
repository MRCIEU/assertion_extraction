"""PART 13 — main downstream transfer sweep report + summary + integration note."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .paths import OUT_ROOT, REPORTS, ensure_dirs


def compose() -> None:
    ensure_dirs()
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tier2_done = (REPORTS / "tables" / "tier2_multiseed_results.csv").is_file() and (
        REPORTS / "tables" / "tier2_multiseed_results.csv"
    ).stat().st_size > 200
    tier2_sec = (
        "## 10. Tier-2 multi-seed findings\n\n"
        "Complete — see `reports/tier2_multiseed_report.md`, `reports/tables/tier2_multiseed_results.csv`, "
        "`tier2_seed_stability_table.csv`, `manifests/tier2_run_verification.json`.\n\n"
        if tier2_done
        else "## 10. Tier-2 multi-seed findings\n\nPending — submit `scripts/run_transfer_tier2_gpu.sbatch` after `tier2_family_selection_decision.json` is frozen.\n\n"
    )
    final_sec = (
        "## 17. Final downstream transfer dossier\n\n"
        "`reports/downstream_transfer_final_report.md`, `data/processed/final_downstream_transfer_selection_rule.json`, "
        "`reports/tables/final_project_joined_model_table.csv`.\n\n"
        if tier2_done
        else ""
    )
    body = f"""# Downstream transfer sweep for knowledge-grounded oncology evidence auditing

*Generated {utc} — output root: `{OUT_ROOT}`*

## 1. Why benchmark-selected models may fail downstream

Benchmark macro-F1 optimizes **relation extraction** on fixed datasets; audit utility depends on **proposal space**, **linkage**, **context**, and **conservative vs surfacing** behavior — orthogonal axes.

## 2. Experimental objectives

- **RQ-DT1:** Which training metrics predict audit utility?
- **RQ-DT2:** Which training designs transfer under improved settings?
- **RQ-DT3:** Setting interaction (realistic vs oracle-like).
- **RQ-DT4:** Operating profiles vs single winner.

## 3. Model universe and tiering

See `manifests/downstream_candidate_model_universe.csv`, `design/downstream_model_tiering_policy.json`, `manifests/tier1_model_selection.csv`.

## 4. Downstream setting families

See `design/downstream_setting_definitions.json` and `reports/tables/downstream_setting_matrix.csv`.

## 5. Gold-lite evaluation core

See `reports/goldlite_reassessment.md`, `data/processed/goldlite_eval_subsets.csv`.

## 6. Retrieval/context transfer results

`reports/tables/transfer_retrieval_context_*.csv`, `reports/retrieval_context_transfer_analysis.md`.

## 7. Proposal-space transfer results

`reports/tables/transfer_proposal_*.csv`, `reports/proposal_transfer_analysis.md`.

## 8. Oracle upper-bound transfer results

`reports/tables/transfer_oracle_*.csv`, `reports/transfer_oracle_analysis.md`.

## 9. Linkage sensitivity results

`reports/tables/transfer_linkage_*.csv`, `reports/transfer_linkage_analysis.md`.

{tier2_sec}## 11. Training-to-downstream transfer analysis

`reports/tables/training_to_downstream_master_table.csv`, `reports/training_to_downstream_transfer_analysis.md`.

## 12. Revised downstream decision framework

`data/processed/downstream_transfer_selection_rule.json`, `reports/downstream_transfer_selection_report.md`, `downstream_transfer_model_roles.csv`.

## 13. Main scientific insights

Table-driven: compare **HR best macro-F1** vs **downstream_nn_yield_R1C1** and **oracle O3** columns — benchmark leadership does not imply audit leadership.

## 14. Limitations

Heuristic gold-lite; Tier-1 uses **s01** per family; external BioRED/BC5CDR macro-F1 not auto-joined in master table yet.

## 15. Implications for the overall project

Downstream audit policy must be **profile-based**; default benchmark line remains a **research default**, not an operational audit default without transfer evidence.

## 16. Recommended next step

{"Tier-2 complete — maintain joined tables and master report; optional external-eval column refresh if bundle version changes." if tier2_done else "Run Tier-2 multi-seed (`scripts/run_transfer_tier2_gpu.sbatch`); then final consolidation."}

{final_sec}---
"""
    (REPORTS / "downstream_transfer_sweep_report.md").write_text(body, encoding="utf-8")
    (REPORTS / "downstream_transfer_sweep_summary.md").write_text(
        f"""# Downstream transfer sweep — summary

**Generated:** {utc}

## Read first

1. `reports/downstream_transfer_sweep_report.md`
2. `reports/tables/training_to_downstream_master_table.csv`
3. `manifests/downstream_transfer_baseline_freeze.json`

## Execution

Heavy phases: **Slurm GPU only** — Tier-1: `scripts/run_transfer_tier1_gpu.sbatch`; Tier-2 / final: `scripts/run_transfer_tier2_gpu.sbatch`.

---
""",
        encoding="utf-8",
    )
    (OUT_ROOT / "integration_note_downstream_transfer_sweep.md").write_text(
        f"""# Integration note — downstream transfer sweep

## For the master report author

- **Versus strengthening pass:** this sweep **widens the model universe** (13 families × s01 Tier-1) and **joins training metrics** to downstream yields — answering whether benchmark F1 predicts audit utility.
- **Default policy:** may **change** only with evidence in `downstream_transfer_selection_rule.json`; do not assume M015 is best for audit surfacing.
- **Oncology contribution:** explains **when and why** extraction benchmarks misalign with KB-anchored audit — not discovery.

**Generated:** {utc}
---
""",
        encoding="utf-8",
    )

    mirror = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "knowledge_grounded_evidence_audit"
    mirror.mkdir(parents=True, exist_ok=True)
    for name in (
        "downstream_transfer_sweep_report.md",
        "downstream_transfer_sweep_summary.md",
        "downstream_transfer_final_report.md",
        "downstream_transfer_final_summary.md",
        "integration_note_final_downstream_transfer.md",
    ):
        src = REPORTS / name
        if src.is_file():
            (mirror / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    integ = OUT_ROOT / "integration_note_downstream_transfer_sweep.md"
    if integ.is_file():
        (mirror / "integration_note_downstream_transfer_sweep.md").write_text(
            integ.read_text(encoding="utf-8"), encoding="utf-8"
        )

