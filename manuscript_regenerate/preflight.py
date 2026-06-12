"""Pre-flight checks before manuscript regeneration."""

from __future__ import annotations

from pathlib import Path

from .paths import REPO, STEPS, step_paths

REQUIRED_ARTIFACTS: dict[str, list[str]] = {
    "00": [
        "evaluable_target_summary.csv",
        "entity_pair_breakdown.csv",
        "abstract_alignment_summary.csv",
    ],
    "01": [
        "corpus_civic_relevance.csv",
        "corpus_alignment_matrix.csv",
        "pmid_leakage.csv",
        "granularity_ladder.csv",
        "oncology_criteria_agreement.csv",
    ],
    "02": ["frozen_protocol.json"],
    "03": [
        "03_candidate_pool_entity_type_alignment_summary.json",
        "03_candidate_pool_positive_coverage.csv",
        "ranking_baselines.csv",
    ],
    "04": [
        "04_pilot_study_benchmark_vs_kb.csv",
        "04_pilot_study_ranking_metrics.csv",
    ],
    "05": ["quality_gate_results.json", "quality_gate_checks.csv"],
    "10": [
        "sweep/recipe_decision_table.csv",
        "matrix/matrix_encoder_summary.csv",
        "matrix/matrix_per_run.csv",
    ],
    "20": [
        "20_checkpoint_inventory.csv",
        "20_within_seed_paired_changes.csv",
        "20_pair_type_breakdown.csv",
        "20_seed_erosion_distribution.csv",
        "20_gene_disease_subset_breakdown.csv",
    ],
}

STYLE_MODULE = REPO / "shared" / "plot_style.py"


def check_style_module() -> bool:
    ok = STYLE_MODULE.exists()
    print(f"Shared style module: {STYLE_MODULE} {'OK' if ok else 'MISSING'}")
    if ok:
        from shared.plot_style import COLORS, DPI, OKABE_ITO, apply_style

        apply_style()
        print(f"  Palette: Okabe-Ito ({len(OKABE_ITO)} colours)")
        print(f"  Semantic roles: benchmark={COLORS['benchmark']}, kb={COLORS['kb']}, "
              f"gene-drug={COLORS['gene_drug']}, gene-disease={COLORS['gene_disease']}")
        print(f"  DPI={DPI}, sans-serif, no top/right spines, light gridlines")
    return ok


def check_artifacts(steps: list[str] | None = None) -> bool:
    keys = steps or list(REQUIRED_ARTIFACTS.keys())
    all_ok = True
    print("\n=== Pre-flight: source artifacts ===")
    for key in keys:
        out = step_paths(STEPS[key])["outputs"]
        missing = []
        for rel in REQUIRED_ARTIFACTS[key]:
            if not (out / rel).exists():
                missing.append(rel)
        if missing:
            all_ok = False
            print(f"  step {key}: FAIL missing {missing}")
        else:
            print(f"  step {key}: OK ({len(REQUIRED_ARTIFACTS[key])} artifacts)")
    return all_ok
