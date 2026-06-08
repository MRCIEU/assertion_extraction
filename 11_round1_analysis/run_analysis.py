"""Orchestrate Round 1 analysis (CPU): reads stage-1 scores only unless --rescore."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .analysis import (
    benchmark_ece_correlation,
    benchmark_f1_range_check,
    collapsed_seed_sensitivity,
    encoder_summary,
    encoder_summary_seed_bootstrap,
    filter_clean_runs,
    filter_easy_hard_runs,
    flag_degenerate_runs,
    mean_level_correlations,
    print_deberta_kb_audit,
    seed_level_association_table,
    variance_components_table,
)
from .build_auxiliary import build_all_auxiliary
from .config import EXPECTED_RUNS, FIGURE_DIR, OUTPUT_DIR, PER_RUN_CSV
from .figures import generate_publication_figures
from .report import write_readme, write_report
from .roberta_analysis import roberta_gene_disease_analysis, roberta_report_paragraph
from .score_runs import count_scored_runs, load_scored_summary, score_all_runs


def _require_scoring_complete() -> None:
    n = count_scored_runs()
    if n < EXPECTED_RUNS:
        raise SystemExit(
            f"KB scoring incomplete: {n}/{EXPECTED_RUNS} scoring_complete markers. "
            "Submit step_score.sbatch or run: python run.py --score-only"
        )


def run_analysis(*, rescore: bool = False, force_score: bool = False) -> None:
    """Full Round 1 analysis from folder-10 matrix checkpoints."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if rescore:
        print("Rescoring KB at best checkpoints (GPU)...")
        per_run_all = score_all_runs(force=force_score)
    else:
        _require_scoring_complete()
        per_run_all = load_scored_summary()
        if per_run_all.empty:
            raise SystemExit("No scoring markers found under data/11_round1_analysis/scores/")

    per_run_all.to_csv(PER_RUN_CSV, index=False)
    print(f"Per-run summary: {len(per_run_all)} scored runs ({count_scored_runs()}/{EXPECTED_RUNS} markers)")

    print_deberta_kb_audit(per_run_all)
    build_all_auxiliary(per_run_all)
    easy_hard = pd.read_csv(OUTPUT_DIR / "11_easy_hard_ranking.csv")

    degenerate = flag_degenerate_runs(per_run_all)
    degenerate.to_csv(OUTPUT_DIR / "11_degenerate_runs.csv", index=False)
    print(f"Degenerate runs flagged: {len(degenerate)}/{len(per_run_all)}")

    per_run_clean = filter_clean_runs(per_run_all)
    encoder_primary = encoder_summary_seed_bootstrap(per_run_clean)
    encoder_primary.to_csv(OUTPUT_DIR / "11_encoder_summary.csv", index=False)

    encoder_all = encoder_summary(per_run_all)
    range_check = benchmark_f1_range_check(encoder_primary)
    pd.DataFrame([{k: v for k, v in range_check.items() if k != "encoder_f1_values"}]).to_csv(
        OUTPUT_DIR / "11_benchmark_f1_range.csv", index=False
    )
    print(
        f"Benchmark F1 encoder-mean range: {range_check['min_f1']:.3f} to {range_check['max_f1']:.3f} "
        f"(spread {range_check['spread']:.3f})"
    )

    variance_primary = variance_components_table(per_run_clean)
    variance_primary.to_csv(OUTPUT_DIR / "11_variance_components.csv", index=False)
    for _, row in variance_primary.iterrows():
        print(
            f"Variance {row['metric']}: encoder {row['encoder_variance_share']:.0%}, "
            f"seed {row['seed_variance_share']:.0%}, ICC {row['icc']:.3f}"
        )

    mean_primary = mean_level_correlations(encoder_primary, "primary_clean_seeds")
    mean_sensitivity = mean_level_correlations(encoder_all, "sensitivity_all_seeds")
    pd.concat([mean_primary, mean_sensitivity], ignore_index=True).to_csv(
        OUTPUT_DIR / "11_benchmark_kb_correlations.csv", index=False
    )

    seed_assoc_primary = seed_level_association_table(per_run_clean, "primary_clean_seeds")
    seed_assoc_primary.to_csv(OUTPUT_DIR / "11_benchmark_kb_seed_association.csv", index=False)

    ece_corr_primary = benchmark_ece_correlation(encoder_primary, "primary_clean_seeds")
    ece_corr_all = benchmark_ece_correlation(encoder_all, "sensitivity_all_seeds")
    pd.concat([ece_corr_primary, ece_corr_all], ignore_index=True).to_csv(
        OUTPUT_DIR / "11_benchmark_ece_correlations.csv", index=False
    )

    sensitivity = collapsed_seed_sensitivity(per_run_all, per_run_clean)
    sensitivity.to_csv(OUTPUT_DIR / "11_collapsed_seed_sensitivity.csv", index=False)

    easy_hard_clean = filter_easy_hard_runs(easy_hard, per_run_clean)
    eh_summary: dict[str, Any] = {}
    for subset_key, label in [("hard_cross_sentence", "hard"), ("easy_co_sentence", "easy")]:
        sub = easy_hard_clean[easy_hard_clean["subset"] == subset_key]
        dr = sub[sub["model_id"] == "distance_ranker"]["mrr"].iloc[0]
        enc = sub[sub["model_id"] != "distance_ranker"].groupby("model_id")["mrr"].mean()
        eh_summary[label] = {"distance_mrr": float(dr), "n_beats": int((enc > dr).sum())}
        print(f"Easy/hard {label}: distance MRR={dr:.3f}, encoders beating distance={eh_summary[label]['n_beats']}/9")

    pool_size_path = OUTPUT_DIR / "11_pool_size_robustness.csv"
    pool_size_df = pd.read_csv(pool_size_path) if pool_size_path.exists() else pd.DataFrame()
    if not pool_size_df.empty:
        for pt in ["gene-drug", "gene-disease"]:
            sub = pool_size_df[pool_size_df["pair_type"] == pt]
            print(
                f"Pool-size vs MRR ({pt}): mean Spearman across runs "
                f"{sub['spearman_r'].mean():.3f} (median {sub['spearman_r'].median():.3f})"
            )

    dist_path = OUTPUT_DIR / "11_distance_score_correlation.csv"
    dist_df = pd.read_csv(dist_path) if dist_path.exists() else pd.DataFrame()

    roberta = roberta_gene_disease_analysis(per_run_all, easy_hard_clean)
    pd.DataFrame([roberta]).to_csv(OUTPUT_DIR / "11_roberta_analysis.csv", index=False)

    deberta_all_mean = (
        float(encoder_all.loc[encoder_all["model_id"] == "deberta_base", "benchmark_f1_mean"].iloc[0])
        if (encoder_all["model_id"] == "deberta_base").any()
        else None
    )

    generate_publication_figures(
        encoder_primary,
        easy_hard_clean,
        deberta_all_seeds_mean=deberta_all_mean,
    )
    print(f"Figures -> {FIGURE_DIR}")

    write_report(
        per_run_clean=per_run_clean,
        degenerate=degenerate,
        encoder_primary=encoder_primary,
        encoder_all=encoder_all,
        range_check=range_check,
        mean_corr_primary=mean_primary,
        mean_corr_sensitivity=mean_sensitivity,
        variance_primary=variance_primary,
        seed_assoc_primary=seed_assoc_primary,
        ece_corr=ece_corr_primary,
        ece_corr_all=ece_corr_all,
        sensitivity=sensitivity,
        easy_hard_summary=eh_summary,
        pool_size_df=pool_size_df,
        dist_df=dist_df,
        roberta_paragraph=roberta_report_paragraph(roberta),
    )
    write_readme(
        per_run_clean=per_run_clean,
        degenerate=degenerate,
        range_check=range_check,
        variance_primary=variance_primary,
        seed_assoc_primary=seed_assoc_primary,
        eh_summary=eh_summary,
        roberta=roberta,
    )
    print("\n=== Round 1 analysis complete ===")
