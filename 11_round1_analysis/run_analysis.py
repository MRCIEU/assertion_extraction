"""Orchestrate Round 1 analysis (CPU): reads stage-1 scores only unless --rescore."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .analysis import (
    absolute_kb_levels,
    benchmark_ece_correlation,
    benchmark_f1_range_check,
    collapsed_seed_sensitivity,
    encoder_summary,
    encoder_summary_seed_bootstrap,
    filter_clean_runs,
    filter_easy_hard_runs,
    finetuning_lift_table,
    flag_degenerate_runs,
    load_pool_baselines,
    mean_level_correlations,
    print_deberta_kb_audit,
    seed_level_association_table,
    variance_components_bootstrap_table,
    variance_components_table,
)
from .build_auxiliary import build_all_auxiliary
from .config import (
    ABSOLUTE_KB_CSV,
    EXPECTED_RUNS,
    FIGURE_DIR,
    OUTPUT_DIR,
    PER_RUN_CSV,
    UNTRAINED_LIFT_CSV,
    VARIANCE_BOOTSTRAP_CSV,
)
from .figures import generate_publication_figures
from .report import write_readme, write_report
from .roberta_analysis import roberta_gene_disease_analysis, roberta_report_paragraph
from .score_runs import count_scored_runs, load_scored_summary, score_all_runs
from .score_untrained import count_untrained_scored, load_untrained_summary
from shared.models import MODELS


def _require_scoring_complete() -> None:
    n = count_scored_runs()
    if n < EXPECTED_RUNS:
        raise SystemExit(
            f"KB scoring incomplete: {n}/{EXPECTED_RUNS} scoring_complete markers. "
            "Submit step_score.sbatch or run: python run.py --score-only"
        )


def _require_untrained_complete() -> None:
    n = count_untrained_scored()
    expected = len(MODELS)
    if n < expected:
        raise SystemExit(
            f"Untrained-floor scoring incomplete: {n}/{expected} markers. "
            "Submit step_score_untrained.sbatch or run: python run.py --score-untrained-only"
        )


def _print_saturation_diagnostic(variance: pd.DataFrame, variance_boot: pd.DataFrame) -> None:
    print("\n=== Benchmark discriminative-power / saturation diagnostic ===")
    boot_map = {row["metric"]: row for _, row in variance_boot.iterrows()}
    for key, label in [
        ("benchmark_f1", "Benchmark F1 (in-distribution)"),
        ("kb_mrr_gene_drug", "KB MRR gene-drug (out-of-distribution)"),
        ("kb_mrr_gene_disease", "KB MRR gene-disease (out-of-distribution)"),
    ]:
        row = variance[variance["metric"] == key]
        if row.empty:
            continue
        r = row.iloc[0]
        enc = float(r["encoder_variance_share"])
        seed = float(r["seed_variance_share"])
        ci = ""
        if key in boot_map:
            b = boot_map[key]
            lo, hi = b.get("encoder_share_ci_lo"), b.get("encoder_share_ci_hi")
            if lo is not None and hi is not None:
                ci = f" [encoder share 95% CI {lo:.0%} to {hi:.0%}]"
        print(f"  {label}: between-encoder {enc:.0%}, within-encoder (seed) {seed:.0%}{ci}")


def _print_lift_table(lift: pd.DataFrame) -> None:
    print("\n=== Fine-tuning lift (fine-tuned minus untrained floor) ===")
    print(
        f"{'Encoder':<16} {'Δ bench':>8} {'Δ KB gd':>8} {'Δ KB gdis':>10} "
        f"{'FT bench':>9} {'UT bench':>9}"
    )
    for _, r in lift.sort_values("lift_benchmark_f1", ascending=False).iterrows():
        print(
            f"{r['short_name']:<16} {r['lift_benchmark_f1']:>8.3f} "
            f"{r['lift_kb_mrr_gene_drug']:>8.3f} {r['lift_kb_mrr_gene_disease']:>10.3f} "
            f"{r['finetuned_benchmark_f1']:>9.3f} {r['untrained_benchmark_f1']:>9.3f}"
        )
    print(
        f"  Mean lift: benchmark {lift['lift_benchmark_f1'].mean():.3f}, "
        f"KB gene-drug {lift['lift_kb_mrr_gene_drug'].mean():.3f}, "
        f"KB gene-disease {lift['lift_kb_mrr_gene_disease'].mean():.3f}"
    )


def run_analysis(*, rescore: bool = False, force_score: bool = False) -> None:
    """Full Round 1 analysis from folder-10 matrix checkpoints."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if rescore:
        print("Rescoring KB at best checkpoints (GPU)...")
        per_run_all = score_all_runs(force=force_score)
    else:
        _require_scoring_complete()
        _require_untrained_complete()
        per_run_all = load_scored_summary()
        if per_run_all.empty:
            raise SystemExit("No scoring markers found under data/11_round1_analysis/scores/")

    per_run_all.to_csv(PER_RUN_CSV, index=False)
    print(f"Per-run summary: {len(per_run_all)} scored runs ({count_scored_runs()}/{EXPECTED_RUNS} markers)")

    untrained_df = load_untrained_summary()
    print(f"Untrained-floor baselines: {len(untrained_df)}/{len(MODELS)} encoders")

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
    variance_boot = variance_components_bootstrap_table(per_run_clean)
    variance_boot.to_csv(VARIANCE_BOOTSTRAP_CSV, index=False)
    _print_saturation_diagnostic(variance_primary, variance_boot)

    lift = finetuning_lift_table(encoder_primary, untrained_df)
    lift.to_csv(UNTRAINED_LIFT_CSV, index=False)
    _print_lift_table(lift)

    rand_mrr, dist_mrr = load_pool_baselines()
    abs_kb = absolute_kb_levels(
        encoder_primary,
        easy_hard,
        random_mrr_overall=rand_mrr,
        distance_mrr_overall=dist_mrr,
    )
    abs_kb.to_csv(ABSOLUTE_KB_CSV, index=False)
    ft_row = abs_kb[abs_kb["reference"] == "finetuned_encoders_mean"].iloc[0]
    print(
        f"\n=== Absolute KB levels (encoder means) ===\n"
        f"  Random baseline MRR: {rand_mrr:.3f}\n"
        f"  Distance ranker MRR: {dist_mrr:.3f}\n"
        f"  Fine-tuned mean KB MRR gene-drug: {ft_row['mrr_gene_drug']:.3f}\n"
        f"  Fine-tuned mean KB MRR gene-disease: {ft_row['mrr_gene_disease']:.3f}"
    )

    mean_primary = mean_level_correlations(encoder_primary, "primary_clean_seeds")
    mean_sensitivity = mean_level_correlations(encoder_all, "sensitivity_all_seeds")
    pd.concat([mean_primary, mean_sensitivity], ignore_index=True).to_csv(
        OUTPUT_DIR / "11_benchmark_kb_correlations.csv", index=False
    )

    seed_assoc_primary = seed_level_association_table(per_run_clean, "primary_clean_seeds")
    seed_assoc_primary.to_csv(OUTPUT_DIR / "11_benchmark_kb_seed_association.csv", index=False)
    for _, r in seed_assoc_primary.iterrows():
        print(
            f"Seed-level benchmark–KB Spearman ({r['pair_type']}): {r['spearman']:.3f} "
            f"[{r.get('ci_lo', float('nan')):.3f}, {r.get('ci_hi', float('nan')):.3f}]"
        )

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

    generate_publication_figures(
        encoder_primary,
        easy_hard_clean,
        variance_primary,
        variance_boot=variance_boot,
        lift_df=lift,
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
        variance_boot=variance_boot,
        seed_assoc_primary=seed_assoc_primary,
        ece_corr=ece_corr_primary,
        ece_corr_all=ece_corr_all,
        sensitivity=sensitivity,
        easy_hard_summary=eh_summary,
        abs_kb=abs_kb,
        lift_df=lift,
        pool_size_df=pool_size_df,
        dist_df=dist_df,
        roberta_paragraph=roberta_report_paragraph(roberta),
        rand_mrr=rand_mrr,
        dist_mrr=dist_mrr,
    )
    write_readme(
        per_run_clean=per_run_clean,
        degenerate=degenerate,
        range_check=range_check,
        variance_primary=variance_primary,
        variance_boot=variance_boot,
        seed_assoc_primary=seed_assoc_primary,
        eh_summary=eh_summary,
        lift_df=lift,
        abs_kb=abs_kb,
        roberta=roberta,
    )
    print("\n=== Round 1 analysis complete ===")
