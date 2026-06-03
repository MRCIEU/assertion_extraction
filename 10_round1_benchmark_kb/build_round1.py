"""Orchestrate Round 1 training, evaluation, and analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .analysis import (
    benchmark_ece_correlation,
    benchmark_f1_range_check,
    benchmark_kb_correlation,
    encoder_summary,
    encoder_vs_seed_noise,
)
from .benchmark_eval import build_biored_test_examples, evaluate_benchmark_f1
from .config import (
    COMPLETE_MARKER,
    MODELS,
    MODEL_BY_ID,
    OUTPUT_DIR,
    PAIR_TYPES,
    RESULTS_DIR,
    TRAIN_SEEDS,
    ModelSpec,
)
from .distance_analysis import (
    distance_ranker_subset_metrics,
    enrich_with_proximity,
    score_proximity_correlations,
    subset_ranking_metrics,
)
from .figures import generate_all_figures
from .inference import load_all_scores, score_checkpoint
from .metrics_calibration import calibration_baselines, calibration_for_scores, expected_calibration_error
from .metrics_ranking import metrics_by_pair_type, ranking_metrics_for_scores
from .pool_loader import load_primary_candidates
from .report import write_report
from .train import checkpoint_path, is_complete, result_path, train_model
from .train_data import build_train_val_examples


def _write_marker(spec: ModelSpec, seed: int, payload: dict) -> Path:
    path = result_path(spec, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def evaluate_one(
    spec: ModelSpec,
    seed: int,
    candidates: pd.DataFrame,
    pool: pd.DataFrame,
    test_examples: list[dict],
    force: bool = False,
) -> dict:
    marker = result_path(spec, seed)
    if marker.exists() and not force:
        return json.loads(marker.read_text(encoding="utf-8"))

    ckpt = checkpoint_path(spec, seed)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint {ckpt}; run training first.")

    bench = evaluate_benchmark_f1(spec, seed, test_examples)
    scores = score_checkpoint(spec, seed, candidates, force=force)

    kb_rows = metrics_by_pair_type(scores, f"{spec.model_id}_seed_{seed}")
    kb_map = {row["pair_type"]: row for _, row in kb_rows.iterrows()}

    cal = calibration_for_scores(scores, f"{spec.model_id}_seed_{seed}")

    payload = {
        "model_id": spec.model_id,
        "short_name": spec.short_name,
        "seed": seed,
        "run_id": f"{spec.model_id}_seed_{seed}",
        **bench,
        "kb_mrr_gene_drug": float(kb_map.get("gene-drug", {}).get("mrr", 0)),
        "kb_mrr_gene_disease": float(kb_map.get("gene-disease", {}).get("mrr", 0)),
        "kb_mrr_overall": float(kb_rows["mrr"].mean()) if not kb_rows.empty else 0.0,
        "kb_auc_pr_gene_drug": float(kb_map.get("gene-drug", {}).get("auc_pr", 0)),
        "kb_auc_pr_gene_disease": float(kb_map.get("gene-disease", {}).get("auc_pr", 0)),
        "ece": float(cal["ece"]),
    }
    for pt in PAIR_TYPES:
        row = kb_map.get(pt, {})
        for k in (1, 3, 5):
            payload[f"recall_at_{k}_{pt.replace('-', '_')}"] = float(row.get(f"recall_at_{k}", 0))

    path = _write_marker(spec, seed, payload)
    print(
        f"  COMPLETE {spec.model_id} seed={seed}: "
        f"benchmark_f1={payload['benchmark_f1']:.3f} "
        f"KB MRR gd={payload['kb_mrr_gene_drug']:.3f} "
        f"gdisease={payload['kb_mrr_gene_disease']:.3f} "
        f"ECE={payload['ece']:.3f}"
    )
    return payload


def run_matrix(
    train: bool = True,
    eval_models: bool = True,
    analyze: bool = True,
    force_train: bool = False,
    force_eval: bool = False,
    force_train_data: bool = False,
    model_ids: list[str] | None = None,
    seeds: list[int] | None = None,
) -> None:
    specs = MODELS if model_ids is None else [MODEL_BY_ID[m] for m in model_ids]
    seed_list = TRAIN_SEEDS if seeds is None else seeds

    train_examples, val_examples = build_train_val_examples(force=force_train_data)
    test_examples = build_biored_test_examples()
    candidates = load_primary_candidates()
    pool = enrich_with_proximity(candidates)

    if train:
        for spec in specs:
            for seed in seed_list:
                if is_complete(spec, seed) and not force_train and not force_eval:
                    print(f"  skip train (complete): {spec.model_id} seed={seed}")
                    continue
                train_model(spec, train_examples, val_examples, seed, force=force_train)

    if eval_models:
        for spec in specs:
            for seed in seed_list:
                evaluate_one(spec, seed, candidates, pool, test_examples, force=force_eval)

    if analyze:
        run_analysis()


def run_analysis() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = load_primary_candidates()
    pool = enrich_with_proximity(candidates)

    rows: list[dict] = []
    for spec in MODELS:
        for seed in TRAIN_SEEDS:
            marker = result_path(spec, seed)
            if marker.exists():
                rows.append(json.loads(marker.read_text(encoding="utf-8")))
    if not rows:
        print("No completed models for analysis.")
        return

    per_run = pd.DataFrame(rows)
    per_run.to_csv(OUTPUT_DIR / "10_per_run_scores.csv", index=False)

    encoder_df = encoder_summary(per_run)
    encoder_df.to_csv(OUTPUT_DIR / "10_encoder_summary.csv", index=False)

    range_check = benchmark_f1_range_check(encoder_df)
    pd.DataFrame([range_check]).to_csv(OUTPUT_DIR / "10_benchmark_f1_range.csv", index=False)
    print(
        f"\nBenchmark F1 range: {range_check['min_f1']:.3f}–{range_check['max_f1']:.3f} "
        f"(spread {range_check['spread']:.3f})"
    )

    corr_rows = []
    for pt, col in [
        ("gene-drug", "kb_mrr_gene_drug_mean"),
        ("gene-disease", "kb_mrr_gene_disease_mean"),
        ("overall", "kb_mrr_overall_mean"),
    ]:
        if col not in encoder_df.columns:
            continue
        res = benchmark_kb_correlation(encoder_df, col, pt)
        corr_rows.append(
            {
                "pair_type": pt,
                "metric": "spearman",
                "estimate": res["spearman"].get("estimate"),
                "ci_lo": res["spearman"].get("ci_lo"),
                "ci_hi": res["spearman"].get("ci_hi"),
                "n": res["spearman"].get("n"),
            }
        )
        corr_rows.append(
            {
                "pair_type": pt,
                "metric": "pearson",
                "estimate": res["pearson"].get("estimate"),
                "ci_lo": res["pearson"].get("ci_lo"),
                "ci_hi": res["pearson"].get("ci_hi"),
                "n": res["pearson"].get("n"),
            }
        )
        res["rank_flips"].to_csv(OUTPUT_DIR / f"10_rank_flips_{pt.replace('-', '_')}.csv", index=False)

    pd.DataFrame(corr_rows).to_csv(OUTPUT_DIR / "10_benchmark_kb_correlations.csv", index=False)

    ece_corr = benchmark_ece_correlation(encoder_df)
    pd.DataFrame(
        [
            {"metric": "spearman", **ece_corr["spearman"]},
            {"metric": "pearson", **ece_corr["pearson"]},
        ]
    ).to_csv(OUTPUT_DIR / "10_benchmark_ece_correlations.csv", index=False)

    noise = encoder_vs_seed_noise(per_run)
    noise.to_csv(OUTPUT_DIR / "10_encoder_vs_seed_noise.csv", index=False)

    try:
        scores_df = load_all_scores()
    except FileNotFoundError:
        scores_df = pd.DataFrame()

    if not scores_df.empty:
        subset = subset_ranking_metrics(scores_df, pool)
        dr_subset = distance_ranker_subset_metrics(pool)
        subset_all = pd.concat([subset, dr_subset], ignore_index=True)
        subset_all.to_csv(OUTPUT_DIR / "10_easy_hard_ranking.csv", index=False)

        prox = score_proximity_correlations(scores_df, pool)
        prox.to_csv(OUTPUT_DIR / "10_distance_score_correlation.csv", index=False)

        cal_rows = []
        for run_id, sub in scores_df.groupby("run_id"):
            cal_rows.append(calibration_for_scores(sub, run_id))
        cal_df = pd.DataFrame(cal_rows)
        cal_df.to_csv(OUTPUT_DIR / "10_calibration_ece.csv", index=False)

        base_rows = []
        for name, sub in calibration_baselines(candidates).items():
            base_rows.append(calibration_for_scores(sub, name))
        pd.DataFrame(base_rows).to_csv(OUTPUT_DIR / "10_calibration_baselines.csv", index=False)

        kb_detail = []
        for run_id, sub in scores_df.groupby("run_id"):
            for pt, ss in sub.groupby("pair_type"):
                row = ranking_metrics_for_scores(ss, run_id)
                row["run_id"] = run_id
                row["model_id"] = ss["model_id"].iloc[0]
                row["seed"] = int(ss["seed"].iloc[0])
                row["pair_type"] = pt
                kb_detail.append(row)
        pd.DataFrame(kb_detail).to_csv(OUTPUT_DIR / "10_kb_metrics_by_pair_type.csv", index=False)

        generate_all_figures(encoder_df, per_run, subset_all, scores_df, candidates)

    write_report(
        per_run=per_run,
        encoder_df=encoder_df,
        range_check=range_check,
        corr_rows=corr_rows,
        ece_corr=ece_corr,
        noise=noise,
    )
    print("\n=== Round 1 analysis complete ===")
