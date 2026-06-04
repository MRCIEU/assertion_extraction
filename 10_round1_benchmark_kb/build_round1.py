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
    easy_hard_encoder_summary,
    encoder_summary,
    encoder_vs_seed_noise,
    flag_degenerate_runs,
    sensitivity_correlations,
)
from .benchmark_eval import build_biored_test_examples, evaluate_benchmark_f1
from .config import (
    COMPLETE_MARKER,
    FIGURE_DIR,
    MODELS,
    MODEL_BY_ID,
    OUTPUT_DIR,
    PAIR_TYPES,
    RESULTS_DIR,
    SCORES_DIR,
    TRAIN_SEEDS,
    TRAINING_STRATEGY,
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
from .train import checkpoint_path, has_valid_checkpoint, is_complete, result_path, train_model
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
) -> dict | None:
    marker = result_path(spec, seed)
    if marker.exists() and not force:
        return json.loads(marker.read_text(encoding="utf-8"))

    ckpt = checkpoint_path(spec, seed)
    if not ckpt.exists():
        print(f"  skip eval (no checkpoint yet): {spec.model_id} seed={seed}")
        return None
    if not has_valid_checkpoint(spec, seed):
        print(f"  skip eval (stale checkpoint): {spec.model_id} seed={seed}")
        return None

    bench = evaluate_benchmark_f1(spec, seed, test_examples)
    scores = score_checkpoint(spec, seed, candidates, force=force)

    meta_path = ckpt / "10_train_metadata.json"
    train_meta = {}
    if meta_path.exists():
        train_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    kb_rows = metrics_by_pair_type(scores, f"{spec.model_id}_seed_{seed}")
    kb_map = {row["pair_type"]: row for _, row in kb_rows.iterrows()}

    cal = calibration_for_scores(scores, f"{spec.model_id}_seed_{seed}")

    payload = {
        "model_id": spec.model_id,
        "short_name": spec.short_name,
        "seed": seed,
        "run_id": f"{spec.model_id}_seed_{seed}",
        "training_strategy": TRAINING_STRATEGY,
        "best_epoch_val_f1": train_meta.get("best_epoch_val_f1"),
        "best_val_f1": train_meta.get("best_val_f1"),
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
        f"best_epoch={payload.get('best_epoch_val_f1')} "
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

    for spec in specs:
        for seed in seed_list:
            if train:
                if is_complete(spec, seed) and not force_train and not force_eval:
                    print(f"  skip train (complete): {spec.model_id} seed={seed}")
                elif has_valid_checkpoint(spec, seed) and not force_train:
                    print(f"  skip train (checkpoint): {spec.model_id} seed={seed}")
                else:
                    train_model(spec, train_examples, val_examples, seed, force=force_train)

            if eval_models:
                evaluate_one(spec, seed, candidates, pool, test_examples, force=force_eval)

    if analyze:
        run_analysis()


def run_analysis() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nTraining strategy: {TRAINING_STRATEGY}")

    n_markers = sum(
        1
        for spec in MODELS
        for seed in TRAIN_SEEDS
        if result_path(spec, seed).exists()
    )
    n_scores = len(list(SCORES_DIR.glob("*/*.jsonl"))) if SCORES_DIR.exists() else 0
    print(f"Completion markers: {n_markers} / {len(MODELS) * len(TRAIN_SEEDS)}")
    print(f"Score files: {n_scores} / {len(MODELS) * len(TRAIN_SEEDS)}")
    if n_markers < len(MODELS) * len(TRAIN_SEEDS):
        raise SystemExit("Analysis aborted: not all round1_complete.json markers present.")

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
    degenerate = flag_degenerate_runs(per_run)
    if not degenerate.empty:
        degenerate.to_csv(OUTPUT_DIR / "10_degenerate_runs.csv", index=False)
        print(f"WARNING: {len(degenerate)} degenerate run(s) flagged (see 10_degenerate_runs.csv)")
    per_run.to_csv(OUTPUT_DIR / "10_per_run_scores.csv", index=False)

    encoder_df = encoder_summary(per_run)
    encoder_df.to_csv(OUTPUT_DIR / "10_encoder_summary.csv", index=False)

    range_check = benchmark_f1_range_check(encoder_df)
    range_row = {k: v for k, v in range_check.items() if k != "encoder_f1_values"}
    pd.DataFrame([range_row]).to_csv(OUTPUT_DIR / "10_benchmark_f1_range.csv", index=False)
    pd.DataFrame(
        [{"short_name": n, "benchmark_f1_mean": v} for n, v in range_check.get("encoder_f1_values", [])]
    ).to_csv(OUTPUT_DIR / "10_benchmark_f1_by_encoder.csv", index=False)
    print(
        f"\nBenchmark F1 gradient ({range_check['n_encoders']} encoders): "
        f"min={range_check['min_f1']:.3f}, max={range_check['max_f1']:.3f}, "
        f"mean={range_check['mean_f1']:.3f}, median={range_check['median_f1']:.3f}, "
        f"spread={range_check['spread']:.3f}"
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
    for row in corr_rows:
        if row.get("metric") == "spearman":
            print(
                f"  Benchmark vs KB ({row['pair_type']}): "
                f"Spearman={row.get('estimate')} "
                f"[{row.get('ci_lo')}, {row.get('ci_hi')}]"
            )

    ece_corr = benchmark_ece_correlation(encoder_df)
    pd.DataFrame(
        [
            {"metric": "spearman", **ece_corr["spearman"]},
            {"metric": "pearson", **ece_corr["pearson"]},
        ]
    ).to_csv(OUTPUT_DIR / "10_benchmark_ece_correlations.csv", index=False)
    sp = ece_corr.get("spearman", {})
    print(
        f"  Benchmark vs ECE: Spearman={sp.get('estimate')} "
        f"[{sp.get('ci_lo')}, {sp.get('ci_hi')}]"
    )

    noise = encoder_vs_seed_noise(per_run)
    noise.to_csv(OUTPUT_DIR / "10_encoder_vs_seed_noise.csv", index=False)

    sens_rows = sensitivity_correlations(encoder_df, per_run)
    pd.DataFrame(sens_rows).to_csv(OUTPUT_DIR / "10_benchmark_kb_correlations_sensitivity.csv", index=False)

    try:
        scores_df = load_all_scores()
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Analysis aborted: score files required for analyses A/C/D ({exc})"
        ) from exc

    if scores_df.empty:
        raise SystemExit("Analysis aborted: no score files loaded.")

    if len(scores_df["run_id"].unique()) < len(MODELS) * len(TRAIN_SEEDS):
        print(
            f"WARNING: expected {len(MODELS) * len(TRAIN_SEEDS)} score runs, "
            f"got {len(scores_df['run_id'].unique())}"
        )

    subset = subset_ranking_metrics(scores_df, pool)
    dr_subset = distance_ranker_subset_metrics(pool)
    subset_all = pd.concat([subset, dr_subset], ignore_index=True)
    subset_all.to_csv(OUTPUT_DIR / "10_easy_hard_ranking.csv", index=False)
    eh_summary = easy_hard_encoder_summary(subset_all)
    eh_summary.to_csv(OUTPUT_DIR / "10_easy_hard_encoder_summary.csv", index=False)
    n_beats_hard = int(eh_summary[eh_summary["subset"] == "hard"]["beats_distance_ranker"].sum())
    print(
        f"  Easy/hard: {n_beats_hard}/{len(MODELS)} encoders beat distance ranker on hard subset "
        f"(step-03 distance MRR ref ≈ 0.489)"
    )

    prox = score_proximity_correlations(scores_df, pool)
    prox.to_csv(OUTPUT_DIR / "10_distance_score_correlation.csv", index=False)
    print(f"  Mean score–proximity Pearson r: {prox['pearson_r'].mean():.3f}")

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
    print(f"  Figures -> {FIGURE_DIR}")

    write_report(
        per_run=per_run,
        encoder_df=encoder_df,
        range_check=range_check,
        corr_rows=corr_rows,
        ece_corr=ece_corr,
        noise=noise,
        degenerate=degenerate,
        sens_rows=sens_rows,
    )
    print("\n=== Round 1 analysis complete ===")
