"""Orchestrate Round 1 training, evaluation, and analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
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
from .figures import generate_publication_figures
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


def _load_per_run_from_disk() -> pd.DataFrame:
    """Load seventy-two runs from stored summary or completion markers (no rescoring)."""
    summary = OUTPUT_DIR / "10_per_run_scores.csv"
    if summary.exists():
        df = pd.read_csv(summary)
        if len(df) >= len(MODELS) * len(TRAIN_SEEDS):
            return df

    rows: list[dict] = []
    for spec in MODELS:
        for seed in TRAIN_SEEDS:
            marker = result_path(spec, seed)
            if not marker.exists():
                raise SystemExit(
                    f"Analysis aborted: missing {marker}. "
                    "Need all 72 completion markers or 10_per_run_scores.csv."
                )
            rows.append(json.loads(marker.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def _load_required_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise SystemExit(f"Analysis aborted: missing required file {path}")
    return pd.read_csv(path)


def run_analysis() -> None:
    """Re-analyse from disk only: no training, no rescoring, no GPU."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nRound 1 re-analysis (stored results only). Training strategy: {TRAINING_STRATEGY}")

    per_run_all = _load_per_run_from_disk()
    per_run_all.to_csv(OUTPUT_DIR / "10_per_run_scores.csv", index=False)

    degenerate = flag_degenerate_runs(per_run_all)
    degenerate.to_csv(OUTPUT_DIR / "10_degenerate_runs.csv", index=False)

    print("\n=== Collapsed seeds (training failures) ===")
    for _, d in degenerate.iterrows():
        print(f"  {d['model_id']} seed={int(d['seed'])}  flags={d.get('flags', '')}")

    print_deberta_kb_audit(per_run_all)

    per_run_clean = filter_clean_runs(per_run_all)
    encoder_primary = encoder_summary_seed_bootstrap(per_run_clean)
    encoder_primary.to_csv(OUTPUT_DIR / "10_encoder_summary.csv", index=False)

    encoder_all = encoder_summary(per_run_all)
    range_check = benchmark_f1_range_check(encoder_primary)
    pd.DataFrame([{k: v for k, v in range_check.items() if k != "encoder_f1_values"}]).to_csv(
        OUTPUT_DIR / "10_benchmark_f1_range.csv", index=False
    )

    variance_primary = variance_components_table(per_run_clean)
    variance_primary.to_csv(OUTPUT_DIR / "10_variance_components.csv", index=False)

    mean_primary = mean_level_correlations(encoder_primary, "primary_clean_seeds")
    mean_sensitivity = mean_level_correlations(encoder_all, "sensitivity_all_seeds")
    pd.concat([mean_primary, mean_sensitivity], ignore_index=True).to_csv(
        OUTPUT_DIR / "10_benchmark_kb_correlations.csv", index=False
    )

    seed_assoc_primary = seed_level_association_table(per_run_clean, "primary_clean_seeds")
    seed_assoc_primary.to_csv(OUTPUT_DIR / "10_benchmark_kb_seed_association.csv", index=False)

    ece_corr_primary = benchmark_ece_correlation(encoder_primary, "primary_clean_seeds")
    ece_corr_all = benchmark_ece_correlation(encoder_all, "sensitivity_all_seeds")
    ece_corr = pd.concat([ece_corr_primary, ece_corr_all], ignore_index=True)
    ece_corr.to_csv(OUTPUT_DIR / "10_benchmark_ece_correlations.csv", index=False)

    sensitivity = collapsed_seed_sensitivity(per_run_all, per_run_clean)
    sensitivity.to_csv(OUTPUT_DIR / "10_collapsed_seed_sensitivity.csv", index=False)

    easy_hard = filter_easy_hard_runs(
        _load_required_csv("10_easy_hard_ranking.csv"), per_run_clean
    )
    eh_summary: dict[str, Any] = {}
    for subset_key, label in [("hard_cross_sentence", "hard"), ("easy_co_sentence", "easy")]:
        sub = easy_hard[easy_hard["subset"] == subset_key]
        dr = sub[sub["model_id"] == "distance_ranker"]["mrr"].iloc[0]
        enc = sub[sub["model_id"] != "distance_ranker"].groupby("model_id")["mrr"].mean()
        eh_summary[label] = {
            "distance_mrr": float(dr),
            "n_beats": int((enc > dr).sum()),
        }

    deberta_all_mean = float(
        encoder_all.loc[encoder_all["model_id"] == "deberta_base", "benchmark_f1_mean"].iloc[0]
    )

    generate_publication_figures(
        encoder_primary,
        easy_hard,
        deberta_all_seeds_mean=deberta_all_mean,
    )
    print(f"Figures (4 PNG) -> {FIGURE_DIR}")

    _print_analysis_stdout(
        degenerate=degenerate,
        variance_primary=variance_primary,
        seed_assoc_primary=seed_assoc_primary,
        mean_primary=mean_primary,
        mean_sensitivity=mean_sensitivity,
        encoder_primary=encoder_primary,
        encoder_all=encoder_all,
        range_check=range_check,
        ece_corr_primary=ece_corr_primary,
        ece_corr_all=ece_corr_all,
    )

    write_report(
        per_run_clean=per_run_clean,
        degenerate=degenerate,
        encoder_primary=encoder_primary,
        range_check=range_check,
        mean_corr_primary=mean_primary,
        mean_corr_sensitivity=mean_sensitivity,
        variance_primary=variance_primary,
        seed_assoc_primary=seed_assoc_primary,
        ece_corr=ece_corr_primary,
        ece_corr_all=ece_corr_all,
        sensitivity=sensitivity,
        easy_hard_summary=eh_summary,
        encoder_all=encoder_all,
    )
    print("\n=== Round 1 re-analysis complete ===")


def _print_analysis_stdout(
    *,
    degenerate: pd.DataFrame,
    variance_primary: pd.DataFrame,
    seed_assoc_primary: pd.DataFrame,
    mean_primary: pd.DataFrame,
    mean_sensitivity: pd.DataFrame,
    encoder_primary: pd.DataFrame,
    encoder_all: pd.DataFrame,
    range_check: dict,
    ece_corr_primary: pd.DataFrame,
    ece_corr_all: pd.DataFrame,
) -> None:
    print("\n=== Variance shares (primary, clean seeds) ===")
    for _, row in variance_primary.iterrows():
        print(
            f"  {row['metric']}: encoder share={row['encoder_variance_share']:.3f} "
            f"seed share={row['seed_variance_share']:.3f} icc={row['icc']:.3f}"
        )

    print("\n=== Seed-level benchmark vs KB (cluster bootstrap, primary) ===")
    for _, row in seed_assoc_primary.iterrows():
        lo, hi = row.get("ci_lo"), row.get("ci_hi")
        ci_txt = f"[{lo:.3f}, {hi:.3f}]" if lo is not None and hi is not None else "[n/a]"
        print(f"  {row['pair_type']}: Spearman={row['spearman']:.3f} {ci_txt}")

    print("\n=== Mean-level correlations (weaker, nine encoder means) ===")
    for label, mdf in [("primary", mean_primary), ("sensitivity_all_seeds", mean_sensitivity)]:
        for pt in ["gene-drug", "gene-disease"]:
            sub = mdf[(mdf["pair_type"] == pt) & (mdf["metric"] == "spearman")]
            if not sub.empty:
                r = sub.iloc[0]
                print(
                    f"  [{label}] {pt}: Spearman={r['estimate']:.3f} "
                    f"[{r.get('ci_lo')}, {r.get('ci_hi')}]"
                )

    others = encoder_primary[encoder_primary["model_id"] != "deberta_base"]
    print(
        f"\nEight-encoder benchmark F1 (primary): "
        f"{others['benchmark_f1_mean'].min():.3f} to {others['benchmark_f1_mean'].max():.3f} "
        f"(spread {range_check['spread']:.3f})"
    )
    deb = encoder_primary[encoder_primary["model_id"] == "deberta_base"]["benchmark_f1_mean"].iloc[0]
    deb_all = encoder_all[encoder_all["model_id"] == "deberta_base"]["benchmark_f1_mean"].iloc[0]
    print(f"DeBERTa benchmark F1 (primary, 6 clean seeds): {deb:.3f}")
    print(f"DeBERTa benchmark F1 (all 8 seeds): {deb_all:.3f}")

    gd = encoder_primary["kb_mrr_gene_drug_mean"]
    print(
        f"\n=== Gene-drug KB MRR range (primary, nine encoder means) ===\n"
        f"  {gd.min():.3f} to {gd.max():.3f} (DeBERTa at {encoder_primary.loc[encoder_primary['model_id']=='deberta_base','kb_mrr_gene_drug_mean'].iloc[0]:.3f})"
    )

    sp = ece_corr_primary[ece_corr_primary["metric"] == "spearman"].iloc[0]
    print(
        f"Benchmark vs ECE (mean-level, primary clean seeds): Spearman={sp['estimate']:.3f} "
        f"[{sp.get('ci_lo')}, {sp.get('ci_hi')}]"
    )
    sp_all = ece_corr_all[ece_corr_all["metric"] == "spearman"].iloc[0]
    print(
        f"Benchmark vs ECE (mean-level, all seeds incl. collapsed): Spearman={sp_all['estimate']:.3f} "
        f"[{sp_all.get('ci_lo')}, {sp_all.get('ci_hi')}]"
    )
