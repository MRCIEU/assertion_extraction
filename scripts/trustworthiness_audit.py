#!/usr/bin/env python3
"""Trustworthiness audit: compute or extract Block 1–8 diagnostics."""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from shared.constants import NEGATIVES_PER_POSITIVE, SAMPLING_SEED, TRAIN_PAIR_TYPES
from shared.metrics_ranking import compute_mrr, compute_recall_at_k, per_abstract_mrr, ranking_metrics_for_scores
from shared.models import MODELS, MODEL_BY_ID

OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", REPO.parent / "projects" / "project_1")).resolve()
AUDIT_DIR = OUTPUT_ROOT / "outputs" / "trustworthiness_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
REGEN_CUTOFF = datetime(2026, 7, 2, 16, 40, 0, tzinfo=timezone.utc)


def _save(df: pd.DataFrame, name: str) -> Path:
    p = AUDIT_DIR / name
    df.to_csv(p, index=False)
    return p


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def block1_benchmark_counts() -> pd.DataFrame:
    """1.1 from stored matrix_complete (PubMedBERT seed 42 exemplar; counts identical across runs)."""
    p = OUTPUT_ROOT / "data/10_recipe_sweep_and_training/matrix/results/pubmedbert_base/seed_42/matrix_complete.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    rows = [
        {
            "subset": "overall",
            "n_examples": m["benchmark_f1_overall_combined_n"],
            "n_positives": m["benchmark_f1_overall_combined_n_positives"],
            "n_negatives": m["benchmark_f1_overall_combined_n"] - m["benchmark_f1_overall_combined_n_positives"],
            "positive_rate": m["benchmark_f1_overall_combined_n_positives"] / m["benchmark_f1_overall_combined_n"],
            "source_file": str(p),
            "source_field_prefix": "benchmark_f1_overall_combined",
        },
        {
            "subset": "gene-disease",
            "n_examples": m["benchmark_f1_gene_disease_n"],
            "n_positives": m["benchmark_f1_gene_disease_n_positives"],
            "n_negatives": m["benchmark_f1_gene_disease_n"] - m["benchmark_f1_gene_disease_n_positives"],
            "positive_rate": m["benchmark_f1_gene_disease_n_positives"] / m["benchmark_f1_gene_disease_n"],
            "source_file": str(p),
            "source_field_prefix": "benchmark_f1_gene_disease",
        },
        {
            "subset": "gene-drug",
            "n_examples": m["benchmark_f1_gene_drug_biored_n"],
            "n_positives": m["benchmark_f1_gene_drug_biored_n_positives"],
            "n_negatives": m["benchmark_f1_gene_drug_biored_n"] - m["benchmark_f1_gene_drug_biored_n_positives"],
            "positive_rate": m["benchmark_f1_gene_drug_biored_n_positives"] / m["benchmark_f1_gene_drug_biored_n"],
            "source_file": str(p),
            "source_field_prefix": "benchmark_f1_gene_drug_biored",
        },
    ]
    return pd.DataFrame(rows)


def _build_test_examples_offline() -> list[dict]:
    from shared.benchmark_eval import build_biored_test_examples

    return build_biored_test_examples()


def block1_trivial_baselines(examples: list[dict]) -> pd.DataFrame:
    """1.3 trivial F1 baselines under sklearn binary F1 definition."""
    rows = []
    subsets = {
        "overall": examples,
        "gene-disease": [e for e in examples if e.get("pair_type") == "gene-disease"],
        "gene-drug": [e for e in examples if e.get("pair_type") == "gene-drug"],
    }
    rng = np.random.default_rng(42)
    for subset, exs in subsets.items():
        labels = np.array([e["label"] for e in exs], dtype=int)
        n = len(labels)
        prev = float(labels.mean())
        # always positive
        pred_pos = np.ones(n, dtype=int)
        f1_pos = float(f1_score(labels, pred_pos, average="binary", zero_division=0))
        # always negative
        pred_neg = np.zeros(n, dtype=int)
        f1_neg = float(f1_score(labels, pred_neg, average="binary", zero_division=0))
        # random at empirical positive rate, 1000 draws
        f1_rand = []
        for _ in range(1000):
            pred = (rng.random(n) < prev).astype(int)
            f1_rand.append(f1_score(labels, pred, average="binary", zero_division=0))
        rows.append(
            {
                "subset": subset,
                "baseline": "always_positive",
                "f1": f1_pos,
                "predicted_positive_rate": 1.0,
                "n_examples": n,
                "empirical_positive_rate": prev,
            }
        )
        rows.append(
            {
                "subset": subset,
                "baseline": "always_negative",
                "f1": f1_neg,
                "predicted_positive_rate": 0.0,
                "n_examples": n,
                "empirical_positive_rate": prev,
            }
        )
        rows.append(
            {
                "subset": subset,
                "baseline": "random_at_empirical_rate",
                "f1": float(np.mean(f1_rand)),
                "f1_sd": float(np.std(f1_rand, ddof=1)),
                "predicted_positive_rate": prev,
                "n_examples": n,
                "empirical_positive_rate": prev,
            }
        )
    return pd.DataFrame(rows)


def block1_beat_always_positive(enc: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset, bench_col in [
        ("overall", "benchmark_f1_mean"),
        ("gene-disease", "benchmark_f1_gene_disease_mean"),
        ("gene-drug", "benchmark_f1_gene_drug_combined_mean"),
    ]:
        thr = float(baselines.loc[baselines["subset"] == subset, "f1"].iloc[0])
        beats = int((enc[bench_col].astype(float) > thr).sum())
        rows.append(
            {
                "subset": subset,
                "always_positive_f1": thr,
                "n_encoders_beat_baseline": beats,
                "n_encoders": len(enc),
                "encoder_column": bench_col,
                "source_file": str(OUTPUT_ROOT / "outputs/11_round1_analysis/11_encoder_summary.csv"),
            }
        )
    return pd.DataFrame(rows)


def block1_untrained_confusion() -> pd.DataFrame:
    """1.5 from untrained markers: precision/recall -> inferred confusion + predicted-positive rate."""
    rows = []
    for spec in MODELS:
        mp = OUTPUT_ROOT / "data/11_round1_analysis/scores" / f"untrained_{spec.model_id}" / "untrained_scoring_complete.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        n = int(m.get("n_test_examples", 14395))
        n_pos = 8875  # from quality gate / matrix_complete
        n_neg = n - n_pos
        prec = float(m["benchmark_precision"])
        rec = float(m["benchmark_recall"])
        tp = rec * n_pos
        fn = n_pos - tp
        fp = (tp / prec - tp) if prec > 0 else 0.0
        tn = n_neg - fp
        pred_pos_rate = (tp + fp) / n
        rows.append(
            {
                "model_id": spec.model_id,
                "benchmark_f1": float(m["benchmark_f1"]),
                "benchmark_precision": prec,
                "benchmark_recall": rec,
                "predicted_positive_rate": pred_pos_rate,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "collapse_all_positive": pred_pos_rate > 0.95,
                "collapse_all_negative": pred_pos_rate < 0.05,
                "source_file": str(mp),
            }
        )
    return pd.DataFrame(rows)


def block2_prevalence_chain() -> pd.DataFrame:
    train_train = OUTPUT_ROOT / "data/10_recipe_sweep_and_training/cache/train_examples_train.jsonl"
    train_val = OUTPUT_ROOT / "data/10_recipe_sweep_and_training/cache/train_examples_val.jsonl"
    if not train_train.exists():
        train_train = OUTPUT_ROOT / "data/05_marker_quality_gate/train_cache/train_examples_train.jsonl"
        train_val = OUTPUT_ROOT / "data/05_marker_quality_gate/train_cache/train_examples_val.jsonl"

    def rate_from_jsonl(path: Path, corpus_tag: str | None = None) -> dict:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if corpus_tag:
            rows = [r for r in rows if r.get("corpus") == corpus_tag]
        labels = [r["label"] for r in rows]
        return {"n": len(labels), "n_pos": int(sum(labels)), "rate": float(np.mean(labels))}

    pooled_train = rate_from_jsonl(train_train)
    pooled_val = rate_from_jsonl(train_val)
    biored_train = rate_from_jsonl(train_train, "biored")
    drugprot_train = rate_from_jsonl(train_train, "drugprot")

    bench = block1_benchmark_counts()
    pool = pd.read_csv(OUTPUT_ROOT / "outputs/03_candidate_pool/03_candidate_pool_composition.csv")
    primary = pool[pool["scope"] == "primary"]
    pool_n = int(primary["n_candidates"].sum())
    pool_pos = int(primary["n_civic_positives"].sum())

    rows = [
        {"stage": "training_pooled_train_cap24k", "corpus": "BioRED+DrugProt pooled", "n": pooled_train["n"], "n_positives": pooled_train["n_pos"], "positive_rate": pooled_train["rate"], "source": str(train_train)},
        {"stage": "training_pooled_val", "corpus": "BioRED+DrugProt pooled", "n": pooled_val["n"], "n_positives": pooled_val["n_pos"], "positive_rate": pooled_val["rate"], "source": str(train_val)},
        {"stage": "training_biored_train_cap", "corpus": "BioRED only (in pooled train cache)", "n": biored_train["n"], "n_positives": biored_train["n_pos"], "positive_rate": biored_train["rate"], "source": str(train_train)},
        {"stage": "training_drugprot_train_cap", "corpus": "DrugProt only (in pooled train cache)", "n": drugprot_train["n"], "n_positives": drugprot_train["n_pos"], "positive_rate": drugprot_train["rate"], "source": str(train_train)},
    ]
    for _, r in bench.iterrows():
        rows.append(
            {
                "stage": f"benchmark_test_{r['subset']}",
                "corpus": "BioRED test (gene-drug subset = BioRED only)",
                "n": int(r["n_examples"]),
                "n_positives": int(r["n_positives"]),
                "positive_rate": float(r["positive_rate"]),
                "source": r["source_file"],
            }
        )
    for _, r in primary.iterrows():
        rows.append(
            {
                "stage": f"curation_pool_{r['pair_type']}",
                "corpus": "CIViC frozen primary pool",
                "n": int(r["n_candidates"]),
                "n_positives": int(r["n_civic_positives"]),
                "positive_rate": float(r["n_civic_positives"] / r["n_candidates"]),
                "source": str(OUTPUT_ROOT / "outputs/03_candidate_pool/03_candidate_pool_composition.csv"),
            }
        )
    rows.append(
        {
            "stage": "curation_pool_primary_total",
            "corpus": "CIViC frozen primary pool",
            "n": pool_n,
            "n_positives": pool_pos,
            "positive_rate": pool_pos / pool_n,
            "source": str(OUTPUT_ROOT / "outputs/03_candidate_pool/03_candidate_pool_composition.csv"),
        }
    )
    return pd.DataFrame(rows)


def block3_association_decomposition(per_run: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_between = []
    rows_within = []
    rows_per_enc = []
    for pt, kb_col, bench_col in [
        ("gene-drug", "kb_mrr_gene_drug", "benchmark_f1_gene_drug_combined"),
        ("gene-disease", "kb_mrr_gene_disease", "benchmark_f1_gene_disease"),
    ]:
        enc = per_run.groupby("model_id").agg(
            benchmark=(bench_col, "mean"),
            kb=(kb_col, "mean"),
        ).reset_index()
        r_between, _ = spearmanr(enc["benchmark"], enc["kb"])
        rows_between.append({"pair_type": pt, "component": "between_encoder", "spearman": float(r_between), "n": 9, "x_col": bench_col, "y_col": kb_col})

        centred = per_run.copy()
        means = per_run.groupby("model_id")[[bench_col, kb_col]].transform("mean")
        centred["x_c"] = per_run[bench_col] - means[bench_col]
        centred["y_c"] = per_run[kb_col] - means[kb_col]
        r_within, _ = spearmanr(centred["x_c"], centred["y_c"])
        rows_within.append({"pair_type": pt, "component": "within_encoder_pooled", "spearman": float(r_within), "n": 72})

        for mid, g in per_run.groupby("model_id"):
            if g[bench_col].nunique() < 2 or g[kb_col].nunique() < 2:
                rho = np.nan
            else:
                rho, _ = spearmanr(g[bench_col], g[kb_col])
            rows_per_enc.append({"pair_type": pt, "model_id": mid, "within_encoder_spearman": float(rho) if not np.isnan(rho) else None, "n_seeds": len(g)})

    return pd.DataFrame(rows_between + rows_within), pd.DataFrame(rows_per_enc)


def block3_regenerate_association(per_run: pd.DataFrame) -> pd.DataFrame:
    """3.5 regenerate association with both x-column definitions."""
    from importlib import import_module

    analysis = import_module("11_round1_analysis.analysis")
    rows = []
    for pt, kb_col in [("gene-drug", "kb_mrr_gene_drug"), ("gene-disease", "kb_mrr_gene_disease")]:
        for x_col, label in [
            ("benchmark_f1", "pooled_benchmark_f1"),
            ("benchmark_f1_gene_drug_combined", "pair_matched_gene_drug_combined"),
            ("benchmark_f1_gene_disease", "pair_matched_gene_disease"),
        ]:
            if pt == "gene-drug" and x_col == "benchmark_f1_gene_disease":
                continue
            if pt == "gene-disease" and x_col == "benchmark_f1_gene_drug_combined":
                continue
            sub = per_run.copy()
            sub["benchmark_f1"] = sub[x_col]  # cluster_bootstrap uses benchmark_f1 column
            boot = analysis.cluster_bootstrap_benchmark_kb(sub, kb_col, pt)
            enc = per_run.groupby("model_id").agg(x=(x_col, "mean"), y=(kb_col, "mean")).reset_index()
            r_mean, _ = spearmanr(enc["x"], enc["y"])
            rows.append(
                {
                    "pair_type": pt,
                    "x_definition": label,
                    "x_column": x_col,
                    "seed_level_spearman": boot["spearman"],
                    "seed_level_ci_lo": boot["ci_lo"],
                    "seed_level_ci_hi": boot["ci_hi"],
                    "encoder_mean_spearman": float(r_mean),
                    "n_runs": boot["n_runs"],
                }
            )
    return pd.DataFrame(rows)


def block3_icc_disattenuated(var_df: pd.DataFrame, enc_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, kb_col in [
        ("benchmark_f1", "benchmark_f1_mean"),
        ("kb_mrr_gene_drug", "kb_mrr_gene_drug_mean"),
        ("kb_mrr_gene_disease", "kb_mrr_gene_disease_mean"),
    ]:
        vrow = var_df[var_df["metric"] == metric]
        if vrow.empty:
            continue
        icc = float(vrow["icc"].iloc[0])
        x = enc_df["benchmark_f1_mean"].astype(float).values if metric == "benchmark_f1" else enc_df[kb_col].astype(float).values
        y_col = "kb_mrr_gene_drug_mean" if "gene_drug" in metric else ("kb_mrr_gene_disease_mean" if "gene_disease" in metric else None)
        if y_col is None:
            rows.append({"metric": metric, "icc_seed_reliability": icc, "source": "11_variance_components.csv"})
            continue
        y = enc_df[y_col].astype(float).values
        r_obs, _ = spearmanr(enc_df["benchmark_f1_mean"], enc_df[y_col])
        r_dis = float(r_obs / np.sqrt(icc * icc)) if icc > 0 else np.nan
        rows.append(
            {
                "metric": metric,
                "icc_seed_reliability": icc,
                "observed_spearman_vs_other_axis": float(r_obs) if metric == "benchmark_f1" else None,
                "disattenuated_spearman": r_dis if metric == "benchmark_f1" else None,
                "source": "11_variance_components.csv + 11_encoder_summary.csv",
            }
        )
    # cross-axis disattenuated
    bench_icc = float(var_df.loc[var_df["metric"] == "benchmark_f1", "icc"].iloc[0])
    gd_icc = float(var_df.loc[var_df["metric"] == "kb_mrr_gene_drug", "icc"].iloc[0])
    gdis_icc = float(var_df.loc[var_df["metric"] == "kb_mrr_gene_disease", "icc"].iloc[0])
    for pt, kb_col, icc_y in [("gene-drug", "kb_mrr_gene_drug_mean", gd_icc), ("gene-disease", "kb_mrr_gene_disease_mean", gdis_icc)]:
        r, _ = spearmanr(enc_df["benchmark_f1_mean"], enc_df[kb_col])
        r_dis = r / np.sqrt(bench_icc * icc_y) if bench_icc > 0 and icc_y > 0 else np.nan
        rows.append(
            {
                "pair_type": pt,
                "observed_encoder_mean_spearman": float(r),
                "benchmark_icc": bench_icc,
                "kb_icc": icc_y,
                "disattenuated_spearman": float(r_dis),
                "source": "11_variance_components.csv",
            }
        )
    return pd.DataFrame(rows)


def block4_tradeoff(paired: pd.DataFrame) -> pd.DataFrame:
    p = paired[paired["pairable_val_f1_best"]].copy()
    n = len(p)
    bench_up = int((p["delta_benchmark_f1_val_f1_best"] > 0).sum())
    kb_hard_down = int((p["delta_kb_mrr_hard_val_f1_best"] < 0).sum())
    erosion = int(p["erosion_pattern_val_f1_best"].sum())
    p_bench = bench_up / n
    p_kb = kb_hard_down / n
    expected_joint = p_bench * p_kb
    obs_joint = erosion / n
    rows = [
        {"quantity": "n_pairable_seeds", "value": n, "source": "20_within_seed_paired_changes.csv"},
        {"quantity": "delta_benchmark_f1_positive", "value": bench_up, "fraction": p_bench},
        {"quantity": "delta_kb_hard_negative", "value": kb_hard_down, "fraction": p_kb},
        {"quantity": "erosion_pattern_bench_up_kb_hard_down", "value": erosion, "fraction": obs_joint},
        {"quantity": "expected_joint_under_independence", "value": expected_joint * n, "fraction": expected_joint},
        {"quantity": "enrichment_ratio_obs_over_expected", "value": obs_joint / expected_joint if expected_joint > 0 else np.nan},
    ]
    # within-seed correlations
    for ycol, label in [
        ("delta_kb_mrr_hard_val_f1_best", "kb_hard"),
        ("delta_kb_mrr_gene_disease_val_f1_best", "kb_gene_disease"),
        ("delta_kb_mrr_gene_drug_val_f1_best", "kb_gene_drug"),
    ]:
        sp, _ = spearmanr(p["delta_benchmark_f1_val_f1_best"], p[ycol])
        pe, _ = pearsonr(p["delta_benchmark_f1_val_f1_best"], p[ycol])
        rows.append({"quantity": f"within_seed_spearman_delta_bench_vs_{label}", "value": float(sp)})
        rows.append({"quantity": f"within_seed_pearson_delta_bench_vs_{label}", "value": float(pe)})

    # encoder-level negatives
    enc = p.groupby("model_id").agg(
        mean_delta_gdis=("delta_kb_mrr_gene_disease_val_f1_best", "mean"),
        mean_delta_gdis_hard=("delta_kb_mrr_hard_val_f1_best", "mean"),
    )
    rows.append({"quantity": "encoders_negative_mean_delta_gene_disease", "value": int((enc["mean_delta_gdis"] < 0).sum()), "denominator": 9})
    rows.append({"quantity": "encoders_negative_mean_delta_gene_disease_hard", "value": int((enc["mean_delta_gdis_hard"] < 0).sum()), "denominator": 9})
    return pd.DataFrame(rows)


def _per_abstract_map(df: pd.DataFrame) -> float:
    maps = []
    for _, g in df.groupby("pmid"):
        y = g["label_civic_curated_positive"].astype(int).values
        if y.sum() == 0:
            continue
        maps.append(average_precision_score(y, g["score"].values))
    return float(np.mean(maps)) if maps else 0.0


def block5_kb_metrics() -> pd.DataFrame:
    rows = []
    baselines = pd.read_csv(OUTPUT_ROOT / "outputs/03_candidate_pool/ranking_baselines.csv")
    for _, r in baselines.iterrows():
        rows.append(
            {
                "reference": r["baseline"],
                "mrr": r["mrr"],
                "map_macro": np.nan,
                "auc_pr_global": r.get("auc_pr", np.nan),
                "recall_at_1": r.get("recall_at_1", np.nan),
                "recall_at_3": r.get("recall_at_3", np.nan),
                "recall_at_5": r.get("recall_at_5", np.nan),
                "source": "ranking_baselines.csv",
            }
        )
    # fine-tuned + untrained from jsonl
    for spec in MODELS:
        for kind, path in [
            ("finetuned", OUTPUT_ROOT / "data/11_round1_analysis/scores" / spec.model_id),
            ("untrained", OUTPUT_ROOT / "data/11_round1_analysis/scores" / f"untrained_{spec.model_id}"),
        ]:
            if kind == "finetuned":
                # use seed 42 representative or mean across seeds - use all seeds averaged scores? use seed42
                jl = path / "seed_42.jsonl"
                if not jl.exists():
                    continue
            else:
                jl = path / "scores.jsonl"
                if not jl.exists():
                    continue
            df = _load_jsonl(jl)
            m = ranking_metrics_for_scores(df, f"{kind}_{spec.model_id}")
            m["map_macro"] = _per_abstract_map(df)
            m["reference"] = f"{kind}_{spec.model_id}"
            m["source"] = str(jl)
            rows.append(m)
    return pd.DataFrame(rows)


def block5_mrr_denominator(candidates: pd.DataFrame) -> pd.DataFrame:
    pos = candidates[candidates["label_civic_curated_positive"]]
    per_pmid = pos.groupby("pmid").size()
    rows = [
        {"metric": "abstracts_with_any_positive", "value": pos["pmid"].nunique()},
        {"metric": "abstracts_with_multiple_positives", "value": int((per_pmid > 1).sum())},
        {"metric": "max_positives_per_abstract", "value": int(per_pmid.max())},
        {"metric": "distinct_frozen_targets", "value": int(len(pd.read_csv(OUTPUT_ROOT / "outputs/02_evaluation_protocol/ranking_targets.csv")))},
        {"metric": "distinct_positive_candidates_in_pool", "value": int(pos.groupby(["pmid", "head_entity", "tail_entity", "pair_type"]).ngroups)},
    ]
    return pd.DataFrame(rows)


def block6_error_base_rates() -> pd.DataFrame:
    cases = pd.read_csv(OUTPUT_ROOT / "outputs/20_round2_diagnostic/20_qualitative_error_cases.csv")
    pos = cases[cases["case_type"] == "missed_positive"]
    genuine = pos[pos["error_class"] != "abstract_unsupported"] if "error_class" in pos.columns else pos
    curated_pos = pd.read_csv(OUTPUT_ROOT / "outputs/03_candidate_pool/03_candidate_pool_pubtator_recall_classification.csv")
    # base rates in all curated positives in pool - use positive cases from qualitative pool
    pool_pos = cases  # all cases are from eval
    def rate(col, val):
        return float((pool_pos[col] == val).mean())
    rows = []
    for feat, val in [
        ("subset", "hard_cross_sentence"),
        ("head_multiword", True),
        ("tail_multiword", True),
        ("publication_year", None),
        ("pair_type", "gene-disease"),
    ]:
        if feat == "publication_year":
            base = float((pool_pos["publication_year"] < 2010).mean())
            name = "publication_year_before_2010"
        else:
            base = rate(feat, val)
            name = f"{feat}={val}"
        err = float((genuine[feat] == val).mean()) if feat != "publication_year" else float((genuine["publication_year"] < 2010).mean())
        rows.append({"feature": name, "base_rate_all_cases": base, "rate_in_genuine_errors": err, "n_genuine_errors": len(genuine)})
    return pd.DataFrame(rows)


def block6_slide_pmids() -> pd.DataFrame:
    slide_pmids = ["17470858", "26724472", "9635567", "11070098", "10485475", "10866302", "15118125"]
    cases = pd.read_csv(OUTPUT_ROOT / "outputs/20_round2_diagnostic/20_qualitative_error_cases.csv")
    flagged = pd.read_csv(OUTPUT_ROOT / "outputs/20_round2_diagnostic/20_qualitative_errors_flagged_manual.csv")
    rows = []
    for pmid in slide_pmids:
        sub = cases[cases["pmid"].astype(str) == str(pmid)]
        fl = flagged[flagged["pmid"].astype(str) == str(pmid)]
        rows.append(
            {
                "pmid": pmid,
                "in_qualitative_cases": len(sub) > 0,
                "in_flagged_manual": len(fl) > 0,
                "n_cases": len(sub),
                "ranks": ",".join(str(int(r)) for r in sub["rank_in_pool"].tolist()) if len(sub) else "",
                "entities": "; ".join(f"{r.head_entity}/{r.tail_entity}" for _, r in sub.iterrows()) if len(sub) else "",
            }
        )
    # BCOR search
    bcor = cases[cases["head_entity"].astype(str).str.contains("BCOR", case=False, na=False) | cases["tail_entity"].astype(str).str.contains("BCOR", case=False, na=False)]
    for _, r in bcor.iterrows():
        rows.append({"pmid": r["pmid"], "in_qualitative_cases": True, "in_flagged_manual": False, "n_cases": 1, "ranks": str(int(r["rank_in_pool"])), "entities": f"{r.head_entity}/{r.tail_entity}"})
    return pd.DataFrame(rows)


def block7_oncology() -> pd.DataFrame:
    onc = pd.read_csv(OUTPUT_ROOT / "outputs/01_corpus_relevance/oncology_criteria_agreement.csv")
    return onc


def block8_stale_csvs() -> pd.DataFrame:
    rows = []
    for p in sorted((OUTPUT_ROOT / "outputs").rglob("*.csv")):
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        if mtime < REGEN_CUTOFF:
            rows.append({"path": str(p), "mtime_utc": mtime.isoformat(), "stale_vs_2026-07-02_batch": True})
    return pd.DataFrame(rows)


def block8_environment() -> Path | None:
    try:
        import subprocess

        out = subprocess.run(["conda", "env", "export", "-n", "hf-hpc"], capture_output=True, text=True, check=True)
        p = AUDIT_DIR / "environment_hf-hpc.yml"
        p.write_text(out.stdout, encoding="utf-8")
        req = AUDIT_DIR / "requirements_hf-hpc.txt"
        pkgs = [line.split("=")[0] for line in out.stdout.splitlines() if line.startswith("  - ") and not line.startswith("  - pip:")]
        req.write_text("\n".join(pkgs) + "\n", encoding="utf-8")
        return p
    except Exception as exc:
        (AUDIT_DIR / "environment_export_error.txt").write_text(str(exc))
        return None


def main() -> None:
    print("Trustworthiness audit ->", AUDIT_DIR)
    # Block 1
    counts = block1_benchmark_counts()
    _save(counts, "block1_1_benchmark_test_counts.csv")
    print("1.1 counts done")

    try:
        examples = _build_test_examples_offline()
        baselines = block1_trivial_baselines(examples)
        _save(baselines, "block1_3_trivial_baselines.csv")
        print("1.3 baselines done", len(examples), "examples")
    except Exception as exc:
        (AUDIT_DIR / "block1_3_error.txt").write_text(str(exc))

    enc = pd.read_csv(OUTPUT_ROOT / "outputs/11_round1_analysis/11_encoder_summary.csv")
    if (AUDIT_DIR / "block1_3_trivial_baselines.csv").exists():
        baselines = pd.read_csv(AUDIT_DIR / "block1_3_trivial_baselines.csv")
        beat = block1_beat_always_positive(enc, baselines)
        _save(beat, "block1_4_beat_always_positive.csv")

    unt = block1_untrained_confusion()
    _save(unt, "block1_5_untrained_confusion_inferred.csv")

    # Block 2
    prev = block2_prevalence_chain()
    _save(prev, "block2_prevalence_chain.csv")

    # Block 3
    per_run = pd.read_csv(OUTPUT_ROOT / "outputs/11_round1_analysis/11_per_run_scores.csv")
    decomp, per_enc = block3_association_decomposition(per_run)
    _save(decomp, "block3_1_association_decomposition.csv")
    _save(per_enc, "block3_1_within_encoder_spearman_by_encoder.csv")
    assoc_regen = block3_regenerate_association(per_run)
    _save(assoc_regen, "block3_5_association_both_definitions.csv")
    var_df = pd.read_csv(OUTPUT_ROOT / "outputs/11_round1_analysis/11_variance_components.csv")
    icc = block3_icc_disattenuated(var_df, enc)
    _save(icc, "block3_4_icc_disattenuated.csv")

    # Block 4
    paired = pd.read_csv(OUTPUT_ROOT / "outputs/20_round2_diagnostic/20_within_seed_paired_changes.csv")
    trade = block4_tradeoff(paired)
    _save(trade, "block4_tradeoff_marginals.csv")

    # Block 5
    from shared.pool_loader import load_primary_candidates

    candidates = load_primary_candidates()
    kb = block5_kb_metrics()
    _save(kb, "block5_2_map_mrr_recall_table.csv")
    denom = block5_mrr_denominator(candidates)
    _save(denom, "block5_mrr_denominator.csv")

    # Block 6
    err = block6_error_base_rates()
    _save(err, "block6_1_error_feature_base_rates.csv")
    slides = block6_slide_pmids()
    _save(slides, "block6_3_slide_pmid_lookup.csv")

    # Block 7
    onc = block7_oncology()
    _save(onc, "block7_1_oncology_agreement.csv")

    # Block 8
    stale = block8_stale_csvs()
    _save(stale, "block8_2_stale_outputs_csvs.csv")
    block8_environment()
    print("Audit complete.")


if __name__ == "__main__":
    main()
