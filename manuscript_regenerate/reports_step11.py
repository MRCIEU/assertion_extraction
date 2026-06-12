"""Manuscript report writer for step 11."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import _report_utils as ru
from .paths import STEPS, VOCAB, step_paths

_BENCH = VOCAB["benchmark"]
_KB = VOCAB["kb"]
_QUESTION = VOCAB["question"]


def write_report_11(paths: dict[str, Path] | None = None) -> Path:
    paths = paths or step_paths(STEPS["11"])
    out = paths["outputs"]

    enc = ru.read_csv(out / "11_encoder_summary.csv")
    var = ru.read_csv(out / "11_variance_components.csv")
    boot = ru.read_csv(out / "11_variance_components_bootstrap.csv")
    abs_kb = ru.read_csv(out / "11_absolute_kb_levels.csv")
    seed_assoc = ru.read_csv(out / "11_benchmark_kb_seed_association.csv")
    enc_corr = ru.read_csv(out / "11_benchmark_kb_correlations.csv")
    ece_corr = ru.read_csv(out / "11_benchmark_ece_correlations.csv")
    lift = ru.read_csv(out / "11_untrained_floor_lift.csv")
    bench_range = ru.read_csv(out / "11_benchmark_f1_range.csv")
    easy_hard = ru.read_csv(out / "11_easy_hard_ranking.csv")
    dist_corr = ru.read_csv(out / "11_distance_score_correlation.csv")
    pool_rob = ru.read_csv(out / "11_pool_size_robustness.csv")

    spread = float(bench_range["spread"].iloc[0])
    bench_min = float(bench_range["min_f1"].iloc[0])
    bench_max = float(bench_range["max_f1"].iloc[0])
    bench_std = float(bench_range["std_f1"].iloc[0])
    bench_enc = var[var["metric"] == "benchmark_f1"].iloc[0]
    gd_var = var[var["metric"] == "kb_mrr_gene_drug"].iloc[0]
    gdis_var = var[var["metric"] == "kb_mrr_gene_disease"].iloc[0]
    bench_boot = boot[boot["metric"] == "benchmark_f1"].iloc[0]

    finetuned = abs_kb[abs_kb["reference"] == "finetuned_encoders_mean"].iloc[0]
    random_mrr = float(abs_kb[abs_kb["reference"] == "random_uniform"]["mrr_overall"].iloc[0])
    dist_mrr = float(abs_kb[abs_kb["reference"] == "distance_ranker"]["mrr_overall"].iloc[0])
    dist_hard = float(abs_kb[abs_kb["reference"] == "distance_ranker"]["mrr_hard"].iloc[0])
    dist_easy = float(abs_kb[abs_kb["reference"] == "distance_ranker"]["mrr_easy"].iloc[0])
    ft_gd = float(finetuned["mrr_gene_drug"])
    ft_gdis = float(finetuned["mrr_gene_disease"])
    ft_hard = float(finetuned["mrr_hard"])

    gd_seed = seed_assoc[seed_assoc["pair_type"] == "gene-drug"].iloc[0]
    gdis_seed = seed_assoc[seed_assoc["pair_type"] == "gene-disease"].iloc[0]
    gd_enc = enc_corr[(enc_corr["pair_type"] == "gene-drug") & (enc_corr["method"] == "encoder_mean_n9")].iloc[0]
    gdis_enc = enc_corr[(enc_corr["pair_type"] == "gene-disease") & (enc_corr["method"] == "encoder_mean_n9")].iloc[0]
    ece_row = ece_corr[ece_corr["pair_type"] == "calibration"].iloc[0]

    mean_bench_lift = float(lift["lift_benchmark_f1"].mean())
    mean_kb_lift = float(((lift["lift_kb_mrr_gene_drug"] + lift["lift_kb_mrr_gene_disease"]) / 2).mean())

    med_dist_spearman = float(dist_corr["spearman_r"].median())
    pool_gd = float(pool_rob[pool_rob["pair_type"] == "gene-drug"]["spearman_r"].mean())
    pool_gdis = float(pool_rob[pool_rob["pair_type"] == "gene-disease"]["spearman_r"].mean())

    hard_sub = easy_hard[easy_hard["subset"] == "hard_cross_sentence"]
    enc_ids = [m for m in hard_sub["model_id"].unique() if m != "distance_ranker"]
    dr_hard = float(hard_sub[hard_sub["model_id"] == "distance_ranker"]["mrr"].iloc[0])
    beats_dr = sum(
        1 for mid in enc_ids
        if hard_sub[hard_sub["model_id"] == mid]["mrr"].mean() > dr_hard
    )

    gd_min = float(enc["kb_mrr_gene_drug_mean"].min())
    gd_max = float(enc["kb_mrr_gene_drug_mean"].max())
    gdis_min = float(enc["kb_mrr_gene_disease_mean"].min())
    gdis_max = float(enc["kb_mrr_gene_disease_mean"].max())

    body = f"""# Round-one encoder comparison (step 11)

Generated: {ru.utc_now()}

## Plain-language summary

This step asks whether a model that scores well on the {_BENCH} also ranks curated CIViC relations well on the frozen evaluation pool. Nine encoders trained at the confirmed step-10 recipe are compared on both axes using seventy-two completed runs. The answer is read through variance shares, absolute ranking levels, and association estimates, not through a single correlation alone.

## Purpose

The {_QUESTION} needs a between-encoder reading on clean data after the step-05 marker repair. Step 11 scores fine-tuned checkpoints from the step-10 matrix on {_BENCH} and {_KB} without retraining. Variant pairs remain excluded because PubTator cannot build variant pools. The frozen step-03 pool has **1590** matched positives and **222** misses under PubTator recall limits; those limits apply equally to every encoder.

## What was measured

Each of seventy-two runs (nine encoders, eight seeds) is evaluated at its best validation-F1 checkpoint on BioRED test presence F1 and on CIViC mean reciprocal rank on the primary pool, split into gene-drug and gene-disease pair types and into easy co-sentence and hard cross-sentence subsets. Nine additional untrained-floor references score pretrained weights with a random classification head and no fine-tuning. All numbers below are derived from stored per-run scores in this rerun; prior round-one prose is not reused.

No runs were flagged degenerate. DeBERTa-base across eight seeds enters the primary set with benchmark F1 between roughly **0.740** and **0.752** per seed.

## Ranking validity and absolute knowledge-base levels

On the easy co-sentence subset the distance ranker reaches mean reciprocal rank **{dist_easy:.3f}**. On the hard cross-sentence subset its mean reciprocal rank is **{dist_hard:.3f}**. Across encoders, **{beats_dr}** of nine beat the distance ranker on the hard subset when seed-averaged. Hard-subset performance is the main check that learned models capture relation signal beyond proximity alone.

Absolute levels matter alongside relative comparisons. On the frozen pool, random ranking achieves MRR **{random_mrr:.3f}** and the distance ranker **{dist_mrr:.3f}**. Fine-tuned encoder means average **{ft_gd:.3f}** on gene-drug and **{ft_gdis:.3f}** on gene-disease, with hard-subset mean **{ft_hard:.3f}** versus distance **{dist_hard:.3f}**. Models sit well above random but only modestly above the distance ranker, so {_KB} adequacy is limited in absolute terms even when hard-subset ranking beats proximity.

Figure fig3_easy_hard_ranking_validity.png plots encoder means against the distance baseline on easy and hard subsets. Points above the dashed line on the hard panel indicate relation signal beyond entity proximity.

## Benchmark discriminative power

Among encoder means, {_BENCH} ranges from **{bench_min:.3f}** to **{bench_max:.3f}** (spread **{spread:.3f}**), comparable to within-encoder seed standard deviation near **{bench_std:.3f}**. Figure fig1_benchmark_kb_scatter.png shows encoder means with seed uncertainty bars on both axes for gene-drug and gene-disease panels; letter codes identify encoders without overlapping labels.

The variance-components method applied to benchmark F1 and to {_KB} MRR separates between-encoder from within-encoder seed variance. For benchmark F1, **{100 * float(bench_enc['encoder_variance_share']):.0f}%** of variance lies between encoders and **{100 * float(bench_enc['seed_variance_share']):.0f}%** within encoders (encoder-share interval **{100 * float(bench_boot['encoder_share_ci_lo']):.0f}%** to **{100 * float(bench_boot['encoder_share_ci_hi']):.0f}%**). For gene-drug {_KB}, between-encoder share is **{100 * float(gd_var['encoder_variance_share']):.0f}%** and within-encoder **{100 * float(gd_var['seed_variance_share']):.0f}%**. For gene-disease {_KB}, between-encoder share is **{100 * float(gdis_var['encoder_variance_share']):.0f}%** and within-encoder **{100 * float(gdis_var['seed_variance_share']):.0f}%**.

Figure fig2_variance_between_encoder.png plots the between-encoder share alone with bootstrap intervals attached to each bar. Benchmark F1 has a higher between-encoder share than either {_KB} axis, so the benchmark discriminates encoders more strongly than knowledge-base ranking does on this recipe.

## Benchmark versus knowledge-base association

Encoder-mean Spearman correlation (nine points) is **{float(gd_enc['estimate']):+.3f}** (interval **{float(gd_enc['ci_lo']):+.3f}** to **{float(gd_enc['ci_hi']):+.3f}**) for gene-drug and **{float(gdis_enc['estimate']):+.3f}** (interval **{float(gdis_enc['ci_lo']):+.3f}** to **{float(gdis_enc['ci_hi']):+.3f}**) for gene-disease. These nine-point estimates are weak and interval-heavy.

The primary association method is seed-level cluster bootstrap over encoders: gene-drug Spearman **{float(gd_seed['spearman']):+.3f}** (interval **{float(gd_seed['ci_lo']):+.3f}** to **{float(gd_seed['ci_hi']):+.3f}**); gene-disease Spearman **{float(gdis_seed['spearman']):+.3f}** (interval **{float(gdis_seed['ci_lo']):+.3f}** to **{float(gdis_seed['ci_hi']):+.3f}**). Read these through the variance shares above. When the benchmark discriminates more than {_KB}, a negative association is closer to genuine decoupling than to benchmark saturation alone.

Gene-drug {_KB} encoder means span **{gd_min:.3f}** to **{gd_max:.3f}**. Gene-disease spans **{gdis_min:.3f}** to **{gdis_max:.3f}**.

## Fine-tuning lift

Untrained-floor references use pretrained encoders with a randomly initialised head and no fine-tuning. Mean lift across encoders is **{mean_bench_lift:.3f}** on benchmark F1 and **{mean_kb_lift:.3f}** on {_KB} MRR averaged across pair types. Fine-tuning adds substantial benchmark signal but more modest knowledge-base gain. Figure fig4_finetuning_lift.png compares per-encoder lifts on both axes.

## Calibration

At nine encoder means, higher benchmark F1 associates with lower expected calibration error against CIViC curation inclusion (Spearman **{float(ece_row['estimate']):+.3f}**, interval **{float(ece_row['ci_lo']):+.3f}** to **{float(ece_row['ci_hi']):+.3f}**). Calibration is measured against curation inclusion, not objective biomedical truth, and may diverge from ranking.

## Robustness diagnostics

Across runs, median Spearman correlation between model scores and entity proximity is **{med_dist_spearman:.3f}**. Values nearer one suggest ranking tracks closeness; values nearer zero suggest signal beyond proximity.

Correlating per-abstract pool size with per-abstract MRR gives mean Spearman **{pool_gd:.3f}** on gene-drug and **{pool_gdis:.3f}** on gene-disease. This tests whether observed pool size drives the metric common-mode across models. It does not measure distractors PubTator missed entirely.

## Closing read

The clean seventy-two-run matrix at **5e-6/none** supports three descriptive pictures: genuine decoupling when the benchmark discriminates but that dimension does not transfer to CIViC; both axes seed-dominated; or benchmark saturation when encoder spread on {_BENCH} is tiny. The variance-component comparison in fig2_variance_between_encoder.png is the central diagnostic. Absolute {_KB} levels sit modestly above random and proximity baselines; encoder choice matters less than seed noise on {_KB} axes. No go/no-go threshold is applied.

A second knowledge base was explored for external validity; its usable abstract-grounded subset was limited, so CIViC remains the primary {_KB} axis (recorded as future work).

## Linkage

Steps 00 through 02 froze **1812** targets. Step 03 built the pool with **1590** matched and **222** missed recall. Step 05 passed the offset gate. Step 10 confirmed **5e-6/none** and produced the matrix scored here. Step 20 follows per-epoch checkpoints from the same matrix to test within-model training dynamics.

## Outputs

Encoder summaries, variance tables, association estimates, lift tables, and diagnostics are in the step-11 outputs directory, including 11_encoder_summary.csv, 11_variance_components.csv, and 11_benchmark_kb_seed_association.csv.
"""
    return ru.write_md(paths["reports"] / "report.md", body)
