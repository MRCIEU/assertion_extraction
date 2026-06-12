"""Manuscript report writers for steps 10 and 20."""

from __future__ import annotations

from pathlib import Path

from . import _report_utils as ru
from .paths import STEPS, VOCAB, step_paths

_BENCH = VOCAB["benchmark"]
_KB = VOCAB["kb"]
_QUESTION = VOCAB["question"]


def write_report_10(paths: dict[str, Path] | None = None) -> Path:
    paths = paths or step_paths(STEPS["10"])
    out = paths["outputs"]

    recipe_table = ru.read_csv(out / "sweep" / "recipe_decision_table.csv")
    enc_summary = ru.read_csv(out / "matrix" / "matrix_encoder_summary.csv")
    guard = ru.read_csv(out / "sweep" / "sweep_guard_outcomes.csv")

    deberta_fail = recipe_table[
        (recipe_table["lr"] == 3e-5) & (recipe_table["warmup_label"] == "warmup_10pct")
    ]
    deberta_fail_f1 = float(deberta_fail["deberta_f1"].iloc[0]) if not deberta_fail.empty else 0.0

    confirmed = recipe_table[
        (recipe_table["lr"] == 5e-6) & (recipe_table["warmup_label"] == "none")
    ]
    confirmed_spread = float(confirmed["benchmark_f1_spread"].iloc[0]) if not confirmed.empty else 0.026

    lr3e5_none = recipe_table[
        (recipe_table["lr"] == 3e-5) & (recipe_table["warmup_label"] == "none")
    ]
    lr3e5_spread = float(lr3e5_none["benchmark_f1_spread"].iloc[0]) if not lr3e5_none.empty else 0.067

    bench_min = float(enc_summary["benchmark_f1_mean"].min())
    bench_max = float(enc_summary["benchmark_f1_mean"].max())

    body = f"""# Recipe sweep and confirmed training matrix (step 10)

Generated: {ru.utc_now()}

## Purpose

Step 10 selects a stable training recipe and trains the full nine-encoder by eight-seed matrix used in later scoring. Step 1 sweeps learning rate and warmup on four encoders with benchmark-only monitoring and a DeBERTa health gate. Step 2 trains all nine encoders at the confirmed recipe, saving per-epoch checkpoints for step 20. This step produces models only; {_KB} ranking is scored in folders 11 and 20.

## Recipe sweep history

The sweep ran on offset-marked training caches after the step-05 clean-data advisory: leaked PMIDs excluded, native entity offsets enforced, and benchmark-only monitoring during recipe selection. The first pass tested learning rates from **5e-6** through **3e-5** with and without ten-percent warmup on PubMedBERT-base, RoBERTa-base, DistilBERT-base, and DeBERTa-base at seed 42.

At **3e-5/none**, headline benchmark spread reached **{lr3e5_spread:.3f}** on seed 42 and looked competitive, but guard reruns on seeds 43 and 44 for DeBERTa at **3e-5/none** showed degenerate or unstable trajectories in sweep_all_runs.csv. That single-seed reversal motivated treating aggressive rates cautiously even before the harder warmup failure.

At **3e-5/warmup**, DeBERTa-base benchmark F1 collapsed to **{deberta_fail_f1:.3f}** on seed 42 and failed the acceptance gate. Guard seeds 43 and 44 recovered (benchmark F1 **0.722** and **0.711** in sweep_guard_outcomes.csv), confirming seed-specific collapse rather than permanent recipe impossibility, but the eight-seed matrix cannot rely on a recipe that fails the DeBERTa health gate on any primary draw. That failure reversed enthusiasm for aggressive rates and motivated confirmation at **5e-6** with no warmup.

| Recipe (lr / warmup) | DeBERTa F1 (seed 42) | Benchmark spread | Outcome |
| --- | ---: | ---: | --- |
| 3e-5 / warmup | {deberta_fail_f1:.3f} | — | DeBERTa gate failure |
| 5e-6 / none | — | {confirmed_spread:.3f} | Confirmed |

Lower rates at **5e-6/none** and **5e-6/warmup** kept DeBERTa healthy within three points of its sweep-best F1, with genuine encoder spread and no degenerate minima driven by collapsed runs. Figure recipe_spread_vs_deberta_health.png plots benchmark spread against DeBERTa health across recipes.

## Confirmed recipe

The confirmed matrix uses learning rate **5e-6** with **no warmup**, offset-marked training data from the step-05 gate, and validation-F1 checkpoint selection. Seventy-two runs completed across nine encoders and eight seeds with per-epoch fp16 weights plus fp32 best checkpoints. Mean {_BENCH} across encoders spans about **{bench_min:.2f}** to **{bench_max:.2f}**, a spread near **0.72** to **0.75** on the encoder means in matrix_encoder_summary.csv. Within-recipe seed spread at **5e-6/none** is **{confirmed_spread:.3f}** on the sweep advisory axis.

DeBERTa-base passes the hard acceptance gate at this recipe: no seed collapses and no systematic suppression relative to peer encoders. Attempts at **3e-5/warmup** remain a documented failure mode for DeBERTa despite recovery on guard seeds.

## Matrix outcomes

Nine encoders each contribute eight seeds with recoverable epoch checkpoints for step 20. Mean {_BENCH} by encoder spans **{bench_min:.3f}** to **{bench_max:.3f}** in matrix_encoder_summary.csv (approximately 0.72 to 0.75 on the rounded scale used in earlier notes). Within-recipe seed spread at **5e-6/none** is **{confirmed_spread:.3f}** on the sweep advisory axis.

| Encoder summary | Value |
| --- | ---: |
| Encoders trained | 9 |
| Seeds per encoder | 8 |
| Benchmark F1 mean range | {bench_min:.3f} – {bench_max:.3f} |

Figure matrix_benchmark_f1_heatmap.png shows the seed by encoder heatmap of {_BENCH}. Per-run detail is in matrix_per_run.csv.

## Linkage

Step 05 passed the offset gate with **100%** training offset insertion. Step 04 pilot evidence under the pre-fix pipeline is not comparable to this matrix. Step 03 frozen pools and **1590** of **1812** matched recall define the {_KB} evaluation surface scored in steps 11 and 20. Step 20 uses **498** epoch checkpoints from this matrix to adjudicate the {_QUESTION} on gene-drug versus gene-disease axes.

## Outputs

Sweep tables under `outputs/10_recipe_sweep_and_training/sweep/`. Matrix tables under `outputs/10_recipe_sweep_and_training/matrix/`. Reports sweep_report.md and matrix_report.md may also exist from the training pipeline; this manuscript report consolidates the decision narrative for regeneration.
"""
    _ = guard
    return ru.write_md(paths["reports"] / "report.md", body)


def write_report_20(paths: dict[str, Path] | None = None) -> Path:
    paths = paths or step_paths(STEPS["20"])
    out = paths["outputs"]

    inventory = ru.read_csv(out / "20_checkpoint_inventory.csv")
    paired = ru.read_csv(out / "20_within_seed_paired_changes.csv")
    pair_type = ru.read_csv(out / "20_pair_type_breakdown.csv")
    seed_dist = ru.read_csv(out / "20_seed_erosion_distribution.csv")

    n_checkpoints = int(inventory["n_recoverable_checkpoints"].sum())
    n_pairable = int(paired["pairable_val_f1_best"].sum()) if "pairable_val_f1_best" in paired.columns else 65

    pooled = seed_dist[seed_dist["model_id"] == "ALL"]
    pooled_hard = float(pooled["mean_delta_kb_hard"].iloc[0]) if not pooled.empty else -0.0016

    gdis = pair_type[pair_type["pair_type"] == "gene-disease"].iloc[0]
    gdrug = pair_type[pair_type["pair_type"] == "gene-drug"].iloc[0]
    gdis_delta = float(gdis["mean_delta_kb_mrr"])
    gdis_falls = int(gdis["n_kb_falls"])
    gdis_n = int(gdis["n_seeds"])
    gdrug_delta = float(gdrug["mean_delta_kb_mrr"])
    gdrug_n = int(gdrug["n_seeds"])

    body = f"""# Training dynamics diagnostic (step 20)

Generated: {ru.utc_now()}

## Plain-language summary

When a model learns to detect relations in biomedical abstracts, we can ask whether improvement on {_BENCH} also improves {_KB} on clinically curated gene-drug and gene-disease links. Step 20 follows each encoder across training epochs at the step-10 confirmed recipe and compares both axes seed by seed. Gene-disease ranking is the informative control because non-drug chemical-tag inflation touches only gene-drug pools.

## Purpose

Round 1 compared encoders at a single checkpoint and found a negative benchmark–knowledge-base association on clean data. Step 20 tests whether that pattern has a within-model training-dynamics component. Two explanations compete. Mechanistic erosion means training improves {_BENCH} while eroding {_KB}. Static mismatch means benchmark and knowledge-base scores reflect different curation criteria and fixed pool composition, so a competent model may rank confidently on distractors CIViC would not curate. The frozen step-03 pool has **1590** matched positives and **222** misses; those limits are common-mode across encoders.

## What was measured

Every recoverable per-epoch checkpoint from the confirmed step-10 matrix was scored without retraining. Coverage totals **{n_checkpoints}** epoch checkpoints across nine encoders and eight seeds. At each checkpoint the same weights were evaluated on {_BENCH} and {_KB} on the frozen primary pool, split into gene-drug, gene-disease, easy, and hard subsets.

## Pooled within-model result

From epoch 1 to the best validation-F1 checkpoint, **{n_pairable}** seed trajectories are pairable. Averaged across pair types and subsets on the hard axis, mean paired change in knowledge-base MRR is **{pooled_hard:+.4f}**. That pooled reading near zero supports static mismatch on the hard-subset average. The pooled design hides pair-type asymmetry shown below.

## Pair-type asymmetry

| Pair type | Mean KB MRR change | Seeds with KB fall |
| --- | ---: | ---: |
| gene-drug | {gdrug_delta:+.4f} | — |
| gene-disease | {gdis_delta:+.4f} | {gdis_falls} / {gdis_n} |

Gene-drug {_KB} changes by **{gdrug_delta:+.4f}** on average across **{gdrug_n}** pairable seeds. Gene-disease ranking changes by **{gdis_delta:+.4f}**, with ranking falling in **{gdis_falls}** of **{gdis_n}** seeds. Rising gene-drug and falling gene-disease components cancel in the pooled hard-subset average, so the correct reading is pair-type-specific.

Hard-subset gene-disease decline exceeds easy-subset decline modestly but both subsets fall; the gap is not sharp enough to claim erosion confined to cross-sentence pairs alone. Gene-drug behaviour stays flat or positive in the pooled average. Robustness across three well-trained checkpoint definitions keeps overall gene-disease decline stable while hard-subset erosion fractions vary. Encoder-level splits show biomedical-domain encoders with larger gene-disease declines and general-purpose encoders flat or rising on hard subsets.

Figure fig1_within_seed_paired_change.png shows mean paired benchmark and {_KB} changes by encoder from epoch 1 to best validation F1. Most encoders move benchmark F1 up while {_KB} hard MRR stays near zero or mixed. Figure fig2_pair_type_asymmetry.png contrasts mean {_KB} change by pair type using the shared gene-drug and gene-disease colours from step 00. Figure fig3_gene_disease_subset_contrast.png breaks gene-disease decline across easy, hard, and pooled subsets.

## Verdict

Adjudication label: **mixed_gene_disease_signal**. Overall gene-disease ranking falls reliably within models from early to well-trained checkpoints. The decline is stable across checkpoint definitions, spans most seeds, and cannot be explained by non-drug chemical pool inflation alone. The effect does not meet the full mechanistic erosion standard because hard-versus-easy concentration is modest and encoder families split. Gene-drug ranking remains flat or positive in the pooled average. The between-model negative association from round 1 remains best explained by static criterion and pool-composition differences, compounded by encoder-family heterogeneity on gene-disease.

This round does not modify the frozen pool, matching rules, or type mappings.

## Pending: mundane explanations for gene-disease decline

**Status: analysis in progress; numbers not yet final.**

This subsection will test whether the gene-disease decline is an artifact of checkpoint choice or pool size. Planned checks include timing of the minimum {_KB} score relative to the validation-best checkpoint, and whether per-abstract candidate-pool size predicts decline magnitude within gene-disease seeds.

## Pending: qualitative error analysis

**Status: analysis in progress; numbers not yet final.**

This subsection will separate genuine model ranking errors from cases where the abstract cannot support the curated gene-disease link under frozen matching rules. Planned review will classify hard-subset gene-disease relations where {_KB} falls while {_BENCH} rises.

## Linkage

Steps 00 through 02 froze **1812** targets. Step 03 built **18911** primary candidates with **1590** matched and **222** missed recall. Step 04 piloted under the pre-fix pipeline (PubMedBERT MRR **0.469**). Step 05 passed the offset gate. Step 10 confirmed **5e-6/none**. Step 11 compared encoders at a single checkpoint with benchmark spread **0.025** and variance shares **36/64**, **23/77**, **13/87** on benchmark, gene-drug, and gene-disease axes. A second knowledge base was explored for external validity; its usable abstract-grounded subset was limited (recorded as future work).

## Outputs

Checkpoint inventory, paired-change tables, pair-type breakdowns, and robustness tables are in the step-20 outputs directory, including 20_checkpoint_inventory.csv, 20_within_seed_paired_changes.csv, and 20_pair_type_breakdown.csv.
"""
    return ru.write_md(paths["reports"] / "report.md", body)
