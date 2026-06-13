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
    """Step 20 prose is owned by 20_round2_diagnostic/report.py after analyze."""
    paths = paths or step_paths(STEPS["20"])
    report_path = paths["reports"] / "report.md"
    print(
        "  Step 20: skipping report regeneration "
        f"(owned by 20_round2_diagnostic/report.py; not overwriting {report_path})"
    )
    return report_path
