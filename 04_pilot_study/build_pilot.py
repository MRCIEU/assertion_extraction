"""Orchestrate minimal-training divergence pilot: train, score, analyze, report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .analysis import (
    build_benchmark_kb_table,
    calibration_decoupling,
    decoupling_summary,
    rank_flips,
)
from .config import (
    MODELS,
    MODEL_BY_ID,
    OUTPUT_DIR,
    POSITIVE_FRACTION_PRIOR,
    REPORT_DIR,
    SAMPLING_SEED,
    STEP03_RANKING_BASELINES_CSV,
    STEP03_RANKING_VERIFICATION_JSON,
    TRAIN_MAX_STEPS,
    TRAIN_SEEDS,
)
from .distance_confound import run_distance_confound_diagnostic
from .figures import (
    export_tables,
    plot_benchmark_vs_kb,
    plot_distance_correlation,
    plot_ece_vs_benchmark,
    plot_positive_distance_distribution,
    plot_reliability_diagrams,
    plot_score_distributions,
    plot_subset_ranking,
)
from .inference import load_all_scores, score_all_models
from .metrics_calibration import evaluate_calibration, evaluate_calibration_baselines
from .metrics_ranking import (
    analytic_random_mrr,
    evaluate_ranking,
    verify_ranking_implementation,
)
from .pool_loader import load_primary_candidates
from .train import train_all_models
from .train_data import build_train_examples


def _load_step03_baselines() -> pd.DataFrame:
    df = pd.read_csv(STEP03_RANKING_BASELINES_CSV)
    return df.rename(columns={"baseline": "model_or_baseline"})


def _load_step03_verification() -> dict[str, float]:
    if STEP03_RANKING_VERIFICATION_JSON.exists():
        payload = json.loads(STEP03_RANKING_VERIFICATION_JSON.read_text(encoding="utf-8"))
        return payload.get("ranking_verification", payload)
    template = load_primary_candidates()[
        ["candidate_id", "pmid", "pair_type", "label_civic_curated_positive"]
    ].copy()
    template["score"] = 0.0
    return verify_ranking_implementation(template)


def _score_distribution_summary(scores_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id, sub in scores_df.groupby("model_id"):
        p = sub["score"].astype(float)
        rows.append(
            {
                "model_id": model_id,
                "mean_score": float(p.mean()),
                "std_score": float(p.std()),
                "min_score": float(p.min()),
                "p25": float(p.quantile(0.25)),
                "median": float(p.median()),
                "p75": float(p.quantile(0.75)),
                "max_score": float(p.max()),
                "pct_lt_0_05": float((p < 0.05).mean()),
                "pct_above_0.5": float((p > 0.5).mean()),
            }
        )
    return pd.DataFrame(rows)


def _preliminary_assessment(
    ranking_models: pd.DataFrame,
    ranking_baselines: pd.DataFrame,
    decouple: dict[str, Any],
    cal_summary: dict[str, Any],
    cal_decouple: dict[str, Any],
    score_dist: pd.DataFrame,
    ranking_verify: dict[str, float],
) -> dict[str, Any]:
    rand = ranking_baselines[ranking_baselines["model_or_baseline"] == "random"].iloc[0]
    const = ranking_baselines[ranking_baselines["model_or_baseline"] == "constant"].iloc[0]
    dist = ranking_baselines[ranking_baselines["model_or_baseline"] == "distance_ranker"].iloc[0]
    best_mrr = ranking_models["mrr"].max()
    best_auc = ranking_models["auc_pr"].max()
    margin_mrr = best_mrr - rand["mrr"]
    margin_vs_dist = best_mrr - dist["mrr"]

    # Non-degenerate scores: no model mean < 0.05
    scores_non_degenerate = bool((score_dist["mean_score"] >= 0.05).all())

    ranking_signal = bool(
        scores_non_degenerate
        and margin_mrr >= 0.05
        and margin_vs_dist >= 0.02
        and best_mrr > rand["mrr"]
        and best_mrr > dist["mrr"]
        and best_auc > rand["auc_pr"]
    )
    benchmark_decoupled = bool(decouple["decoupled"])
    ece_spread = cal_summary.get("ece_spread", 0.0)
    cal_nontrivial = bool(ece_spread >= 0.02)
    cal_decoupled = bool(cal_decouple.get("decoupled", False))
    calibration_ok = cal_nontrivial and cal_decoupled

    justified = ranking_signal and benchmark_decoupled and calibration_ok

    return {
        "ranking_verification_pass": bool(ranking_verify.get("verification_pass")),
        "constant_mrr": round(float(const["mrr"]), 4),
        "random_mrr": round(float(rand["mrr"]), 4),
        "analytic_random_mrr": round(float(ranking_verify.get("mrr_analytic", 0)), 4),
        "scores_non_degenerate": scores_non_degenerate,
        "ranking_signal_exists": ranking_signal,
        "ranking_mrr_margin_vs_random": round(float(margin_mrr), 4),
        "ranking_mrr_margin_vs_distance": round(float(margin_vs_dist), 4),
        "distance_ranker_mrr": round(float(dist["mrr"]), 4),
        "benchmark_kb_decoupled": benchmark_decoupled,
        "calibration_nontrivial_and_decoupled": calibration_ok,
        "calibration_nontrivial": cal_nontrivial,
        "calibration_decoupled": cal_decoupled,
        "preliminary_framework_sound": justified,
        "best_model_mrr": round(float(best_mrr), 4),
        "random_baseline_mrr": round(float(rand["mrr"]), 4),
        "ece_spread": round(float(ece_spread), 4),
        "small_n_caveat": "Only 3 encoders and minimal training (~3000 steps); pilot tests existence of signals, not effect sizes.",
    }


def write_report(
    verdict: dict[str, Any],
    ranking_verify: dict[str, float],
    ranking_models: pd.DataFrame,
    ranking_baselines: pd.DataFrame,
    calibration_df: pd.DataFrame,
    calibration_baselines: pd.DataFrame,
    bench_kb: pd.DataFrame,
    flips: pd.DataFrame,
    decouple: dict[str, Any],
    cal_decouple: dict[str, Any],
    score_dist: pd.DataFrame,
    distance_confound: dict[str, Any] | None = None,
) -> None:
    rank_lines = ""
    for row in ranking_models.sort_values("mrr", ascending=False).itertuples():
        rank_lines += (
            f"| {row.model_id} | {row.mrr:.3f} | {row.recall_at_1:.3f} | {row.recall_at_3:.3f} | "
            f"{row.recall_at_5:.3f} | {row.auc_pr:.3f} |\n"
        )

    base_lines = ""
    for row in ranking_baselines.itertuples():
        base_lines += f"| {row.model_or_baseline} | {row.mrr:.3f} | {row.auc_pr:.3f} |\n"

    dist_lines = ""
    for _, row in score_dist.sort_values("mean_score", ascending=False).iterrows():
        dist_lines += (
            f"| {row['model_id']} | {row['mean_score']:.3f} | {row['std_score']:.3f} | "
            f"{row['median']:.3f} | {row['pct_lt_0_05']:.1%} |\n"
        )

    bench_lines = ""
    for row in bench_kb.itertuples():
        bench_lines += (
            f"| {row.short_name} | {row.benchmark_f1:.3f} | {int(row.benchmark_rank)} | "
            f"{row.mrr:.3f} | {int(row.kb_rank)} | {int(row.rank_delta)} |\n"
        )

    flip_lines = ""
    for row in flips.itertuples():
        flip_lines += (
            f"| {row.short_name} | bench rank {int(row.benchmark_rank)} | KB rank {int(row.kb_rank)} | "
            f"Δ={int(row.rank_delta)} | MRR={row.mrr:.3f} |\n"
        )

    cal_lines = ""
    for row in calibration_df.sort_values("ece").itertuples():
        cal_lines += f"| {row.model_or_baseline} | {row.ece:.3f} | {row.mean_score:.3f} |\n"

    cal_base_lines = ""
    for row in calibration_baselines.itertuples():
        cal_base_lines += f"| {row.model_or_baseline} | {row.ece:.3f} |\n"

    cal_table = cal_decouple["table"]
    cal_dec_lines = ""
    for row in cal_table.itertuples():
        cal_dec_lines += (
            f"| {row.short_name} | {row.benchmark_f1:.3f} | {row.ece:.3f} | {int(row.calibration_rank)} |\n"
        )

    model_desc = "\n".join(
        f"- **{m.short_name}** (`{m.model_id}`): {m.benchmark_name} F1≈{m.benchmark_f1:.3f} ({m.benchmark_source})"
        for m in MODELS
    )

    dist_section = ""
    if distance_confound:
        dc = distance_confound
        dv = dc["verdict"]
        corr_df = dc["correlation"]
        subset_df = dc["subset_ranking"]
        pos_hist = dc["positive_distance_histogram"]

        corr_lines = ""
        for row in corr_df.itertuples():
            corr_lines += (
                f"| {row.model_name} | {row.pearson_r_proximity:.3f} | {row.spearman_r_proximity:.3f} | "
                f"{row.pointbiserial_r_co_sentence:.3f} |\n"
            )

        subset_lines = ""
        for subset_label, subset_key in [("Easy (co-sentence)", "easy_co_sentence"), ("Hard (cross-sentence)", "hard_cross_sentence"), ("All (known distance)", "all")]:
            sub = subset_df[subset_df["subset"] == subset_key]
            if sub.empty:
                continue
            n_cand = int(sub.iloc[0]["n_candidates"])
            n_pm = int(sub.iloc[0]["n_pmids"])
            subset_lines += f"\n**{subset_label}** — {n_cand} candidates, {n_pm} PMIDs\n\n"
            subset_lines += "| Ranker | MRR | R@1 | R@3 | R@5 |\n| --- | ---: | ---: | ---: | ---: |\n"
            for row in sub.sort_values("mrr", ascending=False).itertuples():
                name = row.ranker
                if name in {"pubmedbert_base", "biolinkbert_base", "roberta_base"}:
                    name = MODEL_BY_ID[name].short_name if name in MODEL_BY_ID else name
                elif name == "distance_ranker":
                    name = "Distance ranker"
                subset_lines += (
                    f"| {name} | {row.mrr:.3f} | {row.recall_at_1:.3f} | {row.recall_at_3:.3f} | {row.recall_at_5:.3f} |\n"
                )

        hist_lines = ""
        for row in pos_hist.itertuples():
            hist_lines += f"| {int(row.sentence_distance)} | {int(row.n_positives)} | {row.fraction_of_known:.1%} |\n"

        dist_section = f"""
### Distance-confound diagnostic

The best trained model (PubMedBERT MRR={verdict['best_model_mrr']:.3f}) does **not** beat the step-03 distance ranker (MRR={verdict['distance_ranker_mrr']:.3f}) on the full pool. Three post-hoc checks use saved model scores and entity proximity in the frozen pool — **no re-training**.

#### Diagnostic 1 — Model score vs entity proximity

| Model | Pearson r (proximity) | Spearman r (proximity) | r (co-sentence indicator) |
| --- | ---: | ---: | ---: |
{corr_lines}

Higher correlation suggests scores track entity proximity; lower correlation with weak overall MRR suggests other signal may exist but is under-exploited at pilot training scale.

Figure: `figures/04_distance_score_correlation.png` · Table: `outputs/04_distance_score_correlation.csv`

#### Diagnostic 2 — Easy vs hard subset ranking (key)

Pool split by sentence distance: **easy** = both entities in the same sentence; **hard** = different sentences (proximity shortcut weak).

{subset_lines}

Figure: `figures/04_distance_hard_subset_mrr.png` · Table: `outputs/04_distance_subset_ranking.csv`

#### Diagnostic 3 — Distance distribution of CIViC-curated positives

| Sentence distance | Positives | Share of known |
| --- | ---: | ---: |
{hist_lines}

**{dv['fraction_positives_co_sentence']:.1%}** of CIViC-curated positives with known offsets are co-sentence.

Figure: `figures/04_positive_distance_distribution.png` · Tables: `outputs/04_positive_distance_distribution.csv`, `outputs/04_positive_distance_summary.csv`

#### Verdict (descriptive)

{dv['verdict_summary']}

**Implication (not implemented here):** {dv['next_step_implication']}

*{verdict['small_n_caveat']}*
"""

    report = f"""# Step 04: Divergence Pilot Report (minimal training)

Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

## Pilot study overview

Minimal-training check with three encoders (PubMedBERT-base, BioLinkBERT-base, RoBERTa-base) trained on BioRED + DrugProt binary relation-presence, evaluated on the frozen step-03 candidate pool. Training excludes the three PMIDs flagged in step 01.

**Caveat (read everywhere):** This is a scaled-down preview with **only 3 models**, **~{TRAIN_MAX_STEPS} optimizer steps**, and **{len(TRAIN_SEEDS)} seeds** per model. It tests whether ranking / decoupling / calibration **signals exist**, not their magnitude in the full factorial experiment.

---

## Ranking metric bug fix (blocking, verified first)

The prior constant-score baseline scored MRR≈0.754 — a tie-handling bug (best rank among tied positives always 1). Fixed by **average-rank on scores + random tiebreak** per abstract so all-equal scores behave like random ordering.

| Check | Value |
| --- | ---: |
| Random baseline MRR | {ranking_verify['mrr_random']:.4f} |
| Constant-score MRR | {ranking_verify['mrr_constant']:.4f} |
| Analytic E[MRR] (uniform random) | {ranking_verify['mrr_analytic']:.4f} |
| \\|constant − random\\| | {ranking_verify['abs_constant_random_diff']:.4f} |
| Verification | **{'PASS' if verdict['ranking_verification_pass'] else 'FAIL'}** |

Constant ≈ random confirms the fix before any model evaluation.

---

## Training setup

**Task:** binary relation-presence — entity pair + abstract → P(present). Labels are **presence only** (step-01 incommensurability: no fine-grained relation types).

**Training corpora:** BioRED (gene–drug, gene–disease) + DrugProt (gene–drug), entity-marked input `[E1]…[/E1]` / `[E2]…[/E2]` aligned with CIViC eval framing.

**Encoders (benchmark-quality gradient):**

{model_desc}

**Scale:** {TRAIN_MAX_STEPS} steps, batch 16, lr 2e-5, max seq 256; seeds {list(TRAIN_SEEDS)}; checkpoints under `data/checkpoints/`.

**Evaluation pool:** frozen step-03 gene–drug / gene–disease candidate pool; calibration ground truth = CIViC curation inclusion, not objective biomedical truth.

---

## Sanity checks (score distributions)

| Model | Mean P(present) | Std | Median | % scores < 0.05 |
| --- | ---: | ---: | ---: | ---: |
{dist_lines}

Off-the-shelf run had means ~0.003–0.016. Trained models should show **non-degenerate** spread. Non-degenerate check: **{'PASS' if verdict['scores_non_degenerate'] else 'FAIL'}**.

---

## A. Ranking signal

### Trained models (primary pool)

| Model | MRR | R@1 | R@3 | R@5 | AUC-PR |
| --- | ---: | ---: | ---: | ---: | ---: |
{rank_lines}

### Step-03 baselines (random, constant, distance ranker)

| Baseline | MRR | AUC-PR |
| --- | ---: | ---: |
{base_lines}

Best MRR={verdict['best_model_mrr']:.3f} vs random={verdict['random_baseline_mrr']:.3f} (margin={verdict['ranking_mrr_margin_vs_random']:.3f}); vs distance ranker={verdict['distance_ranker_mrr']:.3f} (margin={verdict['ranking_mrr_margin_vs_distance']:.3f}).

**Preliminary assessment (A):** {'Ranking signal observed above random and distance-ranker baselines.' if verdict['ranking_signal_exists'] else 'Ranking signal not clearly above baselines at this pilot scale.'}
{dist_section}
---

## B. Benchmark vs KB-ranking decoupling

| Model | Benchmark F1 | Bench rank | MRR | KB rank | Δ rank |
| --- | ---: | ---: | ---: | ---: | ---: |
{bench_lines}

Spearman ρ(benchmark F1, MRR) = {decouple['spearman_rho']}; rank flips = {decouple['n_rank_flips']}.

**Small-n caveat:** with only 3 models, ρ and flips are **illustrative** — interpret cautiously.

### Rank-flip examples

{flip_lines if flip_lines else '_No rank flips (or all three tied on benchmark ordering)._'}

**Preliminary assessment (B):** {'Benchmark and KB ranking orders differ (decoupling observed).' if verdict['benchmark_kb_decoupled'] else 'Benchmark and KB ranking orders largely aligned at n=3.'}

---

## C. Calibration vs CIViC inclusion

**Caveat:** ECE measures alignment with **CIViC curation inclusion**, not ground-truth correctness. High scores on non-positives may reflect uncurated true relations.

### C1 — Non-trivial variation

| Model | ECE | Mean score |
| --- | ---: | ---: |
{cal_lines}

ECE spread: **{verdict['ece_spread']:.3f}**. Compare to trivial baselines (~{POSITIVE_FRACTION_PRIOR:.1%} positive rate):

| Baseline | ECE |
| --- | ---: |
{cal_base_lines}

If trained ECE ≈ `constant_low`, calibration may be task-driven-to-low, not informative.

### C2 — Decoupling from benchmark

| Model | Benchmark F1 | ECE | Cal rank (lower better) |
| --- | ---: | ---: | ---: |
{cal_dec_lines}

Spearman ρ(benchmark F1, ECE) = {cal_decouple['spearman_rho_benchmark_vs_ece']}.

**Preliminary assessment (C):** {'Calibration varies across models and does not track benchmark rank.' if verdict['calibration_nontrivial_and_decoupled'] else 'Calibration signal weak or tracks benchmark at this pilot scale.'}

---

## Summary

This pilot study (n=3 encoders, minimal training) provides **descriptive preliminary evidence** on whether ranking, benchmark–KB decoupling, and calibration signals exist. It is not powered for effect-size claims.

{verdict['small_n_caveat']}

---

**Outputs:** `data/04_pilot_study/model_scores/`; tables under `outputs/04_pilot_study/`; figures under `figures/04_pilot_study/`; logs under `runs/04_pilot_study/`.

Run: `sbatch 04_pilot_study/step.sbatch` (GPU, `conda activate hf-hpc`).
"""

    path = REPORT_DIR / "report.md"
    path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {path}")


def run_pilot(
    train: bool = True,
    score: bool = True,
    analyze: bool = True,
    force_train: bool = False,
    force_score: bool = False,
    force_train_data: bool = False,
    model_ids: list[str] | None = None,
) -> dict[str, Any]:
    template = load_primary_candidates()[
        ["candidate_id", "pmid", "pair_type", "label_civic_curated_positive"]
    ].copy()

    ranking_verify = _load_step03_verification()

    if train:
        examples = build_train_examples(force=force_train_data)
        train_all_models(examples, force=force_train, model_ids=model_ids)

    if score:
        from .config import MODEL_BY_ID
        from .inference import score_model

        candidates = load_primary_candidates()
        specs = [MODEL_BY_ID[m] for m in model_ids] if model_ids else MODELS
        for spec in specs:
            score_model(spec, candidates, force=force_score)

    if not analyze:
        return {"ranking_verification": ranking_verify}

    scores_df = load_all_scores()

    ranking_models = evaluate_ranking(scores_df)
    ranking_baselines = _load_step03_baselines()
    calibration_df = evaluate_calibration(scores_df)
    calibration_baselines = evaluate_calibration_baselines(template)
    score_dist = _score_distribution_summary(scores_df)

    distance_confound = run_distance_confound_diagnostic(scores_df)
    plot_distance_correlation(distance_confound["correlation"])
    plot_subset_ranking(distance_confound["subset_ranking"])
    plot_positive_distance_distribution(
        distance_confound["positive_distance_histogram"],
        distance_confound["verdict"]["fraction_positives_co_sentence"],
    )

    bench_kb = build_benchmark_kb_table(ranking_models)
    flips = rank_flips(bench_kb)
    decouple = decoupling_summary(bench_kb)
    cal_decouple = calibration_decoupling(bench_kb, calibration_df)

    cal_summary = {
        "ece_spread": float(calibration_df["ece"].max() - calibration_df["ece"].min()),
        "ece_min": float(calibration_df["ece"].min()),
        "ece_max": float(calibration_df["ece"].max()),
    }

    verdict = _preliminary_assessment(
        ranking_models,
        ranking_baselines,
        decouple,
        cal_summary,
        cal_decouple,
        score_dist,
        ranking_verify,
    )

    export_tables(
        ranking_models,
        ranking_baselines,
        calibration_df,
        calibration_baselines,
        bench_kb,
        flips,
        score_dist,
    )
    plot_benchmark_vs_kb(bench_kb)
    plot_ece_vs_benchmark(bench_kb.merge(calibration_df, on="model_id", how="inner"))
    plot_reliability_diagrams(scores_df)
    plot_score_distributions(scores_df)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pilot_type": "minimal_training",
        "sampling_seed": SAMPLING_SEED,
        "n_models": len(MODELS),
        "train_steps": TRAIN_MAX_STEPS,
        "train_seeds": TRAIN_SEEDS,
        "n_candidates_scored": len(scores_df),
        "ranking_verification": ranking_verify,
        "analytic_random_mrr": float(
            json.loads(STEP03_RANKING_VERIFICATION_JSON.read_text(encoding="utf-8")).get(
                "analytic_random_mrr", 0
            )
        )
        if STEP03_RANKING_VERIFICATION_JSON.exists()
        else analytic_random_mrr(template),
        "verdict": verdict,
        "decoupling": decouple,
        "calibration_decoupling": {k: v for k, v in cal_decouple.items() if k != "table"},
        "score_distribution": score_dist.to_dict(orient="records"),
        "distance_confound": {
            "verdict": distance_confound["verdict"],
            "correlation": distance_confound["correlation"].to_dict(orient="records"),
            "subset_ranking": distance_confound["subset_ranking"].to_dict(orient="records"),
        },
    }
    (OUTPUT_DIR / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== Divergence pilot results ===")
    print(f"  Ranking verification: {verdict['ranking_verification_pass']}")
    print(f"  Scores non-degenerate: {verdict['scores_non_degenerate']}")
    print(f"  Ranking signal: {verdict['ranking_signal_exists']}")
    print(f"  Benchmark-KB decoupled: {verdict['benchmark_kb_decoupled']}")
    print(f"  Calibration OK: {verdict['calibration_nontrivial_and_decoupled']}")
    print(f"  Main experiment justified: {verdict['preliminary_framework_sound']}")

    print(f"  Distance-confound favours: {distance_confound['verdict']['favoured_explanation']}")

    write_report(
        verdict,
        ranking_verify,
        ranking_models,
        ranking_baselines,
        calibration_df,
        calibration_baselines,
        bench_kb,
        flips,
        decouple,
        cal_decouple,
        score_dist,
        distance_confound=distance_confound,
    )
    return summary
