"""Generate Round 1 report (descriptive, no pass/fail language)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import MODELS, OUTPUT_DIR, REPORT_DIR, TRAIN_SEEDS


def _fmt_ci(val: float | None, lo: float | None, hi: float | None) -> str:
    if val is None:
        return "n/a"
    if lo is not None and hi is not None:
        return f"{val:.3f} (95% CI {lo:.3f}–{hi:.3f})"
    return f"{val:.3f}"


def write_report(
    per_run: pd.DataFrame,
    encoder_df: pd.DataFrame,
    range_check: dict[str, Any],
    corr_rows: list[dict],
    ece_corr: dict[str, Any],
    noise: pd.DataFrame,
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "report.md"

    n_complete = len(per_run)
    n_expected = len(MODELS) * len(TRAIN_SEEDS)

    lines = [
        "# Round 1: Does benchmark rank predict KB ranking and calibration?",
        "",
        "## Design",
        "",
        "This is the first main-experiment round. Training is fixed: BioRED plus DrugProt, "
        "binary relation-presence labels, entity-marked inputs, validation early stopping. "
        "Only the encoder and random seed vary.",
        "",
        f"- Encoders: {len(MODELS)} architectures spanning domain-specialised and general models",
        f"- Seeds per encoder: {len(TRAIN_SEEDS)}",
        f"- Completed runs: {n_complete} / {n_expected}",
        "",
        "Two axes from each checkpoint:",
        "",
        "1. **Benchmark axis** — self-measured BioRED test presence F1 under one unified protocol",
        "2. **KB axis** — ranking quality on the frozen CIViC candidate pool (gene-drug and gene-disease separately)",
        "",
        "CIViC is never used for training. Calibration ground truth is CIViC curation inclusion, "
        "not objective biomedical truth.",
        "",
        "## Benchmark gradient check",
        "",
        f"Self-measured benchmark F1 spans **{range_check['min_f1']:.3f}–{range_check['max_f1']:.3f}** "
        f"(spread {range_check['spread']:.3f}) across encoder means.",
    ]

    if not range_check.get("wide_enough"):
        lines.append(
            "\nThe measured gradient is narrow; this round may under-test whether benchmark rank "
            "predicts KB performance."
        )
    else:
        lines.append("\nThe measured gradient is wide enough to compare encoders meaningfully.")

    lines.extend(["", "## A. Ranking validity (easy vs hard subsets)", ""])

    hard_path = OUTPUT_DIR / "10_easy_hard_ranking.csv"
    if hard_path.exists():
        subset = pd.read_csv(hard_path)
        hard = subset[subset["subset"] == "hard_cross_sentence"]
        dr_mrr = hard[hard["model_id"] == "distance_ranker"]["mrr"].mean()
        model_hard = hard[hard["model_id"] != "distance_ranker"].groupby("model_id")["mrr"].mean()
        beats = (model_hard > dr_mrr).sum()
        lines.append(
            f"On the **hard** (cross-sentence) subset, the distance ranker MRR is {dr_mrr:.3f}. "
            f"{beats} of {len(model_hard)} encoders (seed-averaged) exceed this baseline, "
            "indicating relation signal beyond proximity where it matters most."
        )
    else:
        lines.append("_Easy/hard subset table not yet available._")

    lines.extend(["", "## B. Benchmark vs KB ranking", ""])

    for row in corr_rows:
        if row.get("metric") != "spearman":
            continue
        pt = row["pair_type"]
        est = row.get("estimate")
        lo = row.get("ci_lo")
        hi = row.get("ci_hi")
        lines.append(
            f"- **{pt}**: Spearman ρ = {_fmt_ci(est, lo, hi)} "
            f"(benchmark F1 vs KB MRR, encoder-level means, n={row.get('n', 'n/a')})"
        )

    if not noise.empty:
        row = noise.iloc[0]
        lines.extend(
            [
                "",
                f"Between-encoder SD in benchmark F1 ({row.get('between_encoder_sd', 0):.4f}) "
                f"vs mean within-encoder seed SD ({row.get('mean_within_encoder_sd', 0):.4f}): "
                f"{'encoder differences exceed seed noise' if row.get('between_exceeds_within') else 'seed noise is comparable to encoder differences'}.",
            ]
        )

    lines.extend(["", "## C. Calibration vs CIViC inclusion", ""])

    sp = ece_corr.get("spearman", {})
    lines.append(
        f"Spearman correlation between benchmark F1 and ECE: "
        f"{_fmt_ci(sp.get('estimate'), sp.get('ci_lo'), sp.get('ci_hi'))}. "
        "Lower ECE indicates better calibration against CIViC inclusion. "
        "A confident score on a non-CIViC-curated pair may reflect an uncurated true relation, not necessarily model error."
    )

    lines.extend(["", "## D. Distance-confound diagnostic", ""])

    prox_path = OUTPUT_DIR / "10_distance_score_correlation.csv"
    if prox_path.exists():
        prox = pd.read_csv(prox_path)
        mean_r = prox["pearson_r"].mean()
        lines.append(
            f"Mean Pearson correlation between model scores and entity proximity: **{mean_r:.3f}**. "
            "Higher values suggest ranking leans on proximity."
        )
    else:
        lines.append("_Distance correlation table not yet available._")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "This round describes whether self-measured benchmark rank aligns with KB ranking "
            "and calibration on CIViC. Either alignment or divergence is a valid finding; "
            "effect sizes and confidence intervals above should be read descriptively, not as pass/fail criteria.",
            "",
            "### Encoder means (benchmark F1 and KB MRR)",
            "",
        ]
    )

    if not encoder_df.empty:
        cols = ["short_name", "benchmark_f1_mean", "kb_mrr_gene_drug_mean", "kb_mrr_gene_disease_mean", "ece_mean"]
        show = [c for c in cols if c in encoder_df.columns]
        lines.append(encoder_df[show].sort_values("benchmark_f1_mean", ascending=False).to_markdown(index=False, floatfmt=".3f"))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report -> {path}")
    return path
