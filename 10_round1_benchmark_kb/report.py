"""Generate Round 1 report (descriptive, no pass/fail language)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import MODELS, OUTPUT_DIR, REPORT_DIR, TRAIN_LR, TRAIN_SEEDS, TRAINING_STRATEGY, TRAIN_WARMUP_RATIO


def _fmt_ci(val: float | None, lo: float | None, hi: float | None) -> str:
    if val is None:
        return "n/a"
    if lo is not None and hi is not None:
        return f"{val:.3f} (95% CI {lo:.3f}–{hi:.3f})"
    return f"{val:.3f}"


def _simple_table(df: pd.DataFrame, float_cols: set[str] | None = None) -> str:
    float_cols = float_cols or set()
    if df.empty:
        return "_No data._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in float_cols or isinstance(v, float):
                cells.append(f"{float(v):.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    per_run: pd.DataFrame,
    encoder_df: pd.DataFrame,
    range_check: dict[str, Any],
    corr_rows: list[dict],
    ece_corr: dict[str, Any],
    noise: pd.DataFrame,
    degenerate: pd.DataFrame | None = None,
    sens_rows: list[dict] | None = None,
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
        "This round asks a symmetric question: does a model's self-measured BioRED benchmark rank "
        "predict its ranking quality and score calibration on the frozen CIViC candidate pool — or not? "
        "Either alignment or divergence is a valid descriptive finding.",
        "",
        "Training is fixed for all models: BioRED plus DrugProt (leakage-excluded), binary "
        "relation-presence labels, entity-marked inputs. Only the encoder and random seed vary.",
        "",
        f"- Encoders: {len(MODELS)} architectures (domain-specialised and general)",
        f"- Seeds per encoder: {len(TRAIN_SEEDS)} ({min(TRAIN_SEEDS)}–{max(TRAIN_SEEDS)})",
        f"- Completed runs: {n_complete} / {n_expected}",
        "",
        "### Fixed training strategy (from objective sweep)",
        "",
        f"- Learning rate: **{TRAIN_LR:.0e}**, **no warmup**",
        "- Checkpoint selection: **best validation F1** (not validation loss)",
        "- Early stopping: max 10 epochs, patience 3, on validation F1 plateau",
        f"- Strategy tag: `{TRAINING_STRATEGY}`",
        "",
        "Two evaluation axes from the **same** checkpoint:",
        "",
        "1. **Benchmark axis** — self-measured BioRED test presence F1 (unified protocol; not paper-reported values)",
        "2. **KB axis** — ranking on the frozen CIViC pool (gene-drug and gene-disease separately)",
        "",
        "CIViC is never used for training. It defines the KB evaluation universe and calibration reference "
        "(curation inclusion, not objective biomedical truth).",
        "",
        "## Benchmark gradient (9 encoders)",
        "",
        f"Across encoder means, self-measured benchmark F1 ranges from **{range_check['min_f1']:.3f}** "
        f"to **{range_check['max_f1']:.3f}** (spread **{range_check['spread']:.3f}**).",
    ]

    if "encoder_f1_values" in range_check:
        lines.append("")
        lines.append("Per-encoder mean benchmark F1:")
        for name, val in range_check["encoder_f1_values"]:
            lines.append(f"- {name}: {val:.3f}")

    spread = range_check.get("spread", 0)
    lines.append("")
    if spread < 0.05:
        lines.append(
            "The benchmark gradient is **narrow** (spread below 0.05). Any rank-correlation between "
            "benchmark and KB performance will have wide confidence intervals; correlational conclusions "
            "are correspondingly **uncertain**. This is reported as a limitation, not hidden."
        )
    elif spread < 0.08:
        lines.append(
            "The benchmark gradient is **moderate** (spread between 0.05 and 0.08). Correlations are "
            "estimable but may remain sensitive to a few encoders; confidence intervals should be read carefully."
        )
    else:
        lines.append(
            "The benchmark gradient is **substantial** (spread ≥ 0.08). Encoder ordering on the benchmark "
            "axis is distinguishable, though correlational strength with KB performance is still an empirical question."
        )

    lines.extend(["", "## A. Ranking validity (easy vs hard subsets)", ""])

    hard_path = OUTPUT_DIR / "10_easy_hard_ranking.csv"
    easy_path = hard_path
    if hard_path.exists():
        subset = pd.read_csv(hard_path)
        hard = subset[subset["subset"] == "hard_cross_sentence"]
        easy = subset[subset["subset"] == "easy_co_sentence"]
        dr_hard = hard[hard["model_id"] == "distance_ranker"]["mrr"].mean()
        dr_easy = easy[easy["model_id"] == "distance_ranker"]["mrr"].mean()
        model_hard = hard[hard["model_id"] != "distance_ranker"].groupby("model_id")["mrr"].mean()
        model_easy = easy[easy["model_id"] != "distance_ranker"].groupby("model_id")["mrr"].mean()
        beats_hard = int((model_hard > dr_hard).sum())
        beats_easy = int((model_easy > dr_easy).sum())
        lines.append(
            f"Distance ranker MRR: **easy** (co-sentence) {dr_easy:.3f}, **hard** (cross-sentence) {dr_hard:.3f}. "
            f"Among {len(model_hard)} encoders (seed-averaged), **{beats_hard}** beat the distance ranker on the "
            f"hard subset and **{beats_easy}** on the easy subset."
        )
        lines.append(
            "Hard-subset performance is the primary check that models capture relation signal beyond entity proximity."
        )
    else:
        lines.append("_Easy/hard subset table not yet available._")

    if degenerate is not None and not degenerate.empty:
        lines.extend(
            [
                "",
                "## Data quality note",
                "",
                f"**{len(degenerate)}** run(s) had validation F1 = 0 or benchmark F1 = 0 "
                f"(listed in `10_degenerate_runs.csv`). These are included in primary tables; "
                "sensitivity correlations excluding them are in `10_benchmark_kb_correlations_sensitivity.csv`.",
            ]
        )

    lines.extend(["", "## B. Benchmark vs KB ranking", ""])

    for row in corr_rows:
        if row.get("metric") != "spearman":
            continue
        pt = row["pair_type"]
        est = row.get("estimate")
        lo = row.get("ci_lo")
        hi = row.get("ci_hi")
        n_enc = row.get("n", "n/a")
        lines.append(
            f"- **{pt}**: Spearman ρ = {_fmt_ci(est, lo, hi)} "
            f"(benchmark F1 vs KB MRR, encoder means, n={n_enc})"
        )
        if lo is not None and hi is not None and lo <= 0 <= hi:
            lines.append(
                f"  - The confidence interval for {pt} spans zero; the correlation is "
                f"**not conclusively different from zero**."
            )

    if not noise.empty:
        for _, nrow in noise.iterrows():
            metric = nrow.get("metric", "metric")
            between = nrow.get("between_encoder_sd", 0)
            within = nrow.get("mean_within_encoder_sd", 0)
            exceeds = nrow.get("between_exceeds_within")
            lines.extend(
                [
                    "",
                    f"**Encoder vs seed noise ({metric}):** between-encoder SD = {between:.4f}, "
                    f"mean within-encoder seed SD = {within:.4f}. "
                    f"Encoder differences {'exceed' if exceeds else 'do not clearly exceed'} seed noise.",
                ]
            )

    if sens_rows:
        lines.append("")
        lines.append("Sensitivity (Spearman, encoder means):")
        for row in sens_rows:
            if row.get("metric") != "spearman":
                continue
            lines.append(
                f"- {row['analysis_set']}, {row['pair_type']}: "
                f"{_fmt_ci(row.get('estimate'), row.get('ci_lo'), row.get('ci_hi'))} "
                f"(n={row.get('n_encoders')})"
            )

    flip_note = OUTPUT_DIR / "10_rank_flips_gene_drug.csv"
    if flip_note.exists():
        lines.append("")
        lines.append("Concrete rank-flip cases (benchmark rank ≠ KB rank) are listed in "
                     "`outputs/10_round1_benchmark_kb/10_rank_flips_*.csv`.")

    lines.extend(["", "## C. Calibration vs CIViC inclusion", ""])

    sp = ece_corr.get("spearman", {})
    lines.append(
        f"Spearman correlation between benchmark F1 and ECE: "
        f"{_fmt_ci(sp.get('estimate'), sp.get('ci_lo'), sp.get('ci_hi'))}. "
        "Lower ECE indicates better calibration against CIViC curation inclusion. "
        "A confident score on a non-CIViC-curated pair may reflect an uncurated true relation, not necessarily model error."
    )

    base_path = OUTPUT_DIR / "10_calibration_baselines.csv"
    if base_path.exists():
        bases = pd.read_csv(base_path)
        lines.append("")
        lines.append("Trivial calibration baselines (ECE) for context:")
        for _, r in bases.iterrows():
            lines.append(f"- {r['model_or_baseline']}: {r['ece']:.3f}")

    lines.extend(["", "## D. Distance-confound diagnostic", ""])

    prox_path = OUTPUT_DIR / "10_distance_score_correlation.csv"
    if prox_path.exists():
        prox = pd.read_csv(prox_path)
        mean_r = prox["pearson_r"].mean()
        lines.append(
            f"Mean Pearson correlation between model scores and entity proximity: **{mean_r:.3f}** "
            f"(across {len(prox)} model×seed runs). Higher values indicate ranking leans on proximity."
        )
    else:
        lines.append("_Distance correlation table not yet available._")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "This report presents four analysis dimensions (ranking validity, benchmark–KB association, "
            "calibration, distance confound) without pre-ranking their importance. Whether benchmark rank "
            "predicts KB ranking and calibration is answered by the effect sizes and confidence intervals above — "
            "alignment and divergence are both valid outcomes.",
            "",
            "### Encoder means (benchmark F1 and KB MRR)",
            "",
        ]
    )

    if not encoder_df.empty:
        show = encoder_df[
            [c for c in [
                "short_name",
                "benchmark_f1_mean",
                "kb_mrr_gene_drug_mean",
                "kb_mrr_gene_disease_mean",
                "ece_mean",
            ] if c in encoder_df.columns]
        ].sort_values("benchmark_f1_mean", ascending=False)
        lines.append(
            _simple_table(
                show,
                float_cols={"benchmark_f1_mean", "kb_mrr_gene_drug_mean", "kb_mrr_gene_disease_mean", "ece_mean"},
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report -> {path}")
    return path
