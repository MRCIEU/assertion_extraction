"""Publication-quality Round 1 report (folder-10 matrix data only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import MODELS, REPORT_DIR, TRAIN_SEEDS


def _fmt(val: float | None, lo: float | None = None, hi: float | None = None) -> str:
    if val is None:
        return "not available"
    if lo is not None and hi is not None:
        return f"{val:.3f} (95% interval {lo:.3f} to {hi:.3f})"
    return f"{val:.3f}"


def _variance_lead(vc_row: pd.Series) -> str:
    enc = float(vc_row["encoder_variance_share"])
    seed = float(vc_row["seed_variance_share"])
    if seed > enc:
        return (
            f"Within-encoder seed noise ({seed:.0%}) exceeds between-encoder differences "
            f"({enc:.0%}), so encoder choice moves this axis less than seed luck."
        )
    return (
        f"Between-encoder differences ({enc:.0%}) exceed within-encoder seed noise "
        f"({seed:.0%}), so encoder choice carries measurable signal on this axis."
    )


def write_report(
    *,
    per_run_clean: pd.DataFrame,
    degenerate: pd.DataFrame,
    encoder_primary: pd.DataFrame,
    encoder_all: pd.DataFrame,
    range_check: dict[str, Any],
    mean_corr_primary: pd.DataFrame,
    mean_corr_sensitivity: pd.DataFrame,
    variance_primary: pd.DataFrame,
    seed_assoc_primary: pd.DataFrame,
    ece_corr: pd.DataFrame,
    ece_corr_all: pd.DataFrame,
    sensitivity: pd.DataFrame,
    easy_hard_summary: dict[str, Any],
    pool_size_df: pd.DataFrame | None = None,
    dist_df: pd.DataFrame | None = None,
    roberta_paragraph: str = "",
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "report.md"

    n_clean = len(per_run_clean)
    n_total = len(MODELS) * len(TRAIN_SEEDS)
    pool_size_df = pool_size_df if pool_size_df is not None else pd.DataFrame()
    dist_df = dist_df if dist_df is not None else pd.DataFrame()

    sp_gd = seed_assoc_primary[seed_assoc_primary["pair_type"] == "gene-drug"].iloc[0]
    sp_gdis = seed_assoc_primary[seed_assoc_primary["pair_type"] == "gene-disease"].iloc[0]
    vc_gd = variance_primary[variance_primary["metric"] == "kb_mrr_gene_drug"].iloc[0]
    vc_gdis = variance_primary[variance_primary["metric"] == "kb_mrr_gene_disease"].iloc[0]
    vc_bench = variance_primary[variance_primary["metric"] == "benchmark_f1"].iloc[0]

    gd_means = encoder_primary["kb_mrr_gene_drug_mean"]
    gdis_means = encoder_primary["kb_mrr_gene_disease_mean"]
    gd_lo, gd_hi = float(gd_means.min()), float(gd_means.max())
    gdis_lo, gdis_hi = float(gdis_means.min()), float(gdis_means.max())

    recipe_lr = per_run_clean["recipe_lr"].iloc[0] if "recipe_lr" in per_run_clean.columns else "1e-5"
    recipe_wu = (
        per_run_clean["recipe_warmup_label"].iloc[0]
        if "recipe_warmup_label" in per_run_clean.columns
        else "none"
    )

    lines = [
        "# Round 1: Does benchmark rank predict knowledge-base ranking and calibration?",
        "",
        "## What this round asked",
        "",
        "Round 1 asks descriptively whether a model's self-measured BioRED benchmark rank "
        "aligns with how well it ranks curated relations on the CIViC knowledge base, and "
        "with how well its scores match CIViC curation inclusion. Nine pretrained encoders "
        "were trained under one fixed recipe (BioRED and DrugProt, binary relation presence, "
        "entity-marked text). Eight random seeds per encoder gave seventy-two runs. Only the "
        "encoder and seed changed. Variant pairs were excluded from evaluation because "
        "PubTator cannot build variant candidate pools.",
        "",
        f"Training used the recipe chosen after the formal sweep in folder 10 (learning rate "
        f"{recipe_lr}, warmup {recipe_wu}, checkpoint by best validation F1). Each run was "
        "scored on a held-out BioRED test set and on a frozen CIViC candidate pool. CIViC "
        "never entered training.",
        "",
        "## The data behind this report",
        "",
        f"This report uses {n_total} completed training runs from folder 10, scored at their "
        f"best checkpoints on the frozen step-03 pool. The primary analysis includes "
        f"{n_clean} runs that pass a dynamic degenerate filter (validation or benchmark F1 "
        "at or near zero). No seed numbers were excluded by identity. All quantities come "
        "from stored per-run scores and metrics.",
        "",
        "## Data quality: degenerate training runs",
        "",
    ]

    if not degenerate.empty:
        for _, d in degenerate.iterrows():
            lines.append(
                f"{d['model_id']}, seed {int(d['seed'])}: validation F1 or benchmark F1 "
                "registered at or near zero. These are treated as training failures, not as "
                "evidence about encoder capability."
            )
        lines.append("")
        lines.append(
            "Exclusion uses metrics only, not seed identity. Degenerate runs are omitted "
            "from primary summaries but retained in sensitivity tables."
        )
    else:
        lines.append(
            f"No degenerate runs were flagged. All {n_total} runs, including DeBERTa-base "
            "across eight seeds, trained stably under the chosen recipe and enter the "
            "primary analysis."
        )

    lines.extend(
        [
            "",
            "## Prerequisite: ranking beyond entity proximity",
            "",
        ]
    )

    hard = easy_hard_summary.get("hard", {})
    easy = easy_hard_summary.get("easy", {})
    lines.extend(
        [
            f"On co-occurring entity pairs within a sentence (the easy subset), the proximity-only "
            f"distance ranker reached mean reciprocal rank {easy.get('distance_mrr', 0):.3f}. "
            f"On cross-sentence pairs (the hard subset), its mean reciprocal rank was "
            f"{hard.get('distance_mrr', 0):.3f}. "
            f"Across nine encoders (clean seeds, seed-averaged), {hard.get('n_beats', 0)} of "
            f"nine exceeded the distance ranker on the hard subset and {easy.get('n_beats', 0)} of "
            "nine on the easy subset. Hard-subset performance is the main check that learned "
            "models capture relation signal rather than proximity alone.",
            "",
            "## Benchmark gradient across encoders",
            "",
            f"Among encoder means from clean seeds, self-measured benchmark F1 ranges from "
            f"{range_check['min_f1']:.3f} to {range_check['max_f1']:.3f} (spread "
            f"{range_check['spread']:.3f}). Figure 2 shows seed intervals within each encoder; "
            "seed spread often rivals or exceeds the between-encoder spread.",
            "",
            "## Simpler analysis: nine encoder means (weaker approach)",
            "",
            "A straightforward analysis averages seeds within each encoder and correlates "
            "benchmark F1 with KB mean reciprocal rank across nine points (one per encoder). "
            "This is kept for transparency. Its limitations matter: only nine data points, "
            "sensitivity to outliers, confidence intervals that often span zero, and complete "
            "loss of within-encoder seed variation.",
            "",
        ]
    )

    for pt in ["gene-drug", "gene-disease"]:
        sub = mean_corr_primary[
            (mean_corr_primary["pair_type"] == pt) & (mean_corr_primary["metric"] == "spearman")
        ]
        if not sub.empty:
            r = sub.iloc[0]
            lines.append(
                f"For {pt}, encoder-mean Spearman correlation is "
                f"{_fmt(r['estimate'], r.get('ci_lo'), r.get('ci_hi'))} (nine encoder means, "
                "bootstrap over encoders)."
            )

    lines.extend(
        [
            "",
            "These mean-level associations must not be read as the primary evidence. They "
            "are fragile and can be driven by a small number of encoders.",
            "",
            "## Primary analysis: seed-level variance and uncertainty",
            "",
            "The primary analysis keeps all clean model-by-seed runs and separates variance "
            "between encoders from variance within encoders (seed noise).",
            "",
            "### How much does encoder choice matter?",
            "",
            f"For gene-drug KB ranking, about {vc_gd['encoder_variance_share']*100:.0f}% of "
            f"total variance lies between encoders and {vc_gd['seed_variance_share']*100:.0f}% "
            f"within encoders (seed noise). The intraclass-style ratio is {vc_gd['icc']:.3f}. "
            f"{_variance_lead(vc_gd)}",
            "",
            f"For gene-disease KB ranking, between-encoder share is "
            f"{vc_gdis['encoder_variance_share']*100:.0f}% and within-encoder share "
            f"{vc_gdis['seed_variance_share']*100:.0f}% (ratio {vc_gdis['icc']:.3f}). "
            f"{_variance_lead(vc_gdis)}",
            "",
            f"For benchmark F1, between-encoder share is "
            f"{vc_bench['encoder_variance_share']*100:.0f}% and within-encoder share "
            f"{vc_bench['seed_variance_share']*100:.0f}% (ratio {vc_bench['icc']:.3f}). "
            f"{_variance_lead(vc_bench)}",
            "",
            "### Seed-level benchmark–KB association (encoder clustering respected)",
            "",
            f"Cluster bootstrap over encoders at the seed level gives Spearman "
            f"{_fmt(sp_gd['spearman'], sp_gd.get('ci_lo'), sp_gd.get('ci_hi'))} for gene-drug "
            f"and {_fmt(sp_gdis['spearman'], sp_gdis.get('ci_lo'), sp_gdis.get('ci_hi'))} for "
            "gene-disease. Intervals are wide. Read these together with the variance shares "
            "above rather than as standalone pass-fail tests.",
            "",
            "## Headline: pair-type-specific pattern from the new data",
            "",
            f"Gene-drug KB mean reciprocal rank spans {gd_lo:.3f} to {gd_hi:.3f} across the "
            f"nine encoder means. {_variance_lead(vc_gd)} Benchmark rank carries limited "
            "information for gene-drug under this recipe.",
            "",
            f"Gene-disease KB mean reciprocal rank spans {gdis_lo:.3f} to {gdis_hi:.3f} "
            f"({gdis_means.max() - gdis_means.min():.3f} from lowest to highest encoder mean). "
            f"{_variance_lead(vc_gdis)} Read this pair-type pattern from the numbers above "
            "rather than assuming the old recipe's tilt carries over.",
            "",
            "## Calibration behaves differently from ranking",
            "",
        ]
    )

    ece_sp = ece_corr[ece_corr["metric"] == "spearman"]
    if not ece_sp.empty:
        r = ece_sp.iloc[0]
        lines.append(
            f"At the nine encoder means (clean seeds), higher benchmark F1 associates with "
            f"lower expected calibration error (Spearman "
            f"{_fmt(r['estimate'], r.get('ci_lo'), r.get('ci_hi'))}). "
            "Ranking and calibration may therefore tell different stories: benchmark score "
            "can track curation inclusion more closely than it tracks KB rank."
        )
    lines.append(
        "Expected calibration error is measured against CIViC curation inclusion, not against "
        "objective biomedical truth. A confident score on a pair CIViC did not curate may "
        "reflect an uncurated true relation rather than model error."
    )

    lines.extend(["", "## Distance-confound diagnostic", ""])
    if not dist_df.empty and "spearman_r" in dist_df.columns:
        med_sp = float(dist_df["spearman_r"].median())
        lines.append(
            f"Across runs, the median Spearman correlation between model scores and entity "
            f"proximity is {med_sp:.3f}. Values nearer one suggest ranking tracks closeness in "
            "the abstract; values nearer zero suggest scores carry signal beyond proximity. "
            "This is descriptive, not a thresholded test."
        )
    else:
        lines.append(
            "Per-run correlations between model scores and entity proximity are stored with "
            "the Round 1 outputs for inspection."
        )

    lines.extend(["", "## Candidate-pool-size robustness", ""])
    if not pool_size_df.empty:
        for pt in ["gene-drug", "gene-disease"]:
            sub = pool_size_df[pool_size_df["pair_type"] == pt]
            if sub.empty:
                continue
            mean_sp = float(sub["spearman_r"].mean())
            med_sp = float(sub["spearman_r"].median())
            lines.append(
                f"For {pt}, correlating per-abstract pool size with per-abstract MRR across "
                f"runs gives mean Spearman {mean_sp:.3f} (median {med_sp:.3f}). "
            )
        lines.append(
            "This is an indirect proxy: it tests whether observed pool size (how many "
            "candidates PubTator placed in the frozen pool for each abstract) drives the "
            "ranking metric. It does not measure distractors PubTator missed entirely, which "
            "remain unobservable. The NER-recall limitation therefore stays in the "
            "limitations section even when this check is weak."
        )
    else:
        lines.append(
            "Pool-size robustness tables are produced with the Round 1 analysis outputs."
        )

    if not degenerate.empty:
        lines.extend(
            [
                "",
                "## Sensitivity: including degenerate runs",
                "",
                "Repeating summaries while including degenerate runs shifts encoder means where "
                "failures occurred and can widen apparent between-encoder spread on the "
                "benchmark axis.",
            ]
        )

    lines.extend(
        [
            "",
            "## What Round 1 does and does not show",
            "",
            "Round 1 describes whether benchmark rank aligns with KB ranking and calibration "
            "on unseen CIViC evidence under one training recipe. Either alignment or "
            "divergence is a valid finding. The primary seed-level analysis, variance shares, "
            "and wide intervals should be read with uncertainty: nine encoders at the mean "
            "level, seed noise on KB axes, and incomplete distractor sets from NER recall.",
            "",
            "Round 1 does not establish a strong predictive rule from benchmark rank to KB "
            "rank unless the data support it with intervals. It also does not rule out weak "
            "or pair-specific structure.",
        ]
    )

    if roberta_paragraph:
        lines.extend(["", "## RoBERTa versus domain-specialised encoders", "", roberta_paragraph])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report -> {path}")
    return path


def write_readme(
    *,
    per_run_clean: pd.DataFrame,
    degenerate: pd.DataFrame,
    range_check: dict[str, Any],
    variance_primary: pd.DataFrame,
    seed_assoc_primary: pd.DataFrame,
    eh_summary: dict[str, Any],
    roberta: dict[str, Any],
) -> Path:
    """Short plain-language README with key numbers (written after analysis)."""
    path = REPORT_DIR / "README.md"
    n_total = len(MODELS) * len(TRAIN_SEEDS)
    vc_gd = variance_primary[variance_primary["metric"] == "kb_mrr_gene_drug"].iloc[0]
    vc_gdis = variance_primary[variance_primary["metric"] == "kb_mrr_gene_disease"].iloc[0]
    sp_gd = seed_assoc_primary[seed_assoc_primary["pair_type"] == "gene-drug"].iloc[0]
    hard = eh_summary.get("hard", {})

    lines = [
        "# Round 1 analysis (folder 11)",
        "",
        "Consumes folder-10 matrix checkpoints. No training.",
        "",
        "## Key numbers",
        "",
        f"- Runs: {n_total} trained, {len(per_run_clean)} clean in primary analysis "
        f"({len(degenerate)} degenerate flagged by metrics)",
        f"- Benchmark F1 encoder-mean range: {range_check['min_f1']:.3f} to "
        f"{range_check['max_f1']:.3f} (spread {range_check['spread']:.3f})",
        f"- KB gene-drug variance: {vc_gd['encoder_variance_share']:.0%} encoder, "
        f"{vc_gd['seed_variance_share']:.0%} seed (ICC {vc_gd['icc']:.3f})",
        f"- KB gene-disease variance: {vc_gdis['encoder_variance_share']:.0%} encoder, "
        f"{vc_gdis['seed_variance_share']:.0%} seed (ICC {vc_gdis['icc']:.3f})",
        f"- Seed-level benchmark–KB Spearman (gene-drug): {sp_gd['spearman']:.3f} "
        f"[{sp_gd.get('ci_lo', float('nan')):.3f}, {sp_gd.get('ci_hi', float('nan')):.3f}]",
        f"- Encoders beating distance ranker on hard subset: {hard.get('n_beats', 0)}/9",
        f"- RoBERTa pattern holds: {roberta.get('pattern_holds', 'see 11_roberta_analysis.csv')}",
        "",
        "## Workflow",
        "",
        "1. GPU scoring: `sbatch step_score.sbatch` (resumable via scoring_complete markers)",
        "2. CPU analysis after 72/72 markers: `sbatch step_analyze.sbatch`",
        "",
        "Full prose: `report.md`. Figures: `../../figures/11_round1_analysis/`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"README -> {path}")
    return path
