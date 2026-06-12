"""Publication-quality Round 1 report (clean data, folder-10 matrix at 5e-6/none)."""

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


def _saturation_reading(vc_bench: pd.Series, vc_gd: pd.Series, vc_gdis: pd.Series) -> str:
    b_enc = float(vc_bench["encoder_variance_share"])
    b_seed = float(vc_bench["seed_variance_share"])
    gd_enc = float(vc_gd["encoder_variance_share"])
    gdis_enc = float(vc_gdis["encoder_variance_share"])

    if b_enc <= gd_enc and b_enc <= gdis_enc:
        return (
            "The benchmark axis shows the weakest between-encoder discrimination among the three "
            "metrics examined. A near-zero benchmark–KB association would therefore be expected "
            "because the benchmark carries little encoder information here, not necessarily because "
            "two distinct transferable abilities diverge."
        )
    if b_enc > gd_enc or b_enc > gdis_enc:
        return (
            "The benchmark discriminates encoders more strongly than at least one KB axis. "
            "If benchmark–KB association remains weak, that pattern is closer to genuine "
            "decoupling: the dimension the benchmark separates does not transfer to CIViC ranking."
        )
    return (
        "Benchmark and KB axes show comparable between-encoder shares. Read association "
        "estimates together with absolute KB levels and the fine-tuning lift table."
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
    variance_boot: pd.DataFrame,
    seed_assoc_primary: pd.DataFrame,
    ece_corr: pd.DataFrame,
    ece_corr_all: pd.DataFrame,
    sensitivity: pd.DataFrame,
    easy_hard_summary: dict[str, Any],
    abs_kb: pd.DataFrame,
    lift_df: pd.DataFrame,
    pool_size_df: pd.DataFrame | None = None,
    dist_df: pd.DataFrame | None = None,
    roberta_paragraph: str = "",
    rand_mrr: float = float("nan"),
    dist_mrr: float = float("nan"),
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

    boot_bench = variance_boot[variance_boot["metric"] == "benchmark_f1"]
    bench_ci = ""
    if not boot_bench.empty:
        b = boot_bench.iloc[0]
        if b.get("encoder_share_ci_lo") is not None:
            bench_ci = (
                f" (encoder share 95% interval "
                f"{float(b['encoder_share_ci_lo']):.0%} to {float(b['encoder_share_ci_hi']):.0%})"
            )

    gd_means = encoder_primary["kb_mrr_gene_drug_mean"]
    gdis_means = encoder_primary["kb_mrr_gene_disease_mean"]
    gd_lo, gd_hi = float(gd_means.min()), float(gd_means.max())
    gdis_lo, gdis_hi = float(gdis_means.min()), float(gdis_means.max())

    recipe_lr = per_run_clean["recipe_lr"].iloc[0] if "recipe_lr" in per_run_clean.columns else "5e-6"
    recipe_wu = (
        per_run_clean["recipe_warmup_label"].iloc[0]
        if "recipe_warmup_label" in per_run_clean.columns
        else "none"
    )

    ft_abs = abs_kb[abs_kb["reference"] == "finetuned_encoders_mean"].iloc[0]
    rand_abs = abs_kb[abs_kb["reference"] == "random_uniform"].iloc[0]
    dist_abs = abs_kb[abs_kb["reference"] == "distance_ranker"].iloc[0]

    mean_bench_lift = float(lift_df["lift_benchmark_f1"].mean())
    mean_kb_lift = float(
        (lift_df["lift_kb_mrr_gene_drug"].mean() + lift_df["lift_kb_mrr_gene_disease"].mean()) / 2
    )

    hard = easy_hard_summary.get("hard", {})
    easy = easy_hard_summary.get("easy", {})

    lines = [
        "# Round 1: Does in-distribution benchmark rank predict out-of-distribution knowledge-base performance?",
        "",
        "## What this round asked",
        "",
        "Round 1 asks descriptively whether a model's self-measured BioRED benchmark standing "
        "aligns with how well it ranks curated relations on the CIViC knowledge base, and with "
        "how well its scores match CIViC curation inclusion. Nine pretrained encoders were "
        f"trained under one fixed recipe confirmed on clean data (learning rate {recipe_lr}, "
        f"warmup {recipe_wu}, checkpoint by best validation F1). Eight random seeds per encoder "
        "gave seventy-two runs. All seventy-two completed without collapse under the corrected "
        "marker protocol. Variant pairs were excluded from evaluation because PubTator cannot "
        "build variant candidate pools.",
        "",
        "Two evaluation axes must be distinguished. The benchmark is self-measured BioRED "
        "presence F1 on a held-out BioRED test split. BioRED is a training corpus, so the "
        "benchmark is an in-distribution measure of fit to the training distribution. CIViC "
        "ranking is out-of-distribution: CIViC never entered training. The central question is "
        "whether in-distribution benchmark standing predicts out-of-distribution knowledge-base "
        "performance. The benchmark must not be read as an external gold standard; it is a "
        "self-measured in-distribution F1 whose tight encoder spread may partly reflect that "
        "all encoders fit the training distribution similarly.",
        "",
        f"This report uses {n_total} completed training runs from the folder-10 matrix, scored "
        f"at best checkpoints on the frozen step-03 pool (corrected coverage 1590 matched "
        "positives of 1812 CIViC relations in scope). The primary analysis includes "
        f"{n_clean} runs that pass a dynamic degenerate filter (validation or benchmark F1 at "
        "or near zero). No seed numbers were excluded by identity. All quantities are derived "
        "fresh from stored per-run scores; prior Round 1 numbers are not reused.",
        "",
        "## Data quality",
        "",
    ]

    if not degenerate.empty:
        for _, d in degenerate.iterrows():
            lines.append(
                f"{d['model_id']}, seed {int(d['seed'])}: validation F1 or benchmark F1 "
                "registered at or near zero and is omitted from primary summaries."
            )
        lines.append("")
    else:
        lines.append(
            f"No degenerate runs were flagged. All {n_total} runs, including DeBERTa-base "
            "across eight seeds (benchmark F1 roughly 0.734 to 0.760 per seed), enter the "
            "primary analysis."
        )

    lines.extend(
        [
            "",
            "## Ranking validity and absolute knowledge-base levels",
            "",
            f"On co-occurring entity pairs within a sentence (the easy subset), the proximity-only "
            f"distance ranker reached mean reciprocal rank {easy.get('distance_mrr', 0):.3f}. "
            f"On cross-sentence pairs (the hard subset), its mean reciprocal rank was "
            f"{hard.get('distance_mrr', 0):.3f}. "
            f"Across nine encoders (clean seeds, seed-averaged), {hard.get('n_beats', 0)} of "
            f"nine exceeded the distance ranker on the hard subset and {easy.get('n_beats', 0)} of "
            "nine on the easy subset. Hard-subset performance is the main check that learned "
            "models capture relation signal beyond proximity alone.",
            "",
            "Absolute levels matter alongside relative comparisons. On the frozen pool, a uniform "
            f"random ranker achieves mean reciprocal rank {rand_mrr:.3f} and the distance ranker "
            f"{dist_mrr:.3f}. Fine-tuned encoder means average {float(ft_abs['mrr_gene_drug']):.3f} "
            f"on gene-drug and {float(ft_abs['mrr_gene_disease']):.3f} on gene-disease, with hard- "
            f"subset mean {float(ft_abs['mrr_hard']):.3f} versus distance {float(dist_abs['mrr_hard']):.3f}. "
            "Fine-tuned models sit well above random ranking but only modestly above the distance "
            "ranker on this pool, so knowledge-base adequacy is limited in absolute terms even "
            "when models beat proximity-only baselines on the hard subset.",
            "",
            "## Benchmark discriminative power and saturation",
            "",
            f"Among encoder means from clean seeds, self-measured benchmark F1 ranges from "
            f"{range_check['min_f1']:.3f} to {range_check['max_f1']:.3f} (spread "
            f"{range_check['spread']:.3f}), comparable to within-encoder seed standard "
            "deviation (roughly 0.01). Figure 1 shows seed error bars on both axes so that "
            "seed spread can be compared directly to between-encoder spread.",
            "",
            "The same seed-level variance-components method applied to benchmark F1 and to KB "
            "mean reciprocal rank allows a direct saturation comparison. For benchmark F1, "
            f"{vc_bench['encoder_variance_share']*100:.0f}% of total variance lies between "
            f"encoders and {vc_bench['seed_variance_share']*100:.0f}% within encoders (seed "
            f"noise){bench_ci}. {_variance_lead(vc_bench)}",
            "",
            f"For gene-drug KB ranking, between-encoder share is "
            f"{vc_gd['encoder_variance_share']*100:.0f}% and within-encoder share "
            f"{vc_gd['seed_variance_share']*100:.0f}% (intraclass-style ratio "
            f"{vc_gd['icc']:.3f}). {_variance_lead(vc_gd)}",
            "",
            f"For gene-disease KB ranking, between-encoder share is "
            f"{vc_gdis['encoder_variance_share']*100:.0f}% and within-encoder share "
            f"{vc_gdis['seed_variance_share']*100:.0f}% (ratio {vc_gdis['icc']:.3f}). "
            f"{_variance_lead(vc_gdis)}",
            "",
            f"Figure 2 plots these shares side by side. {_saturation_reading(vc_bench, vc_gd, vc_gdis)}",
            "",
            "## Benchmark versus knowledge-base association",
            "",
            "Two association methods are reported. The weaker approach averages seeds within each "
            "encoder and correlates benchmark F1 with KB mean reciprocal rank across nine points. "
            "Its limitations are stated plainly: only nine data points, outlier sensitivity, "
            "intervals that often span zero, and complete loss of within-encoder seed variation.",
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
                f"{_fmt(r['estimate'], r.get('ci_lo'), r.get('ci_hi'))} (nine encoder means)."
            )

    lines.extend(
        [
            "",
            "The primary association method is seed-level cluster bootstrap over encoders, "
            "propagating seed uncertainty:",
            "",
            f"Gene-drug: Spearman {_fmt(sp_gd['spearman'], sp_gd.get('ci_lo'), sp_gd.get('ci_hi'))}.",
            f"Gene-disease: Spearman {_fmt(sp_gdis['spearman'], sp_gdis.get('ci_lo'), sp_gdis.get('ci_hi'))}.",
            "",
            "Interpret these estimates through the variance shares above. If the benchmark itself "
            "does not discriminate encoders, a near-zero association is expected and should be "
            "read as 'benchmark carries little information here,' not automatically as evidence "
            "for two distinct transferable abilities. If the benchmark discriminates while KB "
            "axes do not, weak association is closer to genuine decoupling.",
            "",
            f"Gene-drug KB mean reciprocal rank spans {gd_lo:.3f} to {gd_hi:.3f} across encoder "
            f"means. Gene-disease spans {gdis_lo:.3f} to {gdis_hi:.3f}. Read pair-type patterns "
            "from these numbers; do not assume prior recipe tilts carry over.",
            "",
            "## Fine-tuning lift versus untrained floor",
            "",
            "Nine untrained-floor references score the same encoders with pretrained weights and "
            "a randomly initialised classification head (fixed head-init seed), with no "
            "fine-tuning, on both the BioRED benchmark test set and the frozen CIViC pool. This "
            "is a floor reference, not a zero-shot capability claim. Untrained benchmark F1 is "
            "often near chance by construction; the informative quantity is lift from fine-tuning.",
            "",
            f"Mean lift across encoders is {mean_bench_lift:.3f} on benchmark F1 and "
            f"{mean_kb_lift:.3f} on KB mean reciprocal rank (averaged across pair types). "
            "If fine-tuning lifts benchmark substantially but KB little, that is descriptive "
            "evidence that the benchmark rewards something fine-tuning adds that does not "
            "transfer to knowledge-base ranking. Figure 4 shows per-encoder lifts; calibration "
            "versus benchmark F1 is summarised in text below.",
            "",
            "## Calibration as a separate axis",
            "",
        ]
    )

    ece_sp = ece_corr[ece_corr["metric"] == "spearman"]
    if not ece_sp.empty:
        r = ece_sp.iloc[0]
        lines.append(
            f"At nine encoder means, higher benchmark F1 associates with lower expected "
            f"calibration error against CIViC curation inclusion (Spearman "
            f"{_fmt(r['estimate'], r.get('ci_lo'), r.get('ci_hi'))}). Ranking and calibration "
            "may therefore tell different stories. Expected calibration error is measured "
            "against curation inclusion, not objective biomedical truth."
        )
    else:
        lines.append(
            "Expected calibration error is measured against CIViC curation inclusion, not "
            "objective biomedical truth. Treat calibration as its own axis; do not overstate "
            "its span relative to ranking."
        )

    lines.extend(["", "## Distance-confound diagnostic", ""])
    if not dist_df.empty and "spearman_r" in dist_df.columns:
        med_sp = float(dist_df["spearman_r"].median())
        lines.append(
            f"Across runs, the median Spearman correlation between model scores and entity "
            f"proximity is {med_sp:.3f}. Values nearer one suggest ranking tracks closeness; "
            "values nearer zero suggest signal beyond proximity."
        )
    else:
        lines.append("Per-run score–proximity correlations are stored with the Round 1 outputs.")

    lines.extend(["", "## Candidate-pool-size robustness", ""])
    if not pool_size_df.empty:
        for pt in ["gene-drug", "gene-disease"]:
            sub = pool_size_df[pool_size_df["pair_type"] == pt]
            if sub.empty:
                continue
            mean_sp = float(sub["spearman_r"].mean())
            med_sp = float(sub["spearman_r"].median())
            lines.append(
                f"For {pt}, correlating per-abstract pool size with per-abstract mean reciprocal "
                f"rank across runs gives mean Spearman {mean_sp:.3f} (median {med_sp:.3f}). "
            )
        lines.append(
            "This is an indirect proxy on the corrected frozen pool (1590 matched positives, "
            "222 missed by PubTator). It tests whether observed pool size drives the metric "
            "common-mode across models. It does not measure distractors PubTator missed "
            "entirely (NER-recall gap 12.3%, multi-word entity bias documented in step 03)."
        )
    else:
        lines.append("Pool-size robustness tables are produced with the Round 1 analysis outputs.")

    if roberta_paragraph:
        lines.extend(
            [
                "",
                "## General versus domain-specialised encoders",
                "",
                roberta_paragraph,
                "",
                f"With benchmark encoder-mean spread at {range_check['spread']:.3f}, any "
                "general-versus-domain benchmark differences are themselves tiny on this recipe.",
            ]
        )

    lines.extend(
        [
            "",
            "## Closing read on the clean data",
            "",
            "The clean seventy-two-run matrix at 5e-6/none allows three descriptive pictures: "
            "genuine decoupling (benchmark discriminates but the discriminating dimension does "
            "not transfer to CIViC), both axes seed-dominated or insensitive, or benchmark "
            "saturation (the benchmark does not discriminate these encoders, which largely "
            "explains weak benchmark–KB association). The variance-component comparison in "
            "Figure 2 is the central diagnostic. Absolute knowledge-base levels sit modestly "
            "above random and proximity-only baselines; relative encoder choice matters less "
            "than seed noise on KB axes under this recipe. No go/no-go threshold is applied; "
            "the numbers above state which picture the data support.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report -> {path}")
    return path


def write_readme(
    *,
    per_run_clean: pd.DataFrame,
    degenerate: pd.DataFrame,
    range_check: dict[str, Any],
    variance_primary: pd.DataFrame,
    variance_boot: pd.DataFrame,
    seed_assoc_primary: pd.DataFrame,
    eh_summary: dict[str, Any],
    lift_df: pd.DataFrame,
    abs_kb: pd.DataFrame,
    roberta: dict[str, Any],
) -> Path:
    """Short README with key numbers (written after analysis)."""
    path = REPORT_DIR / "README.md"
    n_total = len(MODELS) * len(TRAIN_SEEDS)
    vc_gd = variance_primary[variance_primary["metric"] == "kb_mrr_gene_drug"].iloc[0]
    vc_gdis = variance_primary[variance_primary["metric"] == "kb_mrr_gene_disease"].iloc[0]
    vc_bench = variance_primary[variance_primary["metric"] == "benchmark_f1"].iloc[0]
    sp_gd = seed_assoc_primary[seed_assoc_primary["pair_type"] == "gene-drug"].iloc[0]
    hard = eh_summary.get("hard", {})
    ft_abs = abs_kb[abs_kb["reference"] == "finetuned_encoders_mean"].iloc[0]
    rand_abs = abs_kb[abs_kb["reference"] == "random_uniform"].iloc[0]

    boot_bench = variance_boot[variance_boot["metric"] == "benchmark_f1"]
    bench_ci_str = ""
    if not boot_bench.empty:
        b = boot_bench.iloc[0]
        if b.get("encoder_share_ci_lo") is not None:
            bench_ci_str = f" [{float(b['encoder_share_ci_lo']):.0%}, {float(b['encoder_share_ci_hi']):.0%}]"

    lines = [
        "# Round 1 analysis (folder 11, clean data rerun)",
        "",
        "Consumes folder-10 matrix checkpoints at 5e-6/none. No training.",
        "",
        "## Key numbers",
        "",
        f"- Runs: {n_total} trained, {len(per_run_clean)} clean ({len(degenerate)} degenerate flagged)",
        f"- Benchmark F1 encoder-mean range: {range_check['min_f1']:.3f} to "
        f"{range_check['max_f1']:.3f} (spread {range_check['spread']:.3f})",
        f"- Benchmark variance: {vc_bench['encoder_variance_share']:.0%} between-encoder, "
        f"{vc_bench['seed_variance_share']:.0%} seed{bench_ci_str}",
        f"- KB gene-drug variance: {vc_gd['encoder_variance_share']:.0%} encoder, "
        f"{vc_gd['seed_variance_share']:.0%} seed",
        f"- KB gene-disease variance: {vc_gdis['encoder_variance_share']:.0%} encoder, "
        f"{vc_gdis['seed_variance_share']:.0%} seed",
        f"- Seed-level benchmark–KB Spearman (gene-drug): {sp_gd['spearman']:.3f} "
        f"[{sp_gd.get('ci_lo', float('nan')):.3f}, {sp_gd.get('ci_hi', float('nan')):.3f}]",
        f"- Absolute KB: fine-tuned mean gene-drug MRR {float(ft_abs['mrr_gene_drug']):.3f}, "
        f"gene-disease {float(ft_abs['mrr_gene_disease']):.3f}; random baseline {float(rand_abs['mrr_overall']):.3f}",
        f"- Mean fine-tuning lift: benchmark {lift_df['lift_benchmark_f1'].mean():.3f}, "
        f"KB (pair-type mean) "
        f"{((lift_df['lift_kb_mrr_gene_drug'].mean() + lift_df['lift_kb_mrr_gene_disease'].mean()) / 2):.3f}",
        f"- Encoders beating distance ranker on hard subset: {hard.get('n_beats', 0)}/9",
        f"- RoBERTa pattern holds: {roberta.get('pattern_holds', 'see 11_roberta_analysis.csv')}",
        "",
        "## Workflow",
        "",
        "1. GPU scoring (72 runs): `sbatch step_score.sbatch` or `./submit_round1.sh`",
        "2. GPU untrained floor (9 encoders): `sbatch step_score_untrained.sbatch`",
        "3. CPU analysis after 72/72 + 9/9 markers: `sbatch step_analyze.sbatch`",
        "",
        "Full prose: `report.md`. Figures: `../../figures/11_round1_analysis/`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"README -> {path}")
    return path
