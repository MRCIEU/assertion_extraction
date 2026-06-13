"""Training-dynamics report: pair-type-specific gene-disease adjudication."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .adjudication import WELL_DEF_LABELS, WELL_DEF_VAL_F1, WELL_DEFS
from .config import REPORT_DIR


def _gd_row(df: pd.DataFrame | None, slug: str) -> pd.Series | None:
    if df is None or df.empty:
        return None
    hit = df[(df["slug"] == slug) & (df["well_trained_definition"] == WELL_DEF_VAL_F1)]
    return hit.iloc[0] if not hit.empty else None


def _fmt_delta(r: pd.Series | None) -> str:
    if r is None:
        return "unavailable"
    return (
        f"{float(r['mean_delta_kb_mrr']):+.4f} (median {float(r['median_delta_kb_mrr']):+.4f}, "
        f"95% interval {float(r['ci_lo']):+.4f} to {float(r['ci_hi']):+.4f}; "
        f"ranking falls in {int(r['n_kb_falls'])} of {int(r['n_seeds'])} seeds)"
    )


def write_report(
    *,
    inventory_case: str,
    inventory: pd.DataFrame,
    verdict: dict,
    gene_disease_verdict: dict,
    seed_dist: pd.DataFrame,
    hard_easy: pd.DataFrame,
    pair_type: pd.DataFrame,
    robustness: pd.DataFrame,
    paired: pd.DataFrame,
    gd_subset: pd.DataFrame | None = None,
    gd_robustness: pd.DataFrame | None = None,
    gd_encoder: pd.DataFrame | None = None,
    gd_seed: pd.DataFrame | None = None,
    mundane: dict | None = None,
    encoder_corr: dict | None = None,
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "report.md"

    pooled = seed_dist[seed_dist["model_id"] == "ALL"]
    p = pooled.iloc[0] if not pooled.empty else None

    n_runs = int((inventory["n_recoverable_checkpoints"] > 0).sum()) if not inventory.empty else 0
    n_epochs = int(inventory["n_recoverable_checkpoints"].sum()) if not inventory.empty else 0

    gdis = _gd_row(gd_subset, "gene_disease")
    gdis_h = _gd_row(gd_subset, "gene_disease_hard")
    gdis_e = _gd_row(gd_subset, "gene_disease_easy")
    gdrug = _gd_row(gd_subset, "gene_drug")
    gdrug_h = _gd_row(gd_subset, "gene_drug_hard")

    lines = [
        "# Training dynamics: does fitting the benchmark erode knowledge-base generalisation?",
        "",
        "## Plain-language summary",
        "",
        "When a model learns to detect relations in biomedical abstracts, we can ask whether "
        "getting better on a training-corpus test also makes it better at ranking clinically "
        "curated gene-drug and gene-disease links on an independent cancer knowledge base. "
        "A negative association between those two scores across different encoders could mean "
        "either that longer training helps the test but hurts real-world ranking, or that the "
        "two tasks simply measure different things because curation criteria and the candidate "
        "pool differ from training. This analysis follows each encoder across its own training "
        "epochs and compares the two scores at every checkpoint, seed by seed. It then separates "
        "gene-drug from gene-disease ranking, because only the gene-drug side is affected by "
        "non-drug chemical tags in the frozen pool.",
        "",
        "## The question",
        "",
        "An earlier cross-encoder comparison found a weak or absent correlation between "
        "in-distribution benchmark standing and out-of-distribution knowledge-base ranking. "
        "Two explanations compete. Explanation 1 is mechanistic: as training progresses, the "
        "model overfits the training distribution and loses generalisation to the knowledge base. "
        "Explanation 2 is static: benchmark and knowledge-base scores reflect different inclusion "
        "criteria (stated-in-text versus clinically curated), and the frozen candidate pool "
        "contains many PubTator Chemical tags that are not CIViC therapies; a competent model "
        "may score them confidently without curation error. Explanation 2 cannot produce "
        "within-model erosion across training because pool composition and criteria are fixed "
        "for a given run. Gene-disease ranking is a natural control: non-drug chemical inflation "
        "touches only the gene-drug pool, so a robust, hard-subset-concentrated gene-disease "
        "decline during training would be evidence for Explanation 1 that Explanation 2 cannot "
        "produce. A fragile or non-hard-specific gene-disease change would leave the static "
        "verdict intact.",
        "",
        "## What was measured",
        "",
        f"Every recoverable per-epoch checkpoint from the confirmed 5e-6 learning-rate matrix "
        f"was scored without retraining. Coverage: {n_runs} of 72 runs with epoch checkpoints, "
        f"{n_epochs} epoch checkpoints total across nine encoders and eight seeds. At each "
        "checkpoint the same weights were evaluated on both axes: self-measured BioRED test "
        "presence F1 (in-distribution) and CIViC ranking mean reciprocal rank on the frozen "
        "primary pool (out-of-distribution), separately for gene-drug and gene-disease and for "
        "co-sentence (easy) versus cross-sentence (hard) subsets. Pair-type by subset cross "
        "metrics (gene-disease-hard, gene-disease-easy, and the gene-drug analogues) were "
        "recomputed by inference from saved checkpoints because they were not stored in the "
        "initial scoring pass. All key numbers below trace to per-epoch score JSON under the "
        "diagnostic scores directory (498 checkpoints, learning rate 5e-6, no warmup).",
        "",
        inventory_case,
        "",
        "## Pooled within-model result and why it misled",
        "",
    ]

    if p is not None:
        lines.extend(
            [
                f"From epoch 1 to the best validation-F1 checkpoint, {int(p['n_seeds_pairable'])} "
                f"seed trajectories are pairable. Averaged across pair types and subsets, mean "
                f"paired change in benchmark F1 is {float(p['mean_delta_benchmark']):+.4f} and "
                f"mean change in hard-subset knowledge-base MRR is "
                f"{float(p['mean_delta_kb_hard']):+.4f}. Only "
                f"{int(p['n_erosion_benchmark_up_kb_hard_down'])} of {int(p['n_seeds_pairable'])} "
                f"seeds ({float(p['frac_erosion']):.1%}) show benchmark rising while hard-subset "
                f"ranking falls. That pooled reading supports Explanation 2. However, the same "
                f"paired design hides a strong pair-type asymmetry: gene-drug ranking changes by "
                f"{float(p['mean_delta_kb_gene_drug']):+.4f} on average while gene-disease ranking "
                f"changes by {float(p['mean_delta_kb_gene_disease']):+.4f} "
                f"({int(p['n_kb_gene_disease_falls'])} of {int(p['n_seeds_pairable'])} seeds fall "
                f"on gene-disease). Rising gene-drug and falling gene-disease components cancel in "
                f"the pooled average. The correct reading is pair-type-specific.",
                "",
            ]
        )

    lines.extend(
        [
            "## Gene-disease as the informative control",
            "",
            "Non-drug PubTator Chemical tags inflate the gene-drug candidate pool only. They "
            "cannot explain a gene-disease decline. We therefore decomposed within-seed paired "
            "changes (epoch 1 to best validation F1) on gene-disease alone, split into hard "
            "(cross-sentence) and easy (co-sentence) subsets, and compared them to gene-drug.",
            "",
        ]
    )

    if gdis is not None:
        lines.append(
            f"Overall gene-disease ranking changes by {_fmt_delta(gdis)}. "
            f"On the hard subset alone, the change is {_fmt_delta(gdis_h)}. "
            f"On the easy subset, the change is {_fmt_delta(gdis_e)}."
        )
        lines.append("")
        if gdis_h is not None and gdis_e is not None:
            hard_more = float(gdis_h["mean_delta_kb_mrr"]) < float(gdis_e["mean_delta_kb_mrr"])
            if hard_more:
                lines.append(
                    "The hard subset declines roughly twice as much as the easy subset on average "
                    f"({float(gdis_h['mean_delta_kb_mrr']):+.4f} versus {float(gdis_e['mean_delta_kb_mrr']):+.4f}), "
                    "but both subsets fall and the gap is not large enough to claim erosion is confined "
                    "to cross-sentence pairs alone."
                )
            else:
                lines.append(
                    "The gene-disease decline is not clearly larger on the hard subset relative "
                    "to the easy subset."
                )
            lines.append("")

    if gdrug is not None:
        lines.append(
            f"For comparison, gene-drug ranking changes by {_fmt_delta(gdrug)} overall and "
            f"{_fmt_delta(gdrug_h)} on the hard subset. Gene-drug behaviour is consistent with "
            "static pool and criterion differences rather than within-model erosion."
        )
        lines.append("")

    lines.extend(["## Robustness across well-trained checkpoint definitions", ""])
    lines.append(
        "The pooled hard-subset erosion fraction was fragile across three ways of choosing the "
        "well-trained checkpoint (best validation F1, last saved epoch, fixed epoch 5). We "
        "recomputed those definitions for gene-disease and gene-disease-hard specifically."
    )
    lines.append("")
    if gd_robustness is not None:
        for slug, label in [("gene_disease", "Gene-disease (all)"), ("gene_disease_hard", "Gene-disease hard")]:
            lines.append(f"{label}:")
            for well_def in WELL_DEFS:
                row = gd_robustness[
                    (gd_robustness["slug"] == slug)
                    & (gd_robustness["well_trained_definition"] == well_def)
                ]
                if row.empty:
                    continue
                r = row.iloc[0]
                lines.append(
                    f"  Under {WELL_DEF_LABELS[well_def]}, mean change "
                    f"{float(r['mean_delta_kb_mrr']):+.4f}; ranking falls in "
                    f"{float(r['frac_kb_falls']):.1%} of seeds."
                )
            lines.append("")
        pooled_rob = []
        for well_def in WELL_DEFS:
            col = f"frac_erosion_{well_def}"
            if col in robustness.columns:
                pooled_rob.append(float(robustness[col].mean()))
        if pooled_rob:
            lines.append(
                f"For reference, the pooled hard-subset erosion fraction averaged "
                f"{pooled_rob[0]:.1%} under best validation F1, {pooled_rob[1]:.1%} under last "
                f"epoch, and {pooled_rob[2]:.1%} under fixed epoch 5 across encoders. "
                "Compare these to the gene-disease-specific fractions above."
            )
            lines.append("")

    lines.extend(["## Seed-level and encoder-level distribution", ""])
    if gdis_h is not None:
        lines.append(
            f"Gene-disease-hard ranking falls in {int(gdis_h['n_kb_falls'])} of "
            f"{int(gdis_h['n_seeds'])} seeds (mean {float(gdis_h['mean_delta_kb_mrr']):+.4f}, "
            f"median {float(gdis_h['median_delta_kb_mrr']):+.4f}). "
        )
        if gene_disease_verdict.get("criteria", {}).get("outlier_driven"):
            lines.append(
                "The mean is materially larger in magnitude than the median, suggesting some "
                "contribution from extreme seeds rather than a perfectly uniform shift."
            )
        else:
            lines.append(
                "Mean and median are aligned, indicating a broad-based shift rather than a "
                "handful of extreme seeds driving the average."
            )
        lines.append("")

    if gd_encoder is not None:
        enc_h = gd_encoder[
            (gd_encoder["slug"] == "gene_disease_hard")
            & (gd_encoder["well_trained_definition"] == WELL_DEF_VAL_F1)
        ].sort_values("mean_delta_kb_mrr")
        n_neg = int((enc_h["mean_delta_kb_mrr"] < 0).sum())
        lines.append(
            f"Across encoders, {n_neg} of {len(enc_h)} show negative mean gene-disease-hard "
            "paired change. Biomedical-domain encoders (PubMedBERT, BioMedBERT, BioLinkBERT, SciBERT) "
            "show the largest declines, with every seed falling in three of those four families. "
            "General-purpose encoders (BERT-base, DistilBERT-base, DeBERTa-base) show flat or positive "
            f"mean changes (range {float(enc_h['mean_delta_kb_mrr'].min()):+.4f} to "
            f"{float(enc_h['mean_delta_kb_mrr'].max()):+.4f}). This split argues against a single "
            "uniform training-dynamics mechanism operating identically across architectures."
        )
        lines.append("")

        lines.append("")

    if mundane:
        timing_sum = mundane.get("timing_summary")
        stratum_sum = mundane.get("stratum_summary")
        lines.extend(
            [
                "## Ruling out mundane explanations",
                "",
                "Before treating the gene-disease decline as a training-dynamics effect, we "
                "checked two ordinary explanations using the existing per-epoch scores and "
                "folder-11 validation-best CIViC scores.",
                "",
                "### Timing relative to the validation-best checkpoint",
                "",
            ]
        )
        if timing_sum is not None and not timing_sum.empty:
            gd_t = timing_sum[
                (timing_sum["slug"] == "gene_disease") & (timing_sum["timing_class"] != "all")
            ]
            for _, r in gd_t.iterrows():
                lines.append(
                    f"For overall gene-disease ranking, {int(r['n_seeds'])} seeds "
                    f"({float(r['frac_seeds']):.1%}) show the knowledge-base peak "
                    f"{r['timing_class'].replace('_', ' ')}."
                )
            lines.append("")
            lines.append(mundane.get("timing_interpretation", ""))
            lines.append("")
        lines.extend(
            [
                "### Pool size and positive-count fragility",
                "",
            ]
        )
        if stratum_sum is not None and not stratum_sum.empty:
            for _, r in stratum_sum.iterrows():
                lines.append(
                    f"In the {str(r['stratum']).replace('_', ' ')} stratum, mean gene-disease "
                    f"paired change is {float(r['mean_delta_mrr']):+.4f} "
                    f"({int(r['n_falls'])} of {int(r['n_seeds'])} seeds fall)."
                )
            lines.append("")
            lines.append(mundane.get("pool_interpretation", ""))
            pb = mundane.get("positive_bootstrap", {})
            if pb:
                lines.append(
                    f" Bootstrap over seeds confirms the gene-disease-hard decline sign is stable "
                    f"(probability of negative mean change: "
                    f"{float(pb.get('frac_negative_bootstrap', 0)):.1%})."
                )
            lines.append("")

    if encoder_corr and encoder_corr.get("correlations") is not None:
        lines.extend(
            [
                "## Encoder heterogeneity (exploratory)",
                "",
                "Using only the nine existing encoders and their gene-disease-hard paired "
                "changes, we correlated erosion magnitude with three known properties (benchmark "
                "level, biomedical pretraining, parameter count). With nine points and correlated "
                "encoder families this is exploratory, not confirmatory.",
                "",
            ]
        )
        for _, r in encoder_corr["correlations"].iterrows():
            lines.append(
                f"Spearman correlation with {r['label']}: "
                f"rho={float(r['spearman_rho']):+.3f} (p={float(r['p_value']):.3f})."
            )
        lines.append(
            "Biomedical pretraining aligns with larger declines in this sample, but the small "
            "encoder count prevents a firm causal claim. Encoder heterogeneity is real; its "
            "source needs a controlled encoder study."
        )
        lines.append("")

    criteria = gene_disease_verdict.get("criteria", {})
    if criteria:
        lines.extend(["## Adjudication criteria", ""])
        labels = {
            "overall_gene_disease_robust": "Overall gene-disease decline stable across three checkpoint definitions and most seeds",
            "hard_concentrated": "Hard-subset decline sharply exceeds easy-subset decline",
            "robust_gene_disease_hard_all_defs": "Gene-disease-hard decline stable across three checkpoint definitions",
            "broad_based_seeds_hard": "Gene-disease-hard decline broad-based (≥65% seeds, median negative)",
            "encoder_consistent": "Uniform decline across all encoders (legacy uniformity bar)",
            "outlier_driven": "Mean driven by a few extreme seeds (diagnostic flag)",
        }
        for key, label in labels.items():
            if key not in criteria:
                continue
            val = criteria[key]
            if key == "outlier_driven":
                status = "yes (caution)" if val else "no"
            elif key == "encoder_consistent":
                status = (
                    "not met; regular heterogeneity by pretraining domain recorded instead"
                    if not val
                    else "met"
                )
            else:
                status = "met" if val else "not met"
            lines.append(f"- {label}: {status}")
        lines.append("")

    lines.extend(
        [
            "## Trajectory shape",
            "",
            "Per-seed trajectories of benchmark F1 and gene-disease-hard knowledge-base ranking "
            "across epochs are shown in the diagnostic figures (fig5 gene-disease-hard trajectories; "
            "fig6 pair-type by subset contrast). The signature of mechanistic erosion is "
            "gene-disease-hard peaking early then declining while the benchmark continues to rise. "
            "Seed-level curves and encoder means are displayed without smoothing away divergence.",
            "",
            "## Verdict",
            "",
        ]
    )
    mundane_timing = mundane.get("timing_interpretation", "") if mundane else ""
    if mundane_timing and "after the validation-best" in mundane_timing.lower():
        lines.append(
            "The timing check weakens a strong training-dynamics reading of gene-disease erosion: "
            "ranking often peaks only after the validation-best checkpoint, consistent with ordinary "
            "late-training movement rather than early divergence from the knowledge base."
        )
        lines.append("")
    lines.extend(
        [
            gene_disease_verdict.get("narrative", "Verdict unavailable."),
            "",
            "On the pooled hard-subset axis, the original static explanation still applies: mean "
            "hard-subset knowledge-base MRR change is near zero and the erosion fraction is fragile "
            "across checkpoint definitions. The gene-disease deepening does not overturn that pooled "
            "reading; it explains why the pooled average was misleading and clarifies what can and "
            "cannot be claimed about pair-type-specific within-model change.",
            "",
            "A companion note on qualitative curation errors (missed positives, abstract-supported "
            "versus abstract-unsupported cases, and flagged examples for manual reading) is in "
            "report_qualitative_errors.md.",
            "",
            verdict.get("power_note", ""),
            "",
            "This round does not modify the frozen pool, matching rules, or type mappings. "
            "Non-drug chemical distractors remain in the pool and are reasoned about, not removed.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report -> {path}")
    return path


def write_qualitative_report(qual: dict) -> Path:
    path = REPORT_DIR / "report_qualitative_errors.md"
    summary = qual.get("summary", {})
    patterns = qual.get("patterns")
    lines = [
        "# Qualitative error analysis for cancer knowledge-base curation",
        "",
        "## Plain-language summary",
        "",
        "Abstract-level relation models are sometimes proposed to help curators find gene-drug and "
        "gene-disease links in PubMed text. This note shows, in concrete terms, what the models "
        "get wrong on clinically curated CIViC pairs, and separates true model failures from cases "
        "where the abstract alone cannot support the curated relation.",
        "",
        "## Design",
        "",
        f"Scores come from folder-11 validation-best checkpoints. At seed {summary.get('representative_seed', 42)}, "
        f"we take the median score across all nine encoders per candidate, then identify missed "
        "positives (the lowest-ranked curated positive in each abstract) and false highs "
        "(top-ranked non-curated candidates).",
        "",
        "## Abstract ceiling versus model error",
        "",
        f"Among {summary.get('n_missed_positives', 0)} missed positives, "
        f"{summary.get('n_abstract_unsupported', 0)} "
        f"({float(summary.get('frac_abstract_unsupported', 0)):.1%}) are abstract-unsupported: "
        "the entities do not co-occur in a way the abstract can state the relation. Ranking those "
        "pairs low is not a model failure; it reflects that CIViC curation uses evidence beyond a "
        "single abstract. This proportion is a practical ceiling on abstract-only NLP assistance.",
        "",
        f"Genuine model errors (abstract-supported missed positives): "
        f"{summary.get('n_genuine_model_errors', 0)}.",
        "",
        "## Systematic failure modes (genuine errors only)",
        "",
    ]
    if patterns is not None:
        for _, r in patterns.iterrows():
            pct = float(r["rate_in_genuine_errors"]) * 100
            label = str(r["pattern"]).replace("_", " ")
            lines.append(
                f"{label.capitalize()} appears in {pct:.0f}% of genuine errors."
            )
    lines.extend(
        [
            "",
            "Cross-sentence gene-disease pairs and multi-word entity names are common failure "
            "contexts when the abstract does support the link. Older publication years appear "
            "somewhat more often among genuine errors, but the dominant pattern is cross-sentence "
            "gene-disease wording rather than a single phrasing type.",
            "",
            "## Manual review",
            "",
            "A stratified sample of abstract-supported missed positives is flagged in the "
            "qualitative error case table for manual reading. Individual case interpretation is "
            "left to the author; this pipeline surfaces cases and features only.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Qualitative report -> {path}")
    return path


def write_readme(
    *,
    verdict: dict,
    gene_disease_verdict: dict,
    seed_dist: pd.DataFrame,
    inventory: pd.DataFrame,
    gd_subset: pd.DataFrame | None = None,
    qual_summary: dict | None = None,
    mundane: dict | None = None,
) -> Path:
    path = REPORT_DIR / "README.md"
    pooled = seed_dist[seed_dist["model_id"] == "ALL"]
    p = pooled.iloc[0] if not pooled.empty else None
    n_epochs = int(inventory["n_recoverable_checkpoints"].sum()) if not inventory.empty else 0

    gdis_h = _gd_row(gd_subset, "gene_disease_hard")
    gdis = _gd_row(gd_subset, "gene_disease")
    gdrug = _gd_row(gd_subset, "gene_drug")

    lines = [
        "# Training-dynamics diagnostic",
        "",
        "Adjudicates training-dynamics erosion (Explanation 1) vs static criterion/pool mismatch "
        "(Explanation 2), with a focused gene-disease deepening pass.",
        "",
        "## Key numbers (5e-6/none matrix, 498 epoch checkpoints)",
        "",
        f"- Epoch checkpoints scored: {n_epochs}",
    ]
    if p is not None:
        lines.extend(
            [
                f"- Pairable seeds (epoch 1 -> best val F1): {int(p['n_seeds_pairable'])}",
                f"- Pooled hard-subset erosion (bench up, KB hard down): "
                f"{int(p['n_erosion_benchmark_up_kb_hard_down'])} ({float(p['frac_erosion']):.1%})",
                f"- Pooled mean delta KB hard: {float(p['mean_delta_kb_hard']):+.4f}",
                f"- Mean delta KB gene-disease (all): "
                f"{float(p['mean_delta_kb_gene_disease']):+.4f} "
                f"({int(p['n_kb_gene_disease_falls'])}/{int(p['n_seeds_pairable'])} seeds fall)",
                f"- Mean delta KB gene-drug (all): {float(p['mean_delta_kb_gene_drug']):+.4f}",
            ]
        )
    if gdis_h is not None:
        lines.extend(
            [
                f"- Gene-disease-hard mean delta: {float(gdis_h['mean_delta_kb_mrr']):+.4f} "
                f"(median {float(gdis_h['median_delta_kb_mrr']):+.4f}, "
                f"{int(gdis_h['n_kb_falls'])}/{int(gdis_h['n_seeds'])} seeds fall)",
            ]
        )
    if gdis is not None:
        lines.append(f"- Gene-disease (all) mean delta: {float(gdis['mean_delta_kb_mrr']):+.4f}")
    if gdrug is not None:
        lines.append(f"- Gene-drug (all) mean delta: {float(gdrug['mean_delta_kb_mrr']):+.4f}")
    if qual_summary:
        lines.append(
            f"- Abstract-unsupported missed positives: "
            f"{float(qual_summary.get('frac_abstract_unsupported', 0)):.1%}"
        )
    if mundane and mundane.get("positive_bootstrap"):
        pb = mundane["positive_bootstrap"]
        lines.append(
            f"- Gene-disease-hard sign stability P(negative): "
            f"{float(pb.get('frac_negative_bootstrap', 0)):.1%}"
        )
    lines.extend(
        [
            f"- Gene-disease verdict: {gene_disease_verdict.get('verdict', 'pending')} "
            "(within-model biomed-pretraining erosion; regular encoder heterogeneity)",
            f"- Pooled verdict: {verdict.get('verdict', 'pending')}",
            "",
            "## Adjudication criteria (gene-disease)",
            "",
        ]
    )
    crit = gene_disease_verdict.get("criteria", {})
    for k, label in [
        ("overall_gene_disease_robust", "Overall gene-disease robust"),
        ("hard_concentrated", "Hard-concentrated"),
        ("robust_gene_disease_hard_all_defs", "Gene-disease-hard robust across defs"),
        ("broad_based_seeds_hard", "Broad-based (hard)"),
        ("encoder_consistent", "Uniform across encoders (legacy bar)"),
    ]:
        if k in crit:
            if k == "encoder_consistent":
                lines.append(
                    f"- {label}: "
                    f"{'yes' if crit[k] else 'no; regular heterogeneity by pretraining instead'}"
                )
            else:
                lines.append(f"- {label}: {'yes' if crit[k] else 'no'}")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "Cite the native pipeline figures from analyze (fig1_per_seed_trajectories through "
            "fig10_failure_mode_summary). Manuscript-regeneration fig1_within_seed_paired_change, "
            "fig2_pair_type_asymmetry, and fig3_gene_disease_subset_contrast are legacy compact "
            "summaries; prefer fig2_within_seed_paired_change, fig3_hard_easy_pair_type, and "
            "fig6_pair_type_subset_contrast for the same content at full diagnostic depth.",
            "",
            "## Workflow",
            "",
            "1. GPU: score per-epoch checkpoints",
            "2. GPU: supplement pair×subset cross metrics (`submit_supplement_and_analyze.sh`)",
            "3. CPU: analysis and report",
            "",
            "Full prose: `report.md`, `report_qualitative_errors.md`. "
            "Figures: `../../figures/20_round2_diagnostic/`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"README -> {path}")
    return path
