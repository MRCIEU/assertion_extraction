"""Diagnostic report for Round 2 planning (folder-10 per-epoch data)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import FOCUS_MODEL_IDS, REPORT_DIR
from .three_point_timing import WELL_DEF_LABELS, WELL_DEF_VAL_F1, WELL_DEFS


def write_report(
    *,
    inventory_case: str,
    curve_shape: str,
    timing_notes: str,
    training_summary: pd.DataFrame,
    power_df: pd.DataFrame,
    two_axis: pd.DataFrame,
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "report.md"

    ts = training_summary[training_summary["model_id"].isin(FOCUS_MODEL_IDS)]
    med_peak = float(ts["median_peak_val_f1_epoch"].mean()) if not ts.empty else 0.0

    power_lines = []
    for _, r in power_df.iterrows():
        clears = r.get("effect_clears_detectable_band")
        src = r.get("effect_estimate_source", "unavailable")
        if pd.isna(clears) or src == "unavailable":
            verdict = "not yet assessed (run per-epoch KB scoring first)"
        elif clears:
            verdict = "estimated hard-subset KB shift along training exceeds the rough 10-seed detectable band"
        else:
            verdict = "estimated hard-subset KB shift is smaller than the rough 10-seed detectable band"
        power_lines.append(
            f"{r['short_name']}: hard-subset KB SD at the deployed checkpoint about "
            f"{r['kb_mrr_hard_sd_at_val_f1_ckpt']:.3f}; Round 1 mean within-encoder gene-drug SD "
            f"{r['r1_mean_within_encoder_sd_gene_drug']:.3f} (seed share "
            f"{r['r1_seed_variance_share_gene_drug']:.0%}); estimated training-amount effect "
            f"{r['estimated_training_effect_hard']:.3f} from {src}; {verdict}."
        )

    lines = [
        "# Round 2 diagnostic: training dynamics and power",
        "",
        "This note reads folder-10 step-2 matrix outputs (1e-5 recipe, per-epoch fp16 "
        "checkpoints, fp32 best) and folder-11 Round 1 best-point scores. Nothing was trained. "
        "Optional per-epoch KB and benchmark scoring uses inference on checkpoints already on disk.",
        "",
        "## Checkpoint inventory",
        "",
        inventory_case,
        "",
        "## Training-curve shape (validation metrics, nine encoders)",
        "",
        curve_shape,
        "",
        f"Across the three focus encoders, the median val_f1-best epoch (seed-level) averages "
        f"about {med_peak:.1f}. Validation loss often rises soon after its minimum, so the "
        "interesting region is early training rather than a long flat late plateau.",
        "",
        "## Two-axis timing: benchmark versus KB along training",
        "",
        timing_notes,
        "",
        "Per-epoch fp16 checkpoints under the step-2 matrix allow scoring both BioRED test F1 "
        "and the frozen CIViC pool at each saved epoch for PubMedBERT, RoBERTa, and DistilBERT. "
        "Round 1 already scored the val_f1-best checkpoint; this diagnostic asks whether "
        "benchmark and hard-subset KB peak at different epochs within the same seed.",
        "",
        "## Power check: training lever versus Round 1 seed noise",
        "",
        "Round 1 (folder 11) found most KB variance sits within encoders: about 88% seed noise "
        "on gene-drug and 70% on gene-disease at the pool level. The table below compares, per "
        "focus encoder, the seed-level hard-subset spread at the deployed checkpoint against "
        "the absolute KB shift along the per-epoch trajectory (max minus min hard MRR within "
        "each seed), and a rough detectable band if ten seeds were averaged per cell.",
        "",
        " ".join(power_lines),
        "",
        "## Plain reading for Round 2 planning",
        "",
        "Validation curves support defining training-amount levels around early epochs, not a "
        "wide late over-training plateau on validation. Whether a Round 2 on training "
        "configuration is worth running depends on whether the training-amount effect on "
        "hard-subset KB is large relative to the seed noise Round 1 already measured. If the "
        "effect stays near that noise scale, a large multi-encoder Round 2 on this lever alone "
        "may not repay the compute unless the design widens the training contrast or uses more "
        "seeds per cell. Either outcome is reported descriptively; this diagnostic does not "
        "design or launch Round 2.",
        "",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report -> {path}")
    return path


def _format_definition_block(r: pd.Series, well_def: str, three_pt: pd.DataFrame) -> str:
    mid = r["model_id"]
    n_p = int(three_pt[(three_pt["model_id"] == mid) & (three_pt[f"pairable_{well_def}"])].shape[0])
    return (
        f"Under {WELL_DEF_LABELS[well_def]}, {n_p} pairable seeds: delta benchmark mean "
        f"{r[f'mean_delta_benchmark_{well_def}']:.3f} "
        f"(interval {r[f'delta_benchmark_ci_lo_{well_def}']:.3f} to "
        f"{r[f'delta_benchmark_ci_hi_{well_def}']:.3f}) against Round 1 benchmark seed SD "
        f"{r['r1_benchmark_f1_sd']:.3f}; delta KB hard mean "
        f"{r[f'mean_delta_kb_hard_{well_def}']:.3f} "
        f"(interval {r[f'delta_kb_hard_ci_lo_{well_def}']:.3f} to "
        f"{r[f'delta_kb_hard_ci_hi_{well_def}']:.3f}) against Round 1 hard-subset seed SD "
        f"{r['r1_kb_hard_mrr_sd']:.3f}. Benchmark clears noise band: "
        f"{bool(r[f'benchmark_clears_noise_{well_def}'])}. KB hard clears noise band: "
        f"{bool(r[f'kb_clears_noise_{well_def}'])}. Axes diverge: "
        f"{bool(r[f'axes_diverge_{well_def}'])}."
    )


def _closing_for_overall(overall: str, summary: pd.DataFrame, pairable: int) -> str:
    robust_n = int((summary["robustness_verdict"] == "divergence_robust_to_selection_criterion").sum())
    val_f1_only_n = int(
        summary["robustness_verdict"].isin(
            ("divergence_only_under_val_f1_selection", "divergence_val_f1_only_trajectory_mixed")
        ).sum()
    )

    if pairable == 0:
        return (
            "Pairable three-point contrasts are not yet available because epoch scoring is "
            "incomplete or milestones are missing. Re-run this section after scoring finishes."
        )

    paras: list[str] = []

    if overall == "round2_on_training_amount_may_be_informative_robust":
        paras.append(
            "The benchmark-up and KB-hard-down divergence is robust to how the well-trained "
            "point is chosen. Under val_f1-best, last epoch, and the fixed epoch-5 training "
            "milestone, all three focus encoders show benchmark rising and hard-subset KB "
            "falling from epoch 1, with both axes clearing matched Round 1 seed-noise bands "
            "where pairable seeds allow. Full-trajectory shape readouts show KB hard generally "
            "below epoch 1 across post-epoch-1 checkpoints, not only at the validation-selected "
            "epoch."
        )
        paras.append(
            "As a candidate mechanism, not a proven law, training that improves benchmark "
            "appears to coincide with a modest erosion of cross-sentence KB signal even when "
            "the well-trained point is not chosen by validation F1. That pattern fits the "
            "round's theme that benchmark-driven optimisation and KB downstream performance "
            "are misaligned, and offers one plausible explanation for Round 1's insensitivity."
        )
        paras.append(
            "The caveats stand. Only three focus encoders were scored at every epoch; "
            f"{pairable} of twenty-four seeds pair under val_f1-best, with four excluded as "
            "collapsed under-well pairs. Confidence intervals remain wide. Delta KB hard "
            "magnitudes stay modest. Generalisation to all nine encoders is explicitly "
            "reserved for Round 2 and is not claimed here."
        )
        paras.append(
            "The verdict is that a Round 2 on training configuration remains informative and "
            "worth running, subject to widening the encoder set and increasing seeds per cell "
            "in the follow-up design. The decoupling read strengthens the hypothesis but does "
            "not replace a nine-encoder confirmatory experiment."
        )
    elif overall in (
        "round2_on_training_amount_softened_selection_confound",
        "round2_on_training_amount_may_be_informative",
    ):
        if val_f1_only_n >= 2:
            paras.append(
                "The benchmark-up and KB-hard-down pattern appears under val_f1-best for the "
                "three focus encoders, but it does not fully survive decoupling from the "
                "selection criterion. When the well-trained point is defined by last epoch or "
                "by a fixed training-amount milestone (epoch 5 capped at the last saved epoch, "
                "without using validation F1), the divergence weakens, reverses, or no longer "
                "clears Round 1 seed-noise bands for one or more encoders. Full-trajectory "
                "shape readouts suggest KB hard is not uniformly below epoch 1 across training; "
                "in several seeds the lowest hard-subset KB aligns with the val_f1-best epoch "
                "specifically, or rebounds afterward."
            )
            paras.append(
                "On this reading the diagnostic cannot fully separate a genuine training-amount "
                "dynamic from a selection-criterion artefact. Defining well-trained by "
                "validation F1 predisposes the well-trained checkpoint toward higher benchmark "
                "scores; if benchmark and KB are even modestly misaligned, that milestone can "
                "appear to force opposite-signed deltas without proving that training amount "
                "itself drives both axes."
            )
            paras.append(
                "The caveats stand. Only three focus encoders; pairable counts vary by "
                "milestone definition; intervals wide; delta KB hard magnitudes modest. "
                "Generalisation to all nine encoders is reserved for Round 2."
            )
            paras.append(
                "The Round 2 recommendation is softened. A follow-up on training configuration "
                "may still be worth running if it prespecifies non-benchmark well-trained "
                "milestones and widens encoders, but the diagnostic no longer treats the "
                "val_f1-only divergence as established without the decoupling checks passing."
            )
        else:
            paras.append(
                "On the paired basis, at least one focus encoder shows a training-amount shift "
                "outside Round 1 seed noise with clear separation between the benchmark and KB "
                "hard axes under val_f1-best, but decoupling readouts are mixed across "
                "encoders. A Round 2 experiment may therefore be informative with explicit "
                "milestone prespecification, though this remains a descriptive reading with "
                "wide intervals."
            )
    else:
        paras.append(
            "On the paired basis, training amount from epoch 1 to the well-trained checkpoint "
            "does not produce a detectable, axis-divergent effect on KB hard relative to Round "
            "1 seed noise once selection-criterion decoupling is applied. A full Round 2 on "
            "this lever alone is likely to yield a null or inconclusive result."
        )

    return "\n\n".join(paras)


def append_three_point_section(
    *,
    three_pt: pd.DataFrame,
    summary: pd.DataFrame,
    overall: str,
) -> Path:
    """Append paired three-point section with selection decoupling to existing report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "report.md"
    base = path.read_text(encoding="utf-8") if path.exists() else ""

    if "## Three-point paired timing" in base:
        base = base.split("## Three-point paired timing")[0].rstrip()

    pairable = int(three_pt["pairable_val_f1_best"].sum())
    total = len(three_pt)

    decoupling_blocks = []
    for _, r in summary.iterrows():
        parts = [f"{r['short_name']}:"]
        for well_def in WELL_DEFS:
            parts.append(_format_definition_block(r, well_def, three_pt))
        parts.append(
            f"Pool-level secondary readouts under val_f1-best (well minus epoch 1): gene-drug "
            f"delta {r[f'mean_delta_kb_gene_drug_{WELL_DEF_VAL_F1}']:.3f}, gene-disease delta "
            f"{r[f'mean_delta_kb_gene_disease_{WELL_DEF_VAL_F1}']:.3f}."
        )
        parts.append(
            f"Three-point KB hard path (epoch 1 to val_f1-best to last): under-to-well "
            f"{r['mean_delta_kb_under_to_well']:.3f}, well-to-end "
            f"{r['mean_delta_kb_well_to_end']:.3f}. {r['three_point_kb_reading']}."
        )
        parts.append(
            f"Full-trajectory shape: {r['trajectory_shape_reading']}. "
            f"Seeds broadly below epoch 1 after epoch 1: "
            f"{r['trajectory_frac_seeds_broadly_below_e1']:.0%}; val_f1-best is unique KB "
            f"minimum: {r['trajectory_frac_val_f1_unique_kb_min']:.0%}; rebound after "
            f"val_f1-best: {r['trajectory_frac_kb_rebound_after_val_f1']:.0%}. "
            f"Robustness verdict: {r['robustness_verdict']}."
        )
        decoupling_blocks.append(" ".join(parts))

    closing = _closing_for_overall(overall, summary, pairable)

    section = [
        "",
        "## Three-point paired timing (authoritative for the verdict)",
        "",
        "The averaged per-epoch curves above describe training shape but can mislead on "
        "two-axis divergence: averaging across seeds hides within-seed misalignment, and "
        "unequal epoch counts from early stopping introduce survivor bias at late epochs. "
        "This section uses within-seed paired deltas at fixed milestones: epoch 1 "
        "(under-trained), a well-trained point, and the last trained epoch (end). The primary "
        "contrast is well-trained minus under-trained on the same seed.",
        "",
        "Because the original well-trained definition uses val_f1-best, a benchmark-side "
        "selection criterion, this section adds decoupling analyses. The same paired deltas "
        "are recomputed under three well-trained definitions: val_f1-best (existing, "
        "validation-selected), last saved epoch (end of training, not selected by validation), "
        "and fixed epoch 5 capped at the last saved epoch (a training-amount index that does "
        "not use validation F1). Divergence is treated as robust only if benchmark still rises "
        "and KB hard still falls under the non-validation definitions, and if full-trajectory "
        "shape shows KB hard generally below epoch 1 across post-epoch-1 checkpoints, not "
        "only at val_f1-best. Hard-subset KB remains the primary axis; gene-drug and "
        "gene-disease pool-level deltas are secondary readouts from the same scored "
        "trajectory.",
        "",
        f"Of {total} focus-encoder seeds, {pairable} pair under val_f1-best; pairable counts "
        "for last epoch and fixed epoch 5 may differ when milestones collapse. Collapsed or "
        "missing milestones are not averaged over.",
        "",
        " ".join(decoupling_blocks),
        "",
        "Divergence is concluded only if both hold: (a) at least one axis delta distribution "
        "lies stably outside the matched Round 1 seed-noise band for that axis; and (b) the two "
        "axes differ clearly in direction or magnitude. Robustness to selection criterion "
        "requires the same sign pattern under last epoch and fixed epoch 5, plus supportive "
        "full-trajectory shape. If divergence appears only under val_f1-best, the diagnostic "
        "cannot separate a genuine training-amount dynamic from a selection artefact.",
        "",
        "When the averaged curve and this paired view disagree, the paired view is authoritative "
        "for whether Round 2 on training amount is worth running.",
        "",
        closing,
        "",
    ]

    path.write_text(base + "\n".join(section) + "\n", encoding="utf-8")
    print(f"Report updated with three-point section -> {path}")
    return path
