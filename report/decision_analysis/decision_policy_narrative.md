# Decision-policy baseline narrative

## Purpose

This document compares **naive single-metric selection rules** to the project’s **explicit weighted rule** (`benchmark_generalization_heavy` in `final_model_selection_rule.json`). All numbers are **derived from** `final_model_selection_summary.csv`; **no new evaluation runs** were executed for this sweep.

## Winners under each policy

| Policy | Chosen model | Basis |
|--------|--------------|--------|
| Weighted default (DP1) | **M015** | Highest `composite_benchmark_generalization_heavy` (0.76668) |
| Internal HR only (DP2) | **M015** | Highest `internal_hr_mean_macro_f1` (0.7683) |
| BioRED only (DP3) | **S001** | Highest `external_biored_macro_f1` (0.2921) |
| BC5CDR only (DP4) | **M015** | Highest `external_bc5cdr_macro_f1` (0.5572) |
| Stability only — min seed std (DP5) | **M021** | Lowest `biored_seed_std` (0.0041) |
| Pairing clinical only (DP6) | **M021** | Highest `pairing_clinical_index` (0.5639) |

## Interpretation (skeptical)

1. **The default weighted rule is not redundant with “pick best internal.”** In this shortlist, DP1 and DP2 both select **M015**, but that agreement is **empirical**, not logical. A future rerun could split internal vs composite winners.

2. **BioRED-only maximization diverges from the default.** **S001** wins DP3. That illustrates **single-benchmark myopia**: strong BioRED, weaker BC5CDR than **M015**, and **weighted-CE branch risk** documented in the selection rule. The project therefore treats **S001/S002** as **conditional**, not default.

3. **BC5CDR-only agrees with the default here.** DP4 picks **M015**, reinforcing that the benchmark-first policy is **not** hiding chemistry transfer.

4. **Stability-only and pairing-only align with M021.** DP5 and DP6 choose **M021**, supporting the report’s **secondary** recommendation when **reproducibility** or **pairing-centric** deployment is explicit. **M021** is not the BioRED or BC5CDR headline leader versus **M015**/**S001**, so deploying it without acknowledging tradeoffs would be a **policy choice**, not a universal win.

5. **No policy is assumption-free.** Even “stability-first” ignores BC5CDR weighting; “BioRED-first” ignores chemistry transfer. The project’s contribution is to make these **assumptions explicit** and to **prefer a multi-criteria default** over naive single scores.

## Relation to executed pairwise baselines

Executed contrasts (**B2–B4**) show **controlled training differences** on the same protocol. Decision-policy baselines (**DP1–DP6**) show **what would be deployed** if stakeholders optimized one column of the summary table. Together they answer: (a) whether **architecture/training controls** matter on benchmarks, and (b) whether **selection policy** matters for which checkpoint ships.
