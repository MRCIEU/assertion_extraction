# Inter-annotator audit (κ on the heuristic gold-lite mapping)

External validation of the heuristic CIViC → S_pair label mapping used in
`analysis/strengthening/goldlite_build.py::_heuristic_gold_s2`. The
manuscript's §3.4 acknowledges this mapping is heuristic, not manual
gold; reviewers will ask "how do you know it's right?", and this audit
answers that with a single-annotator κ on a stratified 30-target sample
(Tom/Yi review item HIGH #13).

## Pipeline

1. `sample_targets.py` — stratified-by-`entity_pair_family` sample of 30
   evaluable goldlite targets (162 evaluable population after dropping
   the 3 schema-unmapped `VARIANT_GENE` rows). Reproducibility:
   `random.Random(42)`. Allocation = (gene_drug 27, variant_disease 3).
   Outputs `sampled_targets.csv` and `population_summary.json`.
2. `render_prompts.py` — emit `prompts.jsonl`, one labelling prompt per
   sampled row. The prompt deliberately excludes `heuristic_expected_label`
   so the second annotator labels blind.
3. (manual) The second annotator labels each prompt and saves the
   results to `audit_labels.csv` with columns
   `target_id, heuristic_label, llm_label, llm_confidence, llm_rationale`.
4. `compute_kappa.py` — Cohen's κ + bootstrap 95% CI (B = 5000,
   seed = 42), and emits `kappa_summary.json` + `disagreements.csv`.

## Realised second-annotator pass (2026-05-07)

The second annotator was a single LLM curator (Claude) acting in place of
a human curator, with `temperature = 0` semantics (deterministic single
pass per target), per the audit specification supplied by Tom/Yi.

Headline numbers:

* `n = 30` audited, 0 parse errors.
* Cohen's κ = **0.56** (95% bootstrap CI [0.32, 0.80]).
* Agreement = 23/30 = 0.77.
* Disagreement pattern is **directional**: in all 7 disagreements the
  blind annotator chose `__NEGATIVE__` while the heuristic chose
  `DRUG_GENE_REGULATION` (5×) or `ASSOCIATION_GENERAL` (2×).
* The disagreements are concentrated on cases where the CIViC-cited
  drug name is *not* explicitly mentioned in the abstract (e.g.,
  Tanespimycin, Lapatinib, Dacomitinib, Teprotumumab); the heuristic
  inherits the drug from the CIViC evidence row even when the abstract
  itself does not name it.

By Landis & Koch this is "moderate" agreement (κ ∈ [0.41, 0.60]); the
lower CI bound dips into "fair". This is *below* the 0.61 "substantial"
threshold proposed by Tom/Yi, so the manuscript should report the κ
honestly and acknowledge the systematic asymmetry rather than claim the
heuristic is externally validated.

## Author-level human labeling (Excel pack)

For **human** curator labels on the same 30-target sample (blind to heuristic), see
`author_level_iaa/README.md` and `author_level_iaa/AUTHOR_IAA_审核指南.md`
(workbook `.xlsx` + TSV + regeneration script).

## Reproducing the audit

```
sbatch fine_tuning_experiments/phase_b/sbatch/phase_b_review_supplements.sbatch
```

(or run the three scripts in order on the login node — total runtime <
10 s, no GPU required).
