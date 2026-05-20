# Phase 2B — LLM baseline validity checks

**Status:** Verification artefact before any LLM headline numbers enter the unified table or Phase 2D prose.

**Inputs:** `fine_tuning_experiments/schema_exp/eval/inputs/kb_surface_pairs.jsonl` (VARIANT_GENE excluded, **n=162**); `schema_expected_label_set` (**S_pair**, **primary**, **set_valued**); GPT-4o-mini JSON under `outputs/llm_baseline/`.

---

## CHECK 1 — Expected-set width (`expected_set_sv`) distribution

| Summary | Value |
|---------|------:|
| Mean $\|S\|$ across 162 targets | **1.0000** |
| Fraction with $\|S\| = 1$ | **1.0000** (162/162) |
| Fraction with $\|S\| \geq 2$ | **0.0000** (0/162) |

**Histogram (count of targets by $|S|$):**

| $\|S\|$ | Count |
|--------|------:|
| 1 | 162 |

*Interpretation:* On this Goldlite-derived **KB audit slice**, **every** target maps to a **singleton** $S$ under (`S_pair`, primary, set-valued). The “multi-label $S$ inflates Method~A” shortcut **does not apply** here ($|S|\geq 2$ never occurs). High LLM scores must be explained by **label priors / always-positive-head strategies** (see Check~4), not by wide expected sets.

---

## CHECK 2 — LLM prediction distribution by condition

### zero_shot (`gpt4o_mini_zero_shot.json`)

- **n targets:** 162
- **Fraction NEG (`__NEGATIVE__`):** 2/162 = **0.0123**
- **Fraction pred equals heuristic `expected_label` (one-to-one):** 90/162 = **0.5556**

| pred_label | Count | Fraction |
|------------|------:|---------:|
| `DRUG_GENE_REGULATION` | 153 | 0.9444 |
| `VARIANT_DISEASE` | 7 | 0.0432 |
| `__NEGATIVE__` | 2 | 0.0123 |

### six_shot (`gpt4o_mini_six_shot.json`)

- **n targets:** 162
- **Fraction NEG (`__NEGATIVE__`):** 10/162 = **0.0617**
- **Fraction pred equals heuristic `expected_label` (one-to-one):** 92/162 = **0.5679**

| pred_label | Count | Fraction |
|------------|------:|---------:|
| `ASSOCIATION_GENERAL` | 3 | 0.0185 |
| `DRUG_GENE_REGULATION` | 144 | 0.8889 |
| `VARIANT_DISEASE` | 5 | 0.0309 |
| `__NEGATIVE__` | 10 | 0.0617 |

### six_shot_rationale (`gpt4o_mini_six_shot_rationale.json`)

- **n targets:** 162
- **Fraction NEG (`__NEGATIVE__`):** 11/162 = **0.0679**
- **Fraction pred equals heuristic `expected_label` (one-to-one):** 90/162 = **0.5556**

| pred_label | Count | Fraction |
|------------|------:|---------:|
| `ASSOCIATION_GENERAL` | 1 | 0.0062 |
| `DRUG_GENE_REGULATION` | 143 | 0.8827 |
| `VARIANT_DISEASE` | 7 | 0.0432 |
| `__NEGATIVE__` | 11 | 0.0679 |

*Interpretation:* Mass is concentrated on **`DRUG_GENE_REGULATION`** (and a thin tail of **`VARIANT_DISEASE`** on variant–disease rows), with **very few NEG** draws—consistent with **entity-type / label-vocabulary heuristics** rather than abstention-heavy human adjudication.

---

## CHECK 3 — Seven IAA disagreement targets (hard cases)

**Audit note:** Per Phase~2B brief: second annotator (Claude Opus) chose **`__NEGATIVE__`** (drug/gene not named in abstract). Goldlite still yields **positive** `expected_set_sv` under schema projection (singleton DGR for the gene–drug rows below). Repository does not vend machine-readable Opus labels—**“Agree IAA”** means `pred_label == __NEGATIVE__`.

| target_id | pairing_family | `expected_label` | $|S|$ | $S$ (sorted) |
|-----------|----------------|------------------|-----|--------------|
| GL_0031 | gene_drug | ASSOCIATION_GENERAL | 1 | `DRUG_GENE_REGULATION` |
| GL_0039 | gene_drug | ASSOCIATION_GENERAL | 1 | `DRUG_GENE_REGULATION` |
| GL_0043 | gene_drug | ASSOCIATION_GENERAL | 1 | `DRUG_GENE_REGULATION` |
| GL_0068 | gene_drug | ASSOCIATION_GENERAL | 1 | `DRUG_GENE_REGULATION` |
| GL_0070 | gene_drug | ASSOCIATION_GENERAL | 1 | `DRUG_GENE_REGULATION` |
| GL_0118 | gene_drug | ASSOCIATION_GENERAL | 1 | `DRUG_GENE_REGULATION` |
| GL_0131 | gene_drug | ASSOCIATION_GENERAL | 1 | `DRUG_GENE_REGULATION` |

### Predictions — zero_shot (`hit_A_sv_argmax` from run log)

| target_id | `pred_label` | hit | Agree IAA (NEG)? | Agree schema $S$ (hit)? |
|-----------|--------------|-----|------------------|-------------------------|
| GL_0031 | `DRUG_GENE_REGULATION` | 1 | False | True |
| GL_0039 | `DRUG_GENE_REGULATION` | 1 | False | True |
| GL_0043 | `DRUG_GENE_REGULATION` | 1 | False | True |
| GL_0068 | `DRUG_GENE_REGULATION` | 1 | False | True |
| GL_0070 | `DRUG_GENE_REGULATION` | 1 | False | True |
| GL_0118 | `DRUG_GENE_REGULATION` | 1 | False | True |
| GL_0131 | `DRUG_GENE_REGULATION` | 1 | False | True |

### Predictions — six_shot (`hit_A_sv_argmax` from run log)

| target_id | `pred_label` | hit | Agree IAA (NEG)? | Agree schema $S$ (hit)? |
|-----------|--------------|-----|------------------|-------------------------|
| GL_0031 | `DRUG_GENE_REGULATION` | 1 | False | True |
| GL_0039 | `DRUG_GENE_REGULATION` | 1 | False | True |
| GL_0043 | `DRUG_GENE_REGULATION` | 1 | False | True |
| GL_0068 | `DRUG_GENE_REGULATION` | 1 | False | True |
| GL_0070 | `DRUG_GENE_REGULATION` | 1 | False | True |
| GL_0118 | `DRUG_GENE_REGULATION` | 1 | False | True |
| GL_0131 | `DRUG_GENE_REGULATION` | 1 | False | True |

### Predictions — six_shot_rationale (`hit_A_sv_argmax` from run log)

| target_id | `pred_label` | hit | Agree IAA (NEG)? | Agree schema $S$ (hit)? |
|-----------|--------------|-----|------------------|-------------------------|
| GL_0031 | `DRUG_GENE_REGULATION` | 1 | False | True |
| GL_0039 | `DRUG_GENE_REGULATION` | 1 | False | True |
| GL_0043 | `DRUG_GENE_REGULATION` | 1 | False | True |
| GL_0068 | `DRUG_GENE_REGULATION` | 1 | False | True |
| GL_0070 | `DRUG_GENE_REGULATION` | 1 | False | True |
| GL_0118 | `DRUG_GENE_REGULATION` | 1 | False | True |
| GL_0131 | `DRUG_GENE_REGULATION` | 1 | False | True |

*Interpretation:* On these seven IDs the model **never** mirrors Opus-style **NEG**; where `hit=1`, it aligns with the **schema projector** (singleton DGR), **not** the second annotator’s abstention narrative.

---

## CHECK 4 — Random baselines (162-set)

**IID uniform over eight S_pair labels (full 162 targets):** for each row, $\mathbb{P}(\mathrm{hit}) = |S|/8$. Here $|S|=1$ everywhere ⇒ expected accuracy **1/8 = 0.125** (numerical mean **0.125000**).

**Relation to paper Layer~A “0.334 random”:** the CIViCmine $n=162$ **random** line in Case~C is `(correct_on_strict41 + analytic_random_mass_on_uncovered_only) / 162`, **not** the same estimand as “uniform guesser evaluated on all 162 KB rows.” Do **not** equate **0.334** with the **0.125** floor here without that caveat.

| Baseline | Accuracy | Notes |
|----------|---------:|-------|
| IID uniform (per-row $\|S\|/8$) | **0.125000** | = **0.125** with all-singleton $S$. |
| Always `DRUG_GENE_REGULATION` | **0.950617** (154/162) | “Smart” constant if $S$ usually contains DGR. |
| Always modal primary `expected_label` = `DRUG_GENE_REGULATION` | **0.950617** (154/162) | Modal primary prevalence **90/162**. |

**Vs zero-shot ~0.988:** the **0.125** floor is trivially passed; the informative ceiling is the **~0.951** **always-DGR** (or equivalent modal-in-set) strategy—only **~0.04** below the observed LLM mean—so the headline accuracy is **not** “oracle-like” relative to naive structure-aware baselines, even though it **dominates** fine-tuned encoders trained under seed noise + abstention.

---

## Gate for RESULTS_TO_REPORT_PHASE_D.md

**Hold:** Do **not** paste GPT-4o-mini headline KB numbers into the Phase~2D unified table until Freddie reviews this file (especially Check~3 vs IAA and Check~4 estimand separation).
