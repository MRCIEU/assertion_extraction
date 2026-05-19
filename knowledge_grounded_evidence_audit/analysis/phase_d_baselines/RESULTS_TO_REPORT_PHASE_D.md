# Phase D — results to cite (AUTO summary)

Frozen artefacts:

- Coverage + PB subsets: `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/civicmine_baseline_case_c.json`
- $R_{\mathrm{B}}$ extensions: `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/rb_phase_d_extensions.json`

## Layer A — CIViCmine coverage dominance (population $n_{\mathrm{eval}} = 162$)

| Quantity | Value |
|-----------|-------|
| Strict entity-pair coverage | **41 / 162 (25.31%)** |
| PMID-only coverage | **79 / 162 (48.77%)** |
| PMID missing entirely | **83** |
| PMID present, strict tuple missing | **38** |

**Full-162 sensitivity imputations (CIViCmine bookkeeping only)**

| Scenario | Accuracy on all 162 evaluable targets |
|-----------|----------------------------------------|
| **Exclude uncovered** | **0.241** |
| **NEG surrogate** | **0.241** (ties exclude on this curator mix) |
| **Random IID** label analytic expectation ($\|\text{gold set}\|/8$) | **0.334** |

## Layer B — Matched 41-target denominator (means over twenty seeds unless noted)

| System | Mean KB argmax accuracy (41 targets) |
|--------|--------------------------------------|
| CIViCmine strict mapping | **0.951** (39/41) |
| PubMedBERT × FT × **T2** | **0.855** |
| PubMedBERT × FT × **T1F-2048** | **0.549** |
| PubMedBERT × FT × **T1B** | **0.262** |
| PubMedBERT × FT × **T1F-4096** | *(pending Phase 2C)* |

Mandatory caveat for narrative: cite Supplement §13 Layer B boxed caption; do **not** contrast CIViCmine subset accuracy against trainer numbers on all 162 targets.

### Layer C — Thirty-eight PMID-but-no-pair cases

| Category | Count |
|----------|------:|
| At least one gold slot surfaced by CIViCmine elsewhere on PMID | **31** |
| Neither slot surfaced | **7** |

## Unified comparison table (cross-system)

Populate **every** row with:

- **KB acc (162)** — trainer systems with full abstracts; dash for CIViCmine deterministic mapping.
- **KB acc (41)** — subset restricted to **`civicmine_baseline_case_c.json › covered_targets › target_id`**.

Annotate external rows with **`n evaluable by system = 41/162`** for CIViCmine.

## $R_{\mathrm{B}}$ diagnostics (`rb_phase_d_extensions.json`)

| Block | Interpretation |
|-------|----------------|
| **`PB_only_three_schedule_rb_interim_pre_t1f4096`** | Balanced PubMedBERT slice (three schedules today). Point **0.67**, 95% CI **[0.15, 1.41]** (`seed = 20260519`). Rename to **`PB_only_four_schedule_rb`** once T1F-4096 evaluables exist. |
| **`augmented_grid_deferred_matching_pre_registration_nine_cell`** | Nine-cell factorial reference recomputed (**point ~0.214**, consistent with headline **≈ 0.21**). Swap in ten-cell augmentation post-Phase 2C; **never overwrite** headline nine-cell figure from pre-registration text. |

## Refresh commands after new GPU outputs

```bash
PYTHONPATH=$PWD python3.11 knowledge_grounded_evidence_audit/analysis/phase_d_baselines/civicmine/run_civicmine_baseline.py
PYTHONPATH=$PWD python3.11 knowledge_grounded_evidence_audit/analysis/phase_d_baselines/analysis/phase_d_rb_extensions.py
```
