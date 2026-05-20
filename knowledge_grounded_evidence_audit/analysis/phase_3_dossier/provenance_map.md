# Provenance map — Evidence ID → source artefact

Paths relative to `project_1/` unless noted. **Reproduction** commands are indicative; run from repo root unless stated.

### E-001
- **Path:** `fine_tuning_experiments/schema_exp/analysis/phase_a_analysis.json`
- **Keys:** `decision_666.pair_tests.Spair_vs_Sflat__KB_hit_A`
- **Reproduction:** regenerate Phase A analysis pipeline that emits `phase_a_analysis.json`.

### E-002
- **Path:** `fine_tuning_experiments/schema_exp/analysis/phase_a_analysis.json`
- **Keys:** `decision_666.pair_tests.Spair_vs_Smech__KB_hit_A`
- **Reproduction:** same as E-001.

### E-002b
- **Path:** `fine_tuning_experiments/schema_exp/analysis/phase_a_analysis.json`
- **Keys:** `decision_666.pair_tests.Spair_vs_Sflat__BIORED_ex_NEG`
- **Reproduction:** same as E-001.

### E-003
- **Path:** `report/project/sections/04_results.tex`
- **Locator:** RQ1 paragraph (encoder-wise Δ list); cross-check Supplement E tables.
- **Reproduction:** LaTeX source only — verify numerically from Supplement E.

### E-004
- **Path:** `report/project/sections/04_results.tex`
- **Locator:** S_mech collapse sentence; Supplement E schema frequency table.
- **Reproduction:** qualitative + supplement cross-ref.

### E-005
- **Path:** `fine_tuning_experiments/phase_b/analysis/output/phase_b_analysis_20260430T145905Z.json`
- **Keys:** `H2_corpus`
- **Reproduction:** `cd fine_tuning_experiments/phase_b/analysis && python analyze_phase_b.py` (see repo’s locked aggregate CSV input).

### E-006
- **Path:** `report/project/sections/04_results.tex`
- **Locator:** Multi-corpus paragraph (BL/PL deltas).
- **Reproduction:** Supplement F per-encoder extract.

### E-007
- **Path:** `fine_tuning_experiments/phase_b/analysis/output/phase_b_analysis_20260430T145905Z.json`
- **Keys:** `H3_schedule.tests`, `H3_schedule.n_confirmed`, `H3_schedule.verdict`
- **Reproduction:** same aggregate as E-005.

### E-008
- **Path:** `fine_tuning_experiments/phase_b/analysis/output/phase_b_analysis_20260430T145905Z.json`
- **Keys:** `H1_encoder.tests` (`PL_vs_PB`, `PL_vs_BL`)
- **Reproduction:** same as E-005.

### E-009
- **Path:** `fine_tuning_experiments/phase_b/analysis/output/rq3_encoder_kb_interaction_20260430T145905Z.json`
- **Keys:** `anova_partial_ss.terms.encoder_x_kb_metric.partial_ss_share_of_corrected_total`
- **Reproduction:** `python fine_tuning_experiments/phase_b/analysis/rq3_encoder_kb_interaction.py` (see file header).

### E-010
- **Path:** `fine_tuning_experiments/phase_b/analysis/output/phase_b_analysis_20260430T145905Z.json`
- **Keys:** `H7_variance_asymmetry.R_B`, `h7_R_B_bootstrap.{ci_lower,ci_upper,bootstrap_median}`
- **Reproduction:** same as E-005.

### E-011
- **Path:** `fine_tuning_experiments/phase_b/analysis/output/phase_b_analysis_20260430T145905Z.json`
- **Keys:** `H7_variance_asymmetry.decomposition.kb_hit_A_setvalued.schedule`, `.biored_macro_f1_ex_neg.schedule`
- **Reproduction:** same as E-005.

### E-012
- **Path:** `fine_tuning_experiments/phase_b/analysis/output/phase_b_analysis_20260430T145905Z.json`
- **Keys:** `H7_variance_asymmetry.lever_share_biored_ex_neg`, `lever_share_kb_hit_a`
- **Reproduction:** same as E-005.

### E-013
- **Path:** `fine_tuning_experiments/schema_exp/analysis/phase_a_analysis.json`
- **Keys:** `H7_variance_decomposition.kb_hit_A_setvalued.{share_schema,share_encoder,share_interaction}`, `H7_variance_decomposition.biored_macro_f1_ex_neg.{share_schema,share_encoder,share_interaction}` (sum each triple for manuscript narrative).
- **Reproduction:** Phase A analysis JSON regeneration.

### E-014
- **Path:** `fine_tuning_experiments/phase_b/analysis/output/phase_b_analysis_20260430T145905Z.json`
- **Keys:** `rq4_ordinal_instability`
- **Reproduction:** same as E-005.

### E-015
- **Path:** `fine_tuning_experiments/phase_b/analysis/output/rho_sensitivity_20260507T205938Z.json`
- **Keys:** `rows[]` (`rho`, `inversion_rate`, …)
- **Reproduction:** `python fine_tuning_experiments/phase_b/analysis/rho_sensitivity.py` with locked CSV.

### E-016
- **Path:** `fine_tuning_experiments/phase_b/analysis/output/rho_sensitivity_20260507T205938Z.json`
- **Keys:** `rows` element with `"rho": 0.03` → `rate_cp_lo`, `rate_cp_hi`
- **Reproduction:** same as E-015.

### E-017
- **Path:** `fine_tuning_experiments/phase_b/analysis/output/h6_coupling_slopes_20260430T145905Z.json`
- **Keys:** `meta.ci_width_threshold`; `beta_within|beta_schema|beta_encoder|beta_config.{ci_width,label}`; `beta_combined_cell.phase_interaction`
- **Reproduction:** `h6_coupling_slopes.py` with Phase A+B CSVs (see JSON meta).

### E-018
- **Path:** `fine_tuning_experiments/phase_b/analysis/output/phase_b_analysis_20260430T145905Z.json`
- **Keys:** `H4_update_regime`
- **Reproduction:** same as E-005.

### E-019
- **Path:** `knowledge_grounded_evidence_audit/analysis/inter_annotator_audit/author_level_iaa/outputs/author_iaa_kappa_results.json`
- **Keys:** `kappa_heuristic_vs_llm_opus`
- **Reproduction:** `author_audit_labels.csv` + bootstrap script used to populate JSON (see audit folder `README` / `compute_kappa.py` if present).

### E-020 — E-030
- **Path:** **`[PROVENANCE UNKNOWN]`** in this workspace — expected under `knowledge_grounded_evidence_audit/analysis/phase_c_robustness/outputs/` (`RESULTS.md`, CSV summaries, figure exports per Phase C spec).
- **Reproduction:** run Phase C robustness bundle pipeline on Isambard / local clone with outputs committed or `git add -f`.

### E-031
- **Path:** `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/civicmine_baseline_case_c.json`
- **Keys:** `coverage_rates_unfiltered_relaxed`, `n_evaluable_targets`, `n_civicmine_strict_covered`
- **Reproduction:** Phase D Case C driver script referenced in `RESULTS_TO_REPORT_PHASE_D.md`.

### E-032
- **Path:** `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/civicmine_baseline_case_c.json`
- **Keys:** `n_missing_pmid`, `pmid_present_no_pair_breakdown`
- **Reproduction:** same as E-031.

### E-033
- **Path:** `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/civicmine_baseline_case_c.json`
- **Keys:** `civicmine_kb_argmax_accuracy_strict41_mean`
- **Reproduction:** same as E-031.

### E-034
- **Path:** `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/civicmine_baseline_case_c.json`
- **Keys:** `pb_pubmedbert_kb_on_civicmine_strict_subset.PB_T2_on_strict41.subset_kb_hit_mean_seed_mean`
- **Reproduction:** same as E-031.

### E-035
- **Path:** `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/civicmine_baseline_case_c.json`
- **Keys:** `pb_pubmedbert_kb_on_civicmine_strict_subset.PB_T1F_2048_on_strict41`, `PB_T1F_4096_on_strict41`, `PB_T1B_on_strict41` → `subset_kb_hit_mean_seed_mean`
- **Reproduction:** same as E-031.

### E-036
- **Path:** `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/civicmine_baseline_case_c.json`
- **Keys:** `pb_pubmedbert_kb_on_civicmine_strict_subset.civicmine_162_denominator_heuristics_for_external_system_reporting`
- **Reproduction:** same as E-031.

### E-037
- **Path:** `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/civicmine/coverage_probe.json`
- **Keys:** `family_coverage_strict_unfiltered`
- **Reproduction:** probe script in `civicmine/` sibling to Case C JSON.

### E-038
- **Path:** `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/llm_baseline/gpt4o_mini_{zero_shot,six_shot,six_shot_rationale}.json`
- **Keys:** each → `summary.kb_hit_mean_162`, `summary.kb_hit_mean_41`
- **Reproduction:** OpenAI sweep scripts under `phase_d_baselines/` (see supplement M).

### E-039
- **Path:** `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/llm_baseline/llm_validity_checks.md`
- **Locator:** Check 2 frequency tables; optional recount from JSON `records[].pred_label`.
- **Reproduction:** markdown is hand-verified summary of JSONs.

### E-040
- **Path:** `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/llm_baseline/llm_validity_checks.md`
- **Locator:** Check 3 tables; JSON row proof as in E-038 `records[]`.
- **Reproduction:** `grep` seven `target_id` values in each `gpt4o_mini_*.json`.

### E-041
- **Path:** `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/llm_baseline/trivial_baselines.json`
- **Keys:** `IID_uniform_mean_P_hit.162_set`, `always_predict_DRUG_GENE_REGULATION.162_set`
- **Reproduction:** analytic script emitting JSON (see Phase D results doc).

### E-042
- **Path:** `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/matched_compute/COMMITMENT.md`
- **Locator:** α definition + verdict table.
- **Reproduction:** human-signed protocol file.

### E-043 — E-047
- **Path:** `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/phase_d_alpha_attribution.json`
- **Keys:** `point_estimate`, `bootstrap.alpha_ci_95_pct`, `compute_only_contrast`, `content_only_contrast`, `verdict`
- **Reproduction:** `phase_d_alpha_attribution.py` (path inside `phase_d_baselines/`, see JSON `runs_root`).

### E-048
- **Path:** `report/data/phase_b_ft_seedlevel.csv`
- **Row filter:** `encoder==PB` & `schedule==T1F4096` (20 rows) → column `kb_hit_A_setvalued` → `statistics.pstdev`.
- **Reproduction:** small Python/R one-liner on CSV.

### E-049
- **Path:** `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/rb_phase_d_extensions.json`
- **Keys:** `PB_only_four_schedule_rb`, `augmented_ten_cell_encoder_schedule_rb`
- **Reproduction:** driver noted in JSON `source_csv` + Phase D extension script.

### E-050
- **Path:** `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/rb_phase_d_extensions.json`
- **Keys:** `pre_registered_nine_cell_reference_note`
- **Reproduction:** same as E-049.

### E-051
- **Path:** `fine_tuning_experiments/phase_b/analysis/output/h6_coupling_slopes_20260430T145905Z.json`
- **Keys:** `meta.n_runs`, `meta.n_phase_A`, `meta.n_phase_B`
- **Reproduction:** regenerate H6 JSON.

### E-052
- **Path:** `knowledge_grounded_evidence_audit/analysis/inter_annotator_audit/author_level_iaa/outputs/author_iaa_kappa_results.json`
- **Keys:** `kappa_heuristic_vs_llm_opus`, `kappa_heuristic_vs_author`, `kappa_llm_opus_vs_author`
- **Reproduction:** audit CSV + κ script.

### E-053
- **Path:** `…/author_iaa_kappa_results.json`
- **Keys:** `fleiss_kappa_three_raters.point`
- **Reproduction:** same as E-052.

### E-054
- **Path:** `knowledge_grounded_evidence_audit/analysis/inter_annotator_audit/author_level_iaa/outputs/author_iaa_disagreement_structure.md`
- **Keys / rows:** reconciliation bullets; machine-readable `iaa_three_way_labels.csv`.
- **Reproduction:** human audit workbook export pipeline.

### E-055
- **Path:** `report/project/sections/03_methods.tex`
- **Locator:** IAA paragraph “lower bound on expert human curator”.
- **Reproduction:** diff vs Phase 3 prose patch (outside dossier task).

### E-056
- **Path:** `report/project/sections/03_methods.tex`
- **Locator:** ICC(1,1) and within-cell SD comparison sentence.
- **Replication to raw data:** `[PROVENANCE UNKNOWN]` — locate ICC script / frozen supplement output if re-deriving from `phase_b_eval_aggregate_*.csv`.

---

**Source short codes** (for `quick_reference_numbers_card.md`): **E** schema (`phase_a_analysis.json`); **F** config (`phase_b_analysis_*.json`); **H** ordinal / ρ (`rho_sensitivity_*.json` / `phase_b_analysis` `rq4`); **H6** slopes JSON; **RQ3** interaction JSON; **L** Case C (`civicmine_baseline_case_c.json`); **probe** `coverage_probe.json`; **M** GPT-4o-mini JSONs; **trivial** `trivial_baselines.json`; **alpha** `phase_d_alpha_attribution.json`; **rbext** `rb_phase_d_extensions.json`; **author_iaa** `author_iaa_kappa_results.json`; **A1** Phase C robustness outputs (**TBD**); **paper** `.tex` secondary citations.
