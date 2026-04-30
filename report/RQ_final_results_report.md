# Final RQ-Level Results Report (post Phase B lock)

**Status**: results-ready, paper-drafting phase.
**Generated**: 2026-04-30 after PL_FT_T2 backfill and post-lock analysis.
**Primary analysis inputs**:

- `fine_tuning_experiments/phase_b/analysis/output/phase_b_eval_aggregate_20260430T145905Z.csv`
  - rows: 190
  - SHA-256: `84a7150dd916faed849c75050a284aae2b0bbe74bca391e3fd316f502545c117`
- `fine_tuning_experiments/phase_b/analysis/output/phase_b_analysis_20260430T145905Z.json`
  - SHA-256: `ba9d338dc83b06a92ffc38581e7c3d6cd165bbc8b4f028046a8f0891f381f159`
- `fine_tuning_experiments/phase_b/analysis/output/h6_coupling_slopes_20260430T153029Z_rerun.json`
  - SHA-256: `cd580fc50dfe9f8177ba108f39870cb940e69d90d7ffd2368fc51516076619a9`
- `fine_tuning_experiments/phase_b/analysis/output/rq3_encoder_kb_interaction_20260430T145905Z.json`
  - SHA-256: `d9555d87918a5e6efccafae07db6f442d4dd114666f665cf0d1fd94379500f23`

---

## Executive Verdict

The project no longer needs additional model training.  The experimental
phase is closed:

- Phase A: 120/120 complete.
- Phase B realised FT factorial: 190/190 complete.
- LoRA arm: methodologically closed by B.8/B.9/B.24; H4 is a
  methodological null, not a missing experiment.
- RQ3 gap: closed by a clearly labelled post-lock exploratory interaction
  audit, not by new training.

The paper headline should be revised from the original expected
"benchmark variance asymmetry confirmed" to:

> Benchmark performance is not a stable configuration-level proxy for KB
> surfacing.  Multi-corpus training and T2 staging can improve OOD
> benchmarks, but KB surfacing is more schedule-sensitive than BioRED
> performance, and benchmark near-ties show large KB-ordering
> instability with wide uncertainty.

This is less triumphant than the pre-registered H7 expectation, but it is
more interesting and defensible: it directly cautions against treating
benchmark macro-F1 as a sufficient model-selection criterion for
knowledge-facing biomedical RE systems.

---

## RQ1 — Schema Operationalisation

**Question**: How should cancer relation extraction be operationalised in
the absence of a fully aligned public benchmark?

**Answer**: Use the pair-level S_pair schema.  This is the cleanest and
strongest RQ.

**Evidence**:

- First-principle schema rationale in `paper_methods_draft.md` §3.
- Phase A selection: S_pair wins the primary KB metric; S_pair − S_flat
  ΔKB_hit_A = +0.117 with paired CI excluding 0.
- Active-head audit: five active relation heads, two dead / unsupported
  heads, supporting the argument against over-fine S_mech as the main
  paper schema.
- Leakage repair and MeSH C04 oncology projection are documented in the
  locked design and amendment log.

**Critical assessment**:

- The logic is complete: design rationale → empirical selection →
  active-head interpretability → Phase B fixed schema.
- Remaining work is presentation, not experimentation: Figure F3-style
  per-family KB surfacing helps readers see where S_pair is useful, but
  it is not required to validate the schema choice.

**Paper guidance**:

Frame RQ1 as the methodological foundation.  Do not overclaim that S_pair
is universally best; claim it is best under this cancer-assertion audit
and pre-registered metrics.

---

## RQ2 — Training Configurations for OOD Generalisation

**Question**: Which training configurations best support external
benchmark generalisation?

**Answer**: Corpus diversity is the clearest lever; staged T2 training is
partly useful; larger biomedical encoder size is not reliably beneficial;
LoRA and shared-multitask are transparent gaps / nulls.

**Evidence**:

| Component | Result | Source |
|---|---|---|
| H1 encoder | Not confirmed. PL − PB = −0.0076; PL − BL = −0.0211. | `phase_b_analysis_20260430T145905Z.json` |
| H2 corpus | Confirmed. T1F − T1B on BC5CDR = +0.139, CI [+0.108,+0.166], d = 2.04. | `phase_b_analysis_20260430T145905Z.json` |
| H3 staged T2 | Partial. 3/6 tests confirmed. | `phase_b_analysis_20260430T145905Z.json` |
| H4 FT vs LoRA | Methodological null after B.8/B.9/B.24 collapse audit. | Appendix B.24 |
| H5 architecture | Deferred. | Appendix B.2 / analyzer stub |

**Critical assessment**:

- RQ2 is informative but mixed.  It is not a clean "one model wins"
  story.
- H2 is very strong and should be the anchor: exposure to multi-corpus T1
  training materially improves OOD BC5CDR.
- H3 must be written carefully: T2 staging helps PB strongly and helps BL
  on BioRED, but it is not uniformly positive.
- H1 rejects a tempting but simplistic scale story.  PL does not beat PB
  or BL in the realised FT design.
- H4 should not be written as "FT > LoRA" because the LoRA comparator is
  collapsed.  The meaningful insight is a capacity boundary of canonical
  Q/V-only rank-16 LoRA under small, imbalanced biomedical RE.

**Paper guidance**:

Use "corpus diversity and task staging matter more than encoder scale" as
the RQ2 summary.  Put H4 in a methodological-null subsection; put H5 in a
limitations/deferred-design sentence.

---

## RQ3 — Model Family × Audit Formulation

**Question**: How do model family and KB audit formulation affect KB
surfacing yield?

**Answer**: There is no strong encoder × KB-metric interaction in the
realised FT factorial.  Encoder has a modest main effect, KB metric
choice changes the scale, but schedule is the dominant driver of KB
surfacing.

**Evidence**:

Exploratory model:

```text
KB_value ~ encoder + schedule + kb_metric + encoder:kb_metric
```

Source: `rq3_encoder_kb_interaction_20260430T145905Z.json`.

| Term | Partial SS share | Interpretation |
|---|---:|---|
| schedule_block | 0.4516 | Dominant driver of KB surfacing. |
| kb_metric | 0.0433 | Audit formulation changes scale. |
| encoder | 0.0178 | Modest model-family effect. |
| encoder × kb_metric | 0.0025 | Very weak interaction. |

Encoder means:

| Encoder | KB_hit_A | KB_pmass_B | KB_auc_C |
|---|---:|---:|---:|
| PB | 0.461 | 0.381 | 0.614 |
| BL | 0.590 | 0.455 | 0.752 |
| PL | 0.594 | 0.457 | 0.722 |

**Critical assessment**:

- This RQ was originally the weakest because it had no H-numbered
  confirmatory hypothesis.  The exploratory interaction analysis fixes the
  narrative gap without pretending to be confirmatory.
- The result is actually useful: audit formulation changes absolute scale
  (`A`, `B`, and `C` answer different questions), but it does not radically
  change the encoder ranking.
- Schedule dominates, which links RQ3 back to RQ2 and strengthens the
  overall story: training data structure drives downstream surfacing more
  than model family × metric quirks.

**Paper guidance**:

Label this as exploratory.  Do not cite p-values as confirmatory.  Use
partial-SS shares and the stable encoder ranking to answer the RQ.

---

## RQ4 — Benchmark × KB Coupling

**Question**: How strongly and in what direction do benchmark scores
predict KB surfacing?

**Answer**: Phase A showed positive coupling, but Phase B does not support
the original H7 variance-asymmetry direction.  In the realised Phase B FT
factorial, KB surfacing is more configuration-sensitive than BioRED
ex-NEG, H6 slopes are inconclusive, and ordinal-instability point
estimates are large but imprecise.

**Evidence**:

| Component | Result | Source |
|---|---|---|
| H7 R_B | R_B = 0.214, bootstrap CI [0.0275, 0.9902], verdict = null_no_asymmetry. | `phase_b_analysis_20260430T145905Z.json` |
| BioRED lever share | 0.142 | same |
| KB_hit_A lever share | 0.665 | same |
| H6 β_config | −3.012, CI [−13.001,+4.535], label = inconclusive. | `h6_coupling_slopes_20260430T153029Z_rerun.json` |
| Phase B cell Spearman | ρ = −0.250, CI [−0.784,+0.497] | same |
| Ordinal instability | median ΔKB = 0.160; rank inversion rate = 0.50. | `phase_b_analysis_20260430T145905Z.json` |

**Critical assessment**:

- The original H7 expectation is rejected, not confirmed.
- This is not a project failure.  The stronger insight is that benchmark
  macro-F1 is an unstable proxy for downstream KB surfacing: configurations
  that are close on BioRED can differ substantially on KB outcomes.
- H6 slopes are not precise enough to support a strong abstract claim about
  coupling strength.  The abstract should use the "inconclusive/no stable
  proxy" pre-committed framing.
- Ordinal instability is the most compelling RQ4 result, but the wide CIs
  (median ΔKB CI includes 0; rank-inversion CI is broad) mean it should be
  written as a cautionary signal rather than a precise population estimate.

**Paper guidance**:

Do not write "benchmark variance dominates KB variance."  Write:

> Benchmark gains do not reliably translate into KB surfacing gains.  In
> Phase B, KB surfacing varies more across training schedules than BioRED
> ex-NEG performance, and benchmark near-ties show large but uncertain
> KB-ordering instability.

---

## Figure and Table Inventory

Generated final assets:

| Asset | File | Purpose |
|---|---|---|
| F1 | `report/figures/fig01_phase_a_schema_encoder.png` | RQ1 schema selection. |
| F2 | `report/figures/fig02_phase_b_benchmark.png` | RQ2 benchmark outcomes. |
| F3 | `report/figures/fig03_kb_surfacing_profiles.png` | RQ3 KB profiles and partial SS. |
| F4 | `report/figures/fig04_h7_variance_ordinal.png` | RQ4 H7 + ordinal instability. |
| F5 | `report/figures/fig05_h6_slopes.png` | RQ4 H6 slope family. |
| F6 | `report/figures/fig06_lora_collapse_audit.png` | H4 methodological null audit. |
| T2 | `report/tables/table02_phase_b_cell_results.md` | Phase B cell-level results. |
| T3 | `report/tables/table03_phase_b_hypothesis_summary.md` | H1-H7 + RQ3 summary. |
| T4 | `report/tables/table04_rq_evidence_matrix.md` | RQ evidence matrix. |

Historical / superseded assets retained:

- `report/RQ_status_report.md` is marked SUPERSEDED.
- `report/figures/fig04_phase_b_ft_cells.png` and
  `fig05_lora_collapse_vs_ft.png` are 2026-04-27 interim figures.
  They are retained but should not be cited in the final paper; use F2-F6
  above.

---

## Claim Traceability

| Claim | Authoritative source | Safe paper wording |
|---|---|---|
| S_pair selected for Phase B | Phase A locked result and `fig01` | "S_pair was selected under the pre-registered KB_hit_A rule." |
| Multi-corpus T1 improves OOD BC5CDR | H2 in `phase_b_analysis_20260430T145905Z.json` | "Multi-corpus T1 training substantially improves BC5CDR macro-F1." |
| T2 staging helps but not universally | H3 in same JSON | "Staging is beneficial in PB and some endpoints, but not uniformly across encoders/datasets." |
| PL does not beat PB/BL | H1 in same JSON | "Encoder scale alone is not a reliable improvement lever." |
| LoRA result is methodological null | Appendix B.24 + F6 | "Canonical LoRA collapsed under all probed LR/budget settings; H4 is empirically undefined." |
| H7 original direction not confirmed | H7 + bootstrap in same JSON | "R_B < 1; KB surfacing is more design-sensitive than BioRED in Phase B." |
| H6 abstract framing is inconclusive | corrected H6 rerun JSON | "No stable configuration-level benchmark-KB slope was estimated." |
| RQ3 interaction is weak | RQ3 exploratory JSON | "Audit formulation changes scale, not encoder ranking; schedule dominates." |
| Ordinal instability point estimate is large but imprecise | ordinal section in analysis JSON | "Near-tied benchmark configurations can differ materially in KB surfacing, but CIs are wide." |

---

## Remaining Work

No additional experiments are recommended.  Remaining work is paper
production:

1. Lock-v3 commit/tag and tarball backup.
2. Methods polish to align all numbering with post-B.24 realised factorial.
3. Results drafting using F1-F6 and T2-T4.
4. Discussion with explicit negative-result framing for LoRA and revised RQ4
   headline.
5. Abstract using the inconclusive β_config / H7-null framing.
