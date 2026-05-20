# Phase 3 — manuscript evidence dossier

**Audience:** Freddie (writing reference only).  
**Rules honoured:** no manuscript prose here; no `.tex` edits in this artefact; quantitative values either have explicit provenance below or are flagged **`[PROVENANCE UNKNOWN]`** when this workspace clone lacks the generating table/JSON.

**Repo anchor:** all paths below are relative to `project_1/` unless stated otherwise.

---

## Section 0: Status overview (pre-Phase-1 → Phase 3)

Brief orientation — what reframed, not how to phrase it.

| Dimension | Pre-Phase-1 status | Current status (Phase 3 audit) | Net change |
|-----------|-------------------|--------------------------------|------------|
| **R_B headline (9-cell)** | Point ~0.214; 95% CI **[0.028, 0.990]** (`phase_b_analysis_20260430T145905Z.json` → `h7_R_B_bootstrap`) | Same locked statistic **unchanged** by Phase 2C extensions (`rb_phase_d_extensions.json` pre_registered note) | CI crosses unity → **avoid “confirmed reversal”**; extensions add **supplementary** ratios only |
| **Drop-7 IAA sensitivity** | Not in main locked JSON | R_B **0.217**, CI **[0.028, 1.015]**; rank inversion **0.500**; median \|ΔKB\| **0.166** vs **0.160** (`[PROVENANCE UNKNOWN]` — expected `phase_c_robustness/outputs/`) | Upper CI marginally **> 1**; thesis stability claim **needs traced file** |
| **Schedule mechanism framing** | Methods text: staged uses **4 096** total vs **2 048** flat (`03_methods.tex`), implying schedule contrasts embed **compute + oncology content** | Phase 2C decomposition: **d̂_compute** ≈ **+0.142**, **d̂_content** ≈ **+0.137**, **α̂** ≈ **0.509** (`phase_d_alpha_attribution.json`) | Reframe from **pure “content staging”** to **joint compute + content** (both sizable) |
| **IAA evidence layer** | LLM second annotator only; manuscript “**lower bound** on expert human” (`03_methods.tex`) | **Author** third rater: κ(heur, author) **0.434**; κ(LLM, author) **0.835**; **7/7** disagreement cases with LLM (`author_iaa_kappa_results.json`, inserts) | **Convergent** human + LLM vs heuristic; **retire “LLM as lower bound”** |
| **External baselines** | None in main factorial | **CIViCmine** Case C JSON; **GPT-4o-mini** sweep JSONs; **trivial** baselines JSON | Benchmark-facing layer for **calibration**, not headline TE |
| **Trivial baseline** | Unmentioned publicly | Always-**DGR**: **154/162** = **0.950617** (`trivial_baselines.json`) | **Anchors** all 162-set KB accuracies |
| **Matched-compute control** | Absent | **T1F @ 4096** cell, **n=20** seeds, committed rule in `COMMITMENT.md`; outputs in `phase_d_alpha_attribution.json` | Makes **compute attribution** explicit (wide uncertainty at n=20) |

---

## Section 1: Pre-registered Phase A & Phase B findings (STABLE)

### 1.1 Schema selection (RQ1)

### Evidence ID: [E-001]
- **Claim**: S_pair improves KB argmax vs S_family by +0.117.
- **Number(s)**: Δ = **+0.117284**; 95% CI **[0.045370, 0.194444]**; Cohen's d = **+0.554**; permutation **p = 0.015**.
- **Provenance**: `fine_tuning_experiments/schema_exp/analysis/phase_a_analysis.json` → `decision_666.pair_tests.Spair_vs_Sflat__KB_hit_A` (`diff`, `ci_lo`, `ci_hi`, `cohens_d`, `p_value`).
- **Pre-registered?**: **yes**
- **Target paper section**: §4.1 RQ1; Table `tab:t1` (first row).
- **Suggested framing hint**: headline schema win; pair with BioRED safety (`E-002` dual-metric story in supplement E).
- **Avoid framing as**: ignoring that S_pair also wins on benchmark (`E-002b`).

### Evidence ID: [E-002]
- **Claim**: S_pair vs S_mech on KB argmax Δ = +0.242.
- **Number(s)**: Δ = **+0.242438**; CI **[0.170679, 0.312809]**; d = **+1.289**; **p = 0.0** (file).
- **Provenance**: same file → `decision_666.pair_tests.Spair_vs_Smech__KB_hit_A`.
- **Pre-registered?**: **yes**
- **Target paper section**: §4.1; Table `tab:t1` row 2.
- **Suggested framing hint**: secondary contrast; mechanistic collapse context (`E-004`).
- **Avoid framing as**: independent of BioRED ex-NEG evidence (`E-002b`).

### Evidence ID: [E-002b]
- **Claim**: S_pair vs S_family on BioRED ex-NEG: large positive Δ, tight CI.
- **Number(s)**: Δ = **+0.160814**; CI **[0.137122, 0.183384]**; d = **+2.370**; **p = 0.0** (file).
- **Provenance**: `phase_a_analysis.json` → `decision_666.pair_tests.Spair_vs_Sflat__BIORED_ex_NEG`.
- **Pre-registered?**: **yes**
- **Target paper section**: Supplement E / schema decision rationale.
- **Suggested framing hint**: justifies “not worse on benchmark” gate for schema choice.
- **Avoid framing as**: KB-only optimisation without benchmark guardrail.

### Evidence ID: [E-003]
- **Claim**: S_pair beats S_family on KB in **every** main encoder with positive Δ.
- **Number(s)**: **Δ ∈ {+0.052, +0.127, +0.132, +0.159}** (ordered PB, BL, PL, RB per manuscript).
- **Provenance**: `report/project/sections/04_results.tex` lines 15–16 (cross-check Supplement E encoder-wise table).
- **Pre-registered?**: **yes** (reported structure)
- **Target paper section**: §4.1 narrative.
- **Suggested framing hint**: supporting unanimity claim.
- **Avoid framing as**: verified from this JSON alone — **trace Supplement E** if challenged.

### Evidence ID: [E-004]
- **Claim**: S_mech discarded — five of six DrugProt mechanism heads had **zero** positives on BioRED test.
- **Number(s)**: qualitative mechanism count (no single JSON scalar here).
- **Provenance**: `report/project/sections/04_results.tex` lines 17–18; align with Supplement E schema inventory.
- **Pre-registered?**: **yes** (design narrative)
- **Target paper section**: §4.1 collapse rationale.
- **Suggested framing hint**: short design necessitiy paragraph.
- **Avoid framing as**: redundant given S_pair already dominates KB — keep **one** sentence unless reviewer demands detail.

### 1.2 Multi-corpus training (H2)

### Evidence ID: [E-005]
- **Claim**: Multi-corpus T1F improves BC5CDR drug–disease F1 vs BioRED-only T1B.
- **Number(s)**: mean_diff **+0.139213**; 95% CI **[0.108239, 0.166213]**; Cohen's d **2.038**; Wilcoxon **p ≈ 1.03×10⁻⁴**.
- **Provenance**: `fine_tuning_experiments/phase_b/analysis/output/phase_b_analysis_20260430T145905Z.json` → `H2_corpus`.
- **Pre-registered?**: **yes**
- **Target paper section**: §4.3 “Multi-corpus training”.
- **Suggested framing hint**: **d ≈ 2.04** headline for corpus effect.
- **Avoid framing as**: automatic KB lift synonym — KB is separate subsection.

### Evidence ID: [E-006]
- **Claim**: Same H2 direction at BL-base and PL-large encoders.
- **Number(s)**: Δ **+0.158** (BL), **+0.199** (PL) — manuscript rounding.
- **Provenance**: `report/project/sections/04_results.tex` lines 87–88; verify against Supplement F per-encoder H2 replication table.
- **Pre-registered?**: **yes** (family test structure)
- **Target paper section**: §4.3 H2 paragraph.
- **Suggested framing hint**: one parenthetical “same direction other encoders”.
- **Avoid framing as**: independent confirmatory tests without citing supplement detail.

### 1.3 Staged schedule (H3)

### Evidence ID: [E-007]
- **Claim**: Six encoder×benchmark staging tests → **3** confirmed, **2** null, **1** marginal (q≈0.05 boundary).
- **Number(s)**: `n_confirmed: 3`; per-contrast Δ, CI, **q_t** in array (PB BioRED **+0.0529** q_t **0.00066**; PB BC5CDR **+0.0303** q_t **0.00723**; BL BioRED **+0.0228** q_t **0.0157**; BL BC5CDR **−0.00347** q_t **0.652** null; PL BioRED **+0.0106** q_t **0.561** null; PL BC5CDR **+0.0278** q_t **0.0475** marginal).
- **Provenance**: `phase_b_analysis_20260430T145905Z.json` → `H3_schedule.tests`, `n_confirmed`, `verdict`.
- **Pre-registered?**: **yes**
- **Target paper section**: §4.3 “Oncology-projected staging”; Table `tab:t2`.
- **Suggested framing hint**: “partial / encoder-dependent” — matches locked **partial** verdict.
- **Avoid framing as**: uniform staging benefit across encoders+benchmarks.

### 1.4 Encoder scale (H1)

### Evidence ID: [E-008]
- **Claim**: PL vs PB and PL vs BL on BioRED ex-NEG **do not** support scale-up superiority (null family).
- **Number(s)**: PL−PB **−0.00764**, CI **[−0.03209, +0.01318]**, **q_t = 0.519**; PL−BL **−0.02106**, CI **[−0.04309, −0.00146]**, **q_t = 0.159**.
- **Provenance**: `phase_b_analysis_20260430T145905Z.json` → `H1_encoder.tests` (`PL_vs_PB`, `PL_vs_BL` entries).
- **Pre-registered?**: **yes**
- **Target paper section**: §4.3 “Encoder scale”.
- **Suggested framing hint**: concise null — stage and corpus matter more.
- **Avoid framing as**: PL worse on every benchmark — second contrast is sub-threshold but negative vs BL.

### 1.5 Audit-metric robustness (RQ3)

### Evidence ID: [E-009]
- **Claim**: Encoder × KB-metric interaction explains ≈**0.25%** of corrected total SS (manuscript rounds “**0.3%**”).
- **Number(s)**: partial SS share **0.00249229** (**0.249%**).
- **Provenance**: `fine_tuning_experiments/phase_b/analysis/output/rq3_encoder_kb_interaction_20260430T145905Z.json` → `anova_partial_ss.terms.encoder_x_kb_metric.partial_ss_share_of_corrected_total`.
- **Pre-registered?**: **exploratory post-lock** (file states `exploratory_post_lock_not_confirmatory`)
- **Target paper section**: §4.5 RQ3 / secondary metric stability.
- **Suggested framing hint**: supports “ordering stable across metrics”.
- **Avoid framing as**: confirmatory null of interaction — file guardrail says descriptive.

### 1.6 Variance asymmetry (RQ4 / H7)

### Evidence ID: [E-010]
- **Claim**: Nine-cell configuration ratio **R_B ≈ 0.214** with wide bootstrap CI straddling 1.
- **Number(s)**: **R_B = 0.213603**; bootstrap CI **[0.027529, 0.990154]**; bootstrap median **0.221056**.
- **Provenance**: `phase_b_analysis_20260430T145905Z.json` → `H7_variance_asymmetry.R_B`; `h7_R_B_bootstrap.{ci_lower,ci_upper,bootstrap_median}`.
- **Pre-registered?**: **yes**
- **Target paper section**: §4.4 variance asymmetry (keep **verbatim** locked statistic).
- **Suggested framing hint**: cite **with** schedule-share headline (`E-011`) — ratio is **fragile**, components less so in prose ordering.
- **Avoid framing as**: confirmed benchmark–KB reversal; CI includes 1.

### Evidence ID: [E-011]
- **Claim**: Schedule term drives KB variance (**≈59.6%**) vs benchmark (**≈8.2%**) — ~**7×** gap.
- **Number(s)**: KB schedule share **0.596229**; BioRED ex-NEG schedule share **0.082292** (proportions of explained lever SS).
- **Provenance**: `phase_b_analysis_20260430T145905Z.json` → `H7_variance_asymmetry.decomposition.kb_hit_A_setvalued.schedule` and `.biored_macro_f1_ex_neg.schedule`.
- **Pre-registered?**: **yes**
- **Target paper section**: §4.4 (Phase 3 cover letter + abstract emphasis shift).
- **Suggested framing hint**: **lead** quantitative story for general readers vs raw **R_B** alone.
- **Avoid framing as**: exact “seven-fold” without saying “schedule shares of lever variance”.

### Evidence ID: [E-012]
- **Claim**: Total encoder+schedule+interaction lever shares: **14.2%** benchmark vs **66.5%** KB.
- **Number(s)**: `lever_share_biored_ex_neg` **0.142102**; `lever_share_kb_hit_a` **0.665261**.
- **Provenance**: `phase_b_analysis_20260430T145905Z.json` → `H7_variance_asymmetry.lever_share_biored_ex_neg`, `lever_share_kb_hit_a`.
- **Pre-registered?**: **yes**
- **Target paper section**: §4.4 opening imbalance paragraph.
- **Suggested framing hint**: sets up **R_B** definition context.
- **Avoid framing as**: population variance — scope to realised nine-cell design.

### Evidence ID: [E-013]
- **Claim**: Schema wave showed opposite imbalance: ~**92%** BioRED lever variance vs ~**40%** KB (R_A≈2.28).
- **Number(s)**: BioRED sum of schema+encoder+interaction shares **0.9185** (=0.60352+0.24227+0.07267); KB sum **0.4026** (=0.19129+0.17225+0.03902); ratio **2.28** (rounded; matches paper narrative).
- **Provenance**: `phase_a_analysis.json` → `H7_variance_decomposition.biored_macro_f1_ex_neg` and `.kb_hit_A_setvalued` (`share_schema`, `share_encoder`, `share_interaction` summed).
- **Pre-registered?**: **yes** (Phase A locked)
- **Target paper section**: §4.4 contrast schema-wave vs config-wave asymmetry.
- **Suggested framing hint**: “direction flips by experimental epoch”.
- **Avoid framing as**: directly comparable effect sizes — different factorial.

### 1.7 Rank inversion (RQ4)

### Evidence ID: [E-014]
- **Claim**: Among **18** benchmark-tied pairs (ρ=0.03), median \|ΔKB\| = **0.1596**; inversion rate **0.50**; cluster bootstrap CI **[0.1429, 0.8333]**.
- **Number(s)**: `median_delta_KB` **0.159568**; `rank_inversion_rate` **0.5**; CI endpoints as stated; **n_eligible_pairs 18**.
- **Provenance**: `phase_b_analysis_20260430T145905Z.json` → `rq4_ordinal_instability`.
- **Pre-registered?**: **yes**
- **Target paper section**: §4.4 rank-inversion paragraph; Figure `fig:f4`.
- **Suggested framing hint**: instability is **substantive**; precision limited (wide CI).
- **Avoid framing as**: significant deviation from 0.5 without multiplicity context.

### Evidence ID: [E-015]
- **Claim**: ρ sensitivity: inversion rate stays **0.50** (ρ=0.01, 0.03) then **0.548** (ρ=0.05).
- **Number(s)**: see `rows[].inversion_rate` for ρ∈{0.01,0.03,0.05}.
- **Provenance**: `fine_tuning_experiments/phase_b/analysis/output/rho_sensitivity_20260507T205938Z.json` → `rows`.
- **Pre-registered?**: **yes** (robustness spec supplement H)
- **Target paper section**: §4.4 / Supplement H cross-ref.
- **Suggested framing hint**: supports “not a ρ artefact” for central ρ.
- **Avoid framing as**: monotonic — ρ=0.05 changes tie set (**31** pairs).

### Evidence ID: [E-016]
- **Claim**: Exact binomial (Clopper–Pearson) CI for ρ=0.03 inversion rate **[~0.26, ~0.74]**.
- **Number(s)**: `rate_cp_lo = 0.26019`, `rate_cp_hi = 0.73981`.
- **Provenance**: `rho_sensitivity_20260507T205938Z.json` → `rows` item with `"rho": 0.03`.
- **Pre-registered?**: **yes** (reporting dual interval)
- **Target paper section**: §4.4 parenthetical CI method.
- **Avoid framing as**: superseding cluster bootstrap primary — paper uses **both**.

### 1.8 Coupling slopes (RQ4 H6)

### Evidence ID: [E-017]
- **Claim**: All **five** pooled slope summaries flagged **`label: "inconclusive"`** vs **CI width gate 0.30**.
- **Number(s)**: `ci_width` examples: `beta_within` **1.347**; `beta_schema` **0.726**; `beta_encoder` **5.645**; `beta_config` **13.758**; phase-interaction Wald CS width **≈3.95** (`phase_interaction.ci_hi − ci_lo` on `beta_combined_cell` ≈ **0.822 − (−3.126)**).
- **Provenance**: `fine_tuning_experiments/phase_b/analysis/output/h6_coupling_slopes_20260430T145905Z.json` → `meta.ci_width_threshold` (=0.3); `beta_within|beta_schema|beta_encoder|beta_config.label`; `beta_combined_cell.phase_interaction`.
- **Pre-registered?**: **yes** (H6 gate pre-specified)
- **Target paper section**: §4.4 mechanism slopes / Supplement slope figure.
- **Suggested framing hint**: mechanical reporting — **no** slope mythology.
- **Avoid framing as**: evidence for zero slope — **non-estimable** per gate.

### 1.9 LoRA falsification (H4)

### Evidence ID: [E-018]
- **Claim**: LoRA arm **methodological null** — training collapsed to uniform NEG dev predictions across three attempts; no fair FT vs LoRA contrast.
- **Number(s)**: qualitative verdict (no stable accuracy).
- **Provenance**: `phase_b_analysis_20260430T145905Z.json` → `H4_update_regime` (`verdict`, `reason`); manuscript `03_methods.tex` LoRA paragraph.
- **Pre-registered?**: **yes** (attempted arm)
- **Target paper section**: §4.6 / Supplement I.
- **Suggested framing hint**: short **failure report**; not a performance comparison.
- **Avoid framing as**: LoRA “worse” numerically — comparator **undefined**.

### 1.10 IAA second annotator (pre–author layer)

### Evidence ID: [E-019]
- **Claim**: Cohen's κ (heuristic vs **LLM Opus second annotator**) **≈0.561** with bootstrap **95% CI [0.321, 0.790]** on 30-target audit.
- **Number(s)**: κ point **0.560669**; CI **`[0.3208, 0.7902]`** (main bootstrap seed **20260520**); paper rounds **0.56** / **[0.32, 0.80]**.
- **Provenance**: `knowledge_grounded_evidence_audit/analysis/inter_annotator_audit/author_level_iaa/outputs/author_iaa_kappa_results.json` → `kappa_heuristic_vs_llm_opus` (reanalysis on frozen labels); original audit release cited in **Supplement C** (paper lock).
- **Pre-registered?**: **yes** (original Opus audit); author pass **post-hoc extension**
- **Target paper section**: §4.2 intro (restructure around three raters).
- **Suggested framing hint**: reproducible κ row before author κ.
- **Avoid framing as**: human–human agreement — explicitly **LLM** rater originally.

---

## Section 2: Phase 1 post-hoc robustness (`phase_c_robustness`) — **`[PROVENANCE UNKNOWN]`** batch

**Workspace note (2026-05-19):** this clone has **Zenodo-style `*.jsonl` bundles** under `knowledge_grounded_evidence_audit/analysis/phase_c_robustness/outputs/zenodo_bundle/` but **no** exported `RESULTS.md` / `.csv` summary path. Freddie should attach the actual generator outputs (or re-run the Phase C pipeline) and **replace** UNKNOWN blocks with row/key provenance.

### Evidence ID: [E-020]
- **Claim**: Drop-7 (155-target) sensitivity for **R_B** point **0.217** (vs **0.214** on 162).
- **Number(s)**: **0.217** vs **0.214** (`[PROVENANCE UNKNOWN]`).
- **Provenance**: **`[PROVENANCE UNKNOWN]`** — specify CSV/JSON from `phase_c_robustness/outputs/` after export.
- **Pre-registered?**: **no** (post-hoc robustness)
- **Target paper section**: §4.4 sensitivity / supplement robustness.
- **Suggested framing hint**: stability of **directional** asymmetry story if verified.
- **Avoid framing as**: new confirmatory estimator without file cite.

### Evidence ID: [E-021]
- **Claim**: Drop-7 bootstrap CI for R_B widens upper bound to **1.015** (vs **0.990**).
- **Number(s)**: **[0.028, 1.015]** vs **[0.028, 0.990]** (`[PROVENANCE UNKNOWN]`).
- **Provenance**: **`[PROVENANCE UNKNOWN]`**
- **Pre-registered?**: **no**
- **Target paper section**: same as `E-020`.
- **Suggested framing hint**: reinforces **non-reversal** caution.
- **Avoid framing as**: material improvement — tiny shift.

### Evidence ID: [E-022]
- **Claim**: Rank inversion rate **unchanged** at **0.500** drop-7 vs full.
- **Number(s)**: **0.500** (`[PROVENANCE UNKNOWN]`).
- **Provenance**: **`[PROVENANCE UNKNOWN]`**
- **Pre-registered?**: **no**
- **Target paper section**: §4.4 ordinal stability paragraph.
- **Suggested framing hint**: ties to **Panel 3** narrative if sourced.
- **Avoid framing as**: independent replication — same ρ procedure assumed.

### Evidence ID: [E-023]
- **Claim**: Median \|ΔKB\|**162→155** moves **0.160→0.166** (sixth-decimal rounding context).
- **Number(s)**: **0.160**, **0.166** (`[PROVENANCE UNKNOWN]`).
- **Provenance**: **`[PROVENANCE UNKNOWN]`**
- **Pre-registered?**: **no**
- **Target paper section**: optional supplement sentence.
- **Suggested framing hint**: shows IAA exclusions don't erase near-tie structure.
- **Avoid framing as**: large effect.

### Evidence ID: [E-024]
- **Claim**: Nine paired contrasts identical under drop-7 (target-independent geometry claim).
- **Number(s)**: qualitative (`[PROVENANCE UNKNOWN]`).
- **Provenance**: **`[PROVENANCE UNKNOWN]`**
- **Pre-registered?**: **no**
- **Target paper section**: technical supplement.
- **Suggested framing hint**: de-confounds IAA row effects from factorial geometry.
- **Avoid framing as**: pre-registered family.

### Evidence ID: [E-025]
- **Claim**: Subsampling **Panel 1** — per-cell KB CI width **~0.063** at **n=100**, **~0.021** at **n=150**.
- **Number(s)**: **0.063**, **0.021** (`[PROVENANCE UNKNOWN]`).
- **Provenance**: **`[PROVENANCE UNKNOWN]`** — expected figure/table under `phase_c_robustness/outputs/`.
- **Pre-registered?**: **no**
- **Target paper section**: §6 / limitations sample-size guidance; Discussion precision.
- **Suggested framing hint**: replaces round-number **100–200 targets** heuristic with **empirical width** (once sourced).
- **Avoid framing as**: exact unless CSV anchored.

### Evidence ID: [E-026]
- **Claim**: Panel 2 — median \|ΔKB\| CI width plateaus **≈0.467** by **n=50+**.
- **Number(s)**: **~0.467** (`[PROVENANCE UNKNOWN]`).
- **Provenance**: **`[PROVENANCE UNKNOWN]`**
- **Pre-registered?**: **no**
- **Target paper section**: Discussion / supplement power figure caption.
- **Suggested framing hint**: **bottleneck** is not per-cell KB mean precision alone.
- **Avoid framing as**: Gaussian closed form — empirical sim only.

### Evidence ID: [E-027]
- **Claim**: Panel 3 — rank-inversion CI width flat **~0.68** across displayed **n**.
- **Number(s)**: **~0.68** (`[PROVENANCE UNKNOWN]`).
- **Provenance**: **`[PROVENANCE UNKNOWN]`**
- **Pre-registered?**: **no**
- **Target paper section**: §4.4 caveat — **n_pairs=18** fixed combinatorics.
- **Suggested framing hint**: explains why more targets don't tighten inversion interval much.
- **Avoid framing as**: independent of ρ.

### Evidence ID: [E-028]
- **Claim**: Bias mechanism — T2 vs T1B mean KB hit on **disagreement-7** Δ **+0.762** **[+0.712, +0.812]**; on **random-7** Δ **+0.362** **[+0.291, +0.433]**; differential **+0.400**.
- **Number(s)**: as stated (`[PROVENANCE UNKNOWN]`).
- **Provenance**: **`[PROVENANCE UNKNOWN]`**
- **Pre-registered?**: **no**
- **Target paper section**: Discussion mechanism / limitation on directionality.
- **Suggested framing hint**: aligns oncology projection with **extra positives** where humans disagree with heuristic positive.
- **Avoid framing as**: causal identification — descriptive decomposition.

### Evidence ID: [E-029]
- **Claim**: Pooled calibration — **ECE_max_softmax ≈ 0.277**; **ECE_pmass_B ≈ 0.209**; temperature **T\*≈2.10** without pooled gain.
- **Number(s)**: (`[PROVENANCE UNKNOWN]`).
- **Provenance**: **`[PROVENANCE UNKNOWN]`**
- **Pre-registered?**: **no** (secondary metrics family)
- **Target paper section**: Supplement calibration / RQ3 extensions.
- **Suggested framing hint**: supports **miscalibration** especially **T1B** NEG-heavy cells once traced.
- **Avoid framing as**: primary endpoint.

### Evidence ID: [E-030]
- **Claim**: Per-family stratified **R_B** — gene–drug **n=154** reproduces headline magnitude; variant–disease **n=8** **descriptive only**.
- **Number(s)**: **R_B ≈ 0.213** CI **[0.029, 1.035]**; variant–disease **R_B ≈ 2.11** **[0.232, 14.80]** (`[PROVENANCE UNKNOWN]`).
- **Provenance**: **`[PROVENANCE UNKNOWN]`**
- **Pre-registered?**: **no**
- **Target paper section**: §1 / limitations variant subgroup transparency; supplement stratified table.
- **Suggested framing hint**: **all 7** curated IAA disagreement rows are **gene–drug** (`llm_validity_checks.md` target list).
- **Avoid framing as**: variant subgroup confirms asymmetry — **n=8** unstable.

---

## Section 3: Phase 2A — CIViCmine (Case C)

### Evidence ID: [E-031]
- **Claim**: Strict entity-pair coverage **41 / 162 = 25.31%**; PMID-only **79 / 162 = 48.77%**; **121** uncovered on strict definition inside Case C pipeline.
- **Number(s)**: `strict_count=41`, `pmid_only_count=79`, `n_evaluable_targets=162`; implied non-cover **121**.
- **Provenance**: `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/civicmine_baseline_case_c.json` (`coverage_rates_unfiltered_relaxed`, `n_*`); cross-check `coverage_probe.json` `targets_covered_strict_unfiltered: 41`.
- **Pre-registered?**: **no**
- **Target paper section**: Supplement §S13 / external comparison §4.5.
- **Suggested framing hint**: **Layer A** before any score on **41**.
- **Avoid framing as**: population prevalence of extractor success.

### Evidence ID: [E-032]
- **Claim**: **83** targets lack PMID in strict bookkeeping; **38** PMID-present strict-pair-missing (**31** with ≥1 entity slot, **7** with neither).
- **Number(s)**: `n_missing_pmid: 83`; `pmid_present_no_pair_breakdown` object counts.
- **Provenance**: `civicmine_baseline_case_c.json` top-level fields.
- **Pre-registered?**: **no**
- **Target paper section**: Supplement S13 diagnostics.
- **Suggested framing hint**: explains *why* non-coverage dominates.
- **Avoid framing as**: CIViC curation error — pipeline mapping issue.

### Evidence ID: [E-033]
- **Claim**: On strict **41**, CIViCmine mapped argmax hits **39/41 = 0.9512** (KB Method A).
- **Number(s)**: accuracy field `0.9512195121951219`.
- **Provenance**: `civicmine_baseline_case_c.json` → `civicmine_kb_argmax_accuracy_strict41_mean`.
- **Pre-registered?**: **no**
- **Target paper section**: external comparison table (conditional row).
- **Suggested framing hint**: **selection-biased** upper-bound slice; pair with **`E-031`**.
- **Avoid framing as**: comparable to unconditional **162** PB means.

### Evidence ID: [E-034]
- **Claim**: PubMedBERT **T2** mean KB hit on the **same 41** = **0.85488** (20-seed mean of subset means).
- **Number(s)**: `subset_kb_hit_mean_seed_mean: 0.8548780487804878`; subset SD **0.11029**.
- **Provenance**: `civicmine_baseline_case_c.json` → `pb_pubmedbert_kb_on_civicmine_strict_subset.PB_T2_on_strict41`.
- **Pre-registered?**: **no**
- **Target paper section**: Case C row beside **0.951** extractor.
- **Suggested framing hint**: fair PB comparison **only** on matched **41**.
- **Avoid framing as**: weakness vs **162** headline trainer numbers without denominator note.

### Evidence ID: [E-035]
- **Claim**: On strict **41**, **T1F-2048** mean **0.5488**; **T1F-4096** **0.7817**; **T1B** **0.2622**.
- **Number(s)**: `subset_kb_hit_mean_seed_mean` per arm in JSON block.
- **Provenance**: `civicmine_baseline_case_c.json` → `pb_pubmedbert_kb_on_civicmine_strict_subset` (`PB_T1F_2048_on_strict41`, `PB_T1F_4096_on_strict41`, `PB_T1B_on_strict41`).
- **Pre-registered?**: **no**
- **Target paper section**: supplement table S13.1 style grid.
- **Suggested framing hint**: shows schedule+corpus spread **even on** cherry-picked positives-heavy slice.
- **Avoid framing as**: independent of seed variance — SD fields reported.

### Evidence ID: [E-036]
- **Claim**: Full-162 **imputation bookkeeping** for external reporting: exclude-uncovered **0.24074**; **NEG surrogate on uncovered** **0.24074**; random-IID-8 **0.33410**; `expected_hits_from_random_imputed` **54.125**.
- **Number(s)**: see JSON numeric fields.
- **Provenance**: `civicmine_baseline_case_c.json` → `pb_pubmedbert_kb_on_civicmine_strict_subset.civicmine_162_denominator_heuristics_for_external_system_reporting`.
- **Pre-registered?**: **no**
- **Target paper section**: Supplement case C “denominator hygiene”.
- **Suggested framing hint**: **0.334** is **not** the same estimand as **IID 0.125** (`llm_validity_checks.md` Check 4 guardrail).
- **Avoid framing as**: performance score — **scenario label**.

### Evidence ID: [E-037]
- **Claim**: Evaluable pool family sizes **gene_drug 154** / **variant_disease 8** under strict coverage accounting (`[PROBE]`).
- **Number(s)**: `gene_drug.total: 154`, `variant_disease.total: 8` (strict unfiltered probe).
- **Provenance**: `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/civicmine/coverage_probe.json` → `family_coverage_strict_unfiltered`.
- **Pre-registered?**: **no** (instrumentation)
- **Target paper section**: Background variant subgroup transparency (**n=8**).
- **Suggested framing hint**: pairs with **`E-030`** once traced.
- **Avoid framing as**: equal power subgroup analysis.

---

## Section 4: Phase 2B — LLM baseline & trivial anchors

### Evidence ID: [E-038]
- **Claim**: GPT-4o-mini KB hit mean **162-set** / **41-set** by condition.
- **Number(s)**: **zero-shot** **0.987654 / 1.000000**; **six-shot** **0.919753 / 0.926829**; **six-shot+rationale** **0.925926 / 0.951220**.
- **Provenance**: `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/llm_baseline/gpt4o_mini_zero_shot.json` → `summary.kb_hit_mean_162`, `kb_hit_mean_41`; parallel keys in `gpt4o_mini_six_shot.json`, `gpt4o_mini_six_shot_rationale.json`.
- **Pre-registered?**: **no**
- **Target paper section**: Supplement M; external comparison §4.5.
- **Suggested framing hint**: present **after** trivial anchor (`E-041`).
- **Avoid framing as**: fair head-to-head vs trainers without label-prior caveat (`E-040`).

### Evidence ID: [E-039]
- **Claim**: Label distribution — **DGR** dominates (**~88–94%** of 162 predictions by condition); **NEG** only **1–7%**.
- **Number(s)**: e.g. zero-shot **153/162** DGR (**0.9444**), **2/162** NEG (**0.0123**); six-shot **144/162** DGR (**0.8889**), **10/162** NEG (**0.0617**); rationale **143/162** DGR (**0.8827**), **11/162** NEG (**0.0679**).
- **Provenance**: `…/llm_baseline/llm_validity_checks.md` — Check 2 tables; tallies match JSON record counts when recomputed.
- **Pre-registered?**: **no**
- **Target paper section**: Supplement M validity preamble.
- **Suggested framing hint**: sets up **prior / vocabulary leakage** interpretation.
- **Avoid framing as**: prevalence-matched to human audit.

### Evidence ID: [E-040]
- **Claim**: On **7** IAA disagreement targets, **all** GPT-4o-mini conditions predict **`DRUG_GENE_REGULATION`** (never **`__NEGATIVE__`**); **0/7** alignment with Opus NEG adjudication.
- **Number(s)**: **7** targets enumerated; `hit_A_sv_argmax: 1` for each row in each JSON (`[PROVENANCE UNKNOWN]` row-level unless reader greps JSON).
- **Provenance**: primary enumeration `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/llm_baseline/llm_validity_checks.md` (Check 3); **verify** per-row `pred_label` / `hit_A_sv_argmax` in `outputs/llm_baseline/gpt4o_mini_{zero_shot,six_shot,six_shot_rationale}.json` → `records[]`.
- **Pre-registered?**: **no**
- **Target paper section**: Discussion limitation “label-space leakage / not relation discrimination”.
- **Suggested framing hint**: bridge to **author 7/7** (`E-047`).
- **Avoid framing as**: general failure on all negatives — slice-specific.

### Evidence ID: [E-041]
- **Claim**: Trivial baselines on **162** (singleton **S**): IID uniform → **0.125**; always-**DGR** → **154/162 = 0.950617**.
- **Number(s)**: see JSON.
- **Provenance**: `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/llm_baseline/trivial_baselines.json` (`IID_uniform_mean_P_hit.162_set`, `always_predict_DRUG_GENE_REGULATION.162_set`).
- **Pre-registered?**: **no**
- **Target paper section**: §4.5 external comparison (opening row group).
- **Suggested framing hint**: **mandatory calibration** paragraph — “LLM ~0.99 is **not** far above structural ceiling.”
- **Avoid framing as**: cheating metric — it is **honest** for **singleton** expected sets.

---

## Section 5: Phase 2C — matched-compute attribution

### Evidence ID: [E-042]
- **Claim**: Pre-committed **continuous** α estimator **α̂ = mean(d_comp)/mean(d_gap)** with paired-seed bootstrap **B=5000**, seed **20260518**; verdict table maps CI width / point to text categories.
- **Number(s)**: rule thresholds in markdown table (α̂ bins, CI width ≥0.50 ⇒ **mixed; attribution uncertain**).
- **Provenance**: `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/matched_compute/COMMITMENT.md`; numerical outcome in **`E-043`–`E-045`**.
- **Pre-registered?**: **protocol committed pre-unblind**; inference **post-hoc**
- **Target paper section**: Methods (new paragraph); Results subsection; Discussion mechanism rewrite.
- **Suggested framing hint**: transparent **STEM** self-critique component.
- **Avoid framing as**: replacing locked **nine-cell** headline (`E-010`).

### Evidence ID: [E-043]
- **Claim**: Cell means on **KB_hit_A_setvalued** (20-seed) — **T1F-2048** **0.476543**; **T1F-4096** **0.618827**; **T2** **0.756173**.
- **Number(s)**: as in `point_estimate` object.
- **Provenance**: `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/phase_d_alpha_attribution.json` → `point_estimate.mean_Y_T1F2048`, `mean_Y_T1F4096`, `mean_Y_T2`.
- **Pre-registered?**: **design extension** on locked metric
- **Target paper section**: Results §4.4 insertion / supplement.
- **Suggested framing hint**: show **PB×FT** schedule ladder.
- **Avoid framing as**: cross-encoder — **PB only** here.

### Evidence ID: [E-044]
- **Claim**: Mean gap decomposition — **d̂_compute = +0.142284**; **d̂_gap = +0.279630**; **α̂ = +0.508830**.
- **Number(s)**: `mean_d_comp`, `mean_d_gap`, `alpha_hat`.
- **Provenance**: `phase_d_alpha_attribution.json` → `point_estimate` (`mean_d_comp`, `mean_d_gap`, `alpha_hat`).
- **Pre-registered?**: **estimator committed**; **values** post-lock
- **Target paper section**: same subsection as `E-043`.
- **Suggested framing hint**: “roughly **half–half** split of mean gap” story + CI caveat (`E-045`).
- **Avoid framing as**: causal share of variance — **mean-gap bookkeeping**.

### Evidence ID: [E-045]
- **Claim**: Bootstrap **95% CI** for α̂ **[−0.08669, +0.86780]** (width **0.954**); **`verdict`: `"mixed_attribution_uncertain"`**.
- **Number(s)**: `alpha_ci_95_pct.lower`, `.upper`, `.width`; `verdict`.
- **Provenance**: `phase_d_alpha_attribution.json` → `bootstrap.alpha_ci_95_pct`, `verdict`.
- **Pre-registered?**: mapped per **`E-042`**
- **Target paper section**: Results + Discussion caution.
- **Suggested framing hint**: commit **honest uncertainty** — matches **COMMITMENT** width rule.
- **Avoid framing as**: precise compute fraction — interval spans most of [0,1].

### Evidence ID: [E-046]
- **Claim**: Paired **Δ_compute** (T1F4096−T1F2048) mean **+0.142284**; **bootstrap 95% CI [−0.01699, +0.29908]**; paired **t p≈0.102**.
- **Number(s)**: `delta_compute_mean_*`; `paired_bootstrap_mean_95_ci.{lower,upper}`; `paired_ttest_rel_scipy.pvalue` **0.101984**.
- **Provenance**: `phase_d_alpha_attribution.json` → `compute_only_contrast`.
- **Pre-registered?**: **no** (contrast of opportunity)
- **Target paper section**: Results decomposition table.
- **Suggested framing hint**: **compute** leg not detectable at **n=20** seeds under this bootstrap/test.
- **Avoid framing as**: proof of zero compute effect.

### Evidence ID: [E-047]
- **Claim**: Paired **Δ_content** (T2−T1F4096) mean **+0.137346**; **bootstrap 95% CI [+0.03610, +0.24043]**; paired **t p≈0.0171**.
- **Number(s)**: `delta_content_mean_*`; CI; p-value **0.017085**.
- **Provenance**: `phase_d_alpha_attribution.json` → `content_only_contrast`.
- **Pre-registered?**: **no**
- **Target paper section**: same table — **content** leg **excludes 0**.
- **Suggested framing hint**: statistical detectability for **staging increment** beyond matched flat steps.
- **Avoid framing as**: independent of compute — **sequential** decomposition.

### Evidence ID: [E-048]
- **Claim**: **Within-cell** KB SD (population **p=stdev**) across **20** PB **T1F-4096** seeds ≈ **0.1689** on `kb_hit_A_setvalued`.
- **Number(s)**: pstdev **0.16889090813535507** (recomputed from CSV column).
- **Provenance**: `report/data/phase_b_ft_seedlevel.csv` — filter `encoder==PB`, `schedule==T1F4096`, column `kb_hit_A_setvalued` (20 rows); recompute `statistics.pstdev`.
- **Pre-registered?**: **no**
- **Target paper section**: phase 2C diagnostic / supplement noise note.
- **Suggested framing hint**: quantifies seed noise for new cell.
- **Avoid framing as**: SE of mean — this is **across-seed** dispersion.

### Evidence ID: [E-049]
- **Claim**: Extended designs on **R_B** — **PB-only four-schedule** point **0.762**, CI **[0.0107, 6.379]**; **augmented ten-cell** point **0.237**, CI **[0.0385, 1.078]**.
- **Number(s)**: see JSON.
- **Provenance**: `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/rb_phase_d_extensions.json` → `PB_only_four_schedule_rb.{point_estimate,ci_lower,ci_upper}`, `augmented_ten_cell_encoder_schedule_rb.*`.
- **Pre-registered?**: **no** (extensions)
- **Target paper section**: Supplement / Phase D note; **not** main abstract headline.
- **Suggested framing hint**: shows **design sensitivity** of ratio estimator; reinforces **don't overwrite** `E-010`.
- **Avoid framing as**: replacement headline — CI even wider than nine-cell.

### Evidence ID: [E-050]
- **Claim**: Narrative lock — **“Retain locked nine-cell factorial R_B (=0.21) verbatim”** when juxtaposing extensions.
- **Number(s)**: quoted string in JSON.
- **Provenance**: `rb_phase_d_extensions.json` → `pre_registered_nine_cell_reference_note`.
- **Pre-registered?**: **guardrail note**
- **Target paper section**: internal author checklist / cover letter technical accuracy.
- **Suggested framing hint**: prevents accidental **numeric drift** across Phase 2 edits.
- **Avoid framing as**: new statistical claim.

### Evidence ID: [E-051] *(study scale anchor)*
- **Claim**: Coupling-slope machinery sees **n_runs = 310** (120 Phase A + 190 Phase B) in analysis JSONs.
- **Number(s)**: **310**, **120**, **190**.
- **Provenance**: `h6_coupling_slopes_20260430T145905Z.json` → `meta.{n_runs,n_phase_A,n_phase_B}`; `phase_b_analysis_20260430T145905Z.json` → `coverage` block for Phase B counts.
- **Pre-registered?**: **yes** (locked sample sizes)
- **Target paper section**: Methods (two-wave description); abstract word-budget fact.
- **Suggested framing hint**: “**310** runs” integer for scope — **do not** put **20×9** seed algebra in abstract (per brief).
- **Avoid framing as**: independent experiments — shared seeds across waves partially.

---

## Section 6: Author-level IAA (post-hoc human layer) — **MAJOR**

### Evidence ID: [E-052]
- **Claim**: Pairwise Cohen's κ — heuristic vs LLM Opus **0.560669 [0.3208, 0.7902]**; heuristic vs author **0.433962 [0.2143, 0.6544]**; LLM vs author **0.834862 [0.6281, 1.0000]**.
- **Number(s)**: as stated (bootstrap **B=5000**, seed **20260520** documented in JSON).
- **Provenance**: `author_iaa_kappa_results.json` → `kappa_heuristic_vs_llm_opus`, `kappa_heuristic_vs_author`, `kappa_llm_opus_vs_author`.
- **Pre-registered?**: **no** (author pass)
- **Target paper section**: §4.2 rewrite + Methods blurb.
- **Suggested framing hint**: **LLM ↔ author** “almost perfect” vs **heuristic gap** pattern.
- **Avoid framing as**: κ_LLM “lower bound” on human–human agreement — **empirically false** given **`E-052`**.

### Evidence ID: [E-053]
- **Claim**: Fleiss' κ (**3 raters**, **30** targets) **0.603**.
- **Number(s)**: **0.603**.
- **Provenance**: `author_iaa_kappa_results.json` → `fleiss_kappa_three_raters.point`.
- **Pre-registered?**: **no**
- **Target paper section**: §4.2 secondary summary line / supplement table foot.
- **Suggested framing hint**: single scalar for **multi-rater** reliability.
- **Avoid framing as**: pairwise interpretable beyond ordinal band.

### Evidence ID: [E-054]
- **Claim**: **Disagreement-structure** file lists **7** heuristic–LLM–author aligned cases where author=LLM≠heuristic on the classic **7** IAA IDs (evidence rows in markdown table export).
- **Number(s)**: **7/7** agreements LLM–author on those targets (0 distinct third label in export narrative).
- **Provenance**: `…/author_level_iaa/outputs/author_iaa_disagreement_structure.md` (reconciliation table); machine list `iaa_three_way_labels.csv`; inserts `PAPER_INSERTS_AUTHOR_IAA.md`.
- **Pre-registered?**: **no**
- **Target paper section**: §4.2 + Discussion “directional bias” rewrite.
- **Suggested framing hint**: **independent** human agrees with LLM on exact earlier contested rows.
- **Avoid framing as**: majority vote truth — still **heuristic retained**.

### Evidence ID: [E-055]
- **Claim**: Methods text still states LLM κ is a “**lower bound** on agreement with an expert human curator” — **deprecated framing** post-author pass.
- **Number(s)**: n/a (qualitative misalignment).
- **Provenance**: `report/project/sections/03_methods.tex` lines ~112–114 (`03_methods.tex` IAA paragraph).
- **Pre-registered?**: **superseded by `E-052`**
- **Target paper section**: ** Methods patch in Phase 3 (outside this dossier)**.
- **Suggested framing hint**: replace with **convergence / triangulation** language.
- **Avoid framing as**: still accurate after author κ (**0.434** vs heuristic **lower** than LLM **0.561** on same vocabulary).

### Evidence ID: [E-056]
- **Claim**: KB metric reliability — **ICC(1,1) = 0.67** for KB argmax across **9×20** design; pooled within-cell SD **~0.17** vs **~0.06** BioRED ex-NEG (manuscript rounding).
- **Number(s)**: **0.67**, **0.17**, **0.06**.
- **Provenance**: `report/project/sections/03_methods.tex` lines ~151–158 (`Statistical analysis` KB noise paragraph). **`[PROVENANCE UNKNOWN]`** for independent recompute from CSV in this dossier — verify against archived ICC script output if required.
- **Pre-registered?**: **yes** (reporting plan)
- **Target paper section**: Methods; optionally Results precision anchors (`E-025` once sourced replaces conclusion round-number).
- **Suggested framing hint**: justifies bootstrap / seed pairing narrative.
- **Avoid framing as**: decomposed ICC by cell without supplement cite.

---

### Cross-phase “integration” status line (non prose)

- **STABLE** core: **§1** (`E-001`–`E-019`, `E-051`) + **Phase 2B calibration (`E-041`)** + **Phase 2C point estimates (`E-043`–`E-047`)** + **CIViCmine coverage facts (`E-031`–`E-037`)** — **57** ID blocks total (includes **E-002b**).
- **FRAGILE** statistics: **`E-010`**, **`E-049`**, **`E-045`**, LLM/GPT scores **`E-038`** under validity gate.
- **DEPRECATED framing**: **`E-055`** (LLM lower-bound sentence) & any cover-letter exclusive reliance on **`E-010`** without **`E-011`**.
- **`[PROVENANCE UNKNOWN]` pending export**: **`E-020`–`E-030`**, partial **`E-056`** trace to analysis artefact.

---

*End of consolidated dossier.*  