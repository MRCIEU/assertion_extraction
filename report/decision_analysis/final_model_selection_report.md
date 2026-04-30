# Final model selection report

*Machine-readable scores and CSV companions: `final_model_selection_rule.json`, `final_model_selection_summary.csv`, `decision_weight_profiles.csv`, `model_role_assignment.csv`. Regenerate with `python3.11 report/decision_analysis/build_decision_analysis.py` from `project_1` root.*

---

## 1. Objective

Produce an **explicit, reproducible** policy for choosing among HR shortlist checkpoints after **BioRED** and **BC5CDR** external evaluation, using **internal rerun** metrics, **stability**, and **pairing-centric** BioRED test slices—without inventing a fake universal scalar winner.

---

## 2. How to read recommendation categories

These labels are **normative** for this project. They are **not** synonyms.

| Category | Meaning | Misread to avoid |
|----------|---------|------------------|
| **Default recommendation** | The single checkpoint selected by the **primary weight profile** (`benchmark_generalization_heavy`) after penalties. This is the **sanctioned** benchmark-first deployment choice **under that profile**. | Do **not** read as “best on every metric” or “best for every product.” |
| **Secondary recommendation** | The **next** checkpoint under the **same** primary profile when a second line is needed (e.g. documentation, A/B, or architecture diversity). It is **not** a weaker default for the same objective—it answers a **related but distinct** emphasis (e.g. pairing strength). | Do **not** treat as “almost default in all cases.” |
| **Conditional recommendation** | A checkpoint that **wins a named objective** (e.g. max BioRED, max PubMedBERT pipeline line) but **only** when that objective is **explicitly** prioritized **and** known caveats (branch risk, BC5CDR tradeoff) are accepted. | Do **not** treat as informal curiosity; it is **actionable only after** a product/policy decision. |
| **Diagnostic-only model** | Controls, failed controls, or high-penalty lines used to **interpret** training behavior — **not** promoted for production under current evidence. | Do **not** deploy without a separate risk assessment. |

---

## 3. Inputs used

| Source | Content |
|--------|---------|
| `external_evaluation/reports/tables/primary_external_results.csv` | Official **BioRED** and **BC5CDR** test macro-F1 (5 seeds); DrugProt **blocked** |
| `external_evaluation/reports/tables/reliability_stability_table.csv` | BioRED test seed std |
| `external_evaluation/reports/tables/oncology_subset_results.csv` | **pairing_subset** stratification on BioRED test |
| `fine_tuning_experiments/reports/tables/rerun_main_aggregated_results.csv` | Internal HR mean macro-F1 (dev protocol) |
| `manifests/drugprot_unresolved_status.json` | Confirms **no** external DrugProt test metrics |

---

## 4. Selection criteria

1. **Internal HR** — relative strength on the staged trainer (min–max normalized across shortlist).
2. **External BioRED** — document-level relation transfer under S2 mapping.
3. **External BC5CDR** — chemical–disease transfer (often diverges from BioRED ranking).
4. **Stability** — lower BioRED test seed std preferred (transformed to a stability proxy score).
5. **Pairing-centric oncology relevance** — composite **clinical pairing index** over four pairing families (variant–disease weighted equally with drug–gene in the index; see JSON formula).
6. **Control / branch risk penalties** — weighted-CE lines (S001, S002, M026) incur a penalty in the composite; **M005** (T3 aux) is heavily penalized and **not** eligible as default.

---

## 5. Weight profiles

| Profile ID | Intent |
|------------|--------|
| `benchmark_generalization_heavy` | **Primary policy**: emphasize official **BioRED + BC5CDR** (65% combined) — used for default/secondary. |
| `balanced_scientific_default` | Higher pairing weight; **exploratory** tie-break when variant/mechanism slices dominate product goals. |
| `stability_heavy` | Stress reproducibility (seed std). |
| `pairing_clinical_anchored` | Stress variant/mechanism pairing slices. |

Full weights: `decision_weight_profiles.csv`.

---

## 6. Policy dependence (read carefully)

- **Under benchmark-first / `benchmark_generalization_heavy` weighting, the default is M015.** That choice encodes **joint** BioRED and BC5CDR emphasis — it is **not** a claim that M015 wins every column of the summary table.
- **M021 rises** when **pairing-clinical** or **stability** priorities dominate (see **`pairing_clinical_anchored`** and **`stability_heavy`** profiles, and naive policy picks **DP5/DP6** in `decision_policy_baseline_comparison.csv`).
- **S001/S002** are **externally strong on BioRED** but carry **weighted-CE branch risk**; they are **not** demoted out of ignorance — they are **conditionally** promoted when BioRED-first deployment is explicit (**DP3** would pick **S001** as a naive single-metric rule).
- **No recommendation is free of policy assumptions.** If deployment priorities change, rerun the rule or shift profiles — do **not** relabel an old default as a universal winner.

---

## 7. Primary decision outcome

- **Default recommended model (benchmark policy):** **M015** (BioLinkBERT, pipeline, full fine-tune, T1_to_T2, CE). Maximizes **`benchmark_generalization_heavy`** composite — strongest **BC5CDR** macro-F1 in the shortlist and competitive **BioRED**, aligned with internal HR anchor.
- **Secondary recommended:** **M021** — second under the same profile; **best variant–disease** pairing macro-F1; strongest when **shared multitask** and **precision variant** behavior are prioritized alongside BC5CDR.

**No universal winner:** Under **`balanced_scientific_default`**, **M021** ranks first if pairing tilt dominates; see `final_model_selection_rule.json` (`alternate_top_model_under_balanced_profile`).

### Compact recommendation scan

| Type | Model | Why | Main caveat |
|------|-------|-----|-------------|
| Default | **M015** | Max `benchmark_generalization_heavy`; strong BC5CDR | Not BioRED-only leader; not pairing-index leader |
| Secondary | **M021** | Strong pairing + stability; 2nd on primary profile | Lower BC5CDR than M015; BioRED slightly below S001 |
| Conditional | **S001 / S002** | BioRED cluster leader under single-metric lens | Weighted-CE branch; check stability policy |
| Conditional | **M003** | Strong PubMedBERT pipeline; BC5CDR competitive | Not default under composite |
| Diagnostic | **M005**, **M026** | T3 aux / anomaly controls | **M005** loses externally vs M015; **M026** penalized |

---

## 8. Weighted-CE branch (S001 / S002)

- **Position:** **Externally strong on BioRED** (top of the BioRED column on the shortlist).
- **Not automatically default:** The composite encodes **BC5CDR**, **stability**, **pairing**, and **branch penalties** — naive **BioRED-only** selection (**DP3**) picks **S001**, which is **intentionally different** from the benchmark-first default **M015**.
- **Worthy of serious attention** when the product is **BioRED-like** and branch risk is operationally acceptable.
- **Branch-risk baggage remains real** — treat as **conditional**, not a silent upgrade to M015.

---

## 9. Conditional model recommendations

| Condition | Model | Rationale |
|-----------|-------|-----------|
| Maximize **BioRED** mean F1 (explicit) | **S001** / **S002** | Top BioRED cluster; **weighted-CE** branch — use only if stability review accepts |
| Maximize **PubMedBERT** pipeline line | **M003** | Strong BC5CDR; close to M015 on several axes |
| Maximize **variant–disease** pairing | **M021** | Highest `pairing_variant_disease` macro-F1 |
| Parameter-efficient preference | **Not M025** on external — **M025** lags both benchmarks vs M015 |
| T3 weak aux value | **Not supported externally** — **M005** loses vs **M015** on both benchmarks |

---

## 10. Pairing-centric findings

- **Hardest family globally:** **drug–gene** (mechanism-heavy; lowest macro-F1 band ~0.34–0.46) — consistent with **S2_current** collapsing DrugProt mechanisms to **`DRUG_GENE_REGULATION`**.
- **Strongest family for M021:** **variant–disease** (~0.699 mean macro-F1) — oncology-relevant **precision** signal.
- **M015** remains mid-pack on variant–disease vs M021 but leads **drug–disease** / **BC5CDR**-aligned chemistry tasks in aggregate benchmarks.

Tables: `pairing_analysis_table.csv`, `pairing_model_profiles.csv`, `pairing_support_table.csv`.

---

## 11. Clinically anchored findings

`clinical_anchor_analysis_table.csv` and `schema_pressure_points.csv` summarize:

- **Mechanistic vs interpretable regions:** High error pressure on **drug–gene** pairs — where clinical mechanism granularity is lost under coarse schema.
- **Variant-linked behavior:** **M021** captures variant–disease signal best under current heads — relevant for **precision oncology** narratives that stress **alterations**.
- **S2_current sufficiency:** Adequate for **coarse** relation deployment and benchmark-style evaluation; **insufficient** for assertion subtypes (predictive vs diagnostic) without label or head refinement.

---

## 12. What remains uncertain

- **DrugProt official test** not evaluated — drug–gene story **incomplete** vs external standard.
- **Human-coded manual audit** deferred — proxy metrics may **overstate** usability.
- **Default vs balanced profile** disagreement (M015 vs M021) shows **policy sensitivity** — deployment must fix weights explicitly.
- **Weighted-CE lines** (S001/S002): strong BioRED but **branch-risk** — not promoted to default without further stability policy.

---

## 13. Future enhancement path

- **Data:** Package **DrugProt test**; optional **human audit** slice.
- **Baselines:** Executed pairwise **B2–B4**; table-derived **decision-policy** sweep **DP1–DP6**; training baselines **B1**, **B5** remain future (`external_baseline_plan.json`, `executed_vs_unexecuted_baselines.csv`).
- **Ensemble / distillation:** Documented in `future_enhancement_note.md` — **not implemented**; requires agreed teacher set and external validation before replacing **M015** default.

---

## Appendix — External baseline and decision-policy evidence

**Executed pairwise contrasts** (`external_baseline_results.csv`, audit: `genuine_executed_baseline`): **M003 > M009** on BC5CDR, **M015 > M025** on both benchmarks, **M015 > M005** on both — supporting that **stage-aware mainline** choices add value over the tested simpler or control alternatives **on this protocol**.

**Decision-policy baselines** (`decision_policy_baseline_comparison.csv`, `decision_policy_narrative.md`, `external_baseline_comparison_report.md`): The **weighted default (DP1)** differs from **BioRED-only (DP3)** and from **stability/pairing-only (DP5/DP6)**. That is why an **explicit rule** is necessary: naive single metrics **would not** have all chosen **M015**. The final policy is **more defensible** than picking the BioRED column alone while still honoring **BC5CDR** and **penalties**.

---

*End of final model selection report.*
