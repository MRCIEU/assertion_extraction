# External baseline comparison report

*Companion machine-readable files: `baseline_suite_audit.json`, `baseline_suite_audit_table.csv`, `baseline_family_definition.csv`, `external_baseline_plan.json`, `executed_vs_unexecuted_baselines.csv`, `decision_policy_*.csv`, `decision_policy_narrative.md`, `external_baseline_results.csv`, `external_baseline_limitations.json`.*

---

## 1. Why baseline strengthening was necessary

The project’s scientific claims rest on **staged training**, **schema-aware packaging**, and **multi-criteria selection** — not on a single leaderboard score. A thin baseline layer that only restates **pairwise deltas** among shortlisted runs risks looking like **post-hoc rationalization**. Strengthening separates: (a) **controlled executed contrasts** on the official external protocol, (b) **selection-policy** comparisons that show what **naive deployment rules** would have chosen from the same numbers, and (c) **honest gaps** where a heavier baseline suite is not yet run.

---

## 2. What counted as a meaningful baseline here

- **Genuine executed baseline (training/architecture sense):** Two frozen checkpoints evaluated on **split_external** under the **same** BioRED/BC5CDR protocol — pairwise rows **B2–B4** in `external_baseline_results.csv`.
- **Meaningful policy baseline:** A **selection rule** (weighted default vs single-metric argmax) applied to **`final_model_selection_summary.csv`** — **DP1–DP6**. These are **not** new GPU runs; they are **decision-theoretic** baselines.
- **Not counted as evidence:** Planned but **unrun** training baselines (**B1**, **B5**), **DrugProt** test (blocked), and **literature SOTA** comparisons (out of scope).

---

## 3. Baseline families

| Family | Intent | Members |
|--------|--------|---------|
| **1 — Simpler training-policy alternatives** | Stage / aux controls vs flatter or weaker-aux training | **B1** (planned), **B4** (executed) |
| **2 — Architecture / update simplifications** | Pipeline vs shared; full vs top-four; frozen encoder | **B2**, **B3** (executed), **B5** (planned) |
| **3 — Selection-policy baselines** | Naive single-metric picks vs explicit weighted rule | **DP1–DP6** (table-derived) |

Full definitions: `baseline_family_definition.csv`, `external_baseline_plan.json` version 2.

---

## 4. Executed baseline evidence

| Contrast | Result (headline) | Caveat |
|----------|-------------------|--------|
| **M003 vs M009** (pipeline vs shared) | **M003** ahead on BC5CDR; small BioRED delta | One architecture fork; not all shared encoders |
| **M015 vs M025** (full vs top-four) | **M015** ahead on **both** benchmarks | Efficiency ablation — top-four lags transfer here |
| **M015 vs M005** (mainline vs T3 aux) | **M015** ahead on **both** | Weak T3 mixture not vindicated vs mainline on this protocol |

Source: `primary_external_results.csv` via `external_baseline_results.csv`. **Audit classification:** `genuine_executed_baseline` for B2–B4 (`baseline_suite_audit.json`).

---

## 5. Decision-policy baselines

Naive rules on the **same shortlist** (`final_model_selection_summary.csv`) yield:

| Rule | Picks | Divergence from default weighted (DP1) |
|------|-------|----------------------------------------|
| Weighted default (**DP1**) | **M015** | — |
| Internal HR only (**DP2**) | **M015** | Aligned *on this shortlist* |
| BioRED only (**DP3**) | **S001** | **Diverges** — BioRED-first favors weighted-CE line |
| BC5CDR only (**DP4**) | **M015** | Aligned |
| Min BioRED seed std (**DP5**) | **M021** | **Diverges** — stability/pairing-strong line |
| Max pairing_clinical_index (**DP6**) | **M021** | **Diverges** — pairing-first |

**Takeaway:** The **default policy** is **not** identical to “best BioRED” or “best pairing index.” Making the **weighted rule** explicit avoids **accidentally** shipping **S001** (BioRED-only) or **M021** (pairing/stability-only) when the intended product priority is **joint** benchmark generalization.

Details: `decision_policy_baseline_comparison.csv`, `decision_policy_tradeoff_table.csv`, `decision_policy_narrative.md`.

---

## 6. What the strengthened baseline layer now supports

- **Training design:** Executed **B2–B4** support that **mainline choices** (pipeline advantage on BC5CDR in B2, full fine-tune in B3, no T3 aux in B4) are **externally** directionally consistent — on **BioRED + BC5CDR only**.
- **Selection design:** **DP1–DP6** show that **who ships as default** depends on **stated priorities**; the project’s **benchmark-heavy composite** is **more defensible** than BioRED-only maximization for **dual-benchmark** deployment.
- **Honesty:** Unrun **B1/B5** and **DrugProt** are labeled **not executed**, not hidden.

---

## 7. What remains missing

- **B1** naive single-stage, **B5** frozen encoder — no artifacts.
- **DrugProt** official test — **formal blocker** (no packaged test split).
- **Literature / third-party** checkpoints — not compared.
- **Policy agreement** (DP1 = DP2 here) may **not** hold after future registry changes — the **process** (explicit weights) remains the safeguard.

---

## 8. Implications for model selection

1. **Default remains policy-driven:** **M015** under `benchmark_generalization_heavy` — not “winner of all metrics.”
2. **S001/S002** remain **conditional** when **BioRED** is the dominant product constraint; **DP3** formalizes why.
3. **M021** remains **secondary** or **pairing-tilt** default under **`pairing_clinical_anchored`** or explicit pairing-first policy — **DP5/DP6** align with that narrative.
4. **No recommendation is free of assumptions** — the baseline layer makes **assumptions visible** instead of embedding them in an implicit pick.

---

*End of external baseline comparison report.*
