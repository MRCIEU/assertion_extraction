# Claim boundary registry

This document fixes **what the thesis or paper may say strongly**, **only cautiously**, and **must not say**, given the evidence in this repository. It does **not** replace reading the cited tables.

---

## Claims the project can make strongly

- **Resource reality:** No single public dataset defines **full** precision-oncology **assertion** extraction for all desired entity/relation semantics; the inventory **bounds** what is trainable vs blocked.
- **Operational schema:** **`S2_current`** is the **frozen** coarse relation/assertion label space for generated T1–T4 JSONL and the fine-tuning registry under this project’s packaging assumptions.
- **Internal protocol:** The HR rerun produces **reproducible internal anchors and contrasts** (families, schedules, losses) under a **defined** trainer — useful for **relative** comparisons within protocol.
- **External protocol (BioRED + BC5CDR):** Official test evaluation is **complete** for the packaged protocol; results support **policy construction**, not a **single** universal winner across benchmarks and slices.
- **Downstream decoupling (documented proxies):** **Internal HR macro-F1 does not predict** gold-lite **pred_nonnegative** yield on the R1/C1-style proxy; **high HR** can co-occur with **zero** yield.
- **Downstream setting interaction:** **Family-specific** responses to settings — e.g. **M025** Tier-2 mean pred_nonnegative **~13.4 (S1)** vs **~0.2 (S2)** in aggregate over seeds — **documented** in `tier2_multiseed_results.csv`.
- **Policy transparency:** **Benchmark-first** selection (e.g. default **M015** under weighted policy) and **downstream-audit** roles (**surfacing vs conservative**) are **explicitly split** in separate rule artifacts — **not** hidden in one scalar.

---

## Claims the project can make only cautiously (qualifiers required)

- **M015 as default for benchmark-generalization:** Supported **under the stated weighting** in `final_model_selection_report.md` / `final_model_selection_rule.json` — **not** “best on every BioRED/BC5CDR slice” and **not** best for **audit surfacing** on the gold-lite proxy.
- **M003 / M025 as surfacing-oriented:** Supported **relative to other families** on **heuristic gold-lite** settings — **not** clinical usefulness and **not** stable for every setting (**M025** under S2).
- **M021 for variant–disease emphasis:** Supported **within pairing-slice evidence** — not global dominance.
- **Oracle O3 metrics:** May be cited as **diagnostic** magnitudes — **not** as meaningful ranking separation (near-zero band in aggregates).

**Required qualifiers:** “heuristic gold-lite,” “proxy,” “under documented settings,” “policy profile X.”

---

## Claims the project should not make

- A **universally best** single model for **all** objectives (benchmark, surfacing, conservative audit, variant-centric) — **contradicted** by split policies and downstream tables.
- **Clinical validation** or **therapeutic discovery** from extraction outputs, KB-gap counts, or surfacing volume.
- That **KB-absent** or **literature_gap** candidates are **discoveries** — they are **candidates** under audit taxonomy, not verified facts.
- That **oracle** metrics on gold-lite represent the **true** upper bound of usable performance — formulation is **weak/sparse** in aggregate.
- That **benchmark success alone** implies **audit readiness** — **explicitly contradicted** by downstream transfer results.
- That **DrugProt** official test results support any claim — **no** packaged test split; **blocker** is documented.
- That **human audit** backs usability — **deferred**; proxy metrics only.

---

## Epistemic boundary (short)

This project contributes **methods, evidence, and explicit decision rules** under **open-resource constraints**. It does **not** establish **clinical safety or efficacy** of deployed extraction.

*Cross-reference: `master_research_report.md` status table (Claim / epistemic boundary).*
