# Results (§4) — evidence placement map

## §4.1 Schema selection (RQ1)

| Order | ID | Integration note |
|------:|-----|------------------|
| 1 | **E-001**, **E-002** | Primary **KB** contrasts + Table `tab:t1` alignment. |
| 2 | **E-002b** | Benchmark guardrail for schema choice (Supplement E narrative). |
| 3 | **E-003**, **E-004** | Unanimity + **S_mech** collapse — short factual supports. |

## §4.2 Inter-annotator audit — **REWRITE**

| Order | ID | Integration note |
|------:|-----|------------------|
| 1 | **E-019** / **E-052** | Reproduce **κ(heuristic, LLM)** ≈ **0.561** [**0.32**, **0.80**]. |
| 2 | **E-052** | **NEW κ(heuristic, author)** **0.434** [**0.21**, **0.65**]. |
| 3 | **E-052** | **NEW κ(LLM, author)** **0.835** [**0.63**, **1.00**]. |
| 4 | **E-053** | Optional **Fleiss** **0.603** (three raters). |
| 5 | **E-054** | **7/7** on **IAA-disagreement** rows: author sides with **LLM NEG**, not heuristic positive. |
| 6 | **E-055** | **Strip** deprecated “LLM lower bound on human” — replace with convergence framing (`DEPRECATED`). |

## §4.3 Training configurations (RQ2)

| Order | ID | Integration note |
|------:|-----|------------------|
| 1 | **E-008** | **H1** null contrasts (PL vs PB / PL vs BL). |
| 2 | **E-005**, **E-006** | **H2** multi-corpus headline + other encoders. |
| 3 | **E-007** | **H3** partial staging — **3 / 2 / 1** pattern + anchor cells. |

## §4.4 Benchmark–KB coupling / variance (RQ4)

| Order | ID | Integration note |
|------:|-----|------------------|
| 1 | **E-013** | Schema-wave **R_A** imbalance setup (≈92% vs ~40% lever shares). |
| 2 | **E-010** | **Nine-cell R_B ~0.214** [**0.028**, **0.990**] — **keep pre-registered wording**; fragile CI. |
| 3 | **E-011** | **Headline**: schedule share **~59.6%** (KB) vs **~8.2%** (BioRED) — **seven-fold** narrative. |
| 4 | **E-012** | Total lever shares **66.5%** vs **14.2%** — defines **R_B** context. |
| 5 | **E-020–E-023** | **Drop-7** sensitivity block — insert **after** Phase C artefacts pinned (`[PROVENANCE UNKNOWN]`). |
| 6 | **E-042–E-047** | **NEW subsection**: matched-compute decomposition (**Δ_compute**, **Δ_content**, **α̂**, CIs, **verdict**). |
| 7 | **E-049**, **E-050** | Supplement-only **extended R_B** designs + **do-not-overwrite** guardrail. |

### Rank inversion (still §4.4)

| Order | ID | Integration note |
|------:|-----|------------------|
| 1 | **E-014** | **18** tied pairs; **median |ΔKB| ~0.16**; inversion **0.50** [**0.14**, **0.83**]. |
| 2 | **E-015** | **ρ** sensitivity **0.01–0.05** — rate ~**0.50–0.55**. |
| 3 | **E-016** | Exact binomial interval from **`rho_sensitivity`** row **ρ=0.03**. |
| 4 | **E-027** | **Phase 1 Panel 3** flat width narrative (`[PROVENANCE UNKNOWN]`). |

## §4.5 **NEW** — External comparison / calibration layer

| Order | ID | Integration note |
|------:|-----|------------------|
| 1 | **E-041** | **Always-DGR 0.951** + **IID 0.125** — **calibration** caption lead-in. |
| 2 | **E-038**, **E-039**, **E-040** | **GPT-4o-mini** accuracies + **DGR/NEG** mass + **7-target** NEG disagreement table. |
| 3 | **E-031**, **E-033**, **E-034–E-036** | **CIViCmine** coverage + **0.951** on **41** + **PB-T2 0.855** on same **41** + denominator heuristics note. |

## §4.6 LoRA falsification

| Order | ID | Integration note |
|------:|-----|------------------|
| 1 | **E-018** | Collapse + mechanical null — unchanged skeleton. |

## §4.x RQ3 (audit metrics)

| Order | ID | Integration note |
|------:|-----|------------------|
| 1 | **E-009** | **Encoder × KB-metric** interaction SS share ≈**0.25%** (manuscript rounds **0.3%**). |
| 2 | **E-029** | **Phase 1 calibration** secondaries — once sourced (`[PROVENANCE UNKNOWN]`). |

## §4.y Coupling slopes (H6)

| Order | ID | Integration note |
|------:|-----|------------------|
| 1 | **E-017** | **Five** slope summaries **inconclusive** vs **0.30** width gate — reporting obligation only. |

---

## Logical bridges

- **E-010** → **E-011**: *ratio* ↔ *schedule share asymmetry* — **shares lead** for readers; **ratio** follows.
- **E-011** → **E-042–E-047**: *schedule dominates KB variance* ↔ *mean-gap decomposition explains schedule channel*.
- **E-020–E-023** → **E-028**: *target-set stability* ↔ *directional bias mechanism* — **independent** supports for “not an IAA artefact main driver” **once sourced**.
- **E-041** → **E-038–E-040**: *trivial anchor* ↔ *LLM* — **relative** interpretation only.
- **E-033** → **E-034**: *strict 41 coverage* ↔ *PB recomputed on 41* — **fair** comparison slice.
- **E-052** → **E-054**: *κ structure* ↔ *row-level 7/7* — avoids over-weighting a single scalar.
