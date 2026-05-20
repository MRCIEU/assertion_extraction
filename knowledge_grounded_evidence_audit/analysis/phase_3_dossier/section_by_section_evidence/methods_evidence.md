# Materials & methods (§3) — evidence placement map

## §3.1–3.3 — mostly locked

| Order | ID | Integration note |
|------:|-----|------------------|
| 1 | **E-051** | Wave **n** and shared-seed policy (Configuration wave **190** incl. RoBERTa reference per locked JSON `coverage`). |
| 2 | **E-018** | LoRA **methodological null** Methods one-liner (already in `03_methods.tex`). |

## New Methods content (Phase 2/3)

| Order | ID | Integration note |
|------:|-----|------------------|
| 1 | **E-042** | **Matched-compute** design + **α̂** definition + bootstrap **B=5000** seed **20260518**. |
| 2 | **E-041** | **Trivial baselines**: IID-8 vs **always-DGR** on **singleton** expected sets. |
| 3 | **E-052**, **E-054**, **PAPER_INSERTS_AUTHOR_IAA.md** | **Author IAA**: **30** stratified (`random.Random(42)`), blinded protocol; **three-rater** CSV release path. |
| 4 | **E-038** | **GPT-4o-mini** conditions (zero / 6-shot / 6-shot+rationale) + **API** reporting line for **Supplement M**. |
| 5 | **E-031**, **E-032** | **CIViCmine** PMID + entity-slot bookkeeping for Case C — Methods or Supplement S13 pointer. |
| 6 | **E-056** | **ICC(1,1)** + within-cell SD sentence — **verify** against analysis artefact if desk-reject risk on reproducibility. |

---

## Logical bridges

- **E-042** → **E-043**: *definitions* ↔ *Y metric column* (`kb_hit_A_setvalued`).
- **E-041** → **E-038**: *trivial ceiling* ↔ *LLM API baseline* — same **audit metric**.
- **E-055** (DEPRECATED) → **E-052**: *replace LLM “lower bound” sentence* ↔ *triangulation κ matrix*.
- **E-018** → **E-042**: *failed LoRA* unrelated — prevents reviewer conflating **PEFT** with **compute attribution**.
