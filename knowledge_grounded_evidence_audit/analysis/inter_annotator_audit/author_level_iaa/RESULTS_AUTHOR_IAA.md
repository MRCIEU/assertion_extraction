# Author IAA results — quick reference

**Branch:** `phase_d_baselines`  
**Analysis:** `author_level_iaa/compute_author_iaa_kappa.py` (read-only ingest of `author_iaa_labeling_workbook.xlsx`)  
**Date:** 2026-05-20  

---

## Headline numbers

| Contrast | κ (point) | 95% bootstrap CI | Notes |
|----------|-----------|------------------|--------|
| **Heuristic vs LLM Opus** (sanity) | **0.561** | [0.321, 0.790] with seed **20260520**; [0.320, 0.796] with seed **42** | Matches manuscript **0.56** and published CI band (**7**/30 disagreements). |
| **Heuristic vs author** | **0.434** | [0.214, 0.654] | **10**/30 disagreements; Landis & Koch **moderate** (point estimate in fair–moderate band). |
| **LLM Opus vs author** | **0.835** | [0.628, **1.000**] | **3**/30 disagreements; upper CI hits 1.0 (small-$n$ bootstrap artefact). |
| **Fleiss κ (3 raters)** | **0.603** | — | stdlib implementation on 3× category-count matrix. |

**Workbook note:** `author_confidence_0_to_1_FILL_IN` was **absent** from the saved Excel schema (columns compressed to 13); `author_audit_labels.csv` leaves `author_confidence` empty.

---

## On the seven known heuristic–vs–LLM disagreement targets

| Pattern | Count / 7 |
|---------|----------|
| Author agrees with **LLM** (`__NEGATIVE__` vs heuristic positive) | **7** |
| Author agrees with **heuristic** (positive vs LLM NEG) | **0** |
| **Other** (third label) | **0** |

IDs: `GL_0031`, `GL_0039`, `GL_0043`, `GL_0068`, `GL_0070`, `GL_0118`, `GL_0131`.

---

## What this means for the paper

1. **Heuristic vs human:** κ(author, heuristic) = **0.43** is **below** κ(LLM, heuristic) = **0.56** — the first author is **less** aligned with the heuristic projection than the Opus proxy was, mainly via **extra** human–heuristic disagreements (10 vs 7). Per the Phase 3 carry-over rubric, this supports **strengthening §5 limitations** around the gold-lite heuristic / CIViC-text closure, *without* walking back the directional story on the seven audited cases (human sides with the text-only NEG read there).

2. **LLM vs human:** κ(author, LLM) ≈ **0.84** is **high** — the Opus second annotator tracks the author’s literal reading closely on this coarse six-label task. That **supports** continued use of the LLM pass as a **structured validity check** aligned with human judgment, **not** as a substitute for multi-human gold.

3. **“Lower bound” wording in §4.2:** The draft insert in `outputs/PAPER_INSERTS_AUTHOR_IAA.md` should be **edited during Phase 3D**: κ(heuristic, author) **\<** κ(heuristic, LLM) means the LLM was **not** a pessimistic lower bound **for agreement with the heuristic**—it sat **between** human and heuristic on that axis. The accurate statement is that **human and LLM largely agree** (high κ) **while both** diverge from the heuristic on overlapping (but not identical) subsets; on the **seven** canonically disputed rows, **human = LLM** throughout.

---

## Files manifest

| Path | Role |
|------|------|
| `author_level_iaa/author_audit_labels.csv` | Cleaned author labels (from workbook; do not overwrite xlsx). |
| `author_level_iaa/outputs/iaa_three_way_labels.csv` | Heuristic + LLM + author join. |
| `author_level_iaa/outputs/author_iaa_kappa_results.json` | Point + CI + Landis labels. |
| `author_level_iaa/outputs/author_iaa_disagreement_structure.md` | Markdown tables. |
| `author_level_iaa/outputs/author_iaa_disagreement_structure.csv` | Machine-readable pairs. |
| `author_level_iaa/outputs/author_rationale_summary.md` | All 30 rationales transcribed. |
| `author_level_iaa/outputs/PAPER_INSERTS_AUTHOR_IAA.md` | **Draft** prose — **revise “lower bound” sentence** before `\input`/paste. |
| `author_level_iaa/compute_author_iaa_kappa.py` | Reproducible analysis script. |

**Not committed:** `author_iaa_labeling_workbook.xlsx` (raw filled workbook stays local / Zenodo policy per project).

---

## Ready for Phase 3D

**Yes** — integration-ready once Freddie (1) **revises** the §4.2 + supplement insert to reflect κ(heur, author) \< κ(heur, LLM) accurately, and (2) optionally re-export Excel with a **confidence** column for archival `author_audit_labels.csv`.

---

*HALT: no `.tex` files modified in this carry-over.*
