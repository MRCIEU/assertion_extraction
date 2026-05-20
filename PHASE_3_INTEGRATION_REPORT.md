# Phase 3.0 — Manuscript integration sanity check

**Branch:** `phase_d_baselines`  
**Repo root:** `project_1`  
**Manuscript root:** `report/project/`  
**Date:** 2026-05-20  
**Scope:** Read-only audit of file closure, cross-refs, citations, figures, and optional TeX build.

---

## Severity legend

| Level | Meaning |
|-------|--------|
| **CRITICAL** | Phase 3 must not proceed on prose until fixed |
| **WARNING** | Fix during Phase 3; not a hard integration blocker if understood |
| **COSMETIC** | Defer to final pre-submit polish |

---

## ACTION 1 — `\input` closure (`main.tex`)

Checks paths relative to `report/project/` (`stem` → `stem.tex`).

| Input | Status |
|-------|--------|
| `sections/abstract` | Present |
| `sections/01_background` | Present |
| `sections/03_methods` | Present |
| `sections/04_results` | Present |
| `sections/05_discussion` | Present |
| `sections/06_conclusion` | Present |
| `sections/ethical_approval` | Present |
| `sections/data_availability` | Present |
| `sections/funding` | Present |
| `sections/contributorship` | Present |
| `sections/acknowledgements` | Present |

**Missing from `main.tex` but present on disk:** `sections/02_objectives.tex` exists under `sections/` but **is not** `\input` in `main.tex` (document jumps from background to methods).

- **Severity:** **WARNING** — Likely intentional consolidation or an oversight; Phase 3 should explicitly **wire in**, **merge** into §1/§3, or **delete** the file to avoid drift.

**CRITICAL missing inputs:** none among the user’s expected list.

---

## ACTION 2 — `\input` closure (`supplementary.tex`)

All expected supplement chunks resolve:

| Input | Status |
|-------|--------|
| `supplement/A_prereg` … `supplement/K_stats_plan` | Present |
| `supplement/L_civicmine_phase_d` | Present |
| `supplement/M_llm_baseline_phase_d` | Present |

**Issues:** none.

---

## ACTION 3 — Cross-references (`\ref`, `\eqref`, `\pageref`; `\cite` excluded)

**Files scanned:** `main.tex`, `supplementary.tex`, `sections/*.tex`, `supplement/*.tex`.

| Metric | Value |
|--------|------|
| Reference commands expanded (comma-split where applicable) | **52** |
| Unique keys referenced | **37** |
| `\label{...}` instances | **82** |
| Unique labels | **82** |

**Undefined references** (referenced key with no `\label` in the scanned tree): **none**.

**Unreferenced labels (“orphans”):** **45** keys are defined but never referenced by `\ref`/`\eqref`/`\pageref` in the scanned files. Examples include main section labels (`sec:background`, `sec:methods`, `sec:results`, …), many supplement section labels (`supp:data`, `supp:trainer`, …), and some table/figure labels (`tab:s6-hypotheses`, `fig:s1-slopes`, …).

- **Severity:** **COSMETIC** for most — common for supplement TOC/bookmark anchors and tables only cited in passing text elsewhere; verify during Phase 3 if any label was *meant* to be cited from the main text.
- No **CRITICAL** cross-ref breakage detected.

**Note:** `\cref`/`\Cref` (cleveref) do not appear in this tree; the audit does not use cleveref.

---

## ACTION 4 — Citations vs `references.bib`

**Files scanned:** same manuscript `.tex` set as above. `\cite...{...}` arguments were split on commas (simple forms; uncommon natbib optional syntax may need manual spot-check).

| Metric | Value |
|--------|------|
| Citation key occurrences (expanded) | **46** |
| Unique cite keys | **28** |
| **Cite keys missing from `references.bib`** | **none** |

**Bibliography entries never cited in scanned `.tex`:** **1**

- `wishart2018drugbank`

**Severity:** **COSMETIC** — unused entry (remove or cite during Phase 3).

---

## ACTION 5 — `\includegraphics` resolution

Paths resolved relative to `report/project/` (compile directory). All referenced assets exist:

| File | Referenced from |
|------|-----------------|
| `figures/fig1_schema_selection.png` | `sections/04_results.tex` |
| `figures/fig2_training_configs_benchmarks.png` | `sections/04_results.tex` |
| `figures/fig2_forest_plot.png` | `sections/04_results.tex` |
| `figures/fig4_variance_asymmetry_and_ordinal.png` | `sections/04_results.tex` |
| `figures/fig_S2_slope_forest.png` | `supplement/J_additional_figures.tex` |
| `figures/fig_S3_ordinal_histogram.png` | `supplement/J_additional_figures.tex` |
| `figures/fig3_kb_surfacing.png` | `supplement/J_additional_figures.tex` |
| `figures/fig_S4_roberta_reference.png` | `supplement/J_additional_figures.tex` |
| `figures/fig_supp_kb_argmax_distribution.png` | `supplement/J_additional_figures.tex` |

**Missing figures:** none.

**Severity:** none.

---

## ACTION 6 — `pdflatex` (draft mode)

Command: `pdflatex -interaction=nonstopmode -draftmode main.tex` from `report/project/`.

| Result | Detail |
|--------|--------|
| Completed | **No** |
| First blocking error | `Package babel Error: Unknown option 'english'.` |
| Subsequent error | `LaTeX Error: File fancyhdr.sty not found` |

**Interpretation:** **Known environment / TinyTeX incompleteness** on this host (matches pre-Phase 3 audit). Not treated as a manuscript defect.

**Undefined references / warnings from `.log`:** not available (run stopped before a meaningful `.aux` / bibliography pass).

**Severity:** **none** for repo — document as env issue; install `babel-english`, `fancyhdr`, and finish full TeX Live where PDF proofing is needed.

---

## ACTION 7 — Phase 3 readiness summary

| Check | Status |
|--------|--------|
| All `main.tex` `\input` targets (user list) on disk | **Pass** |
| All `supplementary.tex` `\input` targets on disk | **Pass** |
| Cross-ref targets for all `\ref`/`\eqref`/`\pageref` | **Pass** |
| All `\cite` keys in `references.bib` | **Pass** |
| All main/supplement ` \includegraphics` files on disk | **Pass** |
| Clean local `pdflatex` | **Blocked by env** |

**Overall:** Integration closure is **sound** for static checks. **No CRITICAL** items among `\input`/`\ref`/`\cite`/figures. Outstanding points:

1. **WARNING:** `sections/02_objectives.tex` not included from `main.tex`.
2. **COSMETIC:** 45 unreferenced `\label{}`; one unused bib entry `wishart2018drugbank`.
3. **Env:** Full PDF build requires fixing babel/fancyhdr (or equivalent) on the build machine.

**Recommendation:** Phase 3 prose revision may start after review of this report; address **02_objectives** explicitly in the Phase 3 outline.

---

*Generated by automated scan + manual policy labels. TeX build outcome is host-specific.*
