# Pre–Phase 3 hygiene fix — summary

**Repo:** `project_1` (`assertion_extraction`)  
**Branch:** `phase_d_baselines`  
**Commits:** `b3cb8c8` — *Track paper LaTeX source … pre-Phase 3 hygiene fix* (pushed to `origin/phase_d_baselines`)  
**Date:** 2026-05-20  

---

## What was wrong on `master`

The pre-Phase 3 audit showed the **entire** `report/project/` manuscript tree was **absent from the `master` commit** (`?? report/project/`), while Phase 2 work on `phase_d_baselines` had already committed a small subset (**four** paths): discussion §5, supplementary driver, and supplements **S13 / S14**.

This hygiene commit **does not** merge to `master` (see branch policy below). It completes tracking of the **rest** of the LaTeX tree on **`phase_d_baselines`** so `main.tex` and the supplement chain have a single coherent, versioned manuscript root.

---

## `.gitignore` changes (signed-off)

### Block 1 — Ignore tool trees (approved)

```gitignore
report/project/.prism/
report/project/.claudeprism/
```

These directories must not be committed (embedded history / IDE artifacts).

### Block 2 — Force-include manuscript paths (last-block wins)

Placed at **end of** `.gitignore` so negation overrides earlier broad patterns:

```gitignore
!report/project/**/*.tex
!report/project/*.tex
!report/project/**/*.bib
!report/project/**/*.cls
!report/project/**/*.sty
!report/project/figures/**
!report/project/auxiliary/**
```

**Review adjustments applied:** explicit `!report/project/*.tex` for top-level TeX; `figures/**` replaces per-extension lines (`*.png` / `*.pdf`) for SVG/JPEG/subdirectories. **`*.json` / `*.csv` rules unchanged** — Phase 2 JSON remains `git add -f` when regenerated.

---

## `schema_projection.json` (Adjustment 4 — follow-up)

| Check | Result |
|--------|--------|
| `git ls-files knowledge_grounded_evidence_audit/schema_projection.json` | **Empty — not tracked** |
| File present in this workspace | **No** (`Glob` / `test -f` found nothing at that path) |

**Action:** Treat as a **separate** task (correct path in your data layout, then `git add -f` if still under `*.json` ignore). **Not** part of this hygiene commit.

---

## Branch policy (Option B — **confirmed**)

- **`phase_d_baselines`** is the **active manuscript + Phase 2/3 revision branch** until submission.
- **Do not merge to `master`** until submission (or an explicit project milestone).
- **After submit:** merge `phase_d_baselines` → `master` as the submission snapshot; user suggestion for later: annotated tag `paper_submission_v1` (not created in this step).

---

## `report/project/` — tracked manifest (after `b3cb8c8`)

Counts from `git ls-files`:

| Category | Tracked count | Notes |
|----------|---------------|--------|
| **Root** | `main.tex`, `references.bib`, `supplementary.tex` | `supplementary.tex` + §5 + S13/S14 were already on branch before this commit; this commit adds the bulk trunk |
| **`sections/*.tex`** | **13** | Includes `05_discussion.tex` (Phase 2 discussion + limitations) |
| **`supplement/*.tex`** | **13** | `A_prereg` … `K_stats_plan`, plus **`L_civicmine_phase_d.tex`**, **`M_llm_baseline_phase_d.tex`** |
| **`figures/*.png`** | **9** | Matches `\\includegraphics{figures/...}` in `04_results.tex` / `J_additional_figures.tex` |
| **`figures/*.pdf`** | **1** | `fig2_forest_plot.pdf` |
| **`auxiliary/*`** | **1** | `cover_letter.txt` |
| **Total under `report/project/`** | **40** | `git ls-files report/project/ \| wc -l` |

### Explicit file list (high level)

- **Main:** `report/project/main.tex`
- **Sections:** `abstract.tex`, `01_background` … `04_results`, **`05_discussion`**, `06_conclusion`, back-matter snippets (`data_availability`, `ethical_approval`, `funding`, `contributorship`, `acknowledgements`, `plain_language_summary`)
- **Supplement:** `A`–`K`, **`L_civicmine_phase_d`**, **`M_llm_baseline_phase_d`**
- **Bibliography:** `references.bib`
- **Figures:** all PNGs referenced by the main + J supplement figures, plus `fig2_forest_plot.pdf`
- **Auxiliary:** `cover_letter.txt`

### `pdflatex` input closure (no run performed)

With a full TeX installation, **`main.tex`** should resolve `\input{sections/...}` for every section file **present and tracked** above, including **`05_discussion`**. Build output logs (`.log`, `.aux`, etc.) remain ignored by `*.log` / standard artifacts. Supplementary PDF is driven by **`supplementary.tex`** (tracked). Environment issues on the audit host (`babel` / `fancyhdr`) are **out of scope** for this hygiene task.

---

## Phase 2 collateral (already on branch)

- **`knowledge_grounded_evidence_audit/analysis/phase_d_baselines/RESULTS_TO_REPORT_PHASE_D.md`** — **tracked** (`git ls-files` confirms). Use as the §3–§5 / supplement insert guide for Phase 3.
- Phase 2 JSON artefacts under `phase_d_baselines/outputs/` and related paths were handled earlier with **`git add -f`**; global **`*.json`** ignore remains.

---

## Phase 3 — ready-to-launch checklist

- [x] Manuscript tree fully tracked on **`phase_d_baselines`**
- [x] Phase 2 supplements **S13 / S14** integrated (`L_*`, `M_*`) + **`supplementary.tex`** (pre-existing on branch)
- [x] **`RESULTS_TO_REPORT_PHASE_D.md`** present and tracked for editorial insert guidance
- [ ] **`schema_projection.json`** — **not** resolved in this fix; schedule explicit add / path confirmation
- [ ] Author **IAA** — 30 targets; blinding protocol; standby κ for §4.2 when ready

---

## Related artefacts

- **`/tmp/gitignore_audit.md`** — pre-commit ignore analysis (local)
- **`/tmp/proposed_gitignore.diff`** — final unified diff vs `HEAD~1`’s `.gitignore` equivalent (local); matches applied commit

---

*This file was added with `git add -f` to override root `*.md` ignore.*
