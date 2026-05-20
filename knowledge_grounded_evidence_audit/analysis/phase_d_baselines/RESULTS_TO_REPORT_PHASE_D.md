# Phase D — results to cite (AUTO summary)

Frozen artefacts:

- Coverage + PB subsets: `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/civicmine_baseline_case_c.json`
- $R_{\mathrm{B}}$ extensions: `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/rb_phase_d_extensions.json`
- Phase~2C $\widehat\alpha$ + paired contrasts: `knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/phase_d_alpha_attribution.json`

**§S13.1 narrative anchor (selection bias):** Any “41-set” accuracy is conditional on CIViCmine’s **self-selected** strict coverage. Those numbers are an **upper-bound slice**, not a draw from the $n=162$ audit population (~**75%** strict non-coverage). Lead with **Layer A** before score rows. Every manuscript table that prints 41-set accuracies must carry the **selection-bias disclaimer** (see Supplement §S13.1 boxed caption + table note).

---

## Layer A — Coverage limitation ($n_{\mathrm{eval}} = 162$)

Under **strict** PMID + typed entity-pair matching, **121 / 162 (~74.7%)** of targets are **not** strictly covered by CIViCmine (non-coverage dominates). Three **$n=162$ sensitivity treatments** for CIViCmine-equivalent bookkeeping appear in the Case C JSON (`civicmine_162_denominator_heuristics_*`):

| Quantity | Value |
|-----------|------:|
| Strict entity-pair coverage | **41 / 162 (25.31%)** |
| PMID-only coverage (any row on curator PMID) | **79 / 162 (48.77%)** |
| PMID missing entirely (no CIViCmine support) | **83** |
| PMID present, strict pair missing | **38** |

| Scenario (full-162 imputation) | Accuracy |
|----------------------------------|--------:|
| **Exclude uncovered** | **0.241** |
| **NEG surrogate** | **0.241** |
| **Random IID** (analytic $\|\text{gold set}\|/8$) | **0.334** |

---

## Layer B — Strictly covered subset ($n=41$; selection-biased)

**Framing:** Compare systems **only** on CIViCmine’s realised strict-**41** `target_id` list. CIViCmine **0.951** (39/41) is **not** comparable to population PB metrics on 162; PB rows below are **recomputed on the same 41** (20-seed means for trainers).

| System | Mean KB argmax (41 targets) |
|--------|-----------------------------:|
| CIViCmine strict mapping | **0.951** (39/41) |
| PubMedBERT × FT × **T2** | **0.855** |
| PubMedBERT × FT × **T1F-4096** | **0.782** |
| PubMedBERT × FT × **T1F-2048** | **0.549** |
| PubMedBERT × FT × **T1B** | **0.262** |

**Table note (required):** *41-set accuracies condition on extractor support; CIViCmine’s score is an upper-bound slice relative to the broader $n=162$ population. Pair with Layer A.*

---

## Layer C — Non-coverage anatomy

| Bucket | Count |
|--------|------:|
| PMID missing (no CIViCmine row on curator PMID) | **83** |
| PMID present, strict pair missing | **38** |
|   • ≥1 gold slot seen in CIViCmine rows on that PMID | **31** |
|   • neither slot seen | **7** |

---

## Unified comparison table (cross-system)

| System | KB acc (162), mean @ 20 seeds | KB acc (41), mean @ 20 seeds |
|--------|-------------------------------:|-----------------------------:|
| CIViCmine strict | — | **0.951** (39/41) |
| PubMedBERT × FT × **T2** | **0.756** | **0.855** |
| PubMedBERT × FT × **T1F-2048** | **0.477** | **0.549** |
| PubMedBERT × FT × **T1F-4096** | **0.619** | **0.782** |
| PubMedBERT × FT × **T1B** | **0.150** | **0.262** |

**Caption / note:** The **41** column shares CIViCmine’s self-selected support list; **selection bias** applies. The **162** column is population KB audit for trainers. Do **not** headline-compare CIViCmine 0.951 to PB 0.756 without Layer A.

---

## §S13.3 — Phase 2C (matched compute) + Discussion paste block

### Numbers (`phase_d_alpha_attribution.json`)

| Contrast | Point | 95% CI | Notes |
|----------|------:|--------|-------|
| **Δ_compute** (T1F-4096 − T1F-2048), paired mean on 162-set KB argmax | **+0.142** | **[-0.017, 0.299]** | directional; interval includes 0 |
| **Δ_content** (T2 − T1F-4096), paired mean, matched compute | **+0.137** | **[0.036, 0.240]** | excludes 0 |
| **$\widehat\alpha$** = mean Δ_compute / mean Δ_gap | **0.509** | **[-0.087, 0.868]** | ratio bootstrap ($B=5000$, seed 20260518) |

| | |
|--|--|
| Paired $t$ (descriptive), Δ_content | $p \approx 0.017$ |
| Paired $t$ (descriptive), Δ_compute | $p \approx 0.102$ |

### Paste scaffold for §5 Discussion (manuscript writer; Freddie rewrites final prose)

1. **Design (one sentence):** Phase~2C held **PubMedBERT-base × full fine-tuning** fixed and added a **matched-compute** **T1 flat @ 4096** cell vs the pre-registered **T1F @ 2048** and **T2** cells so that **additional gradient steps** (multi-corpus continuation) can be separated—seed-for-seed, on the same **162-evaluable KB surface**—from the **T2 oncology-projected staging** trajectory at **matched total updates**.

2. **Contrasts (point + CI):** **Δ_compute** = **+0.142** (paired bootstrap 95% **[-0.017, 0.299]**) for **T1F-4096 minus T1F-2048**; **Δ_content** = **+0.137** (**[0.036, 0.240]**) for **T2 minus T1F-4096** at matched compute; **$\widehat\alpha$** = **0.509** with bootstrap **[-0.087, 0.868]** on the ratio of mean paired gaps defined in `COMMITMENT.md`.

3. **Verdict (one sentence):** Per the signed **`COMMITMENT.md`** rule table, the bootstrap **α** interval is too wide to assert compute- vs content-dominance—the mapped verdict is **“mixed; attribution uncertain.”**

4. **Reframing vs prior schedule story (one sentence):** Where the paper previously summarised the schedule axis primarily as **“content”** (T2 vs T1), post-hoc decomposition shows the schedule lift **splits into compute-like and content-like components of similar magnitude**, with **content statistically detectable at $n=20$** (Δ_content CI excludes zero) and **compute supported only at the point-estimate level** (Δ_compute CI includes zero).

---

## $R_{\mathrm{B}}$ diagnostics (`rb_phase_d_extensions.json`)

| Block | Notes |
|-------|--------|
| **`PB_only_four_schedule_rb`** | PB × (T1B / T1F-2048 / T1F-4096 / T2). Point **0.762**; 95% CI **[0.011, 6.38]** — very wide; cite JSON. |
| **`augmented_ten_cell_encoder_schedule_rb`** | Exploratory 10 strata. Point **0.237**; CI **[0.039, 1.08]**. Does **not** replace nine-cell headline **≈ 0.21**. |

---

## Refresh commands after new GPU outputs

```bash
export PYTHONPATH=/path/to/project_1
python3.11 report/scripts/aggregate_phase_b_ft.py
python3.11 knowledge_grounded_evidence_audit/analysis/phase_d_baselines/civicmine/run_civicmine_baseline.py
python3.11 knowledge_grounded_evidence_audit/analysis/phase_d_baselines/analysis/phase_d_rb_extensions.py
python3.11 knowledge_grounded_evidence_audit/analysis/phase_d_baselines/analysis/phase_d_alpha_attribution.py
```

## Phase 2B — LLM baseline (`llm_baseline/`)

After `export OPENAI_API_KEY=...` (or `source ~/.bashrc`):

```bash
export PYTHONPATH=/path/to/project_1
export BUDGET_USD=1.0
python3.11 knowledge_grounded_evidence_audit/analysis/phase_d_baselines/llm_baseline/probe_openai.py
python3.11 knowledge_grounded_evidence_audit/analysis/phase_d_baselines/llm_baseline/run_gpt4o_mini_baseline.py
```

Outputs: `outputs/llm_baseline/gpt4o_mini_{zero_shot,six_shot,six_shot_rationale}.json` with **kb_hit_mean_162** and **kb_hit_mean_41** per condition.
