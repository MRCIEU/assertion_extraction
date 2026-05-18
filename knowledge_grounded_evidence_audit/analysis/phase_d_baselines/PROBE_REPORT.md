# Phase 2 Probe Report

Branch: `phase_d_baselines` (from `master`, 2026-05-18). HALTED at the
end of Phase 2.0 per the brief. No sub-phase work has started.

---

## Probe 1 — CIViCmine coverage

**TSV source**:
- Zenodo deposit DOI `10.5281/zenodo.7689629` (Lever et al., latest
  version 40, published 2023-03-01, license CC0).
- Cached locally:
  - `civicmine/civicmine_sentences.tsv.gz` 51.8 MB, MD5
    `f3d65ea0bbb11f9f5f7d5e62ad2ecb31` (matches Zenodo).
  - `civicmine/civicmine_unfiltered.tsv.gz` 117.1 MB, MD5
    `a60fb6abd7dc5b26113cc2fe67729261` (matches Zenodo).

**TSV schema** (36 columns): `matching_id`, `pmid`, `title`, `journal`,
`year`, `section`, `evidencetype` (Predictive / Diagnostic /
Prognostic / Predisposing), `evidencetype_prob`, `cancer_*`,
`gene_*` (hugo_id, entrez_id, text, normalized), `drug_*`,
`variant_*`, `sentence`, `formatted_sentence`. The unfiltered file
includes every predicted relation with predictprob > 0.5 (FP-tolerant);
the sentences file is the high-confidence subset (380,482 rows,
123,983 unique PMIDs).

**Our 162 evaluable target set**:
- 162 evaluable target_ids
- 80 unique PMIDs (multiple targets per paper; e.g. EGFR/erlotinib
  and EGFR/gefitinib both anchor on the same PMID 21531810)
- PMID range 11,208,838 to 23,524,406; median 20,979,469 (≈ 2011
  publication date; well within CIViCmine v40's coverage window so
  recency is not the gap driver)

**Coverage table**:

| Source TSV | Rows | Unique PMIDs | PMIDs covered (our 80) | Targets covered STRICT (PMID + gene/drug match) | Targets covered LOOSE (PMID only) |
|---|---:|---:|---:|---:|---:|
| sentences.tsv (high-confidence) | 380,482 | 123,983 | 11 / 80 (13.8%) | **10 / 162 (6.2%)** | 24 / 162 (14.8%) |
| unfiltered.tsv (predictprob > 0.5) | 968,476 | 207,863 | 39 / 80 (50.0%) | **41 / 162 (25.3%)** | 79 / 162 (48.8%) |

**Coverage by entity-pair family on the unfiltered TSV**:

| Family | n_targets | Strict covered | % |
|---|---:|---:|---:|
| gene_drug | 154 | 39 | 25.3% |
| variant_disease | 8 | 2 | 25.0% |

**Reasons for non-coverage (STRICT, against unfiltered)**:
- PMID not in CIViCmine: 83 / 121 missing targets
- PMID in CIViCmine but no entity-pair match: 38 / 121 missing targets
  (Entity-name normalisation differences and partial relation
  extractions account for this slice.)

**Verdict — Case C (<40%)**:
Even against the FP-tolerant unfiltered output with predictprob > 0.5,
only 25.3% of our 162 evaluable targets carry a strict
(PMID + gene/drug) CIViCmine match; the high-confidence sentences
file covers only 6.2%. CIViCmine systematically does not extract
most of the (gene, drug, variant) tuples that CIViC v2024+
curators have anchored. This is the headline finding for Phase 2A
on the Case C path: **CIViCmine cannot evaluate the bulk of current
CIViC oncology assertions**, regardless of model quality. A
quantitative baseline can still be reported on the 41 covered
targets, but the framing flips from "external system comparison"
to "external-system coverage limitation".

**Recommended Phase 2A scope (Case C)**:
- Report CIViCmine non-coverage as the headline (Section S13.1).
- Compute KB argmax accuracy on the 41 strictly-covered targets and
  contrast with PB-T2 means on the same 41 targets (apples-to-apples).
- Run the three "no coverage" sensitivities (exclude / NEG / random)
  on the full 162-target denominator.
- Estimated effort: ≈1 day.

Coverage probe artefact: `civicmine/coverage_probe.json`.

---

## Probe 2 — LLM API

| Check | Outcome |
|---|---|
| `OPENAI_API_KEY` in env | **NOT SET** |
| `ANTHROPIC_API_KEY` | NOT SET |
| `GEMINI_API_KEY` | NOT SET |
| `OPENROUTER_API_KEY` | NOT SET |
| `HF_TOKEN` | NOT SET |
| `openai` Python package | not installed (`ModuleNotFoundError`) |

**Verdict — BLOCKED**. No API key present in environment; no test
call attempted.

**Cost estimate** for Phase 2B at gpt-4o-mini pricing
($0.15/1M input tok, $0.60/1M output tok):

| Condition | Input tok / call | Output tok / call | 162 calls | Cost |
|---|---:|---:|---|---:|
| zero-shot | ~450 | ~50 | 72,900 in + 8,100 out | $0.016 |
| 6-shot | ~1,950 | ~50 | 315,900 in + 8,100 out | $0.052 |
| 6-shot + rationale | ~1,950 | ~200 | 315,900 in + 32,400 out | $0.066 |
| **3-condition sweep** | --- | --- | --- | **≈ $0.13 + retries** |

Recommended budget cap: $1.00 (8× headroom).

**Action required from user**:
1. Provide `OPENAI_API_KEY` (or alternative provider; Anthropic
   `claude-3-haiku-20240307` and Gemini `gemini-1.5-flash` are
   suitable substitutes at similar price/quality).
2. Confirm budget cap.
3. Once key is exported, `pip install --user openai`; re-run Probe 2's
   test call before proceeding to Phase 2B.

Phase 2B does **not** block Phase 2A or 2C.

---

## Probe 3 — Matched-compute control feasibility

| Check | Outcome |
|---|---|
| Slurm available | yes (24.11.5) |
| Partitions visible | `workq*` only (no separate `gpu` partition) |
| Phase B used | `#SBATCH -p workq` with `--gpus=1` (per `phase_b_PL.sbatch`) |
| Partition load | 21 idle nodes, ~860 allocated, plenty of headroom |
| Training runner | `fine_tuning_experiments/phase_b/trainer/run_experiment.py` |
| `max_updates` configurable | **yes, via per-config YAML key `scientific_trainer.max_updates`** |
| Code patch needed | **none** — change the YAML, no Python change |
| Reference config | `fine_tuning_experiments/phase_b/configs/PB_PB_FT_T1F_s01.yaml` (exact PB × T1F × seed 1; max_updates: 2048; T2 stages disabled; all hyperparams locked) |
| Seed reproducibility | yes; seeds 1-20 already used in Phase B |

**Wall-clock estimate**:
- Phase B reported ~60 GPU-hours / 310 runs ≈ 11.6 min average.
  T1F is single-stage (2,048 updates); T2 is two-stage (4,096 total).
  A T1F-2048 run is roughly 8-10 min; doubling to T1F-4096 should be
  ~16-20 min per run.
- 20 seeds × 20 min = ~6.7 GPU-hours total compute.
- Submitted as `--array=1-20%6` (matching Phase B concurrency), wall
  ≈ 4 batches × 20 min = **~80 minutes**. Higher concurrency on the
  21 idle nodes could compress to ~40 minutes.

**Verdict — READY**. Resources available, no code patch required.
Awaiting `COMMITMENT.md` sign-off on the pre-committed decision rule
before the 20 sbatch submissions launch.

---

## Probe 4 — Phase 1 bundle as alternative data source

| Check | Outcome |
|---|---|
| Bundle path in `phase_c_robustness` branch | `knowledge_grounded_evidence_audit/analysis/phase_c_robustness/outputs/phase_b_per_target_predictions_v1.tar.gz` |
| Sha256 recorded | `a15824ac…d8d1d689` (committed `.sha256` file) |
| Sha256 recomputed from the committed blob | `a15824acea979858e448dda2d5cda1df42151f801a35f9151b81c919d8d1d689` |
| Match | **yes** |
| Size | 5,253,553 bytes |
| Contents | 190 jsonl + README.md + LICENSE + manifest.csv |

The bundle covers everything Phase 2D needs for cross-system
comparison: per-target predictions, hit_A_sv, softmax, pmass_B_sv,
the 162-vs-155 split, and per-cell aggregation.

**Verdict — READY**.

---

## Recommended sub-phase ordering

| Step | Status | Action |
|---|---|---|
| 2A CIViCmine baseline (Case C) | ready | Start once user approves probe report |
| 2B LLM baseline | blocked | Await `OPENAI_API_KEY` + budget approval |
| 2C matched-compute control | ready | Write `COMMITMENT.md` first; user signs off on decision rule; then submit 20-seed sbatch array. Can launch in parallel with 2A |
| 2D shared reporting | last | Consolidates 2A + 2B + 2C |

**Suggested parallelism**:
1. **Now**: user approves 2A scope (Case C framing) and 2C
   `COMMITMENT.md` decision rule.
2. **Parallel kickoff**: 2A analysis (1 day CPU) and 2C sbatch
   submission (~80 min wall-clock + 0.5 day analysis afterwards).
3. **When key arrives**: 2B (≈ 0.5 day end-to-end including API calls
   and 3-condition sweep).
4. **Last**: 2D shared reporting (≈ 0.5 day).

**Total projected wall-clock** (assuming user gives go-ahead on 2A
and 2C now, key for 2B within 1-2 days): **3–4 working days**.

---

HALTED. No sub-phase work has begun. Awaiting user sign-off on:
1. **2A Case C scope** (CIViCmine non-coverage as headline).
2. **2C COMMITMENT.md decision rule** (to be drafted next, but
   I want explicit sign-off on the 0.05-threshold three-way decision
   before any sbatch submission).
3. **2B API key + budget cap** (or instruction to skip 2B for this
   revision).
