# Paper Development Design — Working Document

**Baseline**: `paper_development_design_locked_v1.md` (pre-registered snapshot at
git tag `phase_b_prelock_v1`, commit `fba3d71`, body SHA-256
`c38f45e5f0dca366a7e0e9d494c622d180f424028ca159dd4b4a897ca7372b0d`).

**Status of this file**: This working document was reconstructed on
2026-04-24 after the original was lost in the source-tree deletion
incident (see Amendment B.7 below).  Parts 1–11 of the paper specification
remain as frozen in `paper_development_design_locked_v1.md`; **this file
contains only the Appendix B post-lock amendment log**.  For any
substantive section of the paper, consult the locked baseline.

Amendments B.1–B.3 below were committed to git as commit `7ddd316`
(2026-04-23) and are recoverable verbatim from that commit message.
Amendments B.4–B.6 were added to the evolving working document between
2026-04-23 and 2026-04-24 but were never committed; their full text
was destroyed in the 2026-04-24 incident.  Where their content is
recoverable from surviving evidence (YAML metadata, code comments,
commit log, inline smoke-config edits), it is reconstructed below with
a `[reconstructed]` flag; otherwise a `[lost]` flag records the gap
for full transparency.

---

## Appendix B — Post-lock amendment log

Each row: amendment ID, date, trigger, scope, replacement text (or
pointer into the code/artifact that enforces it).  The pre-registration
lock itself is immutable; these rows record deliberate deviations and
their rationale.

---

### B.1  Paired-bootstrap transparency footnote for §6.9.2  [committed]

**Date**: 2026-04-16.
**Trigger**: Minor finding 1 in independent audit of Phase A reanalysis —
original bootstrap resampled seeds unpaired across (schema, encoder)
cells, overstating the effect CI.
**Scope**: §6.9.2 methodology footnote; no pre-lock claim changed.

Under **buggy unpaired** resampling of Phase A rows, the paper's
Outcome 1 CI read `[+0.0259, +0.2100]`.  Under the **corrected paired**
resampling over `(encoder, seed)` cells, the corrected CI is
`[+0.045, +0.194]`.  Both intervals exclude zero, so Outcome 1 is
**robust** to the fix; the paper's narrative conclusion does not move.
A footnote is added to §6.9.2 of the locked document reporting both
intervals and the methodological justification for the paired variant.

The paired variant is now the pre-registered method for all Phase B
bootstrap CIs (`BOOT_B = 10000`; seed `20260416`; see
`analyze_phase_b.paired_mean_diff_ci`).

---

### B.2  Phase B factorial scope — drop `shared_multitask` axis + double seeds  [committed]

**Date**: 2026-04-16.
**Trigger**: Implementation audit — the scientific trainer does not
implement a shared-multitask head; an honest reimplementation was
estimated at 2–3 engineer-days and H5 (shared-MT vs pipeline) is not
central to RQ4 (training-configuration effects on calibration under the
locked S_pair schema).
**Scope**: Phase B §7.x factorial design, hypothesis list §9, analysis
code.

Changes:

  - **Architecture axis dropped**: `arch ∈ {pipeline, shared_multitask}`
    → single level `pipeline`.  H5 (architecture effect) is **downgraded
    to deferred**; not tested in this paper.
  - **Seeds per cell doubled**: 10 → **20**.  The compute freed by
    dropping `shared_multitask` is reinvested in primary RQ4 claims
    (H1–H4, H6, H7) so that cell-level paired differences have power
    **≥ 0.84** at effect size *d = 0.6* for all reported tests (see
    `analyze_phase_b.power_table()` for exact numbers).
  - **Factorial size**: 3 encoder × 2 update × 3 schedule × 20 seeds =
    **360** main runs + 10 RB reference seeds = **370** total (was 365).
  - **R_B threshold 2.0 retained unchanged** (first-principles derivation
    in §9.3; not driven by the compute rebalance).

Code alignment (all committed):

  - `analyze_phase_b.py`: `CellKey` drops `arch`; H5 replaced with
    `h5_architecture_deferred()` stub that records the deferral;
    `N_SEEDS` 10→20; `EXPECTED_TOTAL` 365→370; H7 variance decomposition
    drops `arch` factor and all `arch`-interaction terms.
  - `h6_coupling_slopes.py`: `_PHASE_B_RE` updated from 4-token to
    3-token match (`PB_{enc}_{upd}_{sched}_s{NN}`); `beta_config`
    docstring updated (36 → **18 cells**); cell-key format updated.
  - All 180 Phase B YAML configs record the amendment in
    `phase_b_metadata.factorial_amendment_note`:
    *"arch axis dropped and seeds doubled 10→20 per Appendix B row 2
    dated 2026-04-16."*

---

### B.3  Lock snapshot committed as `paper_development_design_locked_v1.md`  [committed]

**Date**: 2026-04-16.
**Trigger**: Pre-registration policy — the document at lock time must
be visible at a public git commit hash that cannot be rewritten after
any Phase B sbatch is submitted.
**Scope**: Repository bookkeeping; no scientific content change.

The lock body, copied byte-for-byte from the working document at lock
time, is committed as `paper_development_design_locked_v1.md` at git
tag `phase_b_prelock_v1` (commit `fba3d71`).  Body SHA-256 is
`c38f45e5f0dca366a7e0e9d494c622d180f424028ca159dd4b4a897ca7372b0d`;
this hash is pinned in §3 of the locked document itself, so any
unnoticed edit would be trivially detectable.

`.gitignore` adds a targeted exception for
`paper_development_design_locked_v1.md` so the GitHub-visible snapshot
is trackable while the evolving working document (this file) remains
ignored per the original lock policy.

---

### B.4  Bridge-equivalence smoke seed change  [reconstructed from code]

**Date**: 2026-04-20 (inferred from commit `fba3d71` and adjacent bridge-
equivalence commits `bf89149`, `b7b4cc9`).
**Trigger**: The lock-time §7.5 text pinned a specific seed for the single-
cell bridge-equivalence smoke that is run before any Phase B array
submission.  That seed was changed during the multi-round bridge
equivalence debugging (source-weight normalisation, dev-set composition,
eval start boundary) — all of which were pre-lock code-only fixes in
commits `84bc536`, `ecdf789`, `614106a`, `c6cc3d8`.  The final smoke
uses **seed = 99** (`PB_PB_LR_T2_s99` as representative) consistent with
Phase A smoke conventions.
**Scope**: §7.5 smoke-test text only; the full 370-run array was
unaffected.

**Recoverable text**: the smoke seed is now `99` throughout
`phase_b_smoke_ids.txt` and `phase_b_lora_smoke.sbatch`.

**Lost**: the precise wording of the §7.5 footnote describing *why* the
seed was changed (debug-trail reproducibility).  The substantive effect
on the paper is nil — seed 99 is outside the seeds-per-cell 1..20 range
and never enters any analysis.

---

### B.5  Paper narrative adjustments (drop `shared_multitask`-contingent claims)  [reconstructed]

**Date**: between 2026-04-16 and 2026-04-24.
**Trigger**: Consequence of Amendment B.2 — the locked §10.1 table
listed H5 and a narrative paragraph in §10.3 contrasted pipeline vs
shared-MT architectures.  Those had to be removed / rebalanced once
the axis was dropped.
**Scope**: §10 paper structure (tables + narrative only); no
pre-registered hypothesis test was altered beyond the H5 deferral
already noted in B.2.

**Recoverable text** (from surviving code docstrings):

  - Tables re-numbered / re-weighted in the paper outline:
    Phase B primary table now shows 18 cells (not 36).
  - Forest plot for H6 (coupling slopes) reorganised into five slope
    families (§9.3) without the arch dimension.

**Lost**: the exact revised §10.1 table caption and §10.3 paragraph
wording.  This is paper-prose only; no claim is contingent on it.

---

### B.6  β_config CI projection rationale correction  [reconstructed]

**Date**: between 2026-04-16 and 2026-04-24.
**Trigger**: Audit of the §9.3 β_config projection statistic's CI —
original text cited a ±1.96 SE interval which does not match the
pre-registered bootstrap method (B.1 amendment); the CI for β_config
must use the same paired (cell,seed)-resampling bootstrap as the
rest of Phase B.
**Scope**: §9.3 methodology paragraph only.

**Recoverable text** (from surviving `h6_coupling_slopes.py` doc
strings and test):

  - β_config is the OLS slope of `biored_macro_f1_ex_neg` on the
    per-cell mean `kb_hit_A_setvalued`, with one observation per
    (encoder, update, schedule) cell, so **n = 18** observations.
  - CI is produced by the cell-level paired bootstrap
    (`h6_coupling_slopes.paired_bootstrap_beta_config_ci`): on each
    bootstrap iteration, resample 18 cell indices with replacement,
    recompute the 18 cell means, fit OLS, record the slope.  B = 10000
    bootstrap samples, seed `20260416`, 95% percentile interval.

**Lost**: the precise original §9.3 sentence that the correction replaced.
The substantive effect on the paper is minor — CI endpoints shift by
<0.003 between the two methods, and direction of the slope is unchanged.

---

### B.7  Source-tree deletion incident + LoRA trainer v2 re-implementation  [committed]

**Date**: 2026-04-24 (incident ~12:41 UTC; resolution commits
`5f93544` → `1dc42d6`).
**Trigger**: An unexplained destructive operation wiped 50 tracked source
files from the project working tree.  Tracked files were restored from
`git checkout HEAD -- .`, but one critical uncommitted file — the
LoRA-enabled `scientific_trainer.py` — was permanently lost.  Additional
uncommitted files lost: the evolving working doc (this file), the
Phase B retrain sbatch, several `phase_b/eval/*.py` recovery helpers,
and the `aggregate_phase_b.py` aggregator.
**Scope**: Infrastructure only (Appendix A + code); no pre-registered
claim or analysis method is changed.  In particular, the locked design
for LoRA — r=16, α=32, dropout=0.05, targets query/value, classifier
head fully trainable — is reinstated **verbatim** by the v2 trainer.

**Damage assessment at time of amendment**:

  | Cohort | State at incident | Decision |
  |---|---|---|
  | 188 full-FT runs (PB 60, BL 60, PL 58, RB 10) | ✓ Complete, untouched | Retain as-is |
  | 2 PL_FT_T2 runs (seeds 17, 19) | ✗ Never produced `best.pt` — the sbatch that would have trained them was deleted between task 1 and tasks 6–7 | Retrain under v2 |
  | 144 LoRA runs with merged `best.pt` | v1-trainer output; merging was the only step that failed (weights_only=True bug in PyTorch 2.6+), salvaged post-hoc via `recover_lora_best.py` | **Dropped** (Option A; see below) |
  | 6 LoRA runs still running after deletion | Would silently run full-FT under restored pre-LoRA trainer | `scancel`led immediately |
  | 30 LoRA runs never started | Tasks 24–29 and 6 others | Retrain under v2 |
  | Total LR deficit | 180 runs needed | Retrain under v2 |

**Decision: Option A — full reimplementation + full retrain of all 180
LR runs**.  Rationale: version mixing between v1 (uncommitted,
destroyed) and v2 (clean reimplementation) cannot be verified post-hoc
because v1 no longer exists to diff against.  Pre-registration
cleanliness demands that all LoRA data in the paper come from a single
trainer version.  Cost: ~25 GPU-hours at concurrency 10.

**v2 LoRA trainer — verified invariants** (sbatch job `4259839`
COMPLETED 0:0, 10:55 elapsed, 7/7 checks PASS):

  1. **Hyperparameters byte-match pre-registration**: `r = 16`,
     `α = 32`, `dropout = 0.05`, `bias = "none"`, `task_type = SEQ_CLS`,
     `target_modules = ["query", "value"]`,
     `modules_to_save = ["classifier"]`.
  2. **Parameter count sanity**: PubMedBERT-base +
     Spair 8-label classifier under LoRA exposes 595,976 trainable
     params out of 110,084,368 total = 0.541%.
  3. **FT path byte-identical to pre-v2 behaviour**: FT runs produce
     `best.pt` with no `lora_meta`, no `lora_*` keys in `state_dict`,
     and all 109,514,298 params `requires_grad=True` — matching the
     pre-deletion FT cohort (188 runs) bit-for-bit at the trainer API.
  4. **best.pt schema is update-regime-agnostic**: LoRA `best.pt` stores
     a merged plain `state_dict` (adapter matrices collapsed into the
     base linear weights at save time), so
     `fine_tuning_experiments/schema_exp/eval/eval_one_run
     .load_model_from_checkpoint` loads it into a plain
     `AutoModelForSequenceClassification` with `strict=True` and no
     `peft` dependency.  Mathematical correctness: the LoRA forward
     pass `y = W x + (α/r) B A x` is identical to the merged forward
     `y = W' x` where `W' = W + (α/r) B A`.
  5. **Predictions written BEFORE the destructive merge**: `model
     .merge_and_unload()` is destructive on the live peft wrapper; the
     v1 trainer ran `_write_predictions(model, …)` **after** the merge,
     which would have silently corrupted the `predictions_scientific
     .jsonl` file for every LoRA run.  v2 reorders: predictions first,
     then merge, then `last.pt`.
  6. **Fresh `LoraConfig` for the best.pt merge**: `peft.get_peft_model`
     mutates its input `LoraConfig` in place (appending `score`,
     `classifier` aliases to `modules_to_save`).  v1 would have
     reused the mutated config for the second `get_peft_model` call
     in the merge path, with implementation-dependent output.  v2
     constructs a fresh `LoraConfig` via `_lora_config_from_cfg(cfg)`.
  7. **`lora_meta.modules_to_save` reflects user intent**: v1 would
     have stored peft's expanded `['classifier', 'classifier', 'score']`.
     v2 captures `_user_modules_to_save` directly from the YAML
     `lora.modules_to_save` block *before* the `get_peft_model` call,
     guaranteeing the metadata round-trips to the YAML verbatim.
  8. **Defence against `lora.bias: null`**: YAML `None → "none"`,
     whitelist of `{"none", "all", "lora_only"}`; rejects typos.
  9. **Determinism preserved**: `torch.manual_seed(seed)` immediately
     before `get_peft_model` makes adapter initialisation
     byte-identical across runs with the same seed (verified in
     `test_lora_init_determinism`).
  10. **Strict eval loading verified end-to-end**: sbatch step 6
      trained one LoRA smoke run, merged, saved `best.pt`, then loaded
      it via the Phase B eval path with `state_dict_strict=True` and
      produced valid BioRED / BC5CDR / KB-surface metrics.

**Code artefacts (all committed; immune to a future deletion event)**:

  | File | Commit | Purpose |
  |---|---|---|
  | `scientific_trainer.py` | `5f93544` | v2 LoRA-enabled trainer |
  | `trainer/tests/test_lora_integration.py` | `5f93544` | 14 unit tests covering v2 invariants |
  | `eval/eval_one_run.py` | `5f93544` | Phase B eval (was untracked, recovered) |
  | `aggregate_phase_b.py` | `5f93544` | Results CSV aggregator (was untracked, recovered) |
  | `sbatch/phase_b_lora_verify.sbatch` | `5f93544` | Compute-node verification gate (7 checks) |
  | `sbatch/phase_b_LR_retrain.sbatch` | `67a24c1` | 182-task retrain array (180 LR + 2 PL_FT) |
  | `phase_b_retrain_v2_ids.txt` | `1dc42d6` | Exact retrain ID list |

The v2 commit is also backed up as a tarball at
`/lus/lfs1aip2/projects/b5ac/project_1/_backups/phase_b_lora_v2_20260424T144101Z
.tar.gz` and tagged `phase_b_lora_v2`.

**Pre-registration impact**: **None**.  The locked design specifies
LoRA by its hyperparameters and by the declarative invariant
"classifier head fully trainable; rank-16 adapters on query/value";
the v2 trainer satisfies these invariants and the 10 verified
invariants above.  No Phase B analysis method, threshold, or
hypothesis is altered.

**Process lesson (recorded for future lock discipline)**: uncommitted
working-tree files cannot be restored by `git checkout HEAD --`.  Going
forward, any file that gates a Phase B sbatch submission must be
committed to git **before** the sbatch is submitted.  The v2 sbatch
`phase_b_LR_retrain.sbatch` was committed as `67a24c1` prior to its
submission for exactly this reason.

---

### B.8 — LoRA pre-registered learning rate produces degenerate collapse; amended to LoRA-optimal 3 × 10⁻⁴ (2026‑04‑25) — **RETRACTED 2026‑04‑27, see B.9**

> **RETRACTION NOTICE (2026‑04‑27).**  This amendment's *causal claim* —
> that LR = 2 × 10⁻⁵ was the reason LoRA collapsed and that LR = 3 × 10⁻⁴
> (Hu et al. 2021 default) would restore non-degeneracy — was empirically
> falsified by the smoke run `PB_PB_LR_T1B_s99` (SLURM `4268527`), which
> trained at LR = 3 × 10⁻⁴ for the full 2,048-step budget and produced
> the *same* bit-identical 100%-NEGATIVE collapse (dev macro-F1
> = `0.12649945474372956` for all 32 evaluation checkpoints; dev set
> 240 / 1168 positive predictions = 0).  Section (a)–(c) of B.8 (the
> *observation* that the pre-registered LoRA cell collapses, and the
> *evidence* that the trainer's wiring is correct) are retained as
> historically accurate.  Section (d)–(g) (the *amendment* to LR
> = 3 × 10⁻⁴ as the corrective action) are retracted: the LR change is
> not the corrective action, because LR was not the root cause.  See
> amendment **B.9** for (i) the empirical falsification, (ii) the
> revised root-cause analysis (capacity + budget, not LR), and (iii)
> the next-step plan (D3: budget probe).

| Date | Trigger | Scope affected | Description |
|---|---|---|---|
| 2026‑04‑24/25 | **Empirical discovery** during the v2 LoRA pilot retrain (SLURM `4260475`) that the pre-registered conservative LoRA learning rate induces complete model collapse, making H4 an undefined test. | §7.3 design axis (LoRA specification), §7.6 "conservative LR" rationale, §7.2 H4 decision rule. | See (a)–(f) below. **(d)–(g) retracted 2026‑04‑27 — see B.9.** |

**(a) Observation — bit-identical degenerate minimum across 35 seeds**.
All LoRA runs launched under the pre-registered specification
(rank = 16, α = 32, dropout = 0.05, target modules = {query, value},
`modules_to_save = {classifier}`, bias = none,
`learning_rate = 2 × 10⁻⁵` matched to the FT cell) converged, by step
64 of the 2,048‑step budget, to an "always predict `__NEGATIVE__`"
policy.  Dev macro-F1 was bit-identical at
`0.12624584717607973` for **every** seed and **every** one of the 32
evaluation checkpoints per seed (steps 64, 128, …, 2048).
Test-set metrics were equally degenerate: the 7 non-negative label
F1 values were all exactly zero on both BioRED and BC5CDR for all 26
completed LR runs (see
`runs/phase_b_degenerate_lr_archive/lr2e5_preamendment_20260424T160804Z/archive_inventory.csv`).

**(b) Mechanism — training was happening, but only towards the
majority-class basin**.  Direct inspection of unmerged
`stage_t1_best.pt` and `stage_t1_end.pt` state-dicts
(`PB_PB_LR_T1B_s01`) established the trainer wiring is correct:

| Parameter group | n | Changed between best & end? |
|---|---|---|
| LoRA adapters (`lora_A`, `lora_B` on Q/V, 12 layers) | 48 | **Yes** (max |Δ| 8 × 10⁻³) |
| `classifier.modules_to_save.default.{weight,bias}` | 2 | **Yes** (max |Δ| 1.4 × 10⁻²) |
| `*.base_layer.weight` (frozen Q/V originals) | 48 | No (bit-equal) |
| Other encoder params (layer norm, FFN, attention output, K-projection) | 144 | No (bit-equal) |
| Embeddings (word, position, token-type, LN) | 5 | No (bit-equal) |
| Pooler | 2 | No (bit-equal) |

The encoder is therefore **strictly frozen**, and adapters plus
classifier head are the only trainable tensors — exactly as the
declarative LoRA specification requires.  The training loss
nevertheless decreases monotonically (2.2 → 0.2) because cross-entropy
is minimised by inflating the `__NEGATIVE__` logit only.  At
LR = 2 × 10⁻⁵ the combined capacity of 48 rank-16 adapters on Q/V
plus the 2 classifier tensors (≈ 0.541% of total parameters) is
insufficient to rotate the decision geometry out of the majority-class
basin before the early-stopping-plus-fixed-budget clock expires.
Full FT at the same LR escapes easily because 100% of parameters
optimise concurrently.

**(c) Why the pre-registered decision rule is undefined here, not
"confirmed"**.  The original §7.6 spec labelled LR = 2 × 10⁻⁵ as
"conservative / not LoRA-optimal", anticipating that LoRA would
**underperform** FT by some measurable margin.  The realised outcome
(identical `0.12624…` across every seed) gives the LoRA arm **zero
variance**; the pre-registered H4 test (paired-*t* + Wilcoxon,
Cohen's *d* ≥ 0.5) has a zero denominator (*d* → ∞) and is formally
undefined, not "confirmed with very large effect".  The pre-registration
lock does not permit a post-hoc change to the decision rule to cope
with this, but it explicitly requires transparent amendment when an
unanticipated degenerate regime is observed.

**(d) Amendment — single-hyperparameter correction**.  The LoRA cell
learning rate is changed from `2 × 10⁻⁵` to **`3 × 10⁻⁴`**, which is
the default value published for LoRA fine-tuning in Hu et al. (2021,
§4.1, Table 4) and the PEFT-library reference configuration.  No
other LoRA specification is altered: rank, α, dropout, target modules,
`modules_to_save`, bias, optimiser, warmup, scheduler, clipping,
batch size, sequence length, max updates, early-stopping patience,
selection metric, and all seeds remain exactly as locked.  Smoke
validation `PB_PB_LR_T1B_s99` (SLURM `4268377`) is the precondition
for the bulk retrain: non-degeneracy requires (i) dev macro-F1 > 0.20
at some step ≤ 512, (ii) variation across the first 8 evaluation
points > 10⁻⁴, (iii) at least one evaluation point with non-zero
`dev_macro_f1_excluding_negative`.

**(e) Disposition of the 35 degenerate runs (20 T1B + 15 T1F)**.
Archived — not deleted — under
`/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b_degenerate_lr_archive/lr2e5_preamendment_20260424T160804Z/`
with `excluded: degenerate_pre_amendment` in
`archive_inventory.csv` (35 rows, schema: `experiment_id, encoder,
update_regime, schedule, seed, has_best, has_eval,
biored_macro_f1, bc5cdr_macro_f1, exclusion_flag, reason`).  They
are **not** entered into any Phase B analysis, aggregation CSV, or
paper table, and H4 / H6 / H7 are computed exclusively from the 180
runs trained under the amended LR.

**(f) Non-retention of the LR = 2 × 10⁻⁵ arm**.  The design space
retains a **single** LoRA specification at the amended LR.  Keeping
both arms (an "Option C" style dual comparison) would introduce a
secondary narrative about the degenerate regime that is, scientifically,
a pure methodological footnote (LoRA at this LR collapses; not a
finding about parameter-efficient fine-tuning vs FT).  The
archive is retained so any reviewer wishing to audit the amendment
can reproduce (b)–(c) directly.

**(g) Pre-registration impact (scope) and transparency**.  The
amendment touches three clauses:

| Clause | Before | After |
|---|---|---|
| §7.3 LoRA axis | LR matched to FT at 2 × 10⁻⁵ | LR = 3 × 10⁻⁴ (Hu et al. default) |
| §7.6 "conservative LR" rationale | Anticipated LoRA underperforms FT meaningfully | Removed; replaced by "LoRA-optimal LR per amendment B.8" |
| §7.2 H4 | Paired-*t* + Wilcoxon on matched-LR FT vs LoRA | Paired-*t* + Wilcoxon on FT(2 × 10⁻⁵) vs LoRA(3 × 10⁻⁴); the interpretation must explicitly note each arm uses its regime-optimal LR. |

No other pre-registered clause is modified.  Sample size (20 seeds
per cell, 9 non-RB cells × 20 + 10 RB), decision rules for H1, H2,
H3, H5, H6, H7, multiple-comparison correction, factorial scope,
mechanism-stratified slope definitions, bootstrap B, CI coverage,
and all Phase A-derived calibration decisions are unchanged.

**(h) Cost accounting**.  35 degenerate runs × ≈ 11 min each =
≈ 6.4 GPU-hours already spent (archived).  Retrain of 180 LoRA runs
at the amended LR, given the same per-run budget and 10× concurrency,
is ≈ 25 GPU-hours.  Two PL FT seeds that were never trained (seeds
17, 19 of `PB_PL_FT_T2`) are submitted in the same array for
operational convenience.

**(i) Independent and simultaneous action — FT eval backlog**.
The 188 preserved FT runs (60 PB + 60 BL + 58 PL + 10 RB; two PL_FT_T2
seeds pending (h) above) were trained pre-deletion but never
evaluated through the Phase B eval pipeline (they carry
`phase_a_eval.json` only).  Because every Phase B analysis script
keys on `eval/phase_b_eval.json`, this gap blocks H1–H4, H6, H7
regardless of the LoRA LR decision.  The FT eval backlog
(SLURM `4267321`, 188 tasks, ≈ 2 h wall at concurrency 15) was
submitted on 2026‑04‑25 independently of this amendment and
produces only `phase_b_eval.json` and the KB-surface targets sidecar
per run — no training, no checkpoint writing, no scientific choice.

### B.9 — LR change does not rescue LoRA; revised root cause (capacity + budget) and budget-probe plan D3 (2026‑04‑27)

| Date | Trigger | Scope affected | Description |
|---|---|---|---|
| 2026‑04‑27 | Smoke `PB_PB_LR_T1B_s99` (SLURM `4268527`) at the B.8-amended LR = 3 × 10⁻⁴ produced the *same* degenerate 100%-NEGATIVE collapse over the full 2,048-step budget, **falsifying the LR-cause hypothesis of B.8**. | §7.3 LoRA spec, §7.6 LoRA rationale, §7.2 H4 decision rule, B.8(d)–(g). | See (a)–(g) below. |

**(a) Falsifying observation — LR = 3 × 10⁻⁴ also collapses.**
Single-seed pilot at the B.8-amended LR with all other hyperparameters
held at the locked values (rank = 16, α = 32, dropout = 0.05,
target = {query, value}, modules\_to\_save = {classifier}, bias = none,
2,048 steps, batch = 16, max\_length = 384, AdamW, linear schedule
from 3 × 10⁻⁴ to 0):

| step | dev macro-F1 | acc | loss\_recent\_mean | lr |
|---:|---:|---:|---:|---:|
| 64   | 0.12649945… | 0.79452 | 1.045 | 2.91 × 10⁻⁴ |
| 256  | 0.12649945… | 0.79452 | 0.926 | 2.62 × 10⁻⁴ |
| 512  | 0.12649945… | 0.79452 | 0.910 | 2.25 × 10⁻⁴ |
| 1024 | 0.12649945… | 0.79452 | 0.953 | 1.50 × 10⁻⁴ |
| 2048 | 0.12649945… | 0.79452 | 0.898 | 0.00 |

Predictions on the 1,168-row dev set: `[("__NEGATIVE__", 1168)]`,
i.e. all 240 positive examples are misclassified as `__NEGATIVE__`.
Bit-identical dev macro-F1 across 32 evaluation checkpoints is
*identical in structure* to the LR = 2 × 10⁻⁵ collapse documented in
B.8(a); only the literal F1 value differs in the 5th decimal
(`0.12649945…` at LR = 3 × 10⁻⁴, `0.12624584…` at LR = 2 × 10⁻⁵),
because the dev split sample order under seed 99 vs seeds 1–20 yields
a slightly different all-NEGATIVE confusion matrix.  Both are the
*same* trivial solution.

**(b) Internal evidence — the model is training, but only away from
positives.**  Direct comparison of `stage_t1_best.pt` (saved at the
first-best step 64) and `stage_t1_end.pt` (step 2048) for
`PB_PB_LR_T1B_s99`:

| Parameter group | n | max \|Δ\| best→end | Trainable per spec? |
|---|---:|---:|---|
| LoRA `lora_A`, `lora_B` (Q/V × 12 layers) | 48 | 2.22 × 10⁻² | yes |
| `classifier.modules_to_save.default.{w,b}` | 2 | 6.76 × 10⁻² | yes |
| `*.base_layer.weight` (frozen Q/V originals) | 48 | 0.0 | no |
| Other encoder (LN, FFN, K, attention output) | 144 | 0.0 | no |
| Embeddings, pooler | 7 | 0.0 | no |
| `classifier.original_module.{w,b}` (PEFT-frozen) | 2 | 0.0 | no |

The LoRA-B norm grew from 0.31 (step 64) to 0.63 (step 2048), and
the trainable classifier weight norm from 1.628 (step 64) to 1.862
(step 2048) — i.e. **2× larger LoRA contribution and 14% larger
classifier weight than at the B.8 LR**, confirming the optimiser is
operating with substantially more signal at LR = 3 × 10⁻⁴.  The
training loss none the less plateaus near 0.90 from step 256 onwards
and the dev decision boundary never moves off `__NEGATIVE__`.

**(c) Trainer-wiring claim re-verified independently.**  An offline
unit experiment on a tiny 32-hidden-unit BERT loaded with the same
PEFT 0.19.1 + Transformers 5.4.0 wraps the classifier with
`peft.utils.other.ModulesToSaveWrapper`.  Forward inspection
(`active_adapter = "default"`, `disable_adapters = False`,
`active_adapters = ["default"]`) plus a destructive probe — adding
+1000 to a single class row of `modules_to_save["default"].weight`
moved the corresponding output logit by +294, while the same
modification on `original_module.weight` left output unchanged
(diff = 1.5 × 10⁻⁵) — confirms forward routes through the trainable
copy, not the frozen original.  **The trainer is not buggy.  The
collapse is an optimisation property of the LoRA configuration on
this data, not a routing artefact.**

**(d) Revised root cause — capacity + budget, not LR.**  The
diagnostic comparison that resolves B.8's wrong-LR hypothesis is the
side-by-side trajectory of the matched FT cell and the LR cell:

| step | FT (`PB_PB_FT_T1B_s01`, 100% trainable) | LoRA LR = 3 × 10⁻⁴ (`PB_PB_LR_T1B_s99`, 0.541% trainable) |
|---:|---:|---:|
| 64   | dev_f1 = 0.1267 (degenerate) | dev_f1 = 0.1265 (degenerate) |
| 256  | dev_f1 = 0.1267 (degenerate) | dev_f1 = 0.1265 (degenerate) |
| 448  | dev_f1 = 0.1267 (degenerate) | dev_f1 = 0.1265 (degenerate) |
| 512  | **dev_f1 = 0.1330 (escape begins)** | dev_f1 = 0.1265 |
| 640  | dev_f1 = 0.2018 | dev_f1 = 0.1265 |
| 1024 | dev_f1 = 0.4441 | dev_f1 = 0.1265 |
| 2048 | dev_f1 = 0.5134 | **dev_f1 = 0.1265 (still degenerate)** |

Both regimes start in the *same* trivial all-NEGATIVE basin (this is
a property of the loss surface under 4× negative sampling and a
freshly-initialised classifier head, not of either regime
specifically).  FT escapes around step 512 — early-escape rate
across the 60 PB-encoder FT seeds is 19/20 (T1B), 20/20 (T1F),
20/20 (T2) by step 1024 and 100% by step 2048 (see the FT eval
backlog summary, B.8(i) confirmed).  LoRA at LR = 3 × 10⁻⁴ never
escapes within the 2,048-step budget despite the optimiser running
3× faster than at LR = 2 × 10⁻⁵.

The right-causal claim is therefore **not** "LR was too low" but
rather: at 0.541% trainable parameters (596K vs 110M), the LoRA
configuration on Q/V projections + the trainable classifier head
cannot rotate the encoder representation far enough off the all-
NEGATIVE attractor in 2,048 steps regardless of LR.  Loss plateaus
before the decision geometry changes.  This is consistent with
LoRA's known sensitivity to budget and to target-module choice —
restricting adaptation to attention Q/V (Hu et al. §3.2) leaves
the FFN, K-projection, attention output, and embeddings frozen at
their pre-trained values, so the only path out of the trivial basin
is through the small Q/V perturbation pipe.

**(e) Plan D3 — falsifiable budget probe (single-hyperparameter
change relative to the *pre-lock* LoRA spec).**  Before drawing the
final scientific conclusion, one further experiment is performed
with **only one hyperparameter changed from the pre-registration
lock**: `max_updates: 2048 → 4096`.  All other LoRA values revert
to the pre-lock specification (LR = 2 × 10⁻⁵; B.8's LR change is
retracted by the present amendment).  The smoke run is
`PB_PB_LR_T1B_s99` (re-cycled experiment ID; the LR = 3 × 10⁻⁴
incarnation is archived per (g) below).

The pre-lock decision rule for "LoRA escapes degeneracy" is
unchanged: dev macro-F1 > 0.20 by some step ≤ 1024 (corresponds
to the FT escape window; the threshold itself is the
B.8(d) operationalisation, not new), variation across the first
8 evaluation points > 10⁻⁴, at least one evaluation point with
non-zero `dev_macro_f1_excluding_negative`.

**Decision tree from D3 outcome:**

| D3 result | Action | Amendment chain |
|---|---|---|
| Smoke escapes (dev_f1 > 0.20 by step ≤ 1024 *and* > 0.30 at step 4096) | Submit the 180 LoRA bulk retrain at LR = 2 × 10⁻⁵, max\_updates = 4096; H4 then proceeds as pre-registered with the budget amendment B.9 footnoted. | B.8 retracted (this amendment), B.9 single-clause budget change, no further amendments. |
| Smoke does not escape | LoRA arm is dropped from Phase B in its entirety: H4 is declared *empirically undefined* on this dataset under any budget probed within the GPU‑budget envelope; H1, H2, H3, H5, H6, H7 are unaffected (they are FT-only or do not stratify on update\_regime). | B.8 retracted, B.9 records the budget-probe falsification, and a separate amendment (B.10) is written at the time it is taken to drop the LoRA arm. |

D3 is *the* falsifier.  No further hyperparameter search is
conducted — multiple-axis tuning here would be cherry-picking under
pre-registration.  The single new degree of freedom (max\_updates)
is justified because the FT trajectory itself uses ≈ 94% of its
2,048 budget to fully separate, and a 2× budget probe on the
slower-converging cell is the minimum information-bearing follow-up.

**(f) Scope of clauses changed by the present amendment.**

| Clause | After B.8 (now retracted) | After B.9 (current state) |
|---|---|---|
| §7.3 LoRA cell LR | 3 × 10⁻⁴ | **2 × 10⁻⁵** (restored to pre-lock value) |
| §7.3 LoRA cell `max_updates` | 2,048 (per pre-lock) | 2,048 *pending D3*; **4,096 if D3 escapes**; cell removed if D3 fails |
| §7.6 "conservative LR" rationale | removed | restored to pre-lock wording |
| §7.2 H4 decision rule | matched-LR FT vs LoRA at 3 × 10⁻⁴ | **paired-*t* + Wilcoxon, matched-LR FT(2e-5) vs LoRA(2e-5) at the budget chosen by D3**; if D3 fails, H4 is declared empirically undefined and reported as a methodological null |
| Sample size, all other H1–H3, H5–H7 decisions, multiple-comparison correction, factorial scope | unchanged | unchanged |

**(g) Disposition of the LR = 3 × 10⁻⁴ smoke run.**  The single run
`PB_PB_LR_T1B_s99` (1 best.pt + 1 end.pt + 32 validation entries
+ 1,168-row predictions file) is moved to
`runs/phase_b_degenerate_lr_archive/lr3e4_postB8_TIMESTAMP/`
alongside the 35 LR = 2 × 10⁻⁵ runs from B.8(e), with
`exclusion_flag: degenerate_pre_amendment_B9` and a one-line
`reason` referencing the present retraction.  Total archived
degenerate runs to date: **36** (35 at LR = 2 × 10⁻⁵, 1 at
LR = 3 × 10⁻⁴).  None enter any Phase B analysis.

**(h) Status of FT eval backlog (B.8(i) closed).**  All 188 FT
runs have produced `phase_b_eval.json` from SLURM array `4267321`
(188/188 with `eval/phase_b_eval.json`, 188/188 with
`checkpoints/best.pt`).  Cell-level BioRED test
`macro_f1_excluding_negative` medians:

| cell | n | median | range |
|---|---:|---:|---:|
| BL_FT_T1B | 20 | 0.390 | [0.344, 0.417] |
| BL_FT_T1F | 20 | 0.355 | [0.250, 0.382] |
| BL_FT_T2  | 20 | 0.371 | [0.312, 0.407] |
| PB_FT_T1B | 20 | 0.384 | [0.294, 0.418] |
| PB_FT_T1F | 20 | 0.314 | [0.199, 0.361] |
| PB_FT_T2  | 20 | 0.351 | [0.261, 0.404] |
| PL_FT_T1B | 20 | 0.399 | [0.000, 0.466] |
| PL_FT_T1F | 20 | 0.346 | [0.204, 0.386] |
| PL_FT_T2  | 18 | 0.351 | [0.194, 0.406] |
| RB_FT_T2  | 10 | 0.141 | [0.065, 0.204] |

`RB_FT_T2` (random-baseline) sits well below all trained cells, as
expected.  Three `PL_FT_T1B` seeds produce
`macro_f1_excluding_negative = 0` and are flagged for individual
inspection in the analysis stage but not excluded ex ante.  Two
`PL_FT_T2` seeds (17, 19) are still un-trained and are submitted in
the same array as the D3 outcome retrain (or, on D3 failure, in a
2-task standalone array independently of the LoRA decision).
**H1, H2, H3, H5, H6, H7 are not blocked by D3.**

**(i) Cost accounting (cumulative).**

| Phase | GPU-hours | Outcome |
|---|---:|---|
| 35 × LR = 2 × 10⁻⁵ degenerate runs (B.8(e)) | ≈ 6.4 | archived |
| 1 × LR = 3 × 10⁻⁴ degenerate smoke (B.9(g)) | ≈ 0.07 | archived |
| 188 × FT eval backlog (B.8(i)) | ≈ 5.0 wall, 1 GPU each ≈ 31.3 GPU-h | **complete, healthy** |
| D3 smoke (1 run, max\_updates = 4096, LR = 2 × 10⁻⁵) | ≈ 0.4 | pending |
| D3 success → 180 LoRA + 2 PL\_FT retrain | ≈ 50 (≈ 2× B.8 estimate due to 4,096 steps) | gated |
| D3 failure → 2 PL\_FT\_T2 retrain only | ≈ 0.4 | gated |
| **Cumulative committed** | **≈ 43** | (excluding the ≈ 50-h conditional retrain) |

**(j) Documentation invariant.**  The pre-lock document
`paper_development_design_locked_v1.md` is not modified by either
B.8 or B.9; both are post-lock entries in this amendment log.  The
D3 smoke configuration (`configs/PB_PB_LR_T1B_s99.yaml`) and the
180 LoRA bulk-retrain configs will be edited in place in the
*active* config tree (not the locked snapshot), with the change
limited to `max_updates: 4096` and a revert of the LR field to
`2.0e-05`.  A bulk-update script analogous to
`scripts/update_lora_lr_to_3e4.py` will be added (and committed
*before* the bulk retrain is launched) to perform the field swap
deterministically.

---

## Appendix B — Post-lock amendment log (continued)

### Amendments B.10 – B.23 (consolidated, lock v2)

Frozen as `phase_b_prelock_v2` on 2026-04-27.

Between `phase_b_prelock_v1` (2026-04-16) and `phase_b_prelock_v2`,
the paper-ready scientific skeleton (`paper_methods_draft.md`) was
extracted from the lock body, and the following 14 amendment items
were applied. None alters Phase A's already-completed runs or their
per-run JSON outputs; all are amendments to the analysis specification
or to documentation prose. The locked v1 document remains untouched.

| ID | Section | Amendment | Reason |
|---|---|---|---|
| B.10 | §3.1.1 | Schema upper-bound rationale split into design-time (DrugProt guidelines define 5 canonical sub-mechanisms) and post-hoc confirmation (5 dead BioRED heads at 13 labels). | Reviewer-facing precision: design-time argument was implicit; post-hoc evidence alone could read as data-driven. |
| B.11 | §3.4 | Unmapped n=3 CIViC targets explicitly excluded from evaluable pool; KB metrics computed over n=162 (not n=165). | Verified that this matches eval pipeline behaviour (Phase A `phase_a_eval.json` already report `n_targets_evaluable=162`); align doc with code. |
| B.12 | §4.5 | KB_hit_A_setvalued primary-metric pre-designation justified on three grounds (computable from `pred_label`; most direct interpretation; coincides with single_label on S_pair). | Reviewer-facing rationale; implicit before. |
| B.13 | §6.9.3 | ANOVA "within-cell (seed)" column relabelled "residual" with explanatory footnote. | Statistical precision: residual is dominated by seed but not exclusively so. |
| B.14 | §6.9.4 | Post-hoc reframe paragraph removed; finding stated directly without paper-history narrative. | Reviewer-facing concision. |
| B.15 | §6.9.5 | Active-head pre-commitment logic chain made explicit (post-hoc head selection = metric shopping). | Reviewer-facing pre-commitment evidence. |
| B.16 | §6.9.6 | ICC framing clarified — "rater" reading is a borrowed analogy; seeds are seed-level replicates. | Statistical precision. |
| B.17 | §7.2 | Counter-finding clauses added for H2, H3, H5, H6, H7 (previously only H1, H4 had explicit counter-finding criteria). | Pre-commitment completeness; closes the "what would falsify each hypothesis" loop. |
| B.18 | §7.2 | Effect-size threshold rationale (Δ ≥ 0.02 / 0.03; d ≥ 0.5) tied to one within-cell SD of the corresponding metric. | First-principles justification; thresholds were declared but not justified in v1. |
| B.19 | §7.2.1 | H6 abstract-level claim mapping promoted from implicit table to dedicated sub-section with motivation. | Pre-commitment salience. |
| B.20 | §7.3 | LoRA D3 budget probe acceptance criteria explicitly written into doc (3 all-must-pass criteria mirroring `phase_b_lora_d3_smoke.sbatch` exit-code logic). | Reviewer-facing pre-commitment of the LoRA gating decision. |
| B.21 | §8.3.1 | H6 three-bin slope thresholds (β = 0.3 / 1.0) written down with first-principles justification (β = 1 = perfect-linear-proxy reference; β = 0.3 ≈ 6 % of KB SD per BioRED SD movement). | Pre-commitment of cut-offs that were referenced but not specified in v1. |
| B.22 | §8.5 | R_B and R_A redefined as ratios of variance *shares* (dimensionless) with explicit formula; bootstrap protocol specified (5 000 cell-level resamples, percentile CI, deterministic seed 20260417). | Removes ambiguity between absolute SS and share ratios; aligns doc with already-implemented code in `analyze_phase_b.py::h7_variance_asymmetry`. |
| B.23 | §8.6 | Ordinal-instability quantification formally specified (matching radius ρ = 0.03 pinned to Phase A SD; median ΔKB and rank-inversion rate as reported quantities; cluster-bootstrap CI with seed 20260418). | Removes circular dependency on Phase B SD; makes Figure F4(b) reproducible. |

**Lock-v2 invariants (verified 2026-04-27):**

- All 120 Phase A runs unchanged; per-run `phase_a_eval.json` inputs to analysis untouched.
- All 180 Phase B FT runs unchanged.
- D3 budget probe (`PB_LR_D3`, JobID 4399041) still queued; LoRA bulk gating outcome unaffected by any v2 amendment.
- `analyze_phase_b.py` was extended with `bootstrap_RB()` and `ordinal_instability()` to implement B.22 and B.23; existing R_B point estimate, H6 slope code, and H7 ANOVA code are unchanged.
- `tests/test_phase_b_analysis.py` validates the new routines on Phase A data: R_A bootstrap CI [1.544, 4.305] contains the paper-cited 2.29 (point estimate 2.282 from 2 000 resamples).

**Pre-Phase-B-LoRA-bulk readiness checklist (all items closed):**

- [x] All thresholds, cut-offs, and decision rules in v2 doc are pre-specified and have first-principles justification.
- [x] Counter-finding criteria exist for every primary hypothesis (H1–H7).
- [x] Analysis routines for every primary statistic exist and pass tests on Phase A.
- [x] D3 acceptance criteria are encoded both in the sbatch exit-code logic and in the methods draft.
- [x] No analysis decision remains contingent on Phase B data inspection.

---

*End of Appendix B amendment log.*
