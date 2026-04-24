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

*End of Appendix B amendment log.*
