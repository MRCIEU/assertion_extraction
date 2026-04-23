<!--
================================================================================
FROZEN SNAPSHOT — DO NOT EDIT

This file is a read-only copy of paper_development_design.md at the moment of
the pre-registration lock.

  Lock tag:        phase_b_prelock_v1
  Lock commit:    fba3d7149d6ae0420468b6c5071f4b5d7be00c3f
  Document SHA:   c38f45e5f0dca366a7e0e9d494c622d180f424028ca159dd4b4a897ca7372b0d
                  (sha256sum of the body below this HTML comment block)
  Tarball SHA:    12b65c4be97faa64ee21c14a56bee3b2eb52b5129b5c9403648b6602f05af043
                  (phase_b_prelock_v1.tar.gz)
  Frozen on:      2026-04-16

For the evolving working document (with post-lock amendments in Appendix B),
see paper_development_design.md. All post-lock changes are logged there; no
changes to this frozen snapshot are permitted.
================================================================================
-->

# Paper Development Design
## Heterogeneous Supervision and Evaluation Validity for Cancer Assertion Extraction

**Target venue:** *Bioinformatics* (Oxford Academic) — Original Research
**Alternative venues:** *Briefings in Bioinformatics*, *NAR Genomics and Bioinformatics*
**Document status:** Living research record; sole source of truth for methodology, pre-registration, and analysis plan.

**How to read this document.** Parts 1–4 define the paper (story, research questions, data, schemas). Parts 5–6 specify the trainer and evaluation framework that are common to both experimental phases. Part 7 is Phase A (schema selection); Part 8 is Phase B (training-configuration factorial). Part 9 is the statistical analysis plan. Part 10 is paper structure (sections, figures, tables). Part 11 is the pre-committed list of known weaknesses and responses. Appendix A is the code-and-artifact layout; Appendix B is the post-lock amendment log (empty at lock).

---

## Part 1 — Paper Position and Core Narrative

### 1.1 The one-paragraph story

Cancer-focused relation/assertion extraction differs from general biomedical RE in two underappreciated ways: (1) no single public dataset provides sufficient span-supervised coverage of oncology assertion semantics, forcing practitioners to compose heterogeneous supervision from general RE corpora, KB priors, and weak mining; and (2) standard benchmark metrics (BioRED macro-F1, BC5CDR macro-F1) are routinely used as proxies for downstream clinical-informatics utility despite the absence of any empirical validation of that proxy relationship. This paper addresses both gaps. We define a schema-aware multi-stage training framework that explicitly separates gold-span, weak-prior, and unlabeled-adaptation supervision; train and evaluate a systematic factorial of encoder–architecture–schedule configurations against held-out external benchmarks; and evaluate the same checkpoints in a knowledge-grounded audit pipeline anchored to CIViC-curated evidence. The central empirical contribution is a quantitative audit of the benchmark-to-downstream proxy relationship. Across 3 schemas × 4 encoders × 10 seeds (Phase A pilot, 120 runs) and an expanded 3 × 2 × 2 × 3 configuration factorial under the selected schema (Phase B, 360 runs + RB reference), we observe (i) a pronounced *variance asymmetry* — schema choice explains 60.4 % of BioRED macro-F1 ex-NEG variance but only 19.1 % of `KB_hit_A` variance in Phase A, a 3.2× disparity on the primary downstream metric (and a 40× disparity against the out-of-domain BC5CDR benchmark), which Phase B tests against configuration variance; (ii) *mechanism-dependent positive coupling* — the benchmark-to-KB slope is strong on the schema dimension, near-zero on seed noise, and (to be estimated) on the configuration dimension, with no single "benchmark-to-KB" slope representing the system; and (iii) consequent *ordinal instability* — two configurations whose benchmark F1 values differ by the typical within-cell SD of ≈ 0.03 have KB_hit_A rankings (within-cell SD ≈ 0.16) that cannot be reliably predicted from their benchmark ordering. Benchmark rank is therefore a low-fidelity signal for the KB-surface ordering of otherwise-comparable configurations, with direct implications for how extraction models should be selected for clinical-informatics deployment.

### 1.2 Research questions → experimental mapping

| RQ | Question | Primary experiment | Primary output |
|----|----------|-------------------|----------------|
| **RQ1** (task setting) | How should cancer-focused assertion extraction be operationalised given the absence of a single sufficient public dataset or canonical schema? | Data inventory + schema design | Schema definitions (S_flat, S_pair, S_mech); T1–T4 supervision table; gap audit |
| **RQ2** (training & generalisation) | Which training configurations and model choices best support generalisation from internal development to held-out external evaluation? | Phase B factorial (encoder × architecture × update regime × schedule × seed) under the S_pair schema selected by Phase A, plus statistical tests | Main-results table with effect sizes and FDR-corrected significance |
| **RQ3** (downstream sensitivity) | In a knowledge-grounded oncology audit, how do model family and audit formulation jointly affect surfacing yield? | CIViC-anchored KB audit over 165 targets, applied to every Phase A + Phase B checkpoint | Per-family surfacing table; KB-hit rates stratified by schema (Phase A) and by configuration (Phase B) |
| **RQ4** (evaluation validity) | How strongly, and in which direction, do benchmark metrics predict downstream KB-audit surfacing across schema and configuration variation? | Variance decomposition of benchmark F1 vs `KB_hit_A` under both schema variance (Phase A) and configuration variance (Phase B); joint seed-level coupling slope | Variance-share table (H7); seed-level BioRED↔KB scatter with coupling slope and 95 % CI (H6) |

### 1.3 Contributions (for Introduction bullet list)

1. **Schema and packaging framework** — A principled mapping from heterogeneous public corpora (BioRED, DrugProt, BC5CDR, CIViC, CIViCmine, CancerMine) to three oncology-oriented relation schemas (flat, entity-pair, mechanism) at different granularity levels, with explicit supervision-level routing (T1 gold, T2 MeSH-C04 oncology projection, T3 weak/KB, T4 unlabeled) and verifiable integrity checks.
2. **Systematic training comparison** — Multi-seed evaluation of encoder (PubMedBERT-base/large, BioLinkBERT, RoBERTa reference), architecture (pipeline vs shared-multitask), update regime (full fine-tune vs LoRA), and schedule (single-corpus, multi-corpus flat, staged T1→T2) against official BioRED and BC5CDR test benchmarks, with formal paired significance tests.
3. **KB-grounded audit evaluation** — A reproducible downstream evaluation framework using 165 CIViC-anchored targets with three correctness-aware metrics (argmax hit, expected-label probability mass, abstention–recall AUC), under set-valued and single-label projection modes.
4. **Evaluation-validity audit** — A quantitative characterisation of how benchmark macro-F1 and `KB_hit_A` respond to the same design choices. Phase A (schema × encoder) and Phase B (configuration factorial) together decompose variance contributions, report a **family of mechanism-stratified coupling slopes** (within-cell seed noise, between-schema, between-encoder, between-configuration, and pooled-between-cell — each with its own estimand and CI, §9.4), and quantify ordinal rank instability. The headline finding is a pronounced variance asymmetry (benchmark-directed design levers move KB surfacing much less than they move the benchmark itself), paired with mechanism-dependent positive coupling that is strong on the schema mechanism but weaker on the configuration mechanism — together implying that benchmark rank is an unreliable guide to KB-surface ordering when the design changes being compared happen within the configuration regime typical of modern biomedical-NLP practice.

---

## Part 2 — Data Layer

### 2.1 Source corpora

| Corpus | Role | Counts after leakage fix |
|--------|------|------------------------|
| BioRED | Gold span RE (entity-typed; multi-relation) | 500 train/dev docs, 100 test docs |
| DrugProt | Gold span RE (chemical-protein mechanisms) | 4,250 train/dev docs |
| BC5CDR | Gold span RE (chemical-disease binary) | 1,000 train/val docs, 500 test docs |
| CIViC | KB audit anchor (curated oncology assertions) | 165 evidence-supported targets |
| CIViCmine | Weak sentence mining (T3) | ≥ 415k weak rows |
| CancerMine | Weak prior labels (T3) | Included with CIViCmine |
| Oncology PubMed abstracts | T4 unlabeled domain adaptation | 9,463 abstracts |

### 2.2 Train–test leakage remediation (completed 2026-04-15)

The original JSONL packages (`t1_biored.jsonl`, `t1_bc5cdr.jsonl`, `t1_supervised_backbone_merged.jsonl`, `t2_biored_projected.jsonl`, `t2_bc5cdr_cancer_slice.jsonl`) bundled official test-split documents (16.7 % BioRED, 33.3 % BC5CDR) together with the training splits. Because the trainer samples a random dev fraction without filtering by `source_split`, retaining these test documents in the training shards would cause test leakage. The leakage-remediation pass produces `*_train_only.jsonl` shards for T1 and T2 with all test-split documents removed at the source-pipeline level; all Phase A and Phase B experiments in this paper train on the leakage-corrected shards only.

The leakage was fixed by `data_pipeline/01_filter_t1_t2_leakage.py`, producing `*_trn.jsonl` packages that contain no test-split documents, verified by `data_pipeline/05_validate_integrity.py`. All training in this paper uses the `*_trn.jsonl` shards exclusively.

### 2.3 Oncology projection (T2, MeSH C04)

T2 is the oncology-facing subset used as the second training stage. The operational criterion is: documents in T1 whose PubMed record carries at least one MeSH descriptor under the Neoplasms tree (C04.*) are retained.

```
For each PMID in T1:
    fetch MeSH terms via NCBI E-utilities (efetch, db=pubmed, rettype=xml)
    include document in T2 if any descriptor falls under C04.*
```

Resulting counts: 733 T2 documents (BioRED 94, BC5CDR 114, DrugProt 525), 6,969 gold relations. All T2 documents are a subset of the leakage-fixed T1 shards (verified).

Methods statement: *"Documents in T1 with at least one MeSH descriptor under the Neoplasms (C04) hierarchy, as indexed by NCBI PubMed, were designated as the T2 oncology-facing subset."*

Two explicit limitations that must be reported:

- **Limitation 1 — Document-level, not relation-level.** All relations within a matching document enter T2 regardless of whether each individual relation's entities are cancer-related. T2 therefore provides oncology *context* supervision rather than relation-level oncology gold.
- **Limitation 2 — DrugProt T2 is mechanism-in-oncology-context.** DrugProt T2 relations are chemical-protein mechanism labels occurring in abstracts with oncology MeSH terms, not oncology-specific assertion labels.

Both limitations are stated in the Methods and motivate the paper's argument (RQ1) that no single resource suffices.

### 2.4 Weak and unlabeled stages (T3, T4)

T3 (weak, auxiliary only) comprises CIViC semantic priors, CIViCmine weak sentences, and CancerMine priors. It is not used as gold supervision at any point in the main experiments; the corresponding `lambda_auxiliary` and `lambda_distill` hyperparameters are zero in every Phase A/B cell.

T4 (unlabeled) is a set of 9,463 lung-oncology PubMed abstracts reserved for optional domain-adaptation runs. The main factorial disables T4 (`T4_mode: none`).

### 2.5 Frozen shard manifests

The canonical shard files are stored under `/lus/lfs1aip2/projects/b5ac/project_1/training_data_generation/data/processed/`. Each schema has its own pre-projected shard set. The full inventory is in `training_data_generation/data/processed/t1_manifest.json`. SHA-256 checksums of the shards are written to the manifest and re-verified at the start of every training run (see §5).

---

## Part 3 — Schema Design

### 3.1 Three schema candidates

The paper compares three oncology-oriented relation schemas at three granularity levels. All three are run in Phase A (schema selection pilot, §6). Phase A's schema-selection rule (§6.6) and Outcome 1 (§6.9.2) designate **S_pair** as the sole Phase B schema; S_flat and S_mech are not carried into Phase B but their Phase A numbers remain the schema-variance evidence for the RQ4 variance-asymmetry finding (H7).

**S_flat — 4 labels.** Family-level discrimination: `ASSOCIATION_GENERAL`, `DRUG_DISEASE`, `DRUG_GENE_REGULATION`, `__NEGATIVE__`. A BioRED variant–disease association maps to `ASSOCIATION_GENERAL`; a BC5CDR drug-disease CID maps to `DRUG_DISEASE`; DrugProt chemical-protein mechanisms all collapse to `DRUG_GENE_REGULATION`.

**S_pair — 8 labels.** Entity-pair type discrimination: `GENE_DISEASE`, `VARIANT_DISEASE`, `DRUG_DISEASE`, `DRUG_GENE_REGULATION`, `GENE_GENE_ASSOC`, `DRUG_VARIANT_ASSOC`, `ASSOCIATION_GENERAL` (catch-all), `__NEGATIVE__`. A BioRED variant–disease association maps to `VARIANT_DISEASE`; a gene–gene association maps to `GENE_GENE_ASSOC`. `ASSOCIATION_GENERAL` retains the BioRED associations whose entity-pair type is not one of the specialised heads (notably DRUG-DRUG or VARIANT-VARIANT, which are excluded from `spair_legal_endpoints`).

**S_mech — 13 labels.** S_pair plus five DRUG_GENE_REGULATION sub-mechanisms: `DGR_INHIBIT`, `DGR_ACTIVATE`, `DGR_METABOLIC`, `DGR_REGULATE`, `DGR_STRUCTURAL`. Sub-mechanism grouping follows the DrugProt annotation guidelines (Miranda-Escalada et al., 2021).

### 3.2 Label-space derivation and regression test

The classifier head for a given schema is sized by `derive_label_space(shard_paths, pair_type_filter)` (`fine_tuning_experiments/phase_b/trainer/scientific_data.py`). The function scans gold `mapped_label` values across the active T1+T2 shards, sorts non-negative labels alphabetically, and appends `__NEGATIVE__` last. **Labels are enumerated before applying `pair_type_filter`**: a label is retained in the classifier as long as some gold relation somewhere carries that label, even if every gold instance of that label has entity endpoints outside the filter. Without this behaviour, S_pair would silently drop `ASSOCIATION_GENERAL` to a 7-class head (since all S_pair AG rows have illegal endpoints under `spair_legal_endpoints`), diverging from the schema definition. The regression test in `fine_tuning_experiments/phase_b/trainer/tests/test_label_space.py` asserts the expected counts (4 / 8 / 13) and presence of `ASSOCIATION_GENERAL` across all three schemas.

### 3.3 Pair-type filters

Each schema has a legal entity-pair type set. A relation is eligible for training/evaluation only if its `(head_label, tail_label)` tuple is in the filter. Negative sampling draws pairs from the same filter, guaranteeing distributional parity with positive relations.

| Filter | Legal pairs |
|--------|-------------|
| `sflat_legal_endpoints` | GENE↔DISEASE, GENE↔DRUG, DRUG↔DISEASE, GENE-GENE, VARIANT↔DISEASE, VARIANT↔GENE, DRUG↔VARIANT, VARIANT-VARIANT |
| `spair_legal_endpoints` | same as S_flat **minus** VARIANT↔GENE and VARIANT-VARIANT |
| `smech_legal_endpoints` | same as S_pair |

### 3.4 Schema-expected-label mapping for the KB audit

Each CIViC audit target has a pairing family (gene-drug, variant-disease, etc.) and a heuristic gold label drawn from the CIViC evidence. For each (family, gold, schema, projection_mode) combination, the `schema_expected_label_set()` function (`fine_tuning_experiments/schema_exp/eval/schema_expected_label.py`, mapping rationale in `schema_expected_label_mapping_rationale.md`) returns the set of schema-vocabulary labels that count as a correct prediction. The mapping projects CIViC ground truth *up* into each schema's own vocabulary, so each schema is evaluated in its own frame and comparisons across schemas are fair.

| CIViC family × heuristic gold | S_flat | S_pair | S_mech (set_valued) | S_mech (single_label) |
|---|---|---|---|---|
| gene_drug × DGR (n=90) | `{DGR}` | `{DGR}` | `{DGR, DGR_INHIBIT, DGR_ACTIVATE, DGR_METABOLIC, DGR_REGULATE, DGR_STRUCTURAL}` | `{DGR}` |
| gene_drug × AG (n=64) | `{DGR}` | `{DGR}` | same | `{DGR}` |
| variant_disease × AG (n=8) | `{AG}` | `{VARIANT_DISEASE}` | `{VARIANT_DISEASE}` | `{VARIANT_DISEASE}` |
| variant_disease × VARIANT_GENE (n=3) | unmapped | unmapped | unmapped | unmapped |

Two projection modes are reported for S_mech, where a fine-grained schema has multiple heads covering a single CIViC family:

- **set_valued** — any of the covering heads counts as a hit (permissive; favourable to fine-grained schemas).
- **single_label** — only the canonical catch-all head counts (strict; unfavourable to fine-grained schemas).

For S_flat and S_pair, each family maps to exactly one label under either mode, so the two projections are numerically identical on Phase B's active schema (S_pair). The projection-mode distinction therefore only matters for Phase A's S_mech evidence, where it exposes S_mech's single-label collapse (§6.9.1).

---

## Part 4 — Evaluation Framework

### 4.1 BioRED test

Official BioRED test split (100 documents) with relations projected into each schema's label vocabulary. Metrics:

- **BioRED macro-F1 (ex-NEG)** — the paper's headline benchmark signal. Macro-F1 over non-negative labels only, averaged over the labels that appear in at least one gold instance of that schema's test set (`sklearn.metrics.f1_score` with `average="macro"` and default `labels=` behaviour on the union of predictions and gold). The in-training dev F1 is computed with the same convention so that early-stopping and external evaluation are directly comparable.
- **Per-head F1** — reported in supplementary; exposes dead heads and low-support heads.

### 4.2 BC5CDR test

Official BC5CDR test split (500 documents), binary classification of DRUG–DISEASE pairs. Metric: **BC5CDR DRUG_DISEASE F1** (`sklearn.metrics.f1_score` with `pos_label="DRUG_DISEASE"`). The binary random baseline is 0.50; models improving over 0.50 by Δ > 0.05 exhibit measurable DD discrimination.

### 4.3 CIViC KB-surface audit

Over 165 CIViC-anchored targets, each with a PubMed abstract and two pre-identified entity spans, the model produces a full softmax (and pre-softmax logits) over the schema vocabulary. This is a **relation-classification** setting — spans are provided, so the audit measures relation discrimination only, not entity detection. The audit artifacts are `kb_surface_pairs.jsonl` (165 targets) and `kb_surface_targets.jsonl` (per-target prediction output, one row per (target × run)).

### 4.4 Three correctness-aware KB metrics

Each metric is reported under `set_valued` and `single_label` projection modes where applicable (see §3.4). For S_flat and S_pair the two modes coincide.

**Method A — argmax hit (primary).**
$$\text{KB\_hit\_A}(\text{schema}, \text{mode}) = \text{mean}_{\text{evaluable targets}}\; \mathbb{1}\!\bigl[\arg\max_L P_\text{schema}(L) \in \mathcal{E}(\text{target}, \text{schema}, \text{mode})\bigr]$$
where $\mathcal{E}$ is `schema_expected_label_set`. Computable from `pred_label` only (no logits required).

**Method B — expected-label probability mass (sensitivity, calibration-robust).**
$$\text{KB\_pmass\_B}(\text{schema}, \text{mode}) = \text{mean}_{\text{evaluable targets}}\; \sum_{L \in \mathcal{E}} P_\text{schema}(L).$$
Requires the full softmax vector per target. Treats argmax ties and near-ties symmetrically; reports the probability mass assigned to correct labels regardless of whether that mass wins the argmax.

**Method C — AUC of abstention–recall curve (sensitivity, threshold-free surfacing).** On a 21-point grid of abstention thresholds $\tau \in \{0.00, 0.05, \dots, 1.00\}$:
$$\text{reject\_rate}(\tau) = \frac{|\{\text{targets : } P(\_\_\text{NEGATIVE}\_\_) > \tau\}|}{n_\text{eval}}, \quad \text{precision\_kept}(\tau) = \text{KB\_hit\_A on non-rejected subset}.$$
$\text{KB\_auc\_C}$ is the trapezoidal integral of precision_kept over reject_rate (points with n_kept = 0 treated as NaN and dropped from integration; `scipy.integrate.trapezoid`). The 21-point grid is pre-specified; finer/coarser grids and higher-order integration (Simpson, Gauss–Legendre) are explicitly not used.

### 4.5 Primary metric designation

To prevent post-hoc metric selection:

- **Primary** for H6 (coupling slope) and H7 (variance asymmetry): `KB_hit_A` (Method A, set_valued). Simplest to interpret; identical to single_label on Phase B's single active schema (S_pair); directly comparable to Phase A.
- **Sensitivity** (reported alongside, not headlining): `KB_pmass_B` (Method B, set_valued) and `KB_auc_C` (Method C, set_valued).
- **Projection-mode sensitivity** (only for Phase A S_mech): single_label variants of A/B/C, Supplementary only.
- **Deprecated diagnostic**: `KB_surface_mean = mean(1 − P(__NEGATIVE__))` is emitted by the eval pipeline for continuity with earlier sensitivity checks and is retained in per-run JSONs for reference. It is never reported as a primary or sensitivity metric in this paper because it is correctness-blind: a model that spreads probability mass uniformly over wrong non-negative heads scores high without producing correct predictions. `KB_hit_A_setvalued` supersedes it in every reporting role.

### 4.6 Eval pipeline v1.0 (pinned)

The evaluation pipeline is `fine_tuning_experiments/schema_exp/eval/eval_one_run.py` and its dependencies (`schema_expected_label.py`, `aggregate_phase_a.py`, `prepare_eval_inputs.py`). Any change to this code requires re-running evaluation on every checkpoint that will appear in the paper; in practice, this code is frozen at the start of Phase A and any amendment is logged in the appendix.

Per-run output: `<run_dir>/eval/phase_a_eval.json` with three blocks (`biored_test`, `bc5cdr_test`, `kb_surface`) and all correctness-aware metrics described above.

---

## Part 5 — Trainer Specification

### 5.1 Stage structure

Phase A and Phase B both use a two-stage schedule by default (`schedule: T1_to_T2` in the config). Stage T1 trains on the full leakage-fixed supervised backbone (BioRED + DrugProt + BC5CDR); stage T2 is a staged continuation on the MeSH-C04 oncology subset. Single-stage schedules are available as Phase B axis levels:

- `T1_biored_only` — T1 on BioRED alone.
- `T1_flat` — T1 only, no T2.
- `T1_to_T2` — the staged schedule (default).

At stage transition, the optimiser and scheduler are **reinitialised**. The schedule is a HuggingFace linear-with-warmup from `lr = 2e-5` decaying to 0 over the stage's `max_updates = 2048`; warmup = 0.

### 5.2 Optimizer, clipping, precision

- **Optimizer**: AdamW (HuggingFace defaults).
- **Gradient clipping**: `max_grad_norm = 1.0`.
- **Mixed precision**: off (pure FP32); PB-base checkpoints are 438 MB FP32.

### 5.3 Early stopping and checkpoint selection

- **Evaluation frequency**: every 64 optimizer updates.
- **Selection metric**: macro-F1 on the in-training dev split (built from a seeded 12% holdout of each stage's gold positive pool, with negatives materialised at `negative_ratio` to match training distribution).
- **Early stopping**: patience of 10 evaluations (= 640 steps); triggered only once `steps ≥ early_stopping_min_updates = 256`. Dev evaluation runs from the first 64-step boundary and is not gated by `min_updates`.
- **Checkpoint selection**: per stage, the checkpoint with the highest dev macro-F1 is kept; best.pt is the overall-best across stages. Stage-best and stage-end checkpoints are saved explicitly.

The `dev_macro_f1` recorded in `metrics/loss_history.jsonl`, `metrics/validation_history.json`, and the `best_checkpoint_meta.dev_macro_f1` field is always this **in-training dev metric**, computed over the **full schema vocabulary** (S_pair = 8 labels including dead heads; macro with `sklearn.metrics.f1_score(average="macro")`). It is never BioRED or BC5CDR external test F1, and it is never the active-head macro-F1 of §6.9.5. External-benchmark F1 values and active-head-macro variants are computed only post-hoc by the §4.6 eval pipeline and live in `eval/phase_a_eval.json`; they do not feed checkpoint selection.

### 5.4 Negative sampling

Negatives are online-sampled per batch, not pre-materialised. For each positive drawn at batch build time:

1. Sample up to `max_negatives_per_sample = 64` within-document entity pairs that satisfy `pair_type_filter`, excluding the positive's own head–tail pair and its reverse.
2. Keep `negative_ratio = 4.0` negatives per positive.
3. Mix positives and negatives, shuffle, truncate to `batch_size = 4`.

The per-document sampling is seeded by `seed + 101` (T1) and `seed + 202` (T2) so each training run produces a reproducible negative stream.

### 5.5 Source weighting

Each training row carries a `source_weight` drawn from `cfg["source_weights"]`, applied as a per-sample multiplier in the cross-entropy loss. Phase A configs have `biored: 1.0, drugprot: 1.0, bc5cdr: 1.0`, which we interpret as uniform weighting; the `inverse_freq_family_softmax` function described in the original config is available in `scientific_data.py` but disabled because it destabilised T1 convergence in preliminary runs relative to the simple uniform setting.

### 5.6 RNG determinism

All four RNG streams (`random`, `numpy`, `torch` CPU, `torch` CUDA) are seeded from `cfg["seed"]` at the start of training. The seed is also passed to the dev-split sampler, the data shuffle generator, and the online negative sampler. A full RNG-state snapshot is captured at every checkpoint save (stage-best, stage-end, `best.pt`, `last.pt`) and stored in the checkpoint payload so that any future analysis can reproduce the exact batch stream after the checkpoint.

### 5.7 Code implementation

The trainer lives entirely in version-controlled Python source under `fine_tuning_experiments/phase_b/trainer/`:

- `scientific_trainer.py` — training loop, stage dispatch, checkpoint save, metrics JSON output.
- `scientific_data.py` — shard loading, text encoding (`"{head} [ENT] {tail} [SEP] {doc}"[:8000]`), dev split, label-space derivation, online negative sampling, source weighting.
- `minimal_trainer.py` — a NotImplementedError shim retained for backwards compatibility with Phase A configs that set `minimal_trainer.enabled: true`.

Entry point: `python3.11 -m fine_tuning_experiments.train.run_experiment --experiment-id ... --config-path ... --run-root ...`. Both Phase A and Phase B dispatch through this entry point; the only thing that changes between phases is the config YAML.

### 5.8 Per-run artifacts

Every run directory contains:

```
<run_dir>/
  run_manifest.json           (config echo + git_commit + config_sha256 + trainer_source)
  training.log                (one-line-per-event plain text)
  checkpoints/
    stage_t1_best.pt          (includes model, label2id, rng_state, best_checkpoint_meta)
    stage_t1_end.pt
    stage_t2_best.pt
    stage_t2_end.pt
    best.pt                   (copy of the overall-best)
    last.pt
  metrics/
    metrics_standard.json
    metrics_bundle.json
    metrics_best_checkpoint.json
    metrics_by_family.json
    metrics_by_source.json
    metrics_projected_slice.json
    validation_history.json
    calibration_summary.json
    loss_history.jsonl        (per-eval-step: loss mean, lr, dev metrics, is_best)
    dev_row_ids_t1.json       (dev-split row identifiers for T1)
    dev_row_ids_t2.json       (dev-split row identifiers for T2)
  predictions/
    predictions_scientific.jsonl
  eval/
    phase_a_eval.json         (auto-computed at end of training, §4.6)
```

Each checkpoint's `rng_state` captures Python, NumPy, Torch CPU, and (if present) Torch CUDA RNG streams. `dev_row_ids_*.json` freezes the exact dev membership so that any two runs can be compared on a common dev intersection without retraining. `config_sha256` + `git_commit` make every run's methodological provenance auditable. `loss_history.jsonl` carries per-evaluation-step records of the form `{stage, step, loss_recent_mean, lr, dev_accuracy, dev_macro_f1, is_best}`; all `dev_*` fields refer to the in-training dev split defined in §5.3 and are **not** external benchmark F1.

---

## Part 6 — Phase A: Schema Selection Experiment

### 6.1 Purpose and design

Phase A fixes the architecture (pipeline), update regime (full fine-tune), and schedule (T1→T2 staged), and varies the two axes most likely to drive both benchmark and KB behaviour: **schema** (S_flat, S_pair, S_mech) and **encoder** (RB, PB, BL, PL). Ten seeds per cell give 3 × 4 × 10 = **120 runs**.

Phase A's role is two-fold: (a) characterise the effect of schema on benchmark and KB behaviour independently of training-configuration variation, providing the evidence for H7's Phase A arm (variance asymmetry) and supplying the β_schema, β_encoder, and Phase A β_within slopes for H6's mechanism-stratified family (§9.4); (b) decide, via the schema-selection rule in §6.6, which schemas enter Phase B.

### 6.2 Cells

| Encoder | Shorthand | Checkpoint |
|---------|-----------|-----------|
| RoBERTa-base | RB | `FacebookAI/roberta-base` |
| PubMedBERT-base | PB | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` |
| BioLinkBERT-base | BL | `michiyasunaga/BioLinkBERT-base` |
| PubMedBERT-large | PL | `microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract` |

| Schema | Shorthand | # labels |
|--------|-----------|---------|
| S_flat | Sflat | 4 |
| S_pair | Spair | 8 |
| S_mech | Smech | 13 |

Naming: `PA_{ENC}_{SCHEMA}_s{NN}` with NN ∈ 01..10.

### 6.3 Locked hyperparameters

These values are shared across all 120 cells and are also the anchor-cell defaults for Phase B:

| Parameter | Value |
|-----------|-------|
| Architecture | pipeline |
| Update regime | full fine-tune |
| Schedule | T1 → T2 staged |
| Optimizer | AdamW |
| Learning rate | 2.0e-5 |
| Batch size | 4 |
| Max sequence length | 384 (RB uses 512) |
| T1 `max_updates` | 2048 |
| T2 `max_updates` | 2048 |
| `eval_every_steps` | 64 |
| `dev_fraction` | 0.12 |
| `early_stopping_patience` | 10 |
| `early_stopping_min_updates` | 256 |
| Selection metric | dev macro-F1 |
| `negative_ratio` | 4.0 |
| `max_negatives_per_sample` | 64 |
| `pair_type_filter` | `{schema}_legal_endpoints` |
| Source weights | biored = drugprot = bc5cdr = 1.0 |
| T3 / T4 | disabled |
| Save logits | true |

### 6.4 Data inputs

- T1 training shards (per schema): `t1_biored_trn_{SCHEMA}.jsonl`, `t1_drugprot_trn_{SCHEMA}.jsonl`, `t1_bc5cdr_trn_{SCHEMA}.jsonl`.
- T2 training shards (per schema): `t2_biored_mesh_{SCHEMA}.jsonl`, `t2_drugprot_mesh_{SCHEMA}.jsonl`, `t2_bc5cdr_mesh_{SCHEMA}.jsonl`.
- Evaluation inputs (all under `fine_tuning_experiments/schema_exp/eval/inputs/`): `biored_test_pairs_{SCHEMA}.jsonl`, `bc5cdr_test_pairs.jsonl`, `kb_surface_pairs.jsonl`.

### 6.5 Smoke test (required before the 120-run submission)

Three cells are trained first to verify end-to-end correctness of the Phase A pipeline:

1. `PA_PB_Sflat_s01` — baseline encoder, baseline schema; establishes that all pipelines (training → metrics → eval) run.
2. `PA_PB_Spair_s01` — exercises the 8-class classifier head and the label-space regression test (`derive_label_space` must return 8 labels including `ASSOCIATION_GENERAL`; covered by `tests/test_label_space.py`).
3. `PA_BL_Sflat_s01` — exercises a non-PubMedBERT encoder tokenizer/backbone.

Smoke acceptance criteria (all must pass before the 120-run array is submitted):

- Training completes without error (`run_manifest.json` written, `best.pt` ≥ 100 MB).
- `metrics/validation_history.json` has ≥ 20 dev entries (training didn't stop prematurely).
- `eval/phase_a_eval.json` is present and well-formed; `biored_test` and `bc5cdr_test` blocks both have finite macro-F1; `kb_surface` block has all three `kb_hit_A/B/C` metrics populated (not NaN).
- Checkpoints contain the `rng_state` key (artifact-enhancement verification).
- `metrics/dev_row_ids_t1.json` and `dev_row_ids_t2.json` exist and are non-empty.
- The label-space regression test (§3.2) passes independently.

#### 6.5.1 Smoke run record

The three cells `PA_PB_Sflat_s01`, `PA_PB_Spair_s01`, `PA_BL_Sflat_s01` were executed first as a single-GH200 smoke; elapsed ≈ 15 min per cell (≈ 13 min training + ≈ 105 s inline evaluation). All §6.5 acceptance criteria passed:

- `best.pt` ≈ 438 MB on every run; six checkpoint files (`stage_t1_best/end.pt`, `stage_t2_best/end.pt`, `best.pt`, `last.pt`) each carry a `rng_state` dict with four streams (`python`, `numpy`, `torch_cpu`, `torch_cuda`).
- `metrics/validation_history.json` has 64 entries per run (32 T1 + 32 T2, step 64 through 2048, interval 64); well above the ≥ 20 floor.
- `metrics/loss_history.jsonl` has 64 rows per run with `{stage, step, loss_recent_mean, lr, dev_accuracy, dev_macro_f1, is_best}`; T1 first-step loss 0.75–1.13 decays to 0.34–0.40 by step 2048; T2 loss trajectory is well-behaved.
- `metrics/dev_row_ids_t1.json` (≈ 3550 row IDs) and `dev_row_ids_t2.json` (≈ 3030 row IDs) are both populated on every run.
- `eval/phase_a_eval.json` is well-formed on every run with all `biored_test`, `bc5cdr_test`, `kb_surface` fields finite; `eval/kb_surface_targets.jsonl` contains 165 per-target records with full logit vectors; every eval file carries `eval_version = 1.0`.
- `run_manifest.json` carries a non-empty `_git_commit`, a distinct `_config_sha256` per cell, and `_trainer_source = fine_tuning_experiments.phase_b.trainer.scientific_trainer`.
- The `S_pair` label space is 8 labels including `ASSOCIATION_GENERAL`.
- `fine_tuning_experiments/phase_b/trainer/tests/run_all.py` (the label-space + schema-expected-label + negative-sampler + eval-fields + checkpoint-roundtrip + training-determinism unit-test bundle) passes locally.

Qualitative direction check (three single-seed cells, no statistical weight): the expected schema × encoder asymmetry is visible — schema granularity moves BC5CDR DRUG_DISEASE F1 by only ≈ 0.03 and BioRED macro by ≈ 0.13, while moving `KB_hit_A_setvalued` by ≈ 0.13; an encoder swap on S_flat (PB → BL) independently shifts `KB_hit_A_setvalued` by ≈ 0.25. These are not pre-registered tests; they only confirm the pipeline does not produce visibly pathological metrics. The qualitative pattern ("schema moves BioRED substantially more than it moves BC5CDR DD; both shift KB_hit_A by a magnitude comparable to or smaller than the BioRED shift") survives seed-averaging in the full 120-run variance decomposition (§6.9.3).

### 6.6 Schema-selection rule for Phase B

For each schema, compute the pooled mean (over 4 encoders × 10 seeds = 40 runs) of:

- BioRED macro-F1 (ex-NEG)
- BC5CDR DRUG_DISEASE F1
- `KB_hit_A_setvalued`
- Per-head F1 for each active head

Decision tree (pre-committed):

- **Outcome 1 — single schema dominates.** One schema is strictly higher on `KB_hit_A_setvalued` with the paired-bootstrap 95 % CI on the difference to the second-best schema excluding zero, AND not worse on BioRED macro-F1 (ex-NEG) by more than a Cohen's d of 0.3. → Phase B runs that schema only.
- **Outcome 2 — dual schema required.** The top two schemas are within a Cohen's d of 0.3 on `KB_hit_A_setvalued` OR they disagree in direction between `KB_hit_A` and BioRED macro-F1 (a schema-effect-direction disagreement). → Phase B runs both as parallel factorials.
- **Outcome 3 — all schemas dominated by benchmark noise.** None of the three schemas separates on any primary metric beyond within-cell SD. → Phase B retains S_pair alone and reports Phase A as a null result.

**Pre-committed bootstrap specification.** The "CI non-overlapping the second-best" clause in Outcome 1 is evaluated with a *paired* bootstrap at the `(encoder, seed)` cell level, not an unpaired two-sample bootstrap. For each of the 40 matched cells we form the per-cell difference (`top_schema_value − runner_up_value`); the 40 per-cell differences are resampled with replacement B = 10 000 times, and the 2.5 % / 97.5 % quantiles of the mean-diff distribution are reported. Pairing controls for between-encoder variance (H7 estimates encoder alone at ≈ 17 % of `KB_hit_A` SS), yielding a correctly calibrated CI that is tighter than the unpaired version without relying on distributional assumptions. Cohen's d (primary and BioRED-guard) uses the unpaired pooled-SD formula.

### 6.7 Pipeline sanity expectations

The 120-run factorial is submitted only after the 3-cell smoke (§6.5) confirms the pipeline produces metrics in the expected sign and order of magnitude on a single-seed subset. The qualitative expectations are: (i) schema granularity moves BioRED macro-F1 substantially more than it moves `KB_hit_A` (the variance-asymmetry phenomenon that H7 quantifies); (ii) S_mech's single-label KB metric is collapsed by the dead mechanism heads; (iii) S_pair's specialised pair heads absorb gold relations that nominally belong to `ASSOCIATION_GENERAL` (the "AG-collapse" effect that motivates §6.9.5's active-head secondary). These are not pre-registered tests; they are operational checks that the pipeline is producing sensible output before committing to the full factorial.

### 6.8 Analyses (pre-committed)

The canonical Phase A analysis pipeline is three scripts, each with a single, non-overlapping responsibility:

1. `schema_exp/eval/eval_one_run.py` — invoked once per sbatch task (120 total), writes `runs/.../PA_*/eval/phase_a_eval.json` with `eval_version = 1.0`, the three benchmark blocks (`biored_test`, `bc5cdr_test`, `kb_surface`), and the 165 per-target logit vectors in `eval/kb_surface_targets.jsonl`.
2. `schema_exp/eval/aggregate_phase_a.py` — reads the 120 per-run JSONs, enforces the `EVAL_VERSION.txt` read gate, and writes `phase_a_results.csv` (flat run × metric table), `phase_a_aggregate.json` (mean ± SE by encoder × schema and by schema), `phase_a_schema_selection.json` (applies the §6.6 decision tree on `kb_hit_A_setvalued` with paired difference-CI and BioRED ex-NEG guard), and `phase_a_report.md` (human-readable summary).
3. `schema_exp/analysis/phase_a_analysis.py` — reads `phase_a_results.csv` and writes `analysis/phase_a_analysis.{json,md}` containing the full scientific bundle: pooled and cell-level means with bootstrap CIs, H7 variance decomposition, cross-metric correlation preview, per-head F1 with support and 0.05 feasibility floor, permutation tests with BH-FDR over the 21-member schema × metric family, encoder-stratified schema comparison, and cell-level ICC(1,1) for the primary metrics.
4. `schema_exp/build_runs_inventory.py` — walks the 120 run directories and writes `runs_inventory.csv` with the SHA-256 of every `best.pt`, the config hash, and the training manifest fields required for reproducibility.

Results are reported as §6.9 after Phase A completes.

### 6.9 Results

**Status**: 120 / 120 runs completed (3 schemas × 4 encoders × 10 seeds); every run carries the seven required artifacts (`run_manifest.json`, six checkpoint files each with a four-stream `rng_state`, `validation_history.json` with 32 T1 + 32 T2 dev evaluations, per-step `loss_history.jsonl`, dev row-ID dumps for both stages, `eval/phase_a_eval.json` with finite benchmark and KB metrics, and `eval/kb_surface_targets.jsonl` with 165 per-target logit vectors). Every run carries `eval_version = 1.0`. The full machine-readable outputs live at `fine_tuning_experiments/schema_exp/analysis/phase_a_analysis.{json,md}` (and the aggregated CSV/JSON at `.../schema_exp/phase_a_{results.csv, aggregate.json, schema_selection.json, report.md}`). Per-run SHA-256 of `best.pt`, config hash, and training manifest live in `fine_tuning_experiments/schema_exp/runs_inventory.csv` (120 rows).

#### 6.9.1 Pooled schema means (n = 40 each)

| Schema | KB_hit_A_sv | KB_hit_A_sl | KB_pmass_B_sv | KB_auc_C_sv | BioRED macro | BioRED ex-NEG | BC5CDR DD |
|---|---|---|---|---|---|---|---|
| S_flat | 0.578 ± 0.238 | 0.578 ± 0.238 | 0.466 ± 0.151 | 0.785 ± 0.219 | 0.310 ± 0.030 | 0.139 ± 0.036 | 0.765 ± 0.143 |
| S_pair | **0.695 ± 0.181** | **0.695 ± 0.181** | **0.525 ± 0.100** | **0.807 ± 0.137** | **0.365 ± 0.080** | **0.300 ± 0.089** | **0.796 ± 0.082** |
| S_mech | 0.453 ± 0.195 | 0.002 ± 0.004 | 0.445 ± 0.116 | 0.664 ± 0.201 | 0.191 ± 0.047 | 0.139 ± 0.049 | 0.789 ± 0.089 |

Cells show pooled mean ± SD (n = 40 runs per schema). 95 % bootstrap CIs for the primary KB metric: S_pair [0.638, 0.748] > S_flat [0.501, 0.649] > S_mech [0.395, 0.515]. The single-label variant (`KB_hit_A_singlelabel`) collapses on S_mech (0.002) because five of the six mechanism heads (`DGR_ACTIVATE/INHIBIT/METABOLIC/REGULATE/STRUCTURAL`) have no BioRED test support and are never picked as argmax — confirming that S_mech's 13-way split is unsupportable on the present evaluation corpus.

#### 6.9.2 Schema selection (§6.6 applied)

Applying the pre-committed decision tree on `kb_hit_A_setvalued`. All difference CIs below are the paired-bootstrap CI pre-committed in §6.6 (40 matched `(encoder, seed)` cells, B = 10 000).

- S_pair vs S_flat: diff = +0.117, paired CI [+0.045, +0.194] **excludes zero**; Cohen's d = +0.55; permutation p = 0.015 (BH-adj p = 0.024).
- S_pair vs S_flat on BioRED ex-NEG: diff = +0.161, paired CI [+0.137, +0.183], d = +2.37 → S_pair is strictly better on the benchmark, so the "not worse by d > 0.3" clause is trivially satisfied.
- S_pair vs S_mech: diff = +0.242, paired CI [+0.171, +0.313], d = +1.29, p < 0.0001.

**Outcome 1 (single schema dominates). Phase B runs S_pair only.**

The encoder-stratified robustness check is consistent: S_pair > S_flat on `kb_hit_A_setvalued` in all four encoders (RB +0.159, PB +0.052, BL +0.132, PL +0.127; all four positive = S_pair higher) and S_pair > S_mech in all four (RB +0.144, PB +0.177, BL +0.374, PL +0.275). The weakest encoder-stratified separation is on PB where RB and PB each close with the benchmark-ceiling encoders.

#### 6.9.3 H7 variance decomposition (Phase A preview — to be replicated under configuration variance in Phase B)

Fraction of total SS attributable to each factor (3 schemas × 4 encoders × 10 seeds, one-way decomposition):

| Metric | schema | encoder | schema × encoder | within-cell (seed) |
|---|---|---|---|---|
| `kb_hit_A_setvalued` | 19.1 % | 17.2 % | 3.9 % | 59.7 % |
| `biored_macro_f1_ex_neg` | **60.4 %** | 24.2 % | 7.3 % | 8.2 % |
| `bc5cdr_drug_disease_f1` | 1.5 % | **37.2 %** | 3.6 % | 57.7 % |
| `biored_macro_f1` | 63.0 % | 23.8 % | 6.3 % | 6.9 % |

Schema granularity explains 60.4 % of BioRED ex-NEG variance but only 19.1 % of `kb_hit_A_setvalued` variance and 1.5 % of BC5CDR DD variance — a 3.2× asymmetry between the benchmark a schema is designed to affect and the primary downstream metric it is meant to serve, and a 40× asymmetry against the out-of-domain BC5CDR benchmark. Encoder choice is the dominant driver of BC5CDR DD (37.2 %) and is comparable to schema on KB_hit_A (17.2 %). Seed-level within-cell noise is 59.7 % of KB_hit_A variance vs 8.2 % of BioRED ex-NEG variance — the key signal for Phase B ICC planning (§7 and §9.3).

#### 6.9.4 Cross-metric correlations (Phase A pooled, n = 120 seed-level)

| Metric pair | Pearson r [95 % CI] | Spearman ρ [95 % CI] |
|---|---|---|
| BioRED ex-NEG × KB_hit_A | +0.523 [+0.396, +0.637] | +0.457 [+0.297, +0.596] |
| BC5CDR DD × KB_hit_A | +0.435 [+0.258, +0.570] | +0.360 [+0.182, +0.520] |
| BioRED ex-NEG × BC5CDR DD | +0.503 [+0.432, +0.595] | +0.571 [+0.425, +0.694] |
| BioRED macro (all) × KB_hit_A | +0.563 [+0.441, +0.677] | +0.569 [+0.448, +0.683] |

At cell level (n = 12 encoder × schema cells), Pearson r rises to +0.77 for BioRED × KB and +0.58 for BC5CDR × KB. All point estimates are positive and all 95 % CIs exclude zero by a comfortable margin. Under the correctness-aware KB family, the benchmark–KB relationship is therefore positive-coupled, and the scientifically supported central claim of the paper is **variance asymmetry** ("schema / configuration choice shifts benchmark F1 far more than it shifts KB_hit_A"; §7.2 H7) rather than rank reversal. RQ4 writing follows the variance-asymmetry framing; the slope magnitudes for Phase B's β_config are reported in H6 (§7.2, §9.4) as a mechanism-stratified coupling family.

#### 6.9.5 Per-head BioRED F1 (feasibility against the 0.05 floor)

BioRED test-support (gold positive count) and mean per-head F1 by schema:

| Schema | Heads with support > 0 | Heads above 0.05 F1 | Dead heads | Notes |
|---|---|---|---|---|
| S_flat | 2 (`ASSOCIATION_GENERAL` sup ≈ 969, `DRUG_GENE_REGULATION` sup ≈ 21) | 1 (`AG` F1 = 0.412) | `DGR` F1 = 0.005 | AG is the only populated head; DGR is test-support limited. |
| S_pair | 7 | 5 (`GENE_DISEASE` 0.479, `DRUG_DISEASE` 0.451, `VARIANT_DISEASE` 0.429, `GENE_GENE_ASSOC` 0.385, `DRUG_GENE_REGULATION` 0.354) | `AG` F1 = 0 (sup ≈ 104, absorbed by specialised heads); `DRUG_VARIANT_ASSOC` F1 = 0 (sup ≈ 15, data-thin) | 5 active heads + NEGATIVE; AG-collapse is a predictable category-competition effect. |
| S_mech | 7 (5 DGR sub-heads have 0 BioRED test support) | 4 (`DRUG_DISEASE` 0.449, `GENE_DISEASE` 0.452, `GENE_GENE_ASSOC` 0.403, `VARIANT_DISEASE` 0.361) | all 5 `DGR_*` sub-heads, `AG`, `DRUG_GENE_REGULATION`, `DRUG_VARIANT_ASSOC` | Evaluation corpus cannot populate the mechanism sub-heads — S_mech is structurally unsupportable on BioRED. |

The AG-collapse in S_pair is a category-competition effect: once S_pair's five specialised pair heads are available, BioRED's residual AG-annotated pairs are almost entirely re-routable to one of them, so the AG head receives near-zero test-support-at-argmax signal and the classifier learns to predict zero mass for it. Combined with `DRUG_VARIANT_ASSOC` (data-thin at support ≈ 15), the effective S_pair classifier operates as a **5 active pair heads + NEGATIVE** space rather than the nominal 8-way head, even though all 8 classifier outputs remain present.

**Pre-registered secondary: active-head macro-F1.** For every primary H1–H4 and H6 test on BioRED ex-NEG, a parallel test is computed on "active-head macro-F1" = macro over the frozen set **{`GENE_DISEASE`, `DRUG_DISEASE`, `VARIANT_DISEASE`, `GENE_GENE_ASSOC`, `DRUG_GENE_REGULATION`}** — the set of S_pair heads with Phase A mean test-F1 > 0.05. `ASSOCIATION_GENERAL` and `DRUG_VARIANT_ASSOC` are excluded. **The active-head identity is pre-committed at this lock and does not adapt to any Phase B per-head result**: if Phase B produces an AG head with F1 > 0.05 or a currently-active head that collapses to F1 ≤ 0.05, the active-head set is **not** re-computed — both cases are reported as exploratory findings with their own stand-alone active-head-set definitions in a Supplementary table. Decision rules (confirmed / partial / null / inverted) on the active-head macro-F1 are identical to the primary test. Disagreements between the primary and active-head secondary are flagged in the Results narrative and in a Supplementary robustness table. This is a planned-secondary comparison under §9.1 and is included in the Bonferroni-corrected secondary family (not in the BH-FDR primary family). The active-head secondary is also the default metric for the H5 equivalence test where dead heads would otherwise dilute the margin.

Active-head macro-F1 is computed **post-hoc at evaluation time only** from the per-head F1 values already present in `eval/phase_a_eval.json`. It is **not** used as the in-training dev metric (see §5.3): every Phase A and Phase B run is trained and early-stopped against the in-training dev macro-F1 computed over the full schema vocabulary. Switching the training objective to an active-head variant would alter the checkpoint selection and therefore the per-run numbers themselves, which is not the intended secondary analysis.

This does not invalidate the S_pair schema selection (the `KB_hit_A_setvalued` ranking in §6.9.2 is metric-independent of which BioRED heads are active), but it does refine the reported benchmark story: we publish both the nominal-schema macro-F1 and the active-head macro-F1 so readers can judge the S_pair label-space design on both. A reviewer who argues "why not simply use a 5-class + NEG schema?" will find the active-head macro-F1 column and see that the practical answer is: we report the nominal-schema macro-F1 because it is the schema we trained on (and switching schemas requires re-training), and we report the active-head macro-F1 because it is what the model effectively distinguishes.

#### 6.9.6 Intraclass correlation (cell-level, 12 cells × 10 seeds)

| Metric | ICC(1,1) | interpretation |
|---|---|---|
| `kb_hit_A_setvalued` | 0.360 | fair |
| `biored_macro_f1_ex_neg` | 0.916 | excellent |
| `biored_macro_f1` | 0.929 | excellent |
| `bc5cdr_drug_disease_f1` | 0.383 | fair |

ICC for KB_hit_A and for BC5CDR DD is structurally lower than for BioRED because their seed-level SDs within a cell are large relative to between-cell differences — consistent with the 59.7 % and 57.7 % within-cell variance shares in §6.9.3. The 0.36 ICC on KB_hit_A for Phase A's schema × encoder design is sufficient to make pooled-schema comparisons meaningful (§9.3); Phase B's paired design (§7.3) further tightens the effective SE because seeds are matched across compared cells.

---

## Part 7 — Phase B: Training Configuration Factorial

### 7.1 Purpose and pre-commitments

Phase B is the confirmatory phase. It answers RQ2 (training configurations) with a factorial design under the schema selected by Phase A (S_pair; §6.9.2), produces the seed-level evidence for RQ3 (KB surfacing under configuration variation) and RQ4 (variance asymmetry and seed-level coupling; §6.9.3–§6.9.4), and is the sole source of evidence used for the main results table. Phase A informs the schema set and provides the schema-induced-variance arm of the RQ4 analysis; Phase B provides the configuration-induced-variance arm.

Because Phase A and Phase B use the same trainer source (§5), seed-level observations from both phases are combined in the joint RQ4 analysis (§9.4) without a cross-trainer adjustment.

The pre-registration mechanism locks this document at a public git commit hash before any Phase B sbatch is submitted. Amendments after lock are logged in Appendix B with date, trigger, and replacement text.

### 7.2 Research hypotheses

Phase B tests seven primary hypotheses (H1–H7). The earlier H8 "conclusions of H1–H5 hold under both S_pair and S_flat" is dropped because Phase A's Outcome 1 (§6.9.2) designates S_pair as the sole Phase B schema, making a dual-schema robustness test untestable with the allocated sample. Schema-robustness of H1–H5 is instead reported descriptively against Phase A's existing S_flat and S_mech evidence, with no confirmatory statistical claim.

| ID | Claim | Tier | Primary test | Decision rule |
|----|-------|------|-------------|---------------|
| **H1** | PubMedBERT-large > {PubMedBERT-base, BioLinkBERT-base} on BioRED macro-F1 ex-NEG under matched config. | Primary | Paired-t + Wilcoxon (dual report); 3 pairwise tests. | Confirmed: PL > both by Δ ≥ 0.02 with FDR-q < 0.05 on both tests. Null: both gaps < 0.01 or q > 0.10. Inverted: PL < either by Δ ≥ 0.02, q < 0.05. |
| **H2** | Multi-corpus T1 (BioRED+DrugProt+BC5CDR) > BioRED-only T1 on BC5CDR DD F1 (held-out OOD). | Primary | Paired-t + Wilcoxon on T1_flat vs T1_biored_only at PB × pipeline × full-FT. | Confirmed: Δ ≥ 0.03 with q < 0.05 on both. Null: \|Δ\| < 0.02. Inverted: Δ ≤ −0.03 with q < 0.05. |
| **H3** | T1→T2 staged > T1_flat on both BioRED and BC5CDR generalisation, across all three biomedical encoders. | Primary | Paired-t + Wilcoxon on BioRED ex-NEG and BC5CDR DD separately at PB/BL/PL × pipeline × full-FT; 6 tests; FDR over 6. | Confirmed: ≥ 4 of 6 tests show Δ ≥ 0.02 with q < 0.05. Partial: 2–3 of 6. Null: otherwise. |
| **H4** | Full fine-tune > LoRA on the small-data oncology bridge with Cohen's d ≥ 0.5 for biomedical encoders. | Primary | Paired-t + Wilcoxon on BioRED ex-NEG, full-FT vs LoRA, at PB/BL/PL × pipeline × T1→T2. | Confirmed: all 3 encoders show full-FT > LoRA with d ≥ 0.5 and q < 0.05. Partial: 2 of 3. LoRA-preferred counter-finding: any encoder shows LoRA > full-FT with q < 0.05. |
| **H5** | Pipeline ≈ shared-multitask on macro-F1 ex-NEG (equivalence). | Primary (TOST) | TOST on pipeline vs shared-multitask at PB and PL × full-FT × T1→T2; equivalence margin ±0.03. | Equivalent: 90 % CI of Δ ⊂ [−0.03, +0.03] for both encoders. Not equivalent: otherwise. |
| **H6** | The benchmark-to-KB coupling slope is a **family of five mechanism-stratified slopes** (β_within, β_schema, β_encoder, β_config, β_combined_cell), not a single quantity — each slope has its own estimand and may take a different value. | Primary (characterisation family) | See §9.4 for fit specification; §9.3 for per-slope CI-width precondition; §7.2 *H6 abstract-level claim mapping* (below) for the pre-committed slope-to-narrative mapping. | Three-bin label per slope (strong / moderate / weak), with an "inconclusive" label for any slope whose 95 % CI width exceeds 0.30; no single slope carries the headline alone. |
| **H7** | Design levers (schema in Phase A; configuration in Phase B) exhibit variance asymmetry — they explain more variance in BioRED macro-F1 ex-NEG than in `KB_hit_A_setvalued`. | Primary (headline for RQ4) | Variance decomposition of both metrics across (schema, encoder, seed) in Phase A and across (encoder, architecture, update regime, schedule, seed) in Phase B. Compute ratio R = (design-lever share in BioRED_F1) / (design-lever share in KB_hit_A), per phase. | **Phase A arm (descriptive — data already observed, §6.9.3).** Under the same formula used for R_B (design-lever share = schema + encoder + schema×encoder interaction), R_A = 91.9 / 40.2 ≈ 2.29. The "schema-only" narrowing of the ratio — 60.4 / 19.1 ≈ 3.16 — is reported alongside because it isolates the schema-granularity effect from the correlated encoder variance. No threshold is pre-committed against observed data; R_A is reported as a pair of point estimates. **Phase B arm (pre-committed threshold, confirmatory).** R_B ≥ 2 → "configuration-induced asymmetry confirmed" (a ≥ 2× disparity in variance share between benchmark and KB, chosen because R = 1 corresponds to a perfect proxy and any R < 2 leaves insufficient margin to distinguish asymmetry from noise given 36 Phase B cells; R = 2 is a conventional "large" effect-size threshold in variance-share ratio literature). 1 < R_B < 2 → "borderline, reported descriptively". R_B ≤ 1 → "null, no asymmetry under configuration variance". The Phase B threshold is justified independently of the Phase A value and does not adapt post-hoc to R_A. |

**Honest disclosure on H6.** Phase A's pooled Spearman ρ(BioRED ex-NEG, `KB_hit_A_setvalued`) = +0.46 [+0.30, +0.60] and Pearson r = +0.52 [+0.40, +0.64] exclude zero with comfortable margin. A single-slope pooled-null formulation of H6 is therefore not tenable. What the pooled correlation does *not* identify is which design mechanism (schema, encoder, configuration, or within-cell seed noise) drives the observed coupling — that decomposition is precisely what §9.4's mechanism-stratified slopes quantify, and §9.5's variance decomposition complements. The mechanism-specific nulls ("β_within = 0", "β_config = 0", etc.) remain testable on Phase B data; each is reported with its own CI and three-bin label.

**H6 secondary (active-head).** Each of the five slopes in the H6 family is re-fit using the active-head macro-F1 (§6.9.5) as the benchmark side; the full-schema macro-F1 remains the primary. This guards against dead-head artifacts dominating the coupling estimate.

**H6 abstract-level claim mapping (pre-commitment).** The Abstract can accommodate at most one-line characterisations of the H6 family; we pre-commit here which slope's behaviour controls the headline wording, so that we do not story-fit after Phase B unblinds:

| Observed H6 pattern | Pre-committed Abstract-level framing |
|---|---|
| β_config is **weak** (\|β̂\| < 0.3 or CI straddles zero) | "Benchmark rank is a low-fidelity guide to KB surfacing in the regime of typical configuration choices." **(the currently anticipated framing)** |
| β_config is **moderate** (0.3 ≤ \|β̂\| < 1.0) and β_schema strong | "Benchmark-to-KB coupling is mechanism-dependent: strong under schema choice, weaker within a fixed schema across configurations." |
| β_config is **strong** (\|β̂\| ≥ 1.0) and same-sign CI | "Variance asymmetry (H7) is the primary RQ4 finding; within-regime benchmark-to-KB coupling is stronger than we anticipated, so rank-based model selection under fixed schema is better-calibrated than a pure decoupling story would predict." |
| β_config is **inconclusive** (CI width > 0.30) | "We report variance asymmetry (H7) and the H6 slope family descriptively; β_config's CI width exceeds the 0.30 reportability gate, and no Abstract-level coupling claim is made within a fixed schema." |

This mapping is the full set of admissible headlines for H6 + H7 combined; we will not draft a fifth variant that happens to read well after seeing the data. If the observed pattern does not cleanly match one of the four rows (e.g., β_schema inverts sign), that is itself reported as a counter-finding and the Abstract says so.

### 7.3 Design axes

- **Encoder** (3 levels in main factorial + 1 descriptive reference):
  - In factorial: PB (PubMedBERT-base), BL (BioLinkBERT-base), PL (PubMedBERT-large).
  - Reference cell only: RB (RoBERTa-base). 5 seeds at the anchor configuration. RB serves as a general-domain pretraining baseline to quantify the benefit of biomedical pretraining on the present task; it is excluded from every primary and planned-secondary test (H1–H7) and from the FDR correction because it represents a pretraining paradigm outside the hypothesis space. Its value is descriptive: the Methods section will report RB numbers alongside the biomedical encoders so that readers can contextualise the biomedical-pretraining premium. Phase A numbers already establish that RB sits below the biomedical encoders on `KB_hit_A` (RB S_flat 0.26 / S_pair 0.52 vs PL S_flat 0.62 / S_pair 0.69, §6.9.1); Phase B's RB reference confirms this under the anchor configuration.
  - Excluded: SciBERT, BlueBERT, BioBERT (older biomedical encoders), BiomedBERT, BiomedLM, GatorTron-base — modelling decision that PB/BL/PL span "biomedical-pretraining variant + scale" cleanly.
- **Architecture** (2 levels): `pipeline` (single classifier head, Phase A's design) and `shared_multitask` (joint NER+RE with shared encoder and task-specific heads, loss `0.3·L_NER + 0.7·L_RE`).
- **Update regime** (2 levels): `full_ft` (all params) and `lora` (rank 16, α = 32, dropout 0.05, target modules = attention Q/V projections, classifier head fully trained, LR unchanged at 2e-5 for a matched-budget comparison).
- **Schedule** (3 levels): `T1_biored_only`, `T1_flat`, `T1_to_T2_staged`.
- **Schema**: S_pair only (single schema, selected by Phase A Outcome 1 in §6.9.2). S_flat and S_mech are excluded from Phase B; their Phase A numbers are carried into the paper as the schema-variance arm of the RQ4 analysis.
- **Seed** (10 levels): seed indices 1..10 match Phase A's seed indices (same `torch.manual_seed(seed)` calls) so that seed is a meaningful cross-phase random effect.

### 7.4 Factorial plan

3 encoders × 2 architectures × 2 update regimes × 3 schedules = **36 cells**. 10 seeds per cell = **360 main runs**. Plus 5 RB reference runs at the anchor cell = **365 runs total**.

Cell coverage:

| Config | T1_biored_only | T1_flat | T1→T2 (anchor) |
|---|---|---|---|
| PB × pipeline × full-FT | 1 | 2 | 3 (anchor) |
| PB × pipeline × LoRA | 4 | 5 | 6 |
| PB × shared-MT × full-FT | 7 | 8 | 9 |
| PB × shared-MT × LoRA | 10 | 11 | 12 |
| BL × pipeline × full-FT | 13 | 14 | 15 |
| BL × pipeline × LoRA | 16 | 17 | 18 |
| BL × shared-MT × full-FT | 19 | 20 | 21 |
| BL × shared-MT × LoRA | 22 | 23 | 24 |
| PL × pipeline × full-FT | 25 | 26 | 27 |
| PL × pipeline × LoRA | 28 | 29 | 30 |
| PL × shared-MT × full-FT | 31 | 32 | 33 |
| PL × shared-MT × LoRA | 34 | 35 | 36 |

Run naming: `PB_{ENC}_{ARCH}_{UPD}_{SCHED}_s{NN}` with `ENC ∈ {PB, BL, PL, RB}`, `ARCH ∈ {P, MT}`, `UPD ∈ {FT, LR}`, `SCHED ∈ {T2, T1F, T1B}`, `NN ∈ 01..10`. The `SCHEMA` field is dropped — all Phase B runs are S_pair.

### 7.5 Why a full factorial, not a star design

With GPU not the binding constraint, the full factorial enables every two-way and three-way interaction to be estimated. The star-design alternative would leave encoder × update, encoder × architecture, and H1 × H5 interactions confounded. H4's cross-encoder generalisation requires BL × LoRA, which a star design would omit. The full factorial at n = 10 seeds per cell also gives H7's variance decomposition adequate power to partition configuration variance across its four design axes — the single most important Phase B estimand because it closes the loop on the RQ4 variance-asymmetry headline.

### 7.6 Training configuration

All cells share the §6.3 locked hyperparameters. LoRA cells add rank 16, α = 32, dropout 0.05, `target_modules = ["query", "value"]`, classifier fully trained, LR unchanged at 2e-5 (matched budget; not LoRA-optimal — this is a conservative specification for H4). Shared-multitask cells use a token-level NER head (`{GENE, VARIANT, DRUG, DISEASE, O}`) in addition to the RE head; both losses backprop through the shared encoder with fixed weights 0.3 (NER) and 0.7 (RE).

### 7.7 Phase B+ — LLM robustness (deferred)

A follow-on track tests whether the RQ4 findings (variance asymmetry and seed-level coupling characterisation) persist under LLM-paradigm training. Pre-commitments retained:

- LLaMA-3-8B-Instruct via LoRA (rank 16) + zero-shot reference.
- T1→T2 staged under S_pair; 5 seeds.
- Hypothesis Hllm: the variance asymmetry observed under encoder-based RE (H7) is reproduced under LLM-paradigm training.

Full specification deferred until Phase B-main completes.

### 7.8 Infrastructure hardening checklist (pre-lock)

Phase A ran 120 training jobs, produced 720 checkpoint files, and exercised the eval pipeline on every one. Phase B is 3× that volume at 360 runs + 365 × 6 = ~2,190 checkpoints + re-evaluation on every Phase A checkpoint if the eval pipeline version changes. The cost of an undetected infrastructure failure scales accordingly. Before the pre-registration lock we verify and document the following:

**Trainer unit tests (must pass in CI on the pre-lock commit):**

- `tests/test_label_space.py` — S_flat = 4, S_pair = 8 (includes `ASSOCIATION_GENERAL`), S_mech = 13. (Already present.)
- `tests/test_schema_expected_label.py` — covers every (family × heuristic gold × schema × projection-mode) cell in §3.4's mapping table and asserts the expected label set matches.
- `tests/test_negative_sampler.py` — given a tiny synthetic document with three entity mentions, confirms the sampler excludes the positive's head-tail and its reverse, respects `pair_type_filter`, produces exactly `negative_ratio × n_positives` negatives in expectation, and is reproducible under a fixed seed.
- `tests/test_eval_fields.py` — loads a saved `phase_a_eval.json` and asserts every field listed in §4.6 is present and finite; guards against accidental field deletion.
- `tests/test_checkpoint_roundtrip.py` — saves a checkpoint with an `rng_state` dict, reloads it, and asserts all four RNG streams (`python`, `numpy`, `torch_cpu`, `torch_cuda`) are bitwise-equal before and after.
- `tests/test_training_determinism.py` — guards the **two cheap invariants** that together span the full random path in `_train_one_stage`: (a) `_set_all_seeds(k)` produces bit-identical draws across invocations for all four RNG streams (`random`, `numpy`, `torch_cpu`, `torch_cuda`); (b) `_OnlineCollator(seed=k)` produces byte-identical batches given fixed inputs (covering shuffling + document-negative sampling). End-to-end training determinism is covered separately by the §7.8 1-cell paranoia smoke test (`seed = 11`, run once pre-lock), which retrains a complete Phase B cell and compares its full artifact set against the manifest.

**Filesystem / retention:**

- Scratch `/lus/lfs1aip2/projects/b5ac/` retention policy: no automatic purge within the project's active quota allocation (confirmed with site support).
- Phase A's 120 run directories and the 720 `.pt` checkpoints are inventoried in `fine_tuning_experiments/schema_exp/runs_inventory.csv` with SHA-256 of `best.pt` per run, so that any later silent corruption or deletion can be detected.
- Disk headroom for Phase B is checked before submission: 365 runs × (6 × 438 MB + eval artifacts) ≈ 1.1 TB estimated; current free space ≥ 2 × estimated is the gate.

**Code backup:**

- The pre-registration commit (to be tagged `phase_b_prelock_v1`) is pushed to the GitHub remote `project_1` repository. A second, site-local mirror of the same commit is produced as a full-repo tarball (`.git` + working tree, `git clone --mirror` equivalent) archived to `/lus/lfs1aip2/projects/b5ac/backups/phase_b_prelock_v1.tar.gz`. A dedicated site-local git server is not available on the Isambard Lustre filesystem, so a tarball on an independent Lustre path satisfies the "two independent copies of the lock commit" requirement. Verification at lock time: (i) `git rev-parse HEAD` in the live repo equals the SHA under the tag; (ii) `git rev-parse HEAD` on a fresh extract of the tarball into a scratch directory returns the same SHA; (iii) the SHA-256 of the tarball itself is recorded alongside the tag.
- A `RESTORE.md` in the repository root documents the exact steps to reproduce a training run from either copy: GitHub clone or tarball extract → `pip install -r requirements.txt` → fetch shards from scratch → invoke `run_experiment.py` with the `run_manifest.json`'s `config_path` argument. This is dry-run-verified on a smoke cell before lock.

**Evaluation-pipeline version pin:**

- `fine_tuning_experiments/schema_exp/eval/` contains a `EVAL_VERSION.txt` file; bumping the string requires re-running every Phase A eval (and, once Phase B lands, every Phase B eval). The policy is enforced at **two** gates:
  1. **Inline-eval write gate.** `scientific_trainer._try_inline_eval` asserts that the `EVAL_VERSION.txt` string loaded at training time matches an expected constant pinned in the trainer (the trainer and eval are compiled together). A mismatched version aborts the training job *before* writing `phase_a_eval.json`, so the job fails loudly rather than silently producing a wrong-version eval file that later needs re-running.
  2. **Aggregator read gate.** `aggregate_phase_a.py` / `aggregate_phase_b.py` refuse to ingest records whose `phase_a_eval.json` does not carry the current `EVAL_VERSION` string, logging the offending run path for re-evaluation.

**Phase B 1-cell paranoia smoke (post-lock, pre-array-submit gate).** Between the pre-registration lock and the submission of the Phase B array, a single Phase B cell is submitted as a standalone sbatch: `PB × pipeline × full-FT × T1→T2`, seed = 11 (outside the Phase A seed range 1–10, so it does not collide with any Phase A artifact and does not consume a Phase B primary seed). The smoke cell verifies:

1. The Phase B sbatch template launches correctly on the cluster (array indexing, scratch staging, GPU request format).
2. The Phase B config YAML generated for this cell is parsed correctly by the trainer.
3. The inline eval version check triggers as intended — a deliberate `EVAL_VERSION.txt` bump in a throwaway branch causes the training job to fail fast before writing an eval file, confirming the §7.8 two-gate enforcement.
4. `h6_coupling_slopes.py` (built during the pilot-fit step) can parse the resulting per-run JSON even with n = 1 cell, failing gracefully rather than silently returning garbage.
5. The Phase B aggregator (`aggregate_phase_b.py`) ingests the resulting eval record successfully.

If any of the five checks fail, the failure is diagnosed and fixed in a post-lock amendment (Appendix B) before the main Phase B array is submitted. The lock itself is not rolled back — the spec is unchanged; only infrastructure is fixed. Seed 11's run directory is **retained** (not deleted) and flagged `excluded: true` in `runs_inventory.csv`; it is not used in any Phase B analysis (seed-wise, cell-wise, or aggregated). This keeps a known-good single-cell debug artifact available for future use and keeps the inventory manifest accurate at all times.

This checklist (unit tests + filesystem + code backup + eval version pin + 1-cell smoke) is itself pre-registered: any item that fails at lock time is recorded in Appendix B with an explicit remediation plan. No Phase B array sbatch is submitted until all items pass.

---

## Part 8 — Evaluation Pipeline Runs (Pinned)

### 8.1 Per-run evaluation

The inline v1.0 evaluator (§4.6) runs automatically at the end of every Phase A training job via `scientific_trainer._try_inline_eval`. Phase B evaluations invoke the same pipeline post-training as an array job. Evaluation outputs live at `<run_dir>/eval/phase_a_eval.json` (same schema for Phase B; the filename is historical).

### 8.2 Aggregation

- Phase A aggregator: `fine_tuning_experiments/schema_exp/eval/aggregate_phase_a.py` → `schema_exp/phase_a_aggregate.json` + `schema_exp/phase_a_results.csv`.
- Phase B aggregator: `fine_tuning_experiments/phase_b/aggregate_phase_b.py` (to be added) → `phase_b/phase_b_aggregate.json` + `phase_b/phase_b_results.csv`.

### 8.3 Re-evaluation policy

The eval pipeline code is pinned (§4.6). Any modification bumps `EVAL_VERSION.txt`, which requires re-running the evaluation on every checkpoint that appears in the paper; the re-evaluation is recorded as a post-lock amendment in Appendix B.

---

## Part 9 — Statistical Analysis Plan

### 9.1 Three-tier comparison framework

To control multiple-testing inflation while preserving honest reporting of the full factorial:

- **Primary tier.** H1–H7 plus the enumerated pairwise tests within them. Approximately 21 primary comparisons (single schema): H1 (3 pairwise encoder contrasts) + H2 (1) + H3 (6 encoder × benchmark tests) + H4 (3 encoder contrasts) + H5 (2 TOST) + H6 (5 slope-against-null tests) + H7 (1 R_B decision). Correction: Benjamini–Hochberg FDR at q = 0.05 over all primary comparisons combined. H6's mechanism-stratified family (§9.4) contributes **one p-value per slope against the null β = 0** (five slopes: β_within, β_schema, β_encoder, β_config, β_combined_cell), included in the FDR family. The three-bin labelling (strong / moderate / weak / inconclusive per §7.2 H6) is reported **independently** of FDR status — it is a descriptive characterisation of the CI, not a null test.
- **Planned-secondary tier.** (a) Replications of primary comparisons in non-anchor cells (e.g., H4 also tested at PB × shared-multitask × LoRA); (b) **active-head macro-F1 replicas** of H1–H4 and H6 using the 5-active-head macro from §6.9.5. Total ~40 comparisons, pre-enumerated in `fine_tuning_experiments/phase_b/comparisons_inventory.csv`. Correction: Bonferroni at α = 0.05 within each hypothesis's secondary set.
- **Exploratory tier.** All other comparisons enabled by the factorial — higher-order interactions, per-head differences, per-family KB breakdowns, Phase A descriptive replicas of H1–H4 under S_flat and S_mech. No multiple-testing correction. Raw p-values reported with the label "exploratory; not corrected for multiple testing." No exploratory result is quoted in the Abstract or Conclusions.

### 9.2 Test sensitivity (paired-t + Wilcoxon dual reporting)

Every primary and planned-secondary comparison reports both paired-t and Wilcoxon signed-rank. Headline result is **paired-t** unless paired-t and Wilcoxon disagree at the chosen threshold (defined as one having p < 0.05 and the other p > 0.10), in which case the **Wilcoxon result is reported as headline** and a footnote flags the disagreement.

**Normality omnibus (standardised):** because per-cell Shapiro–Wilk on 72 cells at 5% false-positive rate would spuriously flag 3–4 cells, normality is checked once for the pooled z-standardised differences across all primary comparisons:

1. For each primary comparison *i*, compute paired seed-level differences $d_i[s]$ across seeds $s$.
2. Standardise: $z_i[s] = d_i[s]/\operatorname{SD}(d_i)$.
3. Pool all $z_i[s]$ into one vector (≈ 30 primary × 10 seeds = 300 values).
4. One Shapiro–Wilk test.

Decision rule: pooled-z Shapiro–Wilk p < 0.01 → headlines switch to Wilcoxon globally. Otherwise paired-t is the published headline and Wilcoxon remains the sensitivity check.

### 9.3 Precondition on coupling-slope CIs (reportability, not acceptance)

H6 is a mechanism-stratified coupling-slope characterisation (not a null test), so the CI-width check is a *reportability* gate, not an acceptance gate. It is applied **per slope** (β_within, β_schema, β_encoder, β_config, β_combined-cell).

**CI-width threshold = 0.30, per slope.** A slope whose 95 % CI width exceeds 0.30 on the natural slope scale (units of Δ`KB_hit_A` per Δ`BioRED_F1` = 1) is reported as **inconclusive** rather than mapped to strong / moderate / weak; the other slopes in the H6 family carry the narrative.

**Justification of the 0.30 threshold (not ad hoc).** The three-bin characterisation (§7.2 H6) has boundaries at |β| = 0.3 and |β| = 1.0, so its categories have widths 0.3 (weak) and 0.7 (moderate). A CI wider than 0.30 spans at least the width of the narrowest category (weak), so the estimate cannot cleanly place a slope inside weak vs inside moderate — the boundary it might cross is exactly the weak / moderate line at |β| = 0.3, which is the line that most directly governs the paper's "benchmark is a low-fidelity proxy" narrative. Anything coarser than 0.30 therefore cannot adjudicate the headline claim, so is reported as inconclusive rather than mis-labelled.

**ICC diagnostic (secondary, descriptive).** Alongside the slope CIs we report Cicchetti's (1994) one-way ICC of BioRED macro-F1 ex-NEG across Phase B cells:

$$\text{ICC} = \frac{\sigma^2_{\text{between cells}}}{\sigma^2_{\text{between cells}} + \sigma^2_{\text{within cells}}}$$

(`pingouin.intraclass_corr` or equivalent.) ICC < 0.30 does not block reporting — the CI-width check governs that — but is flagged as a context-setting diagnostic: "Phase B configuration variance was [adequate / modest / insufficient] to distinguish cells on the benchmark," used to explain wide β_config CIs if they occur.

Phase B-only Spearman ρ and Phase A-only Spearman ρ are additionally reported in all cases as a non-parametric transparency check, with Fisher-z 95 % CIs.

### 9.4 H6 fit specification — mechanism-stratified slopes

The RQ4 coupling question "how does BioRED macro-F1 predict `KB_hit_A_setvalued`?" does not admit a single answer, because different design mechanisms (schema choice, encoder choice, configuration choice, seed noise) produce BioRED variance by different pathways and can induce different slopes into KB. A single pooled mixed-effects slope therefore estimates a *mechanism-weighted average* whose interpretation depends on which mechanism happened to dominate x-variance in the data — in Phase A, that is schema; in Phase B, configuration — and those mechanisms need not yield the same coupling.

We therefore fit **five mechanism-stratified slopes**, each with a clear estimand. All are OLS (or OLS-with-cluster-robust-SE) on appropriately aggregated data; no single mixed-effects model is used for H6.

**(a) β_within — within-cell (seed-level) coupling.**
Estimand: "at fixed configuration, how much does seed-induced BioRED noise predict seed-induced KB noise?" For each of the 48 cells (12 Phase A × (schema × encoder) + 36 Phase B × (encoder × arch × update × schedule)), fit an OLS slope of `KB_hit_A` on `BioRED_F1` across the 10 seeds in that cell. Report:

- the seed-count-weighted mean slope $\bar\beta_{\text{within}} = (\sum_c n_c \hat\beta_{\text{within},c}) / \sum_c n_c$;
- the per-cell slope variance $\operatorname{Var}_c(\hat\beta_{\text{within},c})$;
- a paired 95 % CI on $\bar\beta_{\text{within}}$ via cluster-bootstrap (5,000 resamples, resampling whole cells).

Expected value: near zero, because within a cell the seed-to-seed variation in BioRED and in KB are dominated by approximately independent sources of randomness (different classifier-head initialisations, dropout noise, data-order perturbations), so the within-cell Pearson r is expected near zero and therefore β_within ≈ 0 *irrespective of the SD ratio*. (Note: β = r × SD_y / SD_x, so a large SD_y / SD_x ratio alone would *inflate* a noise-driven slope — what pins β_within near zero is r ≈ 0, not the SD ratio.)

**(b) β_schema — Phase A between-schema slope at fixed encoder.**
Estimand: "across S_flat / S_pair / S_mech at matched encoder, how much does schema-induced BioRED variation predict schema-induced KB variation?" For each encoder $e \in \{RB, PB, BL, PL\}$, compute three cell means $(\bar{x}_{e,s}, \bar{y}_{e,s})$ for $s \in \{\text{Sflat, Spair, Smech}\}$ (each a mean over 10 seeds) and fit OLS slope of $\bar{y}$ on $\bar{x}$ across the three schemas (n = 3 per encoder). Report the inverse-variance-weighted pooled slope across encoders with its 95 % CI (paired-bootstrap over encoder clusters, 5,000 resamples). This is a between-mechanism slope on the schema dimension, independent of encoder-mean level.

**(c) β_encoder — Phase A between-encoder slope at fixed schema.**
Estimand: "at fixed schema, how much does encoder-induced BioRED variation predict KB variation?" For each schema $s$, compute four cell means across encoders and fit OLS slope (n = 4 per schema). Report the inverse-variance-weighted pooled slope across schemas with its 95 % CI (paired-bootstrap over schema clusters).

**(d) β_config — Phase B between-config slope (S_pair only).**
Estimand: "under the selected schema (S_pair), how much does configuration-induced BioRED variation predict KB variation?" Compute the 36 Phase B cell means $(\bar{x}_c, \bar{y}_c)$ (10 seeds averaged per cell) and fit OLS slope across cells (n = 36). Report slope with 95 % Wald CI (each cell mean is itself based on 10 seeds so Wald on cell-mean observations is appropriate; as a sensitivity, also report cluster-bootstrap CI resampling whole cells with their 10-seed compositions).

**(e) β_combined_cell — pooled between-cell slope across phases.**
Estimand: "across all 48 cells in the study (mechanism-unspecified, worst-case most-pooled reading)." Compute the 48 cell means $(\bar{x}_c, \bar{y}_c)$ and fit OLS slope of $\bar{y}$ on $\bar{x}$ with a phase dummy (Phase_A = 0, Phase_B = 1) as a fixed covariate to absorb phase-level intercept differences. Report $\hat\beta_{\text{combined}}$ with 95 % CI and a phase-interaction test $H_0: \beta_A = \beta_B$ (Wald test on the slope × phase interaction). **This is the single most pooled estimate in the H6 family and is reported last, with the explicit caveat that it averages over mechanisms whose individual slopes may differ.** Phase-interaction decision tree:

- **Interaction CI excludes zero** → $\beta_A \ne \beta_B$ is established; β_combined_cell is **not** reported as a headline estimate and the narrative defers to the per-phase slopes (β_schema for Phase A, β_config for Phase B).
- **Interaction CI is inconclusive** (includes both zero and values of magnitude ≥ 0.5) → β_combined_cell is reported with the explicit caveat "mechanism-pooled across phases; the phase-interaction test was under-powered (12 Phase A cells give limited identification of β_A), so this slope may not generalise to pure-configuration regimes"; the narrative prefers β_config as the most practice-relevant slope.
- **Interaction CI cleanly includes zero and excludes ±0.3** → the single-slope reading is defensible; β_combined_cell is reported alongside the per-mechanism slopes as a pooled summary.

**All five slopes are additionally computed with the active-head macro-F1 (§6.9.5) in place of the full-schema macro-F1 as a planned-secondary (§9.1).**

**Implementation:** `statsmodels.OLS` for each regression; `scipy.stats.bootstrap` (or explicit resampling) for cluster-bootstrap CIs; `numpy.linalg` for the cell-mean aggregations. All fit code lives in `fine_tuning_experiments/phase_b/analysis/h6_coupling_slopes.py` (to be added). The fit is fully deterministic given a fixed bootstrap seed (`numpy.random.default_rng(20260416)`), which is also pre-registered.

**Why a mechanism-stratified slope family, not a single pooled slope.** Two single-model alternatives were considered and rejected before settling on the family decomposition:

1. A mixed-effects model `KB ~ BioRED + (1|phase) + (1|cell_id) + (1|seed_id)` estimates the fixed-effect slope from within-cell seed-level variation only. In Phase A that regime has BioRED SD ≈ 0.03 against KB SD ≈ 0.20 — noise-limited — so β̂ ≈ 0 would be recovered regardless of the real between-cell coupling. That β is β_within in the current decomposition, and its near-zero expectation is correctly diagnosed as "zero by construction" rather than as a scientific finding.
2. A fixed-intercept model `KB ~ BioRED + phase + BioRED × phase + (1|seed_id)` (dropping the cell intercept to let between-cell variation drive the slope) produces a mechanism-pooled between-observation slope whose interpretation depends on which design mechanism dominates x-variance in each phase (schema in Phase A, configuration in Phase B). Because those mechanisms need not yield the same coupling, a pooled β̂ averages over fundamentally different estimands; standard errors are additionally anti-conservative because the 10 seeds within a cell are not independent observations of the between-cell slope.

Neither single-model specification answers RQ4 cleanly. The mechanism-stratified decomposition above reports each β with its own estimand, its own CI, and its own three-bin label, letting "mechanism-dependence of coupling" be an empirical finding rather than an artefact of specification choice.

**Note on β_schema and β_encoder small-sample limitations.** The schema axis has 3 levels and the encoder axis has 4. An OLS slope fit to 3 or 4 cell means has, respectively, 1 or 2 residual degrees of freedom, so the per-cluster Wald CI is very wide and the slope estimate is numerically sensitive to any single cell mean. The inverse-variance-weighted pool across encoders (for β_schema) or schemas (for β_encoder) improves but does not rescue this: with 4 encoders (β_schema) or 3 schemas (β_encoder), the pooled effective sample size remains an order of magnitude below β_config's 36 cell means. We therefore report β_schema and β_encoder **as descriptive between-mechanism contrasts with cluster-bootstrap CIs as the primary interval estimate** (bootstrap is preferred over Wald here precisely because Wald CIs on n = 3 points are poorly calibrated) and **we do not expect either to resolve cleanly under the §9.3 CI-width gate of 0.30**. The statistically strongest mechanism-specific slope is β_config (n = 36 cells), and β_config accordingly carries the mechanism-specific evidence load at the Abstract level (see the H6 abstract-level claim mapping in §7.2). β_schema and β_encoder are reported as orientation for readers asking "is schema or encoder the dominant coupling mechanism in Phase A?" and are diagnostic, not confirmatory.

**Reported outputs per KB metric (three metrics: KB_hit_A, KB_pmass_B, KB_auc_C, all `set_valued`):**

- β_within: $\bar\beta_{\text{within}}$ with cluster-bootstrap CI; per-cell slope distribution histogram.
- β_schema: inverse-variance-weighted pooled slope with bootstrap CI.
- β_encoder: inverse-variance-weighted pooled slope with bootstrap CI.
- β_config: slope with Wald + cluster-bootstrap CI.
- β_combined_cell: slope with Wald CI and phase-interaction test.
- Phase A-only and Phase B-only Spearman ρ (Fisher-z CIs, descriptive).
- Full re-fit with active-head macro-F1 as benchmark side (planned secondary).

### 9.5 Variance decomposition for H7

For Phase B (n = 360 seed-level observations, S_pair only), decompose variance of BioRED macro-F1 ex-NEG and of `KB_hit_A_setvalued` across (encoder, architecture, update regime, schedule, encoder×schedule, encoder×update, seed). For Phase A (n = 120 seed-level observations, three schemas), decompose variance across (schema, encoder, schema×encoder, seed). Report variance share per factor per phase. Compute

$$R_A = \frac{\text{(schema + encoder + schema×encoder) share in BioRED\_F1}}{\text{same share in KB\_hit\_A}}, \qquad R_B = \frac{\text{(encoder + arch + update + schedule + 2-way interactions) share in BioRED\_F1}}{\text{same share in KB\_hit\_A}}.$$

R_A is reported **descriptively**: Phase A data are already observed (§6.9.3 gives R_A ≈ 2.29 under the design-lever-share formula, and ≈ 3.16 under the schema-only narrowing), and no pre-committed threshold is applied to an observed value. R_B is the confirmatory statistic for H7; the pre-committed threshold R_B ≥ 2 (§7.2 H7) is independent of R_A and is justified from first principles (R = 1 is a perfect-proxy reference, and R ≥ 2 corresponds to a 2× disparity between benchmark and KB variance absorption under the same design lever — a commonly used "meaningful asymmetry" cut in variance-share ratio literature). A non-parametric CI on R_B is obtained by cluster-bootstrapping whole cells (5,000 resamples) and re-computing both numerator and denominator shares per bootstrap; the report is $\hat R_B$ + 95 % bootstrap CI + the R_B ≥ 2 decision.

### 9.6 Power

Phase A within-cell SDs (mean over the four encoders at each schema; from the 120-run factorial): BioRED ex-NEG SD ≈ 0.028 (S_flat) / 0.032 (S_pair) / 0.023 (S_mech); BC5CDR DD SD ≈ 0.096 (S_flat) / 0.031 (S_pair) / 0.043 (S_mech); KB_hit_A_setvalued SD ≈ 0.155 (S_pair mean within-cell). Phase B is run under S_pair only with 10 paired seeds per cell; H1–H5 are paired comparisons across cells (seed-paired). With α = 0.05 and power = 0.80:

| Metric | Within-cell SD (pooled S_pair) | Detectable paired Δ (Cohen's d ≈ 0.89) |
|---|---|---|
| BioRED macro-F1 ex-NEG | 0.098 | 0.087 |
| BioRED active-head macro-F1 | ≈ 0.08 (re-estimate post-Phase B) | 0.07 |
| BC5CDR DRUG_DISEASE F1 | 0.078 | 0.07 |
| KB_hit_A_setvalued | 0.191 | 0.17 |
| KB_pmass_B_setvalued | 0.124 | 0.11 |
| KB_auc_C_setvalued | 0.120 | 0.11 |

The S_pair within-cell SD on BioRED ex-NEG is larger than earlier estimates because the pool includes encoder-induced variance; the per-cell (fixed encoder) SD is smaller. H1–H5 will be powered against the per-cell SD, which is expected to be ≈ 0.03–0.05 based on Phase A single-cell variance. H7's variance decomposition uses 360 observations — high-power for variance partitioning.

**H6 power varies by slope** (under the mechanism-stratified specification of §9.4; expected 95 % CI widths are rough Monte-Carlo / Phase-A-derived estimates under baseline variance conditions, not guarantees):

| Slope | n (unit of analysis) | Expected CI width | Likely relation to the 0.30 gate |
|---|---|---|---|
| β_within | 48 cells (cluster-bootstrap over whole cells) | 0.20–0.35, driven by observed per-cell slope dispersion (Phase A suggests ≈ 0.3–0.5 SD across cells) | Borderline; may pass or just miss the gate |
| β_schema | 4 encoder-specific OLS slopes, each on n = 3 schema means; inverse-variance-weighted pool | 0.5–1.0 (wide due to n = 3; see §9.4 small-sample note) | Expected to **trip** the gate → reported descriptively |
| β_encoder | 3 schema-specific OLS slopes, each on n = 4 encoder means; inverse-variance-weighted pool | 0.4–0.8 | Likely to trip the gate → reported descriptively |
| β_config | 36 Phase B cell means (Wald + cluster-bootstrap) | 0.15–0.25 under baseline variance, 0.3–0.5 under high-noise regimes (LoRA or frozen encoders dominating the factorial) | Most likely to **cleanly resolve**; carries the Abstract-level coupling evidence (see H6 abstract-level claim mapping, §7.2) |
| β_combined_cell | 48 cell means + phase dummy | 0.10–0.20 | Narrowest CI, but its interpretability is conditional on the phase-interaction test (§9.4 (e)) |

So β_config is the Phase-B-sensitive slope that the 0.30 CI-width gate is actually designed to adjudicate; β_schema and β_encoder are orientation statistics that we expect to be wide, and §9.4's small-sample note is the pre-committed acknowledgement of that fact. β_within depends on empirical between-cell dispersion of the per-cell slope estimates — Phase A's pilot fit (Step 1 of the lock plan) will tighten this estimate before Phase B is submitted.

**Note on the "Expected CI width" column.** The expected-CI-width estimates in the H6 power table use Phase A **within-cell standard deviations** (the scalar SD summaries already reported publicly in §6.9.1) as sample-size calibration inputs. They are not β̂ estimates; they are the width of the sampling distribution of a slope under the design's n and observed noise, which is what any honest power analysis must use. The pre-registration lock contains no numerical β̂ values in §7.2 H6's claim-mapping table or §9.4's slope specification. This is the same rule the power analyses for H1–H5 already follow (their SDs are also Phase-A-derived scalar summaries), extended to H6 with its more complex per-slope structure.

---

## Part 10 — Paper Structure

### 10.1 Section layout (Bioinformatics format)

1. **Abstract** (250 words). Motivation (oncology gap + unvalidated benchmark proxy) → methods (heterogeneous supervision + schema factorial + KB audit) → central finding (variance asymmetry + mechanism-stratified coupling characterisation per the §7.2 H6 Abstract-level claim mapping + ordinal instability, quantified) → implication (contingent on β_config's three-bin label at the time of drafting; the four admissible Abstract-level framings are pre-committed in §7.2). The currently anticipated framing (β_config weak → "benchmark rank is a low-fidelity guide to KB surfacing within a fixed schema") is not locked; the final wording is chosen from the pre-committed table after Phase B unblinds.
2. **Introduction.** Oncology assertion extraction gap → heterogeneous supervision → evaluation-validity question → contributions (matching §1.3).
3. **Methods.**
   - §2.1 Data and schemas (three schemas, supervision stages, leakage remediation, MeSH C04 projection, limitations).
   - §2.2 Training framework (trainer spec §5; Phase A schema pilot + Phase B configuration factorial §6/§7).
   - §2.3 Evaluation (BioRED/BC5CDR tests; KB audit; three correctness-aware metrics §4; active-head macro-F1 secondary).
   - §2.4 Statistical analysis (three-tier framework §9.1; variance decomposition §9.5; mechanism-stratified coupling slope family §9.4 with the per-slope CI-width gate from §9.3 disclosed).
4. **Results.**
   - §3.1 Schema selection (Phase A pilot; §6.9).
   - §3.2 Main benchmark results under S_pair (Phase B; H1–H5).
   - §3.3 KB-surface yield (Phase B; per-family breakdown; active-head-macro sensitivity).
   - §3.4 Evaluation-validity audit: H7 variance asymmetry as headline; H6 mechanism-stratified coupling slope family with per-slope CIs, three-bin labels, and per-slope CI-width-gate disclosure; ordinal-instability summary.
5. **Discussion.** How to read a benchmark-leaderboard ranking when downstream utility is the goal; why variance asymmetry implies clinical evaluators should not select on F1 alone; W1–W8 weaknesses and responses; limitations including the positive (not negative) coupling finding.
6. **Conclusion.**

### 10.2 Figure plan (5 main figures; within *Bioinformatics* original-research guidance of 4–6)

| Figure | Content |
|--------|---------|
| **F1** | Supervision pipeline schematic: T1 → T2 → T3/T4, with schema labels and corpus provenance. |
| **F2** | Phase B main results: BioRED ex-NEG and BC5CDR DD F1 by encoder × schedule under S_pair. Mean ± 95% CI across 10 seeds. Anchor cell highlighted. Phase A S_flat and S_mech rows inset as "schema-variance context". |
| **F3** | Per-family KB surfacing yield: per-family `KB_hit_A_setvalued` for the anchor and encoder-edge configurations, S_pair, with CIViC baseline overlay. |
| **F4** | RQ4 evaluation-validity audit — variance-asymmetry headline. (a) Bar chart of variance share (schema/encoder in Phase A; encoder/arch/update/schedule in Phase B) for BioRED ex-NEG vs `KB_hit_A` vs BC5CDR DD — visualises H7. (b) Ordinal-instability summary: histogram of ΔKB_hit_A between configurations that are within 1 × within-cell BioRED SD of each other — visualises the practical consequence of H7 for model selection. |
| **F5** | RQ4 evaluation-validity audit — mechanism-stratified coupling characterisation. (a) Forest plot of the five H6 slopes (β_within, β_schema, β_encoder, β_config, β_combined_cell) with 95 % CIs and the three-bin background shading (weak / moderate / strong) — shows that coupling is mechanism-dependent, not a single number. (b) Cell-level scatter of BioRED ex-NEG × `KB_hit_A` across Phase A + Phase B cells (cell means with seed-SD whiskers), coloured by phase, with the β_config and β_schema slope lines overlaid. |

### 10.3 Table plan (3 main tables)

| Table | Content |
|-------|---------|
| **T1** | Data and schema inventory: T1/T2/T3/T4 counts; per-schema label list; active-head support per schema; BioRED test support per head. |
| **T2** | Phase B main benchmark results under S_pair: one row per cell (36 cells), columns for BioRED ex-NEG, BioRED active-head macro, BC5CDR DD, `KB_hit_A_setvalued`, `KB_pmass_B`, `KB_auc_C`; mean ± SE over 10 seeds. Plus the RB reference row. |
| **T3** | Evaluation-validity audit summary: H7 variance-decomposition table (rows: metric; cols: schema / encoder / arch / update / schedule / interactions / seed), with R_A (descriptive) and R_B (confirmatory, with pre-committed threshold and bootstrap CI); H6 mechanism-stratified slope table (rows: β_within, β_schema, β_encoder, β_config, β_combined_cell; cols: KB_hit_A_sv, KB_pmass_B_sv, KB_auc_C_sv, with point estimate, 95 % CI, CI-width-gate status, three-bin label, active-head secondary re-fit). |

---

## Part 11 — Known Weaknesses and Pre-planned Responses

### W1 — DrugProt official test absent

DrugProt's processed package does not include the official test split. We state this explicitly in Methods and do not report DrugProt external evaluation. DrugProt contributes as T1 training data only.

### W2 — Absolute BioRED macro-F1 is low (≈ 0.28)

BioRED test has extreme class imbalance under the coarse schemas (DRUG_GENE_REGULATION and VARIANT_GENE together ≈ 2% of test relations). Even a perfect ASSOCIATION_GENERAL + NEGATIVE classifier ceilings at macro-F1 = 0.50. The Methods report random (0.25) and majority (0.20) baselines; the Discussion frames observed 0.27–0.35 as substantial gains over random under imbalance, not total failure.

### W3 — BC5CDR per-seed SDs are moderate (0.02–0.06)

We report mean ± SE (not SD) throughout, annotate per-seed variability in Supplementary, and use paired tests across seeds so that cross-seed noise is differenced out.

### W4 — CIViC-anchored KB recall is biased toward CIViC coverage

KB_hit_A measures recall-among-CIViC; out-of-CIViC assertions are not counted. Stated explicitly in Methods: the KB audit is a lower bound on surfacing yield with respect to the CIViC oncology KB, not a universal downstream-utility measurement.

### W5 — n = 165 KB targets is small

Per-target surfacing is reported with Wilson binomial CIs; per-family breakdown (n ≈ 20–90 per family) is reported descriptively, not as primary significance evidence.

### W6 — No human audit validation of KB-consistency labels

The schema-expected-label mapping has been peer-reviewed informally and documented in `schema_expected_label_mapping_rationale.md`. A human re-audit is listed as future work. The set_valued / single_label projection-mode duality is specifically designed to bracket the mapping's uncertainty.

### W7 — No LLM baseline

A full LLM paradigm comparison is a separate study. Phase B+ (§7.7) will address this with LLaMA-3-8B-Instruct via LoRA + zero-shot reference on the same S_pair × T1→T2 anchor; results deferred. The current paper restricts itself to encoder-based RE and states so explicitly in the Discussion.

### W8 — Benchmark and KB metrics are positively correlated

The Phase A 120-run data under `KB_hit_A_setvalued` show a positive seed-level coupling between BioRED macro-F1 ex-NEG and KB_hit_A (Pearson r = +0.52 [+0.40, +0.64]; §6.9.4). The central finding is therefore framed as variance asymmetry (§7.2 H7) and ordinal instability (Figure 4), which are compatible with positive coupling and are the aspects directly relevant to model-selection practice.

---

## Appendix A — Code layout and file roles

```
project_1/
├── data_pipeline/                             # leakage fix, MeSH C04 projection, shard generation
│   ├── 01_filter_t1_t2_leakage.py
│   ├── 03_rederive_t2_mesh.py
│   └── reports/
├── training_data_generation/                  # canonical shard files (scratch: /lus/.../processed/)
├── schema_exploration/
│   └── definitions/schema_definitions.py      # S_flat / S_pair / S_mech projection functions
├── oncology_projection/                       # MeSH C04 audit + manifests
├── knowledge_grounded_evidence_audit/
│   ├── data/                                  # CIViC 165 targets, PubMed abstract cache
│   └── inference/predict_checkpoint.py        # checkpoint loader used by eval_one_run
├── external_evaluation/                       # BioRED + BC5CDR test loaders and pair builders
├── fine_tuning_experiments/
│   ├── train/run_experiment.py                # shared entry point (Phase A + Phase B)
│   ├── phase_b/
│   │   └── trainer/
│   │       ├── scientific_trainer.py          # training loop, stage dispatch
│   │       ├── scientific_data.py             # shard loading, label space, negative sampling
│   │       ├── minimal_trainer.py             # NotImplementedError shim
│   │       ├── tests/test_label_space.py      # schema regression test
│   │       └── trainer_inventory/             # API contracts (reference docs)
│   ├── schema_exp/                            # Phase A
│   │   ├── generate_phase_a_configs.py        # one-shot config generator (120 cells)
│   │   ├── configs/PA_*.yaml                  # 120 configs (4 encoders × 3 schemas × 10 seeds)
│   │   ├── sbatch/                            # per-encoder array + smoke
│   │   ├── submit_phase_a.sh
│   │   ├── build_runs_inventory.py            # → runs_inventory.csv (SHA-256 of best.pt, 120 rows)
│   │   ├── runs_inventory.csv
│   │   ├── phase_a_{results.csv, aggregate.json, schema_selection.json, report.md}
│   │   ├── analysis/
│   │   │   ├── phase_a_analysis.py            # scientific analysis bundle (§6.9)
│   │   │   └── phase_a_analysis.{json,md}
│   │   └── eval/                              # pinned v1.0 eval pipeline
│   │       ├── EVAL_VERSION.txt               # "1.0" — enforced read gate
│   │       ├── eval_one_run.py                # per-run eval (writes phase_a_eval.json)
│   │       ├── aggregate_phase_a.py           # 120-run aggregate + §6.6 selection
│   │       ├── prepare_eval_inputs.py
│   │       ├── schema_expected_label.py
│   │       ├── sbatch/phase_a_eval.sbatch     # re-eval array (used only if eval code changes)
│   │       └── inputs/                        # biored_test_pairs_{Sflat,Spair,Smech}.jsonl, bc5cdr_test_pairs.jsonl, kb_surface_pairs.jsonl
│   └── runs/                                  # on scratch: /lus/.../fine_tuning_experiments/runs/
│       ├── schema_exp/                        # Phase A run artifacts (120 runs)
│       └── phase_b/                           # Phase B run artifacts (populated after lock)
└── paper_development_design.md                # this document
```

Shard files and run artifacts live on the Lustre scratch filesystem (`/lus/lfs1aip2/projects/b5ac/project_1/...`) because of their size; the repository tracks code only.

---

## Appendix B — Post-lock amendment log

Amendments to this document after the pre-registration commit are logged below with date, trigger, affected sections, and replacement rationale. No entries prior to the lock are recorded: the lock commit is the single canonical reference for methodology, hypotheses, data, and analysis plan.

| Date | Trigger | Section(s) | Rationale |
|---|---|---|---|

*(This table is empty at lock. Post-lock amendments are appended as the project progresses.)*
