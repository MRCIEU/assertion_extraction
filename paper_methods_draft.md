# Paper Methods Draft
## Heterogeneous Supervision and Evaluation Validity for Cancer Assertion Extraction

**Document role.** This file is the paper-ready scientific skeleton for the
Methods and Results sections. It is derived from
`paper_development_design_locked_v1.md` (the pre-registration lock; frozen,
not to be edited) and from `paper_development_design.md` (the post-lock
amendment log). Both source documents are retained verbatim as the audit
trail. This draft contains only content that a reader of the published
paper needs to follow the science; project-process material (debugging
chains, infrastructure incidents, attempt-by-attempt narratives) lives in
the source documents, not here.

**Target venue.** *Bioinformatics* (Oxford Academic) — Original Research.
**Alternatives.** *Briefings in Bioinformatics*, *NAR Genomics and Bioinformatics*.

---

## Section 1 — Paper Position

Cancer-focused relation and assertion extraction differs from general
biomedical relation extraction in two underappreciated ways. First, no
single public corpus provides sufficient span-supervised coverage of
oncology assertion semantics; practitioners therefore compose
heterogeneous supervision from general biomedical relation corpora (BioRED,
DrugProt, BC5CDR), curated knowledge-base priors (CIViC), and weakly
labelled mining (CIViCmine, CancerMine). Second, public benchmark metrics
such as BioRED macro-F1 and BC5CDR drug–disease F1 are routinely treated
as proxies for downstream clinical-informatics utility, despite the
absence of any empirical validation of that proxy relationship in the
oncology setting.

This paper addresses both gaps. We define a schema-aware multi-stage
training framework that explicitly separates gold-span (T1), oncology-
projected (T2), weak-prior (T3), and unlabeled-adaptation (T4)
supervision; we train a systematic factorial of encoder × architecture ×
update-regime × schedule configurations against held-out external
benchmarks; and we evaluate every checkpoint in a knowledge-grounded
audit pipeline anchored to 165 CIViC-curated oncology evidence targets.

Our central empirical finding is that benchmark and KB-audit metrics
are positively correlated but exhibit pronounced variance asymmetry:
schema-level design choices that explain 60 % of BioRED variance
explain only 19 % of KB-audit variance — a 3× disparity that produces
ordinal instability between the two metric orderings, so
configurations within typical seed noise on BioRED can swap rank on
the KB audit. We characterise this asymmetry through a family of
mechanism-stratified coupling slopes and a pre-registered variance-
decomposition test.

We organise the work around four research questions:
**RQ1** how cancer-focused assertion extraction should be operationalised
given the corpus and schema landscape;
**RQ2** which training configurations best support generalisation from
internal development to held-out external evaluation;
**RQ3** how model family and audit formulation jointly affect KB
surfacing yield;
**RQ4** how strongly, and in which direction, benchmark metrics predict
KB surfacing across schema and configuration variation.

The four contributions are: a principled mapping from heterogeneous
public corpora to three oncology relation schemas at three granularity
levels with verifiable integrity checks; a multi-seed factorial of
encoder, architecture, update regime, and schedule under formal paired
significance tests; a reproducible KB-grounded audit using 165
CIViC-anchored targets evaluated under three correctness-aware metrics;
and an evaluation-validity audit that decomposes variance contributions
across design levers, reports a family of mechanism-stratified coupling
slopes, and quantifies ordinal rank instability between benchmark and KB
metrics.

---

## Section 2 — Data

### 2.1 Source corpora

| Corpus | Role | Counts after leakage fix |
|---|---|---|
| BioRED | Gold span RE (entity-typed, multi-relation) | 500 train/dev docs, 100 test docs |
| DrugProt | Gold span RE (chemical–protein mechanisms) | 4,250 train/dev docs |
| BC5CDR | Gold span RE (chemical–disease binary) | 1,000 train/val docs, 500 test docs |
| CIViC | KB audit anchor (curated oncology assertions) | 165 evidence-supported targets |
| CIViCmine | Weak sentence mining (T3) | ≥ 415k weak rows |
| CancerMine | Weak prior labels (T3) | bundled with CIViCmine |
| Oncology PubMed abstracts | T4 unlabeled adaptation | 9,463 abstracts |

### 2.2 Oncology projection (T2) and the MeSH C04 tree

T2 is the oncology-facing subset used as the second training stage. C04
refers to the *Neoplasms* branch of the U.S. National Library of
Medicine's Medical Subject Headings (MeSH) tree, an indexed
classification curated for every PubMed record. The operational
criterion is document-level: a document enters T2 if its PubMed record
carries at least one MeSH descriptor under C04.

```
For each PMID in T1:
    fetch MeSH terms via NCBI E-utilities (efetch, db=pubmed, rettype=xml)
    include the document in T2 if any descriptor falls under C04.*
```

This yields 733 T2 documents (BioRED 94, BC5CDR 114, DrugProt 525) and
6,969 gold relations; every T2 document is a strict subset of the
leakage-fixed T1 shards.

Two limitations of the construction must be reported explicitly in
Methods.

*Limitation 1 — document-level, not relation-level.* A document is
admitted to T2 because the abstract is indexed as oncology-relevant, but
each gold relation inside that abstract enters T2 unmodified. A concrete
case: a PubMed abstract describing BRAF inhibition in melanoma may also
mention that some patients took aspirin for unrelated cardiovascular
prophylaxis; the aspirin–disease relation enters T2 even though it is
not an oncology assertion. T2 therefore provides *oncology context*
supervision, not relation-level oncology gold.

*Limitation 2 — DrugProt T2 is mechanism-in-oncology-context.* DrugProt
relations are chemical–protein mechanism labels (`INHIBITOR`,
`ACTIVATOR`, `SUBSTRATE`, …) annotated independently of cancer relevance.
After C04 filtering, DrugProt T2 contains chemical-protein mechanism
labels that *happen to occur in abstracts whose MeSH index lists at
least one neoplasms term*; it does not contain oncology-specific
assertion labels. A DGR_INHIBIT relation in a DrugProt T2 row indicates
"this chemical inhibits this protein, in an abstract that also discusses
cancer", not "this chemical inhibits this protein *in a cancer
mechanism*".

Both limitations motivate the paper's RQ1 argument that no single public
resource suffices for cancer assertion extraction, and they are stated
verbatim in the Methods section.

### 2.3 Weak (T3) and unlabeled (T4) supervision — disabled in main factorial

T3 (CIViC priors, CIViCmine, CancerMine) and T4 (9,463 unlabeled
oncology abstracts) are reserved for optional auxiliary objectives. The
main Phase A and Phase B factorials disable both
(`lambda_auxiliary = lambda_distill = 0`, `T4_mode = none`); reporting in
this paper is therefore confined to T1+T2 supervision.

### 2.4 Frozen shard manifests

The canonical shards live under
`/lus/lfs1aip2/projects/b5ac/project_1/training_data_generation/data/processed/`;
SHA-256 checksums recorded in `t1_manifest.json` are re-verified at the
start of every training run, so any silent shard mutation produces an
immediate run-time failure rather than a silently inconsistent dataset.

---

## Section 3 — Schema Design

### 3.1 Three schema candidates

We compare three oncology relation schemas at three granularity levels.
All three are exercised in Phase A; Phase B uses the schema selected by
Phase A's pre-committed selection rule (see §6.6).

**S_flat — 4 labels.** Family-level discrimination:
`ASSOCIATION_GENERAL`, `DRUG_DISEASE`, `DRUG_GENE_REGULATION`,
`__NEGATIVE__`. A BioRED variant–disease association maps to
`ASSOCIATION_GENERAL`; a BC5CDR drug–disease CID maps to `DRUG_DISEASE`;
all DrugProt chemical–protein mechanisms collapse to
`DRUG_GENE_REGULATION`.

**S_pair — 8 labels.** Entity-pair-type discrimination: `GENE_DISEASE`,
`VARIANT_DISEASE`, `DRUG_DISEASE`, `DRUG_GENE_REGULATION`,
`GENE_GENE_ASSOC`, `DRUG_VARIANT_ASSOC`, `ASSOCIATION_GENERAL`
(catch-all), `__NEGATIVE__`.

**S_mech — 13 labels.** S_pair plus five DRUG_GENE_REGULATION sub-
mechanisms — `DGR_INHIBIT`, `DGR_ACTIVATE`, `DGR_METABOLIC`,
`DGR_REGULATE`, `DGR_STRUCTURAL` — defined according to the DrugProt
annotation guidelines (Miranda-Escalada et al., 2021).

#### 3.1.1 Why three schemas, and why these three

The three schemas are deliberately chosen to make the design space
*nested* and to span three distinct scientific questions, rather than to
cover a continuous resolution sweep:

- *Lower bound (S_flat).* A four-label schema is the minimum that still
  preserves the basic clinical distinctions a downstream user cares
  about: drug–disease causality versus drug–gene regulation versus other
  associations. A coarser schema (e.g. binary "associated / not") would
  collapse the very labels the KB audit must discriminate, defeating the
  purpose of the surfacing comparison.
- *Middle (S_pair).* Adding entity-pair-type heads to S_flat captures
  the most consequential ambiguity in BioRED: many gold relations are
  natively annotated by entity pair (gene–disease, variant–disease,
  gene–gene, etc.) rather than by mechanism, so an entity-pair head
  recovers structure that S_flat throws away.
- *Upper bound (S_mech).* Splitting `DRUG_GENE_REGULATION` into five
  mechanism heads tests whether mechanism resolution is *supportable on
  the present evaluation corpus*. The five sub-mechanisms (`INHIBIT`,
  `ACTIVATE`, `METABOLIC`, `REGULATE`, `STRUCTURAL`) are the canonical
  partition of chemical–protein mechanism in the DrugProt annotation
  guidelines (Miranda-Escalada et al., 2021); going finer would require
  guideline categories that DrugProt itself does not annotate
  consistently. *Post-hoc confirmation.* The empirical eval landscape
  reinforces the same upper bound: at thirteen labels, five S_mech
  heads receive zero BioRED test positives (§6.9.5), so any further
  split would produce a schema in which the external benchmark cannot
  disambiguate the additional heads.

The three schemas therefore form a hierarchy: S_flat → S_pair adds
entity-pair-type discrimination; S_pair → S_mech adds mechanism
discrimination. Each successive refinement asks a single, isolable
empirical question, and each can be compared against the others in a
common evaluation frame (§3.4).

### 3.2 Label-space derivation and regression test

The classifier head is sized by `derive_label_space(shard_paths,
pair_type_filter)` (`fine_tuning_experiments/phase_b/trainer/
scientific_data.py`). Labels are enumerated *before* applying
`pair_type_filter`, so a label is retained as long as some gold relation
somewhere carries it, even if every gold instance has entity endpoints
outside the legal-pair filter. Without this rule, S_pair would silently
drop `ASSOCIATION_GENERAL` to a 7-class head (because all S_pair AG rows
have illegal endpoints under `spair_legal_endpoints`), diverging from
the schema definition.
A regression test (`tests/test_label_space.py`) asserts the expected
counts (4 / 8 / 13) and the presence of `ASSOCIATION_GENERAL` across all
three schemas; the test runs in CI and on every release.

### 3.3 Pair-type filters

A relation is eligible for training and evaluation only if its
`(head_label, tail_label)` is in the schema's legal-pair filter.
Negatives are sampled from the *same* filter, which guarantees positive
and negative samples have matching distributional support and prevents
the negative pool from leaking into endpoint pairs that the positives
cannot inhabit.

| Filter | Legal pairs |
|---|---|
| `sflat_legal_endpoints` | GENE↔DISEASE, GENE↔DRUG, DRUG↔DISEASE, GENE-GENE, VARIANT↔DISEASE, VARIANT↔GENE, DRUG↔VARIANT, VARIANT-VARIANT |
| `spair_legal_endpoints` | same as S_flat **minus** VARIANT↔GENE and VARIANT-VARIANT |
| `smech_legal_endpoints` | same as S_pair |

### 3.4 Schema-expected-label mapping for the KB audit

Each CIViC audit target carries a pairing family (gene–drug, variant–
disease, etc.) and a heuristic gold label drawn from the curated
evidence. For each `(family, gold, schema, projection_mode)` tuple the
function `schema_expected_label_set()` returns the set of schema-
vocabulary labels that count as a correct prediction. The mapping
projects the CIViC ground truth *up* into each schema's own vocabulary,
so each schema is evaluated in its own frame and cross-schema
comparisons are fair.

| CIViC family × heuristic gold | S_flat | S_pair | S_mech (set_valued) | S_mech (single_label) |
|---|---|---|---|---|
| gene_drug × DGR (n=90) | `{DGR}` | `{DGR}` | `{DGR, DGR_INHIBIT, DGR_ACTIVATE, DGR_METABOLIC, DGR_REGULATE, DGR_STRUCTURAL}` | `{DGR}` |
| gene_drug × AG (n=64) | `{DGR}` | `{DGR}` | same | `{DGR}` |
| variant_disease × AG (n=8) | `{AG}` | `{VARIANT_DISEASE}` | `{VARIANT_DISEASE}` | `{VARIANT_DISEASE}` |
| variant_disease × VARIANT_GENE (n=3) | unmapped | unmapped | unmapped | unmapped |

The 3 `variant_disease × VARIANT_GENE` targets are unmapped under all
three schemas (no schema vocabulary admits a variant–gene pair as the
covering set for a variant–disease curated assertion) and are
**excluded from the evaluable target pool**. KB metrics (KB_hit_A,
KB_pmass_B, KB_auc_C) are therefore computed over n = 162 effective
targets per run; the headline "165" refers to the curated CIViC anchor
set, while the unmapped 3 are reported as a structural coverage gap
rather than as miss predictions.

#### 3.4.1 Projection modes — a worked example

The curated CIViC evidence for a gene–drug pair typically records the
*family* of the relation (e.g. "DGR — drug–gene regulation") but not the
specific sub-mechanism (e.g. "INHIBIT" vs "ACTIVATE"). When an S_mech
model predicts a sub-mechanism, the KB-audit must decide whether that
sub-mechanism counts as a correct prediction of the CIViC DGR ground
truth. Two projection modes formalise the two reasonable answers:

- **set_valued** — any label in the covering set counts as correct.
  Friendly to fine-grained schemas: a DGR_INHIBIT prediction against a
  CIViC DGR ground truth is treated as a hit.
- **single_label** — only the canonical catch-all head counts. Strict
  against fine-grained schemas: only a literal `DRUG_GENE_REGULATION`
  prediction counts, and a DGR_INHIBIT prediction is treated as a miss.

A concrete example for an S_mech model on a CIViC gene–drug × DGR target:

| Model prediction | set_valued correct? | single_label correct? |
|---|---|---|
| `DRUG_GENE_REGULATION` | yes | yes |
| `DGR_INHIBIT` | yes | **no** |
| `DGR_ACTIVATE` | yes | **no** |
| `GENE_DISEASE` | no | no |
| `__NEGATIVE__` | no | no |

For S_flat and S_pair, the family-to-label mapping is one-to-one (each
gold target maps to a single schema label), so the two projection modes
yield numerically identical KB metrics. The set_valued / single_label
distinction is therefore *consequential only for S_mech*, where it
brackets the schema-mapping uncertainty introduced by the absence of
sub-mechanism annotations in CIViC.

Neither mode is uniquely correct: set_valued treats CIViC's absence of
sub-mechanism annotations as "any sub-mechanism counts"; single_label
treats it as "only the family-level catch-all counts". Reporting both
modes for S_mech, and noting their numerical equivalence on S_flat and
S_pair, brackets the schema-mapping uncertainty without requiring a
normative choice.

---

## Section 4 — Evaluation Framework

### 4.1 BioRED test

Official BioRED test split (100 documents) with relations projected into
each schema's label vocabulary.
**BioRED macro-F1 (ex-NEG)** is the primary benchmark metric, computed
as macro-F1 over non-negative labels only, averaged over labels that
appear in at least one gold instance of the schema's test set
(`sklearn.metrics.f1_score` with `average="macro"`). The in-training
dev metric uses the same convention so that early-stopping and external
evaluation are directly comparable. Per-head F1 is reported in the
supplementary material to expose dead heads and low-support heads.

### 4.2 BC5CDR test

Official BC5CDR test split (500 documents), binary classification of
DRUG–DISEASE pairs. Metric: **BC5CDR DRUG_DISEASE F1**
(`sklearn.metrics.f1_score` with `pos_label="DRUG_DISEASE"`). The binary
random baseline is 0.50; a model that improves over 0.50 by Δ > 0.05
exhibits measurable drug–disease discrimination.

### 4.3 CIViC KB-surface audit

Across 165 CIViC-anchored targets — of which **162 map to the schema
vocabularies** under §3.4 and **3 are unmapped and excluded from the
evaluable pool** as a structural coverage gap — each target carries a
PubMed abstract and two pre-identified entity spans. The model
produces a full softmax (and pre-softmax logits) over the schema
vocabulary. Spans are *given*, so the audit measures relation
discrimination only, not entity detection; this isolates the relation
classifier from upstream NER errors. The eval pipeline emits
`n_targets_total = 165` and `n_targets_evaluable = 162` per run; all
KB_hit_A / KB_pmass_B / KB_auc_C numbers reported in §6.9 and §8 are
means over the 162 evaluable targets. Audit artifacts:
`kb_surface_pairs.jsonl` (165 targets) and `kb_surface_targets.jsonl`
(one row per target × run, with an `evaluable` flag).

### 4.4 Three correctness-aware KB metrics

Each metric is reported under both projection modes where applicable
(see §3.4); for S_flat and S_pair the two modes coincide.

**Method A — argmax hit (primary).**
$$\text{KB\_hit\_A}(\text{schema}, \text{mode}) = \text{mean}_{\text{evaluable targets}}\; \mathbb{1}\!\bigl[\arg\max_L P(L) \in \mathcal{E}(\text{target})\bigr].$$
Computable from `pred_label` only; no logits needed.

**Method B — expected-label probability mass (sensitivity, calibration-robust).**
$$\text{KB\_pmass\_B} = \text{mean}_{\text{evaluable targets}}\; \sum_{L \in \mathcal{E}} P(L).$$
Requires the full softmax. Treats argmax ties and near-ties symmetrically.

**Method C — AUC of abstention–recall curve (sensitivity, threshold-free surfacing).**
On a 21-point grid of abstention thresholds $\tau \in \{0.00, 0.05, \dots, 1.00\}$:
$$\text{reject\_rate}(\tau) = \tfrac{|\{P(\text{NEG}) > \tau\}|}{n_\text{eval}}, \quad \text{precision\_kept}(\tau) = \text{KB\_hit\_A on non-rejected}.$$
$\text{KB\_auc\_C}$ is the trapezoidal integral of `precision_kept` over
`reject_rate`. The 21-point grid is pre-specified; finer grids and
higher-order integration (Simpson, Gauss–Legendre) are explicitly *not*
used.

#### 4.4.1 Why three KB metrics

Each metric isolates a different aspect of "the model surfaces this
target".

- **Method A** is the most direct: did the argmax land in the correct
  set? It matches the natural reading of "the model predicts the right
  thing" and is the metric a clinical informaticist would compute first.
- **Method B** is *calibration-robust*. If the model splits probability
  mass roughly equally between the correct head and a closely-related
  wrong head, A may flip with seed noise while the actual mass on the
  correct head is stable; B captures that stability.
- **Method C** is the *deployment-relevant* metric: in clinical use the
  model is allowed to abstain when uncertain, and what matters is the
  precision–recall trade-off as the abstention threshold sweeps. C is
  the area under that curve.

The three metrics are conceptually independent (an argmax-correct
prediction can have low mass on the correct head; a high mass on the
correct head can lose the argmax; and a model that wins on A and B can
still have a poor abstention curve), so reporting all three brackets
the surfacing finding against any single-metric artefact.

### 4.5 Primary metric pre-designation

The primary KB metric for H6 (coupling slope) and H7 (variance
asymmetry) is `KB_hit_A_setvalued` (Method A, set_valued). The choice
is committed *before* Phase A unblinds, for two reasons. First, A is
the simplest to interpret and matches the natural reading of "the model
surfaces the right answer", which is the practical question the paper
poses. Second, pre-designation prevents *metric shopping* — if all
three metrics were reported as primary candidates, the metric whose
slope or variance share happened to be most favourable to the
narrative could be selected after the fact, inflating false-positive
risk. KB_pmass_B and KB_auc_C are reported alongside as
sensitivities, never as headlines.

A is preferred over B and C as the *primary* metric on three grounds:
it is computable from `pred_label` alone and therefore robust to any
future eval-pipeline change that affects logit storage; it carries the
most direct interpretation ("did the argmax land in the correct set");
and on S_pair (the schema selected for Phase B) the set_valued and
single_label projections coincide, so A under set_valued has no residual
projection-mode ambiguity to bracket. B and C add information about
calibration and abstention behaviour respectively but each requires the
full softmax and each introduces a distinct interpretive layer
(probability mass, threshold sweep) that is unnecessary for the headline.

The legacy diagnostic `KB_surface_mean = mean(1 − P(__NEGATIVE__))` is
emitted by the eval pipeline for continuity but is *deprecated* and
never reported as primary or sensitivity in this paper, because it is
**correctness-blind**. A concrete failure case: if a model assigns 0.85
probability mass to the wrong non-negative head (e.g.
`GENE_GENE_ASSOC` for a gene–drug DGR target) and 0.10 to the correct
head, KB_surface_mean records 0.90 (high "non-NEG mass") while
KB_hit_A scores 0 (wrong argmax) and KB_pmass_B scores 0.10 (low mass
on correct). The legacy metric therefore rewards confidently *wrong*
predictions; we replace it with the correctness-aware family above.

### 4.6 Eval pipeline pinning

The evaluation pipeline lives in
`fine_tuning_experiments/schema_exp/eval/` and is pinned by
`EVAL_VERSION.txt`. The version string is enforced at *two* gates:
the inline-eval write gate (the trainer aborts a run before writing
`phase_a_eval.json` if its compiled-in version disagrees with
`EVAL_VERSION.txt`), and the aggregator read gate (aggregators refuse
to ingest records whose `eval_version` field does not match the
current string). Any change to the eval code therefore requires
re-running evaluation on every checkpoint that appears in the paper.

---

## Section 5 — Trainer Specification

### 5.1 Stage structure

Phase A and Phase B share a two-stage schedule by default
(`schedule: T1_to_T2`). Stage T1 trains on the leakage-fixed supervised
backbone (BioRED + DrugProt + BC5CDR); stage T2 is a staged continuation
on the MeSH-C04 oncology subset. Three schedule levels are exposed as
Phase B axis values: `T1_biored_only`, `T1_flat`, and `T1_to_T2`. At
the stage transition the optimiser and scheduler are re-initialised.
The schedule is HuggingFace linear-with-warmup, decaying from
`lr = 2e-5` to 0 over each stage's `max_updates = 2048` (warmup = 0).

### 5.2 Optimizer, clipping, precision

AdamW (HuggingFace defaults); gradient clipping at `max_grad_norm = 1.0`;
mixed precision off (pure FP32, so PB-base checkpoints are 438 MB).

### 5.3 Early stopping and checkpoint selection

Evaluation runs every 64 optimiser updates. Selection is by macro-F1 on
the *in-training* dev split — a seeded 12 % holdout of each stage's gold
positive pool with negatives materialised at the run's `negative_ratio`
to match training distribution. Early-stopping patience is 10
evaluations (640 steps), triggered only after `steps ≥ 256`. The
checkpoint with the highest dev macro-F1 *per stage* is kept; `best.pt`
is the overall best across stages. The dev metric is computed over the
*full* schema vocabulary (S_pair = 8 labels including dead heads) and is
never the BioRED/BC5CDR external test F1 nor an active-head variant —
external metrics live only in the post-training eval (§4.6) and do not
feed checkpoint selection.

### 5.4 Negative sampling

Negatives are online-sampled per batch, not pre-materialised. For each
positive: (1) sample up to `max_negatives_per_sample = 64` within-
document entity pairs satisfying `pair_type_filter`, excluding the
positive's own head–tail and its reverse; (2) keep `negative_ratio = 4.0`
negatives per positive; (3) shuffle and truncate to
`batch_size = 4`. Per-document sampling is seeded by `seed + 101` (T1)
and `seed + 202` (T2), giving a reproducible negative stream per run.

### 5.5 Source weighting

Each row carries a `source_weight` applied as a per-sample multiplier in
the cross-entropy loss. All Phase A/B configs use uniform weights
(`biored = drugprot = bc5cdr = 1.0`); the `inverse_freq_family_softmax`
alternative is implemented but disabled because uniform weighting was
more stable in T1 convergence preliminaries.

### 5.6 RNG determinism

All four RNG streams (`random`, `numpy`, `torch` CPU, `torch` CUDA) are
seeded from `cfg["seed"]`; the same seed is propagated to the dev-split
sampler, the data-shuffle generator, and the online negative sampler.
A full RNG-state snapshot is captured at every checkpoint save
(stage-best, stage-end, `best.pt`, `last.pt`), so any future analysis
can reproduce the exact batch stream from the checkpoint forward.

### 5.7 Code implementation

The trainer is a small set of version-controlled Python sources under
`fine_tuning_experiments/phase_b/trainer/`:

- `scientific_trainer.py` — training loop, stage dispatch, checkpoint
  save, metrics JSON output.
- `scientific_data.py` — shard loading, text encoding
  (`"{head} [ENT] {tail} [SEP] {doc}"[:8000]`), dev split, label-space
  derivation, online negative sampling, source weighting.

Entry point:
`python3.11 -m fine_tuning_experiments.train.run_experiment
 --experiment-id ... --config-path ... --run-root ...`. Both phases
dispatch through this entry point; only the YAML config differs.

### 5.8 Per-run artifacts

Every run directory carries the artifacts required for full
reproducibility:

```
<run_dir>/
  run_manifest.json           (config echo + git_commit + config_sha256 + trainer_source)
  training.log
  checkpoints/                (stage_t1_best/end.pt, stage_t2_best/end.pt, best.pt, last.pt;
                               each with rng_state for all four RNG streams)
  metrics/                    (metrics_*.json, validation_history.json, loss_history.jsonl,
                               dev_row_ids_t1.json, dev_row_ids_t2.json)
  predictions/predictions_scientific.jsonl
  eval/phase_a_eval.json or phase_b_eval.json   (auto-computed at end of training, §4.6)
```

`config_sha256` and `git_commit` make every run's methodological
provenance auditable; `dev_row_ids_*.json` freezes the exact dev
membership so two runs can be compared on a common dev intersection
without retraining.

---

## Section 6 — Phase A: Schema Selection

### 6.1 Role of Phase A

Phase A serves two purposes: it selects the schema(s) that enter Phase B
under the pre-committed rule of §6.6, and it provides the schema-arm
evidence that Phase B cannot supply — specifically, the schema-induced
variance contribution to H7 and the `β_schema` and `β_encoder` slopes in
the H6 family (§9.4). Architecture, update regime, and schedule are all
fixed at the anchor configuration (pipeline, full fine-tune, T1→T2
staged) so that schema and encoder are the only varying axes.

### 6.2 Cells

| Encoder | Shorthand | Checkpoint |
|---|---|---|
| RoBERTa-base | RB | `FacebookAI/roberta-base` |
| PubMedBERT-base | PB | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` |
| BioLinkBERT-base | BL | `michiyasunaga/BioLinkBERT-base` |
| PubMedBERT-large | PL | `microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract` |

| Schema | Shorthand | # labels |
|---|---|---|
| S_flat | Sflat | 4 |
| S_pair | Spair | 8 |
| S_mech | Smech | 13 |

Naming: `PA_{ENC}_{SCHEMA}_s{NN}` with NN ∈ 01..10. Total: 4 × 3 × 10 =
120 runs.

### 6.3 Locked hyperparameters

These values are shared across all 120 cells and serve as the anchor
defaults for Phase B.

| Parameter | Value |
|---|---|
| Architecture | pipeline |
| Update regime | full fine-tune |
| Schedule | T1 → T2 staged |
| Optimizer | AdamW |
| Learning rate | 2.0e-5 |
| Batch size | 4 |
| Max sequence length | 384 (RB uses 512) |
| T1 / T2 `max_updates` | 2048 / 2048 |
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

T1 shards: `t1_biored_trn_{SCHEMA}.jsonl`,
`t1_drugprot_trn_{SCHEMA}.jsonl`, `t1_bc5cdr_trn_{SCHEMA}.jsonl`. T2
shards: `t2_biored_mesh_{SCHEMA}.jsonl`, `t2_drugprot_mesh_{SCHEMA}
.jsonl`, `t2_bc5cdr_mesh_{SCHEMA}.jsonl`. Evaluation inputs:
`biored_test_pairs_{SCHEMA}.jsonl`, `bc5cdr_test_pairs.jsonl`,
`kb_surface_pairs.jsonl`.

### 6.5 Smoke test

Three cells (`PA_PB_Sflat_s01`, `PA_PB_Spair_s01`, `PA_BL_Sflat_s01`)
are trained first to verify end-to-end correctness of the Phase A
pipeline before the 120-run array is submitted. Acceptance criteria:
training completes without error; `metrics/validation_history.json` has
≥ 20 dev entries; `eval/phase_a_eval.json` has finite values for every
metric block; checkpoints carry `rng_state` with all four RNG streams;
the label-space regression test passes. All criteria are met on every
smoke run before the full array launches.

### 6.6 Schema-selection rule

For each schema, compute the pooled mean (over 4 encoders × 10 seeds =
40 runs) of BioRED macro-F1 ex-NEG, BC5CDR drug–disease F1,
`KB_hit_A_setvalued`, and per-head F1 for each active head. The
decision tree is pre-committed:

- **Outcome 1 — single schema dominates.** One schema is strictly
  higher on `KB_hit_A_setvalued` with the paired-bootstrap 95 % CI on
  the difference to the second-best schema excluding zero, *and* not
  worse on BioRED macro-F1 ex-NEG by more than a Cohen's d of 0.3.
  → Phase B runs that schema only.
- **Outcome 2 — dual schema required.** The top two schemas are within
  d = 0.3 on KB_hit_A or disagree in direction between KB_hit_A and
  BioRED macro-F1.
  → Phase B runs both as parallel factorials.
- **Outcome 3 — null result.** No schema separates beyond within-cell
  SD on any primary metric.
  → Phase B retains S_pair alone and reports Phase A as a null result.

The Outcome-1 difference CI is computed by *paired* bootstrap at the
`(encoder, seed)` cell level, with B = 10 000 resamples and a
deterministic seed (`20260416`).
*Why paired bootstrap.* Pairing across `(encoder, seed)` differences
the between-encoder variance out of the difference distribution; in
Phase A this produces a CI that is correctly calibrated and roughly
30 % tighter than the unpaired alternative without invoking
distributional assumptions.
*Why Cohen's d = 0.3 as the BioRED guard.* d = 0.3 is the conventional
small-to-medium boundary in psychometric and biomedical effect-size
literature; it is loose enough not to block a schema whose KB advantage
is large, and tight enough to refuse a schema whose BioRED loss is
practically meaningful.

### 6.9 Results

**Status.** 120 / 120 runs completed. Every run carries the seven
required artifacts (`run_manifest.json`, six checkpoints with four-stream
`rng_state`, `validation_history.json` with 32 T1 + 32 T2 evaluations,
`loss_history.jsonl`, both `dev_row_ids_*.json`, `eval/phase_a_eval
.json`, and `eval/kb_surface_targets.jsonl`). Machine-readable outputs
live at
`fine_tuning_experiments/schema_exp/analysis/phase_a_analysis.{json,md}`.

#### 6.9.1 Pooled schema means (n = 40 each)

| Schema | KB_hit_A_sv | KB_hit_A_sl | KB_pmass_B_sv | KB_auc_C_sv | BioRED macro | BioRED ex-NEG | BC5CDR DD |
|---|---|---|---|---|---|---|---|
| S_flat | 0.578 ± 0.238 | 0.578 ± 0.238 | 0.466 ± 0.151 | 0.785 ± 0.219 | 0.310 ± 0.030 | 0.139 ± 0.036 | 0.765 ± 0.143 |
| **S_pair** | **0.695 ± 0.181** | **0.695 ± 0.181** | **0.525 ± 0.100** | **0.807 ± 0.137** | **0.365 ± 0.080** | **0.300 ± 0.089** | **0.796 ± 0.082** |
| S_mech | 0.453 ± 0.195 | 0.002 ± 0.004 | 0.445 ± 0.116 | 0.664 ± 0.201 | 0.191 ± 0.047 | 0.139 ± 0.049 | 0.789 ± 0.089 |

All KB columns are means over the **n = 162 evaluable targets** per
run (the 3 unmapped CIViC targets are excluded from the denominator,
not counted as misses; §3.4, §4.3); BioRED and BC5CDR columns are
test-set means over the standard splits.

S_mech's `KB_hit_A_singlelabel` collapses to 0.002 because five of the
six DGR sub-heads have zero BioRED test support and are therefore never
picked as argmax — a structural failure of the 13-way schema on the
present evaluation corpus, not a parameter-tuning artefact.

#### 6.9.2 Schema-selection rule applied → Outcome 1, S_pair

All difference CIs are paired-bootstrap on 40 matched `(encoder, seed)`
cells, B = 10 000.

- **S_pair vs S_flat:** Δ = +0.117 on KB_hit_A; paired CI [+0.045,
  +0.194] (excludes 0); Cohen's d = +0.55; permutation p = 0.015
  (BH-adj 0.024). On BioRED ex-NEG, Δ = +0.161 (d = +2.37) — S_pair is
  strictly better, so the Cohen's d ≤ 0.3 guard is satisfied trivially.
- **S_pair vs S_mech:** Δ = +0.242; paired CI [+0.171, +0.313]; d = +1.29;
  p < 0.0001.

→ **Outcome 1 (single schema dominates). Phase B runs S_pair only.**

The encoder-stratified robustness check is consistent: S_pair > S_flat
on KB_hit_A in all four encoders (RB +0.159, PB +0.052, BL +0.132,
PL +0.127) and S_pair > S_mech in all four (RB +0.144, PB +0.177,
BL +0.374, PL +0.275).

#### 6.9.3 H7 variance decomposition

Variance shares are produced by a Type-I sum-of-squares ANOVA
decomposition that allocates the total sum of squares of each metric to
the design factors `(schema, encoder, schema × encoder)` and the
within-cell residual.
*Why ANOVA.* The Type-I SS partition tells the reader directly what
fraction of the metric's total variance is "absorbed" by each design
lever, which is exactly the question H7 asks. It does not require the
factors to be orthogonal in design (Phase A is balanced, so they are),
and it produces a single, non-negative variance share per factor that
sums to 100 %.

| Metric | schema | encoder | schema × encoder | residual |
|---|---|---|---|---|
| `kb_hit_A_setvalued` | 19.1 % | 17.2 % | 3.9 % | 59.7 % |
| `biored_macro_f1_ex_neg` | **60.4 %** | 24.2 % | 7.3 % | 8.2 % |
| `bc5cdr_drug_disease_f1` | 1.5 % | **37.2 %** | 3.6 % | 57.7 % |

The residual column collects all variance not absorbed by the named
factors; under the balanced 10-seeds-per-cell design it is dominated by
seed-to-seed variation (initialisation, dropout noise, online negative-
sample order) rather than by unmodelled higher-order interactions.

Schema explains 60.4 % of BioRED ex-NEG variance but only 19.1 % of
KB_hit_A variance — a 3.16× asymmetry on the schema-only narrowing — and
1.5 % of BC5CDR DD variance, a 40× asymmetry on the out-of-domain
benchmark. Reporting the union of design levers as
`(schema + encoder + schema × encoder)`, the Phase A asymmetry ratio is
$R_A = 91.9 / 40.2 \approx 2.29$. This is the schema-arm of H7 and is
reported descriptively (no pre-committed threshold is applied to an
already-observed value).

#### 6.9.4 Cross-metric correlations

| Metric pair (n = 120 seed-level) | Pearson r [95 % CI] | Spearman ρ [95 % CI] |
|---|---|---|
| BioRED ex-NEG × KB_hit_A | +0.523 [+0.396, +0.637] | +0.457 [+0.297, +0.596] |
| BC5CDR DD × KB_hit_A | +0.435 [+0.258, +0.570] | +0.360 [+0.182, +0.520] |
| BioRED ex-NEG × BC5CDR DD | +0.503 [+0.432, +0.595] | +0.571 [+0.425, +0.694] |
| BioRED macro × KB_hit_A | +0.563 [+0.441, +0.677] | +0.569 [+0.448, +0.683] |

We report Pearson and Spearman together because they answer different
questions: Pearson tests whether the relationship is *linear* (relevant
when one of the metrics is used as a regression input for the other,
e.g. β_config in §9.4); Spearman tests whether the relationship is
*monotonic* (relevant when the metrics are used to *rank* models, which
is the practical concern raised by RQ4). When both agree in sign and
magnitude, as they do here, the coupling is both linear and monotonic;
when they disagree, the discrepancy is informative about non-linearity.

At cell level (n = 12 cells = 4 encoders × 3 schemas), Pearson r rises
to +0.77 for BioRED × KB and +0.58 for BC5CDR × KB. All point estimates
are positive and all 95 % CIs comfortably exclude zero.

The positive coupling shown by these correlations is the empirical
premise on which the RQ4 framing of variance asymmetry under positive
coupling rests; the variance-asymmetry finding itself is reported in
§6.9.3.

#### 6.9.5 Per-head BioRED F1 and the active-head secondary

| Schema | Heads with support > 0 | Heads above 0.05 F1 | Dead heads |
|---|---|---|---|
| S_flat | 2 (`AG` sup ≈ 969, `DGR` sup ≈ 21) | 1 (`AG` F1 = 0.412) | `DGR` F1 = 0.005 |
| S_pair | 7 | 5 (`GENE_DISEASE` 0.479, `DRUG_DISEASE` 0.451, `VARIANT_DISEASE` 0.429, `GENE_GENE_ASSOC` 0.385, `DRUG_GENE_REGULATION` 0.354) | `AG` F1 = 0 (sup ≈ 104, absorbed by specialised heads); `DRUG_VARIANT_ASSOC` F1 = 0 (sup ≈ 15, data-thin) |
| S_mech | 7 (5 DGR sub-heads have 0 BioRED test support) | 4 (4 specialised pair heads) | All 5 `DGR_*` sub-heads, `AG`, `DGR`, `DRUG_VARIANT_ASSOC` |

The S_pair `ASSOCIATION_GENERAL` head's F1 = 0 is a category-competition
artefact: once the five specialised pair heads are available, BioRED's
residual AG-annotated pairs are routable to one of them, so the AG head
receives near-zero test-support-at-argmax signal. The effective S_pair
classifier therefore behaves as a 5-active-head + NEGATIVE space, even
though all 8 outputs remain present.

**Pre-registered secondary: active-head macro-F1.** For every primary
H1–H4 and H6 test on BioRED ex-NEG, a parallel test is computed on
"active-head macro-F1" = macro over the *frozen* set
**{`GENE_DISEASE`, `DRUG_DISEASE`, `VARIANT_DISEASE`, `GENE_GENE_ASSOC`,
`DRUG_GENE_REGULATION`}** — the S_pair heads with Phase A mean test-F1
> 0.05. The active-head identity is **fixed at this point**: if Phase
B produces an AG head with F1 > 0.05 or a currently-active head that
collapses to F1 ≤ 0.05, the active-head set is *not* recomputed; both
cases are reported as exploratory findings under their own
stand-alone definitions in a Supplementary table. Fixing the
active-head set at lock time prevents post-hoc head selection —
re-computing "active" on Phase B data would be a form of metric
shopping where the head set chosen happens to favour whichever
H1–H4 outcome is observed.

Active-head macro-F1 is computed *post-hoc at evaluation time only* from
the per-head F1 values in `phase_a_eval.json`. It is *not* used as the
in-training dev metric (§5.3); switching the training objective to an
active-head variant would alter checkpoint selection and therefore the
per-run numbers themselves.

#### 6.9.6 Intraclass correlation (ICC)

ICC(1,1) is the fraction of total variance attributable to between-cell
differences in a one-way random-effects model, where each cell is
treated as a population of seed-level replicates and the seed-to-seed
variation within a cell plays the role of within-subject error. (The
"rater" framing of ICC is borrowed: seeds are not independent
observers of a fixed quantity but independently-initialised models
whose variation is the relevant analogue of measurement noise here.)
Cicchetti (1994) labels ICC ≥ 0.40 as "fair", ≥ 0.60 as "good", ≥ 0.75
as "excellent"; the same conventions are used in psychometric
reliability literature.

| Metric | ICC(1,1) | Interpretation |
|---|---|---|
| `kb_hit_A_setvalued` | 0.360 | fair |
| `biored_macro_f1_ex_neg` | 0.916 | excellent |
| `biored_macro_f1` | 0.929 | excellent |
| `bc5cdr_drug_disease_f1` | 0.383 | fair |

KB_hit_A's "fair" ICC quantifies what §6.9.3 already implied: roughly
60 % of KB_hit_A variance is within-cell seed noise, so a single seed's
KB outcome is a poor estimate of the cell's expected KB. The
methodological consequence for Phase B is that all KB-side comparisons
must be *paired across seeds* (so cell-level seed noise differences
out) rather than unpaired; this is one motivation for the
seed-matched factorial in §7.

**Bridge to Phase B.** Phase A fixes three of the four design levers
the paper varies — schema is selected to S_pair (§6.9.2), the
schema-arm of H7 is observed at $R_A \approx 2.29$ (§6.9.3), and the
seed-noise floor on KB_hit_A is large enough that all Phase B
comparisons must be paired across seeds (this section). Phase B
varies encoder, update regime, and schedule under that fixed schema,
so it can supply the configuration-arm of H7 ($R_B$) and the
configuration-arm of the H6 slope family ($\beta_\text{config}$),
neither of which Phase A alone can identify.

---

## Section 7 — Phase B: Training-Configuration Factorial

### 7.1 Role and pre-commitment

Phase B is the confirmatory phase. It answers RQ2 with a paired
factorial under S_pair and supplies the configuration-arm of H7
(variance asymmetry under configuration variance, complementing the
schema-arm from Phase A) and `β_config` of the H6 family. The full
factorial design and every primary hypothesis are pre-registered at the
public git tag `phase_b_prelock_v1`; the lock SHA-256 and tarball
checksums are listed in Appendix A.

### 7.2 Hypotheses

| ID | Claim | Tier | Primary test | Decision rule |
|---|---|---|---|---|
| **H1** | PubMedBERT-large > {PubMedBERT-base, BioLinkBERT-base} on BioRED macro-F1 ex-NEG under matched config. | Primary | Paired-t + Wilcoxon (dual report); 3 pairwise tests. | Confirmed: PL > both by Δ ≥ 0.02 with FDR-q < 0.05 on both tests. Null: gaps < 0.01 or q > 0.10. Inverted: PL < either by Δ ≥ 0.02, q < 0.05. |
| **H2** | Multi-corpus T1 > BioRED-only T1 on BC5CDR DD F1 (held-out OOD). | Primary | Paired-t + Wilcoxon at PB × pipeline × full-FT. | Confirmed: Δ ≥ 0.03, q < 0.05 on both. Null: \|Δ\| < 0.02. Counter-finding: BioRED-only > multi-corpus by Δ ≥ 0.03 with q < 0.05 (multi-corpus actively *hurts* OOD). |
| **H3** | T1→T2 staged > T1_flat on BioRED and BC5CDR, across all three biomedical encoders. | Primary | Paired-t + Wilcoxon on BioRED ex-NEG and BC5CDR DD at PB/BL/PL × pipeline × full-FT (6 tests; FDR over 6). | Confirmed: ≥ 4 of 6 with Δ ≥ 0.02 and q < 0.05. Partial: 2–3 of 6. Counter-finding: T1_flat > T1→T2 in ≥ 4 of 6 tests at Δ ≥ 0.02 and q < 0.05 (oncology-projection stage actively *hurts*). |
| **H4** | Full fine-tune > LoRA on the small-data oncology bridge with d ≥ 0.5 for biomedical encoders. | Primary | Paired-t + Wilcoxon on BioRED ex-NEG, FT vs LoRA, at PB/BL/PL × pipeline × T1→T2. | Confirmed: 3/3 encoders show FT > LoRA with d ≥ 0.5 and q < 0.05. Counter-finding: any encoder shows LoRA > FT with q < 0.05. |
| **H5** | Pipeline ≈ shared-multitask on macro-F1 ex-NEG (equivalence). | Primary (TOST) | TOST at PB and PL × full-FT × T1→T2; equivalence margin ±0.03. | Equivalent: 90 % CI of Δ ⊂ [−0.03, +0.03] for both encoders. Counter-finding: TOST 90 % CI excludes \[−0.03, +0.03\] in either direction (positive disequivalence — pipeline and shared-multitask differ by more than one within-cell SD). |
| **H6** | The benchmark-to-KB coupling slope is a *family of five mechanism-stratified slopes* (β_within, β_schema, β_encoder, β_config, β_combined_cell), not a single quantity. | Primary (characterisation family) | See §8.4. | Three-bin label per slope (strong / moderate / weak), with "inconclusive" when 95 % CI width > 0.30. Counter-finding: β_within is **strong** (currently anticipated weak / near-zero, §7.2.1) — would indicate seed-level coupling not captured by any cell-level mechanism, motivating a re-specification of the H6 family. |
| **H7** | Design levers (schema in Phase A; configuration in Phase B) exhibit variance asymmetry — they explain more variance in BioRED ex-NEG than in `KB_hit_A_setvalued`. | Primary (RQ4 headline) | ANOVA variance decomposition; ratio R = (design-lever share in BioRED) / (same share in KB_hit_A). | Phase A descriptive (R_A ≈ 2.29 / 3.16, §6.9.3). Phase B threshold R_B ≥ 2 confirms; 1 < R_B < 2 borderline; R_B ≤ 1 null. Counter-finding: R_B < 1 with bootstrap 95 % CI fully below 1 (design levers explain *more* variance in KB than in BioRED — the asymmetry runs the other way, and Abstract framing reverses accordingly). |

*Rationale for paired-t + Wilcoxon dual report.* The paired-t statistic
tests the mean of the per-seed difference distribution and is most
powerful when that distribution is approximately normal; Wilcoxon
signed-rank tests the median and is robust to heavy tails or outliers.
Reporting both protects against the small but non-negligible failure
mode in which a single bad seed inflates the t-test variance and
suppresses a real effect that Wilcoxon would still detect.

*Rationale for the unstandardised effect-size thresholds.* The Δ
thresholds in H1–H3 (Δ ≥ 0.02 for BioRED, Δ ≥ 0.03 for BC5CDR) are
each set to roughly one within-cell SD of the corresponding metric as
observed in Phase A (BioRED ex-NEG SD ≈ 0.03; BC5CDR DD SD ≈ 0.04). A
mean difference of one within-cell SD is the smallest effect that is
reliably distinguishable from seed noise under paired comparison at
20 seeds, so the thresholds are calibrated to the noise floor of the
metric rather than to a uniform numeric value. H4 reports a
standardised threshold (Cohen's d ≥ 0.5, the conventional
medium-effect boundary; Cohen 1988) instead of an unstandardised Δ
because the FT and LoRA arms have visibly unequal seed-level variance,
and a standardised cut keeps the threshold comparable across the two
regimes.

*Rationale for TOST in H5.* Two one-sided tests is the standard
equivalence-testing procedure in biostatistics: it converts the null of
"a real difference exists" into the null of "the difference is at least
as large as some pre-specified margin", so a non-rejection yields a
positive equivalence claim rather than a vacuous "we failed to detect a
difference". The equivalence margin ±0.03 is set to one within-cell SD
of BioRED ex-NEG observed in Phase A, so a difference smaller than ±0.03
is *literally indistinguishable from seed noise* under the lock-time
data.

*Rationale for the H6 mechanism-stratified slope family.* A single
mixed-effects pooled slope on `KB ~ BioRED + (1|cell) + (1|seed)`
identifies its fixed effect from within-cell seed-level variation only,
which is noise-limited (Phase A within-cell BioRED SD ≈ 0.03 vs KB SD
≈ 0.20) and would recover β ≈ 0 *regardless of the real between-cell
coupling*. A pooled fixed-effects slope mixes mechanisms whose
individual slopes need not agree (schema and configuration produce
BioRED variance through different pathways). The family decomposition
in §8.4 reports each mechanism's coupling separately, so
"mechanism-dependence of coupling" can be read as an empirical finding
rather than an artefact of model specification.

*Rationale for the H7 R_B ≥ 2 threshold.* R = 1 is the perfect-proxy
reference (a design lever absorbs the same fraction of variance in both
metrics). R ≥ 2 corresponds to a 2× disparity between benchmark and KB
variance under the same lever, which is a conventional "meaningful
asymmetry" cut in variance-share ratio literature; R < 2 leaves
insufficient margin to distinguish asymmetry from sampling noise given
the Phase B cell count. The threshold is set independently of the
already-observed Phase A R_A.

### 7.2.1 H6 Abstract-level claim pre-commitment

Because H6's three-bin slope labels (weak / moderate / strong)
translate into different Abstract framings of RQ4, the mapping from
observed β_config bin to Abstract claim is pre-registered at lock to
prevent post-hoc adjustment of the paper's central narrative based on
which bin the data lands in.

| Observed H6 pattern | Pre-committed Abstract-level framing |
|---|---|
| β_config is **weak** (\|β̂\| < 0.3 or CI straddles zero) | "Benchmark rank is a low-fidelity guide to KB surfacing in the regime of typical configuration choices." *(currently anticipated)* |
| β_config is **moderate** (0.3 ≤ \|β̂\| < 1.0) and β_schema strong | "Benchmark-to-KB coupling is mechanism-dependent: strong under schema choice, weaker within a fixed schema across configurations." |
| β_config is **strong** (\|β̂\| ≥ 1.0, same-sign CI) | "Variance asymmetry (H7) is the primary RQ4 finding; within-regime benchmark-to-KB coupling is stronger than anticipated, so rank-based selection under fixed schema is better-calibrated than a pure decoupling story would predict." |
| β_config is **inconclusive** (CI width > 0.30) | "We report variance asymmetry (H7) and the H6 slope family descriptively; β_config's CI width exceeds the 0.30 reportability gate, and no Abstract-level coupling claim is made within a fixed schema." |

This is the full set of admissible H6/H7 headlines. If the observed
pattern does not match one of the four rows (e.g. β_schema inverts
sign), that mismatch is itself reported as a counter-finding and the
Abstract states so in those terms.

### 7.3 Design axes

- **Encoder** (3 levels in main factorial + 1 reference): PB, BL, PL in
  the factorial; **RB** as a descriptive general-domain reference (10
  seeds at the anchor cell only). RB is excluded from H1–H7 and from FDR
  correction because it represents a pretraining paradigm outside the
  hypothesis space; its role is to contextualise the
  biomedical-pretraining premium.
  Excluded by modelling decision: SciBERT, BlueBERT, BioBERT,
  BiomedLM, GatorTron — PB/BL/PL span "biomedical pretraining variant +
  scale" cleanly.
- **Architecture**: `pipeline` only. The originally planned
  `shared_multitask` arm is deferred to follow-up work, and H5 is
  reported as deferred rather than confirmed or rejected here.
- **Update regime** (2 levels): `full_ft` and `lora` (rank 16, α = 32,
  dropout 0.05, target modules = attention Q/V projections, classifier
  head fully trained, LR matched to FT). The LoRA arm is gated on a
  single-seed budget probe at the doubled training budget
  (`max_updates = 4096`, LR = 2.0e-5, all other LoRA hyperparameters
  unchanged). The pre-committed acceptance criteria are all-must-pass:
  (i) `dev_macro_f1 > 0.20` at some optimiser step ≤ 1024;
  (ii) `dev_macro_f1 > 0.30` at the final step (4096); and
  (iii) at least one evaluation point with `dev_macro_f1_excluding_negative > 0`.
  These three criteria are encoded as exit-code logic in
  `phase_b/sbatch/phase_b_lora_d3_smoke.sbatch` and apply to the
  trainer's full-vocabulary in-training dev metric (§5.3). If any
  criterion fails, the LoRA arm is dropped and H4 is reported as
  *empirically undefined for the present budget* rather than as a
  confirmation of full-FT superiority on a trivially-collapsed
  comparator.
- **Schedule** (3 levels): `T1_biored_only`, `T1_flat`, `T1_to_T2_staged`.
- **Schema**: S_pair only (Phase A Outcome 1).
- **Seed** (20 per cell). Seeds 1..10 are reused from Phase A so seed
  remains a meaningful cross-phase random effect; seeds 11..20 are
  fresh.

### 7.4 Factorial

The factorial is 3 biomedical encoders × 2 update regimes × 3 schedules
× 20 seeds = 360 main runs, plus a 10-seed RB reference cell at the
anchor configuration; the 180-run LoRA half is gated on the
update-regime budget probe described above.

Run naming: `PB_{ENC}_{UPD}_{SCHED}_s{NN}` with `ENC ∈ {PB, BL, PL, RB}`,
`UPD ∈ {FT, LR}`, `SCHED ∈ {T2, T1F, T1B}`, `NN ∈ 01..20`. The schema
field is dropped — every Phase B run is S_pair.

### 7.5 Why a full factorial rather than a star design

A star design at the anchor cell would leave encoder × update and
encoder × schedule interactions confounded; in particular, H4's
cross-encoder generalisation claim requires BL × LoRA and PL × LoRA
cells that a star design omits. The full factorial at 20 paired seeds
per cell also gives H7's variance decomposition adequate power to
partition configuration variance across its design axes — the single
most important Phase B estimand because it closes the loop on the RQ4
variance-asymmetry headline.

### 7.6 Training configuration

All cells share the §6.3 anchor hyperparameters. LoRA cells add rank
16, α = 32, dropout 0.05, `target_modules = ["query", "value"]`,
classifier fully trained, LR matched to FT (this is a *conservative*
specification for H4 — LoRA-optimal LR sweeps are out of scope for the
present paper).

The factorial therefore yields ≈ 18 main cells × 20 paired seeds, plus
the RB reference cell. §8 specifies how every primary and secondary
hypothesis in §7.2 is tested on this design, including the
multiple-comparison correction tier that each hypothesis falls into and
the slope-by-slope fit specifications that operationalise the H6 family.

---

## Section 8 — Statistical Analysis Plan

### 8.1 Three-tier comparison framework

To control multiple-testing inflation while preserving honest reporting
of the full factorial:

- **Primary tier.** H1–H7 plus the enumerated pairwise tests within
  them; approximately 21 primary comparisons. Correction:
  Benjamini–Hochberg FDR at q = 0.05 over all primary comparisons
  combined.
  *Why BH-FDR for primary.* BH-FDR controls the expected proportion of
  false discoveries, not the family-wise error; it tolerates positive
  dependence among tests (tests within H3 share BC5CDR seeds, for
  example) and is uniformly more powerful than Bonferroni when the
  family contains many true positives, which is the regime we expect.
- **Planned-secondary tier.** Replications of primary comparisons in
  non-anchor cells, plus active-head macro-F1 replicas of H1–H4 and
  H6 using the 5-active-head set frozen in §6.9.5. Total ~40
  comparisons, pre-enumerated in `comparisons_inventory.csv`.
  Correction: Bonferroni at α = 0.05 within each hypothesis's
  secondary set.
  *Why Bonferroni for secondary.* Secondary comparisons are
  confirmatory only conditional on a primary effect; the worst-case
  family-wise control of Bonferroni is appropriate to the more
  conservative inferential weight we assign to them, and the smaller
  number of secondaries per hypothesis (≤ 6) keeps Bonferroni's
  conservatism manageable.
- **Exploratory tier.** All other comparisons enabled by the factorial.
  No multiple-testing correction; raw p-values are reported alongside
  the explicit label "exploratory; not corrected".
  *Why no correction for exploratory.* These analyses are descriptive
  hypothesis-generators; correcting them would falsely promote them
  toward primary status. As a counter-protection, exploratory results
  are not quoted in the Abstract or Conclusions.

### 8.2 Test sensitivity (paired-t + Wilcoxon)

Every primary and planned-secondary comparison reports both paired-t
and Wilcoxon signed-rank. Headline result is paired-t unless the two
disagree at the chosen threshold (one having p < 0.05, the other p >
0.10), in which case the Wilcoxon result is the headline and a
footnote flags the disagreement.

Normality is assessed once on the *pooled z-standardised* differences
across all primary comparisons rather than per-comparison Shapiro–Wilk
(which would spuriously flag 3–4 of 72 cells at α = 0.05). The
procedure: per primary comparison i, compute paired seed-level
differences d_i, standardise z_i = d_i / SD(d_i), pool all z_i (≈ 300
values), and apply one Shapiro–Wilk test. Decision rule: pooled-z p <
0.01 → all headlines switch to Wilcoxon globally; otherwise paired-t
remains the headline.

### 8.3 Coupling-slope CI-width gate (reportability, not acceptance)

H6 is a characterisation family, not a null test. The CI-width check
is therefore a *reportability gate* applied per slope, not an
acceptance gate.

**Threshold = 0.30, per slope.** A slope whose 95 % CI width exceeds
0.30 on the natural slope scale (units of Δ`KB_hit_A` per
Δ`BioRED_F1`) is reported as **inconclusive** and does not receive a
strong / moderate / weak label. The threshold matches the width of
the narrowest three-bin category (the weak band, |β| < 0.3), so a CI
wider than 0.30 cannot adjudicate the boundary between "weak coupling"
and "moderate coupling" — exactly the boundary that controls the
paper's "benchmark is a low-fidelity proxy" narrative.

#### 8.3.1 H6 three-bin slope categorisation (pre-committed)

The strong / moderate / weak / inconclusive labels for every slope in
the H6 family are pre-committed at the following thresholds on the
natural slope scale (units of Δ`KB_hit_A` per Δ`BioRED_F1`):

| Bin | Criterion |
|---|---|
| **Strong** | \|β̂\| ≥ 1.0 *and* the 95 % CI excludes zero (same-sign) |
| **Moderate** | 0.3 ≤ \|β̂\| < 1.0 *and* the 95 % CI excludes zero |
| **Weak** | \|β̂\| < 0.3 *or* the 95 % CI straddles zero |
| **Inconclusive** | 95 % CI width > 0.30, regardless of point estimate (§8.3) |

*Rationale for β = 1.0 (strong cut).* β = 1.0 is the perfect-linear-
proxy reference — one unit of BioRED F1 movement maps to one unit of
KB_hit_A movement, so benchmark rank and KB rank carry the same
information up to a constant offset. A slope at or above this value is
the natural definition of "the benchmark *is* a faithful proxy for
KB surfacing".

*Rationale for β = 0.3 (weak cut).* Phase A within-cell standard
deviations are BioRED ex-NEG ≈ 0.03 and KB_hit_A ≈ 0.16. A coupling
slope of 0.3 therefore means that one within-cell SD of BioRED
movement corresponds to 0.3 × 0.03 = 0.009 in KB_hit_A — about **6 %**
of KB_hit_A's own within-cell SD. At this slope, BioRED has to move by
roughly five within-cell SDs before KB_hit_A moves by one within-cell
SD; on the practical measurement scale, KB barely moves at all when
BioRED moves by typical seed noise. β < 0.3 is therefore the regime in
which the benchmark cannot reliably rank-order configurations on the
KB axis. The 0.3 cut also coincides with the §8.3 reportability gate
of 0.30, so the inconclusive band sits exactly at the
weak/moderate boundary that the paper's narrative depends on.

### 8.4 H6 fit specification — five mechanism-stratified slopes

All slopes are OLS or OLS-with-cluster-robust-SE on appropriately
aggregated data; no single mixed-effects model is used.

**(a) β_within — within-cell (seed-level) coupling.** For each cell
(12 Phase A cells + 18 Phase B cells), fit an OLS slope of
`KB_hit_A` on `BioRED_F1` across the seeds in that cell.
Report the seed-count-weighted mean slope and a paired 95 % CI via
cluster-bootstrap (5,000 resamples, resampling whole cells).
*Expected value: near zero*, because within a cell the seed-to-seed
variation in BioRED and in KB are dominated by approximately
independent sources of randomness (different classifier-head
initialisations, dropout noise, data-order perturbations).

**(b) β_schema — Phase A between-schema slope at fixed encoder.** For
each encoder, compute three cell means $(\bar{x}_{e,s}, \bar{y}_{e,s})$
for $s \in \{$Sflat, Spair, Smech$\}$ and fit OLS slope across the
three schemas (n = 3 per encoder). Pool across encoders by inverse-
variance weighting; CI via cluster-bootstrap.

**(c) β_encoder — Phase A between-encoder slope at fixed schema.** For
each schema, compute four cell means (RB, PB, BL, PL) and fit OLS slope
(n = 4 per schema). Pool across schemas by inverse-variance weighting;
CI via cluster-bootstrap. β_encoder is the only place in the H1–H7
family where RB is included in a primary statistic: H6 is a
*characterisation* of the empirical benchmark-to-KB coupling across
the encoder population *as observed*, including the general-domain
reference, rather than a hypothesis test specific to biomedical
pretraining. The biomedical-pretraining-specific test is H1, from
which RB is excluded as stated in §7.3.

**(d) β_config — Phase B between-config slope (S_pair only).** Compute
the cell means across the 18 Phase B cells (or 9 if the LoRA half is
gated out by the update-regime budget probe of §7.3) and fit OLS slope.
Report Wald + cluster-bootstrap CIs.

**(e) β_combined_cell — pooled between-cell slope across phases.**
Compute all cell means, fit OLS slope on `KB ~ BioRED + phase_dummy`
with phase as a fixed covariate to absorb intercept differences. Wald
CI on the slope; phase-interaction test on $H_0: \beta_A = \beta_B$.

All five slopes are additionally re-fit using active-head macro-F1
(§6.9.5) as the benchmark side, as the planned secondary.

### 8.5 H7 variance decomposition

For Phase B (n = 360 seed-level observations, S_pair only) decompose
variance of BioRED ex-NEG and of `KB_hit_A_setvalued` across the
factor set `(encoder, update_regime, schedule)` and all of their
two-way interactions, plus the residual. For Phase A (n = 120) the
factor set is `(schema, encoder, schema × encoder)` plus residual.
Type-I SS partition is used in both phases (cf. §6.9.3).

The asymmetry ratios are defined as **ratios of variance shares**, so
that they are dimensionless and directly comparable across phases and
across metrics with different total SS. For each metric $m$, let
$\text{Share}_m(\mathcal{F})$ denote the fraction of $m$'s total
sum-of-squares attributable to a factor set $\mathcal{F}$:
$$\text{Share}_m(\mathcal{F}) = \frac{\sum_{f \in \mathcal{F}} SS_m(f)}{SS_m^{\text{total}}}.$$

Phase A asymmetry ratio (descriptive, §6.9.3):
$$R_A = \frac{\text{Share}_{\text{BioRED}}(\{\text{schema}, \text{encoder}, \text{schema}\times\text{encoder}\})}{\text{Share}_{\text{KB\_hit\_A}}(\{\text{schema}, \text{encoder}, \text{schema}\times\text{encoder}\})} = \frac{91.9\,\%}{40.2\,\%} \approx 2.29.$$

Phase B asymmetry ratio (confirmatory, identical-factor form):
$$R_B = \frac{\text{Share}_{\text{BioRED}}(\mathcal{F}_B)}{\text{Share}_{\text{KB\_hit\_A}}(\mathcal{F}_B)},
\qquad \mathcal{F}_B = \{\text{enc}, \text{upd}, \text{sch}, \text{enc}\times\text{upd}, \text{enc}\times\text{sch}, \text{upd}\times\text{sch}\}.$$

Three-way interactions and the residual are excluded from
$\mathcal{F}_B$. Numerator and denominator are bound to the same
factor set, so $R_B$ is the ratio of variance shares attributable to
identically-defined design levers under the two metrics, and is
directly comparable in form to $R_A$. R_A is descriptive; R_B is the
confirmatory statistic for H7 with the pre-committed threshold
$R_B \ge 2$ (§7.2).

**Bootstrap protocol for $R_B$.** Five thousand cluster-bootstrap
resamples are drawn at the cell level: for each resample, draw 18
cells (or 9 if the LoRA arm is gated out) **with replacement** from
the realised Phase B cells, preserving all 20 seeds within each
sampled cell. The Type-I SS ANOVA decomposition is recomputed on the
resample, and $R_B$ is recomputed from the share definition above.
The reported 95 % CI is the percentile interval (2.5th and 97.5th
percentiles of the bootstrap distribution); the bootstrap-distribution
median is reported alongside the point estimate as a stability check.

### 8.6 Ordinal-instability quantification

For each ordered pair of Phase B cells $(c_i, c_j)$ whose mean
BioRED ex-NEG differs by less than the **pre-specified matching
radius** $\rho = 0.03$, define the **KB swap magnitude** as
$$\Delta\text{KB}(c_i, c_j) = |\overline{\text{KB\_hit\_A}}(c_i) - \overline{\text{KB\_hit\_A}}(c_j)|.$$
The matching radius is fixed at the Phase A within-cell BioRED ex-NEG
SD ($\rho = 0.03$, §6.9). Pinning $\rho$ to a Phase A quantity rather
than to the Phase B within-cell SD avoids a circular dependency: the
Phase B factorial size depends on the LoRA-arm gating outcome (§7.3),
so a Phase-B-derived SD would yield a matching radius that is itself
a function of the unblinded data and of an as-yet-unobserved
gating verdict.

Reported quantities:

- the distribution of $\Delta\text{KB}$ over the eligible cell pairs,
  visualised as a histogram (Figure F4(b));
- the **median** $\Delta\text{KB}$, as the point estimate of "how much
  KB_hit_A can swap while BioRED is statistically tied";
- the **rank-inversion rate**, defined as the fraction of eligible
  pairs whose BioRED ranking and KB_hit_A ranking disagree.

Cell pairs are de-duplicated (each unordered pair counted once); pairs
involving the RB reference cell are excluded (RB is descriptive only,
§7.3). Cluster-bootstrap CIs (5,000 resamples over whole cells) are
reported on both the median and the rank-inversion rate.

### 8.7 Power

Phase A within-cell SDs (BioRED ex-NEG ≈ 0.03; BC5CDR DD ≈ 0.03–0.10
depending on schema; KB_hit_A_sv ≈ 0.16) yield, under α = 0.05 and
power = 0.80, detectable paired effect sizes of Δ ≈ 0.03–0.05 on BioRED,
≈ 0.07 on BC5CDR DD, and ≈ 0.15–0.17 on KB_hit_A. With 20 paired seeds
per Phase B cell, the H1–H4 paired-difference power exceeds 0.84 at
d = 0.6. H7's variance decomposition uses 360 observations, which is
high-power for variance partitioning. H6 power varies by slope:
β_config (n = 18 cells) is the most likely to clear the 0.30 CI-width
gate; β_schema (n = 3 means per encoder) and β_encoder (n = 4 means
per schema) are expected to trip the gate and be reported
descriptively (small-sample limitation explicitly acknowledged in §8.4).

*Caveat on seed reuse.* Of the 20 seeds per Phase B cell, seeds 1–10
are reused from Phase A (preserving cross-phase pairing) and seeds
11–20 are fresh. The reuse does not affect Phase B's *internal* paired
comparisons (each cell uses its own 20 seeds against its own
counterpart's 20 seeds, and noise from the shared seeds differences
out of paired contrasts), so the H1–H5 power figures above are
unaffected. The cross-phase β_combined_cell test (§8.4(e)) does
treat all 20 seeds as independent observations within Phase B; the
shared seeds introduce mild dependence between Phase A and Phase B
estimates of cell means, and the cross-phase slope should be
interpreted with this caveat.

---

## Section 9 — Paper Structure

### 9.1 Section layout (Bioinformatics format)

1. **Abstract** (250 words). Motivation → methods → central finding
   (variance asymmetry + mechanism-stratified coupling characterisation +
   ordinal instability) → implication.
2. **Introduction.** Oncology gap → heterogeneous supervision →
   evaluation-validity question → contributions.
3. **Methods.** Data and schemas, training framework, evaluation,
   statistical analysis.
4. **Results.**
   - §4.1 Schema selection (Phase A; §6.9).
   - §4.2 Main benchmark results under S_pair (Phase B; H1–H4).
   - §4.3 KB-surface yield (Phase B; per-family breakdown; active-head
     sensitivity).
   - §4.4 Evaluation-validity audit: H7 variance asymmetry as headline;
     H6 mechanism-stratified slopes; ordinal-instability summary.
5. **Discussion.** How to read benchmark leaderboards when downstream
   utility is the goal; clinical-evaluation implications; W1–W8.
6. **Conclusion.**

### 9.2 Figure plan (5 main figures)

| Figure | Content |
|---|---|
| **F1** | Supervision pipeline schematic (T1 → T2 → T3/T4) with schema labels. |
| **F2** | Phase B main results: BioRED ex-NEG and BC5CDR DD F1 by encoder × schedule under S_pair. Mean ± 95 % CI across seeds; anchor cell highlighted; Phase A S_flat/S_mech rows inset for context. |
| **F3** | Per-family KB surfacing yield: per-family `KB_hit_A_setvalued` for the anchor and edge configurations; CIViC baseline overlay. |
| **F4** | RQ4 evaluation-validity audit — variance-asymmetry headline. (a) Variance-share bar chart for BioRED ex-NEG vs `KB_hit_A` vs BC5CDR DD. (b) Ordinal-instability histogram: ΔKB_hit_A between configurations within 1 × within-cell BioRED SD of each other. |
| **F5** | RQ4 — mechanism-stratified coupling. (a) Forest plot of the five H6 slopes with 95 % CIs and three-bin shading. (b) Cell-level scatter of BioRED × `KB_hit_A` across phases, β_config and β_schema lines overlaid. |

### 9.3 Table plan (3 main tables)

| Table | Content |
|---|---|
| **T1** | Data and schema inventory: T1/T2/T3/T4 counts; per-schema label list; active-head support per schema; BioRED test support per head. |
| **T2** | Phase B main benchmark results under S_pair: one row per cell, columns BioRED ex-NEG, BioRED active-head macro, BC5CDR DD, KB_hit_A_sv, KB_pmass_B, KB_auc_C; mean ± SE over seeds. RB reference row included. |
| **T3** | Evaluation-validity audit summary: H7 variance decomposition with R_A (descriptive) and R_B (confirmatory + CI); H6 mechanism-stratified slope table with point estimate, 95 % CI, CI-width-gate status, three-bin label, and active-head re-fit. |

---

## Section 10 — Limitations

**W1 — DrugProt official test absent.** DrugProt's processed package
does not include the official test split, so DrugProt external
evaluation is not reported. DrugProt contributes as T1 training data
only.

**W2 — Absolute BioRED macro-F1 is low (≈ 0.28).** BioRED test has
extreme class imbalance under the coarse schemas; even a perfect AG +
NEG classifier ceilings at macro-F1 = 0.50. Random (0.25) and majority
(0.20) baselines are reported alongside, so observed 0.27–0.35 is
interpretable as the gain over chance under the imbalanced label
distribution.

**W3 — BC5CDR per-seed SDs are moderate (0.02–0.06).** We report mean
± SE throughout, annotate per-seed variability in Supplementary, and
use paired tests so that cross-seed noise is differenced out.

**W4 — KB-audit measures recall against CIViC, not universal oncology
assertion truth.** KB_hit_A counts how many of the 162 evaluable
CIViC-curated targets (of 165 anchored, with 3 unmapped to the schema
vocabularies) the model surfaces correctly; assertions absent from
CIViC are not assessed. The metric is therefore a lower bound on
surfacing yield with respect to CIViC.

**W5 — n = 162 evaluable KB targets is small.** Per-target surfacing
is reported with Wilson binomial CIs; per-family breakdown (n ≈ 20–90)
is descriptive, not primary significance evidence.

**W6 — No human audit of KB-consistency labels.** The schema-expected-
label mapping is documented in `schema_expected_label_mapping_rationale
.md`; a human re-audit is future work. The set_valued / single_label
duality is specifically designed to bracket the mapping's uncertainty.

**W7 — No LLM baseline.** This study is restricted to encoder-based
relation extraction; comparison with LLM-paradigm assertion extraction
is reserved for follow-up work.

**W8 — Benchmark and KB metrics are positively correlated.** The Phase
A 120-run data show a positive seed-level Pearson r = +0.52 [+0.40,
+0.64]. The central finding is therefore framed as variance asymmetry
(H7) and ordinal instability (Figure 4), which are compatible with
positive coupling and are the aspects directly relevant to model-
selection practice.

---

## Appendix A — Reproducibility Materials

**Code layout (paths under repository root):**

```
project_1/
├── data_pipeline/                              # leakage fix, MeSH C04 projection, shard generation
├── training_data_generation/                   # canonical shard files (scratch)
├── schema_exploration/definitions/             # S_flat / S_pair / S_mech projection functions
├── oncology_projection/                        # MeSH C04 audit and manifests
├── knowledge_grounded_evidence_audit/          # CIViC 165 targets and inference helper
├── external_evaluation/                        # BioRED + BC5CDR loaders / pair builders
└── fine_tuning_experiments/
    ├── train/run_experiment.py                 # shared entry point (Phase A + Phase B)
    ├── phase_b/trainer/                        # scientific_trainer.py, scientific_data.py, tests/
    ├── schema_exp/                             # Phase A configs, sbatch, runs_inventory, analysis
    │   └── eval/                               # pinned v1.0 eval pipeline + EVAL_VERSION.txt
    └── runs/                                   # run artifacts on Lustre scratch
```

Shard files and run artifacts live on the Lustre scratch filesystem
(`/lus/lfs1aip2/projects/b5ac/project_1/...`); the repository tracks
code only.

**Pre-registration anchors.**

*Lock v1 (initial pre-registration, frozen 2026-04-16):*

- Lock tag: `phase_b_prelock_v1`
- Lock commit: `fba3d7149d6ae0420468b6c5071f4b5d7be00c3f`
- Document SHA-256 (body of `paper_development_design_locked_v1.md`):
  `c38f45e5f0dca366a7e0e9d494c622d180f424028ca159dd4b4a897ca7372b0d`
- Tarball SHA-256 (`phase_b_prelock_v1.tar.gz`):
  `12b65c4be97faa64ee21c14a56bee3b2eb52b5129b5c9403648b6602f05af043`

*Lock v2 (consolidating amendments B.10–B.23, frozen 2026-04-27):*

- Lock tag: `phase_b_prelock_v2`
- Lock commit: resolvable as `git rev-parse phase_b_prelock_v2` after fetch
- File-level SHA-256 manifest: `prelock_v2_shas.txt` (committed at the same commit as this Appendix; covers `paper_methods_draft.md`, `paper_development_design.md`, `paper_development_design_locked_v1.md`, `fine_tuning_experiments/phase_b/analysis/analyze_phase_b.py`, `fine_tuning_experiments/phase_b/analysis/tests/test_phase_b_analysis.py`, `fine_tuning_experiments/phase_b/sbatch/phase_b_lora_d3_smoke.sbatch`, `fine_tuning_experiments/schema_exp/eval/EVAL_VERSION.txt`)
- Tarball: `/lus/lfs1aip2/projects/b5ac/backups/phase_b_prelock_v2.tar.gz`, with companion `phase_b_prelock_v2.tar.gz.sha256` next to it on the Lustre archive volume
- The committed manifest content is computed over file states immediately prior to the Appendix-A insertion of these references (the standard solution to the SHA-paste recursion); the lock tag and commit hash are the authoritative reproducibility anchors and are unaffected.

The locked v1 document remains the immutable reference for the original pre-registration. v2 adds analysis-specification refinements and prose precision (B.10–B.23 in `paper_development_design.md`'s Appendix B); no Phase A or Phase B FT run output is altered. The paper's full pre-registered design at v2 is the union of `paper_development_design_locked_v1.md` (v1 body, untouched) and the B.10–B.23 amendment block. All referenced files are part of the paper's reproducibility package.