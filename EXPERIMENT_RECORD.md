# Experiment Record — Biomedical Relation Encoder Evaluation Project

**Purpose:** Single reference for an LLM collaborator unfamiliar with the codebase. Describes completed experiments, protocols, results, and file locations. All numeric values are taken from on-disk artifacts unless explicitly marked otherwise.

**Code root:** `project_1/` (this repository)  
**Artifact root:** `../projects/project_1/` (data, outputs, figures, reports)  
**Last verified:** 2026-07-02 (post–BlueBERT naming fix; analyze + figures regenerated)

---

## 1. Project overview

This project fine-tunes nine biomedical and general-purpose transformer encoders on pooled BioRED + DrugProt relation-presence data, then evaluates each model on two axes: (1) **in-distribution benchmark** — relation-presence F1 on held-out training-corpus test splits; (2) **out-of-distribution knowledge-base (KB) ranking** — mean reciprocal rank (MRR) of clinically curated CIViC gene–drug and gene–disease pairs within a frozen PubTator3-derived candidate pool. Three analyses use these measurements: cross-encoder comparison (Round 1), within-model training-dynamics trajectories across per-epoch checkpoints (Round 2), and qualitative error characterization on missed CIViC positives.

**Three-stage design:**
- **Stage A — Data & training:** Build CIViC evaluation targets (1812), training corpora alignment, PubTator3 candidate pool, marker quality gate, learning-rate sweep, full 9×8 fine-tuning matrix at confirmed recipe (5e-6, no warmup).
- **Stage B — Two-axis evaluation:** Score all fine-tuned and untrained baselines on benchmark F1 and CIViC ranking (MRR, recall@k, AUPRC).
- **Stage C — Three analyses:** (1) cross-model variance and benchmark–KB association; (2) epoch-wise training dynamics with robustness checks; (3) qualitative error taxonomy on worst-ranked curated positives.

**Study design figure:** `assets/figures/study_design.png` — **not found** in the repository at time of writing (see Section 11). Conceptual flow is: CIViC targets → frozen pool → fine-tune encoders → dual-axis scoring → three analyses.

---

## 2. Data sources

### 2.1 Training corpora (BioRED, DrugProt)

**Sources and loading**

| Corpus | HuggingFace ID | BigBio config | Loader |
|--------|----------------|---------------|--------|
| BioRED | `bigbio/biored` | `biored_bigbio_kb` | `shared/benchmark_eval.py` L26–L54; `01_corpus_relevance/inventories.py` L67–L162 |
| DrugProt | `bigbio/drugprot` | `drugprot_bigbio_kb` | same |
| BC5CDR (reference only) | `bigbio/bc5cdr` | `bc5cdr_bigbio_kb` | `01_corpus_relevance/inventories.py` |

`datasets` library version **2.21.0** (`data/01_corpus_relevance/corpus_inventories.json`).

**Document splits (relation counts are across all splits; training uses train+val)**

| Corpus | Train docs | Val docs | Test docs | Total relations (all splits) |
|--------|------------|----------|-----------|------------------------------|
| BioRED | 400 | 100 | 100 | 128,460 |
| DrugProt | 3,500 | 750 | 10,750 (test_background) | 21,035 |
| BC5CDR | 500 | — | — | 47,813 |

**Train+validation relation counts used for fine-tuning** (`outputs/01_corpus_relevance/corpus_train_stats.json`):

| Corpus | gene-drug | gene-disease | variant-disease | variant-drug | Total |
|--------|-----------|--------------|-----------------|--------------|-------|
| BioRED | 16,049 | 31,834 | 6,792 | 717 | 107,548 |
| DrugProt | 21,035 | 0 | 0 | 0 | 21,035 |

**Entity-type alignment (corpus → CIViC role)** — `01_corpus_relevance/entity_normalization.py` L7–18:

| Raw entity label | CIViC role |
|------------------|------------|
| BioRED `GeneOrGeneProduct`; DrugProt `GENE`, `GENE-Y`, `GENE-N` | gene |
| BioRED `ChemicalEntity`; DrugProt `CHEMICAL` | drug |
| BioRED `DiseaseOrPhenotypicFeature` | disease |
| BioRED `SequenceVariant` | variant |

**Relation-type handling:** Training uses **relation presence** (binary positive/negative) on gene-drug and gene-disease pairs only (`shared/constants.py` L12–13: `TRAIN_PAIR_TYPES`). BioRED labels (Association, Positive_Correlation, etc.) are not mapped 1:1 to CIViC clinical significance; DrugProt mechanistic labels are documented as ambiguous or unmapped (`outputs/01_corpus_relevance/01_corpus_drugprot_mapping_summary.csv`: 0 clean, 6 ambiguous, 8 no_mapping).

**Leakage exclusion:** Three PMIDs overlap between DrugProt training and CIViC evaluation inventory — `16434489`, `18794803`, `23430109` (`shared/constants.py` L5; `outputs/01_corpus_relevance/excluded_pmids.json`). **31** DrugProt training relations removed.

**BioRED test benchmark size:** 14,395 examples, 8,875 positives (`outputs/05_marker_quality_gate/quality_gate_results.json`, check `benchmark_positive_examples_built`).

### 2.2 CIViC evaluation set

**CIViC version:** `nightly` release; fetched **2026-06-03T14:23:45 UTC** via GraphQL `https://civicdb.org/api/graphql` (`data/00_civic_feasibility/fetch_metadata.json`; `outputs/02_evaluation_protocol/frozen_protocol.json`).

**Filtering pipeline** (counts from `outputs/00_civic_feasibility/`):

| Step | Criterion | Count remaining | Source |
|------|-----------|-----------------|--------|
| 0 | Accepted evidence items | 4,856 | `evaluable_target_summary.csv` |
| 1 | PubMed-sourced + two distinct entity types (evaluable) | 4,674 | same |
| 2 | Abstract-grounded (both entities mentioned in abstract) | 2,074 | `abstract_alignment_summary.csv` |
| 3 | Freeze: gene-drug + gene-disease only; exclude variant pairs | **1,812** | `frozen_protocol.json` |

**Entity-pair breakdown at step 1** (`entity_pair_breakdown.csv`): gene-drug 2,499; gene-disease 1,433; variant-disease 420; variant-drug 322.

**Not abstract-grounded at step 2:** 2,600 items (head absent 646, tail absent 1,451, both absent 503).

**Final frozen targets** (`outputs/02_evaluation_protocol/frozen_protocol.json`):

| Quantity | Value |
|----------|-------|
| Total evaluable ranking targets | **1,812** |
| gene-drug | **1,230** |
| gene-disease | **582** |
| Unique PMIDs | **915** |
| Variant pairs excluded from evaluation | **262** |

**Train–eval leakage check** (`outputs/01_corpus_relevance/pmid_leakage.csv`): BioRED overlap **0**; DrugProt overlap **3** PMIDs (removed from training as above). Eval inventory: 1,079 unique PMIDs, 2,074 ranking-target PMIDs (`pmid_diagnostics.json`).

**Oncology coverage** (`outputs/01_corpus_relevance/oncology_criteria_agreement.csv`): On BioRED gene-disease train+val (31,834 relations), strict intersection of three oncology criteria = **1,086** (3.41%). Per-criterion fractions: disease_neoplasm 21.6%, gene_civic 13.2%, literature_mesh 32.0% (`oncology_fractions_by_criterion.csv`).

### 2.3 Candidate pool construction

**NER tool:** PubTator3 precomputed annotations; API `https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson`; fetched **2026-06-03T15:21:26 UTC**; 1,079/1,079 PMIDs cached (`data/03_candidate_pool/pubtator3_fetch_metadata.json`).

**Construction rule:** For each abstract, enumerate all co-occurring PubTator entity pairs by allowed type combinations (gene×drug, gene×disease). A candidate is positive if it matches a frozen CIViC target under normalized entity matching. Variant pairs are descriptive only. Code: `03_candidate_pool/pool_builder.py` L169–244.

**Co-sentence vs cross-sentence:** `sentence_distance = |head_sentence_idx − tail_sentence_idx|` from PubTator character offsets (`shared/distance_analysis.py` L28–67). `easy_co_sentence`: distance == 0; `hard_cross_sentence`: distance > 0.

**Pool composition** (`outputs/03_candidate_pool/03_candidate_pool_composition.csv`):

| Scope | pair_type | candidates | CIViC+ candidates |
|-------|-----------|------------|-------------------|
| primary | gene-drug | 5,165 | 781 |
| primary | gene-disease | 13,746 | 385 |
| **primary total** | — | **18,911** | **1,166** |

**Coverage of 1,812 frozen targets** (`outputs/03_candidate_pool/03_candidate_pool_pubtator_recall_buckets.csv`):

| Bucket | n | Fraction of 1,812 |
|--------|---|-------------------|
| Matched (positive in pool) | **1,590** | **87.7%** |
| Miss: entity type absent in abstract | 183 | 10.1% |
| Miss: entity present but string/span mismatch | 39 | 2.2% |
| **Total misses** | **222** | **12.3%** |

**Trivial baselines on primary pool** (`outputs/03_candidate_pool/ranking_baselines.csv`):

| Baseline | MRR |
|----------|-----|
| random | 0.322 |
| distance_ranker | 0.489 |

---

## 3. Models

### 3.1 Encoder list

From `shared/models.py` and `20_round2_diagnostic/encoder_properties.py`:

| model_id | Display name | HuggingFace path | Params (M) | Biomed pretrain | Pretraining source |
|----------|--------------|------------------|------------|-----------------|-------------------|
| pubmedbert_base | PubMedBERT-base | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | 110 | Yes | PubMed abstracts |
| bluebert_base | BlueBERT-base | `bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12` | 110 | Yes | PubMed + MIMIC |
| biolinkbert_base | BioLinkBERT-base | `michiyasunaga/BioLinkBERT-base` | 110 | Yes | PubMed + link prediction |
| biobert_base | BioBERT-base | `dmis-lab/biobert-base-cased-v1.2` | 110 | Yes | PubMed + PMC |
| scibert_base | SciBERT | `allenai/scibert_scivocab_uncased` | 110 | Yes | Semantic Scholar biomedical+scientific |
| roberta_base | RoBERTa-base | `roberta-base` | 125 | No | BookCorpus + Wikipedia |
| bert_base | BERT-base | `bert-base-uncased` | 110 | No | BookCorpus + Wikipedia |
| distilbert_base | DistilBERT-base | `distilbert-base-uncased` | 66 | No | Distilled BERT |
| deberta_base | DeBERTa-base | `microsoft/deberta-base` | 100 | No | General English NLU |

### 3.2 Fine-tuning recipe

**Confirmed matrix recipe:** learning rate **5e-6**, warmup **none** (ratio 0.0), max **10** epochs, **8** seeds (42–49).

| Hyperparameter | Value | Code location |
|----------------|-------|---------------|
| Learning rate | 5e-6 | `shared/models.py` L7; matrix `recipe_lr` in checkpoints |
| Warmup | none (0.0) | `shared/models.py` L8–11; `shared/train_core.py` L145–148 |
| Max epochs | 10 | `shared/constants.py` L20 |
| Seeds | 42–49 (8) | `shared/constants.py` L8 |
| Train batch size | 16 | `shared/constants.py` L17 |
| Infer batch size | 32 | `shared/constants.py` L18 |
| Max sequence length | 256 | `shared/constants.py` L18 |
| Max train examples | 24,000 | `shared/constants.py` L15 |
| Negatives per positive | 2 | `shared/constants.py` L16 |
| Optimizer | AdamW | `shared/train_core.py` L144 |
| Early-stopping patience | 3 | `shared/constants.py` L21 (matrix runs completed all 10 epochs) |
| Checkpoint criterion | Pooled validation F1 micro across BioRED+DrugProt | slurm logs; `shared/constants.py` L22 |
| Checkpoint policy | fp32 `best/` at val-best; fp16 per-epoch under `epochs/epoch_NN/` | `shared/train_core.py` L100–199 |

**Entity markers:** `[E1]` / `[E2]` inserted at annotated character offsets (fallback: string match, then wrap). Code: `shared/marker_insert.py` L49–88 (`insert_entity_markers`, `format_marked_pair`).

**Marker quality gate (step 05):** `outputs/05_marker_quality_gate/quality_gate_results.json` — **overall_pass: true**. Training offset insertion **100%** (64,452 relations); same-sentence rate **39.3%** under native offsets; CIViC pool offset insertion **97.3%**; benchmark offset **100%**.

**Run count:** 9 encoders × 8 seeds = **72** fine-tuned runs. Per-epoch checkpoints: **720** recoverable (`outputs/20_round2_diagnostic/20_checkpoint_inventory.csv`, sum of `n_recoverable_checkpoints`).

**Learning-rate sweep (step 10):** Four LRs {5e-6, 1e-5, 2e-5, 3e-5} × {none, 10% warmup} on 4 sweep encoders, seed 42 (`outputs/10_recipe_sweep_and_training/sweep/recipe_decision_table.csv`).

| Recipe | Spread (seed 42) | DeBERTa F1 | Stable? |
|--------|------------------|------------|---------|
| 3e-5/none | 0.0667 | 0.763 | Yes (advisory winner on spread) |
| 5e-6/none | 0.0259 | 0.734 | Yes |
| 3e-5/warmup | 0.7205 | **0.000** | No (DeBERTa collapse) |
| 2e-5/none | 0.7376 | **0.000** | No |

**Final choice 5e-6/none:** Initial full-matrix run at 3e-5/none failed eight-seed DeBERTa stability gate (seeds 45–46 collapsed). Confirmed 5e-6/none matrix: **72/72** runs complete; DeBERTa F1 range 0.731–0.754 across seeds; encoder-mean benchmark spread **0.035** (BlueBERT 0.711 → BioLinkBERT 0.746). See `reports/10_recipe_sweep_and_training/report.md`.

### 3.3 Pre-trained baseline (no fine-tuning)

**Protocol** (`11_round1_analysis/score_untrained.py`): Pretrained HF weights + randomly initialized classification head (head seed **4242**); no gradient updates. Nine references (`untrained_{model_id}`). Scored on same BioRED test benchmark and frozen CIViC pool as fine-tuned runs. Entry: `python 11_round1_analysis/run.py --score-untrained-only`.

**Role:** Quantifies fine-tuning lift (`outputs/11_round1_analysis/11_untrained_floor_lift.csv`).

---

## 4. Evaluation protocol

### 4.1 In-distribution benchmark axis

**Metric:** Relation-presence **F1** on held-out test splits (micro-averaged positives/negatives).

- **Gene-disease F1:** BioRED test only.
- **Gene-drug F1 (combined):** BioRED + DrugProt test pooled micro-average (DrugProt holdout unavailable in some runs → combined equals BioRED gene-drug only).
- **Overall benchmark F1:** Pooled across pair types on BioRED test.

**Computation:** `shared/benchmark_eval.py` — builds marked examples via `format_marked_pair`, runs `AutoModelForSequenceClassification`, sklearn `f1_score` (L12–L13, L26–L54+).

### 4.2 Out-of-distribution ranking axis

**Primary metric:** **MRR** — per abstract, reciprocal rank of best-ranked CIViC-curated positive; macro-averaged over abstracts with ≥1 positive.

**Auxiliary metrics:** recall@k for k ∈ {1, 3, 5}; global **AUPRC** (`shared/constants.py` L10).

**Baselines:**
- **Random:** random scores with fixed tie-break seed 42.
- **Distance/proximity ranker:** score = `1 / (1 + |sentence_idx_head − sentence_idx_tail|)`; co-sentence → 1.0. Code: `03_candidate_pool/distance_ranker.py` L28–37 (`proximity_score`).

**Ranking scope:** Scores computed per candidate; **ranking is within each abstract** (grouped by `pmid`), then MRR macro-averaged across abstracts. Pair-type subsets (gene-drug, gene-disease) and easy/hard subsets computed on filtered score tables.

**Computation:** `shared/metrics_ranking.py` — `_rank_within_abstracts` L13–24, `compute_mrr` L39–41, `compute_recall_at_k` L44–54, `compute_global_auc_pr` L57–61, `ranking_metrics_for_scores` L64–75.

---

## 5. Analysis 1: Cross-model comparison across the two axes

**Step:** `11_round1_analysis/` on 72 clean fine-tuned runs + 9 untrained references.

### 5.1 Variance decomposition

**Method:** ICC-style variance decomposition partitioning total variance into between-encoder and within-encoder (seed) components (`11_round1_analysis/analysis.py` L211–309). Bootstrap N=5000 over encoder clusters (`shared/constants.py` L28).

**Results** (`outputs/11_round1_analysis/11_variance_components.csv`):

| metric | ICC | encoder share | seed share | between_encoder_sd | within_encoder_sd |
|--------|-----|---------------|------------|--------------------|-------------------|
| benchmark_f1 | 0.570 | **53.4%** | **46.6%** | 0.0125 | 0.0108 |
| kb_mrr_gene_drug | 0.374 | **36.1%** | **63.9%** | 0.0126 | 0.0162 |
| kb_mrr_gene_disease | 0.530 | **49.3%** | **50.7%** | 0.0470 | 0.0443 |

**Bootstrap 95% CIs on encoder share** (`11_variance_components_bootstrap.csv`):

| metric | encoder_share | CI lo | CI hi |
|--------|---------------|-------|-------|
| kb_mrr_gene_drug | 0.361 | 0.153 | 0.503 |
| kb_mrr_gene_disease | 0.493 | 0.214 | 0.712 |

Note: `benchmark_f1` has no row in bootstrap CSV (gene-disease and gene-drug split rows exist).

### 5.2 Correlation between the two axes

**Primary method:** Seed-level cluster bootstrap over 72 runs (`11_benchmark_kb_seed_association.csv`):

| pair_type | Spearman ρ | 95% CI lo | 95% CI hi |
|-----------|------------|-----------|-----------|
| gene-drug | **+0.018** | −0.232 | +0.289 |
| gene-disease | **−0.440** | −0.660 | −0.089 |

**Encoder-mean n=9 check** (`11_benchmark_kb_correlations.csv`, Spearman):

| pair_type | estimate | CI lo | CI hi |
|-----------|----------|-------|-------|
| gene-drug | −0.15 | −0.821 | +0.670 |
| gene-disease | −0.55 | −0.982 | +0.222 |

### 5.3 Fine-tuning lift (cross-sectional, Round 1)

From `11_untrained_floor_lift.csv` (mean across 9 encoders):

| Quantity | Mean lift |
|----------|-----------|
| Benchmark F1 | **+0.179** |
| KB MRR gene-drug | **+0.121** |
| KB MRR gene-disease | **+0.168** |
| KB MRR (pair-type average) | **+0.144** |

### 5.4 Absolute KB levels

`outputs/11_round1_analysis/11_absolute_kb_levels.csv`:

| reference | MRR overall | gene-drug | gene-disease | hard | easy |
|-----------|-------------|-----------|--------------|------|------|
| random_uniform | 0.322 | — | — | — | — |
| distance_ranker | 0.489 | — | — | 0.369 | 0.692 |
| finetuned_encoders_mean | 0.642 | 0.670 | 0.614 | 0.392 | 0.728 |

**Encoder-mean benchmark spread:** 0.035 (min 0.711 BlueBERT, max 0.746 BioLinkBERT; `11_benchmark_f1_range.csv`).

**Hard-subset validity:** 8/9 encoders beat distance ranker on hard cross-sentence subset (seed-averaged; `reports/11_round1_analysis/report.md`).

### 5.5 Figures associated (Analysis 1)

| Figure | Path | Generator | Purpose |
|--------|------|-----------|---------|
| Fig 1 | `../projects/project_1/figures/11_round1_analysis/fig1_benchmark_kb_scatter.png` | `11_round1_analysis/figures.py` → `figure1_benchmark_kb_scatter`; also `manuscript_regenerate/figures.py` → `regenerate_step11` | Two-panel scatter: encoder-mean benchmark F1 (x) vs KB MRR (y) for gene-drug and gene-disease, with seed-uncertainty error bars. Shows weak/decoupled association between axes. |
| Fig 2 | `../projects/project_1/figures/11_round1_analysis/fig2_variance_between_encoder.png` | `regenerate_step11` | Bar chart of between-encoder variance share for benchmark F1, KB gene-drug MRR, KB gene-disease MRR, with bootstrap error bars where available. Benchmark axis discriminates encoders more than KB axes. |
| Fig 3 | `../projects/project_1/figures/11_round1_analysis/fig3_easy_hard_ranking_validity.png` | `regenerate_step11` | Side-by-side panels: encoder MRR vs distance-ranker baseline on easy (co-sentence) and hard (cross-sentence) subsets. Supports that learned models exceed proximity on hard subset. |
| Fig 4 | `../projects/project_1/figures/11_round1_analysis/fig4_finetuning_lift.png` | `regenerate_step11` | Per-encoder lift from untrained floor to fine-tuned on benchmark and KB axes. |

**Verbal description (Fig 1):** Two scatter panels share the x-axis (benchmark F1, ~0.71–0.75). Left panel plots gene-drug KB MRR (~0.65–0.69); right panel plots gene-disease KB MRR (~0.55–0.69). Nine encoders appear as colored points with horizontal/vertical error bars. No strong positive trend; gene-disease panel shows slight negative tendency at encoder means.

**Step 10 sweep figure:** `../projects/project_1/figures/10_recipe_sweep_and_training/sweep/recipe_spread_vs_deberta_health.png` — LR sweep spread vs DeBERTa health for recipe selection.

---

## 6. Analysis 2: Training dynamics

**Step:** `20_round2_diagnostic/` — scores all **720** per-epoch checkpoints (9 encoders × 8 seeds × 10 epochs) without retraining.

### 6.1 Trajectory measurement

For each encoder × seed, load epoch checkpoints 1–10 from `data/10_recipe_sweep_and_training/matrix/checkpoints/{model_id}/seed_{seed}/epochs/epoch_NN/`. At each epoch, compute benchmark F1 and KB MRR (overall, hard, easy, gene-drug, gene-disease). Code: `20_round2_diagnostic/epoch_scoring.py`, `20_round2_diagnostic/adjudication.py` → `build_trajectory_table`, `build_within_seed_paired_changes`.

**Paired change definition:** Epoch 1 → best-validation-F1 epoch (pooled val F1 micro). Requires `pairable_val_f1_best` = True (valid metrics at both endpoints, different epochs).

### 6.2 Aggregate findings

**Paired seeds:** **69** of 72 (3 seeds not pairable under val_f1_best definition).

From `outputs/20_round2_diagnostic/20_pair_type_breakdown.csv` and `20_seed_erosion_distribution.csv`:

| Quantity | Value |
|----------|-------|
| Mean Δ benchmark F1 (epoch1→best val) | **+0.0219** |
| Mean Δ KB hard MRR | **−0.0083** |
| Mean Δ KB gene-drug MRR | **+0.0042** (41/69 seeds fall) |
| Mean Δ KB gene-disease MRR | **−0.0541** (46/69 seeds fall) |
| Erosion pattern (bench↑ & KB-hard↓) | **36/69 (52.2%)** |

**Hard vs easy** (`20_hard_easy_breakdown.csv`):

| subset | mean Δ KB MRR | 95% CI | falls |
|--------|---------------|--------|-------|
| hard_cross_sentence | −0.0083 | [−0.0148, −0.0013] | 47/69 |
| easy_co_sentence | −0.0027 | [−0.0099, +0.0045] | 41/69 |

**Biomedical vs general encoders (gene-disease-hard, val_f1_best)** — `20_gene_disease_encoder_breakdown.csv`:

| Group | Encoders | mean Δ KB MRR (hard) | Pattern |
|-------|----------|----------------------|---------|
| Biomedical-pretrained | PubMedBERT, BlueBERT, BioLinkBERT, BioBERT, SciBERT | −0.0225, **+0.0144**, −0.0637, −0.0056, −0.0776 | Mixed; BlueBERT positive |
| General-purpose | RoBERTa, BERT, DistilBERT, DeBERTa | −0.0016, +0.0161, +0.0164, +0.0062 | Mostly flat/positive on hard |

5/9 encoders show negative mean gene-disease-hard paired change.

**Encoder-property correlation (exploratory, n=9):** Spearman ρ benchmark F1 vs erosion **+0.800 (p=0.010)**; biomedical pretrain **+0.693 (p=0.039)** — stated in `reports/20_round2_diagnostic/report.md` (not stored as standalone CSV).

### 6.3 Three robustness checks

**Check 1 — Endpoint choice** (`20_gene_disease_robustness.csv`, gene-disease all):

| Well-trained definition | mean Δ KB MRR | frac seeds fall |
|-------------------------|---------------|-----------------|
| best validation F1 | **−0.0541** | 66.7% (46/69) |
| last saved epoch | **−0.0835** | 76.4% (55/72) |
| fixed epoch 5 | **−0.0529** | 68.1% (49/72) |

Gene-disease-hard: −0.0126 / −0.0349 / −0.0269 respectively. Direction (decline) consistent across definitions.

**Check 2 — Peak timing** (`20_kb_peak_timing_summary.csv`, gene-disease):

| Timing vs best-val epoch | n seeds | fraction |
|--------------------------|---------|----------|
| KB peak before best val | 58 | **80.6%** |
| coincident | 13 | 18.1% |
| after | 1 | 1.4% |

**Check 3 — Pool size** (`20_gene_disease_pool_stratum_summary.csv`):

| Stratum | mean Δ gene-disease MRR | falls |
|---------|-------------------------|-------|
| small_pool | −0.0436 | 46/69 |
| large_pool | −0.0729 | 47/69 |
| comparable_to_gene_drug | −0.0349 | 46/69 |

Decline persists in pool-size-matched stratum. Bootstrap P(negative) for gene-disease-hard paired delta: **98.2%** (`reports/20_round2_diagnostic/report.md`).

### 6.4 Figures associated (Analysis 2)

All under `../projects/project_1/figures/20_round2_diagnostic/`, generated by `20_round2_diagnostic/figures.py` → `generate_all_figures` (2026-07-02).

| Figure | Function | Purpose |
|--------|----------|---------|
| fig1_per_seed_trajectories.png | `figure1_per_seed_trajectories` | 3×3 grid: per-encoder benchmark F1 (blue) and KB hard MRR (red) vs epoch; faint per-seed lines + bold means. Shows heterogeneous trajectories. |
| fig2_within_seed_paired_change.png | `figure2_paired_change_distribution` | Scatter/violin of within-seed Δbenchmark vs ΔKB hard (epoch1→best val). |
| fig3_hard_easy_pair_type.png | `figure3_hard_easy_pair_type` | Bar/point summaries of KB MRR change by hard/easy and gene-drug/gene-disease. |
| fig4_robustness_well_trained.png | `figure4_robustness_well_trained` | Erosion fraction under three checkpoint definitions. |
| fig5_gene_disease_hard_trajectories.png | `figure5_gene_disease_hard_trajectories` | 3×3 grid focused on gene-disease-hard MRR vs benchmark across epochs. Key figure for training-dynamics claim. |
| fig6_pair_type_subset_contrast.png | `figure6_pair_type_subset_contrast` | Contrasts pair-type × subset paired changes. |
| fig7_kb_peak_timing.png | `figure7_kb_peak_timing` | Distribution of KB peak timing relative to validation-best epoch. |
| fig8_pool_stratum_gene_disease.png | `figure8_pool_stratum` | Gene-disease paired change by pool-size stratum. |
| fig9_encoder_property_scatter.png | `figure9_encoder_property_scatter` | Encoder properties vs erosion magnitude (exploratory). |
| fig10_failure_mode_summary.png | `figure10_failure_modes` | Qualitative error mode rates among genuine errors. |

**Verbal description (fig5):** Nine panels (PubMedBERT, BlueBERT, BioLinkBERT, BioBERT, SciBERT, RoBERTa, BERT, DistilBERT, DeBERTa). X-axis: training epoch 1–10. Left y-axis: BioRED test F1 (blue circles, solid mean line). Right y-axis: gene-disease-hard CIViC MRR (red squares). Faint lines show individual seeds; bold lines show means. Biomedical encoders often show rising benchmark with flat or declining KB MRR; general encoders more stable.

---

## 7. Analysis 3: Qualitative error analysis

**Step:** `20_round2_diagnostic/qualitative_errors.py` → `run_qualitative_errors`.

### 7.1 Setup

| Parameter | Value |
|-----------|-------|
| Representative seed | **42** |
| Checkpoint | Folder-11 **validation-best** per encoder |
| Score aggregation | **Median** across nine encoders at seed 42 |
| Case selection | Per abstract, worst-ranked curated positive (missed_positive) and false-high negatives |

### 7.2 Abstract grounding filters

**CIViC target grounding (step 00):** 4,674 evaluable → 2,074 abstract-grounded (both entities in abstract). This is upstream of the 1,812 frozen targets.

**Qualitative "abstract-unsupported" filter:** Among 852 missed positives, **34 (4.0%)** flagged `abstract_unsupported` — curated relation not supported by abstract text under offset-based sentence grounding (`20_qualitative_error_summary.csv`). **818** classified as genuine model errors.

### 7.3 Error categories and counts

`20_qualitative_error_summary.csv`:

| Category | Count |
|----------|-------|
| Total cases | 1,699 (852 missed_positive + 847 false_high) |
| Missed positives | 852 |
| Genuine model errors | 818 |
| Abstract-unsupported | 34 |
| False highs | 847 |

**Failure modes among genuine errors** (`20_qualitative_error_patterns.csv`, n=818):

| Pattern | Rate |
|---------|------|
| cross_sentence_hard | **47.8%** |
| multiword_entity | 14.7% |
| publication_year_before_2010 | 18.9% |
| gene_disease_pair | 30.9% |

**Flagged for manual reading:** 24 cases (`20_qualitative_errors_flagged_manual.csv`).

### 7.4 Illustrative examples

From `20_qualitative_errors_flagged_manual.csv` (missed positives, seed-42 median ranking):

1. **KIT – GIST** (PMID 10485475, gene-disease, rank 1/6, easy co-sentence): Model scores a broad co-occurring pair highly but misses clinically specific disease naming.
2. **PTEN – tumor** (PMID 10866302, gene-disease, rank 1/12, easy co-sentence): Tumor suppressor context; generic "tumor" tail vs curated specificity.
3. **EGFR – Iressa** (PMID 15118125, gene-drug, rank 1/5, easy co-sentence): Drug name in abstract; model fails to rank curated gene-drug pair top despite co-sentence co-occurrence.

**Report:** `../projects/project_1/reports/20_round2_diagnostic/report_qualitative_errors.md`

**Figure:** `fig10_failure_mode_summary.png` — bar chart of failure-mode proportions among 818 genuine errors; cross-sentence hardness dominant (~48%).

---

## 8. Auxiliary / supporting analyses

### 8.1 Learning-rate sweep summary

See Section 3.2. Tested 5e-6–3e-5 × {none, 10% warmup}. **Rejected:** 3e-5/warmup and 2e-5/none (DeBERTa F1=0 seed 42); 3e-5/none failed eight-seed DeBERTa gate. **Selected:** 5e-6/none for full matrix — stable DeBERTa, genuine encoder spread 0.035, 72/72 completions.

### 8.2 Marker placement quality gate

Step 05 (`outputs/05_marker_quality_gate/quality_gate_results.json`): **PASS**. 100% native offset insertion on 64,452 training positives; 39.3% same-sentence rate; 97.3% CIViC pool offset success; 0 offset reversals.

### 8.3 Exploratory pilot (step 04)

3 encoders × 3 seeds, ~3,000 steps, pre-marker-fix pipeline. Best MRR PubMedBERT **0.469** < distance ranker **0.489**. Used to validate framework before marker gate; not used for final numbers.

### 8.4 Second knowledge-base probe (step 06 — OncoKB)

`06_oncokb_feasibility/`: Verdict **GO** for API access. 1,010 genes queried; 678 unique associations; 244 evaluable gene-drug + 76 gene-disease single-PMID pairs; 302 retrievable abstracts. **Conclusion:** Usable abstract-grounded subset too limited for primary KB axis; **CIViC remains primary** (noted in `manuscript_regenerate/reports_step11.py`).

---

## 9. Files and reproducibility map

| Role | Path |
|------|------|
| **Constants / encoders** | `shared/constants.py`, `shared/models.py` |
| **Training core** | `shared/train_core.py`, `shared/train_data.py` |
| **Entity markers** | `shared/marker_insert.py` |
| **Benchmark eval** | `shared/benchmark_eval.py` |
| **Ranking metrics** | `shared/metrics_ranking.py` |
| **CIViC inventory** | `00_civic_feasibility/inventory.py` (git HEAD) |
| **Freeze targets** | `02_evaluation_protocol/targets.py` (git HEAD) |
| **Candidate pool** | `03_candidate_pool/build_pool.py`, `pool_builder.py` |
| **Marker gate** | `05_marker_quality_gate/checks.py` (git HEAD) |
| **LR sweep + matrix train** | `10_recipe_sweep_and_training/` (orchestration in git HEAD / slurm logs) |
| **Round 1 scoring** | `11_round1_analysis/run.py`, `score_runs.py`, `run_analysis.py` |
| **Round 1 untrained** | `11_round1_analysis/score_untrained.py` |
| **Round 1 figures** | `11_round1_analysis/figures.py`; `manuscript_regenerate/figures.py` |
| **Round 2 epoch scoring** | `20_round2_diagnostic/epoch_scoring.py` |
| **Round 2 analysis** | `20_round2_diagnostic/run.py` (--analyze-only) |
| **Round 2 figures** | `20_round2_diagnostic/figures.py` |
| **Qualitative errors** | `20_round2_diagnostic/qualitative_errors.py` |
| **Manuscript regen** | `manuscript_regenerate/run_all.py` |
| **Fine-tuned checkpoints** | `../projects/project_1/data/10_recipe_sweep_and_training/matrix/checkpoints/` |
| **Per-epoch scores (R2)** | `../projects/project_1/data/20_round2_diagnostic/scores/` |
| **Round 1 per-run scores** | `../projects/project_1/data/11_round1_analysis/scores/` |
| **Frozen CIViC targets** | `../projects/project_1/outputs/02_evaluation_protocol/ranking_targets.csv` |
| **Candidate pool** | `../projects/project_1/outputs/03_candidate_pool/pool_candidates.csv` |
| **Environment** | Conda env `hf-hpc` (miniforge3); no `requirements.txt` or `environment.yml` in repo |

**SLURM helpers:** `_slurm_env.sh`, `20_round2_diagnostic/step_*.sbatch`, `11_round1_analysis/step_*.sbatch`.

---

## 10. Key numbers, one-glance summary

Paper draft `.tex` files were **not found** in the repository (`sections-main/`, `sections-supp/` absent). "Paper draft value" column is blank or marked N/A unless a number appears in generated `reports/*/report.md` that may lag CSVs.

| Quantity | Latest value | Source file | Paper draft value | Match? |
|----------|-------------|-------------|-------------------|--------|
| CIViC accepted evidence items | 4,856 | `outputs/00_civic_feasibility/evaluable_target_summary.csv` | N/A | — |
| CIViC evaluable two-entity | 4,674 | same | N/A | — |
| CIViC abstract-grounded | 2,074 | `abstract_alignment_summary.csv` | N/A | — |
| CIViC frozen targets total | **1,812** | `outputs/02_evaluation_protocol/frozen_protocol.json` | 1,812 (typical) | — |
| CIViC gene-drug / gene-disease | **1,230 / 582** | same | 1,230 / 582 (typical) | — |
| Unique PMIDs (eval) | **915** | same | 915 (typical) | — |
| Pool coverage matched | **1,590 / 1,812** | `outputs/03_candidate_pool/03_candidate_pool_pubtator_recall_buckets.csv` | 1,590 / 1,812 | — |
| Pool misses | **222** | same | 222 | — |
| Primary pool candidates | **18,911** | `03_candidate_pool_composition.csv` | N/A | — |
| Fine-tuned runs | **72** | 9×8 matrix | 72 | — |
| Per-epoch checkpoints scored (R2) | **720** | `outputs/20_round2_diagnostic/20_checkpoint_inventory.csv` | N/A | — |
| Benchmark encoder variance share | **53.4%** | `outputs/11_round1_analysis/11_variance_components.csv` | N/A | — |
| KB gene-drug encoder variance share | **36.1%** | same | N/A | — |
| KB gene-disease encoder variance share | **49.3%** | same | N/A | — |
| Benchmark F1 spread (9 encoders) | **0.035** | `11_benchmark_f1_range.csv` | 0.025 (example in prompt) | **No** — latest 0.035 |
| Seed-level Spearman (gene-drug) | **+0.018** [−0.232, +0.289] | `11_benchmark_kb_seed_association.csv` | −0.36 (example in prompt) | **No** — verify against paper |
| Seed-level Spearman (gene-disease) | **−0.440** [−0.660, −0.089] | same | −0.46 (example in prompt) | Close |
| Fine-tuning Δ benchmark F1 (untrained→FT, mean) | **+0.179** | `11_untrained_floor_lift.csv` | 0.378 (example in prompt) | **No** — different metric scope |
| Fine-tuning Δ KB MRR gene-drug (untrained→FT) | **+0.121** | same | 0.121 (example in prompt) | Yes (if same definition) |
| Fine-tuning Δ KB MRR gene-disease (untrained→FT) | **+0.168** | same | N/A | — |
| Distance ranker MRR | **0.489** | `11_absolute_kb_levels.csv` | N/A | — |
| Finetuned mean KB MRR gene-drug | **0.670** | same | N/A | — |
| Finetuned mean KB MRR gene-disease | **0.614** | same | N/A | — |
| Pairable seeds (R2) | **69** | `20_pair_type_breakdown.csv` | N/A | — |
| Gene-disease KB Δ over training (epoch1→best val) | **−0.0541** (46/69 fall) | same | −0.057 (example in prompt) | Close |
| Gene-drug KB Δ over training | **+0.0042** | same | N/A | — |
| Pooled hard KB Δ over training | **−0.0083** | `20_hard_easy_breakdown.csv` | N/A | — |
| Gene-disease-hard KB Δ | **−0.0126** | `20_gene_disease_robustness.csv` | N/A | — |
| Bootstrap P(negative) gene-disease-hard | **98.2%** | `reports/20_round2_diagnostic/report.md` | 99.1% (stale README) | **No** |
| KB peak before best val (gene-disease) | **80.6%** seeds | `20_kb_peak_timing_summary.csv` | N/A | — |
| Missed positives (qualitative) | **852** | `20_qualitative_error_summary.csv` | N/A | — |
| Genuine model errors | **818** | same | N/A | — |
| Cross-sentence among genuine errors | **47.8%** | `20_qualitative_error_patterns.csv` | N/A | — |
| Leaked PMIDs removed | **3** | `outputs/01_corpus_relevance/excluded_pmids.json` | 3 | — |
| BioRED test positives (benchmark) | **8,875** | `quality_gate_results.json` | N/A | — |

---

## 11. Items flagged for verification

1. **`assets/figures/study_design.png` not found.** Section 1 references it provisionally. Confirm path or add figure.

2. **No paper `.tex` drafts in repository.** All "paper draft value" comparisons in Section 10 are incomplete. Numbers marked in user prompt (0.378, 0.025, −0.36, etc.) could not be verified against `sections-main/` or `sections-supp/`.

3. **Step 10 training orchestration scripts** (`10_recipe_sweep_and_training/run.py`, etc.) absent from working tree; logic confirmed via `shared/train_core.py` and slurm logs. Included provisionally.

4. **Steps 00, 02, 05 analysis modules** partially absent on disk (present in git HEAD). Counts verified from `outputs/` artifacts.

5. **Bootstrap P(negative) = 98.2%** and **encoder-property Spearman ρ** for Round 2 appear only in `reports/20_round2_diagnostic/report.md`, not in a dedicated CSV. Included from report; verify if paper cites these.

6. **Stale README template** (`20_round2_diagnostic/README.md` via `manuscript_regenerate/readmes.py`) lists pairable seeds=65, gene-disease Δ=−0.0569, bootstrap 99.1%. **Superseded by Jul 2 CSVs** (69 seeds, −0.0541, 98.2%). Do not use README for paper numbers.

7. **BlueBERT naming:** On-disk matrix uses `bluebert_base`; `shared/models.py` now matches. Older reports/slurm logs may say `biomedbert_base`. Latest figures and CSVs use **BlueBERT-base**.

8. **DrugProt test split:** `test_background` (10,750 docs) not used for benchmark F1 in current pipeline; gene-drug combined F1 equals BioRED gene-drug when DrugProt holdout unavailable.

9. **498 vs 720 epoch checkpoints:** Older slurm output and `manuscript_regenerate/provenance_checks.py` reference 498 recoverable checkpoints from an earlier partial scoring pass. **Latest inventory sums to 720.** Paper should use 720.

10. **ECE vs benchmark correlation (Round 1):** Report states Spearman ρ ≈ −0.833 at encoder means; `11_benchmark_ece_correlations.csv` uses pair-type rows, not `pair_type=calibration`. Value taken from generated report; confirm computation if cited.

---

## 12. Authoritative numeric claims catalogue (code-side reference for paper alignment)

> This section is a one-stop catalogue of every numeric claim that the code and result artifacts can currently support. The author will use it manually to check each corresponding sentence in the paper draft. This section does not claim to know what the paper currently says.

| # | Category | Claim (natural-language) | Authoritative value | Source file | Definition note | Confidence |
|---|----------|--------------------------|--------------------|-------------|-----------------|------------|
| 1 | Dataset counts | The CIViC pipeline count at step 'total_accepted_evidence_items' is 4856. | 4856 | outputs/00_civic_feasibility/evaluable_target_summary.csv | From nightly CIViC fetch 2026-06-03. | AUTHORITATIVE |
| 2 | Dataset counts | The CIViC pipeline count at step 'pubmed_sourced_items' is 4834. | 4834 | outputs/00_civic_feasibility/evaluable_target_summary.csv | From nightly CIViC fetch 2026-06-03. | AUTHORITATIVE |
| 3 | Dataset counts | The CIViC pipeline count at step 'evaluable_abstract_two_entity' is 4674. | 4674 | outputs/00_civic_feasibility/evaluable_target_summary.csv | From nightly CIViC fetch 2026-06-03. | AUTHORITATIVE |
| 4 | Dataset counts | At abstract-alignment filtering, 2074 evidence items have status 'both_present'. | 2074 | outputs/00_civic_feasibility/abstract_alignment_summary.csv | Step 00 abstract grounding audit. | AUTHORITATIVE |
| 5 | Dataset counts | At abstract-alignment filtering, 1451 evidence items have status 'tail_absent'. | 1451 | outputs/00_civic_feasibility/abstract_alignment_summary.csv | Step 00 abstract grounding audit. | AUTHORITATIVE |
| 6 | Dataset counts | At abstract-alignment filtering, 646 evidence items have status 'head_absent'. | 646 | outputs/00_civic_feasibility/abstract_alignment_summary.csv | Step 00 abstract grounding audit. | AUTHORITATIVE |
| 7 | Dataset counts | At abstract-alignment filtering, 503 evidence items have status 'both_absent'. | 503 | outputs/00_civic_feasibility/abstract_alignment_summary.csv | Step 00 abstract grounding audit. | AUTHORITATIVE |
| 8 | Dataset counts | The frozen CIViC evaluation set contains 1812 ranking targets across 915 unique PMIDs. | 1812 targets, 915 PMIDs | outputs/02_evaluation_protocol/frozen_protocol.json | Final freeze after variant-pair exclusion. | AUTHORITATIVE |
| 9 | Dataset counts | The frozen evaluation set comprises 1230 gene-drug and 582 gene-disease targets. | 1230 / 582 | outputs/02_evaluation_protocol/frozen_protocol.json | Primary pair types only. | AUTHORITATIVE |
| 10 | Dataset counts | 262 variant-containing pairs were excluded from evaluation. | 262 | outputs/02_evaluation_protocol/frozen_protocol.json | Variant pairs frozen out at protocol step. | AUTHORITATIVE |
| 11 | Dataset counts | In PubTator recall bucketing, bucket 'matched (positive in frozen pool)' contains 1590 targets (87.7%). | 1590 (87.7%) | outputs/03_candidate_pool/03_candidate_pool_pubtator_recall_buckets.csv | Matched vs miss reasons on 1812 frozen targets. | AUTHORITATIVE |
| 12 | Dataset counts | In PubTator recall bucketing, bucket 'miss: entity type absent in abstract' contains 183 targets (10.1%). | 183 (10.1%) | outputs/03_candidate_pool/03_candidate_pool_pubtator_recall_buckets.csv | Matched vs miss reasons on 1812 frozen targets. | AUTHORITATIVE |
| 13 | Dataset counts | In PubTator recall bucketing, bucket 'miss: entity present but string/span mismatch' contains 39 targets (2.2%). | 39 (2.2%) | outputs/03_candidate_pool/03_candidate_pool_pubtator_recall_buckets.csv | Matched vs miss reasons on 1812 frozen targets. | AUTHORITATIVE |
| 14 | Dataset counts | The primary candidate pool has 5165 candidates and 781 CIViC positives for scope 'primary' pair type 'gene-drug'. | 5165 candidates, 781 positives | outputs/03_candidate_pool/03_candidate_pool_composition.csv | Frozen PubTator3-derived pool composition. | AUTHORITATIVE |
| 15 | Dataset counts | The primary candidate pool has 13746 candidates and 385 CIViC positives for scope 'primary' pair type 'gene-disease'. | 13746 candidates, 385 positives | outputs/03_candidate_pool/03_candidate_pool_composition.csv | Frozen PubTator3-derived pool composition. | AUTHORITATIVE |
| 16 | Dataset counts | The primary candidate pool has 2220 candidates and 0 CIViC positives for scope 'descriptive_only' pair type 'variant-disease'. | 2220 candidates, 0 positives | outputs/03_candidate_pool/03_candidate_pool_composition.csv | Frozen PubTator3-derived pool composition. | AUTHORITATIVE |
| 17 | Dataset counts | The primary candidate pool has 1102 candidates and 0 CIViC positives for scope 'descriptive_only' pair type 'variant-drug'. | 1102 candidates, 0 positives | outputs/03_candidate_pool/03_candidate_pool_composition.csv | Frozen PubTator3-derived pool composition. | AUTHORITATIVE |
| 18 | Dataset counts | The primary candidate pool contains 18,911 ranking candidates in total. | 18,911 | outputs/03_candidate_pool/03_candidate_pool_composition.csv | Sum of primary-scope gene-drug (5165) and gene-disease (13746) candidates. | DERIVED |
| 19 | Dataset counts | 3 PMIDs were removed from DrugProt training due to train-eval leakage. | 3 PMIDs, 31 relations removed | outputs/01_corpus_relevance/excluded_pmids.json | Overlap with CIViC eval inventory; 31 DrugProt relations dropped. | AUTHORITATIVE |
| 20 | Dataset counts | Among BioRED gene-disease train+val relations, strict intersection of three oncology criteria yields 1086 relations (3.41%). | 1086 (3.41%) | outputs/01_corpus_relevance/oncology_criteria_agreement.csv | Three-criterion oncology subset on BioRED gene-disease. | AUTHORITATIVE |
| 21 | Training recipe | The confirmed full-matrix recipe uses learning rate 5e-6 with no warmup across 9 encoders and 8 seeds (42–49). | lr=5e-6, warmup=none, 9×8=72 runs | outputs/20_round2_diagnostic/20_checkpoint_inventory.csv | All 72 runs show recipe_lr=5e-06 and recipe_warmup_label=none. | AUTHORITATIVE |
| 22 | Training recipe | The matrix produced 720 recoverable per-epoch checkpoints across 72 encoder×seed runs. | 720 checkpoints / 72 runs | outputs/20_round2_diagnostic/20_checkpoint_inventory.csv | Sum of n_recoverable_checkpoints; policy all_epochs_saved. | AUTHORITATIVE |
| 23 | Auxiliary (LR sweep / OncoKB / marker gate) | At seed 42, recipe 5e-6/none yielded encoder benchmark F1 spread 0.0259 with DeBERTa F1 0.734. | spread=0.0259, DeBERTa=0.734 | outputs/10_recipe_sweep_and_training/sweep/recipe_decision_table.csv | Step-10 advisory sweep; seed 42 only. | AUTHORITATIVE |
| 24 | Auxiliary (LR sweep / OncoKB / marker gate) | The marker quality gate overall verdict is PASS. | PASS | outputs/05_marker_quality_gate/quality_gate_results.json | Step 05 pre-matrix gate. | AUTHORITATIVE |
| 25 | Auxiliary (LR sweep / OncoKB / marker gate) | Native offset insertion succeeded on 100.0% of 64,452 training positives. | 100.0% | outputs/05_marker_quality_gate/quality_gate_results.json | 100.0% of positives use native offset insertion (64452 relations) | AUTHORITATIVE |
| 26 | Auxiliary (LR sweep / OncoKB / marker gate) | The BioRED test benchmark contains 8875 positive examples in 14,395 total examples. | 8875 positives | outputs/05_marker_quality_gate/quality_gate_results.json | 8875 BioRED test positives in 14395 benchmark examples | AUTHORITATIVE |
| 27 | Round 1 — variance/correlation | For metric 'benchmark_f1', between-encoder variance share is 53.4% (ICC=0.570). | 53.4% | outputs/11_round1_analysis/11_variance_components.csv | Between-encoder SD=0.0125, within-encoder SD=0.0108, n=72 runs. | AUTHORITATIVE |
| 28 | Round 1 — variance/correlation | For metric 'benchmark_f1_gene_disease', between-encoder variance share is 58.9% (ICC=0.618). | 58.9% | outputs/11_round1_analysis/11_variance_components.csv | Between-encoder SD=0.0119, within-encoder SD=0.0094, n=72 runs. | AUTHORITATIVE |
| 29 | Round 1 — variance/correlation | For metric 'benchmark_f1_gene_drug_combined', between-encoder variance share is 40.2% (ICC=0.429). | 40.2% | outputs/11_round1_analysis/11_variance_components.csv | Between-encoder SD=0.0226, within-encoder SD=0.0261, n=72 runs. | AUTHORITATIVE |
| 30 | Round 1 — variance/correlation | For metric 'benchmark_f1_gene_drug_biored', between-encoder variance share is 40.2% (ICC=0.429). | 40.2% | outputs/11_round1_analysis/11_variance_components.csv | Between-encoder SD=0.0226, within-encoder SD=0.0261, n=72 runs. | AUTHORITATIVE |
| 31 | Round 1 — variance/correlation | For metric 'benchmark_f1_gene_drug_drugprot', between-encoder variance share is nan% (ICC=0.000). | nan% | outputs/11_round1_analysis/11_variance_components.csv | Between-encoder SD=nan, within-encoder SD=nan, n=72 runs. | AUTHORITATIVE |
| 32 | Round 1 — variance/correlation | For metric 'kb_mrr_gene_drug', between-encoder variance share is 36.1% (ICC=0.374). | 36.1% | outputs/11_round1_analysis/11_variance_components.csv | Between-encoder SD=0.0126, within-encoder SD=0.0162, n=72 runs. | AUTHORITATIVE |
| 33 | Round 1 — variance/correlation | For metric 'kb_mrr_gene_disease', between-encoder variance share is 49.3% (ICC=0.530). | 49.3% | outputs/11_round1_analysis/11_variance_components.csv | Between-encoder SD=0.0470, within-encoder SD=0.0443, n=72 runs. | AUTHORITATIVE |
| 34 | Round 1 — variance/correlation | For metric 'ece', between-encoder variance share is 77.6% (ICC=0.788). | 77.6% | outputs/11_round1_analysis/11_variance_components.csv | Between-encoder SD=0.1149, within-encoder SD=0.0595, n=72 runs. | AUTHORITATIVE |
| 35 | Round 1 — variance/correlation | Bootstrap 95% CI on between-encoder share for 'benchmark_f1_gene_disease' is [36.0%, 64.4%]. | [0.360, 0.644] | outputs/11_round1_analysis/11_variance_components_bootstrap.csv | Cluster bootstrap over encoder resampling, N=5000. | AUTHORITATIVE |
| 36 | Round 1 — variance/correlation | Bootstrap 95% CI on between-encoder share for 'benchmark_f1_gene_drug_combined' is [18.0%, 59.9%]. | [0.180, 0.599] | outputs/11_round1_analysis/11_variance_components_bootstrap.csv | Cluster bootstrap over encoder resampling, N=5000. | AUTHORITATIVE |
| 37 | Round 1 — variance/correlation | Bootstrap 95% CI on between-encoder share for 'kb_mrr_gene_drug' is [15.3%, 50.3%]. | [0.153, 0.503] | outputs/11_round1_analysis/11_variance_components_bootstrap.csv | Cluster bootstrap over encoder resampling, N=5000. | AUTHORITATIVE |
| 38 | Round 1 — variance/correlation | Bootstrap 95% CI on between-encoder share for 'kb_mrr_gene_disease' is [21.4%, 71.2%]. | [0.214, 0.712] | outputs/11_round1_analysis/11_variance_components_bootstrap.csv | Cluster bootstrap over encoder resampling, N=5000. | AUTHORITATIVE |
| 39 | Round 1 — variance/correlation | The seed-level cluster-bootstrap Spearman between pair-type-matched benchmark F1 and gene-drug KB MRR is +0.018 with 95% CI [-0.232, +0.289] across 72 runs. | +0.018 [-0.232, +0.289] | outputs/11_round1_analysis/11_benchmark_kb_seed_association.csv | Uses benchmark_f1_gene_drug_combined as x (stored CSV); cluster bootstrap over 9 encoders, 2000 iterations. | AUTHORITATIVE |
| 40 | Round 1 — variance/correlation | The seed-level cluster-bootstrap Spearman between pair-type-matched benchmark F1 and gene-disease KB MRR is -0.440 with 95% CI [-0.660, -0.089] across 72 runs. | -0.440 [-0.660, -0.089] | outputs/11_round1_analysis/11_benchmark_kb_seed_association.csv | Uses benchmark_f1_gene_disease as x; cluster bootstrap over 9 encoders. | AUTHORITATIVE |
| 41 | Round 1 — variance/correlation | If BioRED-only benchmark_f1 (gene-disease F1) is used as x for gene-drug KB MRR across 72 runs, Spearman ρ is -0.259. | -0.259 | outputs/11_round1_analysis/11_per_run_scores.csv | Alternative definition using legacy benchmark_f1 column for both pair types; matches checked-in analysis.py default. | SUPERSEDED_VARIANT_EXISTS |
| 42 | Round 1 — variance/correlation | If BioRED-only benchmark_f1 is used as x for gene-disease KB MRR across 72 runs, Spearman ρ is -0.278. | -0.278 | outputs/11_round1_analysis/11_per_run_scores.csv | Legacy BioRED-only x column; differs from pair-type-matched stored association CSV. | SUPERSEDED_VARIANT_EXISTS |
| 43 | Round 1 — variance/correlation | At nine encoder means, spearman correlation between benchmark and gene-drug KB MRR is -0.150 (95% CI [-0.821, +0.670]). | -0.150 [-0.821, +0.670] | outputs/11_round1_analysis/11_benchmark_kb_correlations.csv | Weaker n=9 encoder-mean method. | AUTHORITATIVE |
| 44 | Round 1 — variance/correlation | At nine encoder means, pearson correlation between benchmark and gene-drug KB MRR is -0.068 (95% CI [-0.675, +0.650]). | -0.068 [-0.675, +0.650] | outputs/11_round1_analysis/11_benchmark_kb_correlations.csv | Weaker n=9 encoder-mean method. | AUTHORITATIVE |
| 45 | Round 1 — variance/correlation | At nine encoder means, spearman correlation between benchmark and gene-disease KB MRR is -0.550 (95% CI [-0.982, +0.222]). | -0.550 [-0.982, +0.222] | outputs/11_round1_analysis/11_benchmark_kb_correlations.csv | Weaker n=9 encoder-mean method. | AUTHORITATIVE |
| 46 | Round 1 — variance/correlation | At nine encoder means, pearson correlation between benchmark and gene-disease KB MRR is -0.566 (95% CI [-0.908, +0.279]). | -0.566 [-0.908, +0.279] | outputs/11_round1_analysis/11_benchmark_kb_correlations.csv | Weaker n=9 encoder-mean method. | AUTHORITATIVE |
| 47 | Round 1 — variance/correlation | Encoder-mean benchmark F1 ranges from 0.711 to 0.746 (spread 0.035). | 0.711–0.746, spread 0.035 | outputs/11_round1_analysis/11_benchmark_f1_range.csv | BioRED gene-disease benchmark F1 across 9 encoder means. | AUTHORITATIVE |
| 48 | Round 1 — variance/correlation | BERT-base encoder-mean benchmark F1 is 0.725 with gene-drug KB MRR 0.675 and gene-disease KB MRR 0.645. | bench=0.725, gd=0.675, gdis=0.645 | outputs/11_round1_analysis/11_encoder_summary.csv | Seed-averaged per encoder over clean 72-run matrix. | AUTHORITATIVE |
| 49 | Round 1 — variance/correlation | BioBERT-base encoder-mean benchmark F1 is 0.733 with gene-drug KB MRR 0.679 and gene-disease KB MRR 0.621. | bench=0.733, gd=0.679, gdis=0.621 | outputs/11_round1_analysis/11_encoder_summary.csv | Seed-averaged per encoder over clean 72-run matrix. | AUTHORITATIVE |
| 50 | Round 1 — variance/correlation | BioLinkBERT-base encoder-mean benchmark F1 is 0.746 with gene-drug KB MRR 0.656 and gene-disease KB MRR 0.595. | bench=0.746, gd=0.656, gdis=0.595 | outputs/11_round1_analysis/11_encoder_summary.csv | Seed-averaged per encoder over clean 72-run matrix. | AUTHORITATIVE |
| 51 | Round 1 — variance/correlation | BlueBERT-base encoder-mean benchmark F1 is 0.711 with gene-drug KB MRR 0.686 and gene-disease KB MRR 0.691. | bench=0.711, gd=0.686, gdis=0.691 | outputs/11_round1_analysis/11_encoder_summary.csv | Seed-averaged per encoder over clean 72-run matrix. | AUTHORITATIVE |
| 52 | Round 1 — variance/correlation | DeBERTa-base encoder-mean benchmark F1 is 0.743 with gene-drug KB MRR 0.686 and gene-disease KB MRR 0.643. | bench=0.743, gd=0.686, gdis=0.643 | outputs/11_round1_analysis/11_encoder_summary.csv | Seed-averaged per encoder over clean 72-run matrix. | AUTHORITATIVE |
| 53 | Round 1 — variance/correlation | DistilBERT-base encoder-mean benchmark F1 is 0.722 with gene-drug KB MRR 0.665 and gene-disease KB MRR 0.572. | bench=0.722, gd=0.665, gdis=0.572 | outputs/11_round1_analysis/11_encoder_summary.csv | Seed-averaged per encoder over clean 72-run matrix. | AUTHORITATIVE |
| 54 | Round 1 — variance/correlation | PubMedBERT-base encoder-mean benchmark F1 is 0.745 with gene-drug KB MRR 0.651 and gene-disease KB MRR 0.563. | bench=0.745, gd=0.651, gdis=0.563 | outputs/11_round1_analysis/11_encoder_summary.csv | Seed-averaged per encoder over clean 72-run matrix. | AUTHORITATIVE |
| 55 | Round 1 — variance/correlation | RoBERTa-base encoder-mean benchmark F1 is 0.733 with gene-drug KB MRR 0.661 and gene-disease KB MRR 0.549. | bench=0.733, gd=0.661, gdis=0.549 | outputs/11_round1_analysis/11_encoder_summary.csv | Seed-averaged per encoder over clean 72-run matrix. | AUTHORITATIVE |
| 56 | Round 1 — variance/correlation | SciBERT encoder-mean benchmark F1 is 0.745 with gene-drug KB MRR 0.674 and gene-disease KB MRR 0.645. | bench=0.745, gd=0.674, gdis=0.645 | outputs/11_round1_analysis/11_encoder_summary.csv | Seed-averaged per encoder over clean 72-run matrix. | AUTHORITATIVE |
| 57 | Round 1 — variance/correlation | Across 72 runs, mean per-abstract Spearman between pool size and gene-drug MRR is -0.400 (median -0.401). | mean -0.400, median -0.401 | outputs/11_round1_analysis/11_pool_size_robustness.csv | Mean of per-run Spearman over 72 run×pair-type rows. | DERIVED |
| 58 | Round 1 — variance/correlation | Across 72 runs, mean per-abstract Spearman between pool size and gene-disease MRR is -0.473 (median -0.467). | mean -0.473, median -0.467 | outputs/11_round1_analysis/11_pool_size_robustness.csv | Mean of per-run Spearman over 72 run×pair-type rows. | DERIVED |
| 59 | Round 1 — variance/correlation | Across 72 runs, median Spearman between model scores and entity proximity is 0.246. | median 0.246 | outputs/11_round1_analysis/11_distance_score_correlation.csv | Median of per-run spearman_r column. | DERIVED |
| 60 | Round 1 — variance/correlation | At nine encoder means, spearman between benchmark_f1_gene_drug_combined and ECE on gene-drug is -0.667. | -0.667 | outputs/11_round1_analysis/11_benchmark_ece_correlations.csv | Pair-type-specific benchmark axis vs calibration error. | AUTHORITATIVE |
| 61 | Round 1 — variance/correlation | At nine encoder means, pearson between benchmark_f1_gene_drug_combined and ECE on gene-drug is -0.688. | -0.688 | outputs/11_round1_analysis/11_benchmark_ece_correlations.csv | Pair-type-specific benchmark axis vs calibration error. | AUTHORITATIVE |
| 62 | Round 1 — variance/correlation | At nine encoder means, spearman between benchmark_f1_gene_disease and ECE on gene-disease is -0.667. | -0.667 | outputs/11_round1_analysis/11_benchmark_ece_correlations.csv | Pair-type-specific benchmark axis vs calibration error. | AUTHORITATIVE |
| 63 | Round 1 — variance/correlation | At nine encoder means, pearson between benchmark_f1_gene_disease and ECE on gene-disease is -0.818. | -0.818 | outputs/11_round1_analysis/11_benchmark_ece_correlations.csv | Pair-type-specific benchmark axis vs calibration error. | AUTHORITATIVE |
| 64 | Round 1 — fine-tuning lift | Mean fine-tuning lift on benchmark F1 across 9 encoders is +0.179. | +0.179 | outputs/11_round1_analysis/11_untrained_floor_lift.csv | Per-encoder (FT encoder-mean minus UT single score) then mean over 9 encoders. | AUTHORITATIVE |
| 65 | Round 1 — fine-tuning lift | Mean fine-tuning lift on gene-drug KB MRR is +0.121 and on gene-disease KB MRR is +0.168. | gd=+0.121, gdis=+0.168 | outputs/11_round1_analysis/11_untrained_floor_lift.csv | Same aggregation as benchmark lift. | AUTHORITATIVE |
| 66 | Round 1 — fine-tuning lift | An earlier Round 1 analyze pass (2026-06-09 slurm log) reported mean benchmark lift 0.378 before BlueBERT rescoring. | 0.378 | project_1/11_round1_analysis/slurm-5116372.out | Same formula but pre–Jul-02 untrained-floor scoring and biomedbert labelling. | SUPERSEDED_VARIANT_EXISTS |
| 67 | Round 1 — fine-tuning lift | BERT-base lift from untrained floor is benchmark F1 +0.195, gene-drug KB +0.123, gene-disease KB +0.228. | Δbench=+0.195, Δgd=+0.123, Δgdis=+0.228 | outputs/11_round1_analysis/11_untrained_floor_lift.csv | UT = pretrained encoder + random head (seed 4242); FT = validation-best checkpoint encoder-mean. | AUTHORITATIVE |
| 68 | Round 1 — fine-tuning lift | BioBERT-base lift from untrained floor is benchmark F1 -0.110, gene-drug KB +0.111, gene-disease KB +0.165. | Δbench=-0.110, Δgd=+0.111, Δgdis=+0.165 | outputs/11_round1_analysis/11_untrained_floor_lift.csv | UT = pretrained encoder + random head (seed 4242); FT = validation-best checkpoint encoder-mean. | AUTHORITATIVE |
| 69 | Round 1 — fine-tuning lift | BioLinkBERT-base lift from untrained floor is benchmark F1 +0.522, gene-drug KB +0.167, gene-disease KB +0.215. | Δbench=+0.522, Δgd=+0.167, Δgdis=+0.215 | outputs/11_round1_analysis/11_untrained_floor_lift.csv | UT = pretrained encoder + random head (seed 4242); FT = validation-best checkpoint encoder-mean. | AUTHORITATIVE |
| 70 | Round 1 — fine-tuning lift | BlueBERT-base lift from untrained floor is benchmark F1 -0.144, gene-drug KB +0.145, gene-disease KB +0.236. | Δbench=-0.144, Δgd=+0.145, Δgdis=+0.236 | outputs/11_round1_analysis/11_untrained_floor_lift.csv | UT = pretrained encoder + random head (seed 4242); FT = validation-best checkpoint encoder-mean. | AUTHORITATIVE |
| 71 | Round 1 — fine-tuning lift | DeBERTa-base lift from untrained floor is benchmark F1 +0.635, gene-drug KB +0.080, gene-disease KB +0.107. | Δbench=+0.635, Δgd=+0.080, Δgdis=+0.107 | outputs/11_round1_analysis/11_untrained_floor_lift.csv | UT = pretrained encoder + random head (seed 4242); FT = validation-best checkpoint encoder-mean. | AUTHORITATIVE |
| 72 | Round 1 — fine-tuning lift | DistilBERT-base lift from untrained floor is benchmark F1 +0.606, gene-drug KB +0.076, gene-disease KB +0.086. | Δbench=+0.606, Δgd=+0.076, Δgdis=+0.086 | outputs/11_round1_analysis/11_untrained_floor_lift.csv | UT = pretrained encoder + random head (seed 4242); FT = validation-best checkpoint encoder-mean. | AUTHORITATIVE |
| 73 | Round 1 — fine-tuning lift | PubMedBERT-base lift from untrained floor is benchmark F1 +0.138, gene-drug KB +0.120, gene-disease KB +0.138. | Δbench=+0.138, Δgd=+0.120, Δgdis=+0.138 | outputs/11_round1_analysis/11_untrained_floor_lift.csv | UT = pretrained encoder + random head (seed 4242); FT = validation-best checkpoint encoder-mean. | AUTHORITATIVE |
| 74 | Round 1 — fine-tuning lift | RoBERTa-base lift from untrained floor is benchmark F1 -0.161, gene-drug KB +0.132, gene-disease KB +0.131. | Δbench=-0.161, Δgd=+0.132, Δgdis=+0.131 | outputs/11_round1_analysis/11_untrained_floor_lift.csv | UT = pretrained encoder + random head (seed 4242); FT = validation-best checkpoint encoder-mean. | AUTHORITATIVE |
| 75 | Round 1 — fine-tuning lift | SciBERT lift from untrained floor is benchmark F1 -0.070, gene-drug KB +0.138, gene-disease KB +0.204. | Δbench=-0.070, Δgd=+0.138, Δgdis=+0.204 | outputs/11_round1_analysis/11_untrained_floor_lift.csv | UT = pretrained encoder + random head (seed 4242); FT = validation-best checkpoint encoder-mean. | AUTHORITATIVE |
| 76 | Round 1 — absolute levels | Reference 'random_uniform' achieves KB MRR levels: overall=0.322. | overall=0.322 | outputs/11_round1_analysis/11_absolute_kb_levels.csv | Analytic expectation on frozen pool (step 03) | AUTHORITATIVE |
| 77 | Round 1 — absolute levels | Reference 'distance_ranker' achieves KB MRR levels: overall=0.489, hard=0.369, easy=0.692. | overall=0.489, hard=0.369, easy=0.692 | outputs/11_round1_analysis/11_absolute_kb_levels.csv | Proximity-only ranker on frozen pool | AUTHORITATIVE |
| 78 | Round 1 — absolute levels | Reference 'finetuned_encoders_mean' achieves KB MRR levels: overall=0.642, gd=0.670, gdis=0.614, hard=0.392, easy=0.728. | overall=0.642, gd=0.670, gdis=0.614, hard=0.392, easy=0.728 | outputs/11_round1_analysis/11_absolute_kb_levels.csv | Mean of nine encoder seed-averaged means (clean runs) | AUTHORITATIVE |
| 79 | Round 1 — absolute levels | Reference 'finetuned_encoders_best' achieves KB MRR levels: gd=0.686, gdis=0.691, hard=0.416, easy=0.755. | gd=0.686, gdis=0.691, hard=0.416, easy=0.755 | outputs/11_round1_analysis/11_absolute_kb_levels.csv | Best encoder mean per pair type / subset | AUTHORITATIVE |
| 80 | Round 1 — absolute levels | On hard cross-sentence subset, 8 of 9 encoders beat the distance ranker baseline MRR 0.369. | 8/9 beat 0.369 | outputs/11_round1_analysis/11_easy_hard_ranking.csv | Seed-averaged encoder means vs distance ranker. | DERIVED |
| 81 | Round 1 — absolute levels | On easy co-sentence subset, 9 of 9 encoders beat the distance ranker baseline MRR 0.692. | 9/9 beat 0.692 | outputs/11_round1_analysis/11_easy_hard_ranking.csv | Seed-averaged encoder means vs distance ranker. | DERIVED |
| 82 | Round 2 — training dynamics | Under val_f1_best pairing, 69 of 72 seeds are pairable (epoch 1 and best-val endpoints valid). | 69/72 pairable | outputs/20_round2_diagnostic/20_pair_type_breakdown.csv | Requires different epochs at both endpoints with valid metrics. | AUTHORITATIVE |
| 83 | Round 2 — training dynamics | Mean paired KB MRR change (epoch 1→best val) for gene-drug is +0.0042 (41/69 seeds fall). | +0.0042, 41/69 fall | outputs/20_round2_diagnostic/20_pair_type_breakdown.csv | Well-trained definition val_f1_best. | AUTHORITATIVE |
| 84 | Round 2 — training dynamics | Mean paired KB MRR change (epoch 1→best val) for gene-disease is -0.0541 (46/69 seeds fall). | -0.0541, 46/69 fall | outputs/20_round2_diagnostic/20_pair_type_breakdown.csv | Well-trained definition val_f1_best. | AUTHORITATIVE |
| 85 | Round 2 — training dynamics | Pooled across 69 pairable seeds, mean Δ benchmark F1 is +0.0219 and mean Δ KB hard MRR is -0.0083. | Δbench=+0.0219, Δhard=-0.0083 | outputs/20_round2_diagnostic/20_seed_erosion_distribution.csv | ALL row aggregates pairable seeds. | AUTHORITATIVE |
| 86 | Round 2 — training dynamics | The erosion pattern (benchmark rises and KB-hard falls) occurs in 36/69 pairable seeds (52.2%). | 36/69 (52.2%) | outputs/20_round2_diagnostic/20_seed_erosion_distribution.csv | Within-seed paired change epoch1→best val. | AUTHORITATIVE |
| 87 | Round 2 — training dynamics | Mean paired Δ KB MRR on hard_cross_sentence is -0.0083 (95% CI [-0.0148, -0.0013]). | -0.0083 [-0.0148, -0.0013] | outputs/20_round2_diagnostic/20_hard_easy_breakdown.csv | val_f1_best endpoint. | AUTHORITATIVE |
| 88 | Round 2 — training dynamics | Mean paired Δ KB MRR on easy_co_sentence is -0.0027 (95% CI [-0.0099, +0.0045]). | -0.0027 [-0.0099, +0.0045] | outputs/20_round2_diagnostic/20_hard_easy_breakdown.csv | val_f1_best endpoint. | AUTHORITATIVE |
| 89 | Round 2 — training dynamics | For gene-disease (all), mean paired Δ KB MRR is -0.0541 with 46/69 seeds falling. | -0.0541, 46/69 | outputs/20_round2_diagnostic/20_gene_disease_subset_breakdown.csv | Metric column kb_mrr_gene_disease. | AUTHORITATIVE |
| 90 | Round 2 — training dynamics | For gene-disease hard, mean paired Δ KB MRR is -0.0126 with 39/69 seeds falling. | -0.0126, 39/69 | outputs/20_round2_diagnostic/20_gene_disease_subset_breakdown.csv | Metric column kb_mrr_gene_disease_hard. | AUTHORITATIVE |
| 91 | Round 2 — training dynamics | For gene-disease easy, mean paired Δ KB MRR is -0.0126 with 52/69 seeds falling. | -0.0126, 52/69 | outputs/20_round2_diagnostic/20_gene_disease_subset_breakdown.csv | Metric column kb_mrr_gene_disease_easy. | AUTHORITATIVE |
| 92 | Round 2 — training dynamics | For gene-drug (all), mean paired Δ KB MRR is +0.0042 with 41/69 seeds falling. | +0.0042, 41/69 | outputs/20_round2_diagnostic/20_gene_disease_subset_breakdown.csv | Metric column kb_mrr_gene_drug. | AUTHORITATIVE |
| 93 | Round 2 — training dynamics | For gene-drug hard, mean paired Δ KB MRR is -0.0058 with 43/69 seeds falling. | -0.0058, 43/69 | outputs/20_round2_diagnostic/20_gene_disease_subset_breakdown.csv | Metric column kb_mrr_gene_drug_hard. | AUTHORITATIVE |
| 94 | Round 2 — training dynamics | For gene-drug easy, mean paired Δ KB MRR is +0.0030 with 30/69 seeds falling. | +0.0030, 30/69 | outputs/20_round2_diagnostic/20_gene_disease_subset_breakdown.csv | Metric column kb_mrr_gene_drug_easy. | AUTHORITATIVE |
| 95 | Round 2 — training dynamics | BERT-base mean paired Δ benchmark F1 is +0.0385, Δ gene-drug KB +0.0372, Δ gene-disease KB +0.0357, Δ gene-disease-hard +0.0161. | Δbench=+0.0385, Δgd=+0.0372, Δgdis=+0.0357, Δhard=+0.0161 | outputs/20_round2_diagnostic/20_within_seed_paired_changes.csv | Encoder mean over pairable seeds; val_f1_best. | DERIVED |
| 96 | Round 2 — training dynamics | BioBERT-base mean paired Δ benchmark F1 is +0.0277, Δ gene-drug KB -0.0148, Δ gene-disease KB -0.0945, Δ gene-disease-hard -0.0056. | Δbench=+0.0277, Δgd=-0.0148, Δgdis=-0.0945, Δhard=-0.0056 | outputs/20_round2_diagnostic/20_within_seed_paired_changes.csv | Encoder mean over pairable seeds; val_f1_best. | DERIVED |
| 97 | Round 2 — training dynamics | BioLinkBERT-base mean paired Δ benchmark F1 is +0.0159, Δ gene-drug KB -0.0160, Δ gene-disease KB -0.1261, Δ gene-disease-hard -0.0637. | Δbench=+0.0159, Δgd=-0.0160, Δgdis=-0.1261, Δhard=-0.0637 | outputs/20_round2_diagnostic/20_within_seed_paired_changes.csv | Encoder mean over pairable seeds; val_f1_best. | DERIVED |
| 98 | Round 2 — training dynamics | BlueBERT-base mean paired Δ benchmark F1 is +0.0136, Δ gene-drug KB +0.0329, Δ gene-disease KB +0.1082, Δ gene-disease-hard +0.0144. | Δbench=+0.0136, Δgd=+0.0329, Δgdis=+0.1082, Δhard=+0.0144 | outputs/20_round2_diagnostic/20_within_seed_paired_changes.csv | Encoder mean over pairable seeds; val_f1_best. | DERIVED |
| 99 | Round 2 — training dynamics | DeBERTa-base mean paired Δ benchmark F1 is +0.0222, Δ gene-drug KB -0.0094, Δ gene-disease KB -0.0664, Δ gene-disease-hard +0.0062. | Δbench=+0.0222, Δgd=-0.0094, Δgdis=-0.0664, Δhard=+0.0062 | outputs/20_round2_diagnostic/20_within_seed_paired_changes.csv | Encoder mean over pairable seeds; val_f1_best. | DERIVED |
| 100 | Round 2 — training dynamics | DistilBERT-base mean paired Δ benchmark F1 is +0.0312, Δ gene-drug KB +0.0860, Δ gene-disease KB +0.0758, Δ gene-disease-hard +0.0164. | Δbench=+0.0312, Δgd=+0.0860, Δgdis=+0.0758, Δhard=+0.0164 | outputs/20_round2_diagnostic/20_within_seed_paired_changes.csv | Encoder mean over pairable seeds; val_f1_best. | DERIVED |
| 101 | Round 2 — training dynamics | PubMedBERT-base mean paired Δ benchmark F1 is +0.0096, Δ gene-drug KB -0.0266, Δ gene-disease KB -0.1306, Δ gene-disease-hard -0.0225. | Δbench=+0.0096, Δgd=-0.0266, Δgdis=-0.1306, Δhard=-0.0225 | outputs/20_round2_diagnostic/20_within_seed_paired_changes.csv | Encoder mean over pairable seeds; val_f1_best. | DERIVED |
| 102 | Round 2 — training dynamics | RoBERTa-base mean paired Δ benchmark F1 is +0.0131, Δ gene-drug KB -0.0266, Δ gene-disease KB -0.1593, Δ gene-disease-hard -0.0016. | Δbench=+0.0131, Δgd=-0.0266, Δgdis=-0.1593, Δhard=-0.0016 | outputs/20_round2_diagnostic/20_within_seed_paired_changes.csv | Encoder mean over pairable seeds; val_f1_best. | DERIVED |
| 103 | Round 2 — training dynamics | SciBERT mean paired Δ benchmark F1 is +0.0253, Δ gene-drug KB -0.0273, Δ gene-disease KB -0.1229, Δ gene-disease-hard -0.0776. | Δbench=+0.0253, Δgd=-0.0273, Δgdis=-0.1229, Δhard=-0.0776 | outputs/20_round2_diagnostic/20_within_seed_paired_changes.csv | Encoder mean over pairable seeds; val_f1_best. | DERIVED |
| 104 | Round 2 — robustness | For gene-disease (all) under 'val_f1_best', mean Δ KB MRR is -0.0541 (46/69 seeds fall). | -0.0541, 46/69 | outputs/20_round2_diagnostic/20_gene_disease_robustness.csv | Slug gene_disease. | AUTHORITATIVE |
| 105 | Round 2 — robustness | For gene-disease (all) under 'last_epoch', mean Δ KB MRR is -0.0835 (55/72 seeds fall). | -0.0835, 55/72 | outputs/20_round2_diagnostic/20_gene_disease_robustness.csv | Slug gene_disease. | AUTHORITATIVE |
| 106 | Round 2 — robustness | For gene-disease (all) under 'fixed_epoch5', mean Δ KB MRR is -0.0529 (49/72 seeds fall). | -0.0529, 49/72 | outputs/20_round2_diagnostic/20_gene_disease_robustness.csv | Slug gene_disease. | AUTHORITATIVE |
| 107 | Round 2 — robustness | For gene-disease hard under 'val_f1_best', mean Δ KB MRR is -0.0126 (39/69 seeds fall). | -0.0126, 39/69 | outputs/20_round2_diagnostic/20_gene_disease_robustness.csv | Slug gene_disease_hard. | AUTHORITATIVE |
| 108 | Round 2 — robustness | For gene-disease hard under 'last_epoch', mean Δ KB MRR is -0.0349 (56/72 seeds fall). | -0.0349, 56/72 | outputs/20_round2_diagnostic/20_gene_disease_robustness.csv | Slug gene_disease_hard. | AUTHORITATIVE |
| 109 | Round 2 — robustness | For gene-disease hard under 'fixed_epoch5', mean Δ KB MRR is -0.0269 (46/72 seeds fall). | -0.0269, 46/72 | outputs/20_round2_diagnostic/20_gene_disease_robustness.csv | Slug gene_disease_hard. | AUTHORITATIVE |
| 110 | Round 2 — robustness | For gene-drug (all) under 'val_f1_best', mean Δ KB MRR is +0.0042 (41/69 seeds fall). | +0.0042, 41/69 | outputs/20_round2_diagnostic/20_gene_disease_robustness.csv | Slug gene_drug. | AUTHORITATIVE |
| 111 | Round 2 — robustness | For gene-drug (all) under 'last_epoch', mean Δ KB MRR is -0.0119 (53/72 seeds fall). | -0.0119, 53/72 | outputs/20_round2_diagnostic/20_gene_disease_robustness.csv | Slug gene_drug. | AUTHORITATIVE |
| 112 | Round 2 — robustness | For gene-drug (all) under 'fixed_epoch5', mean Δ KB MRR is +0.0008 (47/72 seeds fall). | +0.0008, 47/72 | outputs/20_round2_diagnostic/20_gene_disease_robustness.csv | Slug gene_drug. | AUTHORITATIVE |
| 113 | Round 2 — robustness | For gene-disease KB MRR, 58 seeds (80.6%) have KB peak before best val relative to validation-best epoch. | 58 seeds (80.6%) | outputs/20_round2_diagnostic/20_kb_peak_timing_summary.csv | Peak timing classification per seed. | AUTHORITATIVE |
| 114 | Round 2 — robustness | For gene-disease KB MRR, 13 seeds (18.1%) have KB peak coincident best val relative to validation-best epoch. | 13 seeds (18.1%) | outputs/20_round2_diagnostic/20_kb_peak_timing_summary.csv | Peak timing classification per seed. | AUTHORITATIVE |
| 115 | Round 2 — robustness | For gene-disease KB MRR, 1 seeds (1.4%) have KB peak after best val relative to validation-best epoch. | 1 seeds (1.4%) | outputs/20_round2_diagnostic/20_kb_peak_timing_summary.csv | Peak timing classification per seed. | AUTHORITATIVE |
| 116 | Round 2 — robustness | In pool stratum 'small_pool', mean gene-disease paired Δ MRR is -0.0436 (46/69 seeds fall). | -0.0436 | outputs/20_round2_diagnostic/20_gene_disease_pool_stratum_summary.csv | Pool-size matched robustness check. | AUTHORITATIVE |
| 117 | Round 2 — robustness | In pool stratum 'large_pool', mean gene-disease paired Δ MRR is -0.0729 (47/69 seeds fall). | -0.0729 | outputs/20_round2_diagnostic/20_gene_disease_pool_stratum_summary.csv | Pool-size matched robustness check. | AUTHORITATIVE |
| 118 | Round 2 — robustness | In pool stratum 'comparable_to_gene_drug', mean gene-disease paired Δ MRR is -0.0349 (46/69 seeds fall). | -0.0349 | outputs/20_round2_diagnostic/20_gene_disease_pool_stratum_summary.csv | Pool-size matched robustness check. | AUTHORITATIVE |
| 119 | Round 2 — bootstrap | Bootstrap over pairable seeds puts P(negative mean) for gene-disease-hard paired Δ KB MRR at 98.2% (2000 iterations). | 98.2% | reports/20_round2_diagnostic/report.md | From mundane_explanations.bootstrap_positive_sign_stability; mean Δ=−0.01265. | DERIVED |
| 120 | Round 2 — bootstrap | Stale README template lists gene-disease-hard bootstrap P(negative) as 99.1% from pre–BlueBERT-fix analyze. | 99.1% | project_1/20_round2_diagnostic/README.md | Superseded template via manuscript_regenerate/readmes.py. | SUPERSEDED_VARIANT_EXISTS |
| 121 | Qualitative errors | Qualitative analysis identified 852 missed curated positives, of which 818 are genuine model errors and 34 are abstract-unsupported (4.0%). | 852 missed, 818 genuine, 34 abstract-unsupported | outputs/20_round2_diagnostic/20_qualitative_error_summary.csv | Representative seed 42; median across nine encoders. | AUTHORITATIVE |
| 122 | Qualitative errors | Among 818 genuine model errors, failure pattern 'cross_sentence_hard' occurs at rate 47.8%. | 47.8% | outputs/20_round2_diagnostic/20_qualitative_error_patterns.csv | n_genuine=818. | AUTHORITATIVE |
| 123 | Qualitative errors | Among 818 genuine model errors, failure pattern 'multiword_entity' occurs at rate 14.7%. | 14.7% | outputs/20_round2_diagnostic/20_qualitative_error_patterns.csv | n_genuine=818. | AUTHORITATIVE |
| 124 | Qualitative errors | Among 818 genuine model errors, failure pattern 'publication_year_before_2010' occurs at rate 18.9%. | 18.9% | outputs/20_round2_diagnostic/20_qualitative_error_patterns.csv | n_genuine=818. | AUTHORITATIVE |
| 125 | Qualitative errors | Among 818 genuine model errors, failure pattern 'gene_disease_pair' occurs at rate 30.9%. | 30.9% | outputs/20_round2_diagnostic/20_qualitative_error_patterns.csv | n_genuine=818. | AUTHORITATIVE |
| 126 | Round 2 — training dynamics | Exploratory n=9 Spearman between encoder-mean benchmark F1 and gene-disease-hard erosion magnitude is ρ=+0.800 (p=0.010). | ρ=+0.800, p=0.010 | reports/20_round2_diagnostic/report.md | Not stored as standalone CSV; computed in encoder_correlation analysis over 9 encoder means. | DERIVED |
| 127 | Round 2 — training dynamics | Exploratory n=9 Spearman between biomedical-pretrain flag and gene-disease-hard erosion is ρ=+0.693 (p=0.039). | ρ=+0.693, p=0.039 | reports/20_round2_diagnostic/report.md | Binary biomedical vs general over 9 encoders. | DERIVED |
| 128 | Round 1 — variance/correlation | The 2026-06-09 Round 1 slurm log reported encoder-mean benchmark spread 0.025 (min 0.720, max 0.746). | spread 0.025 | project_1/11_round1_analysis/slurm-5116372.out | Pre–BlueBERT-fix scoring vintage. | SUPERSEDED_VARIANT_EXISTS |

### Known definition splits requiring author decision

Four conceptual quantities have more than one defensible number on disk. **Gene-drug benchmark–KB association (rows 39–42):** the stored CSV uses pair-type-matched x (`benchmark_f1_gene_drug_combined`) and yields ρ≈+0.018 (CI spans zero); the legacy `benchmark_f1` column yields ρ≈−0.26. **Gene-disease association (rows 40 vs 42):** pair-type-matched x gives ρ≈−0.44; BioRED-only `benchmark_f1` gives ρ≈−0.28. **Fine-tuning lift (rows 64–66):** current per-encoder mean is +0.179; the Jun 9 slurm log reports +0.378 under the same formula but older untrained-floor inputs. **Round 2 gene-disease Δ endpoint (rows 88–90 vs robustness rows 107–112):** val_f1_best gives −0.0541; last_epoch gives −0.0835; fixed epoch 5 gives −0.0529 — all defensible depending on which training endpoint the paper emphasises.

<!-- Task A complete -->

---

## 13. Targeted verification of three flagged items (Task B)

### 13.1 Gene-drug seed-level Spearman = +0.018

**Recalculation method:** Loaded all 72 rows from `outputs/11_round1_analysis/11_per_run_scores.csv`. Computed Spearman ρ between benchmark and KB columns using `scipy.stats.spearmanr` on the full 72-run table (no bootstrap). Also tested the column pairs implied by the stored CSV.

| x column | y column | Recalculated ρ | CSV ρ |
|----------|----------|----------------|-------|
| `benchmark_f1` (BioRED gene-disease only) | `kb_mrr_gene_drug` | **−0.259** | — |
| `benchmark_f1_gene_drug_combined` | `kb_mrr_gene_drug` | **+0.018** | **+0.018** |
| `benchmark_f1_gene_disease` | `kb_mrr_gene_disease` | **−0.440** | **−0.440** |

**Comparison to CSV:** `11_benchmark_kb_seed_association.csv` gene-drug row: ρ = 0.017622, CI [−0.232, +0.289], n = 72, method = `seed_level_cluster_bootstrap`. Pair-type-matched recalculation matches to machine precision.

**Conclusion:** **Confirmed** for the stored CSV value (+0.018), but only when x = `benchmark_f1_gene_drug_combined`. The current `cluster_bootstrap_benchmark_kb` function in `11_round1_analysis/analysis.py` (L428–431) uses `benchmark_f1` (BioRED gene-disease F1) for **both** pair types; re-running that function on current data yields ρ = −0.259 (gene-drug) and ρ = −0.278 (gene-disease) — a **code–artifact mismatch**. The CSV and report reflect pair-type-matched benchmark columns; the checked-in association function does not. Paper authors should confirm which definition the manuscript intends before updating numbers.

### 13.2 Fine-tuning ΔF1 = +0.179 definition

**Q1 — What is "untrained baseline"?** Pretrained HuggingFace encoder weights with a **freshly random-initialised binary classification head** (no fine-tuning). Head init seed = `UNTRAINED_HEAD_SEED` in `11_round1_analysis/config.py`. Implementation: `score_untrained.py` L51–58 (`_load_untrained_model`).

**Q2 — What is "fine-tuned"?** Per-encoder **seed-averaged** scores from the **validation-best checkpoint** (`matrix/checkpoints/{model_id}/seed_{seed}/best/`), selected by pooled validation F1 micro. Lift table uses `encoder_summary(per_run_clean)` means, not a single global checkpoint.

**Q3 — Aggregation:** Lift is computed **per encoder** (FT encoder-mean minus UT encoder-single-score), then **averaged across 9 encoders**. Not pooled over 72 runs first. Code: `finetuning_lift_table` in `analysis.py` L312–342; mean printed in `run_analysis.py` L100–102.

**Q4 — Alternative ~0.378 figure:** **Yes.** `11_round1_analysis/slurm-5116372.out` (2026-06-09) reports `Mean lift: benchmark 0.378`. That run predates the BlueBERT naming/rescoring fix and used inflated lifts for DeBERTa (0.746) and DistilBERT (0.720) where untrained `benchmark_f1 = 0.000`. No separate CSV stores 0.378; it exists only in the Jun 9 slurm log. Current authoritative value 0.179 is in `11_untrained_floor_lift.csv` (mean of `lift_benchmark_f1` column).

**0.179 vs 0.378:** Both use the same formula (per-encoder FT-mean minus UT, then mean over 9 encoders). The difference is **input data vintage**: Jun 9 had mis-scored untrained floors (zeros for collapsed-head encoders) and `biomedbert_base` labelling; Jul 2 rescoring yields lower FT benchmarks for several encoders and higher UT scores (e.g. BlueBERT UT = 0.741 → lift = −0.144). 0.378 is **stale**; 0.179 is **current**.

### 13.3 Bootstrap P(negative) = 98.2% vs README 99.1%

**Source script:** `20_round2_diagnostic/mundane_explanations.py` → `bootstrap_positive_sign_stability` (L284–327). Called from `run_mundane_explanations` and written to `reports/20_round2_diagnostic/report.md` via `report.py`.

**Parameters traced:**
- **Bootstrap iterations:** `n_boot = 2000` (default argument L284).
- **Target metric:** **gene-disease-hard** paired Δ KB MRR, epoch 1 → validation-best epoch (`val_f1_best` well-trained definition).
- **Unit of resampling:** One scalar per pairable encoder×seed trajectory (positives at both endpoints); bootstrap resamples these scalars with replacement and computes fraction of bootstrap means < 0.

**CSV storage:** No dedicated CSV column stores 98.2%. The value appears in `reports/20_round2_diagnostic/report.md`, slurm logs (`slurm-5473442.out` line 95: `P(negative)=98.2%`), and stdout from `run_mundane_explanations`. README template (`20_round2_diagnostic/README.md` via `manuscript_regenerate/readmes.py`) still lists **99.1%** from an earlier analyze pass (`slurm-5473046.out`: 99.3% with mean Δ = −0.0157 before Jul 2 rescoring).

**Cross-validation (rerun, <2 min):** Reloaded `20_epoch_trajectory.csv`, called `bootstrap_positive_sign_stability(traj, n_boot=2000)`. Result: `frac_negative_bootstrap = 0.982`, `mean_delta = −0.01265`, CI [−0.0237, −0.0010]. **Matches 98.2%.**

**README 99.1% origin:** Stale template from pre–BlueBERT-fix analyze job when gene-disease-hard mean was −0.0157 and P(negative) ≈ 99.3%. Superseded by Jul 2 pipeline (69 pairable seeds, mean −0.0126, P = 98.2%).

<!-- Task B complete -->

---

## 14. BlueBERT exception and biomedical-encoder narrative (Task C)

### 14.1 Full encoder × paired-change table (epoch 1 → best val, val_f1_best)

Values are encoder means over pairable seeds from `20_within_seed_paired_changes.csv` (KB overall = mean of gene-drug and gene-disease Δ).

| Encoder | Δ bench F1 | Δ KB MRR (overall) | Δ gene-drug | Δ gene-disease | Δ gene-disease-hard | Δ gene-disease-easy |
|---------|------------|--------------------|-------------|----------------|---------------------|---------------------|
| BERT-base | +0.0385 | +0.0364 | +0.0372 | +0.0357 | +0.0161 | −0.0091 |
| BioBERT-base | +0.0277 | −0.0547 | −0.0148 | −0.0945 | −0.0056 | −0.0216 |
| BioLinkBERT-base | +0.0159 | −0.0710 | −0.0160 | −0.1261 | −0.0637 | −0.0248 |
| **BlueBERT-base** | **+0.0136** | **+0.0705** | **+0.0329** | **+0.1082** | **+0.0144** | **+0.0088** |
| DeBERTa-base | +0.0222 | −0.0379 | −0.0094 | −0.0664 | +0.0062 | −0.0060 |
| DistilBERT-base | +0.0312 | +0.0809 | +0.0860 | +0.0758 | +0.0164 | +0.0229 |
| PubMedBERT-base | +0.0096 | −0.0786 | −0.0266 | −0.1306 | −0.0225 | −0.0199 |
| RoBERTa-base | +0.0131 | −0.0929 | −0.0266 | −0.1593 | −0.0016 | −0.0366 |
| SciBERT | +0.0253 | −0.0751 | −0.0273 | −0.1229 | −0.0776 | −0.0273 |

Cross-check: `20_gene_disease_encoder_breakdown.csv` gene-disease-hard means match BlueBERT +0.0144, PubMedBERT −0.0225, BioLinkBERT −0.0637, SciBERT −0.0776, etc.

### 14.2 Hypotheses for BlueBERT exception

1. **Lowest benchmark F1 among encoders (0.711 encoder-mean; Section 5.4):** BlueBERT starts from the weakest in-distribution benchmark position but achieves the **highest** gene-disease KB MRR among biomedical encoders (0.691 finetuned mean in `11_encoder_summary.csv`). Its training trajectory adds KB ranking capacity without the sharp gene-disease erosion seen in PubMedBERT/BioLinkBERT/SciBERT. Plausible interpretation: MIMIC clinical text in pretraining (Section 3.1) may preserve disease-context signal that PubMed-abstract-only encoders lose under fine-tuning — but this is inferential, not directly measured.

2. **High untrained KB floor (UT gene-disease MRR = 0.455; `11_untrained_floor_lift.csv`):** BlueBERT's random-head baseline already ranks gene-disease pairs relatively well; fine-tuning adds +0.236 KB lift (largest among biomedical encoders) while benchmark F1 actually **falls** (−0.144 lift). The "exception" may partly reflect a different untrained starting point rather than a distinct erosion mechanism. **No clear explanation from available data** for why only BlueBERT rises on hard subset while four other biomedical encoders fall.

### 14.3 Narrative adjustment guidance (for paper authors)

The data do not support a single sentence that "biomedical encoders uniformly lose gene-disease KB ranking during fine-tuning." Four of five biomedical encoders show negative gene-disease and gene-disease-hard means, but BlueBERT is a clear counterexample on both all and hard subsets (+0.1082, +0.0144). General-purpose encoders are also heterogeneous: DistilBERT and BERT rise on gene-disease; RoBERTa falls sharply (−0.1593). Any narrative should (a) report encoder-level tables, not only group means; (b) separate "PubMed-abstract BERT family" (PubMedBERT, BioLinkBERT, SciBERT, BioBERT) from BlueBERT (PubMed+MIMIC); (c) acknowledge that hard-subset declines are smaller in magnitude than all-subset declines and that easy-subset declines co-occur, weakening "hard-concentrated" language; (d) avoid attributing the pattern to a single pretraining variable without direct ablation. The honest framing is **regular but architecture-dependent heterogeneity**, with BlueBERT as the primary biomedical exception requiring explicit mention.

<!-- Task C complete -->

---

## 15. Figure inventory and provenance (code-side reference for paper alignment)

> This section catalogues every figure that has been generated by the current pipeline, with enough provenance for the author to check paper `\includegraphics` references manually. This section does not know what the paper currently references.

### Table 15.A — On-disk figure inventory

| # | Category | Figure filename | Absolute path | Generator script and function | mtime (UTC) | Fresh vs latest regeneration batch? | Suggested caption angle |
|---|----------|-----------------|---------------|------------------------------|-------------|--------------------------------------|-------------------------|
| 1 | Study design | study_design.png | /home/b5ac/freddieyu.b5ac/projects/project_1/assets/figures/study_design.png | none (asset not generated) | — | missing | Conceptual three-stage study flow referenced in Section 1. |
| 2 | Dataset construction (steps 00–05) | entity_pair_distribution.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/00_civic_feasibility/entity_pair_distribution.png | manuscript_regenerate/figures.py::regenerate_step00 | 2026-07-02T16:41:36Z | yes | CIViC evaluable targets by entity-pair type at feasibility step. |
| 3 | Dataset construction (steps 00–05) | 01_corpus_granularity_ladder.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/01_corpus_relevance/01_corpus_granularity_ladder.png | manuscript_regenerate/figures.py::regenerate_step01 | 2026-07-02T16:41:36Z | yes | Training-corpus relation counts by granularity level. |
| 4 | Dataset construction (steps 00–05) | 01_corpus_pmid_leakage.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/01_corpus_relevance/01_corpus_pmid_leakage.png | manuscript_regenerate/figures.py::regenerate_step01 | 2026-07-02T16:41:36Z | yes | Train-eval PMID overlap between corpora and CIViC. |
| 5 | Dataset construction (steps 00–05) | 02_evaluation_protocol_composition.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/02_evaluation_protocol/02_evaluation_protocol_composition.png | manuscript_regenerate/figures.py::regenerate_step02 | 2026-07-02T16:41:36Z | yes | Frozen evaluation target composition by pair type. |
| 6 | Dataset construction (steps 00–05) | 03_candidate_pool_coverage.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/03_candidate_pool/03_candidate_pool_coverage.png | manuscript_regenerate/figures.py::regenerate_step03 | 2026-07-02T16:41:36Z | yes | Abstract coverage of frozen CIViC primary targets. |
| 7 | Dataset construction (steps 00–05) | 03_candidate_pool_pubtator_recall_gap.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/03_candidate_pool/03_candidate_pool_pubtator_recall_gap.png | manuscript_regenerate/figures.py::regenerate_step03 | 2026-07-02T16:41:36Z | yes | PubTator recall gap: matched vs unmatched CIViC targets. |
| 8 | Dataset construction (steps 00–05) | 04_pilot_study_benchmark_vs_kb.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/04_pilot_study/04_pilot_study_benchmark_vs_kb.png | manuscript_regenerate/figures.py::regenerate_step04 | 2026-07-02T16:41:36Z | yes | Exploratory pilot: benchmark vs KB scatter (pre-marker-gate vintage). |
| 9 | Dataset construction (steps 00–05) | 05_marker_quality_gate_before_after.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/05_marker_quality_gate/05_marker_quality_gate_before_after.png | manuscript_regenerate/figures.py::regenerate_step05 | 2026-07-02T16:41:36Z | yes | Marker-insertion quality gate before/after offset fix. |
| 10 | LR sweep / matrix training (step 10) | matrix_benchmark_f1_heatmap.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/10_recipe_sweep_and_training/matrix/matrix_benchmark_f1_heatmap.png | manuscript_regenerate/figures.py::regenerate_step10 | 2026-07-02T16:41:37Z | yes | Full 9×8 matrix BioRED benchmark F1 heatmap. |
| 11 | LR sweep / matrix training (step 10) | recipe_spread_vs_deberta_health.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/10_recipe_sweep_and_training/sweep/recipe_spread_vs_deberta_health.png | manuscript_regenerate/figures.py::regenerate_step10 | 2026-07-02T16:41:37Z | yes | Recipe sweep: encoder spread vs DeBERTa health for recipe selection. |
| 12 | Round 1 analysis (step 11) | fig1_benchmark_kb_scatter.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/11_round1_analysis/fig1_benchmark_kb_scatter.png | manuscript_regenerate/figures.py::regenerate_step11 | 2026-07-02T16:41:37Z | yes | Encoder-mean benchmark F1 vs KB MRR by pair type with seed uncertainty. |
| 13 | Round 1 analysis (step 11) | fig2_variance_between_encoder.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/11_round1_analysis/fig2_variance_between_encoder.png | manuscript_regenerate/figures.py::regenerate_step11 | 2026-07-02T16:41:37Z | yes | Between-encoder variance share discriminates benchmark more than KB axes. |
| 14 | Round 1 analysis (step 11) | fig3_easy_hard_ranking_validity.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/11_round1_analysis/fig3_easy_hard_ranking_validity.png | manuscript_regenerate/figures.py::regenerate_step11 | 2026-07-02T16:41:37Z | yes | Encoder MRR vs distance ranker on easy and hard subsets. |
| 15 | Round 1 analysis (step 11) | fig4_finetuning_lift.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/11_round1_analysis/fig4_finetuning_lift.png | manuscript_regenerate/figures.py::regenerate_step11 | 2026-07-02T16:41:38Z | yes | Per-encoder fine-tuning lift from untrained floor on benchmark and KB. |
| 16 | Round 2 dynamics (step 20) | fig1_per_seed_trajectories.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/20_round2_diagnostic/fig1_per_seed_trajectories.png | 20_round2_diagnostic/figures.py::figure1_per_seed_trajectories | 2026-07-02T17:09:46Z | yes | Per-encoder benchmark F1 and KB-hard MRR trajectories across epochs. |
| 17 | Round 2 dynamics (step 20) | fig2_within_seed_paired_change.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/20_round2_diagnostic/fig2_within_seed_paired_change.png | 20_round2_diagnostic/figures.py::figure2_paired_change_distribution | 2026-07-02T17:09:47Z | yes | Distribution of within-seed paired Δbenchmark vs ΔKB-hard. |
| 18 | Round 2 dynamics (step 20) | fig3_hard_easy_pair_type.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/20_round2_diagnostic/fig3_hard_easy_pair_type.png | 20_round2_diagnostic/figures.py::figure3_hard_easy_pair_type | 2026-07-02T17:09:47Z | yes | KB MRR paired change by hard/easy and gene-drug/gene-disease. |
| 19 | Round 2 dynamics (step 20) | fig4_robustness_well_trained.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/20_round2_diagnostic/fig4_robustness_well_trained.png | 20_round2_diagnostic/figures.py::figure4_robustness_well_trained | 2026-07-02T17:09:47Z | yes | Fraction of seeds showing KB decline under three checkpoint definitions. |
| 20 | Round 2 dynamics (step 20) | fig5_gene_disease_hard_trajectories.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/20_round2_diagnostic/fig5_gene_disease_hard_trajectories.png | 20_round2_diagnostic/figures.py::figure5_gene_disease_hard_trajectories | 2026-07-02T17:09:48Z | yes | Gene-disease-hard MRR vs benchmark F1 across training (key dynamics figure). |
| 21 | Round 2 dynamics (step 20) | fig6_pair_type_subset_contrast.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/20_round2_diagnostic/fig6_pair_type_subset_contrast.png | 20_round2_diagnostic/figures.py::figure6_pair_type_subset_contrast | 2026-07-02T17:09:48Z | yes | Pair-type × subset contrast of paired KB changes. |
| 22 | Round 2 dynamics (step 20) | fig7_kb_peak_timing.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/20_round2_diagnostic/fig7_kb_peak_timing.png | 20_round2_diagnostic/figures.py::figure7_kb_peak_timing | 2026-07-02T17:09:49Z | yes | When gene-disease KB peak occurs relative to validation-best epoch. |
| 23 | Round 2 dynamics (step 20) | fig8_pool_stratum_gene_disease.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/20_round2_diagnostic/fig8_pool_stratum_gene_disease.png | 20_round2_diagnostic/figures.py::figure8_pool_stratum | 2026-07-02T17:09:49Z | yes | Gene-disease decline by pool-size stratum (robustness check 3). |
| 24 | Round 2 dynamics (step 20) | fig9_encoder_property_scatter.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/20_round2_diagnostic/fig9_encoder_property_scatter.png | 20_round2_diagnostic/figures.py::figure9_encoder_property_scatter | 2026-07-02T17:09:49Z | yes | Exploratory encoder properties vs erosion magnitude. |
| 25 | Qualitative errors | fig10_failure_mode_summary.png | /home/b5ac/freddieyu.b5ac/projects/project_1/figures/20_round2_diagnostic/fig10_failure_mode_summary.png | 20_round2_diagnostic/figures.py::figure10_failure_modes | 2026-07-02T17:09:49Z | yes | Failure-mode rates among genuine missed-positive errors. |

### Table 15.B — Figures documented in Sections 5.5 / 6.4 of this record

| # | Section reference | Figure filename | Present in Table 15.A? |
|---|-------------------|-----------------|------------------------|
| 1 | Section 5.5 | fig1_benchmark_kb_scatter.png | yes |
| 2 | Section 5.5 | fig2_variance_between_encoder.png | yes |
| 3 | Section 5.5 | fig3_easy_hard_ranking_validity.png | yes |
| 4 | Section 5.5 | fig4_finetuning_lift.png | yes |
| 5 | Section 5.5 | recipe_spread_vs_deberta_health.png | yes |
| 6 | Section 6.4 | fig1_per_seed_trajectories.png | yes |
| 7 | Section 6.4 | fig2_within_seed_paired_change.png | yes |
| 8 | Section 6.4 | fig3_hard_easy_pair_type.png | yes |
| 9 | Section 6.4 | fig4_robustness_well_trained.png | yes |
| 10 | Section 6.4 | fig5_gene_disease_hard_trajectories.png | yes |
| 11 | Section 6.4 | fig6_pair_type_subset_contrast.png | yes |
| 12 | Section 6.4 | fig7_kb_peak_timing.png | yes |
| 13 | Section 6.4 | fig8_pool_stratum_gene_disease.png | yes |
| 14 | Section 6.4 | fig9_encoder_property_scatter.png | yes |
| 15 | Section 6.4 | fig10_failure_mode_summary.png | yes |

**Missing figures referenced elsewhere in this record.** Section 1 references `assets/figures/study_design.png` as the study-design schematic. That path does not exist on disk (Table 15.A row 1, status missing). The author should either add the asset or remove the reference from the manuscript.

**Orphan on-disk figures.** Nine figures from preparation steps 00–05 (rows 2–9) and Round 2 diagnostic figures fig7–fig10 (rows 22–25) are on disk but not narrated in Sections 5.5 or 6.4. They may belong in Methods, Supplement, or may have been intentionally omitted from the record's figure summaries.

**Rename or version risk.** No on-disk figure filenames contain `biomedbert`, `_v1`, `_old`, or `_deprecated`. The Jul 2026 regeneration batch uses BlueBERT labels in step-20 trajectory figures. Legacy slurm logs may still say BioMedBERT; figure files themselves are clean.

<!-- Task D complete -->

---

## 16. Priority action list for paper alignment (Task E)

*Based on Sections 12–15 (provisional where `.tex` unavailable), Tasks B–C, and Section 11 flags.*

### Priority 1 — Must change (impacts main claims)

1. **Association analysis (Results/Discussion):** Confirm whether manuscript uses BioRED-only `benchmark_f1` or pair-type-matched columns; gene-drug ρ flips between −0.26 and +0.02.
2. **Gene-drug association claim:** If paper states negative gene-drug benchmark–KB correlation, revisit; latest pair-matched estimate is +0.018 with CI spanning zero.
3. **Biomedical-encoder erosion narrative:** Replace uniform-decline wording; BlueBERT rises on gene-disease (+0.108) and hard (+0.0144).
4. **Fine-tuning lift magnitude:** Update any 0.378 benchmark-lift citation to 0.179 (current `11_untrained_floor_lift.csv` mean).
5. **Bootstrap probability:** Replace 99.1% with 98.2% wherever gene-disease-hard sign stability is cited.

### Priority 2 — Should change (numeric update, claim direction unchanged)

6. **Encoder benchmark spread:** 0.025 → 0.035 if cited from early Round 1 log.
7. **Pairable seeds (Round 2):** 65 → 69 in methods/results footnotes.
8. **Gene-disease pooled Δ:** −0.0569 → −0.0541 if README-template number used.
9. **BlueBERT naming:** Ensure all figure labels say BlueBERT, not BioMedBERT (`bluebert_base` on disk).
10. **Epoch checkpoints scored:** 498 → 720 if older partial-scoring count appears.
11. **Per-encoder lift table:** Rescore fig4/table after BlueBERT fix; several lifts changed sign.

### Priority 3 — Nice to have (cleanup and consistency)

12. **`20_round2_diagnostic/README.md`:** Regenerate from latest CSVs (still shows 99.1%, 65 seeds).
13. **`assets/figures/study_design.png`:** Add or remove `\includegraphics` reference.
14. **Association code/doc alignment:** Reconcile `analysis.py` (uses `benchmark_f1`) with stored CSV (pair-type-matched x).
15. **Bootstrap CSV export:** Persist `frac_negative_bootstrap` to CSV for traceability.
16. **Round 2 fig7–fig10:** Verify paper includes or deliberately omits newer diagnostic figures.

### Priority 4 — Consider revisiting (needs human judgment)

17. **Gene-drug association interpretation:** Near-zero positive ρ may not warrant a directional claim either way.
18. **Hard-concentrated erosion:** Easy-subset also declines; "hard-specific" framing may overstate.
19. **Encoder-property Spearman (+0.80):** n = 9 exploratory; cite with caution or move to supplement.
20. **ECE–benchmark ρ (−0.833):** Confirm computation path before citing (`11_benchmark_ece_correlations.csv` vs report).
21. **Full `.tex` numeric audit:** Re-run Section 12 once paper repository path is supplied.

<!-- Task E complete -->

---

## 17. Missing figure resolution

**Table 15.B reconciliation (2026-07-03).** Table 15.B lists fifteen figures from Sections 5.5 and 6.4; **all fifteen rows show "Present in Table 15.A? = yes"**. Disk verification at artifact-root paths under `/home/b5ac/freddieyu.b5ac/projects/project_1/figures/` confirms every filename exists (mtime 2026-07-02 regeneration batch). **No Table 15.B inline corrections were required.** The "twelve missing / Present in Table 15.A? = no" premise in the Section 17 task does not match the current document after the Jul 2026 regeneration pass.

**Likely source of a "missing" impression.** (1) Figures live under **artifact root** `../projects/project_1/figures/`, not under code-root `project_1/figures/` (that directory does not exist). (2) Legacy filenames from pre-regeneration Round 2 (`fig1_within_seed_paired_change.png`, etc.) were deleted by `manuscript_regenerate/figures.py::regenerate_step20`. (3) `11_round1_analysis/figures.py` emits **different** Round 1 names (`fig2_variance_components.png`, `fig3_easy_hard_prerequisite.png`) than Sections 5.5 / Table 15.B (`fig2_variance_between_encoder.png`, `fig3_easy_hard_ranking_validity.png`); the manuscript-regeneration path is authoritative.

Because Table 15.B contains zero "no" entries, the table below resolves **twelve plausible missing-filename queries** (legacy aliases, dropped figures, paper-specific assets never in this record, study-design asset). All fifteen Table 15.B figures are on disk; regenerate commands if ever needed: Round 1 quartet + sweep → `cd project_1 && python -c "from manuscript_regenerate.figures import regenerate_step11, regenerate_step10; regenerate_step10(); regenerate_step11()"`; Round 2 decuple → `cd project_1/20_round2_diagnostic && python run.py --analyze-only --skip-stratum-inference` (or `manuscript_regenerate.figures.regenerate_step20()`).

| # | Missing filename | Verdict | Actual path or command | Notes |
|---|------------------|---------|------------------------|-------|
| 1 | study_design.png | GENERATOR_MISSING | — | Referenced Section 1 / Table 15.A row 1. No generator; `projects/project_1/assets/` directory absent. Manual asset or remove reference. |
| 2 | fig2_variance_components.png | FOUND_UNDER_DIFFERENT_NAME | `../projects/project_1/figures/11_round1_analysis/fig2_variance_between_encoder.png` | Legacy output of `11_round1_analysis/figures.py::figure2_variance_components`. Authoritative name from `manuscript_regenerate/figures.py::regenerate_step11`. |
| 3 | fig3_easy_hard_prerequisite.png | FOUND_UNDER_DIFFERENT_NAME | `../projects/project_1/figures/11_round1_analysis/fig3_easy_hard_ranking_validity.png` | Legacy output of `11_round1_analysis/figures.py::figure3_easy_hard_prerequisite`. Rename search: no `*prerequisite*` PNG on disk. |
| 4 | fig1_within_seed_paired_change.png | FOUND_UNDER_DIFFERENT_NAME | `../projects/project_1/figures/20_round2_diagnostic/fig2_within_seed_paired_change.png` | Pre–Jul-2026 manuscript-regen name; explicitly removed in `regenerate_step20` (L527–532). |
| 5 | fig3_gene_disease_subset_contrast.png | FOUND_UNDER_DIFFERENT_NAME | `../projects/project_1/figures/20_round2_diagnostic/fig6_pair_type_subset_contrast.png` | Legacy Round 2 name; removed by `regenerate_step20`. Substring search: `pair_type_subset`, `gene_disease_subset` → only fig6. |
| 6 | fig2_pair_type_asymmetry.png | GENERATOR_MISSING | — | Legacy Round 2 name removed by `regenerate_step20`; no current function emits this filename. Content split across `figure3_hard_easy_pair_type` and `figure6_pair_type_subset_contrast`. |
| 7 | fig4_calibration_benchmark_ece.png | REGENERABLE_NOW | `cd project_1/11_round1_analysis && python -c "import pandas as pd; from figures import generate_publication_figures; from config import OUTPUT_DIR; generate_publication_figures(pd.read_csv(OUTPUT_DIR/'11_encoder_summary.csv'), pd.read_csv(OUTPUT_DIR/'11_easy_hard_ranking.csv'), pd.read_csv(OUTPUT_DIR/'11_variance_components.csv'), pd.read_csv(OUTPUT_DIR/'11_variance_components_bootstrap.csv'), lift_df=None)"` | Optional fifth figure from native Round 1 pipeline; not in Table 15.B. Inputs `11_encoder_summary.csv`, `11_easy_hard_ranking.csv`, `11_variance_components.csv`, `11_variance_components_bootstrap.csv` all present. |
| 8 | fig5_training_trajectory.png | GENERATOR_MISSING | — | Paper Section-2 asset from a removed `manuscript_regenerate/results_section2.py` (not in current repo). Distinct from on-disk `fig5_gene_disease_hard_trajectories.png`. No rename candidate. |
| 9 | fig9_encoder_heterogeneity.png | GENERATOR_MISSING | — | Paper Section-2 asset (erosion vs biomedical pretrain); not the same as on-disk `fig9_encoder_property_scatter.png`. Script absent from current codebase. |
| 10 | direction_balance.png | GENERATOR_MISSING | — | Mentioned only as explicitly dropped in `manuscript_regenerate/figures.py::regenerate_step00` (L79). No PNG anywhere under `figures/`, `outputs/*/figures/`, or `assets/figures/`. |
| 11 | alignment_rates.png | GENERATOR_MISSING | — | Same as row 10; dropped in `regenerate_step00`. Inputs for step 00 (`evaluable_target_summary.csv`, `entity_pair_breakdown.csv`) exist but no figure function remains. |
| 12 | 01_oncology_fraction_by_criterion.png | GENERATOR_MISSING | — | Named in `manuscript_regenerate/reports_step00_02.py` prose only. `regenerate_step01` emits two figures (`01_corpus_granularity_ladder.png`, `01_corpus_pmid_leakage.png`); no oncology figure function in current code. Input `oncology_fractions_by_criterion.csv` present. |

**Summary.** Table 15.B figures (Sections 5.5 and 6.4) are **not missing** — all fifteen are on disk at artifact-root paths. Of the twelve supplementary queries above: **four** resolve by rename to a current on-disk file; **one** (`fig4_calibration_benchmark_ece.png`) is regenerable now via `11_round1_analysis/figures.py` with inputs present; **zero** require an upstream rerun before regeneration; **seven** have no current generator (study design, three legacy Round 2 names, two dropped step-00 figures, two paper-specific Section-2 assets). Authors searching under code-root `project_1/figures/` will find zero PNGs; use `../projects/project_1/figures/` instead.

---

*End of experiment record (Sections 12–17 appended 2026-07-03).*
