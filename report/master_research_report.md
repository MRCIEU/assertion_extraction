# Master research report (writer-facing dossier)

*Single reference for later thesis/paper drafting: narrative synthesis, explicit RQ answers, and pointers to evidence. It is not a raw paste of subproject files. Line-level tables: `report/master_tables_index.csv`. Writing aids: `report/writing_support/`.*

---

### Current status at a glance

| Item | Status |
|------|--------|
| **Schema** | **`S2_current` frozen** for T1–T4 and registry |
| **Rerun** | **Complete** — `rerun_main_aggregated_results.csv` |
| **External eval** | **BioRED + BC5CDR done** (`primary_external_results.csv`, closure) |
| **DrugProt test** | **Blocked** — no packaged `test` (`drugprot_unresolved_status.json`) |
| **Human audit** | **Deferred** — proxies only |
| **Selection policy (benchmarks)** | **Default = M015** (`benchmark_generalization_heavy`); **secondary = M021** — `report/decision_analysis/final_model_selection_report.md` |
| **Final downstream decision framework** | **Frozen** — **split** objectives: `report/decision_analysis/final_model_selection_rule.json` (benchmarks) + `knowledge_grounded_evidence_audit/data/processed/final_downstream_transfer_selection_rule.json` (audit/surfacing/conservative). Same evidence base; **different** default families by objective. |
| **Downstream transfer (audit utility)** | **Complete** — Tier-1 (3714406) + Tier-2 (3716065); `knowledge_grounded_evidence_audit/reports/downstream_transfer_final_report.md`, `final_project_joined_model_table.csv` |
| **Claim / epistemic boundary** | **No** clinical validity or therapeutic discovery claims. Downstream audit metrics are **gold-lite proxies** (heuristic labels), not validated clinical assertions. Oracle-style metrics are **diagnostic** (near-zero magnitudes), not proof of ceiling performance. |
| **Residual risk** | Drug–gene / DrugProt; deployment vs benchmark; **explicit weighting** required; engineering variability |

---

### Phase milestones (scan)

| Phase | Status | Main output | Main conclusion (final-answer form) | Next dependency |
|-------|--------|-------------|--------------------------------------|-----------------|
| Dataset inventory | Complete | Tiering, staging, profiling | **No single public resource** defines full precision-oncology assertion extraction; inventory bounds what is trainable vs blocked. | — |
| Oncology projection | Complete | Retention / pair-world quantification | **Planning/weighting signal only** — projection does **not** create oncology relation gold (see §5). | — |
| Schema exploration | Complete | `S2_current` + refined paths | **`S2_current` chosen and frozen** for operational packaging; refined schemas are **documented non-defaults**. | — |
| Training data generation | Complete | T1–T4 packages + provenance | **T1–T4** packages separate gold vs weak vs unlabeled under `S2_current` mapping with traceability. | — |
| Fine-tuning (HR rerun) | Complete | Multi-seed aggregates | **Internal protocol** establishes **relative** anchors and **controlled contrasts** (e.g. M015 mean macro F1 **0.7683 ± 0.0182**); **internal ranking alone is insufficient** for deployment claims once external/downstream layers exist. | — |
| External evaluation | **Complete for protocol** (BioRED+BC5CDR; DrugProt blocked; audit deferred) | Primary + subset + reliability tables | **No universal winner** across BioRED and BC5CDR + slices; ranking **partially** overlaps internal order; **explicit policy** in `decision_analysis/` resolves tradeoffs. | — |
| Downstream transfer sweep | **Complete** — Tier-1 (3714406) + Tier-2 (3716065) | Joined table + split policy + bridge | **HR macro-F1 is not a reliable proxy** for gold-lite audit yield; **setting effects are family-specific** (e.g. M025 mean pred_nonnegative **~0.2** under S2 vs **~13.4** under S1, Tier-2); **benchmark-balanced and audit-surfacing defaults diverge** — **operating profiles required**. | Optional gold-lite revision only if revisiting proxy labels |

---

## 1. Background and problem context

Biomedical **relation extraction (RE)** classifies whether and how two (or more) entity mentions stand in a stated semantic relation (e.g. drug–target, gene–disease). **Assertion extraction**, as used in this project, targets **clinical and biomarker-oriented statements** whose correct use depends not only on a relation label but on **assertion type**, **evidence scope**, and **interpretive semantics** (e.g. predictive vs diagnostic vs resistance-oriented claims in oncology). Assertion extraction is therefore **not** reducible to generic relation classification on a benchmark ontology: it additionally requires that the **operational schema**, **supervision type**, and **unit of evidence** support the intended downstream use in precision oncology.

In **cancer and precision oncology**, surface forms carry **high-stakes clinical interpretations** (predictive vs diagnostic biomarkers, resistance, therapeutic response), while public corpora remain **heterogeneous**: some are general biomedical RE benchmarks, others are chemistry-focused, others are **sentence-level mining** or **KB rows** without token alignment.

**Domain-specific fine-tuning** is therefore non-trivial: one cannot assume that a single open dataset defines the task. **Relation extraction alone** is insufficient if the **label schema**, **evidence unit** (document vs sentence), and **supervision type** (gold spans vs weak priors) are not aligned with the operational scientific question. This project treats **schema choice**, **supervision design and data packaging**, **training design**, and **external evaluation** as **first-class** research problems—not as preprocessing afterthoughts.

---

## 2. Motivation and research gap

**Motivation (project-level):**

1. **No single mature, public, oncology-only assertion dataset** spans all entity families and relation semantics needed for a defensible precision-oncology extraction system. Accessible resources mix **BigBio-style RE** (BioRED, DrugProt, BC5CDR), **weak mining** (CIViCmine, CancerMine), **clinical assertion semantics** (CIViC KB), **unlabeled oncology text**, and **BRONCO-style** full-text material. **BRONCO** is **not missing**: the English manual release is a **strong oncology relation-world anchor** on disk, but it is **engineering-blocked** in this pipeline until a dedicated span reader maps tabbed full text into the same tensor regime as BigBio RE corpora.

2. **Heterogeneity** implies that **direct fine-tuning** on a naive union of corpora risks **label collision**, **negative transfer**, and **illusory benchmark success** that does not transfer to **realistic oncology assertions** in clinical-genomics text.

3. **Benchmark success is an insufficient proxy for oncology usability.** The project is specifically concerned with failures such as: **transfer collapse** (strong held-out dev on a narrow shard, weak performance on official test splits or shifted corpora); **oncology-subset failure** (good aggregate F1 but poor behavior on biomarker- or variant-centric slices); and **output unusability** (degenerate predictions, unstable loss branches, or schema-incompatible extractions) despite acceptable headline metrics. These risks motivate the **high-budget rerun** and **external evaluation** phases—not cosmetic validation.

These points are documented quantitatively in the **dataset inventory (v2)** and the **oncology projection audit** (see index).

---

## 3. Research questions

The work proceeds as a **causal chain**. **Thesis and paper writing should foreground *headline* RQs** (schema, training, external stress, downstream transfer). **Supporting RQs** (data inventory, oncology projection, packaging) justify **design and feasibility** and belong in Methods/background — not as co-equal “results headlines” unless the chapter scope requires it.

### Headline RQs (foreground in Introduction / Results)

| RQ | Focus | In plain language | Primary answering phase |
|----|--------|-------------------|-------------------------|
| **RQ-schema** | Defensible coarse label space under mapping distortion | Which **operational relation/assertion schema** balances trainability, mapping loss, and external semantic stress tests? | Schema exploration |
| **RQ-train** | Internal stability and relative strength | Which **encoder, architecture, schedule, T3/T4, update regime, loss** configurations are **stable and relatively strong** on the *internal* scientific protocol? | Fine-tuning design + rerun |
| **RQ-ext** | External stress and ranking | Do internally favored configurations **hold up** on **split_external** benchmarks and subsets—without treating weak probes as gold? | External evaluation |
| **RQ-downstream** | Audit utility vs training metrics | Do **internal or benchmark metrics** predict **KB-anchored audit utility** on documented **gold-lite** proxies — and how do **settings and families** interact? | Downstream transfer (`knowledge_grounded_evidence_audit/`, §9A) |

### Supporting RQs (design / operational)

| RQ | Focus | In plain language | Primary answering phase |
|----|--------|-------------------|-------------------------|
| **RQ-data** | What exists and what is trainable *now*? | Which corpora are **supervised span RE**, **weak/KB**, **unlabeled**, or **engineering-blocked**? | Dataset inventory |
| **RQ-onc** | Oncology mass under fixed rules | How much is **oncology-facing** under explicit rules, for **weighting and slice design** (not new gold)? | Oncology projection |
| **RQ-pack** | Supervision roles and stream separation | How are **gold, weak, unlabeled** **typed and routed** in T1–T4? | Training data generation |

**Dependency order (unchanged):** RQ-data / RQ-onc bound candidates → RQ-schema → RQ-pack → RQ-train → RQ-ext; **RQ-downstream** uses the same model families and adds an **orthogonal** audit layer — it does **not** replace RQ-ext.

**Maps:** `report/writing_support/rq_to_final_answer_map.csv`, `rq_to_evidence_map.csv`.

---

## 4. Data landscape and dataset collection

**Purpose:** Without a structured inventory, the project could not honestly claim which corpora are **training-ready supervised RE**, which are **weak only**, and which are **present but not yet in tensor form**.

**How it was designed:** Profiling scripts produced `profiles/*.json`; a **v2 decision brief** added staging (T1–T6), schema audit, and heuristic oncology projections. Companion CSVs list status per `dataset_id`.

**What it found (condensed):**

- **Tier A (training-ready supervised RE):** BioRED, DrugProt, BC5CDR—document-level relations with entity spans in the BigBio view used here.
- **Tier B (weak / KB / unlabeled):** CancerMine, CIViC KB, CIViCmine (sentence + collated), oncology lung PubMed (unlabeled).
- **Tier C (present on disk; engineering-blocked for this pipeline):** English BRONCO 2015 manual release—**not** absent data; **not** comparable “doc counts” to BigBio RE corpora without a dedicated span reader and unified preprocessing.
- **Tier D (truly unavailable here):** Precision Oncology Concept Corpus annotations **not on disk**—strategic for concept-span evaluation when obtained.

**Justificatory note:** Taken together, the selected accessible resources span **supervised biomedical RE**, **oncology-relevant weak evidence**, **clinical assertion semantics** (CIViC), **unlabeled in-domain text**, and **oncology relation-world anchors** (BRONCO, when wired)—covering the main open-resource modalities relevant to this project’s aims, while honestly marking **gaps** where gold is missing or blocked.

**Insight:** The project has a **strong backbone** plus **partial** oncology signal, but **no** complete public substitute for **span-level oncology assertion gold** combining all desired families.

**Implication:** Oncology projection quantified **retention** and **pair-type** structure so “dataset exists” became “dataset *usable for oncology-facing design* under stated rules.”

**Core quantitative snapshot (illustrative rows from inventory v2):**

| dataset_id | supervision_type (summary) | oncology_specificity | immediate_blocker |
|------------|----------------------------|----------------------|-------------------|
| biored | supervised RE/NER | general_biomedical; high disease signal | none for backbone |
| drugprot | supervised RE/NER | indirect via abstract | no official test in mirror (noted in brief) |
| bc5cdr | supervised RE/NER | mixed; many cancer diseases | none |
| civicmine | sentence weak + collated | mined biomarkers | not span gold; noise |
| bronco | full-text tabbed + mapped | high English oncology | span reader not implemented |
| precision_oncology_concepts | concept spans (when available) | very high | annotations not on disk |

**Resource roles (compact — reduces later confusion):**

| Role | Example resources (see inventory for full list) |
|------|-----------------------------------------------|
| **Training (supervised RE, T1–T2)** | BioRED, DrugProt, BC5CDR shards as packaged under `S2_current` |
| **External evaluation (Layer A)** | BioRED / BC5CDR official test pairs (DrugProt **blocked** — no test split in package) |
| **Weak supervision / KB semantics (T3)** | CIViC KB, CIViCmine, CancerMine — **not** span RE gold |
| **Unlabeled adaptation (T4)** | Lung oncology PubMed — MLM/DA only |
| **Downstream audit (proxy)** | Gold-lite targets under `knowledge_grounded_evidence_audit/` — **heuristic** labels, not training gold |
| **Anchoring / external alignment only** | External constraint JSONs; BRONCO **on disk** as relation-world anchor — **not** in tensor pipeline until reader exists |
| **Future engineering only** | BRONCO span reader; Precision-O annotations when obtained |

*Full profiling table: `reports/dataset_inventory_summary_v2.md` §5 and `reports/tables/dataset_status_decision.csv`.*

---

## 5. Oncology projection

**Purpose:** The inventory establishes **availability**; projection establishes **oncology-facing mass and structure** of what could enter **T2** and **evaluation slices**.

**How designed:** Layer A = sources; Layer B = projected slices under **fixed English cancer lexicon** and structural rules (exact for CIViC fields, heuristic for abstract/disease text). Documented as **not clinical gold**.

**What it found (examples):** BioRED train (partial file): ~**22%** document retention, ~**25%** relation retention; DrugProt ~**21%** / **23%**; BC5CDR cancer-like CID slice **very small** (high variance expected). CIViC rows: **high** retention at row level but **no spans**.

**Critical limitation:** Oncology projection is a **selection and quantification layer** for **task construction and weighting** only—it does **not** create **new oncology gold labels** for training or evaluation. It informs **how much** oncology-facing mass exists under fixed rules; it is **not** a substitute for curated oncology relation gold.

**Insight:** Shared **entity families** (gene, disease, chemical, variant) appear across corpora but **native relation ontologies differ**; **coarse collapse** is required for multitask training. Heuristic slices **narrow context** without creating new relation gold.

**Implication:** Projection informed both **relation-family collapse** in schema design and the **expected support and variance** of oncology-focused training slices (T2 thinness, reliance on T1 backbone)—linking quantitative retention to **risk profile**, not to new labels.

---

## 6. Schema exploration

**Purpose:** With heterogeneous sources, **schema selection** determines **mapping distortion**, **trainable heads**, and **what clinical semantics are even representable** in the **operational** label space.

**How designed (unified program):**

1. **Initial round (internal):** Candidates **S1–S4** compared on **projected** evidence—packaging simulation, stability, composite scores. **S2** (oncology-facing coarse buckets including **CLINICAL_ASSERTION** / **BIOMARKER_EVIDENCE**) **near-tied** **S1** (minimal shared) on balanced composite—operational default **S2** chosen for oncology-facing buckets.

2. **Second round (externally anchored):** **External constraint JSONs** (entity, relation/assertion, pairing) formalize BRONCO-style **relation worlds**, CIViC **assertion-type** semantics, and core pairings. **Gap audit** of **S2_current** shows, e.g., **high severity**: CIViC assertion subtypes **collapse** into one **CLINICAL_ASSERTION** head; medium: DrugProt mechanism shades collapsed into **DRUG_GENE_REGULATION**; population/outcome families sparse in span gold.

**What it found:** **S2_current** remains **operationally recommended**: strong on **core span-supervised pairings** trainable on BioRED/DrugProt/BC5CDR; **externally coarse** for CIViC subtype semantics unless weak/KB packaging adopts **S2_refined_relation** or **S2_refined_hybrid**.

**Concrete output of this phase:** The project **froze `S2_current` as the operational relation/assertion label space** for training-data JSONL and the fine-tuning experiment registry, while **documenting** refined schemas (`S2_refined_*`, **S4**) as **non-default upgrade paths** pending packaging and trainer adoption.

**Final operational schema — what is frozen vs not**

| Frozen under `S2_current` for this project | Not frozen / non-operational default |
|-------------------------------------------|-------------------------------------|
| Label set used in **generated T1–T4 JSONL** and **fine-tuning registry** experiments | **`S2_refined_hybrid`**, **`S2_refined_relation`**, **`S4`**: documented upgrade paths only |
| Mapping from source-native labels to coarse heads (with metadata preserving source labels) | BRONCO / Precision-O **tensor** pipelines |
| External anchor **gap audit** outcomes as **constraints on claims**, not as automatic triggers to switch schema | Switching default schema without packaging + trainer adoption |

**Methods placement:** State **`S2_current`** as the **operational** schema; cite **`S2_refined_*`** only as **candidates** in Discussion or Appendix per `report/writing_support/main_text_vs_appendix_plan.csv`.

**Why `S2_refined_hybrid` is not the operational default:** Tabulated **external alignment** scores can make **S2_refined_hybrid** look preferable on paper. Higher scores do **not** automatically justify switching the live default: refined variants add **heads and mappings** that may be **weakly supported** by span gold, **not yet integrated** into packaging and routing, or **imbalanced** relative to auxiliary losses. Operational default status requires **joint** evidence of feasibility, stability, and end-to-end trainer support—not composite metrics alone.

**Trainability scores:** A **train_feas** (or similar) entry reflects **feasibility under current mapping and packaging assumptions** and documented head coverage—not proof that every candidate head receives **substantive** span-level supervision.

**Meaning of labels now:**

- **S1:** Minimal backbone—fallback when minimizing heads; weakest CIViC assertion semantics when collapsed to generic association.
- **S2_current:** **Operational** coarse oncology-oriented **relation/assertion** schema used in training data and fine-tuning registry—not claimed as ontologically complete.
- **S4:** Expressive oncology relations—**deferred** until BRONCO span reader + richer concept resources justify the mapping cost.
- **S2_refined_hybrid:** **Sanctioned upgrade path**—adds population/outcome entity heads (sparse gold) and splits CIViC assertion types on **KB/weak** rows; **not** the default until packaging + trainer explicitly adopt it.

**Refined schema quality (scores from `schema_exploration/reports/tables/refined_schema_quality_metrics.csv`; columns: external entity / relation alignment, train feasibility, composite usefulness):**

| schema_id | ext_entity | ext_relation | train_feas | usefulness |
|-----------|------------|--------------|------------|------------|
| S2_current | 0.72 | 0.68 | 0.88 | 0.84 |
| S2_refined_relation | 0.72 | 0.82 | 0.88 | 0.88 |
| S2_refined_hybrid | 0.78 | 0.82 | 0.88 | 0.89 |

**Candidate coverage (S1–S4 vs this table):** The **first** internal round **screened S1–S4** on projected packaging evidence (composite / stability) and narrative trade-offs (see “Meaning of labels”). **S1** and **S4** remain documented as **minimal** and **deferred expressive** options; they are **not** row-matched in the spreadsheet above. The numeric block compares **`S2_current`** to **`S2_refined_*`** only—because closure documentation **quantifies** external-alignment and train-feasibility for **operational and upgrade-path** S2 variants, not a full S1–S4 score matrix. **S4** is **not** evaluated in that CSV; it stays a **documented** future path pending packaging cost and span resources.

**Implication:** **Frozen operational mapping** to **S2_current** for T1/T2 JSONL generation with **source-native labels preserved alongside mapped labels** for traceability; refined variants documented for **future pilots** on T3/T4-heavy configs.

---

## 7. Training data generation

**Why not trivial:** Packaging must preserve **provenance**, **supervision_level**, **gold vs weak**, **schema mapping metadata**, and **leakage-safe** shard boundaries—while **not materializing** full negative pools in JSONL (trainer-side sampling). Readiness, leakage, and provenance checks exist specifically to **prevent conflating** gold supervised streams, weak priors, unlabeled adaptation text, and merged shards in routing and analysis.

**How T1–T4 were defined:**

| Stage | Role | Contents (summary) |
|-------|------|-------------------|
| **T1** | Supervised backbone | BioRED, DrugProt, BC5CDR shards + optional merged stream |
| **T2** | Oncology bridge | **Projected** BioRED/DrugProt + **small** BC5CDR cancer slice + merged |
| **T3** | Weak oncology semantics | T3a CIViC KB rows; T3b CIViCmine sentences; T3c CancerMine—**not** span RE gold |
| **T4** | Unlabeled adaptation | Lung oncology PubMed docs—**MLM / DA only** |
| **T5/T6** | **Deferred placeholders** (not active training packages) | BRONCO reader engineering; Precision-O acquisition—dependencies for future stages, **not** instantiated like T1–T4 |

**Gold vs weak vs unlabeled:** Enforced via `is_gold_supervision`, `readiness_level`, and trainer routing (`active_t1_shards`, etc.). **All generated T1–T4 packages are instantiated under the operational schema `S2_current`**, with **source-native relation/assertion labels retained in metadata** alongside **mapped** labels for traceability and audit—not a single opaque relabel pass.

**What it enabled:** Scientific multi-stage trainer consuming **real shards**, with **T3** auxiliary smoothing and **T4** MLM **without inventing** relation labels.

**Insight:** T2 BC5CDR slice is **intentionally thin**—high variance unless regularized with T1 BC5CDR or down-weighted.

**Core T1 package scale (from generation report):**

| package_file | n_samples | gold_relation_mentions |
|--------------|-----------|------------------------|
| t1_biored.jsonl | 600 | 5935 |
| t1_drugprot.jsonl | 4250 | 21053 |
| t1_bc5cdr.jsonl | 1500 | 47813 |
| t1_supervised_backbone_merged.jsonl | 6350 | 74801 |

**Methods vs appendix:** **Methods** should state **stage definitions (T1–T4)**, **`S2_current` mapping policy**, **leakage/readiness rules**, and **representative package scales** (table above). **Appendix** can hold **full package readiness matrices**, **per-shard statistics**, and **extended provenance tables** (`package_readiness_matrix.csv`, generation report annexes) — see `report/writing_support/main_text_vs_appendix_plan.csv`.

---

## 8. Fine-tuning design and rerun results

**Purpose:** The registry grid isolates **scientific factors** (encoder, pipeline vs shared multitask, **T1_to_T2** vs **T1_T4_T2** vs **T1_T2_T3_aux**, full vs top-4 update, CE vs weighted CE vs focal) under a **stage-aware trainer** with real checkpoints and exports—not a BioRED-only smoke loop.

**First-pass batch (31 runs):** All completed on **GPU** with artifacts; **result triage** flagged **weighted CE instability (M026)**, **S001/S002 last-step artifacts**, and **inconclusive** encoder/architecture gaps at **short step budgets**. Motivated **targeted high-budget rerun (HR)** with **best-checkpoint on macro F1** and **five seeds**.

**Rerun (HR) — what is established (provisional, internal-dev only):** All statements below are **internal** to the rerun protocol and **dev-side** aggregates; they calibrate **relative** strength and **controlled contrasts** on the documented dev regime—they do **not** substitute for **split_external** tables or **deployment** truth.

- **Strongest mean macro F1 in the main Group A comparison (this rerun):** **M015** (BioLinkBERT, pipeline, T1_to_T2, full finetune, CE)—**0.7683 ± 0.0182** across 5 seeds. This is the **strongest internal rerun configuration under stated conditions**, **not** a validated global winner until external evaluation confirms ranking.
- **PubMedBERT pipeline M003** is a **strong internal baseline** (**0.7554 ± 0.0078**). **Shared-multitask M009** shows **lower mean** macro F1 than the strongest **pipeline** branches in these head-to-head aggregates but **higher seed-to-seed spread**—**not** evidence that shared multitask is universally inferior; it is evidence that **under this protocol**, the **stronger pipeline branch outperformed the strongest shared branch** in the primary comparisons, with **shared** exhibiting **greater instability** in some cases.
- **T3 auxiliary M005:** **negative transfer** signal vs M003 on this protocol (**~0.60** mean F1)—a **control** result, not a universal claim about T3.
- **T4:** **heterogeneous**—can help or hurt depending on paired comparison (e.g. M010 vs M009 vs M004 vs M003).
- **Top-4 CE M025 vs full M009:** In **this** rerun, **M025** achieved **higher mean** macro F1 than **M009** on the aggregate table—**conditional** on shared encoder, schedule, and budget. This should be read as a **protocol-specific** observation that may reflect **optimization stability**, **effective capacity**, or **interaction with the dev shard**, **not** as evidence that **top-4 tuning is generally superior** to full fine-tuning.
- **M021 vs M009:** These lines differ by **encoder** (BioLinkBERT vs PubMedBERT) and **configuration**, not architecture alone—**not** a clean pipeline-vs-shared ablation.
- **Weighted CE branches:** **M026** trails **M025** and remains **high-variance**. **S001** and **S002** show **high mean** F1 on the **weighted-CE secondary branch** but are **retained only as evaluation candidates** for external stress testing—**not** promoted to **default training recommendations** pending stability and external confirmation.
- **Anomaly table** records **high final-stage F1 swing** and **majority-class** flags for specific runs/seeds.

**What remains weak:** **M027** incomplete seed; any conclusion resting on **tiny dev** or **single-metric** snapshots without **external** confirmation.

**Headline comparison table (rerun aggregates, abbreviated; internal only):**

| Base | Encoder | Arch | Schedule | Mean macro F1 | Std |
|------|---------|------|----------|---------------|-----|
| M015 | biolinkbert | pipeline | T1_to_T2 | 0.7683 | 0.0182 |
| S002 | pubmedbert | shared | T1_to_T2 | 0.7612 | 0.0117 |
| M003 | pubmedbert | pipeline | T1_to_T2 | 0.7554 | 0.0078 |
| M021 | biolinkbert | shared | T1_to_T2 | 0.7474 | 0.0739 |
| M009 | pubmedbert | shared | T1_to_T2 | 0.7085 | 0.0954 |
| M005 | pubmedbert | pipeline | T1_T2_T3_aux | 0.5978 | 0.0204 |

*Full rows: `fine_tuning_experiments/reports/tables/rerun_main_aggregated_results.csv`.*

**Implication:** External **split_external** tables and **`report/decision_analysis/`** formalize selection; internal ordering is **calibrated** against BioRED/BC5CDR, not treated as sufficient alone.

**Final writing stance:** The **HR rerun** establishes **internal anchors and controlled contrasts** (families, losses, schedules)—**not** final statements about official-test or production behavior. **External evaluation** and **downstream transfer** **supersede internal-only ranking** for any claim about **benchmark deployment**, **slice behavior**, or **audit utility**. Internal numbers remain **necessary** to interpret *why* a family was trained a certain way — not sufficient for **operational** default choice without the later layers.

---

## 9. External evaluation — design and current status (closure)

**Purpose:** Internal **macro F1 on dev** does not establish **official test** behavior on **BioRED/BC5CDR**, **oncology-facing pairing** subsets, **multi-seed rank stability**, or **schema** stress under shift.

**How designed:** **Evidence taxonomy** preserved: **`split_external`** (Layer A benchmarks), **`realism_probe`**, pairing stratification, ontology/weak entries registered separately. **No score fusion** across evidence types. **Strict pair-level** protocol (`strict_realism_protocol.json`). **Loader gate `proceed`** after checkpoint integrity audit.

**What is complete for the current protocol**

- **BioRED** and **BC5CDR** official test pair evaluation (**five seeds** per shortlist model): `primary_external_results.csv`.
- **Reliability, rank robustness (BioRED), error taxonomy, schema stress, oncology subset** tables under `external_evaluation/reports/tables/`.
- **Closure documentation:** `external_evaluation_closure_status.json`, `external_eval_consistency_audit.json`.
- **Narrative:** `external_evaluation_report.md`, `external_evaluation_comprehensive_report.md`.

**Formal limitations (not “pending” in a vague sense)**

1. **DrugProt official test** — **not evaluated**: packaged **`t1_drugprot.jsonl` has no `test` split**; raw DrugProt lacks a test tree on disk. Documented in **`drugprot_unresolved_status.json`**. No fabricated DrugProt metrics.
2. **Human-coded manual audit** — **deferred**; **`manual_audit_table.csv`** is **proxy-only**.

**Distinction:** Protocol execution and **BioRED/BC5CDR** benchmarks are **complete**; **DrugProt** and **human audit** are **explicit exclusions**, not silent gaps.

**Final answer (one sentence):** External evaluation **partially reproduces** internal ordering on BioRED/BC5CDR but **does not yield a single universal winner** across those benchmarks, slices, and stability — hence the **explicit** selection policy.

**Primary external benchmark snapshot (shortlist, five seeds per source; canonical numbers from `external_evaluation/reports/tables/primary_external_results.csv`):**

| Model | BioRED test mean macro-F1 ± std | BC5CDR test mean macro-F1 ± std | Role (registry) |
|-------|--------------------------------|----------------------------------|-----------------|
| M003 | 0.2837 ± 0.0078 | 0.5483 ± 0.0979 | primary |
| M005 | 0.2746 ± 0.0224 | 0.4798 ± 0.0678 | control |
| M009 | 0.2775 ± 0.0143 | 0.4729 ± 0.0815 | optional |
| M010 | 0.2801 ± 0.0102 | 0.5002 ± 0.0833 | primary |
| M015 | 0.2857 ± 0.0052 | 0.5572 ± 0.0627 | primary |
| M021 | 0.2844 ± 0.0041 | 0.5293 ± 0.0797 | primary |
| M025 | 0.2667 ± 0.0125 | 0.4199 ± 0.0245 | primary |
| M026 | 0.2808 ± 0.0095 | 0.4611 ± 0.0673 | diagnostic |
| S001 | 0.2921 ± 0.0064 | 0.4735 ± 0.0709 | optional |
| S002 | 0.2899 ± 0.0088 | 0.4820 ± 0.0688 | primary |

*DrugProt official test rows are absent (blocked); same as §9 limitations.*

---

## 9A. Downstream transfer to knowledge-grounded audit utilities

**Purpose:** External and internal **relation-extraction** metrics do not specify whether model outputs are **useful for KB-anchored auditing** (proposal volume, linkage buckets, conservative vs surfacing behavior). A dedicated **downstream transfer sweep** under `knowledge_grounded_evidence_audit/` evaluates families on **gold-lite** audit proxies (heuristic labels).

**Evaluation families (routing):**

| Layer | Families | Why (factual) |
|-------|----------|---------------|
| **External BioRED + BC5CDR** | M003, M005, M009, M010, M015, M021, M025, M026, S001, S002 | Completed **`split_external`** rows in `primary_external_results.csv` (five seeds each; DrugProt blocked for all). |
| **Tier-1 downstream** | M003, M004, M005, M009, M010, M015, M021, M025, M026, M027, S001, S002, S003 | Canonical seed **s01** per family in `knowledge_grounded_evidence_audit/manifests/tier1_model_selection.csv` — broad screen of **HR vs audit-proxy decoupling**. |
| **Tier-2 downstream (multiseed)** | M003, M015, M025, M026 | Frozen in `knowledge_grounded_evidence_audit/data/processed/tier2_family_selection_decision.json` — **contrastive** story (surfacing vs benchmark-default conservatism vs diagnostic weighted CE), not top-N by a single metric. |

**Downstream settings (compact definitions; full JSON in `knowledge_grounded_evidence_audit/reports/downstream_setting_definitions.md`):**

| Setting | Retrieval / context / linkage (intent) | What it tests | Note on “improved” |
|---------|------------------------------------------|---------------|---------------------|
| **S1** (`S1_current_realistic`) | **R1** current manifest · **C1** abstract-full · **L1** strict | **Operational** audit surfacing under the **documented** realistic formulation. | Baseline proxy band for proposal volume. |
| **S2** (`S2_improved_realistic`) | **R2** expanded lexical · **C4** richer excerpt window · **L2** relaxed semantic | Whether **lexical + window + relaxed linkage** changes transfer vs S1. | Name reflects **design intent** (more retrieval/context signal); **not** guaranteed to raise yield for every family—**M025** collapses pred_nonnegative here vs S1 under Tier-2 aggregates. |
| **S3** (`S3_oracle_like`) | Oracle **O3** pair+sentence on **C2** evidence sentence, **L1** on predictions | **Upper-bound / diagnostic** slice vs formulation (oracle path). | **macro_f1_heuristic** on this path is **diagnostic only** (near-zero magnitudes)—**not** a ranking signal for model quality. |

**What Tier-1 established (frozen):** **Decoupling** — internal HR macro-F1 does **not** rank **pred_nonnegative** yield on the R1/C1 formulation; several **high-HR** families have **zero** yield, while **M025** and **M003** surface many non-negatives and **M015** (benchmark-default policy line) shows **near-zero** pred_nonnegative on the same proxy. Aggregated **oracle O3** macro-F1 is **near-zero** for almost all families — **not** usable as a sole ranker.

**Tier-2 multiseed summary (job 3716065; `knowledge_grounded_evidence_audit/reports/tables/tier2_multiseed_results.csv`):**

| Family | S1 pred_nonnegative mean ± std | S2 pred_nonnegative mean ± std | S3 oracle-path macro-F1 (heuristic) mean ± std | Interpretation (Tier-2 aggregate) |
|--------|-------------------------------|--------------------------------|------------------------------------------------|-----------------------------------|
| M003 | 14.4 ± 20.032474 | 12.8 ± 24.396721 | 0.00686 ± 0.015339 | Moderate surfacing in both realistic settings; S3 remains diagnostic noise band. |
| M015 | 2.8 ± 4.086563 | 0.8 ± 1.30384 | 0.0021 ± 0.002881 | **Conservative profile** on the proxy (low pred_nonnegative)—consistent with benchmark-default role; not a “failure” of the line. |
| M025 | 13.4 ± 23.943684 | 0.2 ± 0.447214 | 0.00214 ± 0.004785 | **Setting-sensitive** under the proxy: high S1 band vs near-zero S2; not universally “better for downstream.” |
| M026 | 12.8 ± 9.833616 | 1.0 ± 1.224745 | 0.00526 ± 0.005202 | Diagnostic weighted-CE branch; modest surfacing in S1/S2 vs M003/M025. |

**Tier-2 (complete, job 3716065) — final downstream-transfer answers:**

1. **HR macro-F1 is not a reliable proxy** for gold-lite **audit yield** (pred_nonnegative on documented settings).
2. **Setting effects are family-specific** and can be **large**: e.g. **M025** mean pred_nonnegative **13.4 ± 23.943684 (S1)** vs **0.2 ± 0.447214 (S2)** across seeds; **M003** pred_nonnegative stays in a **similar band** between **S1 and S2** (see Tier-2 table).
3. **Benchmark-balanced** defaults (policy composite) and **audit-surfacing** defaults **diverge** — **M015** vs **M025/M003** on the proxy; **operating profiles are required** — no single checkpoint for every objective.
4. **Oracle-like** metrics remain **near-zero** at aggregate level — **diagnostic only**, **not** dominance or ranking signals.

Evidence: **`knowledge_grounded_evidence_audit/reports/transfer_tier2_job_3716065_record.md`**, **`knowledge_grounded_evidence_audit/reports/tables/tier2_multiseed_results.csv`**.

**Outputs:** `knowledge_grounded_evidence_audit/reports/tables/final_project_joined_model_table.csv` (evidence stack), `knowledge_grounded_evidence_audit/data/processed/final_downstream_transfer_selection_rule.json` (**split** policy: benchmark deployment vs audit surfacing vs diagnostic), `knowledge_grounded_evidence_audit/reports/external_baseline_downstream_bridge.md` (how benchmark-first selection interacts with downstream findings).

**Oncology-facing contribution (non-clinical):** The project can now state **when benchmark selection fails to align with audit utility** on a documented proxy — **not** a claim of clinical benefit.

---

## 10. Current integrated findings

**External evaluation is no longer pending** for BioRED and BC5CDR. The integrated story is **not** “pick the single best model,” because **no universal scalar winner exists** across BioRED, BC5CDR, pairing slices, and stability. The project **resolves** that ambiguity through an **explicit selection policy** in `report/decision_analysis/` (`final_model_selection_rule.json`, weight profiles, penalties). The **primary contribution** at this checkpoint is therefore a **defensible decision framework** and **evidence mapping** — not a claim of one dominant checkpoint on every axis.

**Downstream transfer (added):** For **KB-audit-style** use of model proposals, the project **does not** endorse a **single** checkpoint for every objective. **Benchmark-balanced deployment** may still favor **M015** on external composites; **candidate surfacing** favors **M025/M003** on the gold-lite proxy; **conservative support finding** aligns with **M015**’s **near-zero** pred_nonnegative on that proxy—a **conservative operating profile** under the documented formulation, not an indictment of benchmark performance. These are **profile choices**, not contradictions — see `final_downstream_transfer_selection_rule.json`.

### Decision-oriented summary (for Discussion / Conclusion drafting)

| Objective | Preferred model / profile | Basis | Main caveat |
|-----------|---------------------------|-------|-------------|
| **Benchmark-balanced deployment** | **M015** (default under `benchmark_generalization_heavy`) | External BioRED+BC5CDR composites + policy weights in `final_model_selection_rule.json` | Not guaranteed best on every slice; **not** identical to downstream audit yield |
| **BioRED-like deployment** | **S001 / S002** (conditional); policy compares **naive BioRED-first** vs weighted default (`decision_policy_baseline_comparison.csv`) | High BioRED mean macro F1 cluster | Weighted-CE **branch** — explicit risk flags in policy |
| **Variant / pairing emphasis** | **M021** (secondary; variant–disease slice strength) | `pairing_analysis_table.csv`, role assignment | Not the global composite winner |
| **Candidate surfacing (gold-lite audit proxy)** | **M025 / M003** | Tier-1 + Tier-2 pred_nonnegative bands; **M003** more stable across **S1 vs S2** than **M025** in Tier-2 aggregate | **Heuristic** gold-lite; **not** clinical validation; **M025** is **setting-sensitive** and can **collapse** under S2 |
| **Conservative support-finding (same proxy)** | **M015** | Very low pred_nonnegative on R1/C1-style proxy; **conservative profile** that minimizes false surfacing under stated formulation | May miss candidates; **proxy-only** |
| **Diagnostic / unstable families** | **M005**, **M026** | Controls; weighted CE / aux — policy marks **diagnostic_only** | Do **not** promote to defaults without scope |

**Evidence-aligned model story (high level)**

| Layer | Finding |
|-------|---------|
| **Internal HR** | **M015** (BioLinkBERT pipeline) led **mean macro F1** on the dev protocol; **M003** strong **PubMedBERT** pipeline; **S001/S002** high on **weighted-CE** branch with stability caveats; **M005** **negative transfer** control. |
| **External BioRED** | Top **mean macro F1** cluster: **S001**, **S002**, **M015** (~0.285–0.292) — **not** identical to internal ordering. |
| **External BC5CDR** | **M015** and **M003** lead **chemical–disease** transfer; **M015** highest **mean macro F1** on this source. |
| **Pairing-centric (BioRED test slices)** | **M021** strongest on **variant–disease**; **drug–gene** hardest for **all** models (schema collapse to **`DRUG_GENE_REGULATION`**). |
| **Stability** | **M015**, **M021**, **M003** show **tight** BioRED test seed std; **M005** higher variance (**caution**). |
| **Naive vs explicit selection** | Table-derived policy baselines (`decision_policy_baseline_comparison.csv`) show **BioRED-only** would favor **S001** while the **benchmark-first composite** defaults to **M015** — **policy dependence is empirical**, not cosmetic. |

**Roles (see `model_role_assignment.csv` and §2 of `final_model_selection_report.md`)**

- **Default (benchmark-generalization policy):** **M015** — maximizes weighted composite emphasizing **BioRED + BC5CDR** (`benchmark_generalization_heavy` profile).
- **Secondary:** **M021** — second under the same profile; **best variant–disease** pairing when **precision oncology / variant** emphasis matters.
- **Conditional:** **S001/S002** if **BioRED-like** deployment dominates (weighted-CE **branch risk**); **M003** as **PubMedBERT** pipeline alternative; **M009** optional **shared** line.
- **Diagnostic only:** **M005**, **M026** — controls; not default promotions.

**Schema + training integration:** **`S2_current`** remains **operationally adequate** for coarse benchmark-style deployment; **pairing** and **DrugProt** gaps show where **mechanistic** and **third-corpus** validation are still **thin**.

---

## 11. Limitations and unresolved risks

### Data and resource limitations

- **DrugProt official test** — **not evaluated**: packaged `t1_drugprot.jsonl` has **no `test` split** (`drugprot_unresolved_status.json`). **Formal blocker**, not a pending polish item.
- **Thin slices:** T2 BC5CDR cancer slice is **intentionally small** (high variance); oncology **projection** is heuristic — does not add gold.
- **BRONCO / Precision-O:** Present or strategic but **not** integrated as span-supervised training in this pipeline without further engineering.

### Schema and task limitations

- **`S2_current` is coarse** — CIViC assertion subtypes and mechanistic shades **collapse** into broader heads (see schema gap audit); **drug–gene** pairing remains hard for all models under the operational mapping.
- **Multitask vs pipeline** and **T3 aux** are **not** universally vindicated (e.g. **M005** vs **M015** externally).

### Evaluation limitations

- **Human-coded manual audit** — **deferred**; `manual_audit_table.csv` is **proxy-only**.
- **External evaluation** complete for **BioRED + BC5CDR** protocol; **no** single scalar winner across benchmarks and slices.
- **Downstream transfer:** Gold-lite labels are **heuristic**; **pred_nonnegative** counts are **audit proxies**, not validated clinical assertions. Oracle-style metrics are **weak** numerically — **diagnostic only**, **not** model-ranking evidence.

### Claim limitations (epistemic boundary)

- **No** clinical validity or therapeutic **discovery** claims from KB-gap or surfacing counts.
- **No** claim that one model is **universally best** for all objectives; **policy profiles** are explicit.
- **Policy sensitivity:** BioRED-first vs BC5CDR-first vs pairing-first rules **disagree** on winners (`decision_policy_baseline_comparison.csv`) — selection is **not** policy-free.
- **Engineering variability:** HF hub, CUDA, cluster queues affect reproducibility margins — document versions where needed.

---

## 12. From project closure to writing

This section replaces an experiment roadmap. **Empirical phases described in this dossier are complete** unless a future author explicitly opens a new study.

### Method-writing skeleton (recommended order)

| § block (order) | One-line purpose | Do not repeat here |
|-------------------|------------------|--------------------|
| Task framing + **`S2_current`** | What is extracted, under which coarse heads, and why mapping matters. | Full S1–S4 screening tables (appendix pointer only). |
| Data + **T1–T4** packaging | Gold vs weak vs unlabeled routing, leakage rules, representative scales. | Full shard matrices / long provenance dumps. |
| **HR** trainer + registry | What factors the grid isolates; checkpoint selection rule on dev. | Seed-by-seed logs; treating dev macro-F1 as deployment proof. |
| **External** protocol | BioRED/BC5CDR official pair eval, seeds, exclusions (DrugProt). | Re-deriving policy weights (point to `decision_analysis/`). |
| **Downstream** proxy + **S1/S2/S3** | Gold-lite audit definition, settings, and epistemic limits. | Oracle metrics as a quality leaderboard. |

### Results-writing skeleton (recommended order)

| Results block (order) | Core claim (one line) | Canonical table / file |
|------------------------|----------------------|-------------------------|
| Internal HR anchors | Relative strength and contrasts on the **dev** protocol—inputs to later layers, not standalone deployment claims. | `fine_tuning_experiments/reports/tables/rerun_main_aggregated_results.csv` |
| External benchmarks | No universal winner; policy resolves tradeoffs on BioRED/BC5CDR. | `external_evaluation/reports/tables/primary_external_results.csv` (+ slice tables in same tree) |
| Downstream Tier-1 | HR F1 does not rank audit yield; decoupling is systematic. | `knowledge_grounded_evidence_audit/manifests/tier1_model_selection.csv` + Tier-1 findings manifests |
| Downstream Tier-2 | Setting × family interaction; conservative vs surfacing profiles. | `knowledge_grounded_evidence_audit/reports/tables/tier2_multiseed_results.csv` |
| Integrated decision | Split objectives (benchmark vs audit) per frozen JSON rules. | `report/decision_analysis/final_model_selection_rule.json`, `knowledge_grounded_evidence_audit/data/processed/final_downstream_transfer_selection_rule.json` |

*Detail routing: `report/writing_support/chapter_writing_map.md`.*

### Immediate writing priorities

1. **Introduction:** Foreground headline RQs (**schema, train, ext, downstream**); state **split policy** (benchmark vs audit) in one clear paragraph.
2. **Methods:** T1–T4 packaging under **`S2_current`**; HR trainer; external protocol; downstream gold-lite **proxy** definition — use `report/writing_support/chapter_writing_map.md`.
3. **Results:** Internal anchors → external tables → downstream Tier-1/Tier-2 — order per `chapter_writing_map.md`.
4. **Discussion:** Decoupling (HR vs audit yield; setting × family interaction); **no universal winner**.
5. **Limitations:** Copy structure from §11; cite **DrugProt** and **human audit** blockers explicitly.

### Optional future work (not required to defend current claims)

- DrugProt **test** packaging when data engineering unblocks; **B1/B5** or literature checkpoints only if agreed (see `executed_vs_unexecuted_baselines.csv`).
- Ensemble / distillation — see `future_enhancement_note.md` — **research-only**.
- BRONCO span reader; Precision-O acquisition — **data/pipeline dependencies**.

### Not required for a dissertation/paper based on this dossier

- New benchmark sweeps or broad ablations **beyond** what is documented here.
- Treating downstream **oracle** metrics as primary evidence of model quality.
- Presenting **M015** as the audit-surfacing default without stating the **profile** distinction.

---

*End of master research report. Manifest: `report/master_report_manifest.json`. Writing support: `report/writing_support/`. Downstream dossier: `knowledge_grounded_evidence_audit/reports/downstream_transfer_final_report.md`.*
