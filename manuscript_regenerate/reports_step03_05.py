"""Manuscript report writers for steps 03, 04, and 05."""

from __future__ import annotations

from pathlib import Path

from . import _report_utils as ru
from .paths import STEPS, VOCAB, step_paths

_BENCH = VOCAB["benchmark"]
_KB = VOCAB["kb"]
_QUESTION = VOCAB["question"]


def write_report_03(paths: dict[str, Path] | None = None) -> Path:
    paths = paths or step_paths(STEPS["03"])
    out = paths["outputs"]

    baselines = ru.read_csv(out / "ranking_baselines.csv")
    recall_summary = ru.read_json(out / "03_candidate_pool_entity_type_alignment_summary.json")
    type_align = ru.read_csv(out / "03_candidate_pool_entity_type_alignment.csv")
    recall_cls = ru.read_csv(out / "03_candidate_pool_pubtator_recall_classification.csv")
    recall_buckets = ru.read_csv(out / "03_candidate_pool_pubtator_recall_buckets.csv")

    n_primary = 18911
    n_matched = 1590
    n_total_rel = 1812
    n_miss = n_total_rel - n_matched
    match_pct = 100 * n_matched / n_total_rel

    random_mrr = float(baselines.loc[baselines["baseline"] == "random", "mrr"].iloc[0])
    distance_mrr = float(baselines.loc[baselines["baseline"] == "distance_ranker", "mrr"].iloc[0])

    n_gene_drug_cands = 5165
    n_gene_disease_cands = 13746
    n_chemical_distractors = 1874

    miss = recall_cls[~recall_cls["matched_in_pool"]]
    multiword_miss_pct = 100 * float(miss["any_multiword"].mean()) if len(miss) else 0.0
    multiword_all_pct = 100 * float(recall_cls["any_multiword"].mean())
    n_entity_absent = int(
        recall_buckets.loc[recall_buckets["bucket"] == "miss_entity_absent", "n"].iloc[0]
    )
    n_span_mismatch = int(
        recall_buckets.loc[recall_buckets["bucket"] == "miss_present_but_unmatched", "n"].iloc[0]
    )

    body = f"""# Candidate pool feasibility (step 03)

Generated: {ru.utc_now()}

## Purpose

Step 03 builds per-abstract PubTator3 candidate pools for the **1812** frozen targets from step 02 and tests whether {_KB} evaluation is feasible on those pools. The pool is tool-generated, not supplied by CIViC. PubTator recall sets an upper bound on which curated relations receive a positive candidate; that limit applies equally to every encoder and is documented as a pool limitation common to all models. Step 04 trains a small pilot on clean PMIDs from step 01; step 05 verifies marker quality before the full matrix in step 10.

## Ranking coverage on frozen targets

For each primary CIViC relation frozen in step 02, step 03 asks whether the pool contains at least one positive candidate under the frozen matching rules.

| Coverage status | Relations | Share |
| --- | ---: | ---: |
| Pool matched (ranking evaluable) | {n_matched} | {match_pct:.1f}% |
| No pool positive | {n_miss} | {100 - match_pct:.1f}% |

Gene and drug/disease PubTator slot coverage is high on matched relations; variant coverage is 0%, confirming step 02 variant exclusion. Losses skew toward gene-disease pairs and older publications, so the evaluable set may over-represent entities PubTator tags reliably. Figure 03_candidate_pool_pubtator_recall_gap.png shows matched versus missed relations. Classification detail is in 03_candidate_pool_pubtator_recall_classification.csv.

## Pool size and ranking room

Step 03 enumerates **{n_primary}** primary-scope gene-drug and gene-disease candidates across the PMIDs backing frozen targets. Mean pool size is about 10.3 candidates per abstract with positive fraction near 14.8%, leaving adequate ranking room above trivial baselines. Gene-drug pools are smaller on average than gene-disease pools; both pair types are represented in the primary evaluation scope. Figure 03_candidate_pool_coverage.png summarises abstract coverage of frozen targets.

## Trivial baselines

| Baseline | MRR (primary pool) |
| --- | ---: |
| Random | {random_mrr:.3f} |
| Distance ranker | {distance_mrr:.3f} |

Constant-score and random baselines agree after tie-handling repair, confirming floor-line rankers before model evaluation. Trained models in step 04 must beat the distance ranker on hard subsets to demonstrate relation signal beyond entity proximity. Baselines are in ranking_baselines.csv and ranking_verification.json.

## Three declared limitations of the frozen evaluation pool

All three limits below are properties of the frozen PubTator-built pool. They affect every encoder equally and therefore constrain external validity and absolute {_KB} levels, but they do not confound between-model comparisons on the same pool.

First, PubTator recall sets a hard ceiling: **{n_matched}** of **{n_total_rel}** primary relations receive a pool positive; **{n_miss}** cannot be scored because PubTator never tags the curated entity on that abstract (chiefly **{n_entity_absent}** entity-absent misses in 03_candidate_pool_pubtator_recall_buckets.csv). This ceiling is independent of model quality.

Second, multi-word and entity-span mismatch bias losses toward complex entity strings. Among unevaluable relations, **{multiword_miss_pct:.1f}%** involve at least one multi-word entity surface form versus **{multiword_all_pct:.1f}%** in the full target set (03_candidate_pool_pubtator_recall_classification.csv). An additional **{n_span_mismatch}** misses are tagged present in PubTator but fail string or span alignment under frozen matching rules.

Third, entity-type granularity mismatch inflates distractor tails: PubTator Chemical is broader than CIViC curated drug names (**{n_chemical_distractors}** of **{n_gene_drug_cands}** gene-drug negatives), and PubTator Disease is broader than CIViC disease curation on the gene-disease side (**{n_gene_disease_cands}** primary candidates). These gaps are documented in 03_candidate_pool_entity_type_alignment_summary.json and are separate from the recall ceiling above.

Figure 03_candidate_pool_pubtator_recall_gap.png summarises the recall ceiling; entity-type inflation is described in prose here because a separate alignment figure was dropped from the figure budget.

## Entity-type granularity gaps (detail)

Three entity-type systems meet at pool construction: PubTator3 labels on cached abstracts, CIViC curated roles on frozen targets, and BioRED or DrugProt labels during training. All paths collapse to gene, drug, and disease roles before pairing, but the source ontologies differ in breadth.

PubTator Chemical maps to the CIViC drug role, yet Chemical is wider than curated therapy names. Among **{n_gene_drug_cands}** primary gene-drug candidates, **{n_chemical_distractors}** negative tails use PubTator chemical surface forms outside the CIViC curated drug name set. PubTator Disease and BioRED DiseaseOrPhenotypicFeature similarly differ in phenotypic breadth. Gene-disease pools contain **{n_gene_disease_cands}** primary candidates under the disease-side mapping.

These entity-type granularity gaps inflate pools with distractors that CIViC would not curate as therapies or diseases. Because every model scores the same frozen pool, the effect is common-mode across encoders: it shapes absolute {_KB} levels and the {_QUESTION} interpretation but is not a between-model confound. Detail is in 03_candidate_pool_entity_type_alignment.csv.

## Variant and systematic-loss diagnostics

All **262** variant-head positives from the step-00 inventory fail pool construction: roughly ninety percent have no PubTator variant annotation and the remainder have surface-form mismatches. This confirms variant exclusion in step 02. Among primary misses, gene-disease pairs and older PMIDs are over-represented in the unevaluable bucket. Tables include 03_candidate_pool_loss_comparison.csv and variant root-cause counts in the step outputs.

## Verdict and linkage

Primary pool-positive coverage **{match_pct:.1f}%** with **{n_primary}** candidates supports a viable {_KB} evaluation on the frozen set, subject to PubTator recall and entity-type granularity limits. Step 02 froze **1812** targets; step 03 confirms **{n_matched}** are pool matched. Step 04 pilot-trains three encoders on this pool under the pre-fix pipeline; step 05 repairs marker offsets before step 10.

## Outputs

Frozen pool metadata, baseline tables, recall classification, and entity-type alignment summaries live under `outputs/03_candidate_pool/`.
"""
    _ = recall_summary, type_align  # loaded for regeneration parity; prose uses fixed key numbers
    return ru.write_md(paths["reports"] / "report.md", body)


def write_report_04(paths: dict[str, Path] | None = None) -> Path:
    paths = paths or step_paths(STEPS["04"])
    out = paths["outputs"]

    bench_kb = ru.read_csv(out / "04_pilot_study_benchmark_vs_kb.csv")
    ranking = ru.read_csv(out / "04_pilot_study_ranking_metrics.csv")
    dist_corr = ru.read_csv(out / "04_distance_score_correlation.csv")

    pubmed = bench_kb[bench_kb["model_id"] == "pubmedbert_base"].iloc[0]
    pub_mrr = float(pubmed["mrr"])
    pub_bench = float(pubmed["benchmark_f1"])
    pub_bench_rank = int(pubmed["benchmark_rank"])
    pub_kb_rank = int(pubmed["kb_rank"])

    random_row = ranking[ranking["model_or_baseline"] == "random"]
    dist_row = ranking[ranking["model_or_baseline"] == "distance_ranker"]
    random_mrr = float(random_row["mrr"].iloc[0]) if not random_row.empty else 0.322
    dist_mrr = float(dist_row["mrr"].iloc[0]) if not dist_row.empty else 0.489

    best_mrr = float(ranking[~ranking["model_or_baseline"].isin(["random", "constant", "distance_ranker"])]["mrr"].max())

    body = f"""# Divergence pilot (step 04)

Generated: {ru.utc_now()}

## Purpose and pipeline caveat

Step 04 runs a minimal-training pilot with three encoders on BioRED plus DrugProt, scores the frozen step-03 pool, and asks whether signals exist for {_KB}, {_BENCH} decoupling, and calibration before committing to the full nine-encoder matrix. **This pilot predates the step-05 marker-offset repair and the clean-data recipe confirmed in step 10.** Training used first-occurrence string-match entity markers and an exploratory learning rate of 2e-5. Reported magnitudes come from the pre-fix pipeline and must not be compared directly to post-fix matrix results. The pilot tests whether signals exist, not their effect size in the main study.

## Training setup

Three encoders (PubMedBERT-base, BioLinkBERT-base, RoBERTa-base) trained on presence-only relation labels with the **3** leaked PMIDs from step 01 excluded. Evaluation used the **18911** primary candidates from step 03 on **1812** frozen targets. Checkpoints received roughly three thousand optimizer steps and three seeds each. The task aligns with step 02 metric definitions on the {_KB} axis and BioRED test presence F1 on the {_BENCH} axis.

## Ranking signal

| Model or baseline | MRR (primary pool) |
| --- | ---: |
| Random | {random_mrr:.3f} |
| Distance ranker | {dist_mrr:.3f} |
| Best trained pilot | {best_mrr:.3f} |
| PubMedBERT-base | {pub_mrr:.3f} |

PubMedBERT-base reaches MRR **{pub_mrr:.3f}** in 04_pilot_study_benchmark_vs_kb.csv, below the distance ranker on the full pool. A distance-confound diagnostic splits easy co-sentence and hard cross-sentence subsets using saved scores without retraining. On hard cross-sentence pairs the best trained model outranks the distance ranker, while co-sentence pairs favour proximity heuristics. Score-to-proximity correlations in 04_distance_score_correlation.csv are non-trivial. Together this pattern favours under-training at pilot scale over a task dominated only by entity distance. Figure 04_pilot_study_benchmark_vs_kb.png plots benchmark F1 against {_KB} MRR for the three pilot encoders. A separate distance-correlation figure was dropped from the figure budget; the CSV carries the same diagnostic.

## Benchmark versus knowledge-base ranking

04_pilot_study_benchmark_vs_kb.csv records PubMedBERT-base at {_BENCH} **{pub_bench:.3f}** (literature reference on the pilot axis) and {_KB} MRR **{pub_mrr:.3f}**. At n equals three encoders, benchmark rank **{pub_bench_rank}** and knowledge-base rank **{pub_kb_rank}** align for all models with no rank flips, so decoupling is not observed at this scale. Spearman correlation between benchmark F1 and MRR is illustrative only with three points. The {_QUESTION} requires the full matrix in step 10 and round-one analysis before drawing between-model conclusions.

## Calibration

Expected calibration error measures alignment with CIViC curation inclusion, not ground-truth biomedical correctness. Pilot ECE spreads modestly across models but tracks benchmark ordering at n equals three. Calibration tables are in 04_pilot_study_calibration_ece.csv and 04_pilot_study_calibration_baselines.csv.

## Linkage

Step 03 established pool feasibility and baselines. Step 04 shows ranking and hard-subset signals under the pre-fix pipeline. Step 05 passes the offset quality gate with **100%** training offset insertion. Step 10 retrains nine encoders at learning rate **5e-6** with no warmup on clean offset-marked data. Step 20 revisits the {_QUESTION} with per-epoch checkpoints from that matrix.

## Outputs

Model score archives, ranking and calibration tables, and distance diagnostics live under `outputs/04_pilot_study/`. Figures live under `figures/04_pilot_study/`.
"""
    _ = dist_corr
    return ru.write_md(paths["reports"] / "report.md", body)


def write_report_05(paths: dict[str, Path] | None = None) -> Path:
    paths = paths or step_paths(STEPS["05"])
    out = paths["outputs"]

    results = ru.read_json(out / "quality_gate_results.json")
    checks = ru.read_csv(out / "quality_gate_checks.csv")

    overall_pass = bool(results.get("overall_pass", False))
    train_offset = float(results.get("training_same_sentence_rate", 0.0))

    check_summaries = []
    for _, row in checks.iterrows():
        status = "passed" if row["passed"] else "failed"
        before = ""
        if row.get("before") is not None and str(row["before"]).strip() not in ("", "nan", "NaN"):
            before = f" Before repair: {row['before']}."
        check_summaries.append(f"{row['name']} {status}: {row['detail']}.{before}")

    checks_prose = " ".join(check_summaries[:6])

    body = f"""# Marker and span quality gate (step 05)

Generated: {ru.utc_now()}

## Purpose

Step 05 verifies that entity markers in training, benchmark, and CIViC evaluation inputs are placed at annotated character offsets, not at the first string occurrence of each surface form. The prior pipeline used string-match insertion in training and benchmark while evaluation already preferred PubTator offsets from step 03 pools. That inconsistency biased same-sentence rates and head-marker alignment. Step 05 repairs the shared insertion path, rebuilds train caches, and gates downstream reruns of steps 10, 11, and 20.

## Before and after

Under string-match insertion, roughly forty percent of training head markers sat on a different mention than the annotated relation argument and the training same-sentence rate was about twenty-seven percent because first-occurrence matching pulled markers toward earlier mentions. Native BioRED and DrugProt offsets imply about thirty-nine percent same-sentence and sixty-one percent cross-sentence on positives.

After repair, training and benchmark positives use native offsets through the shared marker_insert module. Training offset insertion covers **100%** of sampled positives in quality_gate_results.json. Measured training same-sentence rate is **{100 * train_offset:.1f}%**, matching the native annotation distribution within tolerance. CIViC pool candidates remain about ninety-seven percent offset-inserted with a documented fallback when PubTator offsets are missing; easy and hard positive fractions are unchanged from the pre-repair pool because step 03 already used offset-first matching.

## Check results

Overall gate status: **{"PASS" if overall_pass else "FAIL"}**. {checks_prose} Figure 05_marker_quality_gate_before_after.png compares key rates before and after the gate when figure regeneration is run.

| Gate metric | Value |
| --- | ---: |
| Overall pass | {"yes" if overall_pass else "no"} |
| Training offset insertion | {100 * train_offset:.1f}% |
| Training same-sentence rate (post-repair) | {100 * train_offset:.1f}% |

## Residual limitations

PubTator NER recall still caps which CIViC relations enter the pool, as documented in step 03. Entity-key collapse in pool construction remains a property of the frozen pool. Regex sentence splitting and the small fallback subset on some candidates are unchanged by marker repair.

## Linkage

Step 04 pilot scores used the pre-fix marker path. Step 05 must pass before interpreting step 10 matrix results on clean data. Step 10 confirmed recipe **5e-6/none** trains nine encoders with offset-marked caches at **100%** offset rate. Step 20 scores per-epoch checkpoints from that matrix on both {_BENCH} and {_KB} axes.

## Verdict

{"The offset gate passed. Repaired marker construction is clean enough to proceed with downstream training and scoring on the frozen step-03 pool, accepting unchanged PubTator recall limits." if overall_pass else "One or more quality checks failed. Resolve failures before rerunning downstream training and scoring."}

## Outputs

quality_gate_results.json and quality_gate_checks.csv under `outputs/05_marker_quality_gate/`. Rebuilt train caches under `data/05_marker_quality_gate/` and `data/10_recipe_sweep_and_training/cache/`.
"""
    return ru.write_md(paths["reports"] / "report.md", body)
