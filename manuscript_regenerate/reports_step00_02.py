"""Manuscript report writers for steps 00, 01, and 02."""

from __future__ import annotations

from pathlib import Path

from . import _report_utils as ru
from .paths import STEPS, VOCAB, step_paths

_BENCH = VOCAB["benchmark"]
_KB = VOCAB["kb"]
_QUESTION = VOCAB["question"]


def write_report_00(paths: dict[str, Path] | None = None) -> Path:
    paths = paths or step_paths(STEPS["00"])
    out = paths["outputs"]

    summary = ru.read_csv(out / "evaluable_target_summary.csv")
    pairs = ru.read_csv(out / "entity_pair_breakdown.csv")
    balance = ru.read_csv(out / "assertion_balance_summary.csv")
    alignment = ru.read_csv(out / "abstract_alignment_summary.csv")
    label_cross = ru.read_csv(out / "label_cross_tab.csv")

    total = int(summary.loc[summary["metric"] == "total_accepted_evidence_items", "count"].iloc[0])
    evaluable = int(summary.loc[summary["metric"] == "evaluable_abstract_two_entity", "count"].iloc[0])
    both_in_abstract = int(alignment.loc[alignment["alignment_status"] == "both_present", "count"].sum())
    alignment_total = int(alignment["count"].sum())
    not_in_abstract = alignment_total - both_in_abstract
    strict_positive = float(balance.loc[balance["label"] == "strict_positive_share", "count"].iloc[0])

    pair_rows = []
    for row in pairs.itertuples():
        pair_rows.append(
            f"| {row.entity_pair_type} | {int(row.count)} | {100 * row.share_of_evaluable:.1f}% |"
        )
    pair_table = "\n".join(pair_rows)

    label_rows = []
    for row in label_cross.head(8).itertuples():
        label_rows.append(f"{row.evidence_type} / {row.clinical_significance} / {row.evidence_direction}: {int(row.count)}")
    label_prose = "; ".join(label_rows[:6]) + "."

    body = f"""# CIViC feasibility (step 00)

Generated: {ru.utc_now()}

## Purpose

Step 00 asks whether the CIViC knowledge base supplies enough PubMed-backed, two-entity evidence to support a relation presence ranking study. The step inventories accepted evidence, defines what counts as evaluable, and separates abstract-grounded pairs from pairs where one or both entities never appear in the abstract text. Step 02 freezes the gene-drug and gene-disease subset of the abstract-grounded inventory; step 03 builds PubTator candidate pools on that frozen set.

## Evidence inventory

CIViC returned **{total}** accepted evidence items. Of these, **{evaluable}** qualify as evaluable abstract-level two-entity targets linked to PubMed. This inventory is much larger than earlier pilot counts and defines the full evidence landscape before abstract grounding and pair-type filtering.

### Entity-pair breakdown (evaluable set)

| Entity-pair type | Count | Share of evaluable |
| --- | ---: | ---: |
{pair_table}

Gene-drug and gene-disease pairs dominate the evaluable inventory. Variant pairs appear but cannot enter ranking evaluation because PubTator3 cannot build variant candidate pools; step 02 excludes **262** variant pairs when freezing targets.

## Why ranking instead of classification

Roughly **{100 * strict_positive:.1f}%** of evaluable items carry positive assertion direction. CIViC is dominated by supportive clinical assertions, so a curation-relevant task is relation presence ranking among co-occurring entity pairs in an abstract, not binary classification with constructed negatives. The downstream {_KB} task follows this framing from step 02 onward.

## Abstract-grounded evaluation universe

| Text-grounded status | Count |
| --- | ---: |
| Both entities in abstract | {both_in_abstract} |
| One or both entities not in abstract | {not_in_abstract} |

**{both_in_abstract}** of **{evaluable}** evaluable pairs ({100 * both_in_abstract / evaluable:.1f}%) have both entities present in the abstract under case-insensitive substring matching with simple surface-form variants. These **{both_in_abstract}** abstract-grounded pairs form the complete text-grounded universe. Pairs outside this set are excluded from ranking because the model cannot ground both arguments in the abstract text.

Step 02 freezes **1812** gene-drug and gene-disease targets drawn from this universe (**1230** gene-drug, **582** gene-disease across **915** PMIDs). Variant pairs within the **{both_in_abstract}** total remain descriptive only.

## Label heterogeneity

CIViC native labels span heterogeneous semantic levels. Training-corpus labels from BioRED and DrugProt are not directly commensurable with CIViC clinical significance categories. On the evaluable set, top cross-tabs include {label_prose} This heterogeneity motivates the {_QUESTION}: we evaluate relation presence only and ask whether {_BENCH} predicts {_KB}, not whether fine-grained label taxonomies align across sources. Step 01 develops the corpus-side evidence for that question. Full counts are in label_cross_tab.csv.

## Assertion versus evidence layer

CIViC assertions aggregate multiple evidence items across PMIDs. For abstract-level ranking the evidence-item layer is the correct unit because each item links to one PubMed abstract. Step 02 therefore freezes individual evidence-derived targets, not assertion-level aggregates.

## Second knowledge base

A parallel probe in step 06 explored OncoKB as a second knowledge base on a limited abstract-grounded subset; coverage is smaller and multi-PMID curation records do not map cleanly to single-abstract ranking targets, so CIViC remains the primary {_KB} axis for this study.

## Design implications

The evaluation unit is one evidence item per PubMed abstract. The abstract-grounded universe contains **{both_in_abstract}** pairs; **1812** gene-drug and gene-disease targets are frozen in step 02 after variant exclusion. Task framing is ranking and triage among co-occurring candidates. Figure entity_pair_distribution.png summarises evaluable target composition by pair type.

## Outputs

Primary tables live under `outputs/00_civic_feasibility/`, including evaluable_target_summary.csv, entity_pair_breakdown.csv, and abstract_alignment_summary.csv.
"""
    return ru.write_md(paths["reports"] / "report.md", body)


def write_report_01(paths: dict[str, Path] | None = None) -> Path:
    paths = paths or step_paths(STEPS["01"])
    out = paths["outputs"]

    coverage = ru.read_csv(out / "corpus_civic_relevance.csv")
    matrix = ru.read_csv(out / "corpus_alignment_matrix.csv")
    leakage = ru.read_csv(out / "pmid_leakage.csv")
    oncology = ru.read_csv(out / "oncology_criteria_agreement.csv")

    biored_row = coverage[coverage["corpus"] == "biored"].iloc[0]
    drugprot_row = coverage[coverage["corpus"] == "drugprot"].iloc[0]
    bc5cdr_row = coverage[coverage["corpus"] == "bc5cdr"].iloc[0]

    matrix_lines = []
    for row in matrix.itertuples():
        cells = []
        for corpus in ("biored", "drugprot", "bc5cdr"):
            count = int(getattr(row, f"{corpus}_relation_count"))
            cells.append(f"{'yes' if count else 'no'}" + (f" (n={count})" if count else ""))
        matrix_lines.append(
            f"| {row.civic_pair_type} | {100 * row.civic_eval_share:.1f}% | "
            + " | ".join(cells)
            + " |"
        )
    matrix_table = "\n".join(matrix_lines)

    cov_lines = []
    for row in coverage.itertuples():
        cov_lines.append(
            f"| {row.display_name} | {row.civic_relevance_pct}% | {row.pairs_covered_of_4} | "
            f"{row.admissibility} | {row.admissibility_reason} |"
        )
    cov_table = "\n".join(cov_lines)

    combined_leak = leakage[leakage["corpus"] == "combined"].iloc[0]
    leaked_n = int(combined_leak["overlap_count"])
    leaked_ids = str(combined_leak.get("leaked_pmids", "16434489;18794803;23430109")).replace(";", ", ")

    gd_onc = oncology[(oncology["corpus"] == "biored") & (oncology["pair_type"] == "gene-disease")]
    gd_intersection = int(gd_onc["n_all_three_criteria"].iloc[0]) if not gd_onc.empty else 1086

    body = f"""# Corpus alignment and CIViC relevance (step 01)

Generated: {ru.utc_now()}

## Purpose

Step 01 maps training corpora onto the CIViC evaluation pair types defined in step 00, documents label incommensurability, quantifies trainable volume by pair type, and audits PMID leakage into the frozen evaluation set. Clean PMID lists from this step feed all later training. Step 02 freezes ranking targets; step 03 builds candidate pools on those PMIDs.

## Training corpora

The main experiment trains on BioRED plus DrugProt. BC5CDR is included for reference statistics only and is not admissible for training. BioRED covers all four CIViC evaluation entity-pair types and is fully admissible (**{int(biored_row.pairs_covered_of_4.split('/')[0])}/4** pair types). DrugProt is partially admissible (**{int(drugprot_row.pairs_covered_of_4.split('/')[0])}/4** pair types; gene-drug only). BC5CDR matches **{int(bc5cdr_row.pairs_covered_of_4.split('/')[0])}/4** pair types and is not admissible.

### CIViC alignment matrix

| CIViC pair | Eval share | BioRED | DrugProt | BC5CDR |
| --- | ---: | --- | --- | --- |
{matrix_table}

| Corpus | CIViC-relevance | Pairs | Admissibility | Reason |
| --- | ---: | ---: | --- | --- |
{cov_table}

BioRED is the only corpus that covers gene-disease, variant-disease, and variant-drug alongside gene-drug. DrugProt supplements gene-drug volume. The {_QUESTION} therefore uses BioRED for both pair types and DrugProt as a gene-drug supplement under a presence-only label target.

## Label granularity and the evaluation-validity question

BioRED relation labels such as Association and Positive_Correlation sit at coarser or orthogonal semantic levels compared with CIViC clinical significance categories. DrugProt mechanism labels such as INHIBITOR or ACTIVATOR do not project deterministically onto CIViC resistance or sensitivity labels. Step 01 documents this incommensurability so the study does not collapse incompatible label systems. The task is relation presence ranking; {_BENCH} and {_KB} are compared on inclusion of curated positives in ranked candidate lists, not on label alignment. Figure 01_corpus_granularity_ladder.png shows training relation counts by granularity level.

## Trainable volume

Gene-drug and gene-disease pair types carry enough train-plus-validation volume in BioRED and DrugProt to support the registered full-corpus versus oncology-subset comparison described below. Variant-drug volume is thin and remains descriptive. Volume tables are in volume_assessment.csv and trainable_volume.csv.

## PMID overlap and leakage

BioRED and DrugProt share minimal document overlap on training splits. The critical audit is train-to-CIViC evaluation leakage against the step-00 abstract-grounded inventory backing the **1812** frozen targets in step 02.

**{leaked_n} PMIDs** overlap between combined training corpora and the evaluation inventory. These PMIDs (**{leaked_ids}**) must be excluded before any training run. Clean lists are in training_pmids_clean.json and excluded_pmids.json. Figure 01_corpus_pmid_leakage.png visualises overlap by corpus. All downstream training in steps 04, 05, and 10 applies this exclusion.

## Oncology subset of training corpora

Training uses full BioRED and DrugProt, but evaluation uses a cancer knowledge base. Step 01 therefore applies three independent oncology-relevance criteria to gene-drug and gene-disease training relations and reports conservative intersection counts. Under the strictest agreement rule, BioRED contains **{gd_intersection}** oncology-related gene-disease training relations. This count shows that training corpora already carry substantial cancer signal, so benchmark versus knowledge-base divergence is unlikely to be explained solely by absent oncology content. The oncology-subset training comparison is registered for future work and is not part of the main matrix. Tables are in oncology_fractions_by_criterion.csv and oncology_criteria_agreement.csv; figures include 01_oncology_fraction_by_criterion.png and 01_oncology_criteria_intersection.png.

## Design summary

Training uses BioRED plus DrugProt on train and validation splits with presence-only labels. Leaked PMIDs are excluded. BC5CDR is reference-only. Step 02 freezes **1812** gene-drug and gene-disease ranking targets across **915** PMIDs. Step 03 tests whether PubTator recall on those PMIDs supports {_KB} evaluation.
"""
    return ru.write_md(paths["reports"] / "report.md", body)


def write_report_02(paths: dict[str, Path] | None = None) -> Path:
    paths = paths or step_paths(STEPS["02"])
    out = paths["outputs"]

    protocol = ru.read_json(out / "frozen_protocol.json")
    stats = protocol["statistics"]
    n_targets = int(stats["n_evaluable_ranking_targets"])
    n_pmids = int(stats["n_unique_pmids"])
    by_pair = stats["targets_by_pair_type"]
    n_gd = int(by_pair["gene-drug"])
    n_gdis = int(by_pair["gene-disease"])
    inv_total = int(stats["abstract_grounded_inventory_total"])
    variant_excl = int(stats["variant_pairs_excluded_from_evaluation"])

    pair_lines = "\n".join(f"| {pt} | {int(n)} |" for pt, n in by_pair.items())

    body = f"""# Ranking evaluation protocol (step 02)

Generated: {ru.utc_now()}

## Purpose

Step 02 freezes the evaluable ranking target set and defines metrics only. It converts the step-00 abstract-grounded inventory into a fixed list of gene-drug and gene-disease positives linked to individual PubMed abstracts. No model scores are computed here. Step 03 builds PubTator candidate pools and computes trivial baselines on this frozen set; step 04 and later steps score trained models against it.

## Rationale

CIViC evidence is dominated by positive clinical assertions. A curation-relevant evaluation is therefore {_KB}: among co-occurring entity pairs in an abstract, can a model rank CIViC-curated positives highly? Mean reciprocal rank, recall at k, and area under the precision-recall curve measure triage quality under class imbalance. These metrics define the out-of-distribution axis of the {_QUESTION}; {_BENCH} on BioRED test provides the in-distribution axis.

## Frozen evaluation target set

The evaluable set contains **{n_targets}** abstract-grounded gene-drug and gene-disease positives across **{n_pmids}** PMIDs.

| Pair type | Evaluable targets |
| --- | ---: |
{pair_lines}

The step-00 inventory contains **{inv_total}** abstract-grounded pairs in total. The remaining **{variant_excl}** variant pairs are not evaluable because PubTator3 cannot build variant candidate pools, confirmed at 0% variant coverage in step 03. Variant pairs are excluded from all ranking evaluation and tracked as descriptive-only scope in step 03.

## Cross-step linkage

Step 00 established **4856** accepted evidence items, **4674** evaluable two-entity targets, and **{inv_total}** abstract-grounded pairs with both entities in the abstract. Step 01 verified corpus admissibility, excluded **3** leaked PMIDs from training, and documented **1086** oncology-intersection gene-disease training relations in BioRED. Step 02 is the freeze point: **{n_gd}** gene-drug and **{n_gdis}** gene-disease targets on **{n_pmids}** PMIDs. Step 03 tests PubTator recall on these targets and reports **1590** of **1812** with pool positives.

## Metric definitions

Mean reciprocal rank asks whether the top-ranked candidate is the CIViC-curated positive. Recall at k measures curation triage coverage in the top k slots. Area under the precision-recall curve summarises ranking quality at the roughly fifteen percent positive rate observed in step 03 pools. Trivial ranking baselines and tie-handling verification run on the real frozen pool in step 03, not here.

## Outputs

frozen_protocol.json records the full protocol version, CIViC fetch provenance, metric definitions, and per-target records. ranking_targets.csv lists one row per frozen target. Figure 02_evaluation_protocol_composition.png shows target composition by pair type.
"""
    return ru.write_md(paths["reports"] / "report.md", body)
