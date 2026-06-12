"""Generate step 06 feasibility report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import (
    EVALUABLE_TRIPLES_CSV,
    FETCH_METADATA_JSON,
    GROUNDING_SUMMARY_CSV,
    INFO_JSON,
    MIN_EVALUABLE_GENE_DISEASE,
    MIN_EVALUABLE_GENE_DRUG,
    PMID_RETRIEVABILITY_CSV,
    REPORT_DIR,
    STRUCTURAL_EVIDENCE_CSV,
)


def _pct(n: float) -> str:
    return f"{100 * n:.1f}%"


def generate_report(probe_result: dict) -> Path:
    info = json.loads(INFO_JSON.read_text(encoding="utf-8")) if INFO_JSON.exists() else {}
    metadata = json.loads(FETCH_METADATA_JSON.read_text(encoding="utf-8")) if FETCH_METADATA_JSON.exists() else {}
    summary = pd.read_csv(GROUNDING_SUMMARY_CSV)
    structural = pd.read_csv(STRUCTURAL_EVIDENCE_CSV)
    evaluable = pd.read_csv(EVALUABLE_TRIPLES_CSV) if EVALUABLE_TRIPLES_CSV.exists() else pd.DataFrame()
    retrievability = pd.read_csv(PMID_RETRIEVABILITY_CSV) if PMID_RETRIEVABILITY_CSV.exists() else pd.DataFrame()

    verdict = probe_result["verdict"]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    data_version = info.get("dataVersion", {})
    version_label = data_version.get("version", "unknown") if isinstance(data_version, dict) else str(data_version)

    struct_lines = "\n".join(
        f"| {row.pair_type} | {int(row.pmids_per_association)} | {int(row.association_count)} |"
        for row in structural.itertuples()
    )
    pair_lines = "\n".join(
        f"| {row.pair_type} | {row.grounding_class} | {int(row.association_count)} |"
        for row in summary.itertuples()
    )

    eval_gd = int((evaluable["pair_type"] == "gene-drug").sum()) if len(evaluable) else 0
    eval_gdis = int((evaluable["pair_type"] == "gene-disease").sum()) if len(evaluable) else 0
    retrievable_n = int(evaluable["abstract_retrievable"].sum()) if len(evaluable) and "abstract_retrievable" in evaluable else 0

    if verdict == "GO":
        verdict_text = (
            f"GO. After full cancer-gene annotation coverage, OncoKB yields enough single-PMID gene–drug "
            f"({eval_gd}) and gene–disease ({eval_gdis}) triples for a parallel ranking evaluation alongside CIViC. "
            f"{retrievable_n} of {probe_result['evaluable_triples_total']} evaluable PMIDs returned a PubMed abstract, "
            "so a PubTator pool could be built later using the same protocol as CIViC."
        )
        closing = "OncoKB can serve as a second parallel knowledge base for both pair types on the single-PMID subset."
    elif verdict == "PARTIAL":
        verdict_text = (
            f"PARTIAL (gene–drug only). Comprehensive annotation coverage produced {eval_gd} evaluable single-PMID "
            f"gene–drug triples, enough for a parallel gene–drug external-validity check. Gene–disease single-PMID "
            f"triples are only {eval_gdis}, below the {MIN_EVALUABLE_GENE_DISEASE} threshold for a meaningful parallel "
            "gene–disease ranking set. OncoKB’s strength is therapeutic gene–drug actionability; gene–disease curation "
            "is secondary and often multi-reference. Recommend OncoKB as a gene–drug-only parallel check; leave "
            "gene–disease evaluation to CIViC."
        )
        closing = "Include OncoKB in the study only for gene–drug parallel validation, not gene–disease."
    else:
        verdict_text = (
            f"NO-GO. Even with full cancer-gene annotation coverage ({probe_result['queries_n']} API calls, "
            f"{probe_result['total_associations']} unique associations), the leakage-free single-PMID subset is too "
            f"small or too dominated by multi-reference records for CIViC-style abstract ranking. "
            f"Evaluable triples: {eval_gd} gene–drug, {eval_gdis} gene–disease. "
            f"Multi-PMID associations were {probe_result['multi_pmid_associations']} "
            f"({_pct(probe_result['multi_pmid_share'])}). The study narrative stays on CIViC alone."
        )
        closing = "This probe is recorded for audit; OncoKB is not included in the main narrative."

    report = f"""# OncoKB parallel knowledge base feasibility (step 06, corrected)

Generated: {generated}

## Purpose

CIViC supports ranking evaluation because each evidence item links to one PubMed abstract. This probe asks whether OncoKB actionable associations can support the same abstract-grounded gene–drug and gene–disease ranking task as a parallel knowledge base.

## Correct API method

OncoKB has no bulk export of all actionables. The API is an annotation service: submit gene and alteration queries, receive per-variant actionability (drugs, evidence level, tumour type, and a PMIDs list). This probe used only the authenticated production instance with batch POST requests to:

- annotate/mutations/byProteinChange
- annotate/copyNumberAlterations
- annotate/structuralVariants

No calls were made to non-existent bulk endpoints (allActionables, allAlterations, or download exports).

Access mode: Bearer token on production (token not stored or printed). Data version: {version_label}. OncoTree: {info.get("oncoTreeVersion", "unknown")}.

## Coverage achieved

| Metric | Value |
| --- | ---: |
| Cancer gene list source | {metadata.get("cancer_gene_source", "unknown")} |
| Rows in cancer gene list | {metadata.get("cancer_gene_total_rows", "unknown")} |
| OncoKB-annotated genes queried | {metadata.get("oncokb_annotated_genes", probe_result.get("gene_meta", {}).get("oncokb_annotated_genes", "unknown"))} |
| Phase-1 queries (umbrella, CNA, fusion, MSI-H/TMB-H) | {probe_result.get("phase1_queries_n", metadata.get("phase1_queries", "unknown"))} |
| Phase-2 queries (specific alterations discovered in responses) | {probe_result.get("phase2_queries_n", metadata.get("phase2_queries", "unknown"))} |
| Total annotation API calls | {probe_result["queries_n"]} |
| Unique association records (deduplicated) | {probe_result["total_associations"]} |

Phase 2 re-queried specific alterations returned inside treatment and diagnostic blocks so coverage is not limited to a fixed umbrella panel alone.

## Grounding test (decisive)

| Pair type | Grounding class | Count |
| --- | --- | ---: |
{pair_lines}

PMID list length distribution:

| Pair type | PMIDs per association | Count |
| --- | ---: | ---: |
{struct_lines}

CIViC uses one evidence item per abstract. OncoKB attaches a PMIDs array to each implication. Across all retrieved records, {_pct(probe_result["single_pmid_share"])} had exactly one PMID, {_pct(probe_result["multi_pmid_share"])} had two or more, and the remainder had none. Multi-reference records reflect OncoKB’s accumulated-literature curation style and are not directly usable as single-abstract ranking targets.

## Evaluable target set

Single-PMID associations were reduced to (gene, drug, PMID) or (gene, disease, PMID) triples. Training corpus PMIDs (BioRED and DrugProt clean lists, n={probe_result["training_pmids_n"]}) were excluded.

| Pair type | Single-PMID associations | Evaluable triples (leakage-free) |
| --- | ---: | ---: |
| gene–drug | {probe_result["single_gene_drug_associations"]} | {eval_gd} |
| gene–disease | {probe_result["single_gene_disease_associations"]} | {eval_gdis} |

PubMed abstract retrievability was checked for evaluable PMIDs: {retrievable_n} of {probe_result["evaluable_triples_total"]} returned a non-empty abstract via PubMed efetch, indicating a future PubTator candidate pool could be built for that subset. No pool was built in this step.

## Comparison to CIViC

CIViC step 00 frozen 1,812 gene–drug and gene–disease abstract-grounded pairs. OncoKB’s evaluable single-PMID slice under full annotation coverage is {probe_result["evaluable_triples_total"]} triples ({eval_gd} gene–drug, {eval_gdis} gene–disease), with a large multi-PMID tail that CIViC-style ranking cannot use directly.

## Verdict

{verdict_text}

{closing}
"""

    path = REPORT_DIR / "report.md"
    path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {path}")
    return path


def write_readme(probe_result: dict) -> Path:
    text = f"""# Step 06 — OncoKB feasibility (corrected annotation probe)

Read-only probe using batch POST on the authenticated production OncoKB annotation API over the full cancer-gene list.

**Verdict:** {probe_result["verdict"]}

**Key counts**
- OncoKB-annotated genes queried: {probe_result.get("gene_meta", {}).get("oncokb_annotated_genes", "n/a")}
- Total annotation API calls: {probe_result["queries_n"]}
- Unique associations: {probe_result["total_associations"]}
- gene–drug single-PMID / evaluable: {probe_result["single_gene_drug_associations"]} / {probe_result["evaluable_gene_drug_triples"]}
- gene–disease single-PMID / evaluable: {probe_result["single_gene_disease_associations"]} / {probe_result["evaluable_gene_disease_triples"]}
- Abstract retrievable (evaluable PMIDs): {probe_result.get("evaluable_abstract_retrievable_n", "n/a")}

Run: `bash -lc 'source ~/.bashrc && conda activate hf-hpc && python project_1/06_oncokb_feasibility/run.py --force-fetch'`
"""
    path = Path(__file__).resolve().parent / "README.md"
    path.write_text(text, encoding="utf-8")
    return path
