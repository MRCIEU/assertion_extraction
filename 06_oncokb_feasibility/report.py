"""Generate step 06 feasibility report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import (
    ACCESS_PROBE_JSON,
    ASSOCIATIONS_CSV,
    DATA_DIR,
    EVALUABLE_TRIPLES_CSV,
    GROUNDING_SUMMARY_CSV,
    INFO_JSON,
    OUTPUT_DIR,
    REPORT_DIR,
    STRUCTURAL_EVIDENCE_CSV,
)


def _pct(n: float) -> str:
    return f"{100 * n:.1f}%"


def generate_report(probe_result: dict) -> Path:
    info = json.loads(INFO_JSON.read_text(encoding="utf-8")) if INFO_JSON.exists() else {}
    access = json.loads(ACCESS_PROBE_JSON.read_text(encoding="utf-8")) if ACCESS_PROBE_JSON.exists() else {}
    summary = pd.read_csv(GROUNDING_SUMMARY_CSV)
    structural = pd.read_csv(STRUCTURAL_EVIDENCE_CSV)
    associations = pd.read_csv(ASSOCIATIONS_CSV)
    evaluable = pd.read_csv(EVALUABLE_TRIPLES_CSV) if EVALUABLE_TRIPLES_CSV.exists() else pd.DataFrame()

    verdict = probe_result["verdict"]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    endpoint_lines = "\n".join(
        f"| {row['endpoint']} | {int(row['status_code'])} |"
        for row in access.get("endpoints", [])
    )

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

    if verdict == "GO":
        verdict_text = (
            f"GO. OncoKB exposes a usable single-PMID subset for parallel ranking evaluation. "
            f"After excluding training PMIDs, {probe_result['evaluable_triples_total']} triples remain "
            f"({eval_gd} gene–drug, {eval_gdis} gene–disease). "
            f"A parallel run would reuse the same PubTator candidate pool and ranking protocol as CIViC, "
            f"using the evaluable triple list from this probe as targets."
        )
        closing = (
            "OncoKB may serve as a second evaluation knowledge base alongside CIViC for the "
            "abstract-grounded single-PMID subset identified here."
        )
    else:
        verdict_text = (
            "NO-GO. OncoKB therapeutic and diagnostic records returned by the annotation API carry "
            "PubMed identifiers as reference lists, not as one primary abstract per association. "
            f"Only {probe_result['single_pmid_associations']} of {probe_result['total_associations']} "
            f"associations ({_pct(probe_result['single_pmid_share'])}) had exactly one PMID; "
            f"{probe_result['multi_pmid_associations']} ({_pct(probe_result['multi_pmid_share'])}) had two or more. "
            "That structure does not match the CIViC evidence-item model used in this study. "
            "OncoKB is therefore not included in the main narrative; the study remains centered on CIViC."
        )
        closing = (
            "This probe is recorded for audit purposes. The main study narrative stays on CIViC alone."
        )

    report = f"""# OncoKB parallel knowledge base feasibility (step 06)

Generated: {generated}

## Purpose

CIViC supports ranking evaluation because each accepted evidence item links to one PubMed abstract, which anchors a PubTator candidate pool and a gene–drug or gene–disease target. This probe asks whether OncoKB actionable associations can be used the same way.

## API access

Access mode: Bearer token from the user environment (token value not stored or printed).

| Endpoint | HTTP status |
| --- | ---: |
{endpoint_lines}

Public metadata and curated gene lists were readable. Bulk association exports and alteration inventory endpoints returned 403 Forbidden with this token tier. Actionable therapeutic and diagnostic content was retrieved through variant annotation endpoints only.

Data version: {info.get("dataVersion", {}).get("version", "unknown") if isinstance(info.get("dataVersion"), dict) else info.get("dataVersion", "unknown")}. OncoTree version: {info.get("oncoTreeVersion", "unknown")}.

## What was retrieved

The probe queried {probe_result["therapeutic_genes_n"]} genes with OncoKB therapeutic levels, using a fixed panel of umbrella alterations, copy-number queries, selected fusion queries, and MSI-H/TMB-H checks ({probe_result["queries_n"]} annotation calls total). Responses were deduplicated into {probe_result["total_associations"]} unique association records.

Each record includes a pair type (gene–drug from treatments, gene–disease from diagnostic or prognostic implications), optional disease context, evidence level, and a `pmids` list when present. The API does not expose a single primary PMID field per association.

## Grounding test (decisive)

| Pair type | Grounding class | Count |
| --- | --- | ---: |
{pair_lines}

Distribution of PMID list lengths:

| Pair type | PMIDs per association | Count |
| --- | ---: | ---: |
{struct_lines}

Interpretation: CIViC ties one evidence item to one abstract. OncoKB ties one implication to a list of PMIDs in the `pmids` array. {_pct(probe_result["multi_pmid_share"])} of retrieved associations had two or more PMIDs; {_pct(probe_result["single_pmid_share"])} had exactly one; the remainder had none.

Single-PMID gene–drug associations: {probe_result["groundable_gene_drug_triples"]}. Single-PMID gene–disease associations: {probe_result["groundable_gene_disease_triples"]}.

## Training PMID overlap

Training corpus PMIDs (BioRED and DrugProt clean lists, n={probe_result["training_pmids_n"]}) were checked against single-PMID triples. Evaluable triples after excluding any training PMID: {probe_result["evaluable_triples_total"]} ({eval_gd} gene–drug, {eval_gdis} gene–disease).

## Pair-type composition of the groundable subset

Of single-PMID associations, gene–drug records are {probe_result["groundable_gene_drug_triples"]} and gene–disease records are {probe_result["groundable_gene_disease_triples"]}. CIViC step 00 found 1,812 gene–drug and gene–disease abstract-grounded pairs in its evaluable freeze; OncoKB’s retrievable single-PMID slice is smaller and mixed with a larger multi-reference tail.

## Limitations

Bulk biomarker-drug and all-actionables endpoints were not available (HTTP 403). Counts therefore reflect associations reachable through the annotation query panel, not a guaranteed complete OncoKB inventory. Even under that panel, the PMID structure is the binding constraint for CIViC-style ranking.

## Verdict

{verdict_text}

{closing}
"""

    path = REPORT_DIR / "report.md"
    path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {path}")
    return path


def write_readme(probe_result: dict) -> Path:
    text = f"""# Step 06 — OncoKB feasibility

Read-only probe of OncoKB API abstract grounding for a parallel gene–drug and gene–disease ranking target set.

**Verdict:** {probe_result["verdict"]}

**Key counts**
- Therapeutic genes queried: {probe_result["therapeutic_genes_n"]}
- Annotation queries: {probe_result["queries_n"]}
- Unique associations retrieved: {probe_result["total_associations"]}
- Single-PMID associations: {probe_result["single_pmid_associations"]} ({probe_result["single_pmid_share"]:.1%})
- Multi-PMID associations: {probe_result["multi_pmid_associations"]} ({probe_result["multi_pmid_share"]:.1%})
- Evaluable single-PMID triples (training PMIDs excluded): {probe_result["evaluable_triples_total"]}

Run: `bash -lc 'source ~/.bashrc && conda activate hf-hpc && python project_1/06_oncokb_feasibility/run.py'`
"""
    path = Path(__file__).resolve().parent / "README.md"
    path.write_text(text, encoding="utf-8")
    return path
