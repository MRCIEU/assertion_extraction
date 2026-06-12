"""Fetch OncoKB associations via batch annotation API and run grounding analysis."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .cancer_genes import load_cancer_genes
from .config import (
    ANNOTATIONS_JSONL,
    ASSOCIATIONS_CSV,
    ATYPICAL_NO_GENE,
    BATCH_SIZE,
    CANCER_GENE_LIST_JSON,
    CNA_TYPES,
    EVALUABLE_TRIPLES_CSV,
    FETCH_METADATA_JSON,
    GROUNDABLE_TRIPLES_CSV,
    GROUNDING_SUMMARY_CSV,
    INFO_JSON,
    MIN_EVALUABLE_GENE_DISEASE,
    MIN_EVALUABLE_GENE_DRUG,
    ONCOKB_PRODUCTION_HOST,
    PMID_RETRIEVABILITY_CSV,
    QUERY_MANIFEST_JSON,
    STRUCTURAL_EVIDENCE_CSV,
    TRAINING_PMIDS_JSON,
    UMBRELLA_ALTERATIONS,
)
from .oncokb_client import OncoKBClient

UMBRELLA_SET = set(UMBRELLA_ALTERATIONS) | set(CNA_TYPES) | set(ATYPICAL_NO_GENE)
SKIP_ALTERATION_RE = re.compile(
    r"^(unknown|none|null|na|n/a|other|all|any|wildtype|wt)$",
    re.I,
)


def _build_phase1_queries(genes: list[str]) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for gene in genes:
        for alt in UMBRELLA_ALTERATIONS:
            queries.append(
                {
                    "query_id": f"p1:mut:{gene}:{alt}",
                    "phase": 1,
                    "endpoint": "annotate/mutations/byProteinChange",
                    "payload": {
                        "referenceGenome": "GRCh37",
                        "gene": {"hugoSymbol": gene},
                        "alteration": alt,
                    },
                }
            )
        for cna in CNA_TYPES:
            queries.append(
                {
                    "query_id": f"p1:cna:{gene}:{cna}",
                    "phase": 1,
                    "endpoint": "annotate/copyNumberAlterations",
                    "payload": {
                        "referenceGenome": "GRCh37",
                        "gene": {"hugoSymbol": gene},
                        "copyNameAlterationType": cna,
                    },
                }
            )
        queries.append(
            {
                "query_id": f"p1:sv:{gene}:FUSION",
                "phase": 1,
                "endpoint": "annotate/structuralVariants",
                "payload": {
                    "referenceGenome": "GRCh37",
                    "geneA": {"hugoSymbol": gene},
                    "structuralVariantType": "FUSION",
                    "functionalFusion": True,
                },
            }
        )
    for alt in ATYPICAL_NO_GENE:
        queries.append(
            {
                "query_id": f"p1:atypical:{alt}",
                "phase": 1,
                "endpoint": "annotate/mutations/byProteinChange",
                "payload": {"referenceGenome": "GRCh37", "alteration": alt},
            }
        )
    return queries


def _gene_from_response(response: dict[str, Any]) -> str:
    query = response.get("query") or {}
    for key in ("hugoSymbol", "hugoSymbolA"):
        if query.get(key):
            return str(query[key])
    gene = query.get("gene") or {}
    if gene.get("hugoSymbol"):
        return str(gene["hugoSymbol"])
    gene_a = query.get("geneA") or {}
    if gene_a.get("hugoSymbol"):
        return str(gene_a["hugoSymbol"])
    return ""


def _collect_specific_alterations(path: Path) -> set[tuple[str, str]]:
    discovered: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            response = record["response"]
            gene = _gene_from_response(response)
            if not gene:
                continue
            blocks = (response.get("treatments") or []) + (response.get("diagnosticImplications") or []) + (
                response.get("prognosticImplications") or []
            )
            for block in blocks:
                for alt in block.get("alterations") or []:
                    alt_text = str(alt).strip()
                    if not alt_text or alt_text in UMBRELLA_SET:
                        continue
                    if SKIP_ALTERATION_RE.match(alt_text):
                        continue
                    if len(alt_text) > 80:
                        continue
                    discovered.add((gene, alt_text))
    return discovered


def _build_phase2_queries(discovered: set[tuple[str, str]]) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for gene, alt in sorted(discovered):
        safe_alt = alt.replace("/", "_")[:60]
        queries.append(
            {
                "query_id": f"p2:mut:{gene}:{safe_alt}",
                "phase": 2,
                "endpoint": "annotate/mutations/byProteinChange",
                "payload": {
                    "referenceGenome": "GRCh37",
                    "gene": {"hugoSymbol": gene},
                    "alteration": alt,
                },
            }
        )
    return queries


def _run_annotations(client: OncoKBClient, queries: list[dict[str, Any]], path: Path, append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    written = 0
    total_batches = (len(queries) + BATCH_SIZE - 1) // BATCH_SIZE
    with path.open(mode, encoding="utf-8") as handle:
        for batch_idx, start in enumerate(range(0, len(queries), BATCH_SIZE), start=1):
            batch = queries[start : start + BATCH_SIZE]
            by_endpoint: dict[str, list[dict[str, Any]]] = {}
            for item in batch:
                by_endpoint.setdefault(item["endpoint"], []).append(item)

            for endpoint, items in by_endpoint.items():
                payload = [item["payload"] for item in items]
                status, body, url = client.post_json(endpoint, payload)
                if status in {401, 403}:
                    raise RuntimeError(
                        f"Annotation batch failed with HTTP {status} on production instance. "
                        "Check ONCOKB_API_TOKEN in OncoKB Account Settings."
                    )
                if status != 200:
                    raise RuntimeError(f"Batch annotation failed for {endpoint}: HTTP {status}")
                if not isinstance(body, list) or len(body) != len(items):
                    raise RuntimeError(
                        f"Unexpected batch response for {endpoint}: expected {len(items)} items, got "
                        f"{len(body) if isinstance(body, list) else type(body)}"
                    )
                for item, response in zip(items, body):
                    record = {
                        "query_id": item["query_id"],
                        "phase": item["phase"],
                        "endpoint": endpoint,
                        "url": url,
                        "request": item["payload"],
                        "response": response,
                    }
                    handle.write(json.dumps(record) + "\n")
                    written += 1
            if batch_idx % 20 == 0 or batch_idx == total_batches:
                print(f"    annotation batches: {batch_idx}/{total_batches} ({written} responses written)")
    return written


def _normalize_pmids(value: Any) -> list[str]:
    if not value:
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text.isdigit():
            out.append(text)
    return sorted(set(out))


def _association_key(kind: str, gene: str, tail: str, disease: str, level: str, pmids: list[str]) -> str:
    raw = "|".join([kind, gene, tail, disease, level, ",".join(pmids)])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _extract_associations(path: Path = ANNOTATIONS_JSONL) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            response = record["response"]
            gene = _gene_from_response(response)
            req = record["request"]
            alt_label = (
                req.get("alteration")
                or req.get("copyNameAlterationType")
                or req.get("structuralVariantType")
                or ""
            )

            for treatment in response.get("treatments") or []:
                drugs = [d.get("drugName") for d in treatment.get("drugs") or [] if d.get("drugName")]
                disease = (treatment.get("levelAssociatedCancerType") or {}).get("name") or ""
                pmids = _normalize_pmids(treatment.get("pmids"))
                level = str(treatment.get("level") or "")
                alts = ", ".join(treatment.get("alterations") or [alt_label])
                for drug in drugs or [""]:
                    key = _association_key("gene-drug", gene, drug, disease, level, pmids)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "association_id": key,
                            "pair_type": "gene-drug",
                            "gene": gene,
                            "entity_b": drug,
                            "variant_or_alteration": alts,
                            "tumor_type": disease,
                            "level": level,
                            "pmid_count": len(pmids),
                            "pmids": ";".join(pmids),
                            "grounding_class": (
                                "single_pmid" if len(pmids) == 1 else "multi_pmid" if len(pmids) > 1 else "no_pmid"
                            ),
                            "source_block": "treatments",
                        }
                    )

            for block_name in ("diagnosticImplications", "prognosticImplications"):
                for implication in response.get(block_name) or []:
                    disease = (implication.get("tumorType") or {}).get("name") or ""
                    pmids = _normalize_pmids(implication.get("pmids"))
                    level = str(implication.get("levelOfEvidence") or "")
                    alts = ", ".join(implication.get("alterations") or [alt_label])
                    key = _association_key("gene-disease", gene, disease, disease, level, pmids)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "association_id": key,
                            "pair_type": "gene-disease",
                            "gene": gene,
                            "entity_b": disease,
                            "variant_or_alteration": alts,
                            "tumor_type": disease,
                            "level": level,
                            "pmid_count": len(pmids),
                            "pmids": ";".join(pmids),
                            "grounding_class": (
                                "single_pmid" if len(pmids) == 1 else "multi_pmid" if len(pmids) > 1 else "no_pmid"
                            ),
                            "source_block": block_name,
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(ASSOCIATIONS_CSV, index=False)
    return df


def _load_training_pmids() -> set[str]:
    data = json.loads(TRAINING_PMIDS_JSON.read_text(encoding="utf-8"))
    return set(data.get("biored_training_pmids") or []) | set(data.get("drugprot_training_pmids") or [])


def _fetch_pubmed_abstracts(pmids: list[str], batch_size: int = 100) -> dict[str, bool]:
    """Return PMID -> abstract retrievable (non-empty abstract text)."""
    api_key = os.environ.get("NCBI_API_KEY", "")
    retrievable: dict[str, bool] = {}
    unique_pmids = sorted({p for p in pmids if p})

    for start in range(0, len(unique_pmids), batch_size):
        batch = unique_pmids[start : start + batch_size]
        params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml", "rettype": "abstract"}
        if api_key:
            params["api_key"] = api_key
        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params=params,
            timeout=60,
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        found: set[str] = set()
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            abstract_el = article.find(".//Abstract")
            if pmid_el is None:
                continue
            pmid = pmid_el.text.strip()
            found.add(pmid)
            if abstract_el is None:
                retrievable[pmid] = False
                continue
            parts = ["".join(el.itertext()).strip() for el in abstract_el.findall("AbstractText")]
            text = " ".join(p for p in parts if p)
            retrievable[pmid] = bool(text)
        for pmid in batch:
            if pmid not in found:
                retrievable[pmid] = False
        time.sleep(0.12)
    return retrievable


def _check_pmid_retrievability(evaluable_df: pd.DataFrame) -> pd.DataFrame:
    if evaluable_df.empty:
        pd.DataFrame(columns=["pmid", "abstract_retrievable"]).to_csv(PMID_RETRIEVABILITY_CSV, index=False)
        return evaluable_df.assign(abstract_retrievable=pd.Series(dtype=bool))

    pmids = sorted(evaluable_df["pmid"].astype(str).unique())
    lookup = _fetch_pubmed_abstracts(pmids)
    pd.DataFrame(
        [{"pmid": pmid, "abstract_retrievable": lookup.get(pmid, False)} for pmid in pmids]
    ).to_csv(PMID_RETRIEVABILITY_CSV, index=False)

    out = evaluable_df.copy()
    out["abstract_retrievable"] = out["pmid"].astype(str).map(lambda p: lookup.get(p, False))
    return out


def _decide_verdict(eval_gd: int, eval_gdis: int) -> str:
    if eval_gd >= MIN_EVALUABLE_GENE_DRUG and eval_gdis >= MIN_EVALUABLE_GENE_DISEASE:
        return "GO"
    if eval_gd >= MIN_EVALUABLE_GENE_DRUG and eval_gdis < MIN_EVALUABLE_GENE_DISEASE:
        return "PARTIAL"
    return "NO-GO"


def _build_grounding_outputs(associations: pd.DataFrame) -> dict[str, Any]:
    training_pmids = _load_training_pmids()

    summary_rows = []
    for pair_type in ("gene-drug", "gene-disease"):
        sub = associations[associations["pair_type"] == pair_type]
        for cls in ("single_pmid", "multi_pmid", "no_pmid"):
            n = int((sub["grounding_class"] == cls).sum())
            summary_rows.append({"pair_type": pair_type, "grounding_class": cls, "association_count": n})
        summary_rows.append({"pair_type": pair_type, "grounding_class": "total", "association_count": len(sub)})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(GROUNDING_SUMMARY_CSV, index=False)

    structural_rows = []
    for pair_type in ("gene-drug", "gene-disease"):
        sub = associations[associations["pair_type"] == pair_type]
        for count, n in sorted(Counter(sub["pmid_count"].tolist()).items()):
            structural_rows.append(
                {"pair_type": pair_type, "pmids_per_association": count, "association_count": n}
            )
    structural_df = pd.DataFrame(structural_rows)
    structural_df.to_csv(STRUCTURAL_EVIDENCE_CSV, index=False)

    triple_rows: list[dict[str, Any]] = []
    for row in associations.itertuples():
        if row.grounding_class != "single_pmid":
            continue
        triple_rows.append(
            {
                "pair_type": row.pair_type,
                "gene": row.gene,
                "entity_b": row.entity_b,
                "pmid": row.pmids,
                "evidence_level": row.level,
                "tumor_type": row.tumor_type,
                "variant_or_alteration": row.variant_or_alteration,
                "in_training_pmids": row.pmids in training_pmids,
            }
        )

    triples_df = pd.DataFrame(triple_rows)
    triples_df.to_csv(GROUNDABLE_TRIPLES_CSV, index=False)
    evaluable_df = triples_df[~triples_df["in_training_pmids"]].copy()
    evaluable_df = _check_pmid_retrievability(evaluable_df)
    evaluable_df.to_csv(EVALUABLE_TRIPLES_CSV, index=False)

    single_assoc = int((associations["grounding_class"] == "single_pmid").sum())
    multi_assoc = int((associations["grounding_class"] == "multi_pmid").sum())
    no_pmid = int((associations["grounding_class"] == "no_pmid").sum())
    total_assoc = len(associations)

    single_gd = int(((associations["pair_type"] == "gene-drug") & (associations["grounding_class"] == "single_pmid")).sum())
    single_gdis = int(
        ((associations["pair_type"] == "gene-disease") & (associations["grounding_class"] == "single_pmid")).sum()
    )
    eval_gd = int((evaluable_df["pair_type"] == "gene-drug").sum())
    eval_gdis = int((evaluable_df["pair_type"] == "gene-disease").sum())
    retrievable_n = int(evaluable_df["abstract_retrievable"].sum()) if len(evaluable_df) else 0

    verdict = _decide_verdict(eval_gd, eval_gdis)

    return {
        "total_associations": total_assoc,
        "single_pmid_associations": single_assoc,
        "multi_pmid_associations": multi_assoc,
        "no_pmid_associations": no_pmid,
        "single_pmid_share": single_assoc / total_assoc if total_assoc else 0.0,
        "multi_pmid_share": multi_assoc / total_assoc if total_assoc else 0.0,
        "single_gene_drug_associations": single_gd,
        "single_gene_disease_associations": single_gdis,
        "evaluable_gene_drug_triples": eval_gd,
        "evaluable_gene_disease_triples": eval_gdis,
        "evaluable_triples_total": len(evaluable_df),
        "evaluable_abstract_retrievable_n": retrievable_n,
        "training_pmids_n": len(training_pmids),
        "verdict": verdict,
        "summary_df": summary_df,
        "structural_df": structural_df,
        "evaluable_df": evaluable_df,
    }


def run_probe(force_fetch: bool = False) -> dict[str, Any]:
    client = OncoKBClient()

    status, info, _ = client.get("info")
    if status != 200:
        raise RuntimeError(f"/info failed: HTTP {status}")
    INFO_JSON.write_text(json.dumps(info, indent=2), encoding="utf-8")

    gene_meta = load_cancer_genes(client=client, force_fetch=force_fetch)
    genes = gene_meta["genes"]
    phase1 = _build_phase1_queries(genes)
    print(f"  cancer genes (OncoKB annotated): {gene_meta['oncokb_annotated_genes']} (source: {gene_meta['source']})")
    print(f"  phase-1 annotation queries: {len(phase1)}")

    if force_fetch and ANNOTATIONS_JSONL.exists():
        ANNOTATIONS_JSONL.unlink()

    if force_fetch or not ANNOTATIONS_JSONL.exists():
        _run_annotations(client, phase1, ANNOTATIONS_JSONL, append=False)
        discovered = _collect_specific_alterations(ANNOTATIONS_JSONL)
        phase2 = _build_phase2_queries(discovered)
        print(f"  phase-2 specific-alteration queries: {len(phase2)}")
        if phase2:
            _run_annotations(client, phase2, ANNOTATIONS_JSONL, append=True)
        phase2_n = len(phase2)
    else:
        discovered = _collect_specific_alterations(ANNOTATIONS_JSONL)
        phase2 = _build_phase2_queries(discovered)
        phase2_n = len(phase2)

    total_queries = len(phase1) + phase2_n
    annotation_responses = sum(1 for _ in ANNOTATIONS_JSONL.open(encoding="utf-8"))

    associations = _extract_associations()
    grounding = _build_grounding_outputs(associations)

    metadata = {
        "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
        "instance": ONCOKB_PRODUCTION_HOST,
        "method": "batch POST annotation API only (no bulk export endpoints)",
        "cancer_gene_source": gene_meta["source"],
        "cancer_gene_total_rows": gene_meta["total_rows"],
        "oncokb_annotated_genes": gene_meta["oncokb_annotated_genes"],
        "phase1_queries": len(phase1),
        "phase2_queries": phase2_n,
        "total_annotation_queries": total_queries,
        "annotation_responses": annotation_responses,
        "unique_associations": len(associations),
        "data_version": info.get("dataVersion"),
    }
    FETCH_METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    QUERY_MANIFEST_JSON.write_text(
        json.dumps({"phase1_n": len(phase1), "phase2_n": metadata["phase2_queries"], "genes": genes}, indent=2),
        encoding="utf-8",
    )

    print("\n=== Grounding test (by pair type) ===")
    for pair_type in ("gene-drug", "gene-disease"):
        sub = associations[associations["pair_type"] == pair_type]
        print(f"  {pair_type}:")
        for cls in ("single_pmid", "multi_pmid", "no_pmid"):
            n = int((sub["grounding_class"] == cls).sum())
            print(f"    {cls}: {n}")
    print(f"  evaluable gene-drug triples: {grounding['evaluable_gene_drug_triples']}")
    print(f"  evaluable gene-disease triples: {grounding['evaluable_gene_disease_triples']}")
    print(f"  evaluable abstracts retrievable: {grounding['evaluable_abstract_retrievable_n']}/{grounding['evaluable_triples_total']}")
    print(f"  VERDICT: {grounding['verdict']}")

    return {
        "info": info,
        "gene_meta": gene_meta,
        "phase1_queries_n": len(phase1),
        "phase2_queries_n": phase2_n,
        "queries_n": total_queries,
        "associations_n": len(associations),
        **grounding,
    }
