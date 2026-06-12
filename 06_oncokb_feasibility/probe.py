"""Fetch OncoKB associations via annotation API and run grounding analysis."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    ACCESS_PROBE_JSON,
    ANNOTATIONS_JSONL,
    ASSOCIATIONS_CSV,
    ATYPICAL_NO_GENE,
    BATCH_SIZE,
    CNA_TYPES,
    CURATED_GENES_JSON,
    EVALUABLE_TRIPLES_CSV,
    FUSION_GENES,
    GROUNDABLE_TRIPLES_CSV,
    GROUNDING_SUMMARY_CSV,
    INFO_JSON,
    STRUCTURAL_EVIDENCE_CSV,
    TRAINING_PMIDS_JSON,
    UMBRELLA_ALTERATIONS,
)
from .oncokb_client import OncoKBClient


def _therapeutic_genes(curated: list[dict[str, Any]]) -> list[str]:
    genes: list[str] = []
    for row in curated:
        if row.get("highestSensitiveLevel") or row.get("highestResistanceLevel"):
            genes.append(str(row["hugoSymbol"]))
    return sorted(set(genes))


def _build_queries(genes: list[str]) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []

    for gene in genes:
        for alt in UMBRELLA_ALTERATIONS:
            queries.append(
                {
                    "query_id": f"mut:{gene}:{alt}",
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
                    "query_id": f"cna:{gene}:{cna}",
                    "endpoint": "annotate/copyNumberAlterations",
                    "payload": {
                        "referenceGenome": "GRCh37",
                        "gene": {"hugoSymbol": gene},
                        "copyNameAlterationType": cna,
                    },
                }
            )

    for gene in FUSION_GENES:
        if gene in genes:
            queries.append(
                {
                    "query_id": f"sv:{gene}:FUSION",
                    "endpoint": "annotate/structuralVariants",
                    "payload": {
                        "referenceGenome": "GRCh37",
                        "geneA": {"hugoSymbol": gene},
                        "structuralVariantType": "FUSION",
                        "functionalFusion": True,
                    },
                }
            )

    for alt, _ in ATYPICAL_NO_GENE:
        queries.append(
            {
                "query_id": f"atypical:{alt}",
                "endpoint": "annotate/mutations/byProteinChange",
                "payload": {"referenceGenome": "GRCh37", "alteration": alt},
            }
        )

    return queries


def _probe_access(client: OncoKBClient) -> dict:
    endpoints = [
        "info",
        "utils/allCuratedGenes",
        "utils/allActionables",
        "utils/allAlterations",
        "download/biomarkerDrugAssociationList.tsv",
    ]
    rows = []
    for ep in endpoints:
        status, body, url = client.get(ep)
        rows.append(
            {
                "endpoint": ep,
                "url": url,
                "status_code": status,
                "response_bytes": len(json.dumps(body)) if isinstance(body, (dict, list)) else len(str(body)),
            }
        )
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "endpoints": rows}


def _fetch_curated_genes(client: OncoKBClient, path: Path = CURATED_GENES_JSON) -> list[dict[str, Any]]:
    status, body, _ = client.get("utils/allCuratedGenes")
    if status != 200 or not isinstance(body, list):
        raise RuntimeError(f"Failed to fetch allCuratedGenes (HTTP {status})")
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body


def _run_annotations(client: OncoKBClient, queries: list[dict[str, Any]], path: Path = ANNOTATIONS_JSONL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(queries), BATCH_SIZE):
            batch = queries[start : start + BATCH_SIZE]
            by_endpoint: dict[str, list[dict[str, Any]]] = {}
            for item in batch:
                by_endpoint.setdefault(item["endpoint"], []).append(item)

            for endpoint, items in by_endpoint.items():
                payload = [item["payload"] for item in items]
                status, body, url = client.post_json(endpoint, payload)
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
                        "endpoint": endpoint,
                        "url": url,
                        "request": item["payload"],
                        "response": response,
                    }
                    handle.write(json.dumps(record) + "\n")


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


def _normalize_pmids(value: Any) -> list[str]:
    if not value:
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text.isdigit():
            out.append(text)
    return sorted(set(out))


def _association_key(kind: str, gene: str, head: str, tail: str, disease: str, level: str, pmids: list[str]) -> str:
    raw = "|".join([kind, gene, head, tail, disease, level, ",".join(pmids)])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _extract_associations(path: Path = ANNOTATIONS_JSONL) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            response = record["response"]
            gene = _gene_from_response(response)
            alterations = record["request"].get("alteration") or record["request"].get("copyNameAlterationType")
            if isinstance(alterations, str):
                alt_label = alterations
            else:
                alt_label = str(record["request"].get("structuralVariantType") or "")

            for treatment in response.get("treatments") or []:
                drugs = [d.get("drugName") for d in treatment.get("drugs") or [] if d.get("drugName")]
                disease = (treatment.get("levelAssociatedCancerType") or {}).get("name") or ""
                pmids = _normalize_pmids(treatment.get("pmids"))
                level = str(treatment.get("level") or "")
                alts = ", ".join(treatment.get("alterations") or [alt_label])
                for drug in drugs or [""]:
                    key = _association_key("gene-drug", gene, gene, drug, disease, level, pmids)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "association_id": key,
                            "pair_type": "gene-drug",
                            "gene": gene,
                            "variant_or_alteration": alts,
                            "entity_a": gene,
                            "entity_b": drug,
                            "disease_context": disease,
                            "level": level,
                            "pmid_count": len(pmids),
                            "pmids": ";".join(pmids),
                            "grounding_class": (
                                "single_pmid" if len(pmids) == 1 else "multi_pmid" if len(pmids) > 1 else "no_pmid"
                            ),
                            "source_block": "treatments",
                        }
                    )

            for block_name, pair_type in (
                ("diagnosticImplications", "gene-disease"),
                ("prognosticImplications", "gene-disease"),
            ):
                for implication in response.get(block_name) or []:
                    disease = (implication.get("tumorType") or {}).get("name") or ""
                    pmids = _normalize_pmids(implication.get("pmids"))
                    level = str(implication.get("levelOfEvidence") or "")
                    alts = ", ".join(implication.get("alterations") or [alt_label])
                    key = _association_key(pair_type, gene, gene, disease, disease, level, pmids)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "association_id": key,
                            "pair_type": pair_type,
                            "gene": gene,
                            "variant_or_alteration": alts,
                            "entity_a": gene,
                            "entity_b": disease,
                            "disease_context": disease,
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
        pmid_counter = Counter(sub["pmid_count"].tolist())
        for count, n in sorted(pmid_counter.items()):
            structural_rows.append(
                {
                    "pair_type": pair_type,
                    "pmids_per_association": count,
                    "association_count": n,
                }
            )
    structural_df = pd.DataFrame(structural_rows)
    structural_df.to_csv(STRUCTURAL_EVIDENCE_CSV, index=False)

    triple_rows: list[dict[str, Any]] = []
    for row in associations.itertuples():
        if row.grounding_class != "single_pmid":
            continue
        pmid = row.pmids
        triple_rows.append(
            {
                "pair_type": row.pair_type,
                "gene": row.gene,
                "entity_b": row.entity_b,
                "pmid": pmid,
                "variant_or_alteration": row.variant_or_alteration,
                "disease_context": row.disease_context,
                "level": row.level,
                "in_training_pmids": pmid in training_pmids,
            }
        )

    triples_df = pd.DataFrame(triple_rows)
    triples_df.to_csv(GROUNDABLE_TRIPLES_CSV, index=False)
    evaluable_df = triples_df[~triples_df["in_training_pmids"]].copy()
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

    multi_share = multi_assoc / total_assoc if total_assoc else 0.0
    single_share = single_assoc / total_assoc if total_assoc else 0.0

    # NO-GO if most associations carry aggregated PMIDs and single-PMID share is small.
    verdict = "NO-GO"
    if single_assoc > 0 and single_share >= 0.25 and (eval_gd + eval_gdis) >= 50:
        verdict = "GO"

    return {
        "total_associations": total_assoc,
        "single_pmid_associations": single_assoc,
        "multi_pmid_associations": multi_assoc,
        "no_pmid_associations": no_pmid,
        "single_pmid_share": single_share,
        "multi_pmid_share": multi_share,
        "single_gene_drug_associations": single_gd,
        "single_gene_disease_associations": single_gdis,
        "groundable_gene_drug_triples": single_gd,
        "groundable_gene_disease_triples": single_gdis,
        "evaluable_gene_drug_triples": eval_gd,
        "evaluable_gene_disease_triples": eval_gdis,
        "evaluable_triples_total": len(evaluable_df),
        "training_pmids_n": len(training_pmids),
        "verdict": verdict,
        "summary_df": summary_df,
        "structural_df": structural_df,
        "triples_df": triples_df,
        "evaluable_df": evaluable_df,
    }


def run_probe(force_fetch: bool = False) -> dict[str, Any]:
    client = OncoKBClient()

    status, info, _ = client.get("info")
    if status != 200:
        raise RuntimeError(f"/info failed: HTTP {status}")
    INFO_JSON.write_text(json.dumps(info, indent=2), encoding="utf-8")

    access = _probe_access(client)
    ACCESS_PROBE_JSON.write_text(json.dumps(access, indent=2), encoding="utf-8")

    if force_fetch or not CURATED_GENES_JSON.exists():
        curated = _fetch_curated_genes(client)
    else:
        curated = json.loads(CURATED_GENES_JSON.read_text(encoding="utf-8"))

    genes = _therapeutic_genes(curated)
    queries = _build_queries(genes)
    print(f"  therapeutic genes: {len(genes)}")
    print(f"  annotation queries: {len(queries)}")

    if force_fetch or not ANNOTATIONS_JSONL.exists():
        _run_annotations(client, queries)

    associations = _extract_associations()
    grounding = _build_grounding_outputs(associations)

    print("\n=== Grounding test ===")
    print(f"  total associations retrieved: {grounding['total_associations']}")
    print(f"  single-PMID associations: {grounding['single_pmid_associations']}")
    print(f"  multi-PMID associations: {grounding['multi_pmid_associations']}")
    print(f"  no-PMID associations: {grounding['no_pmid_associations']}")
    print(f"  groundable gene-drug triples (single PMID): {grounding['groundable_gene_drug_triples']}")
    print(f"  groundable gene-disease triples (single PMID): {grounding['groundable_gene_disease_triples']}")
    print(f"  evaluable triples after training-PMID exclusion: {grounding['evaluable_triples_total']}")
    print(f"  VERDICT: {grounding['verdict']}")

    return {
        "info": info,
        "access": access,
        "therapeutic_genes_n": len(genes),
        "queries_n": len(queries),
        "associations_n": len(associations),
        **grounding,
    }
