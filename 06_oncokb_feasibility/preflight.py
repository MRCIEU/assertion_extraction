"""Preflight checks for step 06."""

from __future__ import annotations

import json
from pathlib import Path

import requests

from .cancer_genes import load_cancer_genes
from .config import ONCOKB_BASE_URL, ONCOKB_PRODUCTION_HOST, TRAINING_PMIDS_JSON, WORKSPACE_CANCER_GENE_LIST
from .oncokb_client import OncoKBClient


def check_training_pmids(path: Path = TRAINING_PMIDS_JSON) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Training PMID list missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    biored = set(data.get("biored_training_pmids") or [])
    drugprot = set(data.get("drugprot_training_pmids") or [])
    union = biored | drugprot
    return {
        "path": str(path),
        "biored_n": len(biored),
        "drugprot_n": len(drugprot),
        "union_n": len(union),
        "excluded_pmids": list(data.get("excluded_pmids") or []),
    }


def check_production_annotation(client: OncoKBClient) -> dict:
    """Sanity check: authenticated production annotation returns therapeutic data."""
    status, body, url = client.get(
        "annotate/mutations/byProteinChange",
        params={"hugoSymbol": "BRAF", "alteration": "V600E", "tumorType": "Melanoma"},
    )
    if status == 401:
        raise RuntimeError(
            "Production annotation returned HTTP 401. Check ONCOKB_API_TOKEN in OncoKB Account Settings."
        )
    if status == 403:
        raise RuntimeError(
            "Production annotation returned HTTP 403 on BRAF V600E in Melanoma. "
            "Check token permissions in OncoKB Account Settings."
        )
    if status != 200 or not isinstance(body, dict):
        raise RuntimeError(f"Production annotation sanity check failed (HTTP {status})")
    treatments = body.get("treatments") or []
    if not treatments:
        raise RuntimeError("Production annotation returned HTTP 200 but no treatments for BRAF V600E in Melanoma.")
    return {
        "instance": ONCOKB_PRODUCTION_HOST,
        "endpoint": url,
        "status_code": status,
        "treatments_n": len(treatments),
    }


def run_preflight() -> dict:
    print("=== Step 06 preflight ===")
    training = check_training_pmids()
    print(f"  training PMIDs present: {training['union_n']} (BioRED {training['biored_n']}, DrugProt {training['drugprot_n']})")

    response = requests.get(f"{ONCOKB_BASE_URL}/info", timeout=30)
    print(f"  production /info reachable: HTTP {response.status_code}")

    client = OncoKBClient()
    status, body, _ = client.get("info")
    if status != 200:
        raise RuntimeError(f"Authenticated /info failed with HTTP {status}")
    print(f"  authenticated production access: OK ({client.access_mode})")

    sanity = check_production_annotation(client)
    print(
        f"  annotation sanity check (BRAF V600E, Melanoma): HTTP {sanity['status_code']}, "
        f"{sanity['treatments_n']} treatments"
    )

    if WORKSPACE_CANCER_GENE_LIST.exists():
        gene_meta = load_cancer_genes(force_fetch=False)
        gene_source = "existing workspace cancer gene list"
    else:
        gene_meta = load_cancer_genes(client=client, force_fetch=True)
        gene_source = "fetched from API utils/cancerGeneList.txt"
    print(
        f"  cancer gene list: {gene_meta['oncokb_annotated_genes']} OncoKB-annotated genes "
        f"({gene_source})"
    )

    return {
        "training_pmids": training,
        "access_mode": client.access_mode,
        "instance": ONCOKB_PRODUCTION_HOST,
        "annotation_sanity": sanity,
        "cancer_genes": gene_meta,
        "data_version": body.get("dataVersion") if isinstance(body, dict) else None,
    }
