"""Load the OncoKB cancer gene list from workspace or API."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from .config import CANCER_GENE_LIST_JSON, WORKSPACE_CANCER_GENE_LIST
from .oncokb_client import OncoKBClient


def _parse_tsv(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def _oncokb_annotated_genes(rows: list[dict[str, str]]) -> list[str]:
    genes: list[str] = []
    for row in rows:
        if (row.get("OncoKB Annotated") or "").strip().lower() != "yes":
            continue
        symbol = (row.get("Hugo Symbol") or row.get("HugoSymbol") or "").strip()
        if symbol:
            genes.append(symbol)
    return sorted(set(genes))


def load_cancer_genes(client: OncoKBClient | None = None, force_fetch: bool = False) -> dict[str, Any]:
    """Return cancer gene metadata and the OncoKB-annotated Hugo symbols to query."""
    source = ""
    rows: list[dict[str, str]]

    if WORKSPACE_CANCER_GENE_LIST.exists() and not force_fetch:
        text = WORKSPACE_CANCER_GENE_LIST.read_text(encoding="utf-8")
        rows = _parse_tsv(text)
        source = "workspace_file"
    else:
        if client is None:
            client = OncoKBClient()
        status, body, url = client.get("utils/cancerGeneList.txt")
        if status != 200 or not isinstance(body, str):
            raise RuntimeError(f"Failed to fetch cancerGeneList.txt (HTTP {status})")
        rows = _parse_tsv(body)
        source = "api_utils_cancerGeneList_txt"
        WORKSPACE_CANCER_GENE_LIST.parent.mkdir(parents=True, exist_ok=True)
        WORKSPACE_CANCER_GENE_LIST.write_text(body if isinstance(body, str) else str(body), encoding="utf-8")

    genes = _oncokb_annotated_genes(rows)
    meta = {
        "source": source,
        "total_rows": len(rows),
        "oncokb_annotated_genes": len(genes),
        "genes": genes,
    }
    CANCER_GENE_LIST_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
