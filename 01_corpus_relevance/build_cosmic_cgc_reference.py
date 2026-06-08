#!/usr/bin/env python3
"""Build COSMIC Cancer Gene Census reference for future Round 3 / C1 only.

Does NOT modify step 01F oncology_subset outputs or any Round 1/2 artifacts.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _paths import OUTPUT_ROOT

STEP = "01_corpus_relevance"
DATA_DIR = OUTPUT_ROOT / "data" / STEP
COSMIC_DIR = DATA_DIR / "cosmic" / "Cosmic_CancerGeneCensus_Tsv_v104_GRCh38"
CGC_GZ = COSMIC_DIR / "Cosmic_CancerGeneCensus_v104_GRCh38.tsv.gz"
PRODUCT_DIR = DATA_DIR / "oncology_gene_reference"
OUT_TSV = PRODUCT_DIR / "cosmic_cgc_genes.tsv"
PROVENANCE_JSON = PRODUCT_DIR / "cosmic_cgc_provenance.json"
RECON_JSON = PRODUCT_DIR / "cosmic_civic_reconciliation.json"
README = PRODUCT_DIR / "README.md"

STEP00_FEATURES = OUTPUT_ROOT / "data" / "00_civic_feasibility" / "features.json"
STEP00_FETCH_META = OUTPUT_ROOT / "data" / "00_civic_feasibility" / "fetch_metadata.json"
ONCOLOGY_META = OUTPUT_ROOT / "outputs" / STEP / "oncology_subset_metadata.json"


def _gene_symbol(text: str) -> str:
    t = re.sub(r"\(.*?\)", "", str(text).strip())
    t = re.split(r"[/\s,;]", t)[0].strip()
    return t.upper()


def _load_civic_genes() -> set[str]:
    features = json.loads(STEP00_FEATURES.read_text(encoding="utf-8"))
    return {
        _gene_symbol(f["name"])
        for f in features
        if f.get("featureInstanceType") == "GENE" and f.get("name")
    } - {""}


def _read_cgc_raw() -> pd.DataFrame:
    if not CGC_GZ.exists():
        raise FileNotFoundError(f"Missing COSMIC CGC file: {CGC_GZ}")
    with gzip.open(CGC_GZ, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f, sep="\t", dtype=str)
    return df


def _build_reference(df: pd.DataFrame) -> pd.DataFrame:
    keep = df.copy()
    keep["gene_symbol"] = keep["GENE_SYMBOL"].map(_gene_symbol)
    keep = keep[keep["gene_symbol"].astype(bool)]
    keep = keep.sort_values(["gene_symbol", "TIER"], na_position="last")
    keep = keep.drop_duplicates(subset=["gene_symbol"], keep="first")

    def _synonyms(val: str | float) -> str:
        if pd.isna(val) or not str(val).strip():
            return ""
        parts = [_gene_symbol(x) for x in str(val).split(",") if str(x).strip()]
        parts = [p for p in parts if p and p != ""]
        return ";".join(dict.fromkeys(parts))

    out = pd.DataFrame(
        {
            "gene_symbol": keep["gene_symbol"],
            "cosmic_gene_id": keep["COSMIC_GENE_ID"].fillna(""),
            "name": keep["NAME"].fillna(""),
            "tier": keep["TIER"].fillna(""),
            "role_in_cancer": keep["ROLE_IN_CANCER"].fillna(""),
            "synonyms": keep["SYNONYMS"].map(_synonyms),
        }
    )
    return out.sort_values("gene_symbol").reset_index(drop=True)


def _reconcile(cosmic: set[str], civic: set[str]) -> dict:
    overlap = cosmic & civic
    cosmic_only = cosmic - civic
    civic_only = civic - cosmic
    return {
        "n_cosmic_cgc_genes": len(cosmic),
        "n_civic_genes": len(civic),
        "n_overlap": len(overlap),
        "n_cosmic_only": len(cosmic_only),
        "n_civic_only": len(civic_only),
        "fraction_cosmic_in_civic": round(len(overlap) / max(len(cosmic), 1), 4),
        "fraction_civic_in_cosmic": round(len(overlap) / max(len(civic), 1), 4),
        "note_ncit_mesh": (
            "NCIt Neoplasm and PubMed MeSH are disease/literature criteria in step 01F; "
            "they do not define a gene list. Gene-level reconciliation is against the CIViC "
            "gene set used by the existing oncology gene criterion."
        ),
    }


def _write_readme(recon: dict) -> None:
    text = f"""# Oncology gene reference (COSMIC CGC derived)

This directory holds a **new, separate product** for a possible future Round 3 / C1
(full-corpus vs oncology-subset training). It does **not** replace or update step 01F
oncology-subset outputs.

## Isolation

Round 1 and Round 2 train on the full matrix and evaluate on the full frozen CIViC pool.
They do **not** read this reference. Nothing here retroactively changes Round 1 or Round 2
reproducibility.

## Licensing

COSMIC Cancer Gene Census raw data remain under COSMIC licence in
`data/01_corpus_relevance/cosmic/` and are **not redistributed** in this repository.
This product contains only derived gene symbols and minimal annotation fields needed for
internal oncology-relevance marking.

## Contents

- `cosmic_cgc_genes.tsv` — minimal cancer-gene reference (symbol, COSMIC id, tier, role, synonyms)
- `cosmic_cgc_provenance.json` — source release metadata (v104, GRCh38)
- `cosmic_civic_reconciliation.json` — descriptive overlap with the CIViC gene set from step 01F

## Reconciliation with step 01F gene criterion (descriptive only)

CIViC genes (step 00): {recon['n_civic_genes']}
COSMIC CGC genes (this product): {recon['n_cosmic_cgc_genes']}
Overlap: {recon['n_overlap']}
COSMIC adds beyond CIViC: {recon['n_cosmic_only']}
CIViC genes not in COSMIC CGC: {recon['n_civic_only']}

NCIt and MeSH in step 01F apply to disease entities and literature, not to a gene inventory.
This delta shows what COSMIC would contribute if a future C1 gene criterion were widened;
it does not change the existing 01F definition.

## Consumption

For future Round 3 / C1 design only. Not consumed by folders 10, 11, or 20.
"""
    README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=== COSMIC Cancer Gene Census reference (future C1 only) ===\n")

    raw = _read_cgc_raw()
    print(f"Raw CGC columns ({len(raw.columns)}): {list(raw.columns)}")
    print(f"Raw CGC data rows: {len(raw)}")

    ref = _build_reference(raw)
    PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
    ref.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"Unique gene reference rows written: {len(ref)} -> {OUT_TSV}")

    civic = _load_civic_genes()
    cosmic = set(ref["gene_symbol"])
    recon_body = _reconcile(cosmic, civic)
    recon = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "civic_source": str(STEP00_FEATURES),
        "step01f_metadata_readonly": str(ONCOLOGY_META) if ONCOLOGY_META.exists() else None,
        **recon_body,
    }
    RECON_JSON.write_text(json.dumps(recon, indent=2), encoding="utf-8")

    fetch_ts = "unknown"
    if STEP00_FETCH_META.exists():
        fetch_ts = json.loads(STEP00_FETCH_META.read_text(encoding="utf-8")).get("fetch_timestamp", fetch_ts)

    provenance = {
        "product": "oncology_gene_reference/cosmic_cgc_genes.tsv",
        "derived_from": "COSMIC Cancer Gene Census",
        "cosmic_release": "v104",
        "genome_assembly": "GRCh38",
        "source_file": str(CGC_GZ.name),
        "source_readonly_path": str(CGC_GZ),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_genes": len(ref),
        "redistribution": "COSMIC raw data are not redistributed; internal annotation use only.",
        "consumers": "Future Round 3 / C1 only; not used by Round 1 or Round 2.",
        "civic_fetch_timestamp_readonly": fetch_ts,
    }
    PROVENANCE_JSON.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    _write_readme(recon_body)

    print("\n=== Reconciliation vs CIViC gene set (step 01F gene criterion) ===")
    print(f"  CIViC genes:              {recon_body['n_civic_genes']}")
    print(f"  COSMIC CGC genes:         {recon_body['n_cosmic_cgc_genes']}")
    print(f"  Overlap:                  {recon_body['n_overlap']}")
    print(f"  COSMIC adds beyond CIViC: {recon_body['n_cosmic_only']}")
    print(f"  CIViC not in COSMIC CGC:  {recon_body['n_civic_only']}")
    print(f"\n  {recon_body['note_ncit_mesh']}")
    print(f"\nProduct dir: {PRODUCT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
