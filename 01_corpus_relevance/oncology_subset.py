"""Oncology-subset analysis of BioRED + DrugProt training relations (step 01 add-on)."""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from .config import (
    CORPORA,
    DATA_DIR,
    ONCOLOGY_AGREEMENT_CSV,
    ONCOLOGY_FRACTIONS_CSV,
    ONCOLOGY_METADATA_JSON,
    ONCOLOGY_PMID_MESH_JSON,
)
from .entity_normalization import civic_pair_type, entity_surface, normalize_entity_type

ONCOLOGY_DATA_DIR = DATA_DIR / "oncology"
NCIT_NEoplasm_CORE_URL = "https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/Neoplasm/Neoplasm_Core.txt"
NCIT_NEoplasm_MESH_MAP_URL = (
    "https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/Neoplasm/Neoplasm_Core_Mappings_NCIm_Terms.csv"
)
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
EFETCH_BATCH = 200
EFETCH_INTERVAL_S = 0.34
TRAINING_CORPORA = ("biored", "drugprot")
FOCUS_PAIR_TYPES = ("gene-drug", "gene-disease")


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"  Downloading {url} -> {dest.name}")
        with urllib.request.urlopen(url, timeout=120) as resp:
            dest.write_bytes(resp.read())
    return dest


def _gene_symbol(text: str) -> str:
    t = re.sub(r"\(.*?\)", "", str(text).strip())
    t = re.split(r"[/\s,;]", t)[0].strip()
    return t.upper()


def load_ncit_neoplasm_reference() -> dict[str, Any]:
    """Load NCIt Neoplasm Core codes and MeSH crosswalk from NCI EVS."""
    core_path = _download(NCIT_NEoplasm_CORE_URL, ONCOLOGY_DATA_DIR / "Neoplasm_Core.txt")
    map_path = _download(NCIT_NEoplasm_MESH_MAP_URL, ONCOLOGY_DATA_DIR / "Neoplasm_Core_Mappings_NCIm_Terms.csv")

    ncit_codes: set[str] = set()
    ncit_terms: set[str] = set()
    with core_path.open(encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            code = (row.get("Code") or "").strip()
            if code:
                ncit_codes.add(code)
            term = (row.get("Preferred Term") or "").strip().lower()
            if term:
                ncit_terms.add(term)
            for syn in (row.get("Synonyms") or "").split("||"):
                syn = syn.strip().lower()
                if syn:
                    ncit_terms.add(syn)

    mesh_ids: set[str] = set()
    with map_path.open(encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("NCIm Source") or "").strip() != "MSH":
                continue
            code = (row.get("Source Code") or "").strip()
            if code.startswith("D"):
                mesh_ids.add(code)

    return {
        "ncit_source_url": NCIT_NEoplasm_CORE_URL,
        "ncit_mapping_url": NCIT_NEoplasm_MESH_MAP_URL,
        "ncit_core_file_date": datetime.fromtimestamp(core_path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "n_ncit_neoplasm_codes": len(ncit_codes),
        "n_mesh_neoplasm_ids": len(mesh_ids),
        "ncit_codes": ncit_codes,
        "ncit_terms": ncit_terms,
        "mesh_neoplasm_ids": mesh_ids,
    }


def load_civic_gene_reference() -> dict[str, Any]:
    from .config import STEP00_DATA

    features_path = STEP00_DATA / "features.json"
    fetch_meta_path = STEP00_DATA / "fetch_metadata.json"
    features = json.loads(features_path.read_text(encoding="utf-8"))
    fetch_ts = "unknown"
    if fetch_meta_path.exists():
        fetch_ts = json.loads(fetch_meta_path.read_text(encoding="utf-8")).get("fetch_timestamp", fetch_ts)

    genes = {
        _gene_symbol(f["name"])
        for f in features
        if f.get("featureInstanceType") == "GENE" and f.get("name")
    }
    genes.discard("")
    return {
        "source": str(features_path),
        "civic_fetch_timestamp": fetch_ts,
        "n_civic_genes": len(genes),
        "genes": genes,
        "license_note": "CIViC gene list (CC0); COSMIC Cancer Gene Census excluded due to licence/redistribution limits.",
    }


def _entity_norm_ids(entity: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for norm in entity.get("normalized") or []:
        db = str(norm.get("db_name") or "").strip()
        db_id = str(norm.get("db_id") or "").strip()
        if db and db_id:
            out.append((db, db_id))
    return out


def _disease_is_neoplasm(entity: dict, ncit: dict[str, Any]) -> bool:
    for db, db_id in _entity_norm_ids(entity):
        if db in ("NCIt", "NCI") and db_id in ncit["ncit_codes"]:
            return True
        if db == "MESH" and db_id in ncit["mesh_neoplasm_ids"]:
            return True
    text = entity_surface(entity).lower()
    return text in ncit["ncit_terms"]


def _extract_training_relations(ncit: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for corpus_key in TRAINING_CORPORA:
        spec = CORPORA[corpus_key]
        ds = load_dataset(spec["hf_id"], spec["config"], trust_remote_code=True)
        for split in spec["train_splits"]:
            if split not in ds:
                continue
            for doc in ds[split]:
                pmid = str(doc.get("document_id") or doc.get("id"))
                emap = {e["id"]: e for e in doc.get("entities", [])}
                for rel in doc.get("relations", []):
                    e1 = emap.get(rel["arg1_id"])
                    e2 = emap.get(rel["arg2_id"])
                    if not e1 or not e2:
                        continue
                    pt = civic_pair_type(e1["type"], e2["type"])
                    if pt not in FOCUS_PAIR_TYPES:
                        continue
                    gene_ent = e1 if normalize_entity_type(e1["type"]) == "gene" else e2
                    disease_ent = None
                    disease_neoplasm = None
                    if pt == "gene-disease":
                        disease_ent = e1 if normalize_entity_type(e1["type"]) == "disease" else e2
                        disease_neoplasm = _disease_is_neoplasm(disease_ent, ncit)
                    rows.append(
                        {
                            "corpus": corpus_key,
                            "split": split,
                            "pair_type": pt,
                            "pmid": pmid,
                            "gene_text": entity_surface(gene_ent),
                            "disease_text": entity_surface(disease_ent) if disease_ent else None,
                            "disease_neoplasm": disease_neoplasm,
                        }
                    )
    return pd.DataFrame(rows)


def fetch_pmid_mesh_index(pmids: list[str], force: bool = False) -> dict[str, Any]:
    cache_path = ONCOLOGY_PMID_MESH_JSON
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    ONCOLOGY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pmids = sorted({str(p) for p in pmids if str(p).isdigit()})
    index: dict[str, list[str]] = {}
    n_with_mesh = 0

    for i in range(0, len(pmids), EFETCH_BATCH):
        batch = pmids[i : i + EFETCH_BATCH]
        params = urllib.parse.urlencode(
            {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
        )
        req = urllib.request.Request(f"{EFETCH_URL}?{params}")
        with urllib.request.urlopen(req, timeout=120) as resp:
            root = ET.fromstring(resp.read())
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            if pmid_el is None or not pmid_el.text:
                continue
            pmid = pmid_el.text.strip()
            uis = sorted(
                {
                    el.attrib.get("UI", "")
                    for el in article.findall(".//MeshHeading/DescriptorName")
                    if el.attrib.get("UI", "").startswith("D")
                }
            )
            index[pmid] = uis
            if uis:
                n_with_mesh += 1
        print(f"  MeSH fetch batch {i // EFETCH_BATCH + 1}/{(len(pmids) + EFETCH_BATCH - 1) // EFETCH_BATCH}")
        time.sleep(EFETCH_INTERVAL_S)

    payload = {
        "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
        "source": EFETCH_URL,
        "n_pmids_requested": len(pmids),
        "n_pmids_with_mesh": n_with_mesh,
        "mesh_coverage_rate": round(n_with_mesh / max(len(pmids), 1), 4),
        "pmid_mesh": index,
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _summarize_fractions(classified: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    spec = [
        ("disease_neoplasm", "oncology_disease", "Disease maps to NCIt Neoplasm branch"),
        ("gene_civic", "oncology_gene", "Gene in CIViC gene set"),
        ("literature_mesh", "oncology_mesh", "Source article MeSH neoplasm descriptor"),
    ]
    for corpus in TRAINING_CORPORA:
        for pair_type in FOCUS_PAIR_TYPES:
            sub = classified[(classified["corpus"] == corpus) & (classified["pair_type"] == pair_type)]
            n_total = len(sub)
            for crit_key, col, _label in spec:
                if crit_key == "disease_neoplasm" and pair_type == "gene-drug":
                    rows.append(
                        {
                            "corpus": corpus,
                            "pair_type": pair_type,
                            "criterion": crit_key,
                            "n_total": n_total,
                            "n_oncology": 0,
                            "fraction": None,
                            "note": "No disease entity in gene-drug relations",
                        }
                    )
                    continue
                n_pos = int(sub[col].fillna(False).astype(bool).sum())
                rows.append(
                    {
                        "corpus": corpus,
                        "pair_type": pair_type,
                        "criterion": crit_key,
                        "n_total": n_total,
                        "n_oncology": n_pos,
                        "fraction": round(n_pos / n_total, 4) if n_total else None,
                        "note": "",
                    }
                )
    return pd.DataFrame(rows)


def _summarize_agreement(classified: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for corpus in TRAINING_CORPORA:
        for pair_type in FOCUS_PAIR_TYPES:
            sub = classified[(classified["corpus"] == corpus) & (classified["pair_type"] == pair_type)]
            n = len(sub)
            d = sub["oncology_disease"].fillna(False).astype(bool)
            g = sub["oncology_gene"].astype(bool)
            m = sub["oncology_mesh"].astype(bool)
            if pair_type == "gene-disease":
                all_three = d & g & m
            else:
                all_three = g & m
            rows.append(
                {
                    "corpus": corpus,
                    "pair_type": pair_type,
                    "n_total": n,
                    "n_all_three_criteria": int(all_three.sum()),
                    "fraction_all_three": round(float(all_three.mean()), 4) if n else None,
                    "n_disease_and_gene": int((d & g).sum()) if pair_type == "gene-disease" else None,
                    "n_disease_and_mesh": int((d & m).sum()) if pair_type == "gene-disease" else None,
                    "n_gene_and_mesh": int((g & m).sum()),
                    "n_disease_only": int((d & ~g & ~m).sum()) if pair_type == "gene-disease" else None,
                    "n_gene_only": int((~d & g & ~m).sum()) if pair_type == "gene-disease" else int((g & ~m).sum()),
                    "n_mesh_only": int((~d & ~g & m).sum()) if pair_type == "gene-disease" else int((~g & m).sum()),
                }
            )
    return pd.DataFrame(rows)


def run_oncology_subset(force_mesh: bool = False) -> dict[str, Any]:
    print("\n=== Oncology subset of training corpora ===")
    ncit = load_ncit_neoplasm_reference()
    civic = load_civic_gene_reference()
    print(f"  NCIt neoplasm codes: {ncit['n_ncit_neoplasm_codes']}")
    print(f"  MeSH neoplasm IDs (NCIt crosswalk): {ncit['n_mesh_neoplasm_ids']}")
    print(f"  CIViC genes: {civic['n_civic_genes']}")

    relations = _extract_training_relations(ncit)
    print(f"  Training relations (gene-drug + gene-disease): {len(relations)}")

    pmids = sorted(relations["pmid"].astype(str).unique().tolist())
    mesh_payload = fetch_pmid_mesh_index(pmids, force=force_mesh)
    pmid_mesh = mesh_payload["pmid_mesh"]
    print(
        f"  MeSH coverage: {mesh_payload['n_pmids_with_mesh']}/{mesh_payload['n_pmids_requested']} "
        f"({mesh_payload['mesh_coverage_rate']:.1%})"
    )

    civic_genes = civic["genes"]
    mesh_neoplasm = ncit["mesh_neoplasm_ids"]
    classified = relations.copy()
    classified["oncology_gene"] = classified["gene_text"].map(
        lambda t: _gene_symbol(t) in civic_genes or str(t).upper() in civic_genes
    )
    classified["oncology_disease"] = classified["disease_neoplasm"]
    classified["oncology_mesh"] = classified["pmid"].astype(str).map(
        lambda p: any(ui in mesh_neoplasm for ui in pmid_mesh.get(p, []))
    )
    classified["oncology_all_three"] = classified.apply(
        lambda r: bool(r["oncology_disease"] and r["oncology_gene"] and r["oncology_mesh"])
        if r["pair_type"] == "gene-disease"
        else bool(r["oncology_gene"] and r["oncology_mesh"]),
        axis=1,
    )

    fractions = _summarize_fractions(classified)
    agreement = _summarize_agreement(classified)
    fractions.to_csv(ONCOLOGY_FRACTIONS_CSV, index=False)
    agreement.to_csv(ONCOLOGY_AGREEMENT_CSV, index=False)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ncit": {k: v for k, v in ncit.items() if k not in ("ncit_codes", "ncit_terms", "mesh_neoplasm_ids")},
        "civic_genes": {k: v for k, v in civic.items() if k != "genes"},
        "mesh_fetch": {
            "timestamp": mesh_payload["fetch_timestamp"],
            "n_pmids_requested": mesh_payload["n_pmids_requested"],
            "n_pmids_with_mesh": mesh_payload["n_pmids_with_mesh"],
            "mesh_coverage_rate": mesh_payload["mesh_coverage_rate"],
        },
        "method_notes": [
            "Three criteria computed independently; no union headline.",
            "Disease criterion: BioRED disease entities carry MeSH IDs; MeSH IDs cross-walked to NCIt Neoplasm Core via NCI EVS mapping file (MSH source). Direct NCIt IDs on entities also accepted.",
            "Gene criterion: entity symbol matched against CIViC GENE features (CC0); COSMIC not used.",
            "Literature criterion: PubMed MeSH descriptors checked against MeSH neoplasm ID set derived from NCIt Neoplasm Core crosswalk.",
        ],
    }
    ONCOLOGY_METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== Oncology fractions (conservative intersection) ===")
    for _, row in agreement.iterrows():
        print(
            f"  {row['corpus']} {row['pair_type']}: "
            f"all-three={row['n_all_three_criteria']}/{row['n_total']} ({row['fraction_all_three']:.1%})"
        )
    print(fractions.to_string(index=False))

    return {
        "metadata": metadata,
        "fractions": fractions,
        "agreement": agreement,
        "classified": classified,
    }
