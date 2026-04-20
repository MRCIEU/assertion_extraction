#!/usr/bin/env python3.11
"""
Phase A-eval — prepare evaluation inputs (one-time).

Builds three cached JSONL files that every PA_* run will share:

  1. biored_test_pairs_{Sflat,Spair,Smech}.jsonl
     — Gold relations from t1_biored_test_{SCH}.jsonl plus same-document sampled
       negatives (ratio 2:1), textified into `{h_text} [ENT] {t_text} [SEP] {doc_text}`.
     — One file per schema (BioRED test labels are not identical across schemas).

  2. bc5cdr_test_pairs.jsonl
     — BC5CDR test docs (source_split=='test' from t1_bc5cdr.jsonl) gold + negatives.
     — Schema-agnostic: BC5CDR's only gold label is DRUG_DISEASE which is present in
       every schema's label set (just need to remap the __NEGATIVE__ class name,
       which is already `__NEGATIVE__` uniformly).

  3. kb_surface_pairs.jsonl
     — 165 CIViC goldlite targets, one row each (no negatives), text built from the
       target's head/tail surface strings and the PubMed abstract at primary_pmid.
     — All three schemas share this file; only the *interpretation* of the model's
       softmax differs (each classifier has its own N-way output head).

Outputs land in:
  $FT_ROOT/fine_tuning_experiments/schema_exp/eval/inputs/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FT_DATA_ROOT = Path(os.environ.get(
    "PROJECT_1_DATA_ROOT",
    "/lus/lfs1aip2/projects/b5ac/project_1",
)).resolve()

PROC = FT_DATA_ROOT / "training_data_generation" / "data" / "processed"
KB_ROOT = FT_DATA_ROOT / "knowledge_grounded_evidence_audit" / "data" / "processed"
GOLDLITE = KB_ROOT / "goldlite_audit_targets.csv"
PUBMED_CACHE = KB_ROOT / "pubmed_cache"

OUT_DIR = SCRIPT_DIR / "inputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCHEMAS = ("Sflat", "Spair", "Smech")
NEG_LABEL = "__NEGATIVE__"

# Cancer-scope to surface text mapping (consistent with how BioRED/BC5CDR
# annotate disease mentions). When the exact surface is not resolvable we fall
# back to the scope token itself and let the document context carry semantics.
CANCER_SCOPE_SURFACE = {
    "NSCLC_family": "non-small cell lung cancer",
    "breast_family": "breast cancer",
    "melanoma_family": "melanoma",
    "colorectal_family": "colorectal cancer",
    "pancreatic_family": "pancreatic cancer",
    "prostate_family": "prostate cancer",
    "ovarian_family": "ovarian cancer",
    "leukemia_family": "leukemia",
    "lymphoma_family": "lymphoma",
    "lung_family": "lung cancer",
    "glioblastoma_family": "glioblastoma",
    "renal_family": "renal cell carcinoma",
    "thyroid_family": "thyroid cancer",
    "gastric_family": "gastric cancer",
    "hepatocellular_family": "hepatocellular carcinoma",
    "SOLID_TUMOR": "solid tumor",
}


# ───────────────────────────────────────────────────────────────────
# BioRED / BC5CDR test pair construction
# ───────────────────────────────────────────────────────────────────

def _make_text(h_text: str, t_text: str, doc_text: str, max_chars: int = 8000) -> str:
    return f"{h_text} [ENT] {t_text} [SEP] {doc_text}"[:max_chars]


def doc_to_gold_rows(doc: dict) -> list[dict]:
    ent_by_id = {e["entity_id"]: e for e in (doc.get("entities") or [])}
    id_list = list(ent_by_id.keys())
    gold_pairs: set[tuple[str, str]] = set()
    for rel in doc.get("relations") or []:
        hid, tid = rel.get("head_entity_id"), rel.get("tail_entity_id")
        if hid and tid:
            gold_pairs.add((hid, tid))
            gold_pairs.add((tid, hid))

    text = doc.get("text") or ""
    rows = []
    for rel in doc.get("relations") or []:
        if not rel.get("is_gold_supervision", True):
            continue
        hid, tid = rel.get("head_entity_id"), rel.get("tail_entity_id")
        if not (hid and tid and hid in ent_by_id and tid in ent_by_id):
            continue
        h, t = ent_by_id[hid], ent_by_id[tid]
        rows.append({
            "text": _make_text(h["text"], t["text"], text),
            "label": rel["mapped_label"],
            "doc_id": doc.get("doc_id", ""),
            "sample_id": doc.get("sample_id", ""),
            "head_entity_label": h.get("mapped_label", ""),
            "tail_entity_label": t.get("mapped_label", ""),
            "relation_family": rel.get("relation_family"),
            "source_dataset": doc.get("source_dataset"),
            "supervision": "gold",
            "_ent_by_id": ent_by_id,
            "_id_list": id_list,
            "_gold_pairs": gold_pairs,
            "_doc_text": text,
        })
    return rows


def add_sampled_negatives(
    gold_rows: list[dict], rng: random.Random, *, ratio: float = 2.0,
) -> list[dict]:
    """Match trainer protocol: for each gold row, draw `ratio` random non-gold
    same-document pairs and label them __NEGATIVE__. Deterministic via seeded rng."""
    n_neg_per = max(1, int(ratio))
    out = []
    for row in gold_rows:
        out.append({k: v for k, v in row.items() if not k.startswith("_")})
        ent_by_id = row["_ent_by_id"]
        id_list = row["_id_list"]
        gold_pairs = row["_gold_pairs"]
        doc_text = row["_doc_text"]
        if len(id_list) < 2:
            continue
        added = 0
        for _ in range(256 * n_neg_per):
            if added >= n_neg_per:
                break
            h2 = rng.choice(id_list)
            t2 = rng.choice(id_list)
            if h2 == t2 or (h2, t2) in gold_pairs:
                continue
            h_e, t_e = ent_by_id[h2], ent_by_id[t2]
            out.append({
                "text": _make_text(h_e["text"], t_e["text"], doc_text),
                "label": NEG_LABEL,
                "doc_id": row["doc_id"],
                "sample_id": row["sample_id"],
                "head_entity_label": h_e.get("mapped_label", ""),
                "tail_entity_label": t_e.get("mapped_label", ""),
                "relation_family": None,
                "source_dataset": row["source_dataset"],
                "supervision": "negative_sample",
            })
            added += 1
    return out


def build_biored_test(schema: str, out_path: Path, seed: int = 20260416) -> dict:
    src = PROC / f"t1_biored_test_{schema}.jsonl"
    assert src.exists(), f"Missing {src}"
    rng = random.Random(seed)
    all_rows: list[dict] = []
    n_docs = 0
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        n_docs += 1
        gold = doc_to_gold_rows(doc)
        with_neg = add_sampled_negatives(gold, rng, ratio=2.0)
        all_rows.extend(with_neg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lab_counts = Counter(r["label"] for r in all_rows)
    return {
        "schema": schema,
        "source": str(src),
        "output": str(out_path),
        "n_docs": n_docs,
        "n_rows": len(all_rows),
        "label_counts": dict(lab_counts),
    }


def build_bc5cdr_test(out_path: Path, seed: int = 20260416) -> dict:
    """BC5CDR test pairs (schema-agnostic: only DRUG_DISEASE vs __NEGATIVE__)."""
    src = PROC / "t1_bc5cdr.jsonl"
    assert src.exists(), f"Missing {src}"
    rng = random.Random(seed)
    all_rows: list[dict] = []
    n_docs = 0
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        if doc.get("source_split") != "test":
            continue
        n_docs += 1
        gold = doc_to_gold_rows(doc)
        with_neg = add_sampled_negatives(gold, rng, ratio=2.0)
        all_rows.extend(with_neg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lab_counts = Counter(r["label"] for r in all_rows)
    return {
        "schema": "schema_agnostic",
        "source": str(src),
        "output": str(out_path),
        "n_docs": n_docs,
        "n_rows": len(all_rows),
        "label_counts": dict(lab_counts),
    }


# ───────────────────────────────────────────────────────────────────
# KB surface pairs
# ───────────────────────────────────────────────────────────────────

def extract_abstract(pmid: str) -> str:
    """Parse PubMed XML from pubmed_cache, return title + abstract concatenated."""
    path = PUBMED_CACHE / f"{pmid}.xml"
    if not path.exists():
        return ""
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return ""
    root = tree.getroot()
    parts: list[str] = []
    for title in root.iter("ArticleTitle"):
        t = "".join(title.itertext()).strip()
        if t:
            parts.append(t)
    for abst in root.iter("AbstractText"):
        t = "".join(abst.itertext()).strip()
        if t:
            parts.append(t)
    return " ".join(parts)


def _pick_head_tail(row: dict) -> tuple[str, str, str]:
    """For CIViC target → (head_text, tail_text, pairing_family).

    Convention:
      gene_drug       : head = drug, tail = gene
      variant_disease : head = variant, tail = disease (cancer_scope)
    """
    fam = row.get("expected_pairing_family") or ""
    gene = (row.get("gene") or "").strip()
    drug = (row.get("drug_primary") or "").strip()
    variant = (row.get("variant_text") or "").strip() or gene
    scope = (row.get("cancer_scope") or "").strip()
    disease = CANCER_SCOPE_SURFACE.get(scope, scope.replace("_family", "").replace("_", " ").strip() or "cancer")

    if fam == "gene_drug":
        return drug or gene, gene or drug, fam
    if fam == "variant_disease":
        return variant, disease, fam
    # Fallback (none expected)
    return variant or gene, disease or drug, fam or "unknown"


def build_kb_surface_pairs(out_path: Path) -> dict:
    rows = list(csv.DictReader(open(GOLDLITE)))
    out_rows: list[dict] = []
    n_missing_abs = 0
    for r in rows:
        pmid = (r.get("primary_pmid") or "").strip()
        abstract = extract_abstract(pmid)
        if not abstract:
            n_missing_abs += 1
        h_text, t_text, fam = _pick_head_tail(r)
        expected = r.get("heuristic_gold_s2_label") or ""
        out_rows.append({
            "target_id": r.get("goldlite_target_id"),
            "pmid": pmid,
            "pairing_family": fam,
            "head_text": h_text,
            "tail_text": t_text,
            "expected_label": expected,
            "cancer_scope": r.get("cancer_scope"),
            "text": _make_text(h_text, t_text, abstract),
            "abstract_found": bool(abstract),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "source": str(GOLDLITE),
        "pubmed_cache": str(PUBMED_CACHE),
        "output": str(out_path),
        "n_targets": len(out_rows),
        "n_missing_abstract": n_missing_abs,
        "expected_label_counts": dict(Counter(r["expected_label"] for r in out_rows)),
        "pairing_family_counts": dict(Counter(r["pairing_family"] for r in out_rows)),
    }


# ───────────────────────────────────────────────────────────────────
# Driver
# ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260416)
    args = ap.parse_args()

    manifest: dict[str, Any] = {"outputs": {}}

    for sch in SCHEMAS:
        out = OUT_DIR / f"biored_test_pairs_{sch}.jsonl"
        info = build_biored_test(sch, out, seed=args.seed)
        manifest["outputs"][f"biored_test_{sch}"] = info
        print(f"[biored/{sch}] docs={info['n_docs']} rows={info['n_rows']} "
              f"labels={info['label_counts']}")

    out_bc = OUT_DIR / "bc5cdr_test_pairs.jsonl"
    info_bc = build_bc5cdr_test(out_bc, seed=args.seed)
    manifest["outputs"]["bc5cdr_test"] = info_bc
    print(f"[bc5cdr]    docs={info_bc['n_docs']} rows={info_bc['n_rows']} "
          f"labels={info_bc['label_counts']}")

    out_kb = OUT_DIR / "kb_surface_pairs.jsonl"
    info_kb = build_kb_surface_pairs(out_kb)
    manifest["outputs"]["kb_surface"] = info_kb
    print(f"[kb]        targets={info_kb['n_targets']} "
          f"missing_abs={info_kb['n_missing_abstract']}")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest → {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
