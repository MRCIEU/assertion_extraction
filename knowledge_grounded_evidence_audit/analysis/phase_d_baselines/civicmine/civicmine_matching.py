#!/usr/bin/env python3.11
"""Shared CIViCmine ↔ goldlite matching utilities (Phase 2A, unfiltered TSV).

Strict entity-pair rules mirror the Phase 2.0 probe: PMID must appear in
CIViCmine, then a row must match the gold entity pair using CIViCmine's
``gene_normalized`` / ``drug_normalized`` (gene_drug) or gene + variant +
cancer fields (variant_disease).

``kb_surface_pairs.jsonl`` stores ``head_text``/``tail_text`` (drug/gene order
for gene_drug); gold CSV supplies variant_text and cancer_scope for VD rows.
"""
from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[4]
PROC = REPO / "knowledge_grounded_evidence_audit" / "data" / "processed"
DEFAULT_GOLD = PROC / "goldlite_audit_targets.csv"
DEFAULT_CIVICMINE = Path(__file__).resolve().parent / "civicmine_unfiltered.tsv.gz"

EVALUABLE_DROP_LABEL = "VARIANT_GENE"


def load_kb_surface_rows() -> list[dict[str, Any]]:
    kb_path = (
        REPO / "fine_tuning_experiments" / "schema_exp" / "eval" / "inputs"
        / "kb_surface_pairs.jsonl"
    )
    return [json.loads(l) for l in kb_path.read_text().splitlines() if l.strip()]


def _u(s: str | None) -> str:
    return (s or "").strip().upper()


def _norm_hyphen(s: str | None) -> str:
    return _u(s).replace("–", "-").replace("—", "-")


def load_evaluable_gold_rows(path: Path = DEFAULT_GOLD) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, str]] = []
    for r in rows:
        if r.get("heuristic_gold_s2_label", "").strip() == EVALUABLE_DROP_LABEL:
            continue
        out.append(r)
    return out


def gold_by_target_id() -> dict[str, dict[str, str]]:
    return {r["goldlite_target_id"]: r for r in load_evaluable_gold_rows()}


def load_eval_kb_targets() -> list[dict[str, Any]]:
    return [
        r for r in load_kb_surface_rows()
        if r.get("expected_label") != EVALUABLE_DROP_LABEL
    ]


def enrich_target_for_civicmine(
    kb_row: dict[str, Any], gold: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Attach gene / drug / variant / cancer fields used for CIViCmine matching."""
    tid = kb_row["target_id"]
    g = gold or gold_by_target_id()[tid]
    fam = kb_row["pairing_family"].strip()
    base: dict[str, Any] = {
        "target_id": tid,
        "pmid": kb_row["pmid"].strip(),
        "pairing_family": fam,
        "expected_pairing_family": fam,
        "expected_label": kb_row.get("expected_label", ""),
        "cancer_scope": kb_row.get("cancer_scope") or g.get("cancer_scope", ""),
    }
    if fam == "gene_drug":
        base["gene"] = kb_row["tail_text"].strip()
        base["drug_primary"] = kb_row["head_text"].strip()
        base["variant_text"] = ""
    else:
        base["gene"] = g.get("gene", "").strip()
        base["drug_primary"] = g.get("drug_primary", "").strip()
        base["variant_text"] = g.get("variant_text", "").strip()
    return base


def load_enriched_eval_targets() -> list[dict[str, Any]]:
    gb = gold_by_target_id()
    return [enrich_target_for_civicmine(r, gb[r["target_id"]]) for r in load_eval_kb_targets()]


def iter_civicmine_rows(tsv_gz: Path) -> Iterator[dict[str, str]]:
    with gzip.open(tsv_gz, "rt", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            yield row


def index_civicmine_by_pmid(tsv_gz: Path) -> dict[str, list[dict[str, str]]]:
    by_pmid: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in iter_civicmine_rows(tsv_gz):
        pm = (row.get("pmid") or "").strip()
        if pm:
            by_pmid[pm].append(row)
    return by_pmid


def gene_hit(row: dict[str, str], gold_gene: str) -> bool:
    g = _u(gold_gene)
    if not g:
        return False
    gn = _norm_hyphen(row.get("gene_normalized"))
    gt = _norm_hyphen(row.get("gene_text"))
    gh = _u(row.get("gene_hugo_id"))
    if g == gn or g == gt:
        return True
    if g == gh:
        return True
    if gh.startswith("HGNC:"):
        return False
    return False


def drug_hit(row: dict[str, str], gold_drug: str) -> bool:
    d = _u(gold_drug)
    if not d:
        return False
    dn = _norm_hyphen(row.get("drug_normalized"))
    return bool(dn) and d == dn


def variant_token_hit(row: dict[str, str], gold_variant: str) -> bool:
    gv = _norm_hyphen(gold_variant)
    if not gv:
        return False
    vn = _norm_hyphen(row.get("variant_normalized"))
    vt = _norm_hyphen(row.get("variant_text"))
    if vn and (gv == vn or gv in vn or vn in gv):
        return True
    if vt and (gv in vt or vt in gv):
        return True
    toks = [t for t in gv.split() if len(t) > 2]
    blob = f"{vn} {vt}"
    if blob and toks and all(t in blob for t in toks if t != "MUTATION"):
        return True
    return False


def cancer_hit(row: dict[str, str], gold_scope: str) -> bool:
    cn = _u(row.get("cancer_normalized"))
    ct = _u(row.get("cancer_text"))
    bundle = f"{cn} {ct}"
    if not bundle.strip():
        return False
    if "NSCLC" in _u(gold_scope):
        return (
            "NSCLC" in bundle
            or "NON-SMALL" in bundle
            or "NON SMALL" in bundle
            or "LUNG ADENOCARCINOMA" in bundle
            or ("LUNG" in bundle and "ADENOCARCINOMA" in bundle)
            or ("LUNG" in bundle and "CARCINOMA" in bundle)
        )
    sc = _u(gold_scope).replace("_FAMILY", "")
    return bool(sc) and sc in bundle


def variant_mention_hit(row: dict[str, str], target: dict[str, Any]) -> bool:
    """Gene-linked variant / alteration mention (no disease requirement)."""
    if not gene_hit(row, str(target.get("gene", ""))):
        return False
    if variant_token_hit(row, str(target.get("variant_text", ""))):
        return True
    gv = _u(target.get("variant_text", ""))
    civic_v = (row.get("variant_normalized") or row.get("variant_text") or "").strip().lower()
    if not civic_v:
        return False
    if civic_v == "mutation" and (
        "MUTATION" in gv or "G12" in gv or "G13" in gv or "L858" in gv or "EXON" in gv
    ):
        return True
    if civic_v == "amplification" and "AMPLIFICATION" in gv:
        return True
    if civic_v == "overexpression" and "OVEREXPRESSION" in gv:
        return True
    return False


def variant_disease_strict(row: dict[str, str], target: dict[str, Any]) -> bool:
    """Variant–disease tuple match (same PMID).

    CIViCmine often normalises alleles to the generic token ``mutation`` while
    goldlite retains HGVS-like surfaces (e.g. ``KRAS G12``). We therefore
    accept (gene + cancer + ``mutation``) when the gold variant string clearly
    denotes a sequence alteration (``mutation``, ``G12``, ``amplification``,
    etc.) — mirroring the Phase 2.0 probe's 2/8 VD coverage.
    """
    return variant_mention_hit(row, target) and cancer_hit(
        row, str(target.get("cancer_scope", ""))
    )


def strict_pair_match(row: dict[str, str], target: dict[str, Any]) -> bool:
    fam = target.get("expected_pairing_family") or target.get("pairing_family") or ""
    fam = fam.strip()
    if fam == "gene_drug":
        return gene_hit(row, str(target.get("gene", ""))) and drug_hit(
            row, str(target.get("drug_primary", ""))
        )
    if fam == "variant_disease":
        return variant_disease_strict(row, target)
    return False


def civicmine_predicted_spair_label(evidencetype: str, pairing_family: str) -> str:
    et = _u(evidencetype)
    fam = pairing_family.strip()
    if fam == "gene_drug":
        if et == "PREDICTIVE":
            return "DRUG_GENE_REGULATION"
        return "ASSOCIATION_GENERAL"
    if fam == "variant_disease":
        if et == "PREDICTIVE":
            return "VARIANT_DISEASE"
        return "ASSOCIATION_GENERAL"
    return "ASSOCIATION_GENERAL"


def pair_slot_presence(
    rows: list[dict[str, str]], target: dict[str, Any]
) -> tuple[bool, bool]:
    """Whether each gold slot has at least one extracted mention (any row, same PMID)."""
    fam = (target.get("expected_pairing_family") or target.get("pairing_family") or "").strip()
    if fam == "gene_drug":
        g_any = any(gene_hit(row, str(target.get("gene", ""))) for row in rows)
        d_any = any(drug_hit(row, str(target.get("drug_primary", ""))) for row in rows)
        return (g_any, d_any)
    if fam == "variant_disease":
        v_any = any(variant_mention_hit(row, target) for row in rows)
        c_any = any(cancer_hit(row, str(target.get("cancer_scope", ""))) for row in rows)
        return (v_any, c_any)
    return (False, False)
