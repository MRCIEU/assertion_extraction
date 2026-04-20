#!/usr/bin/env python3
"""
Knowledge-grounded oncology evidence audit pipeline.

Audits KB files, builds scoped ledgers, retrieves PubMed XML, runs **checkpoint
relation classification** (default) or optional placeholder co-occurrence mode
for debugging only, then KB linkage and evidence-audit tables.

**Code lives only under this directory.** Writable outputs default to
``~/projects/project_1/knowledge_grounded_evidence_audit`` (``KG_AUDIT_OUTPUT_ROOT``).

**Execution policy:** Do **not** run this script on a login node for real jobs.
Submit a **GPU Slurm** job, e.g. ``sbatch scripts/run_kg_audit_gpu.sbatch``.
Local execution is blocked unless ``KG_AUDIT_ALLOW_LOCAL=1`` (debug only).

All comments and logs are in English.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from inference.checkpoint_assertions import extract_neural_pair_assertions
from inference.predict_checkpoint import load_model_from_checkpoint


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
# Inference package (local only): inference.*
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_1_ROOT", str(DEFAULT_PROJECT_ROOT))).resolve()

RAW_CIVIC = PROJECT_ROOT / "data" / "raw" / "civic"
RAW_ONCOKB = PROJECT_ROOT / "data" / "raw" / "oncoKB"

_DEFAULT_OUTPUT_ROOT = Path.home() / "projects" / "project_1" / "knowledge_grounded_evidence_audit"
OUT_ROOT = Path(os.environ.get("KG_AUDIT_OUTPUT_ROOT", str(_DEFAULT_OUTPUT_ROOT))).resolve()
MANIFESTS = OUT_ROOT / "manifests"
PROC = OUT_ROOT / "data" / "processed"
CACHE = PROC / "pubmed_cache"
ASSERTIONS_DIR = PROC / "assertions"
REPORTS = OUT_ROOT / "reports"

PANEL_GENES = ("EGFR", "ALK", "KRAS", "BRAF", "ROS1", "MET")
LUNG_PAT = re.compile(
    r"Lung|NSCLC|non[- ]small|Pulmonary|Adenocarcinoma|Bronch",
    re.I,
)


# Model calibration: pairing_family -> multiplicative weight on confidence;
# min_confidence gates retention. Informed by decision_analysis pairing story,
# not neural outputs.
MODEL_CALIBRATION: Dict[str, Dict[str, Any]] = {
    "M015": {
        "min_confidence": 0.42,
        "pairing_weight": {
            "variant_disease": 1.0,
            "gene_disease": 1.05,
            "drug_gene": 1.0,
            "drug_disease": 0.98,
        },
        "global_recall_bias": 1.0,
    },
    "M021": {
        "min_confidence": 0.38,
        "pairing_weight": {
            "variant_disease": 1.28,
            "gene_disease": 1.02,
            "drug_gene": 0.92,
            "drug_disease": 0.95,
        },
        "global_recall_bias": 1.05,
    },
    "M003": {
        "min_confidence": 0.44,
        "pairing_weight": {
            "variant_disease": 0.95,
            "gene_disease": 1.08,
            "drug_gene": 1.02,
            "drug_disease": 1.06,
        },
        "global_recall_bias": 0.97,
    },
    "S002": {
        "min_confidence": 0.36,
        "pairing_weight": {
            "variant_disease": 1.05,
            "gene_disease": 1.0,
            "drug_gene": 1.12,
            "drug_disease": 1.0,
        },
        "global_recall_bias": 1.1,
        "note": "Weighted-CE branch proxy: more drug–gene hypotheses retained.",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Part 1 — Resource audit
# ---------------------------------------------------------------------------


def scan_raw_kb_dirs() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source_name, base in (("civic", RAW_CIVIC), ("oncokb", RAW_ONCOKB)):
        if not base.is_dir():
            continue
        for p in sorted(base.iterdir()):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            fmt = {
                ".tsv": "tsv",
                ".csv": "csv",
                ".xlsx": "xlsx",
                ".json": "json",
                ".jsonl": "jsonl",
            }.get(suf, "unknown")
            rows.append(
                {
                    "filepath": str(p.resolve()),
                    "filename": p.name,
                    "file_format": fmt,
                    "size_bytes": p.stat().st_size,
                    "kb_source_folder": source_name,
                }
            )
    return rows


def resource_semantics(filename: str, kb: str) -> Tuple[str, str, str, str, str, str]:
    """Returns inferred_type, semantic_role, usable(yes/no/conditional), role, why_not, caveat."""
    fn = filename.lower()
    caveats = ""
    if kb == "civic":
        if "acceptedclinical" in fn and "submitted" not in fn:
            return (
                "civic_nightly_export",
                "clinical_evidence_summaries;publisher_assertions;PMID-linked",
                "yes",
                "primary_evidence_summaries_track_A_pmids",
                "",
                "Accepted only; use as anchor + literature linkage, not patient truth.",
            )
        if "acceptedandsubmittedclinical" in fn:
            return (
                "civic_nightly_export",
                "evidence_mixed_submitted",
                "conditional",
                "secondary_or_qa_only",
                "",
                "Submitted rows add noise; not primary ledger without review.",
            )
        if "acceptedassertion" in fn and "submitted" not in fn:
            return (
                "civic_nightly_export",
                "assertion_summaries",
                "conditional",
                "assertion_level_metadata",
                "",
                "Cross-check against clinical evidence file for PMID linkage.",
            )
        if "variant" in fn or "molecular" in fn or "variantgroup" in fn:
            return (
                "civic_nightly_export",
                "variant_profile_metadata",
                "yes",
                "normalization_and_display_strings",
                "",
                "",
            )
        if "feature" in fn:
            return (
                "civic_nightly_export",
                "genomic_feature_metadata",
                "conditional",
                "supporting_metadata",
                "",
                "Use for contextualization; not primary assertion table.",
            )
        return ("civic_nightly_export", "unknown_civic_table", "conditional", "", "", "Review columns before use.")
    # oncokb
    if fn == "oncokb_biomarker_drug_associations.tsv":
        return (
            "oncokb_tabular_export",
            "gene_alteration_cancer_drug_implication",
            "yes",
            "therapeutic_implication_anchor_rows",
            "",
            "Levels 1–4 are OncoKB curation, not literature-automatic labels.",
        )
    if re.match(r"oncokb_biomarker_drug_associations-\d+\.tsv$", filename.lower()):
        return (
            "oncokb_tabular_export",
            "duplicate_or_partial_slice",
            "conditional",
            "",
            "",
            "Compare checksum/size against primary TSV to avoid redundant merges.",
        )
    if "fda_approved" in fn and fn.endswith(".xlsx"):
        return (
            "oncokb_supplement",
            "fda_oncology_therapy_reference",
            "conditional",
            "therapy_vocab_enrichment",
            "",
            "Requires openpyxl; optional for drug string harmonization.",
        )
    return ("oncokb_unknown", "unknown", "conditional", "", "", "Inspect manually.")


def write_resource_audit(rows: List[Dict[str, Any]]) -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    enriched = []
    for r in rows:
        it, sr, us, role, why, cc = resource_semantics(r["filename"], r["kb_source_folder"])
        enriched.append({**r, "inferred_resource_type": it, "probable_semantic_role": sr, "usable_for_this_subproject": us, "role_if_yes": role, "why_not_if_no": why, "conditional_caveat": cc})

    inv_csv = MANIFESTS / "kb_resource_inventory.csv"
    fields = [
        "filepath",
        "filename",
        "file_format",
        "size_bytes",
        "inferred_resource_type",
        "probable_semantic_role",
        "usable_for_this_subproject",
        "role_if_yes",
        "why_not_if_no",
        "conditional_caveat",
        "kb_source_folder",
    ]
    with open(inv_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in enriched:
            w.writerow({k: e.get(k, "") for k in fields})

    with open(MANIFESTS / "kb_resource_inventory.json", "w", encoding="utf-8") as f:
        json.dump({"generated_utc": utc_now(), "files": enriched}, f, indent=2)

    # Usage decisions (one row per file)
    ud = MANIFESTS / "kb_resource_usage_decision.csv"
    with open(ud, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "decision", "primary_use", "avoid_or_limit"])
        for e in enriched:
            decision = "USE_PRIMARY" if e["usable_for_this_subproject"] == "yes" else (
                "USE_CONDITIONAL" if e["usable_for_this_subproject"] == "conditional" else "DO_NOT_USE"
            )
            avoid = ""
            if "duplicate" in e["inferred_resource_type"]:
                avoid = "Redundant export; prefer primary TSV."
            if "submitted" in e["filename"].lower() and "clinical" in e["filename"].lower():
                avoid = "Submitted evidence mixed; ledger uses accepted-only."
            w.writerow([e["filename"], decision, e["role_if_yes"] or e["probable_semantic_role"], avoid])

    rationale = MANIFESTS / "kb_resource_usage_rationale.md"
    rationale.write_text(
        """# KB resource usage rationale

## Principles

- **CIViC** and **OncoKB** are **external clinical knowledge anchors**, not interchangeable gold labels for automatic extraction.
- **Accepted-only** CIViC clinical evidence summaries are the **primary** table for **PMID-linked** Track A retrieval.
- **OncoKB** biomarker–drug export supplies **curated therapeutic implication rows** without per-row PMIDs in this file; rows **harmonize** with CIViC on gene / cancer / drug where semantics align.
- **Duplicate** `oncokb_biomarker_drug_associations-N.tsv` slices are **not** merged blindly — use the **primary** `oncokb_biomarker_drug_associations.tsv` unless a checksum audit shows substantive differences.
- **FDA oncology therapies XLSX** is **optional** enrichment for drug name normalization; not required for the minimal audit path.
- **Submitted** CIViC clinical evidence is **out of scope** for the canonical ledger to limit noise.

## Files not used as primary evidence anchors

- `nightly-AcceptedAndSubmittedClinicalEvidenceSummaries.tsv` — too large; mixes submitted; use accepted-only file instead.
- OncoKB `*-2.tsv` … `*-5.tsv` — treat as **conditional** duplicates pending diff review.

---
*Auto-generated skeleton; see `kb_resource_usage_decision.csv` for per-file flags.*
""",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Scope artifacts
# ---------------------------------------------------------------------------


def write_scope_files() -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    with open(MANIFESTS / "scope_option_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope_id", "description", "kb_coverage", "retrievability", "heterogeneity_risk", "audit_tractability", "verdict"])
        w.writerow([
            "nsclc_precision_panel",
            "NSCLC + six-gene precision panel (EGFR, ALK, KRAS, BRAF, ROS1, MET)",
            "High in both CIViC and OncoKB for NSCLC rows",
            "Rich PMID coverage via CIViC; OncoKB adds implication levels",
            "Moderate — bounded histology mix within NSCLC family",
            "Strong — explicit harmonization and PMID traceability",
            "SELECTED",
        ])
        w.writerow([
            "pan_lung_broad",
            "All lung cancer histologies without tight gene panel",
            "Uneven — many non-panel CIViC diseases",
            "Noisier queries; harder PMID–anchor pairing",
            "High — small cell vs NSCLC vs metastatic patterns",
            "Weaker for controlled audit",
            "REJECTED_FOR_THIS_PASS",
        ])

    scope = {
        "chosen_scope_id": "nsclc_precision_panel",
        "target_cancer_focus": "Non–small cell lung cancer family (NSCLC / lung adenocarcinoma patterns)",
        "target_genes": list(PANEL_GENES),
        "target_variants": "Variants and fusion patterns appearing in CIViC molecular_profile or OncoKB Alterations for panel genes within scope",
        "target_drugs": "Therapies listed in CIViC therapies column or OncoKB Drugs column for scoped rows",
        "target_assertion_families": [
            "CIViC Predictive",
            "CIViC Diagnostic",
            "CIViC Prognostic",
            "OncoKB therapeutic implication levels",
        ],
        "exclusion_rules": [
            "Exclude CIViC rows outside panel genes unless clearly lung-involved and manually reviewed (automatic filter uses panel genes + lung disease pattern)",
            "Exclude non-lung cancers from OncoKB ledger rows",
            "Do not use submitted-only CIViC export as primary ledger",
        ],
        "justification_chosen": (
            "Panel scope maximizes cross-KB overlap, keeps PubMed retrieval bounded, and preserves "
            "clinically meaningful therapy/biomarker contrasts without pan-cancer heterogeneity."
        ),
        "updated_utc": utc_now(),
    }
    with open(MANIFESTS / "scope_definition.json", "w", encoding="utf-8") as f:
        json.dump(scope, f, indent=2)

    (MANIFESTS / "scope_definition.md").write_text(
        f"""# Clinical scope definition

## Chosen scope: `{scope["chosen_scope_id"]}`

**Cancer focus:** {scope["target_cancer_focus"]}

**Panel genes:** {", ".join(PANEL_GENES)}

**Assertion families:** {", ".join(scope["target_assertion_families"])}

**Exclusions:** 
{chr(10).join("- " + x for x in scope["exclusion_rules"])}

## Why not pan-lung unbounded

Broad lung scope increases histology and staging heterogeneity with modest audit benefit for this protocol. The precision panel aligns with **standard precision-oncology biomarkers** well covered in both CIViC and OncoKB NSCLC exports.

---
*See `scope_option_comparison.csv`.*
""",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Part 2 — Ledgers
# ---------------------------------------------------------------------------


def civic_panel_gene(molecular_profile: str) -> Optional[str]:
    """Return first matching panel gene in standard oncology mention order."""
    u = (molecular_profile or "").upper()
    if "EML4" in u and "ALK" in u:
        return "ALK"
    priority = ("EGFR", "ALK", "KRAS", "BRAF", "ROS1", "MET")
    for g in priority:
        if re.search(rf"\b{re.escape(g)}\b", u):
            return g
    return None


def parse_civic_ledger() -> List[Dict[str, str]]:
    path = RAW_CIVIC / "nightly-AcceptedClinicalEvidenceSummaries.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    out: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            disease = row.get("disease") or ""
            if not LUNG_PAT.search(disease):
                continue
            mp = row.get("molecular_profile") or ""
            gene = civic_panel_gene(mp)
            if gene is None:
                continue
            evid = row.get("evidence_id") or ""
            pmid = (row.get("citation_id") or "").strip()
            if row.get("source_type", "").strip().lower() != "pubmed":
                continue
            if not pmid.isdigit():
                continue
            out.append(
                {
                    "source_kb": "CIViC",
                    "record_id": f"CIViC_EVID_{evid}",
                    "gene": gene,
                    "variant": mp.strip(),
                    "cancer_disease": disease.strip(),
                    "drug_therapy": (row.get("therapies") or "").strip(),
                    "evidence_category": (row.get("evidence_type") or "").strip(),
                    "evidence_level": (row.get("evidence_level") or "").strip(),
                    "normalized_identifiers": f"DOID:{row.get('doid','').strip()}",
                    "source_labels": mp.strip(),
                    "pmid": pmid,
                    "evidence_direction": (row.get("evidence_direction") or "").strip(),
                    "significance": (row.get("significance") or "").strip(),
                    "notes": "CIViC accepted clinical evidence summary; anchor not patient-level truth.",
                }
            )
    return out


def parse_oncokb_ledger() -> List[Dict[str, str]]:
    path = RAW_ONCOKB / "oncokb_biomarker_drug_associations.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    out: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader, start=2):
            cancer = row.get("Cancer Types") or ""
            if "Non-Small Cell Lung" not in cancer and "lung" not in cancer.lower():
                continue
            gene = (row.get("Gene") or "").strip().upper()
            if gene not in PANEL_GENES:
                continue
            drug = (row.get("Drugs (for therapeutic implications only)") or "").strip()
            alt = (row.get("Alterations") or "").strip()
            lvl = (row.get("Level") or "").strip()
            out.append(
                {
                    "source_kb": "OncoKB",
                    "record_id": f"OncoKB_ROW_{i}",
                    "gene": gene,
                    "variant": alt,
                    "cancer_disease": cancer.strip(),
                    "drug_therapy": drug,
                    "evidence_category": "therapeutic_implication",
                    "evidence_level": lvl,
                    "normalized_identifiers": "",
                    "source_labels": alt,
                    "pmid": "",
                    "evidence_direction": "",
                    "significance": "",
                    "notes": "OncoKB curated implication level; anchor not automatic literature proof.",
                }
            )
    return out


def norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def harmonize_ledgers(civic: List[Dict[str, str]], oncokb: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    harm: List[Dict[str, str]] = []
    audit: List[Dict[str, str]] = []
    rules = {
        "version": 1,
        "harmonization_goals": "Align gene + therapy + coarse alteration for audit targeting without erasing source semantics.",
        "matching": {
            "L1_coordinate": "Same gene (symbol), drug name overlap after normalization, both NSCLC-family cancer strings.",
            "L2_partial": "Same gene and (drug overlap OR alteration token overlap) with cancer compatible.",
        },
        "non_goals": ["Do not assert clinical equivalence of CIViC evidence_level to OncoKB Level.", "Do not merge conflicting significance."],
        "updated_utc": utc_now(),
    }
    (PROC / "kb_harmonization_rules.json").parent.mkdir(parents=True, exist_ok=True)
    with open(PROC / "kb_harmonization_rules.json", "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)

    def drug_tokens(d: str) -> Set[str]:
        parts = re.split(r"[,/+]| and ", d)
        return {norm_token(p) for p in parts if len(norm_token(p)) > 2}

    used_onco = set()
    for cr in civic:
        cg = cr["gene"]
        cd = cr["drug_therapy"]
        cdtoks = drug_tokens(cd) if cd else set()
        best: Optional[Tuple[int, Dict[str, str]]] = None
        for orow in oncokb:
            if orow["gene"] != cg:
                continue
            od = orow["drug_therapy"]
            odtoks = drug_tokens(od) if od else set()
            overlap = len(cdtoks & odtoks) if cdtoks and odtoks else 0
            if overlap > 0:
                score = 3 + overlap
            elif cdtoks and not odtoks:
                score = 1
            elif norm_token(cr["variant"])[:12] == norm_token(orow["variant"])[:12]:
                score = 2
            else:
                continue
            if best is None or score > best[0]:
                best = (score, orow)
        key = hashlib.sha1(
            "|".join([cr["record_id"], best[1]["record_id"] if best else ""]).encode()
        ).hexdigest()[:12]
        prov = "CIViC_only"
        matched_onco = ""
        harmon_reason = ""
        amb_flags = ""
        if best:
            prov = "CIViC_plus_OncoKB"
            matched_onco = best[1]["record_id"]
            harmon_reason = "gene_match_plus_drug_or_variant_overlap"
            used_onco.add(matched_onco)
            if cr["evidence_level"] and best[1]["evidence_level"]:
                if cr["evidence_level"][0] in ("A", "B") and best[1]["evidence_level"] in ("3", "4"):
                    amb_flags = "evidence_taxonomies_differ"
        else:
            harmon_reason = "no_oncokb_row_meeting_overlap_rule"

        harm.append(
            {
                "harmonized_key": f"H_{key}",
                "provenance": prov,
                "civic_record_id": cr["record_id"],
                "oncokb_record_id": matched_onco,
                "gene": cg,
                "variant_civic": cr["variant"],
                "variant_oncokb": best[1]["variant"] if best else "",
                "drug_therapy": cr["drug_therapy"] or (best[1]["drug_therapy"] if best else ""),
                "cancer_scope": "NSCLC_family",
                "assertion_family_civic": cr["evidence_category"],
                "oncokb_level": best[1]["evidence_level"] if best else "",
                "pmid_track_a": cr["pmid"],
                "harmonization_basis": harmon_reason,
                "ambiguity_flags": amb_flags,
                "anchor_disclaimer": "knowledge_anchor_space_not_patient_truth",
            }
        )
        audit.append(
            {
                "civic_record_id": cr["record_id"],
                "oncokb_record_id": matched_onco,
                "match_tier_description": harmon_reason,
                "ambiguity": amb_flags,
            }
        )

    for orow in oncokb:
        if_orow_id = orow["record_id"]
        if if_orow_id in used_onco:
            continue
        key = hashlib.sha1(orow["record_id"].encode()).hexdigest()[:12]
        harm.append(
            {
                "harmonized_key": f"H_ONCO_{key}",
                "provenance": "OncoKB_only",
                "civic_record_id": "",
                "oncokb_record_id": orow["record_id"],
                "gene": orow["gene"],
                "variant_civic": "",
                "variant_oncokb": orow["variant"],
                "drug_therapy": orow["drug_therapy"],
                "cancer_scope": "NSCLC_family",
                "assertion_family_civic": "",
                "oncokb_level": orow["evidence_level"],
                "pmid_track_a": "",
                "harmonization_basis": "oncokb_row_without_civic_PMID_partner",
                "ambiguity_flags": "no_track_A_pmid_in_oncokb_file",
                "anchor_disclaimer": "knowledge_anchor_space_not_patient_truth",
            }
        )
        audit.append(
            {
                "civic_record_id": "",
                "oncokb_record_id": orow["record_id"],
                "match_tier_description": "oncokb_only_no_auto_civic_pair",
                "ambiguity": "literature_linkage_requires_separate_retrieval",
            }
        )

    return harm, audit


def write_ledgers(civic: List[Dict[str, str]], oncokb: List[Dict[str, str]], harm: List[Dict[str, str]], audit: List[Dict[str, str]]) -> None:
    PROC.mkdir(parents=True, exist_ok=True)

    def dump(path: Path, flds: Sequence[str], data: Iterable[Dict[str, str]]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(flds))
            w.writeheader()
            for row in data:
                w.writerow({k: row.get(k, "") for k in flds})

    cf = tuple(civic[0].keys()) if civic else ()
    of = tuple(oncokb[0].keys()) if oncokb else ()
    dump(PROC / "civic_target_ledger.csv", cf, civic)
    dump(PROC / "oncokb_target_ledger.csv", of, oncokb)
    hf = tuple(harm[0].keys()) if harm else ()
    dump(PROC / "kb_target_ledger_harmonized.csv", hf, harm)
    af = ("civic_record_id", "oncokb_record_id", "match_tier_description", "ambiguity")
    dump(PROC / "kb_harmonization_audit.csv", af, audit)


# ---------------------------------------------------------------------------
# PubMed
# ---------------------------------------------------------------------------


def fetch_pubmed_xml(pmid: str, email: str, cache_dir: Path) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_dir / f"{pmid}.xml"
    if cpath.exists():
        return cpath.read_text(encoding="utf-8", errors="replace")
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        f"db=pubmed&id={pmid}&retmode=xml&tool=kg_evidence_audit&email={urllib.parse.quote(email)}"
    )
    time.sleep(0.35)
    req = urllib.request.Request(url, headers={"User-Agent": f"kg_evidence_audit ({email})"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read().decode("utf-8", errors="replace")
    cpath.write_text(data, encoding="utf-8")
    return data


def parse_pubmed_article(xml_text: str) -> Tuple[str, str]:
    root = ET.fromstring(xml_text)
    title_el = root.find(".//ArticleTitle")
    title = "".join(title_el.itertext()) if title_el is not None else ""
    abstract_parts: List[str] = []
    for at in root.findall(".//AbstractText"):
        label = at.attrib.get("Label", "")
        txt = "".join(at.itertext())
        abstract_parts.append(f"{label}: {txt}".strip() if label else txt)
    abstract = " ".join(abstract_parts).strip()
    return title, abstract


# ---------------------------------------------------------------------------
# Normalization & pairs
# ---------------------------------------------------------------------------


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def collect_lexicons(harm: List[Dict[str, str]]) -> Tuple[Set[str], Set[str], Set[str]]:
    genes: Set[str] = set()
    drugs: Set[str] = set()
    variants: Set[str] = set()
    for h in harm:
        genes.add(h["gene"].upper())
        for part in re.split(r"[,/+]| and ", h.get("drug_therapy", "")):
            p = part.strip()
            if len(p) > 2:
                drugs.add(p.lower())
        for v in (h.get("variant_civic"), h.get("variant_oncokb")):
            if v:
                for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9+\-]*", v):
                    if len(tok) >= 3:
                        variants.add(tok.upper())
    return genes, drugs, variants


def find_genes(sentence: str, genes: Set[str]) -> List[Tuple[int, int, str]]:
    found = []
    for g in genes:
        for m in re.finditer(rf"\b{re.escape(g)}\b", sentence, re.I):
            found.append((m.start(), m.end(), g.upper()))
    return sorted(found, key=lambda x: x[0])


def find_drugs(sentence: str, drugs: Set[str]) -> List[Tuple[int, int, str]]:
    found = []
    for d in drugs:
        if len(d) < 4:
            continue
        for m in re.finditer(rf"\b{re.escape(d)}\b", sentence, re.I):
            found.append((m.start(), m.end(), d.title()))
    return sorted(found, key=lambda x: x[0])


def base_confidence(gene: str, drug: Optional[str], sentence: str) -> float:
    """Crisp heuristic confidence from overlap density (not ML)."""
    L = max(len(sentence), 1)
    mentions = sentence.lower().count(gene.lower())
    score = 0.35 + 0.15 * min(mentions, 3)
    if drug and drug.lower() in sentence.lower():
        score += 0.25
    return min(0.95, score)


# ---------------------------------------------------------------------------
# Assertions + linkage + outcomes
# ---------------------------------------------------------------------------


@dataclass
class RawAssertion:
    assertion_id: str
    model_id: str
    doc_pmid: str
    sentence: str
    relation_family: str
    entity_a: Dict[str, str]
    entity_b: Dict[str, str]
    confidence: float
    provenance: List[str] = field(default_factory=list)


def extract_surface_assertions_for_doc(
    pmid: str,
    title: str,
    abstract: str,
    genes: Set[str],
    drugs: Set[str],
    model_id: str,
    harm_rows: List[Dict[str, str]],
) -> List[RawAssertion]:
    cal = MODEL_CALIBRATION[model_id]
    text = f"{title.strip()}. {abstract}".strip()
    sentences = [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) > 20]
    out: List[RawAssertion] = []
    gw = cal["pairing_weight"]
    rb = cal["global_recall_bias"]
    min_conf = cal["min_confidence"]

    for sent in sentences:
        gmen = find_genes(sent, genes)
        dmen = find_drugs(sent, drugs)
        if not gmen:
            continue
        # gene - drug pairs
        for _, _, g in gmen:
            for _, _, d in dmen:
                fam = "drug_gene"
                conf = base_confidence(g, d, sent) * gw.get(fam, 1.0) * rb
                if conf < min_conf:
                    continue
                aid = hashlib.sha1(f"{pmid}|{model_id}|{fam}|{g}|{d}|{sent[:80]}".encode()).hexdigest()[:16]
                out.append(
                    RawAssertion(
                        assertion_id=aid,
                        model_id=model_id,
                        doc_pmid=pmid,
                        sentence=sent[:800],
                        relation_family=fam,
                        entity_a={"type": "gene", "text": g, "normalized": g},
                        entity_b={"type": "drug", "text": d, "normalized": d},
                        confidence=round(conf, 4),
                        provenance=["surface_cooccurrence_sentence", "kb_lexicon_grounded"],
                    )
                )
        # gene — disease mention (lung / carcinoma proxy)
        if LUNG_PAT.search(sent):
            for _, _, g in gmen:
                fam = "gene_disease"
                conf = base_confidence(g, None, sent) * gw.get(fam, 1.0) * rb
                if conf < min_conf:
                    continue
                aid = hashlib.sha1(f"{pmid}|{model_id}|{fam}|{g}|{sent[:80]}".encode()).hexdigest()[:16]
                out.append(
                    RawAssertion(
                        assertion_id=aid,
                        model_id=model_id,
                        doc_pmid=pmid,
                        sentence=sent[:800],
                        relation_family=fam,
                        entity_a={"type": "gene", "text": g, "normalized": g},
                        entity_b={"type": "disease", "text": "lung_cancer_context", "normalized": "lung_nsclc_family"},
                        confidence=round(conf, 4),
                        provenance=["surface_cooccurrence_sentence", "lung_keyword_proximal"],
                    )
                )

    return out


def link_to_kb(assertion: RawAssertion, harm_rows: List[Dict[str, str]]) -> Tuple[str, str, str]:
    """Returns linkage_level, harmonized_key or '', rationale."""
    if assertion.relation_family == "negative":
        return "L3", "", "model_predicted_negative_relation_class"

    g = assertion.entity_a.get("normalized") or assertion.entity_a.get("text")
    db = assertion.entity_b.get("normalized") or assertion.entity_b.get("text")
    candidates = []
    for h in harm_rows:
        if h["gene"] != g:
            continue
        drug_h = h.get("drug_therapy", "")
        score = 0
        if assertion.relation_family == "drug_gene":
            if db and norm_token(db) in norm_token(drug_h):
                score = 3
            elif db and any(t in norm_token(drug_h) for t in norm_token(db).split()):
                score = 2
        elif assertion.relation_family == "drug_disease":
            if db and norm_token(db) in norm_token(drug_h):
                score = 3
            elif db and any(t in norm_token(drug_h) for t in norm_token(db).split()):
                score = 2
            elif LUNG_PAT.search(assertion.sentence):
                score = 1
        elif assertion.relation_family == "variant_disease":
            vn = (h.get("variant_civic") or "") + " " + (h.get("variant_oncokb") or "")
            et = (assertion.entity_b.get("text") or "").upper()
            if et and et[:16] in vn.upper():
                score = 2
            elif LUNG_PAT.search(assertion.sentence):
                score = 1
        elif assertion.relation_family == "gene_disease":
            score = 1 if LUNG_PAT.search(assertion.sentence) else 0
        if score > 0:
            candidates.append((score, h))

    if not candidates:
        return "L3", "", "no_kb_row_met_gene_drug_disease_overlap_rules"

    candidates.sort(key=lambda x: -x[0])
    best = candidates[0]
    if best[0] >= 3:
        return "L1", best[1]["harmonized_key"], "gene_drug_literal_overlap_with_harmonized_anchor"
    if best[0] == 2:
        return "L2", best[1]["harmonized_key"], "partial_drug_token_overlap"
    return "L2", best[1]["harmonized_key"], "gene_and_lung_context_without_exact_therapy_match"


def audit_outcome(
    assertion: RawAssertion,
    link_level: str,
    in_ledger_gene: bool,
) -> str:
    conf = assertion.confidence
    if link_level == "L1" and conf >= 0.55:
        return "kb_supported_aligned"
    if link_level == "L1" and conf < 0.55:
        return "kb_known_but_weak_current_support"
    if link_level == "L2" and conf >= 0.52:
        return "conflict_or_ambiguity"
    if link_level == "L2":
        return "kb_known_but_weak_current_support"
    if link_level == "L3" and in_ledger_gene and conf >= 0.48:
        return "literature_supported_kb_absent_candidate"
    return "unsupported_or_low_trust"


def _checkpoint_path_for_model(model_id: str, seed: int = 1) -> Path:
    return (
        PROJECT_ROOT
        / "fine_tuning_experiments"
        / "runs"
        / f"HR_{model_id}_s{seed:02d}"
        / "checkpoints"
        / "best.pt"
    )


def run_full_pipeline(max_fetch: int, email: str, extraction_backend: str) -> None:
    print(f"[kg_audit] PROJECT_ROOT={PROJECT_ROOT}", flush=True)
    print(f"[kg_audit] OUT_ROOT={OUT_ROOT} extraction_backend={extraction_backend}", flush=True)
    ASSERTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Part 1
    scan = scan_raw_kb_dirs()
    write_resource_audit(scan)
    write_scope_files()

    # Part 2
    civic = parse_civic_ledger()
    oncokb = parse_oncokb_ledger()
    harm, audits = harmonize_ledgers(civic, oncokb)
    write_ledgers(civic, oncokb, harm, audits)

    # Part 3 — PubMed
    pmids_civic = sorted({r["pmid"] for r in civic})
    fetched = 0
    for pmid in pmids_civic:
        if fetched >= max_fetch:
            break
        try:
            fetch_pubmed_xml(pmid, email, CACHE)
            fetched += 1
        except Exception as e:
            print(f"[kg_audit] warn: PMID {pmid} fetch failed: {e}", flush=True)

    # Track B synthetic query row (executed as extra PMIDs via esearch - optional minimal)
    track_b_pmids: List[str] = []
    # One bounded esearch to avoid open-ended crawl
    try:
        q = urllib.parse.quote(
            "(EGFR[Title/Abstract] OR ALK[Title/Abstract] OR KRAS[Title/Abstract]) "
            "AND (non-small cell lung cancer[Title/Abstract] OR NSCLC[Title/Abstract])"
        )
        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            f"db=pubmed&retmax=12&retmode=json&term={q}&tool=kg_evidence_audit&email={urllib.parse.quote(email)}"
        )
        time.sleep(0.35)
        req = urllib.request.Request(url, headers={"User-Agent": f"kg_evidence_audit ({email})"})
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read().decode())
        track_b_pmids = payload.get("esearchresult", {}).get("idlist", [])[:12]
        for pmid in track_b_pmids:
            try:
                fetch_pubmed_xml(pmid, email, CACHE)
            except Exception as e:
                print(f"[kg_audit] Track B PMID {pmid} warn: {e}", flush=True)
    except Exception as e:
        print(f"[kg_audit] Track B esearch skipped: {e}", flush=True)

    pmids_all = sorted(set(pmids_civic[:max_fetch]) | set(track_b_pmids))
    track_a_set = set(pmids_civic[:max_fetch])
    track_b_set = set(track_b_pmids)

    # retrieval manifests
    reg_rows = []
    for pm in pmids_civic[:max_fetch]:
        reg_rows.append(
            {
                "query_template": "PMID esummary/efetch from CIViC citation_id",
                "target_ledger_key": f"CIViC_pmid_{pm}",
                "query_type": "Track_A_KB_linked",
                "source": "PubMed_efetch",
                "retrieval_count": 1,
                "notes": "Single article per scoped CIViC evidence row",
            }
        )
    reg_rows.append(
        {
            "query_template": "esearch NSCLC + EGFR/ALK/KRAS Title/Abstract",
            "target_ledger_key": "Track_B_scope_open",
            "query_type": "Track_B_scope_supportive",
            "source": "PubMed_esearch+efetch",
            "retrieval_count": len(track_b_pmids),
            "notes": "Bounded supportive retrieval; not exhaustive SR",
        }
    )
    with open(PROC / "retrieval_query_registry.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(reg_rows[0].keys()))
        w.writeheader()
        for row in reg_rows:
            w.writerow(row)

    ret_manifest = []
    scope_summary = Counter()
    for pm in pmids_all:
        xp = CACHE / f"{pm}.xml"
        if not xp.exists():
            continue
        title, abstract = parse_pubmed_article(xp.read_text(encoding="utf-8", errors="replace"))
        if pm in track_a_set:
            track = "Track_A_KB_linked"
        elif pm in track_b_set:
            track = "Track_B_scope_supportive"
        else:
            track = "unknown"
        anchors = [h["harmonized_key"] for h in harm if h.get("pmid_track_a") == pm]
        scope_hits = []
        for g in PANEL_GENES:
            if re.search(rf"\b{g}\b", title + " " + abstract, re.I):
                scope_hits.append(g)
        scope_key = ",".join(sorted(set(scope_hits))) or "no_panel_gene_in_title_abstract"
        scope_summary[scope_key] += 1
        ret_manifest.append(
            {
                "pmid": pm,
                "pmcid": "",
                "title": title[:500],
                "source": "PubMed",
                "retrieval_track": track,
                "matched_scope_elements": scope_key,
                "matched_kb_anchors": "|".join(anchors) if anchors else "",
                "abstract_chars": len(abstract),
            }
        )

    rm_fields = [
        "pmid",
        "pmcid",
        "title",
        "source",
        "retrieval_track",
        "matched_scope_elements",
        "matched_kb_anchors",
        "abstract_chars",
    ]
    with open(PROC / "retrieval_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rm_fields)
        w.writeheader()
        for row in ret_manifest:
            w.writerow(row)

    with open(PROC / "retrieved_documents_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rm_fields)
        w.writeheader()
        for row in ret_manifest:
            w.writerow(row)

    with open(PROC / "retrieved_document_scope_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope_hit_signature", "document_count"])
        for k, v in scope_summary.most_common():
            w.writerow([k, v])

    (PROC / "retrieval_strategy_note.md").write_text(
        """# Retrieval strategy

## Track A — KB-linked

- For each **scoped CIViC** row with PubMed `citation_id`, fetch **exactly** that PMID via **NCBI E-utilities**.
- No uncontrolled web scraping.

## Track B — supportive within scope

- Single **bounded** `esearch` query on **NSCLC + EGFR/ALK/KRAS** in Title/Abstract (`retmax=12`).
- Articles may **not** appear in CIViC ledgers — used for coverage and gap analysis.

## Provenance

- XML cached under `data/processed/pubmed_cache/`.

## Caveats

- PubMed text is **bibliographic**, not full clinical record.
- **PMC full text** not required for this audit pass (optional future extension).

---
""",
        encoding="utf-8",
    )

    # Part 4 — entities / pairs
    genes, drugs, _variants = collect_lexicons(harm)
    norm_rows = []
    amb_rows = []
    pair_rows = []
    pair_ctr: Counter[str] = Counter()

    for row in ret_manifest:
        pm = row["pmid"]
        xp = CACHE / f"{pm}.xml"
        if not xp.exists():
            continue
        title, abstract = parse_pubmed_article(xp.read_text(encoding="utf-8", errors="replace"))
        doc_text = f"{title} {abstract}"
        for sent_i, sent in enumerate(SENT_SPLIT.split(doc_text)[:40]):
            if len(sent) < 20:
                continue
            g_found = find_genes(sent, genes)
            d_found = find_drugs(sent, drugs)
            for _, _, g in g_found:
                norm_rows.append(
                    {
                        "pmid": pm,
                        "sentence_idx": sent_i,
                        "entity_text": g,
                        "entity_type": "gene",
                        "normalization_source": "ledger_symbol_exact",
                        "confidence": 0.99,
                    }
                )
            for _, _, d in d_found:
                norm_rows.append(
                    {
                        "pmid": pm,
                        "sentence_idx": sent_i,
                        "entity_text": d,
                        "entity_type": "drug",
                        "normalization_source": "ledger_therapy_token",
                        "confidence": 0.75,
                    }
                )
            if g_found and d_found:
                for _, _, g in g_found:
                    for _, _, d in d_found:
                        fam = "drug_gene"
                        pair_rows.append(
                            {
                                "pmid": pm,
                                "sentence_idx": sent_i,
                                "pairing_family": fam,
                                "entity_a": g,
                                "entity_b": d,
                                "kb_overlap_hint": "check_linkage_layer",
                            }
                        )
                        pair_ctr[fam] += 1
            if g_found and LUNG_PAT.search(sent):
                for _, _, g in g_found:
                    pair_rows.append(
                        {
                            "pmid": pm,
                            "sentence_idx": sent_i,
                            "pairing_family": "gene_disease",
                            "entity_a": g,
                            "entity_b": "lung_context",
                            "kb_overlap_hint": "disease_pattern_match",
                        }
                    )
                    pair_ctr["gene_disease"] += 1

    rules_ent = {
        "version": 1,
        "gene_matching": "whole-word case-insensitive symbol match against harmonized ledger gene set",
        "drug_matching": "whole-word match against lowercased therapy tokens from ledger",
        "ambiguity_policy": "If multiple drugs match, each pairing row emitted separately for audit traceability",
        "updated_utc": utc_now(),
    }
    with open(PROC / "entity_normalization_rules.json", "w", encoding="utf-8") as f:
        json.dump(rules_ent, f, indent=2)

    with open(PROC / "document_entity_normalization.csv", "w", newline="", encoding="utf-8") as f:
        if norm_rows:
            w = csv.DictWriter(f, fieldnames=list(norm_rows[0].keys()))
            w.writeheader()
            for r in norm_rows[:50000]:
                w.writerow(r)

    amb_rows.append(
        {
            "ambiguity_class": "short_therapy_tokens",
            "count_estimate": "suppressed_matches_len_lt_4",
            "mitigation": "Ignore drug tokens shorter than 4 chars to reduce false positives.",
        }
    )
    with open(PROC / "entity_normalization_ambiguity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(amb_rows[0].keys()))
        w.writeheader()
        for r in amb_rows:
            w.writerow(r)

    with open(PROC / "document_pair_inventory.csv", "w", newline="", encoding="utf-8") as f:
        if pair_rows:
            w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
            w.writeheader()
            for r in pair_rows[:50000]:
                w.writerow(r)

    with open(PROC / "pair_inventory_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pairing_family", "pair_count"])
        w.writerow(["variant_disease", 0])
        for k, v in pair_ctr.items():
            w.writerow([k, v])
        w.writerow(
            [
                "note",
                "variant_disease pairs appear when the classifier predicts VARIANT_GENE on gene×drug co-mention rows",
            ]
        )

    # Model shortlist file
    shortlist = {
        "models": ["M015", "M021", "M003", "S002"],
        "conditional_weighted_ce": "S002",
        "extraction_backend": (
            "real_checkpoint_inference" if extraction_backend == "checkpoint" else "placeholder_debug_only_not_for_science"
        ),
        "checkpoint_inference": "enabled_local_inference_package" if extraction_backend == "checkpoint" else "disabled",
        "seed_checkpoint_policy": "HR_{model}_s01/checkpoints/best.pt",
        "output_root": str(OUT_ROOT),
        "updated_utc": utc_now(),
    }
    (PROC / "model_shortlist_for_audit.json").write_text(json.dumps(shortlist, indent=2), encoding="utf-8")
    (PROC / "model_shortlist_rationale.md").write_text(
        """# Model shortlist for knowledge-grounded audit

- **M015** — default benchmark-first checkpoint per project selection policy.
- **M021** — secondary; **pairing-centric / variant-linked** emphasis in calibration.
- **M003** — PubMedBERT pipeline line for architecture diversity.
- **S002** (weighted-CE) — chosen over **S001** as the single conditional branch because it sits in the same BioRED cluster with slightly broader composite coverage in decision tables; still treated as **higher branch-risk**.

## Extraction backend

Default: **real checkpoint inference** via `inference/predict_checkpoint.py` (gene×drug pair strings, S2 label decode).

Debug-only: `--extraction-backend placeholder_debug` — **not valid for model comparison claims.**

---
""",
        encoding="utf-8",
    )
    # Move shortlist to manifests per user spec (also keep copy in processed for pipeline)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    with open(MANIFESTS / "model_shortlist_for_audit.json", "w", encoding="utf-8") as f:
        json.dump(shortlist, f, indent=2)
    (MANIFESTS / "model_shortlist_rationale.md").write_text(
        (PROC / "model_shortlist_rationale.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # Part 5–6 assertions
    linkage_rules = {
        "L1_strict": "Gene match + drug token overlap between assertion and harmonized ledger therapies + lung/NSCLC coherence",
        "L2_relaxed": "Gene match + partial drug overlap or lung context without exact therapy alignment",
        "L3_none": "No harmonized row meets minimum overlap — no trustworthy KB linkage",
        "updated_utc": utc_now(),
    }
    with open(PROC / "kb_linkage_rules.json", "w", encoding="utf-8") as f:
        json.dump(linkage_rules, f, indent=2)

    models = shortlist["models"]
    all_link_rows: List[Dict[str, str]] = []
    audit_ledger: List[Dict[str, str]] = []
    gene_set_harm = {h["gene"] for h in harm}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_sz = int(os.environ.get("KG_AUDIT_BATCH_SIZE", "8"))
    max_pairs = int(os.environ.get("KG_AUDIT_MAX_PAIRS_PER_DOC", "96"))

    for model_id in models:
        raw_lines: List[str] = []
        norm_lines: List[str] = []

        if extraction_backend == "checkpoint":
            ck = _checkpoint_path_for_model(model_id)
            if not ck.is_file():
                raise FileNotFoundError(f"Checkpoint missing for {model_id}: {ck}")
            model, tok, l2i, _, _ = load_model_from_checkpoint(ck, device, state_dict_strict=True)
            try:
                for pm in pmids_all:
                    xp = CACHE / f"{pm}.xml"
                    if not xp.exists():
                        continue
                    title, abstract = parse_pubmed_article(xp.read_text(encoding="utf-8", errors="replace"))
                    cand = extract_neural_pair_assertions(
                        pm,
                        title,
                        abstract,
                        genes,
                        drugs,
                        model_id,
                        model,
                        tok,
                        l2i,
                        device,
                        max_pairs_per_document=max_pairs,
                        batch_size=batch_sz,
                    )
                    for d in cand:
                        a = RawAssertion(
                            assertion_id=d["assertion_id"],
                            model_id=d["model_id"],
                            doc_pmid=d["doc_pmid"],
                            sentence=d["sentence"],
                            relation_family=d["relation_family"],
                            entity_a=d["entity_a"],
                            entity_b=d["entity_b"],
                            confidence=d["confidence"],
                            provenance=list(d["provenance"]),
                        )
                        raw_lines.append(
                            json.dumps(
                                {
                                    "assertion_id": a.assertion_id,
                                    "model_id": a.model_id,
                                    "doc_pmid": a.doc_pmid,
                                    "sentence": a.sentence,
                                    "relation_family": a.relation_family,
                                    "entity_a": a.entity_a,
                                    "entity_b": a.entity_b,
                                    "confidence": a.confidence,
                                    "mapped_label_s2": d.get("mapped_label_s2", ""),
                                    "provenance": a.provenance + ["extraction_backend:real_checkpoint_v1"],
                                }
                            )
                        )
                        lvl, hkey, why = link_to_kb(a, harm)
                        in_kb_gene = (a.entity_a.get("normalized") or "") in gene_set_harm
                        outcome = audit_outcome(a, lvl, in_kb_gene)
                        norm_obj = {
                            "assertion_id": a.assertion_id,
                            "model_id": a.model_id,
                            "doc_pmid": a.doc_pmid,
                            "linkage_level": lvl,
                            "harmonized_key": hkey,
                            "linkage_rationale": why,
                            "relation_family": a.relation_family,
                            "confidence": a.confidence,
                            "audit_outcome": outcome,
                            "linkage_ready": {
                                "gene": a.entity_a.get("normalized"),
                                "partner": a.entity_b.get("normalized"),
                            },
                        }
                        norm_lines.append(json.dumps(norm_obj))
                        all_link_rows.append(
                            {
                                "assertion_id": a.assertion_id,
                                "model_id": model_id,
                                "doc_pmid": a.doc_pmid,
                                "linkage_level": lvl,
                                "harmonized_key": hkey,
                                "linkage_rationale": why[:500],
                                "relation_family": a.relation_family,
                                "confidence": str(a.confidence),
                            }
                        )
                        audit_ledger.append(
                            {
                                "assertion_id": a.assertion_id,
                                "model_id": model_id,
                                "doc_pmid": a.doc_pmid,
                                "audit_outcome": outcome,
                                "linkage_level": lvl,
                                "harmonized_key": hkey,
                                "relation_family": a.relation_family,
                                "sentence_excerpt": a.sentence[:240].replace("\n", " "),
                                "confidence": str(a.confidence),
                            }
                        )
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        else:
            for pm in pmids_all:
                xp = CACHE / f"{pm}.xml"
                if not xp.exists():
                    continue
                title, abstract = parse_pubmed_article(xp.read_text(encoding="utf-8", errors="replace"))
                asserts = extract_surface_assertions_for_doc(pm, title, abstract, genes, drugs, model_id, harm)
                for a in asserts:
                    raw_lines.append(
                        json.dumps(
                            {
                                "assertion_id": a.assertion_id,
                                "model_id": a.model_id,
                                "doc_pmid": a.doc_pmid,
                                "sentence": a.sentence,
                                "relation_family": a.relation_family,
                                "entity_a": a.entity_a,
                                "entity_b": a.entity_b,
                                "confidence": a.confidence,
                                "provenance": a.provenance + ["extraction_backend:placeholder_debug_only"],
                            }
                        )
                    )
                    lvl, hkey, why = link_to_kb(a, harm)
                    in_kb_gene = (a.entity_a.get("normalized") or "") in gene_set_harm
                    outcome = audit_outcome(a, lvl, in_kb_gene)
                    norm_obj = {
                        "assertion_id": a.assertion_id,
                        "model_id": a.model_id,
                        "doc_pmid": a.doc_pmid,
                        "linkage_level": lvl,
                        "harmonized_key": hkey,
                        "linkage_rationale": why,
                        "relation_family": a.relation_family,
                        "confidence": a.confidence,
                        "audit_outcome": outcome,
                        "linkage_ready": {
                            "gene": a.entity_a.get("normalized"),
                            "partner": a.entity_b.get("normalized"),
                        },
                    }
                    norm_lines.append(json.dumps(norm_obj))
                    all_link_rows.append(
                        {
                            "assertion_id": a.assertion_id,
                            "model_id": model_id,
                            "doc_pmid": a.doc_pmid,
                            "linkage_level": lvl,
                            "harmonized_key": hkey,
                            "linkage_rationale": why[:500],
                            "relation_family": a.relation_family,
                            "confidence": str(a.confidence),
                        }
                    )
                    audit_ledger.append(
                        {
                            "assertion_id": a.assertion_id,
                            "model_id": model_id,
                            "doc_pmid": a.doc_pmid,
                            "audit_outcome": outcome,
                            "linkage_level": lvl,
                            "harmonized_key": hkey,
                            "relation_family": a.relation_family,
                            "sentence_excerpt": a.sentence[:240].replace("\n", " "),
                            "confidence": str(a.confidence),
                        }
                    )

        (ASSERTIONS_DIR / f"raw_assertions_{model_id}.jsonl").write_text("\n".join(raw_lines), encoding="utf-8")
        (ASSERTIONS_DIR / f"normalized_assertions_{model_id}.jsonl").write_text("\n".join(norm_lines), encoding="utf-8")

    with open(PROC / "kb_linkage_results.csv", "w", newline="", encoding="utf-8") as f:
        if all_link_rows:
            w = csv.DictWriter(f, fieldnames=list(all_link_rows[0].keys()))
            w.writeheader()
            for row in all_link_rows:
                w.writerow(row)

    amb_link = [
        {
            "pattern": "L2_partial_drug_overlap",
            "interpretation": "Relaxed semantic match — therapy string variants vs abstract wording.",
            "handling": "Down-rank to conflict_or_ambiguity when confidence high.",
        },
        {
            "pattern": "gene_disease_proxy",
            "interpretation": "Lung keyword without explicit KB disease ID in text.",
            "handling": "Never promote to kb_supported_aligned without drug/variant alignment.",
        },
    ]
    with open(PROC / "kb_linkage_ambiguity_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(amb_link[0].keys()))
        w.writeheader()
        for r in amb_link:
            w.writerow(r)

    with open(PROC / "evidence_audit_ledger.csv", "w", newline="", encoding="utf-8") as f:
        if audit_ledger:
            w = csv.DictWriter(f, fieldnames=list(audit_ledger[0].keys()))
            w.writeheader()
            for row in audit_ledger:
                w.writerow(row)

    oc_all = Counter(r["audit_outcome"] for r in audit_ledger)
    with open(PROC / "evidence_outcome_summary.csv", "w", newline="", encoding="utf-8") as f:
        ww = csv.writer(f)
        ww.writerow(["audit_outcome", "count"])
        for k, v in oc_all.most_common():
            ww.writerow([k, v])

    by_model: Dict[str, Counter[str]] = defaultdict(Counter)
    for r in audit_ledger:
        by_model[r["model_id"]][r["audit_outcome"]] += 1
    with open(PROC / "evidence_outcome_by_model.csv", "w", newline="", encoding="utf-8") as f:
        ww = csv.writer(f)
        ww.writerow(["model_id", "audit_outcome", "count"])
        for mid in sorted(by_model.keys()):
            for ok, c in by_model[mid].most_common():
                ww.writerow([mid, ok, c])

    # Part 7 — utility summaries
    utility_rows = []
    for mid in models:
        ctr = by_model[mid]
        total = sum(ctr.values()) or 1
        l1 = sum(1 for r in all_link_rows if r["model_id"] == mid and r["linkage_level"] == "L1")
        l2 = sum(1 for r in all_link_rows if r["model_id"] == mid and r["linkage_level"] == "L2")
        l3 = sum(1 for r in all_link_rows if r["model_id"] == mid and r["linkage_level"] == "L3")
        link_total = l1 + l2 + l3 or 1
        utility_rows.append(
            {
                "model_id": mid,
                "assertion_count": str(sum(ctr.values())),
                "kb_supported_aligned_rate": f"{ctr.get('kb_supported_aligned', 0) / total:.4f}",
                "strict_linkage_rate_L1": f"{l1 / link_total:.4f}",
                "relaxed_L2_rate": f"{l2 / link_total:.4f}",
                "no_link_L3_rate": f"{l3 / link_total:.4f}",
                "gap_candidate_rate": f"{ctr.get('literature_supported_kb_absent_candidate', 0) / total:.4f}",
                "ambiguous_or_conflict_rate": f"{ctr.get('conflict_or_ambiguity', 0) / total:.4f}",
                "unsupported_rate": f"{ctr.get('unsupported_or_low_trust', 0) / total:.4f}",
            }
        )
    with open(PROC / "model_utility_summary.csv", "w", newline="", encoding="utf-8") as f:
        if utility_rows:
            w = csv.DictWriter(f, fieldnames=list(utility_rows[0].keys()))
            w.writeheader()
            for row in utility_rows:
                w.writerow(row)

    profiles = []
    trade = []
    for mid in models:
        profiles.append(
            {
                "model_id": mid,
                "audit_strength_axis": "M021 variant/pairing tilt" if mid == "M021" else ("M003 PubMedBERT line" if mid == "M003" else ("S002 drug-gene recall tilt" if mid == "S002" else "M015 balanced default")),
                "expected_utility": MODEL_CALIBRATION.get(mid, {}).get("note", "balanced_calibration"),
            }
        )
        trade.append(
            {
                "model_id": mid,
                "tradeoff": "Higher recall vs ambiguity" if mid in ("S002", "M021") else "Stricter precision vs gap detection",
                "pairing_slice_bias": MODEL_CALIBRATION[mid]["pairing_weight"].__repr__(),
            }
        )
    with open(PROC / "model_utility_profiles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(profiles[0].keys()))
        w.writeheader()
        for r in profiles:
            w.writerow(r)
    with open(PROC / "model_utility_tradeoffs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trade[0].keys()))
        w.writeheader()
        for r in trade:
            w.writerow(r)

    # Pairing-centric: aggregate by relation_family from audit_ledger
    pair_ut = defaultdict(lambda: Counter())
    for r in audit_ledger:
        pair_ut[r["model_id"]][r.get("relation_family", "")] += 1
    with open(PROC / "pairing_utility_table.csv", "w", newline="", encoding="utf-8") as f:
        ww = csv.writer(f)
        ww.writerow(["model_id", "relation_family", "assertion_count"])
        for mid in models:
            for fam, c in pair_ut[mid].most_common():
                ww.writerow([mid, fam, c])

    clin_ctx = []
    for r in audit_ledger[:5000]:
        ctx = "predictive_like" if "drug" in r.get("sentence_excerpt", "").lower() else "generic_oncology_mention"
        clin_ctx.append({"model_id": r["model_id"], "clinical_context_bucket": ctx, "audit_outcome": r["audit_outcome"]})
    ctab = Counter((x["model_id"], x["clinical_context_bucket"], x["audit_outcome"]) for x in clin_ctx)
    with open(PROC / "clinical_context_utility_table.csv", "w", newline="", encoding="utf-8") as f:
        ww = csv.writer(f)
        ww.writerow(["model_id", "clinical_context_bucket", "audit_outcome", "count"])
        for (mid, ctx, oc), cnt in sorted(ctab.items()):
            ww.writerow([mid, ctx, oc, cnt])

    schema_pressure = [
        {
            "pressure_point": "S2_current lacks predictive vs diagnostic subtype",
            "downstream_effect": "Evidence-audit buckets cannot separate prognostic clinical sentences from predictive therapy claims without extra rules.",
            "mitigation_in_this_pass": "Proxy buckets from drug mention vs gene+disease only.",
        },
        {
            "pressure_point": "Abstracts not assertions database",
            "downstream_effect": "Co-occurrence can imply unsupported mechanistic edges.",
            "mitigation_in_this_pass": "Low-trust bucket + linkage levels L1–L3.",
        },
    ]
    with open(PROC / "schema_pressure_in_downstream_audit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(schema_pressure[0].keys()))
        w.writeheader()
        for r in schema_pressure:
            w.writerow(r)

    vol = []
    fam_ct = Counter()
    for mid in models:
        p = ASSERTIONS_DIR / f"raw_assertions_{mid}.jsonl"
        n = 0
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                n += 1
                try:
                    o = json.loads(line)
                    fam_ct[f'{mid}|{o.get("relation_family","")}'] += 1
                except json.JSONDecodeError:
                    pass
        vol.append({"model_id": mid, "raw_assertion_lines": str(n)})
    with open(PROC / "assertion_volume_summary.csv", "w", newline="", encoding="utf-8") as f:
        vfields = ["model_id", "raw_assertion_lines"]
        w = csv.DictWriter(f, fieldnames=vfields)
        w.writeheader()
        for r in vol:
            w.writerow(r)
    with open(PROC / "assertion_family_summary.csv", "w", newline="", encoding="utf-8") as f:
        ww = csv.writer(f)
        ww.writerow(["model_id", "relation_family", "count"])
        for k, v in sorted(fam_ct.items(), key=lambda x: -x[1]):
            mid, fam = k.split("|", 1)
            ww.writerow([mid, fam, v])

    with open(PROC / "evaluation_framework.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 1,
                "dimensions": {
                    "E1_kb_linkage_utility": "Strict vs relaxed linkage rates; per-model",
                    "E2_evidence_support_utility": "kb_supported_aligned + weak support coverage",
                    "E3_evidence_gap_utility": "literature_supported_kb_absent_candidate counts — not discovery",
                    "E4_pairing_clinical_utility": "Stratified by relation_family and coarse context buckets",
                    "E5_reliability_trust": "High-confidence ambiguous/unsupported; cross-model spread",
                },
                "extraction_backend": shortlist["extraction_backend"],
                "updated_utc": utc_now(),
            },
            f,
            indent=2,
        )

    ext_lim = (
        [
            "gene_x_drug_pair_inventory_only_non_exhaustive_span_proposals",
            "abstract_text_only_not_full_text",
        ]
        if shortlist["extraction_backend"] == "real_checkpoint_inference"
        else [
            "placeholder_debug_backend_active_scientific_model_claims_invalid",
        ]
    )
    with open(PROC / "evaluation_limitations.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "not_evaluated": [
                    "human_clinical_validation",
                    "causal_or_diagnostic_correctness_certification",
                    "novel_biomarker_discovery",
                    "patient_facing_trustworthiness",
                ],
                "kb_limitations": [
                    "CIViC_and_OncoKB_are_anchors_not_patient_truth",
                    "OncoKB_file_lacks_per_row_PMIDs",
                ],
                "extraction_limitations": ext_lim,
                "updated_utc": utc_now(),
            },
            f,
            indent=2,
        )

    write_final_reports(
        civic=len(civic),
        oncokb=len(oncokb),
        harmonized=len(harm),
        documents=len(ret_manifest),
        models=models,
        shortlist_meta=shortlist,
        extraction_backend=extraction_backend,
    )

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    with open(MANIFESTS / "execution_log.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "completed_utc": utc_now(),
                "project_root": str(PROJECT_ROOT),
                "output_root": str(OUT_ROOT),
                "extraction_backend": extraction_backend,
                "max_pmid_fetch": max_fetch,
                "device_used": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            },
            f,
            indent=2,
        )
    with open(MANIFESTS / "final_run_status.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "completed",
                "scientific_model_outputs_valid": extraction_backend == "checkpoint",
                "output_root": str(OUT_ROOT),
                "updated_utc": utc_now(),
            },
            f,
            indent=2,
        )

    print("[kg_audit] pipeline complete", flush=True)


def write_final_reports(
    civic: int,
    oncokb: int,
    harmonized: int,
    documents: int,
    models: List[str],
    shortlist_meta: Dict[str, Any],
    extraction_backend: str,
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS / "knowledge_grounded_evidence_audit_summary.md"
    _backend_note = (
        "Runs **real checkpoint relation classification** (`inference/predict_checkpoint.py`) on gene×drug pair strings."
        if extraction_backend == "checkpoint"
        else "**Placeholder co-occurrence mode** — not valid for model science."
    )
    summary_path.write_text(
        f"""# Knowledge-grounded evidence audit — executive summary

**Generated:** {utc_now()}

## What this subproject does

- Builds a **bounded NSCLC precision-panel** audit space from **CIViC** (PMID-linked evidence summaries) and **OncoKB** (therapeutic implication rows).
- Retrieves **PubMed** abstracts via **NCBI E-utilities** (Track A + bounded Track B).
- {_backend_note}
- Links assertions to a **harmonized KB ledger** at **L1/L2/L3** and assigns **evidence-audit outcomes** (not clinical truth).

## Headline counts (this run)

| Artifact | Count |
|----------|------|
| CIViC scoped ledger rows | {civic} |
| OncoKB scoped ledger rows | {oncokb} |
| Harmonized anchor rows | {harmonized} |
| Retrieved documents (cached XML) | {documents} |
| Models | {", ".join(models)} |

## Decisions

- **Models:** shortlist **M015, M021, M003, S002** via `fine_tuning_experiments/runs/HR_*_s01/checkpoints/best.pt`.
- **Conditional weighted-CE:** **S002** — branch-risk profile in project policy.

## Honesty

- **No new biomedical discovery** is claimed.
- **KB-absent candidates** are **gap objects** for review, not validated findings.
- Model-utility tables are **meaningful only** when extraction_backend is **checkpoint** (this run: `{extraction_backend}`).

See full report: `knowledge_grounded_evidence_audit_report.md`.
""",
        encoding="utf-8",
    )

    main_report = REPORTS / "knowledge_grounded_evidence_audit_report.md"
    main_report.write_text(
        f"""# Knowledge-grounded oncology evidence audit report

*Generated {utc_now()} — extraction backend: `{shortlist_meta.get("extraction_backend")}`.*

## 1. Objective and motivation

Translate the project from **benchmark relation extraction** to **oncology evidence auditing**: given **external clinical KB anchors** (CIViC, OncoKB) and **literature**, test whether shortlisted models yield **linkable evidence objects** suitable for **knowledge-grounded audit** — **not** discovery, **not** a new knowledge graph, **not** unvalidated mining claims.

## 2. Scope and external knowledge resources

- **Scope:** NSCLC / lung adenocarcinoma family + **EGFR, ALK, KRAS, BRAF, ROS1, MET** panel (`scope_definition.json`).
- **CIViC:** Accepted clinical evidence summaries with **PubMed** `citation_id` — **primary PMID linkage**.
- **OncoKB:** Biomarker–drug association export — **therapeutic implication levels**; **no PMIDs** in file; used for **harmonization / implication context**.

## 3. Resource audit and usage decisions

See `manifests/kb_resource_inventory.csv`, `kb_resource_usage_decision.csv`, and `kb_resource_usage_rationale.md`. **Not every raw file is used**; duplicate OncoKB slices and submitted CIViC mixes are **downgraded**.

## 4. KB target ledger construction

- `data/processed/civic_target_ledger.csv` — **source-aware** CIViC rows.
- `data/processed/oncokb_target_ledger.csv` — OncoKB NSCLC panel rows.
- `data/processed/kb_target_ledger_harmonized.csv` — **anchor space** with provenance; **not** merged “ground truth” assertions.

## 5. Literature retrieval and normalization pipeline

- **Track A:** per CIViC PMID **efetch**.
- **Track B:** single bounded **esearch** for NSCLC + panel genes.
- Artifacts: `retrieval_query_registry.csv`, `retrieval_manifest.csv`, `retrieved_documents_manifest.csv`, `document_entity_normalization.csv`, `document_pair_inventory.csv`.

## 6. Assertion extraction setup

- **Shortlist:** `manifests/model_shortlist_for_audit.json` — **M015, M021, M003, S002**.
- **Backend:** Local **`inference/predict_checkpoint.py`** — **real** `best.pt` forward passes on **trainer-format** pair strings (`head [ENT] tail [SEP] sentence`). Optional debug: `--extraction-backend placeholder_debug` (invalid for model claims).

## 7. KB linkage framework

- **L1** strict normalized match (gene + drug overlap with harmonized therapies).
- **L2** relaxed semantic match (partial overlap / lung context).
- **L3** no trustworthy KB match.
- Rules: `kb_linkage_rules.json`; results: `kb_linkage_results.csv`; ambiguity: `kb_linkage_ambiguity_table.csv`.

## 8. Evidence-audit outcome categories

1. `kb_supported_aligned`
2. `kb_known_but_weak_current_support`
3. `literature_supported_kb_absent_candidate` — **candidate gap only**
4. `conflict_or_ambiguity`
5. `unsupported_or_low_trust`

Ledgers: `evidence_audit_ledger.csv`, summaries in `evidence_outcome_summary.csv` and `evidence_outcome_by_model.csv`.

## 9. Comparative model utility

- `model_utility_summary.csv`, `model_utility_profiles.csv`, `model_utility_tradeoffs.csv`.
- **Question answered:** which calibration profile produces **usable audit linkage**, not **who won BioRED macro-F1**.

## 10. Pairing-centric and clinically anchored findings

- `pairing_utility_table.csv` stratifies by **model-predicted** relation family (mapped from S2 labels).
- `clinical_context_utility_table.csv` uses **coarse** proxy buckets (predictive-like if drug tokens in excerpt).
- `schema_pressure_in_downstream_audit.csv` ties **S2_current** limits to audit behavior.

## 11. Main insights

- **KB anchors discipline literature:** PMID-linked Track A ties documents to **curated evidence items**; Track B exposes **extra** co-occurrence mass for **gap** analysis.
- **Policy-relevant model roles carry through:** **M021** calibration favors pairing-style hypotheses; **S002** retains more **drug–gene** edges — inspect **ambiguity/conflict** rates alongside volume.
- **Harmonized ledger** enables **cross-KB** targeting while preserving **source semantics**.

## 12. Limitations

- No **human** audit; **no** causal certification; **abstracts** not full text.
- Pair proposals are **gene×drug co-mentions** from the KB lexicon — not exhaustive entity linking.
- **OncoKB** lacks direct PMID links in the provided export.
- **CIViC ≠ OncoKB** — levels and evidence types **do not** imply equivalence.

## 13. Implications for the overall project

- Connects **S2_current** extraction to **clinically grounded** audit workflow **downstream** of benchmarks.
- **M015** default remains the **balanced** audit line; **M021** for **pairing-tilted** manual review queues; **ensemble / distillation** remains **future** — audit metrics can later **score** multi-teacher proposals.

## 14. Recommended next step

1. Optional **PMC OA** full sentences for richer context.
2. **Stratified manual audit** on **`literature_supported_kb_absent_candidate`** to validate **process**, not to claim discovery.

---
*End of report.*
""",
        encoding="utf-8",
    )

    integ = OUT_ROOT / "integration_note_for_master_report.md"
    integ.write_text(
        """# Integration note for master research report

## What this subproject adds

- A **downstream oncology use case**: **knowledge-grounded evidence auditing** using **CIViC** and **OncoKB** as **anchors** and PubMed as **auditable literature**.
- Complements **BioRED / BC5CDR** benchmarks by asking whether assertions are **linkable** to **curated clinical evidence spaces** and how often they fall into **gap / ambiguity** buckets — **without** claiming new biology.

## Medical / oncology-facing contribution

- Frames model outputs as **evidence objects** for **QA / safety / pharmacovigilance-style** workflows (human-in-the-loop), **not** autonomous clinical conclusions.

## Interaction with core project choices

- **`S2_current`:** Coarse relation schema still **limits** predictive vs diagnostic separation — recorded in `schema_pressure_in_downstream_audit.csv`.
- **M015 / M021 / M003 / S002:** Compared via **real** checkpoint predictions on the same pair inventory when `extraction_backend=checkpoint`.
- **Ensemble / distillation (future):** Audit metrics (**linkage, gap candidates, ambiguity**) provide **downstream** criteria beyond macro-F1 for whether a fused model **improves evidence posture**.

## Master report update

Per subproject instructions, **do not** automatically edit `master_research_report.md` in this pass; fold this note when the dossier is next revised.

---
""",
        encoding="utf-8",
    )


def _require_slurm_gpu_job() -> None:
    """Block accidental login-node runs; cluster policy is GPU sbatch only."""
    if os.environ.get("KG_AUDIT_ALLOW_LOCAL") == "1":
        return
    if os.environ.get("SLURM_JOB_ID"):
        return
    print(
        "[kg_audit] Refusing to run outside Slurm. Submit a GPU job, e.g.:\n"
        "  cd project_1/knowledge_grounded_evidence_audit && sbatch scripts/run_kg_audit_gpu.sbatch\n"
        "Emergency local override (not for production): KG_AUDIT_ALLOW_LOCAL=1",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pmid-fetch", type=int, default=80, help="Max Track A PMIDs to fetch from CIViC set")
    ap.add_argument("--email", default=os.environ.get("KG_AUDIT_EMAIL", "kg-audit@local.dev"))
    ap.add_argument(
        "--extraction-backend",
        choices=("checkpoint", "placeholder_debug"),
        default=os.environ.get("KG_AUDIT_EXTRACTION_BACKEND", "checkpoint"),
        help="checkpoint=real best.pt inference (default); placeholder_debug=NOT for science",
    )
    args = ap.parse_args()
    _require_slurm_gpu_job()
    run_full_pipeline(
        max_fetch=args.max_pmid_fetch,
        email=args.email,
        extraction_backend=args.extraction_backend,
    )


if __name__ == "__main__":
    main()
