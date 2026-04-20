"""
Construct gold-lite audit slice (heuristic-assisted, not fully manual gold).

Targets are anchored to CIViC/OncoKB harmonized rows with PMID-linked documents where cached.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_KG = Path(__file__).resolve().parent.parent.parent
if str(_KG) not in sys.path:
    sys.path.insert(0, str(_KG))

from run_pipeline import SENT_SPLIT, parse_pubmed_article

from .paths import CACHE, MANIFESTS, PROC, REPORTS, ensure_dirs

LUNG_HINT = re.compile(
    r"lung|nsclc|adenocarcinoma|carcinoma|neoplasm|oncolog", re.I
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _first_drug(therapy: str) -> Optional[str]:
    if not therapy.strip():
        return None
    parts = re.split(r"[,;]", therapy)
    for p in parts:
        t = p.strip()
        if len(t) >= 4:
            return t.split()[0] if t else None
    return None


def _best_sentence_for_anchor(title: str, abstract: str, gene: str, drug: Optional[str], variant: str) -> Tuple[int, str, float]:
    doc = f"{title}. {abstract}".strip()
    sents = [s.strip() for s in SENT_SPLIT.split(doc) if len(s.strip()) > 25]
    best_i, best_s, best_sc = 0, sents[0] if sents else "", 0.0
    for i, s in enumerate(sents):
        sl = s.lower()
        sc = 0.0
        if gene and re.search(rf"\b{re.escape(gene)}\b", s, re.I):
            sc += 0.35
        if drug and len(drug) >= 4 and re.search(rf"\b{re.escape(drug)}\b", s, re.I):
            sc += 0.35
        if variant:
            for tok in re.findall(r"[A-Za-z0-9]{4,}", variant):
                if tok.lower() in sl:
                    sc += 0.08
        if LUNG_HINT.search(s):
            sc += 0.12
        if sc > best_sc:
            best_sc, best_i, best_s = sc, i, s
    return best_i, best_s[:800], round(min(0.95, 0.4 + best_sc), 3)


def _neighbor_window(sents: List[str], idx: int, before: int, after: int) -> str:
    lo = max(0, idx - before)
    hi = min(len(sents), idx + after + 1)
    return " ".join(sents[lo:hi])[:1500]


def _heuristic_gold_s2(
    evidence_category: str,
    gene: str,
    drug: Optional[str],
    variant: str,
    sentence: str,
) -> str:
    """Heuristic proxy label for oracle F1 — documented as NOT manual gold."""
    sl = sentence.lower()
    has_drug = drug and len(drug) >= 4 and drug.lower() in sl
    has_var = any(
        len(tok) >= 4 and tok.lower() in sl for tok in re.findall(r"[A-Za-z0-9]+", variant) if len(tok) >= 4
    )
    ev = (evidence_category or "").lower()
    if "predictive" in ev and has_drug and gene.lower() in sl:
        return "DRUG_GENE_REGULATION"
    if "predictive" in ev and gene.lower() in sl:
        return "ASSOCIATION_GENERAL"
    if "diagnostic" in ev and has_var:
        return "VARIANT_GENE"
    if gene.lower() in sl:
        return "ASSOCIATION_GENERAL"
    return "ASSOCIATION_GENERAL"


def _expected_pairing_family(row: Dict[str, str]) -> str:
    drug = _first_drug(row.get("drug_therapy", ""))
    if drug and row.get("gene"):
        return "gene_drug"
    if row.get("variant_civic") and row.get("gene"):
        return "variant_disease"
    if row.get("gene"):
        return "gene_disease"
    return "gene_disease"


def build_goldlite(
    *,
    target_n: int = 220,
    seed: int = 42,
    harmonized_path: Optional[Path] = None,
) -> Dict[str, Any]:
    ensure_dirs()
    hpath = harmonized_path or (PROC / "kb_target_ledger_harmonized.csv")
    rows = _read_csv(hpath)
    pmid_rows = [r for r in rows if (r.get("pmid_track_a") or "").strip()]
    # Prefer CIViC-anchored rows with cached XML
    eligible: List[Dict[str, str]] = []
    for r in pmid_rows:
        pm = (r.get("pmid_track_a") or "").strip()
        if (CACHE / f"{pm}.xml").is_file():
            eligible.append(r)

    if len(eligible) < 20:
        # fall back: any row with pmid even if cache missing (still emit targets; retrieval metrics penalize)
        eligible = pmid_rows

    rng = random.Random(seed)
    by_fam: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in eligible:
        fam = _expected_pairing_family(r)
        by_fam[fam].append(r)

    families_order = ["gene_drug", "gene_disease", "variant_disease", "drug_disease", "variant_gene"]
    per_fam_cap = max(8, target_n // max(1, len(families_order)))

    picked: List[Dict[str, str]] = []
    for fam in families_order:
        pool = by_fam.get(fam, []) + by_fam.get("gene_drug", []) if fam != "gene_drug" else by_fam["gene_drug"]
        if fam == "drug_disease":
            pool = [r for r in eligible if _first_drug(r.get("drug_therapy", ""))]
        if fam == "variant_gene":
            pool = [r for r in eligible if r.get("variant_civic") and r.get("gene")]
        if not pool:
            continue
        rng.shuffle(pool)
        for r in pool[:per_fam_cap]:
            if r not in picked:
                picked.append(r)
        if len(picked) >= target_n:
            break

    if len(picked) < target_n:
        rest = [r for r in eligible if r not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: target_n - len(picked)])

    targets: List[Dict[str, str]] = []
    doc_links: List[Dict[str, str]] = []
    ev_cands: List[Dict[str, str]] = []

    for i, r in enumerate(picked[:target_n]):
        tid = f"GL_{i+1:04d}"
        pm = (r.get("pmid_track_a") or "").strip() or "NA"
        gene = r.get("gene", "")
        variant = (r.get("variant_civic") or r.get("variant_oncokb") or "").strip()
        drug = _first_drug(r.get("drug_therapy", ""))
        cancer = r.get("cancer_scope", "") or "NSCLC_family"
        fam = _expected_pairing_family(r)
        if fam == "variant_gene" or (variant and gene and fam == "variant_disease"):
            pass

        title, abstract = "", ""
        xp = CACHE / f"{pm}.xml"
        if xp.is_file():
            title, abstract = parse_pubmed_article(xp.read_text(encoding="utf-8", errors="replace"))
        si, best_sent, conf = _best_sentence_for_anchor(title, abstract, gene, drug, variant)
        sents = [s.strip() for s in SENT_SPLIT.split(f"{title}. {abstract}".strip()) if len(s.strip()) > 15]
        win2 = _neighbor_window(sents, min(si, len(sents) - 1), 1, 1) if sents else best_sent
        win4 = _neighbor_window(sents, min(si, len(sents) - 1), 2, 2) if sents else best_sent

        gold_s2 = _heuristic_gold_s2(r.get("assertion_family_civic", ""), gene, drug, variant, best_sent)
        abstract_has_support = "likely" if conf >= 0.7 else ("possible" if conf >= 0.55 else "uncertain")

        targets.append(
            {
                "goldlite_target_id": tid,
                "source_kb": "harmonized_anchor",
                "harmonized_key": r.get("harmonized_key", ""),
                "civic_record_id": r.get("civic_record_id", ""),
                "gene": gene,
                "variant_text": variant,
                "drug_primary": drug or "",
                "cancer_scope": cancer,
                "expected_pairing_family": fam,
                "evidence_category_proxy": r.get("assertion_family_civic", ""),
                "primary_pmid": pm,
                "heuristic_gold_s2_label": gold_s2,
                "construction_confidence": str(conf),
                "human_confirmed": "no",
                "heuristic_assisted": "yes",
                "notes": "Heuristic gold S2 and pairing family for oracle ceiling analysis — not clinical truth.",
            }
        )
        doc_links.append(
            {
                "goldlite_target_id": tid,
                "pmid": pm,
                "link_type": "CIViC_Track_A_citation",
                "cache_present": "yes" if xp.is_file() else "no",
            }
        )
        ev_cands.append(
            {
                "goldlite_target_id": tid,
                "pmid": pm,
                "sentence_idx": str(si),
                "evidence_sentence": best_sent,
                "context_C2_sentence_only": best_sent,
                "context_C3_pm1": win2,
                "context_C4_window": win4,
                "context_C1_abstract": abstract[:2000],
                "oracle_head_entity": gene,
                "oracle_tail_entity": drug if fam == "gene_drug" else (variant or cancer),
            }
        )

    fam_ct = Counter(t["expected_pairing_family"] for t in targets)
    src_ct = Counter("CIViC_harmonized" for t in targets)

    with open(PROC / "goldlite_audit_targets.csv", "w", newline="", encoding="utf-8") as f:
        if targets:
            w = csv.DictWriter(f, fieldnames=list(targets[0].keys()))
            w.writeheader()
            w.writerows(targets)
    with open(PROC / "goldlite_document_links.csv", "w", newline="", encoding="utf-8") as f:
        if doc_links:
            w = csv.DictWriter(f, fieldnames=list(doc_links[0].keys()))
            w.writeheader()
            w.writerows(doc_links)
    with open(PROC / "goldlite_evidence_candidates.csv", "w", newline="", encoding="utf-8") as f:
        if ev_cands:
            w = csv.DictWriter(f, fieldnames=list(ev_cands[0].keys()))
            w.writeheader()
            w.writerows(ev_cands)

    summary = {
        "target_count": len(targets),
        "pairing_family_distribution": dict(fam_ct),
        "source_distribution": dict(src_ct),
        "human_confirmed_fraction": 0.0,
        "heuristic_assisted_fraction": 1.0,
        "cache_hit_pmids": sum(1 for t in targets if (CACHE / f"{t['primary_pmid']}.xml").is_file()),
        "seed": seed,
        "caveat": "Gold-lite labels are heuristic proxies for ceiling analysis, not independent expert annotation.",
    }
    with open(MANIFESTS / "goldlite_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report = REPORTS / "goldlite_audit_construction.md"
    report.write_text(
        f"""# Gold-lite audit set construction

## Intent

Bounded ({len(targets)} targets) **audit-facing** slice to support retrieval, context, proposal, oracle, and linkage experiments.
This is **not** fully manually curated clinical gold.

## Provenance

- Sampled from `data/processed/kb_target_ledger_harmonized.csv` rows with **Track A PMID** where possible.
- PubMed XML read from `data/processed/pubmed_cache/` when present.

## Labeling policy

| Field | Nature |
|-------|--------|
| `heuristic_gold_s2_label` | Rule mapping from CIViC `assertion_family_civic` + token presence in best sentence |
| `expected_pairing_family` | Structural guess from gene / drug / variant columns |
| `construction_confidence` | Heuristic co-mention score in chosen sentence |

**Human confirmed:** none in this pass (`human_confirmed=no` for all rows).

## Family distribution

```json
{json.dumps(dict(fam_ct), indent=2)}
```

## Files

- `data/processed/goldlite_audit_targets.csv`
- `data/processed/goldlite_document_links.csv`
- `data/processed/goldlite_evidence_candidates.csv`
- `manifests/goldlite_audit_summary.json`

---
""",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    build_goldlite()
