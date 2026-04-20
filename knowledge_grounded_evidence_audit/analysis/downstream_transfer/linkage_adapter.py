"""Linkage scoring for transfer sweep (reuses strengthening linkage_modes)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_KG = Path(__file__).resolve().parent.parent.parent
if str(_KG) not in sys.path:
    sys.path.insert(0, str(_KG))

from run_pipeline import RawAssertion

from analysis.strengthening.linkage_modes import audit_outcome_permissive_gap, audit_outcome_variant, link_to_kb_variant


def classify_linkage_outcome(
    assertion: RawAssertion,
    harm_rows: List[Dict[str, str]],
    link_mode: str,
) -> Tuple[str, str]:
    lvl, _, _ = link_to_kb_variant(assertion, harm_rows, link_mode)
    in_g = (assertion.entity_a.get("normalized") or "") in {h["gene"] for h in harm_rows}
    if link_mode == "L2_relaxed":
        oc = audit_outcome_permissive_gap(assertion, lvl, in_g)
    else:
        oc = audit_outcome_variant(assertion, lvl, in_g)
    return oc, lvl


def sweep_linkage_modes(
    cache: List[Dict[str, str]],
    harm: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    from collections import Counter

    results: List[Dict[str, str]] = []
    shift: List[Dict[str, str]] = []
    models = sorted({r["model_base_id"] for r in cache}) if cache else []
    l1_kb: Dict[str, int] = {}
    for mid in models:
        sub = [r for r in cache if r["model_base_id"] == mid]
        for lk in ("L1_strict", "L2_relaxed", "L3_grouped"):
            tot: Counter[str] = Counter()
            for r in sub:
                pred = r["pred"]
                head, tail = r["head"], r["tail"]
                fam = "gene_disease"
                if pred == "DRUG_GENE_REGULATION":
                    fam = "drug_gene"
                elif pred == "VARIANT_GENE":
                    fam = "variant_disease"
                elif pred == "DRUG_DISEASE":
                    fam = "drug_disease"
                a = RawAssertion(
                    assertion_id=f"lk|{r['goldlite_target_id']}|{mid}",
                    model_id=f"{mid}_s{r['model_seed_id']}",
                    doc_pmid="",
                    sentence=r.get("sentence_excerpt", "")[:800],
                    relation_family=fam if pred != "__NEGATIVE__" else "negative",
                    entity_a={"type": "gene", "text": head, "normalized": head},
                    entity_b={"type": "entity", "text": tail, "normalized": tail},
                    confidence=float(r.get("confidence", "0.5")),
                    provenance=["transfer_linkage", lk],
                )
                oc, _ = classify_linkage_outcome(a, harm, lk)
                tot[oc] += 1
            results.append(
                {
                    "model_base_id": mid,
                    "linkage_variant": lk,
                    "kb_supported_aligned": str(tot.get("kb_supported_aligned", 0)),
                    "conflict_or_ambiguity": str(tot.get("conflict_or_ambiguity", 0)),
                    "literature_kb_absent": str(tot.get("literature_supported_kb_absent_candidate", 0)),
                    "unsupported_or_low_trust": str(tot.get("unsupported_or_low_trust", 0)),
                    "rows": str(len(sub)),
                }
            )
            if lk == "L1_strict":
                l1_kb[mid] = int(tot.get("kb_supported_aligned", 0))
        for lk in ("L2_relaxed", "L3_grouped"):
            x = next((r for r in results if r["model_base_id"] == mid and r["linkage_variant"] == lk), None)
            if x and mid in l1_kb:
                shift.append(
                    {
                        "model_base_id": mid,
                        "linkage_variant": lk,
                        "delta_kb_supported_vs_L1": str(int(x["kb_supported_aligned"]) - l1_kb[mid]),
                    }
                )
    return results, shift
