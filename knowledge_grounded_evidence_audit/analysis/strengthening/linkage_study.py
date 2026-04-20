"""Linkage sensitivity L1–L3 using cached C1 predictions (no second forward pass)."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

_KG = Path(__file__).resolve().parent.parent.parent
if str(_KG) not in sys.path:
    sys.path.insert(0, str(_KG))

from run_pipeline import RawAssertion

from .linkage_modes import audit_outcome_permissive_gap, audit_outcome_variant, link_to_kb_variant
from .paths import PROC, TABLES, REPORTS, ensure_dirs


def run_linkage_sensitivity() -> Dict[str, Any]:
    ensure_dirs()
    harm_path = PROC / "kb_target_ledger_harmonized.csv"
    if not harm_path.is_file():
        return {"error": "missing harmonized ledger"}
    import csv as _csv

    with open(harm_path, newline="", encoding="utf-8") as f:
        harm = list(_csv.DictReader(f))

    gene_set = {h["gene"] for h in harm}
    cache = PROC / "strengthening_per_row_C1.jsonl"
    if not cache.is_file():
        return {"error": "run neural_strengthening first (strengthening_per_row_C1.jsonl missing)"}

    rows: List[Dict[str, str]] = []
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    results: List[Dict[str, Any]] = []
    shift_rows: List[Dict[str, str]] = []

    for lk in ["L1_strict", "L2_relaxed", "L3_grouped"]:
        out_tot: Counter[str] = Counter()
        for r in rows:
            pred = r["pred_label"]
            head = r["head"]
            tail = r["tail"]
            fam = "gene_disease"
            if pred == "DRUG_GENE_REGULATION":
                fam = "drug_gene"
            elif pred == "DRUG_DISEASE":
                fam = "drug_disease"
            elif pred == "VARIANT_GENE":
                fam = "variant_disease"
            a = RawAssertion(
                assertion_id=f"lk|{r['goldlite_target_id']}|{r['model_id']}",
                model_id=r["model_id"],
                doc_pmid="",
                sentence=r.get("sentence_excerpt", "")[:800],
                relation_family=fam if pred != "__NEGATIVE__" else "negative",
                entity_a={"type": "gene", "text": head, "normalized": head},
                entity_b={"type": "entity", "text": tail, "normalized": tail},
                confidence=float(r.get("confidence", "0.5")),
                provenance=["linkage_sensitivity", lk],
            )
            lvl, _, _ = link_to_kb_variant(a, harm, lk)
            in_g = head in gene_set
            if lk == "L2_relaxed":
                oc = audit_outcome_permissive_gap(a, lvl, in_g)
            else:
                oc = audit_outcome_variant(a, lvl, in_g)
            out_tot[oc] += 1

        results.append(
            {
                "linkage_variant": lk,
                "kb_supported_aligned": out_tot.get("kb_supported_aligned", 0),
                "kb_known_but_weak_current_support": out_tot.get("kb_known_but_weak_current_support", 0),
                "literature_supported_kb_absent_candidate": out_tot.get(
                    "literature_supported_kb_absent_candidate", 0
                ),
                "conflict_or_ambiguity": out_tot.get("conflict_or_ambiguity", 0),
                "unsupported_or_low_trust": out_tot.get("unsupported_or_low_trust", 0),
                "rows_scored": len(rows),
            }
        )

    # Shift vs L1
    base = next(x for x in results if x["linkage_variant"] == "L1_strict")
    for x in results:
        shift_rows.append(
            {
                "linkage_variant": x["linkage_variant"],
                "delta_kb_supported_aligned": str(
                    x["kb_supported_aligned"] - base["kb_supported_aligned"]
                ),
                "delta_kb_absent_candidate": str(
                    x["literature_supported_kb_absent_candidate"]
                    - base["literature_supported_kb_absent_candidate"]
                ),
                "delta_conflict": str(x["conflict_or_ambiguity"] - base["conflict_or_ambiguity"]),
            }
        )

    rules = {
        "L1_strict": "Production-style scoring; therapy literal or strong partial overlap.",
        "L2_relaxed": "Therapy family map (EGFR-TKI, ALK inhibitors, BRAF/MEK) + permissive audit confidence gates.",
        "L3_grouped": "Gene + lung context fallback to first harmonized row for same gene when no drug hit.",
        "disclaimer": "Relaxations are bounded and logged — not uncontrolled fuzzy KB merging.",
    }
    (PROC / "linkage_variant_rules.json").write_text(json.dumps(rules, indent=2), encoding="utf-8")

    with open(TABLES / "linkage_sensitivity_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow({k: str(r[k]) for k in r})

    with open(TABLES / "linkage_shift_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(shift_rows[0].keys()))
        w.writeheader()
        w.writerows(shift_rows)

    (REPORTS / "linkage_sensitivity_analysis.md").write_text(
        "## Linkage sensitivity\n\n"
        + "\n".join(
            f"- **{r['linkage_variant']}**: kb_supported_aligned={r['kb_supported_aligned']}, "
            f"conflict={r['conflict_or_ambiguity']}, kb_absent_candidate={r['literature_supported_kb_absent_candidate']}"
            for r in results
        )
        + "\n\nSee `data/processed/linkage_variant_rules.json` for explicit relaxation definitions.\n",
        encoding="utf-8",
    )
    return {"modes": len(results), "rows": len(rows)}
