"""PART 4 — gold-lite reassessment, balance tables, eval subsets."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .paths import MANIFESTS, PROC, REPORTS, ensure_dirs


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_goldlite_reassessment() -> None:
    ensure_dirs()
    targets = _read_csv(PROC / "goldlite_audit_targets.csv")
    if not targets:
        return

    fam_ct = Counter(t.get("expected_pairing_family", "") for t in targets)
    anchor_ct = Counter(t.get("harmonized_key", "") for t in targets)

    fam_rows = [{"pairing_family": k, "count": str(v)} for k, v in fam_ct.most_common()]
    with open(REPORTS / "tables" / "goldlite_family_balance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fam_rows[0].keys()))
        w.writeheader()
        w.writerows(fam_rows)

    anch_rows = [{"harmonized_key": k, "count": str(v)} for k, v in anchor_ct.most_common()][:80]
    with open(REPORTS / "tables" / "goldlite_anchor_balance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(anch_rows[0].keys()))
        w.writeheader()
        w.writerows(anch_rows)

    decision = {
        "keep_current_goldlite": True,
        "reason": "165 targets with full PMID cache; stratification limited by CIViC harmonized diversity.",
        "imbalance_note": "gene_drug dominates; variant_disease small — subset analysis required.",
        "human_label_plan": "future_manual_spot_check_optional",
    }
    (MANIFESTS / "goldlite_rebuild_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    (REPORTS / "goldlite_reassessment.md").write_text(
        "# Gold-lite reassessment\n\n"
        f"- **Targets:** {len(targets)}\n"
        f"- **Family distribution:** {dict(fam_ct)}\n"
        f"- **Decision:** keep slice; document imbalance; use subset registry for transfer metrics.\n",
        encoding="utf-8",
    )

    # Subsets
    subsets: Dict[str, List[str]] = {
        "gene_drug": [],
        "gene_disease": [],
        "drug_disease": [],
        "variant_disease": [],
        "variant_gene": [],
        "retrieval_sensitive": [],
        "proposal_sensitive": [],
        "linkage_sensitive": [],
    }
    for t in targets:
        tid = t.get("goldlite_target_id", "")
        fam = t.get("expected_pairing_family", "")
        if fam in subsets:
            subsets[fam].append(tid)
        if fam == "gene_drug":
            subsets["proposal_sensitive"].append(tid)
        if fam == "variant_disease":
            subsets["linkage_sensitive"].append(tid)
        subsets["retrieval_sensitive"].append(tid)

    reg = {"subsets": {k: v for k, v in subsets.items()}, "notes": "Heuristic tagging; refine as labels improve."}
    (PROC / "goldlite_eval_subset_registry.json").write_text(json.dumps(reg, indent=2), encoding="utf-8")

    sub_rows = [{"subset_id": k, "size": str(len(v)), "example_target_ids": "|".join(v[:5])} for k, v in subsets.items()]
    with open(PROC / "goldlite_eval_subsets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sub_rows[0].keys()))
        w.writeheader()
        w.writerows(sub_rows)
