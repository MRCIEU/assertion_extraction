"""PART 9 placeholders until Tier-1 evidence drives Tier-2 family pick."""

from __future__ import annotations

import json

from .paths import MANIFESTS, PROC, REPORTS, TABLES, ensure_dirs


def write_tier2_placeholders() -> None:
    ensure_dirs()
    frozen = PROC / "tier2_family_selection_decision.json"
    if frozen.is_file():
        try:
            cur = json.loads(frozen.read_text(encoding="utf-8"))
            if cur.get("status") == "frozen":
                return
        except json.JSONDecodeError:
            pass
    dec = {
        "status": "pending_tier1_review",
        "candidate_families_rule": "Pick 4–6 from tier1 downstream dispersion + policy relevance",
        "seeds": ["s01", "s02", "s03", "s04", "s05"],
        "note": "Populate after inspecting training_to_downstream_master_table.csv",
    }
    (MANIFESTS / "tier2_family_selection_decision.json").write_text(json.dumps(dec, indent=2), encoding="utf-8")
    (REPORTS / "tier2_family_selection.md").write_text(
        "# Tier-2 family selection\n\nPending Tier-1 sweep review. See `manifests/tier2_family_selection_decision.json`.\n",
        encoding="utf-8",
    )
    import csv as _csv

    with open(TABLES / "tier2_multiseed_downstream_results.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["status", "note"])
        w.writerow(["pending", "submit Tier-2 after family selection"])
    with open(TABLES / "tier2_seed_stability_results.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["status", "note"])
        w.writerow(["pending", "requires tier2_multiseed_downstream_results"])
    (REPORTS / "tier2_multiseed_analysis.md").write_text(
        "# Tier-2 multi-seed analysis\n\n**Pending.**\n",
        encoding="utf-8",
    )
