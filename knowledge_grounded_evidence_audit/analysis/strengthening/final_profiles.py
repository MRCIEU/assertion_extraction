"""Decision-layer operating profiles JSON + markdown."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .paths import PROC, TABLES, REPORTS, ensure_dirs


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_final_profiles() -> Dict[str, Any]:
    ensure_dirs()
    ctx = _read_csv(TABLES / "context_variant_results.csv")
    oracle = _read_csv(TABLES / "oracle_upper_bound_results.csv")
    o3_all = [
        r
        for r in oracle
        if r.get("oracle_condition") == "O3_oracle_pair_sentence" and r.get("pairing_family") == "ALL"
    ]
    o3_sorted = sorted(o3_all, key=lambda x: float(x.get("macro_f1") or 0), reverse=True)

    c1 = [r for r in ctx if r.get("context_variant") == "C1_abstract"]
    by_vol = sorted(c1, key=lambda x: int(x.get("pred_nonnegative_count") or 0))

    # Variant-centric: prefer M021 by policy; override if another model has strictly higher O3 F1 on variant families
    fam = _read_csv(TABLES / "oracle_family_results.csv")
    var_rows = [r for r in fam if "variant" in (r.get("pairing_family") or "")]
    var_best = sorted(var_rows, key=lambda x: float(x.get("macro_f1") or 0), reverse=True)

    profiles = {
        "conservative_support_finding": {
            "intent": "Minimize false candidate assertions in audit queues.",
            "recommended_model": by_vol[0]["model_id"] if by_vol else "M015",
            "rationale_table": "model_operating_profile_summary.csv + lowest pred_nonnegative on C1_abstract",
        },
        "candidate_surfacing": {
            "intent": "Maximize reviewable non-negative hypotheses for triage.",
            "recommended_model": by_vol[-1]["model_id"] if by_vol else "S002",
            "rationale_table": "highest pred_nonnegative on C1_abstract",
        },
        "variant_centric_audit": {
            "intent": "Prioritize variant-bearing pairing families on gold-lite oracle slice.",
            "recommended_model": var_best[0]["model_id"] if var_best else "M021",
            "rationale_table": "oracle_family_results.csv variant_* rows",
        },
        "benchmark_balanced_default": {
            "intent": "Project default checkpoint line — still valid for benchmarks; downstream volume may be sparse.",
            "recommended_model": "M015",
            "rationale_table": "policy + observed zero-yield on first pass (documented separately)",
        },
        "upper_bound_classifier_under_oracle": {
            "intent": "Best achievable S2 alignment on heuristic gold under oracle pair+sentence.",
            "recommended_model": o3_sorted[0]["model_id"] if o3_sorted else "M021",
            "metric_macro_f1_O3": float(o3_sorted[0]["macro_f1"]) if o3_sorted else None,
            "rationale_table": "oracle_upper_bound_results.csv",
        },
        "disclaimer": "Profiles are audit-routing recommendations, not clinical deployment advice.",
    }

    path = PROC / "final_downstream_operating_profiles.json"
    path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")

    md = REPORTS / "final_downstream_operating_profiles.md"
    md.write_text(
        "# Final downstream operating profiles\n\n"
        + "\n".join(f"## {k}\n\n```json\n{json.dumps(v, indent=2)}\n```\n" for k, v in profiles.items() if k != "disclaimer")
        + f"\n## Disclaimer\n\n{profiles['disclaimer']}\n",
        encoding="utf-8",
    )
    return profiles
