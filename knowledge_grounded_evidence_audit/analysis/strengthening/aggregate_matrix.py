"""Model × setting matrix and operating-profile summaries from tables."""

from __future__ import annotations

import csv
from typing import Any, Dict, List

from .paths import TABLES, PROC, REPORTS, ensure_dirs


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_aggregate() -> Dict[str, Any]:
    ensure_dirs()
    ctx = _read_csv(TABLES / "context_variant_results.csv")
    oracle = _read_csv(TABLES / "oracle_upper_bound_results.csv")
    oracle_all = [r for r in oracle if r.get("pairing_family") == "ALL"]
    retr = _read_csv(TABLES / "retrieval_variant_results.csv")
    prop = _read_csv(TABLES / "proposal_density_table.csv")
    link = _read_csv(TABLES / "linkage_sensitivity_results.csv")

    matrix: List[Dict[str, str]] = []
    for r in ctx:
        setting = f"baseline_context_{r['context_variant']}"
        matrix.append(
            {
                "model_id": r["model_id"],
                "setting_label": setting,
                "macro_f1_heuristic": r.get("macro_f1_vs_heuristic_gold", ""),
                "kb_supported_aligned": r.get("kb_supported_aligned", ""),
                "conflict_or_ambiguity": r.get("conflict_or_ambiguity", ""),
                "source_table": "context_variant_results.csv",
            }
        )

    for r in oracle_all:
        matrix.append(
            {
                "model_id": r["model_id"],
                "setting_label": f"oracle_{r['oracle_condition']}",
                "macro_f1_heuristic": r.get("macro_f1", ""),
                "kb_supported_aligned": "",
                "conflict_or_ambiguity": "",
                "source_table": "oracle_upper_bound_results.csv",
            }
        )

    # Best-proxy rows (single row per model — pick max F1 context)
    by_m: Dict[str, List[Dict[str, str]]] = {}
    for r in ctx:
        by_m.setdefault(r["model_id"], []).append(r)
    for mid, rows in by_m.items():
        best = max(rows, key=lambda x: float(x.get("macro_f1_vs_heuristic_gold") or 0))
        matrix.append(
            {
                "model_id": mid,
                "setting_label": "best_context_proxy",
                "macro_f1_heuristic": best["macro_f1_vs_heuristic_gold"],
                "kb_supported_aligned": best.get("kb_supported_aligned", ""),
                "conflict_or_ambiguity": best.get("conflict_or_ambiguity", ""),
                "source_table": "derived_max_context_f1",
            }
        )

    with open(TABLES / "model_setting_matrix_results.csv", "w", newline="", encoding="utf-8") as f:
        if matrix:
            w = csv.DictWriter(f, fieldnames=list(matrix[0].keys()))
            w.writeheader()
            w.writerows(matrix)

    # Operating profile summary — rank models on oracle O3 and conservative ambiguity
    o3 = [r for r in oracle_all if r.get("oracle_condition") == "O3_oracle_pair_sentence"]
    o3_sorted = sorted(o3, key=lambda x: float(x.get("macro_f1") or 0), reverse=True)
    amb_low = sorted(ctx, key=lambda x: int(x.get("conflict_or_ambiguity") or 0))
    profile_rows = []
    if o3_sorted:
        profile_rows.append(
            {
                "profile_name": "upper_bound_classifier_oracle",
                "recommended_model": o3_sorted[0]["model_id"],
                "metric": "macro_f1_O3_oracle_pair_sentence",
                "value": o3_sorted[0].get("macro_f1", ""),
            }
        )
    # M015 conservative: lowest non-negative count proxy — use pred count from context C1
    c1 = [r for r in ctx if r.get("context_variant") == "C1_abstract"]
    c1_by_pred = sorted(c1, key=lambda x: int(x.get("pred_nonnegative_count") or 0))
    if c1_by_pred:
        profile_rows.append(
            {
                "profile_name": "conservative_support_finding",
                "recommended_model": c1_by_pred[0]["model_id"],
                "metric": "lowest_pred_nonnegative_C1_abstract",
                "value": c1_by_pred[0].get("pred_nonnegative_count", ""),
            }
        )
        profile_rows.append(
            {
                "profile_name": "candidate_surfacing",
                "recommended_model": c1_by_pred[-1]["model_id"],
                "metric": "highest_pred_nonnegative_C1_abstract",
                "value": c1_by_pred[-1].get("pred_nonnegative_count", ""),
            }
        )

    with open(TABLES / "model_operating_profile_summary.csv", "w", newline="", encoding="utf-8") as f:
        if profile_rows:
            w = csv.DictWriter(f, fieldnames=list(profile_rows[0].keys()))
            w.writeheader()
            w.writerows(profile_rows)

    # Pairing profile from oracle_family_results
    fam = _read_csv(TABLES / "oracle_family_results.csv")
    with open(TABLES / "model_pairing_profile_results.csv", "w", newline="", encoding="utf-8") as f:
        if fam:
            w = csv.DictWriter(f, fieldnames=list(fam[0].keys()))
            w.writeheader()
            w.writerows(fam)

    (REPORTS / "model_operating_profile_analysis.md").write_text(
        _md_profiles(profile_rows, o3_sorted[:3], c1_by_pred[:2] if c1_by_pred else []),
        encoding="utf-8",
    )
    return {"matrix_rows": len(matrix), "profiles": len(profile_rows)}


def _md_profiles(
    prof: List[Dict[str, str]],
    top_o3: List[Dict[str, str]],
    low_vol: List[Dict[str, str]],
) -> str:
    lines = ["# Model operating profiles (downstream audit)", ""]
    for p in prof:
        lines.append(f"- **{p['profile_name']}** → `{p['recommended_model']}` ({p['metric']}={p['value']})")
    lines.append("\n## Oracle O3 leaders\n")
    for r in top_o3:
        lines.append(f"- {r['model_id']}: macro_F1={r.get('macro_f1')}")
    lines.append("\n## Low assertion volume on C1 (conservative proxy)\n")
    for r in low_vol:
        lines.append(f"- {r['model_id']}: pred_nonnegative={r.get('pred_nonnegative_count')}")
    lines.append("\nNo single universal winner — use profile-specific routing.\n")
    return "\n".join(lines)
