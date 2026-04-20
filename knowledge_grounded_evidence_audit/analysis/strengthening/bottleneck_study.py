"""Ablation-based bottleneck attribution (descriptive, not causal)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .paths import TABLES, MANIFESTS, REPORTS, ensure_dirs


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(csv_rows: List[Dict[str, str]], key: str, val_key: str, want: str) -> float:
    for r in csv_rows:
        if r.get(key) == want:
            try:
                return float(r[val_key])
            except (KeyError, ValueError):
                return 0.0
    return 0.0


def run_bottleneck() -> Dict[str, Any]:
    ensure_dirs()
    retr = _read_csv(TABLES / "retrieval_variant_results.csv")
    ctx = _read_csv(TABLES / "context_variant_results.csv")
    prop = _read_csv(TABLES / "proposal_density_table.csv")
    oracle = _read_csv(TABLES / "oracle_upper_bound_results.csv")
    link = _read_csv(TABLES / "linkage_sensitivity_results.csv")

    r1 = _f(retr, "retrieval_variant", "value", "R1_current")
    r2 = _f(retr, "retrieval_variant", "value", "R2_expanded_lexical")
    ret_gap = max(0.0, r2 - r1)

    # Context: max F1 - min F1 across contexts per model averaged
    by_model: Dict[str, List[float]] = {}
    for r in ctx:
        try:
            by_model.setdefault(r["model_id"], []).append(float(r.get("macro_f1_vs_heuristic_gold") or 0))
        except ValueError:
            continue
    ctx_spread = (
        sum(max(v) - min(v) for v in by_model.values() if v) / max(1, len(by_model))
        if by_model
        else 0.0
    )

    p1 = _f(prop, "proposal_variant", "mean_recall_on_goldlite", "P1_gene_drug")
    p5 = _f(prop, "proposal_variant", "mean_recall_on_goldlite", "P5_oracle_pair")
    prop_gap = max(0.0, p5 - p1)

    o1 = [float(r["macro_f1"]) for r in oracle if r.get("oracle_condition") == "O1_oracle_pair" and r.get("pairing_family") == "ALL"]
    o3 = [float(r["macro_f1"]) for r in oracle if r.get("oracle_condition") == "O3_oracle_pair_sentence" and r.get("pairing_family") == "ALL"]
    model_ceiling_delta = (sum(o3) / max(1, len(o3)) - sum(o1) / max(1, len(o1))) if o1 and o3 else 0.0

    l1_sup = 0.0
    l2_sup = 0.0
    for r in link:
        if r.get("linkage_variant") == "L1_strict":
            l1_sup = float(r.get("kb_supported_aligned", 0) or 0)
        if r.get("linkage_variant") == "L2_relaxed":
            l2_sup = float(r.get("kb_supported_aligned", 0) or 0)
    link_delta = max(0.0, l2_sup - l1_sup)

    # Normalize crude 0-1 "shares"
    raw = [ret_gap, ctx_spread, prop_gap, abs(model_ceiling_delta), link_delta / max(1, l1_sup + 1)]
    s = sum(raw) or 1.0
    shares = [x / s for x in raw]

    rows = [
        {
            "bottleneck_component": "retrieval_manifest_coverage",
            "relative_share_proxy": round(shares[0], 4),
            "evidence_metric": f"R2-R1 hit rate delta ≈ {ret_gap:.4f}",
            "uncertainty": "Proxy uses cached text, not live expanded PubMed crawl.",
        },
        {
            "bottleneck_component": "evidence_localization_context",
            "relative_share_proxy": round(shares[1], 4),
            "evidence_metric": f"mean per-model max-min macro_F1 across C1–C5 ≈ {ctx_spread:.4f}",
            "uncertainty": "Heuristic gold S2 inflates absolute F1 interpretation.",
        },
        {
            "bottleneck_component": "proposal_space",
            "relative_share_proxy": round(shares[2], 4),
            "evidence_metric": f"P5-P1 mean recall delta ≈ {prop_gap:.4f}",
            "uncertainty": "P5 oracle is definitional; gap is upper bound on proposal loss.",
        },
        {
            "bottleneck_component": "checkpoint_classification",
            "relative_share_proxy": round(shares[3], 4),
            "evidence_metric": f"mean Δ macro_F1 O3 vs O1 ≈ {model_ceiling_delta:.4f}",
            "uncertainty": "If O3 still low, schema/supervision mismatch plausible.",
        },
        {
            "bottleneck_component": "kb_linkage_strictness",
            "relative_share_proxy": round(shares[4], 4),
            "evidence_metric": f"kb_supported count gain L2 vs L1 ≈ {link_delta}",
            "uncertainty": "Counts are on gold-lite C1 prediction cache, not full corpus ledger.",
        },
    ]

    with open(TABLES / "bottleneck_attribution_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "method": "ablation_delta_normalization",
        "strong_inference": [
            "Proposal oracle P5 vs P1 quantifies how much gene×drug inventory alone can suppress recall.",
            "Context spread across C1–C4 localizes abstract-only penalty when gold-lite F1 rises in sentence windows.",
        ],
        "uncertain": [
            "Retrieval R2 is lexical proxy only.",
            "Linkage shifts apply to cached predictions, not re-run full pipeline.",
        ],
        "rows": rows,
    }
    with open(MANIFESTS / "bottleneck_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    (REPORTS / "bottleneck_attribution_analysis.md").write_text(
        "# Bottleneck attribution\n\n"
        + "\n".join(
            f"- **{r['bottleneck_component']}** (share_proxy={r['relative_share_proxy']}): {r['evidence_metric']}"
            for r in rows
        )
        + "\n\nShares are **normalized ablation magnitudes**, not causal Shapley decomposition.\n",
        encoding="utf-8",
    )
    return summary
