"""Aggregate Tier-2 raw rows into family-level stability tables."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any, Dict, List

from .paths import MANIFESTS, REPORTS, TABLES, ensure_dirs


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def run_tier2_aggregate() -> Dict[str, Any]:
    ensure_dirs()
    raw = _read_csv(TABLES / "tier2_multiseed_raw.csv")
    if not raw:
        return {"error": "no tier2_multiseed_raw.csv"}

    agg_rows: List[Dict[str, str]] = []
    groups: Dict[tuple, List[Dict[str, str]]] = {}
    for r in raw:
        groups.setdefault((r["model_base_id"], r["downstream_setting_id"]), []).append(r)

    for (base, sid), lst in sorted(groups.items()):
        nn = [_f(x["pred_nonnegative_count"]) for x in lst]
        rates = [_f(x["pred_nonnegative_rate"]) for x in lst]
        sr = [_f(x["support_ready_rate"]) for x in lst]
        amb = [_f(x["ambiguity_rate"]) for x in lst]
        mf1 = [_f(x["macro_f1_heuristic"]) for x in lst]

        def _ms(xs: List[float]) -> tuple:
            clean = [x for x in xs if x == x]
            if not clean:
                return "", ""
            sd = statistics.stdev(clean) if len(clean) > 1 else 0.0
            return str(round(statistics.mean(clean), 6)), str(round(sd, 6))

        m_nn, s_nn = _ms(nn)
        m_rt, s_rt = _ms(rates)
        m_sr, s_sr = _ms(sr)
        m_am, s_am = _ms(amb)
        m_mf, s_mf = _ms(mf1)
        o3 = lst[0].get("oracle_o3_macro_f1_all") or ""

        agg_rows.append(
            {
                "model_base_id": base,
                "downstream_setting_id": sid,
                "n_seeds": str(len(lst)),
                "pred_nonnegative_mean": m_nn,
                "pred_nonnegative_std": s_nn,
                "pred_nonnegative_rate_mean": m_rt,
                "pred_nonnegative_rate_std": s_rt,
                "support_ready_rate_mean": m_sr,
                "support_ready_rate_std": s_sr,
                "ambiguity_rate_mean": m_am,
                "ambiguity_rate_std": s_am,
                "macro_f1_heuristic_mean": m_mf,
                "macro_f1_heuristic_std": s_mf,
                "oracle_o3_macro_f1_all_note": o3 if sid == "S3_oracle_like" else "",
            }
        )

    if agg_rows:
        with open(TABLES / "tier2_multiseed_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            w.writeheader()
            w.writerows(agg_rows)

    stab: List[Dict[str, str]] = []
    for base in sorted({r["model_base_id"] for r in raw}):
        for sid in sorted({r["downstream_setting_id"] for r in raw}):
            sub = [r for r in raw if r["model_base_id"] == base and r["downstream_setting_id"] == sid]
            if len(sub) < 1:
                continue
            nn = [_f(x["pred_nonnegative_count"]) for x in sub]
            stab.append(
                {
                    "model_base_id": base,
                    "setting": sid,
                    "n_seeds": str(len(sub)),
                    "pred_nonnegative_min": str(int(min(nn))),
                    "pred_nonnegative_max": str(int(max(nn))),
                    "pred_nonnegative_mean": str(round(statistics.mean(nn), 4)),
                    "pred_nonnegative_std": str(round(statistics.stdev(nn), 4)) if len(nn) > 1 else "0.0",
                }
            )

    if stab:
        with open(TABLES / "tier2_seed_stability_table.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(stab[0].keys()))
            w.writeheader()
            w.writerows(stab)

    body = (
        "# Tier-2 execution summary\n\n"
        f"- Raw rows: {len(raw)}\n"
        f"- Aggregated groups: {len(agg_rows)}\n"
        f"- Stability rows: {len(stab)}\n"
        "- See `reports/tables/tier2_multiseed_results.csv`, `tier2_seed_stability_table.csv`.\n"
    )
    (REPORTS / "tier2_execution_summary.md").write_text(body, encoding="utf-8")

    return {"aggregated": len(agg_rows), "stability_rows": len(stab)}
