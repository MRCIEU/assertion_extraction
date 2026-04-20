#!/usr/bin/env python3.11
"""
Phase A re-analysis, stage 1 — sanity precompute on existing saved data.

Runs WITHOUT the Step 0.5 logit re-run. Uses the already-saved
`kb_surface_targets.jsonl` per Phase A run (which contains `pred_label`,
`p_negative`, and an S2-vocab `p_expected`). From this we can compute:

  - Method A (set_valued projection)
  - Method A (single_label projection)
  - Method C at fixed rejection fractions (10 %, 25 %, 50 %)
    under either projection (hit uses pred_label; rejection uses p_negative)
  - KB_surface_mean (legacy, for sanity comparison with Phase A's original
    aggregate.json)

We CANNOT compute from saved data:

  - Method B (needs the probability of the schema-aligned expected label,
    which for variant_disease in S_pair/S_mech and for gene_drug in
    S_mech-set_valued is NOT the saved `p_expected`; Step 0.5 required)
  - Method A / C strictly-correctly under the saved data for
    `single_label`-only on variant_disease+AG targets in S_pair/S_mech
    (actually these ARE correct — pred_label is argmax in schema vocab)

Output:
  fine_tuning_experiments/schema_exp/phase_a_reanalysis_precompute.json
  fine_tuning_experiments/schema_exp/phase_a_reanalysis_precompute.md
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fine_tuning_experiments.schema_exp.eval.schema_expected_label import (  # noqa: E402
    is_hit,
    resolve_family,
    schema_expected_label_set,
)

FT_DATA_ROOT = Path(os.environ.get(
    "PROJECT_1_DATA_ROOT",
    "/lus/lfs1aip2/projects/b5ac/project_1",
)).resolve()
RUNS_ROOT = FT_DATA_ROOT / "fine_tuning_experiments" / "runs" / "schema_exp"
GOLDLITE_CSV = FT_DATA_ROOT / "knowledge_grounded_evidence_audit" / "data" / "processed" / "goldlite_audit_targets.csv"

OUT_DIR = SCRIPT_DIR.parent  # fine_tuning_experiments/schema_exp/

SCHEMA_NAME_MAP = {"Sflat": "S_flat", "Spair": "S_pair", "Smech": "S_mech"}
PROJECTIONS = ("set_valued", "single_label")
REJECTION_FRACTIONS = (0.10, 0.25, 0.50)


def load_civic_targets() -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(open(GOLDLITE_CSV)))
    return {r["goldlite_target_id"]: r for r in rows}


def method_a_hits(
    targets: list[dict], civic: dict[str, dict], schema: str, projection: str,
) -> tuple[float, int, int]:
    hits = 0
    n_eval = 0
    n_total = 0
    for t in targets:
        n_total += 1
        civic_row = civic.get(t["target_id"])
        if civic_row is None:
            continue
        _, conf = resolve_family(civic_row, strategy="primary")
        if conf == "unmapped":
            continue
        n_eval += 1
        if is_hit(t["pred_label"], civic_row, schema, "primary", projection):
            hits += 1
    return (hits / n_eval if n_eval else float("nan"), hits, n_eval)


def method_c_rejection_table(
    targets: list[dict], civic: dict[str, dict], schema: str, projection: str,
) -> dict[float, dict[str, float]]:
    # Evaluable targets (drop unmapped)
    rows = []
    for t in targets:
        civic_row = civic.get(t["target_id"])
        if civic_row is None:
            continue
        _, conf = resolve_family(civic_row, strategy="primary")
        if conf == "unmapped":
            continue
        hit = 1 if is_hit(t["pred_label"], civic_row, schema, "primary", projection) else 0
        rows.append((t["p_negative"], hit))
    rows.sort(key=lambda r: -r[0])  # highest P(NEG) first = reject first
    n = len(rows)
    if n == 0:
        return {}
    out: dict[float, dict[str, float]] = {}
    for rf in REJECTION_FRACTIONS:
        n_reject = int(round(rf * n))
        kept = rows[n_reject:]
        if not kept:
            out[rf] = {"precision_kept": float("nan"), "n_kept": 0, "n_reject": n_reject}
            continue
        prec = sum(h for _, h in kept) / len(kept)
        out[rf] = {"precision_kept": prec, "n_kept": len(kept), "n_reject": n_reject}
    # Also point at rf=0 (no rejection = Method A)
    prec0 = sum(h for _, h in rows) / n
    out[0.0] = {"precision_kept": prec0, "n_kept": n, "n_reject": 0}
    return out


def load_run(run_dir: Path) -> dict[str, Any] | None:
    manifest = run_dir / "run_manifest.json"
    targets_path = run_dir / "eval" / "kb_surface_targets.jsonl"
    if not (manifest.exists() and targets_path.exists()):
        return None
    m = json.loads(manifest.read_text())
    targets = [json.loads(l) for l in targets_path.read_text().splitlines() if l.strip()]
    return {
        "run_id": m["experiment_id"],
        "encoder": m["phase_a_metadata"]["encoder_key"],
        "schema_key": m["phase_a_metadata"]["schema_key"],
        "schema_id": m["schema_id"],
        "seed": m["seed"],
        "targets": targets,
    }


def mean_se(vs: list[float]) -> tuple[float, float, int]:
    clean = [v for v in vs if v is not None and not math.isnan(v)]
    if not clean:
        return float("nan"), float("nan"), 0
    m = statistics.mean(clean)
    se = statistics.stdev(clean) / math.sqrt(len(clean)) if len(clean) > 1 else 0.0
    return m, se, len(clean)


def main() -> None:
    print(f"Reading runs from {RUNS_ROOT}")
    civic = load_civic_targets()
    print(f"Loaded {len(civic)} CIViC targets")

    run_dirs = sorted([d for d in RUNS_ROOT.glob("PA_*") if d.is_dir()])
    print(f"Found {len(run_dirs)} run directories")

    per_run: list[dict[str, Any]] = []
    for d in run_dirs:
        r = load_run(d)
        if r is None:
            print(f"  SKIP (missing artifacts): {d.name}")
            continue
        schema = r["schema_id"]
        row: dict[str, Any] = {
            "run_id": r["run_id"],
            "encoder": r["encoder"],
            "schema_key": r["schema_key"],
            "schema_id": schema,
            "seed": r["seed"],
        }
        # Legacy KB_surface_mean (diagnostic)
        ps_neg = [t["p_negative"] for t in r["targets"]]
        row["kb_surface_mean_legacy"] = 1.0 - statistics.mean(ps_neg)

        for proj in PROJECTIONS:
            hit_rate, hits, n_eval = method_a_hits(r["targets"], civic, schema, proj)
            row[f"method_a_{proj}"] = hit_rate
            row[f"method_a_{proj}_hits"] = hits
            row[f"method_a_{proj}_n_eval"] = n_eval
            rej_table = method_c_rejection_table(r["targets"], civic, schema, proj)
            for rf, stats in rej_table.items():
                row[f"method_c_{proj}_precision_at_rej_{int(rf*100):02d}"] = stats["precision_kept"]
        per_run.append(row)

    print(f"\nProcessed {len(per_run)} runs")

    # Aggregate by schema, and by (encoder, schema)
    by_schema: dict[str, list[dict]] = defaultdict(list)
    by_enc_sch: dict[tuple, list[dict]] = defaultdict(list)
    for r in per_run:
        by_schema[r["schema_id"]].append(r)
        by_enc_sch[(r["encoder"], r["schema_id"])].append(r)

    METRIC_KEYS = [
        "kb_surface_mean_legacy",
        "method_a_set_valued",
        "method_a_single_label",
        "method_c_set_valued_precision_at_rej_00",
        "method_c_set_valued_precision_at_rej_10",
        "method_c_set_valued_precision_at_rej_25",
        "method_c_set_valued_precision_at_rej_50",
        "method_c_single_label_precision_at_rej_00",
        "method_c_single_label_precision_at_rej_10",
        "method_c_single_label_precision_at_rej_25",
        "method_c_single_label_precision_at_rej_50",
    ]

    agg_schema: dict[str, dict[str, Any]] = {}
    for sch, rs in by_schema.items():
        cell: dict[str, Any] = {"n": len(rs)}
        for k in METRIC_KEYS:
            m, se, n = mean_se([r[k] for r in rs])
            cell[k] = {"mean": m, "se": se, "n": n}
        agg_schema[sch] = cell

    agg_enc_sch: dict[str, dict[str, Any]] = {}
    for (enc, sch), rs in by_enc_sch.items():
        cell: dict[str, Any] = {"n": len(rs)}
        for k in METRIC_KEYS:
            m, se, n = mean_se([r[k] for r in rs])
            cell[k] = {"mean": m, "se": se, "n": n}
        agg_enc_sch[f"{enc}_{sch}"] = cell

    # Write JSON
    out_json = OUT_DIR / "phase_a_reanalysis_precompute.json"
    out_json.write_text(json.dumps({
        "n_runs": len(per_run),
        "projection_modes": list(PROJECTIONS),
        "rejection_fractions": list(REJECTION_FRACTIONS),
        "per_run": per_run,
        "by_schema": agg_schema,
        "by_encoder_schema": agg_enc_sch,
    }, indent=2))
    print(f"\nWrote {out_json}")

    # Write Markdown summary
    lines = ["# Phase A re-analysis — stage 1 sanity precompute", ""]
    lines.append(f"**Runs processed:** {len(per_run)} / 120")
    lines.append(f"**CIViC evaluable targets per run:** 162 (primary strategy; 3 variant_gene unmapped)")
    lines.append("")
    lines.append("## Method A — argmax hit rate (pooled across 4 encoders × 10 seeds, n=40 per schema)")
    lines.append("")
    lines.append("| schema | set_valued | single_label | Δ (sv − sl) | legacy KB_surface_mean |")
    lines.append("|---|---|---|---|---|")
    for sch in ("S_flat", "S_pair", "S_mech"):
        c = agg_schema.get(sch)
        if not c: continue
        sv = c["method_a_set_valued"]
        sl = c["method_a_single_label"]
        lg = c["kb_surface_mean_legacy"]
        lines.append(
            f"| {sch} "
            f"| {sv['mean']:.4f} ± {sv['se']:.4f} "
            f"| {sl['mean']:.4f} ± {sl['se']:.4f} "
            f"| {sv['mean'] - sl['mean']:+.4f} "
            f"| {lg['mean']:.4f} ± {lg['se']:.4f} |"
        )
    lines.append("")
    lines.append("## Method C — precision-kept at given rejection fraction (set_valued)")
    lines.append("")
    lines.append("| schema | reject 0% (=Method A) | reject 10% | reject 25% | reject 50% |")
    lines.append("|---|---|---|---|---|")
    for sch in ("S_flat", "S_pair", "S_mech"):
        c = agg_schema.get(sch)
        if not c: continue
        lines.append(
            f"| {sch} "
            f"| {c['method_c_set_valued_precision_at_rej_00']['mean']:.4f} "
            f"| {c['method_c_set_valued_precision_at_rej_10']['mean']:.4f} "
            f"| {c['method_c_set_valued_precision_at_rej_25']['mean']:.4f} "
            f"| {c['method_c_set_valued_precision_at_rej_50']['mean']:.4f} |"
        )
    lines.append("")
    lines.append("## Method C — precision-kept at given rejection fraction (single_label)")
    lines.append("")
    lines.append("| schema | reject 0% (=Method A) | reject 10% | reject 25% | reject 50% |")
    lines.append("|---|---|---|---|---|")
    for sch in ("S_flat", "S_pair", "S_mech"):
        c = agg_schema.get(sch)
        if not c: continue
        lines.append(
            f"| {sch} "
            f"| {c['method_c_single_label_precision_at_rej_00']['mean']:.4f} "
            f"| {c['method_c_single_label_precision_at_rej_10']['mean']:.4f} "
            f"| {c['method_c_single_label_precision_at_rej_25']['mean']:.4f} "
            f"| {c['method_c_single_label_precision_at_rej_50']['mean']:.4f} |"
        )
    lines.append("")
    lines.append("## Schema ranking summary")
    lines.append("")
    for metric_label, key in [
        ("Method A set_valued", "method_a_set_valued"),
        ("Method A single_label", "method_a_single_label"),
        ("Method C set_valued @ reject 50%", "method_c_set_valued_precision_at_rej_50"),
        ("Method C single_label @ reject 50%", "method_c_single_label_precision_at_rej_50"),
    ]:
        ranking = sorted(
            agg_schema.items(),
            key=lambda kv: -kv[1][key]["mean"],
        )
        lines.append(
            f"- **{metric_label}**: "
            + " > ".join(f"{s} ({c[key]['mean']:.3f})" for s, c in ranking)
        )
    lines.append("")
    lines.append("## Encoder × schema breakdown (Method A set_valued)")
    lines.append("")
    lines.append("| encoder | S_flat | S_pair | S_mech |")
    lines.append("|---|---|---|---|")
    for enc in ("RB", "PB", "BL", "PL"):
        cells = []
        for sch in ("S_flat", "S_pair", "S_mech"):
            c = agg_enc_sch.get(f"{enc}_{sch}")
            cells.append(f"{c['method_a_set_valued']['mean']:.3f} ± {c['method_a_set_valued']['se']:.3f}" if c else "—")
        lines.append(f"| {enc} | " + " | ".join(cells) + " |")

    out_md = OUT_DIR / "phase_a_reanalysis_precompute.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
