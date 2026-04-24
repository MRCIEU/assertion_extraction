#!/usr/bin/env python3.11
"""aggregate_phase_b — collect per-run `phase_b_eval.json` into the flat
results CSV consumed by `analyze_phase_b.py` and `h6_coupling_slopes.py`.

Walks the Phase B run root (default:
  /lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b),
extracts one row per run (parsed from the run directory name
`PB_{ENC}_{UPD}_{SCHED}_s{NN}`), joins in manifest metadata and the three
eval blocks (BioRED, BC5CDR, KB surface), then writes a single CSV with
the schema defined below.

CSV schema (frozen; matches Phase A aggregate.csv plus the Phase B
factorial axes):

    run_id, role, encoder, update, schedule, seed, update_regime,
    schedule_long, schema_id, n_labels,
    biored_macro_f1, biored_macro_f1_ex_neg, biored_n,
    bc5cdr_drug_disease_f1, bc5cdr_macro_f1, bc5cdr_n,
    kb_surface_mean, kb_surface_50, kb_nonneg_rate,
    kb_hit_A_setvalued, kb_hit_A_singlelabel,
    kb_pmass_B_setvalued, kb_pmass_B_singlelabel,
    kb_auc_C_setvalued, kb_auc_C_singlelabel,
    n_targets_evaluable,
    biored_f1__<LABEL>  (one column per label),
    biored_support__<LABEL>  (one column per label)

Where `role` is:
  - "main"        for PB/BL/PL × FT/LR × T1B/T1F/T2
  - "reference"   for RB × FT × T2
  - "smoke"       for any run with seed == 99

Re-created 2026-04-24 after the source-tree deletion incident; the
contents here are a faithful reconstruction from the pre-deletion
CSV header and from analyze_phase_b's required columns.

Usage:
    python3.11 -m fine_tuning_experiments.phase_b.aggregate_phase_b \\
        --out fine_tuning_experiments/phase_b/phase_b_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_RUN_ROOT = Path(
    "/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "phase_b_results.csv"

# Canonical Spair label order (Phase B fixed schema)
SPAIR_LABELS: tuple[str, ...] = (
    "__NEGATIVE__",
    "ASSOCIATION_GENERAL",
    "DRUG_DISEASE",
    "DRUG_GENE_REGULATION",
    "DRUG_VARIANT_ASSOC",
    "GENE_DISEASE",
    "GENE_GENE_ASSOC",
    "VARIANT_DISEASE",
)

RUN_ID_RE = re.compile(r"^PB_([A-Z]+)_([A-Z]+)_([A-Za-z0-9]+)_s(\d+)$")

# Fixed header order for downstream stability (analyze_phase_b + h6_coupling_slopes
# both read by column name but tools like pandas benefit from a stable order).
BASE_COLUMNS: tuple[str, ...] = (
    "run_id", "role", "encoder", "update", "schedule", "seed",
    "update_regime", "schedule_long", "schema_id", "n_labels",
    "biored_macro_f1", "biored_macro_f1_ex_neg", "biored_n",
    "bc5cdr_drug_disease_f1", "bc5cdr_macro_f1", "bc5cdr_n",
    "kb_surface_mean", "kb_surface_50", "kb_nonneg_rate",
    "kb_hit_A_setvalued", "kb_hit_A_singlelabel",
    "kb_pmass_B_setvalued", "kb_pmass_B_singlelabel",
    "kb_auc_C_setvalued", "kb_auc_C_singlelabel",
    "n_targets_evaluable",
)


def _row_role(encoder: str, seed: int) -> str:
    if seed == 99:
        return "smoke"
    if encoder == "RB":
        return "reference"
    return "main"


def _extract_per_label(biored: dict[str, Any]) -> dict[str, Any]:
    """Flatten per-label F1/support into biored_f1__<LABEL> columns."""
    out: dict[str, Any] = {}
    per_label = biored.get("per_label", {}) or {}
    for label in SPAIR_LABELS:
        entry = per_label.get(label, {}) or {}
        out[f"biored_f1__{label}"] = float(entry.get("f1", 0.0))
        out[f"biored_support__{label}"] = int(entry.get("support", 0))
    return out


def _row_from_run(run_dir: Path) -> dict[str, Any] | None:
    name = run_dir.name
    m = RUN_ID_RE.match(name)
    if not m:
        return None
    enc, upd, sched, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))

    eval_json = run_dir / "eval" / "phase_b_eval.json"
    manifest_json = run_dir / "run_manifest.json"
    if not eval_json.exists():
        return None
    if not manifest_json.exists():
        return None
    e = json.loads(eval_json.read_text())
    m_ = json.loads(manifest_json.read_text())

    biored = e.get("biored_test", {}) or {}
    bc5cdr = e.get("bc5cdr_test", {}) or {}
    kb = e.get("kb_surface", {}) or {}

    row: dict[str, Any] = {
        "run_id": name,
        "role": _row_role(enc, seed),
        "encoder": enc,
        "update": upd,
        "schedule": sched,
        "seed": seed,
        "update_regime": (
            m_.get("update_regime")
            or e.get("update_regime")
            or ("lora" if upd == "LR" else "full_finetune")
        ),
        "schedule_long": e.get("schedule") or m_.get("schedule") or "",
        "schema_id": m_.get("schema_id", "S_pair"),
        "n_labels": len(e.get("labels_ordered", SPAIR_LABELS)),
        # BioRED
        "biored_macro_f1": float(biored.get("macro_f1", 0.0)),
        "biored_macro_f1_ex_neg": float(biored.get("macro_f1_excluding_negative", 0.0)),
        "biored_n": int(biored.get("n", 0)),
        # BC5CDR
        "bc5cdr_drug_disease_f1": float(bc5cdr.get("drug_disease_f1", 0.0)),
        "bc5cdr_macro_f1": float(bc5cdr.get("macro_f1", 0.0)),
        "bc5cdr_n": int(bc5cdr.get("n", 0)),
        # KB surface
        "kb_surface_mean": float(kb.get("kb_surface_mean", 0.0)),
        "kb_surface_50": float(kb.get("kb_surface_50", 0.0)),
        "kb_nonneg_rate": float(kb.get("kb_nonneg_rate", 0.0)),
        "kb_hit_A_setvalued": float(kb.get("kb_hit_A_setvalued", 0.0)),
        "kb_hit_A_singlelabel": float(kb.get("kb_hit_A_singlelabel", 0.0)),
        "kb_pmass_B_setvalued": float(kb.get("kb_pmass_B_setvalued", 0.0)),
        "kb_pmass_B_singlelabel": float(kb.get("kb_pmass_B_singlelabel", 0.0)),
        "kb_auc_C_setvalued": float(kb.get("kb_auc_C_setvalued", 0.0)),
        "kb_auc_C_singlelabel": float(kb.get("kb_auc_C_singlelabel", 0.0)),
        "n_targets_evaluable": int(kb.get("n_targets_evaluable", 0)),
    }
    row.update(_extract_per_label(biored))
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--include-smoke", action="store_true",
        help="Include seed=99 smoke runs (excluded by default).",
    )
    args = ap.parse_args(argv)

    run_dirs = sorted(d for d in args.run_root.iterdir() if d.is_dir())
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for d in run_dirs:
        r = _row_from_run(d)
        if r is None:
            if RUN_ID_RE.match(d.name):
                missing.append(d.name)
            continue
        if r["role"] == "smoke" and not args.include_smoke:
            continue
        rows.append(r)

    # Stable output order: encoder, update, schedule, seed.
    rows.sort(key=lambda r: (r["encoder"], r["update"], r["schedule"], int(r["seed"])))

    columns = list(BASE_COLUMNS)
    for label in SPAIR_LABELS:
        columns.append(f"biored_f1__{label}")
    for label in SPAIR_LABELS:
        columns.append(f"biored_support__{label}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Cell-count sanity: group by (encoder, update, schedule).
    cells: dict[tuple[str, str, str], int] = {}
    for r in rows:
        key = (r["encoder"], r["update"], r["schedule"])
        cells[key] = cells.get(key, 0) + 1

    print(f"Wrote {len(rows)} rows to {args.out}")
    print("Cell fill (encoder/update/schedule → seeds present):")
    for key in sorted(cells):
        print(f"  {key[0]}_{key[1]}_{key[2]}: {cells[key]}")
    if missing:
        print(f"\n{len(missing)} run dirs matched PB_* pattern but lacked eval or manifest:")
        for m_ in missing[:20]:
            print(f"  missing: {m_}")
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
