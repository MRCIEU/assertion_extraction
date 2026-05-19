#!/usr/bin/env python3.11
"""Phase D — supplementary R_B extensions (PB-only schedules vs augmented 10-cell grid).

Loads seed-level aggregates from ``report/data/phase_b_ft_seedlevel.csv``, optionally merges
PB T1F-4096 rows produced by Phase~2C (`PB_PB_FT_T1F4096_s{seed}`), then calls
``analyze_phase_b.bootstrap_RB`` using the canonical BioRED/KB-hit metrics.

When T1F-4096 evaluates are absent, emits **three-schedule interim** diagnostics for PB-only
cells and recomputes the original **nine-cell** encoder×schedule decomposition (no augmentation).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

def _find_project_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(12):
        if (p / "fine_tuning_experiments").is_dir() and (p / "report" / "data").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError("Cannot locate project_1 root from phase_d_rb_extensions.py")


REPO_ROOT = _find_project_root()

CSV_PATH_DEFAULT = REPO_ROOT / "report/data/phase_b_ft_seedlevel.csv"
RUN_ROOT_DEFAULT = Path(
    "/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b"
)

BIORED_COL = "biored_macro_f1_ex_neg"
KB_COL = "kb_hit_A_setvalued"

from fine_tuning_experiments.phase_b.analysis.analyze_phase_b import bootstrap_RB  # noqa: E402


def kb_accuracy_from_kb_surface_targets(path: Path) -> float | None:
    if not path.is_file():
        return None
    hits = nt = 0
    import json as _json

    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = _json.loads(ln)
        if not r.get("evaluable"):
            continue
        hits += int(r["hit_A_sv"])
        nt += 1
    return hits / nt if nt else None


def kb_from_phase_b_eval(path: Path) -> float | None:
    """Aggregate ``kb_hit_A_setvalued`` from ``phase_b_eval.json`` (matches CSV row)."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    kb_surf = data.get("kb_surface")
    if not kb_surf:
        return None
    v = kb_surf.get("kb_hit_A_setvalued")
    return float(v) if v is not None else None


def bio_ex_neg_from_phase_b_eval(path: Path) -> float | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    bt = data.get("biored_test") or {}
    v = bt.get("macro_f1_excluding_negative")
    return float(v) if v is not None else None


def load_t1f4096_seed_rows(run_root: Path) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    ok = False
    for seed in range(1, 21):
        run_dir = run_root / f"PB_PB_FT_T1F4096_s{seed:02d}"
        js_path = run_dir / "eval" / "phase_b_eval.json"
        kb_mean = kb_from_phase_b_eval(js_path)
        if kb_mean is None:
            kb_path = run_dir / "eval" / "kb_surface_targets.jsonl"
            kb_mean = kb_accuracy_from_kb_surface_targets(kb_path)
        biored = bio_ex_neg_from_phase_b_eval(js_path)
        if kb_mean is None or biored is None:
            continue
        rows.append({
            "run_id": run_dir.name,
            "encoder": "PB",
            "update": "FT",
            "seed": seed,
            "schedule_track": "T1F_4096",
            BIORED_COL: biored,
            KB_COL: kb_mean,
        })
        ok = True
    return rows, ok


def normalise_legacy_row(row: dict[str, str]) -> dict[str, Any]:
    sch = row["schedule"]
    if sch == "T1F":
        track = "T1F_2048"
    elif sch == "T1B":
        track = "T1B"
    elif sch == "T2":
        track = "T2"
    else:
        track = sch
    return {
        "run_id": row["run_id"],
        "encoder": row["encoder"],
        "update": row["update"],
        "seed": int(row["seed"]),
        "schedule_track": track,
        BIORED_COL: float(row[BIORED_COL]),
        KB_COL: float(row[KB_COL]),
    }


def bootstrap_block(
    label: str, rows: list[dict[str, Any]], factors: tuple[str, ...],
) -> dict[str, Any]:
    res = bootstrap_RB(
        rows, factors=factors,
        metric_num=BIORED_COL, metric_den=KB_COL,
        seed=20260519,
    )
    uniq_cells = {tuple(r[f] for f in factors) for r in rows}
    out = dict(res)
    out["human_label"] = label
    out["design_factors_requested"] = list(factors)
    out["n_rows_input"] = len(rows)
    out["n_unique_cells"] = len(uniq_cells)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=CSV_PATH_DEFAULT)
    ap.add_argument("--runs-root", type=Path, default=RUN_ROOT_DEFAULT)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT
        / "knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/rb_phase_d_extensions.json",
    )
    args = ap.parse_args()

    with args.csv.open(encoding="utf-8") as fh:
        legacy_rows = list(csv.DictReader(fh))
    base_rows = [
        normalise_legacy_row(r) for r in legacy_rows if r.get("role") == "main"
    ]

    extras, merged_4096 = load_t1f4096_seed_rows(args.runs_root)
    augmented = base_rows[:]
    if merged_4096:
        augmented.extend(extras)

    rows_by_cell_seed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for r in augmented:
        cid = tuple(r[f] for f in ("encoder", "schedule_track", "seed"))
        rows_by_cell_seed[cid] = r
    augmented_unique = sorted(rows_by_cell_seed.values(), key=lambda x: (x["encoder"], x["schedule_track"], x["seed"]))
    pb_unique = sorted(
        (rows_by_cell_seed[k] for k in rows_by_cell_seed if rows_by_cell_seed[k]["encoder"] == "PB"),
        key=lambda x: (str(x["schedule_track"]), int(x["seed"])),
    )

    out: dict[str, Any] = {
        "source_csv": str(args.csv.resolve()),
        "runs_root_checked": str(args.runs_root),
        "t1f4096_rows_merged": merged_4096,
    }

    if merged_4096:
        out["PB_only_four_schedule_rb"] = bootstrap_block(
            "PB_only_four_schedule_design_T1B_T1F2048_T1F4096_T2",
            pb_unique, ("schedule_track",),
        )
        out["augmented_ten_cell_encoder_schedule_rb"] = bootstrap_block(
            "three_encoder_by_schedule_grid_plus_PB_T1F4096_augment_total_10_cells",
            augmented_unique, ("encoder", "schedule_track"),
        )
        out["pre_registered_nine_cell_reference_note"] = (
            "Retain locked nine-cell factorial R_B (=0.21) verbatim in the manuscript; "
            "this augmentation is supplementary only."
        )
    else:
        out["warnings"] = [
            "PB_ONLY R_B emits THREE schedules pending PB_PB_FT_T1F4096 completion.",
            "Augmented 10-cell R_B falls back to the original NINE factorial cells.",
        ]
        pb_three = sorted(
            (r for r in pb_unique if r["schedule_track"] != "T1F_4096"),
            key=lambda x: (x["schedule_track"], x["seed"]),
        )
        out["PB_only_three_schedule_rb_interim_pre_t1f4096"] = bootstrap_block(
            "PB_only_three_schedule_design_interim_missing_T1F4096",
            pb_three, ("schedule_track",),
        )
        nine_only = sorted(
            (r for r in augmented_unique if r["schedule_track"] != "T1F_4096"),
            key=lambda x: (x["encoder"], x["schedule_track"], x["seed"]),
        )
        out["augmented_grid_deferred_matching_pre_registration_nine_cell"] = bootstrap_block(
            "original_nine_cells_encoder_schedule_no_T1F4096_augment_yet",
            nine_only,
            ("encoder", "schedule_track"),
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
