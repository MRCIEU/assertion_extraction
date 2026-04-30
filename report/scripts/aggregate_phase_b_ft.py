"""Aggregate Phase B FT eval JSONs into a tidy CSV for the report.

Walks runs/phase_b/PB_*_FT_*_s*/eval/phase_b_eval.json and emits one row per run
with the metrics most useful for reporting (BioRED ex-NEG, BioRED active-head
macro, BC5CDR DD, KB_hit_A_sv, KB_pmass_B_sv, KB_auc_C_sv, plus per-head F1).

Output: report/data/phase_b_ft_seedlevel.csv  (+ phase_b_ft_cells.csv).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev

RUNS_ROOT = Path("/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b")
OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIVE_HEADS = (
    "GENE_DISEASE",
    "DRUG_DISEASE",
    "VARIANT_DISEASE",
    "GENE_GENE_ASSOC",
    "DRUG_GENE_REGULATION",
)


def macro_active(per_label: dict) -> float:
    vals = [per_label[h]["f1"] for h in ACTIVE_HEADS if h in per_label]
    return mean(vals) if vals else float("nan")


def main() -> None:
    rows = []
    for d in sorted(RUNS_ROOT.iterdir()):
        ev = d / "eval" / "phase_b_eval.json"
        if not ev.exists():
            continue
        try:
            data = json.loads(ev.read_text())
        except Exception:
            continue
        biored = data.get("biored_test", {})
        bc = data.get("bc5cdr_test", {})
        kb = data.get("kb_surface", {})
        rows.append({
            "run_id": data.get("run_id", d.name),
            "encoder": data.get("encoder_key", ""),
            "update": data.get("update_key", ""),
            "schedule": data.get("schedule_key", ""),
            "seed": data.get("seed", ""),
            "biored_macro_f1": biored.get("macro_f1"),
            "biored_macro_f1_ex_neg": biored.get("macro_f1_excluding_negative"),
            "biored_active_head_macro": macro_active(biored.get("per_label", {})),
            "bc5cdr_drug_disease_f1": bc.get("drug_disease_f1"),
            "kb_hit_A_setvalued": kb.get("kb_hit_A_setvalued"),
            "kb_pmass_B_setvalued": kb.get("kb_pmass_B_setvalued"),
            "kb_auc_C_setvalued": kb.get("kb_auc_C_setvalued"),
            "kb_n_evaluable": kb.get("n_targets_evaluable"),
        })

    seed_csv = OUT_DIR / "phase_b_ft_seedlevel.csv"
    if rows:
        with seed_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {seed_csv}")
    else:
        print("no eval files found")
        return

    # Cell-level aggregate (encoder × update × schedule).
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        cells.setdefault((r["encoder"], r["update"], r["schedule"]), []).append(r)

    metrics = [
        "biored_macro_f1_ex_neg",
        "biored_active_head_macro",
        "bc5cdr_drug_disease_f1",
        "kb_hit_A_setvalued",
        "kb_pmass_B_setvalued",
        "kb_auc_C_setvalued",
    ]
    fields = ["encoder", "update", "schedule", "n_seeds"]
    for m in metrics:
        fields += [f"{m}_mean", f"{m}_median", f"{m}_sd",
                   f"{m}_min", f"{m}_max"]
    cell_csv = OUT_DIR / "phase_b_ft_cells.csv"
    with cell_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for (enc, upd, sch), rs in sorted(cells.items()):
            row = {"encoder": enc, "update": upd, "schedule": sch, "n_seeds": len(rs)}
            for m in metrics:
                vals = [r[m] for r in rs if r[m] is not None]
                if vals:
                    row[f"{m}_mean"] = mean(vals)
                    row[f"{m}_median"] = median(vals)
                    row[f"{m}_sd"] = pstdev(vals) if len(vals) > 1 else 0.0
                    row[f"{m}_min"] = min(vals)
                    row[f"{m}_max"] = max(vals)
                else:
                    for sfx in ("mean", "median", "sd", "min", "max"):
                        row[f"{m}_{sfx}"] = ""
            w.writerow(row)
    print(f"wrote cell aggregate -> {cell_csv}")


if __name__ == "__main__":
    main()
