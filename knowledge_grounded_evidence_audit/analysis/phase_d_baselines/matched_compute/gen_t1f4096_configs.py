#!/usr/bin/env python3.11
"""Emit PB_PB_FT_T1F4096_s{01..20}.yaml from the locked T1F template.

Only ``experiment_id`` and ``scientific_trainer.max_updates`` change (2048→4096).
Training shards, seeds, and all other hyper-parameters stay byte-identical to
the pre-registered T1F factorial cell.
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
TEMPLATE = REPO / "fine_tuning_experiments" / "phase_b" / "configs" / "PB_PB_FT_T1F_s01.yaml"
OUT_DIR = Path(__file__).resolve().parent / "configs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    base = args.template.read_text(encoding="utf-8")
    if "max_updates: 2048" not in base:
        raise SystemExit("Template missing expected max_updates: 2048")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ids: list[str] = []
    for i in range(1, 21):
        exp = f"PB_PB_FT_T1F4096_s{i:02d}"
        cfg = base.replace("experiment_id: PB_PB_FT_T1F_s01", f"experiment_id: {exp}")
        cfg = cfg.replace("seed: 1", f"seed: {i}")
        cfg = cfg.replace("max_updates: 2048", "max_updates: 4096", 1)
        out_path = args.out_dir / f"{exp}.yaml"
        out_path.write_text(cfg, encoding="utf-8")
        ids.append(exp)

    ids_path = Path(__file__).resolve().parent / "pb_ft_t1f4096_ids.txt"
    ids_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"Wrote {len(ids)} configs → {args.out_dir}")
    print(f"IDs list: {ids_path}")


if __name__ == "__main__":
    main()
