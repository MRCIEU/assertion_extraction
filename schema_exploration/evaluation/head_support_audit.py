"""
Head support audit for SC0, SC1, SC3.

Computes per-head training instance counts across all T1+T2 packages
for each schema candidate. Identifies trainable vs dead heads.

Outputs:
  - reports/tables/head_support_audit.csv
  - reports/head_support_summary.json
"""
from __future__ import annotations

import csv, json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROC, REPORTS, TABLES, ensure_dirs

N_MIN = 50  # minimum training instances per head

SCHEMA_PACKAGES = {
    "S_flat": [   # Uses remapped _Sflat packages to collapse VARIANT_GENE (4 instances) → ASSOC_GENERAL
        PROC / "t1_biored_trn_Sflat.jsonl",
        PROC / "t1_bc5cdr_trn_Sflat.jsonl",
        PROC / "t1_drugprot_trn_Sflat.jsonl",
        PROC / "t2_biored_mesh_Sflat.jsonl",
        PROC / "t2_bc5cdr_mesh_Sflat.jsonl",
        PROC / "t2_drugprot_mesh_Sflat.jsonl",
    ],
    "S_pair": [
        PROC / "t1_biored_trn_Spair.jsonl",
        PROC / "t1_bc5cdr_trn_Spair.jsonl",
        PROC / "t1_drugprot_trn_Spair.jsonl",
        PROC / "t2_biored_mesh_Spair.jsonl",
        PROC / "t2_bc5cdr_mesh_Spair.jsonl",
        PROC / "t2_drugprot_mesh_Spair.jsonl",
    ],
    "S_mech": [
        PROC / "t1_biored_trn_Smech.jsonl",
        PROC / "t1_bc5cdr_trn_Smech.jsonl",
        PROC / "t1_drugprot_trn_Smech.jsonl",
        PROC / "t2_biored_mesh_Smech.jsonl",
        PROC / "t2_bc5cdr_mesh_Smech.jsonl",
        PROC / "t2_drugprot_mesh_Smech.jsonl",
    ],
}


def count_heads(paths: list[Path]) -> Counter:
    c: Counter = Counter()
    for p in paths:
        if not p.exists(): continue
        with open(p) as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                for rel in rec.get("relations", []):
                    c[rel.get("relation_family", "?")] += 1
    return c


def run() -> None:
    ensure_dirs()
    print("=== Per-Head Support Audit ===\n")
    print(f"Trainability threshold: N_min = {N_MIN}\n")

    all_schema_counts = {}
    rows = []

    for schema, paths in SCHEMA_PACKAGES.items():
        c = count_heads(paths)
        all_schema_counts[schema] = dict(c)
        total = sum(c.values())

        print(f"  {schema} (total={total} relations):")
        print(f"  {'Head':<30} {'Count':>7}  {'%':>6}  {'Status'}")
        print(f"  {'-'*55}")
        for head, cnt in sorted(c.items(), key=lambda x: -x[1]):
            pct = 100 * cnt / max(total, 1)
            status = "✓ TRAINABLE" if cnt >= N_MIN else f"✗ BELOW {N_MIN}"
            print(f"  {head:<30} {cnt:>7}  {pct:>5.1f}%  {status}")
            rows.append({
                "schema": schema, "head": head, "count": cnt,
                "fraction": round(cnt/max(total,1), 4),
                "trainable": cnt >= N_MIN,
                "status": "trainable" if cnt >= N_MIN else "dead",
            })
        dead = [h for h, c_ in c.items() if c_ < N_MIN]
        print(f"  Dead heads: {dead if dead else 'NONE'}")
        print(f"  TRAINABLE: {'YES' if not dead else 'NO'}\n")

    (REPORTS / "head_support_summary.json").write_text(json.dumps(all_schema_counts, indent=2))

    with open(TABLES / "head_support_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["schema","head","count","fraction","trainable","status"])
        w.writeheader(); w.writerows(rows)

    print(f"  Outputs: head_support_summary.json, head_support_audit.csv")
