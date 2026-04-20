"""
Keyword-based oncology projection (original T2 derivation method).

Applies the cancer surface lexicon to BioRED, DrugProt, and BC5CDR
T1_trn packages to produce T2 keyword-filtered slices.

Note: This method is retained for documentation and comparison.
The preferred T2 derivation uses MeSH C04 filtering (mesh_projection.py).

Outputs:
  - reports/tables/keyword_projection_stats.csv
  - reports/keyword_projection_summary.json
"""
from __future__ import annotations

import csv, json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROC, REPORTS, TABLES, DATA_OUT, ensure_dirs
from pipeline.lexicon import is_cancer_like


def project_biored(src: Path, dst: Path) -> dict:
    kept, removed = 0, 0
    label_dist: Counter = Counter()
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            if rec.get("source_split") == "test":
                removed += 1; continue
            # BioRED: include doc if any DISEASE entity mention is cancer-like
            ents = rec.get("entities", [])
            has_cancer = any(
                e.get("mapped_label") == "DISEASE" and is_cancer_like(e.get("text",""))
                for e in ents
            )
            if has_cancer:
                # Update projection_info
                pi = rec.get("projection_info", {})
                pi["oncology_projected_slice"] = True
                pi["method"] = "keyword_cancer_lexicon"
                pi["heuristic"] = True
                rec["projection_info"] = pi
                for rel in rec.get("relations", []):
                    label_dist[rel.get("relation_family", "?")] += 1
                fout.write(json.dumps(rec) + "\n")
                kept += 1
            else:
                removed += 1
    return {"kept": kept, "removed": removed,
            "retention_fraction": round(kept / max(kept + removed, 1), 4),
            "relation_families": dict(label_dist)}


def project_drugprot(src: Path, dst: Path) -> dict:
    kept, removed = 0, 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            if rec.get("source_split") == "test":
                removed += 1; continue
            text = rec.get("text", "")
            if is_cancer_like(text[:800]):
                pi = rec.get("projection_info", {})
                pi["oncology_projected_slice"] = True
                pi["method"] = "keyword_cancer_lexicon_abstract"
                pi["heuristic"] = True
                rec["projection_info"] = pi
                fout.write(json.dumps(rec) + "\n")
                kept += 1
            else:
                removed += 1
    return {"kept": kept, "removed": removed,
            "retention_fraction": round(kept / max(kept + removed, 1), 4)}


def project_bc5cdr(src: Path, dst: Path) -> dict:
    kept, removed = 0, 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            if rec.get("source_split") == "test":
                removed += 1; continue
            ents = rec.get("entities", [])
            cancer_disease_ents = [
                e for e in ents
                if e.get("mapped_label") == "DISEASE" and is_cancer_like(e.get("text",""))
            ]
            if cancer_disease_ents:
                pi = rec.get("projection_info", {})
                pi["oncology_projected_slice"] = True
                pi["method"] = "keyword_cancer_lexicon_disease_arg"
                pi["heuristic"] = True
                rec["projection_info"] = pi
                fout.write(json.dumps(rec) + "\n")
                kept += 1
            else:
                removed += 1
    return {"kept": kept, "removed": removed,
            "retention_fraction": round(kept / max(kept + removed, 1), 4)}


def run() -> None:
    ensure_dirs()
    print("=== Keyword Projection (Lexicon-based T2) ===\n")

    configs = [
        ("biored",   PROC / "t1_biored_trn.jsonl",   DATA_OUT / "t2_biored_keyword.jsonl",   project_biored),
        ("drugprot", PROC / "t1_drugprot_trn.jsonl",  DATA_OUT / "t2_drugprot_keyword.jsonl", project_drugprot),
        ("bc5cdr",   PROC / "t1_bc5cdr_trn.jsonl",   DATA_OUT / "t2_bc5cdr_keyword.jsonl",   project_bc5cdr),
    ]

    stats = {}
    for name, src, dst, fn in configs:
        if not src.exists():
            print(f"  ✗ {name}: source not found")
            continue
        s = fn(src, dst)
        stats[name] = s
        pct = s["retention_fraction"] * 100
        print(f"  {name:<12}: {s['kept']:4d} / {s['kept']+s['removed']:4d} docs ({pct:.1f}% retained)")
        if "relation_families" in s:
            for fam, cnt in sorted(s["relation_families"].items(), key=lambda x: -x[1]):
                print(f"              {fam}: {cnt}")

    # Merge keyword T2
    merged = DATA_OUT / "t2_keyword_merged.jsonl"
    total = 0
    with open(merged, "w") as fout:
        for _, _, src, _ in configs:
            if src.exists():
                with open(src) as fin:
                    for l in fin: fout.write(l); total += 1
    stats["merged"] = {"total_records": total}
    print(f"\n  Merged T2 keyword: {total} records → {merged.name}")

    (REPORTS / "keyword_projection_summary.json").write_text(json.dumps(stats, indent=2))

    rows = [
        {"corpus": k, "kept": v.get("kept",""), "retention_pct": round(v.get("retention_fraction",0)*100,1),
         "method": "keyword_cancer_lexicon"}
        for k, v in stats.items() if k != "merged"
    ]
    with open(TABLES / "keyword_projection_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["corpus","kept","retention_pct","method"])
        w.writeheader(); w.writerows(rows)
    print(f"\nOutputs: keyword_projection_summary.json, keyword_projection_stats.csv")
