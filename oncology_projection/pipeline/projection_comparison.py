"""
Projection comparison: keyword lexicon vs MeSH C04.

Computes and reports differences between the two T2 derivation methods:
  - Document overlap (which docs appear in both)
  - Retention rates
  - Relation family distributions
  - Entity pair type distributions

Outputs:
  - reports/projection_comparison.json
  - reports/tables/projection_comparison_stats.csv
"""
from __future__ import annotations

import csv, json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROC, DATA_OUT, REPORTS, TABLES, ensure_dirs


def load_doc_ids(path: Path) -> set[str]:
    if not path.exists(): return set()
    ids = set()
    with open(path) as f:
        for l in f:
            if l.strip():
                ids.add(str(json.loads(l).get("doc_id","")))
    return ids


def load_relation_dist(path: Path) -> Counter:
    c: Counter = Counter()
    if not path.exists(): return c
    with open(path) as f:
        for l in f:
            if l.strip():
                for rel in json.loads(l).get("relations", []):
                    c[rel.get("relation_family","?")] += 1
    return c


def run() -> None:
    ensure_dirs()
    print("=== Projection Method Comparison ===\n")

    corpora = ["biored", "bc5cdr", "drugprot"]
    results = {}

    for name in corpora:
        kw_path   = DATA_OUT / f"t2_{name}_keyword.jsonl"
        mesh_path = DATA_OUT / f"t2_{name}_mesh.jsonl"

        kw_ids   = load_doc_ids(kw_path)
        mesh_ids = load_doc_ids(mesh_path)

        overlap      = kw_ids & mesh_ids
        kw_only      = kw_ids - mesh_ids
        mesh_only    = mesh_ids - kw_ids

        kw_rels   = load_relation_dist(kw_path)
        mesh_rels = load_relation_dist(mesh_path)

        results[name] = {
            "keyword_n_docs": len(kw_ids),
            "mesh_n_docs":    len(mesh_ids),
            "overlap_n_docs": len(overlap),
            "kw_only_n":      len(kw_only),
            "mesh_only_n":    len(mesh_only),
            "jaccard": round(len(overlap) / max(len(kw_ids | mesh_ids), 1), 4),
            "keyword_relations": dict(kw_rels),
            "mesh_relations":    dict(mesh_rels),
        }

        print(f"  {name}:")
        print(f"    keyword: {len(kw_ids)} docs | mesh: {len(mesh_ids)} docs")
        print(f"    overlap: {len(overlap)} | kw-only: {len(kw_only)} | mesh-only: {len(mesh_only)}")
        print(f"    Jaccard: {results[name]['jaccard']:.3f}")

    (REPORTS / "projection_comparison.json").write_text(json.dumps(results, indent=2))

    rows = [
        {"corpus": k,
         "keyword_docs": v["keyword_n_docs"],
         "mesh_docs":    v["mesh_n_docs"],
         "overlap":      v["overlap_n_docs"],
         "kw_only":      v["kw_only_n"],
         "mesh_only":    v["mesh_only_n"],
         "jaccard":      v["jaccard"],
         "preferred":    "MeSH (reproducible, standard)"}
        for k, v in results.items()
    ]
    with open(TABLES / "projection_comparison_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n  Outputs: projection_comparison.json, projection_comparison_stats.csv")
