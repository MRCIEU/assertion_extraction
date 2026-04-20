"""
MeSH C04 oncology projection (preferred T2 derivation method).

For each PMID in T1_trn packages, queries NCBI PubMed via esearch POST
to determine if the article has a MeSH descriptor under the Neoplasms
(C04) hierarchy.

Uses cache (pubmed_mesh_cache.json) for efficiency.
Rate: 10 req/sec with NCBI_API_KEY; 3 req/sec without.

Outputs:
  - data/processed/t2_biored_mesh.jsonl
  - data/processed/t2_bc5cdr_mesh.jsonl
  - data/processed/t2_drugprot_mesh.jsonl
  - data/processed/t2_mesh_merged.jsonl
  - reports/mesh_projection_summary.json
  - reports/tables/mesh_projection_stats.csv

Paper Methods statement:
  "Documents in T1 with at least one MeSH descriptor under the Neoplasms
   (C04) hierarchy, as indexed by NCBI PubMed, were designated as the T2
   oncology-facing subset."
"""
from __future__ import annotations

import csv, json, os, time, urllib.parse, urllib.request
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROC, REPORTS, TABLES, DATA_OUT, CACHE, ensure_dirs

NCBI_API_KEY  = os.environ.get("NCBI_API_KEY", "")
NCBI_EMAIL    = os.environ.get("KG_AUDIT_EMAIL", "research@example.com")
ESEARCH_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BATCH_SIZE    = 100
REQUEST_DELAY = 0.11 if NCBI_API_KEY else 0.34


def esearch_cancer_batch(pmids: list[str]) -> set[str]:
    """Return subset of pmids indexed with Neoplasms[MeSH Terms]."""
    if not pmids: return set()
    pmid_query = " OR ".join(f"{p}[UID]" for p in pmids)
    post_data = urllib.parse.urlencode({
        "db": "pubmed",
        "term": f"({pmid_query}) AND Neoplasms[MeSH Terms]",
        "retmax": len(pmids) + 10,
        "retmode": "json",
        **({} if not NCBI_API_KEY else {"api_key": NCBI_API_KEY}),
        **({} if not NCBI_EMAIL   else {"email":   NCBI_EMAIL}),
    }).encode("utf-8")
    req = urllib.request.Request(ESEARCH_URL, data=post_data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        return set(result["esearchresult"].get("idlist", []))
    except Exception as exc:
        print(f"    WARNING: esearch failed ({exc})")
        return set()


def get_cancer_pmids(all_pmids: list[str], cache: dict) -> set[str]:
    cancer: set[str] = set()
    to_fetch = [p for p in all_pmids if p not in cache]
    for p in all_pmids:
        if cache.get(p) is True:
            cancer.add(p)

    if to_fetch:
        print(f"  Fetching MeSH for {len(to_fetch)} PMIDs ({len(all_pmids)-len(to_fetch)} cached)...")
        batches = [to_fetch[i:i+BATCH_SIZE] for i in range(0, len(to_fetch), BATCH_SIZE)]
        for idx, batch in enumerate(batches):
            found = esearch_cancer_batch(batch)
            for p in batch:
                is_cancer = p in found
                cache[p] = is_cancer
                if is_cancer: cancer.add(p)
            if idx < len(batches) - 1:
                time.sleep(REQUEST_DELAY)
            if (idx + 1) % 5 == 0 or idx == len(batches) - 1:
                print(f"    {idx+1}/{len(batches)} batches done")
    else:
        print(f"  All {len(all_pmids)} PMIDs served from cache.")
    return cancer


def filter_by_mesh(src: Path, dst: Path, cancer_pmids: set[str]) -> dict:
    kept, removed = 0, 0
    label_dist: Counter = Counter()
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            pid = str(rec.get("doc_id", "")).strip()
            if pid in cancer_pmids:
                pi = rec.get("projection_info", {})
                pi["oncology_projected_slice"] = True
                pi["method"] = "mesh_c04_neoplasms"
                pi["mesh_filter"] = "Neoplasms[MeSH Terms] via NCBI esearch POST"
                rec["projection_info"] = pi
                for rel in rec.get("relations", []):
                    label_dist[rel.get("relation_family", "?")] += 1
                fout.write(json.dumps(rec) + "\n")
                kept += 1
            else:
                removed += 1
    return {
        "kept": kept, "removed": removed,
        "retention_fraction": round(kept / max(kept + removed, 1), 4),
        "cancer_pmids_used": len(cancer_pmids),
        "relation_families": dict(label_dist),
    }


def run() -> None:
    ensure_dirs()
    cache: dict = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    print(f"  Cache: {len(cache)} entries loaded")
    print()
    print("=== MeSH C04 Projection ===\n")

    configs = [
        ("biored",   PROC / "t1_biored_trn.jsonl",  DATA_OUT / "t2_biored_mesh.jsonl"),
        ("bc5cdr",   PROC / "t1_bc5cdr_trn.jsonl",  DATA_OUT / "t2_bc5cdr_mesh.jsonl"),
        ("drugprot", PROC / "t1_drugprot_trn.jsonl", DATA_OUT / "t2_drugprot_mesh.jsonl"),
    ]

    stats = {}
    mesh_parts = []
    for name, src, dst in configs:
        if not src.exists():
            print(f"  ✗ {name}: source not found"); continue
        print(f"  {name}: {src.name}")

        all_pmids = list(dict.fromkeys(
            str(json.loads(l).get("doc_id","")).strip()
            for l in open(src) if l.strip() and json.loads(l).get("doc_id")
        ))
        cancer_pmids = get_cancer_pmids(all_pmids, cache)
        s = filter_by_mesh(src, dst, cancer_pmids)
        stats[name] = s
        mesh_parts.append(dst)
        pct = s["retention_fraction"] * 100
        print(f"  → Kept: {s['kept']}/{len(all_pmids)} docs ({pct:.1f}%)")

    # Save cache
    CACHE.write_text(json.dumps(cache))
    print(f"  Cache saved: {len(cache)} entries → {CACHE.name}")

    # Merge
    merged = DATA_OUT / "t2_mesh_merged.jsonl"
    total = 0
    with open(merged, "w") as fout:
        for p in mesh_parts:
            if p.exists():
                with open(p) as fin:
                    for l in fin: fout.write(l); total += 1
    stats["merged"] = {"total_records": total}
    print(f"\n  Merged MeSH T2: {total} records → {merged.name}")

    # Also copy to training_data_generation/processed so training can use them
    for name, _, src_out in configs:
        dst_proc = PROC / src_out.name
        if src_out.exists() and not dst_proc.exists():
            import shutil; shutil.copy2(src_out, dst_proc)
    merged_proc = PROC / "t2_supervised_oncology_bridge_mesh_merged.jsonl"
    if not merged_proc.exists():
        import shutil; shutil.copy2(merged, merged_proc)

    (REPORTS / "mesh_projection_summary.json").write_text(json.dumps(stats, indent=2))

    rows = [
        {"corpus": k, "kept": v.get("kept",""), "n_pmids": v.get("cancer_pmids_used",""),
         "retention_pct": round(v.get("retention_fraction",0)*100,1),
         "method": "MeSH_C04_Neoplasms"}
        for k, v in stats.items() if k != "merged"
    ]
    with open(TABLES / "mesh_projection_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["corpus","kept","n_pmids","retention_pct","method"])
        w.writeheader(); w.writerows(rows)
    print(f"  Outputs: mesh_projection_summary.json, mesh_projection_stats.csv")
