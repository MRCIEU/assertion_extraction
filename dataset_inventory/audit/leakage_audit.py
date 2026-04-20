"""
Audit 03: Train-test leakage audit and fix.

Identifies test-split documents in T1/T2 JSONL packages and
creates clean *_trn.jsonl versions without test contamination.

Outputs:
  - reports/leakage_audit.json
  - reports/tables/leakage_summary.csv
  - Cleaned packages in training_data_generation/data/processed/*_trn.jsonl
"""
from __future__ import annotations

import csv, json
from collections import Counter
from pathlib import Path
import sys, shutil
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROC, REPORTS, TABLES, ensure_dirs

PACKAGES_TO_CHECK = {
    "t1_biored":    PROC / "t1_biored.jsonl",
    "t1_bc5cdr":    PROC / "t1_bc5cdr.jsonl",
    "t1_drugprot":  PROC / "t1_drugprot.jsonl",
    "t2_biored_proj": PROC / "t2_biored_projected.jsonl",
    "t2_bc5cdr_cancer": PROC / "t2_bc5cdr_cancer_slice.jsonl",
    "t2_drugprot_proj": PROC / "t2_drugprot_projected.jsonl",
}

EXCLUDED_SPLITS = {"test"}


def check_leakage(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing"}
    splits: Counter = Counter()
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            splits[json.loads(line).get("source_split", "unknown")] += 1
    test_count = splits.get("test", 0)
    total = sum(splits.values())
    return {
        "status": "ok",
        "total": total,
        "splits": dict(splits),
        "test_count": test_count,
        "test_fraction": round(test_count / max(total, 1), 4),
        "contaminated": test_count > 0,
    }


def filter_package(src: Path, dst: Path) -> dict:
    kept, removed = Counter(), Counter()
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            split = rec.get("source_split", "unknown")
            if split in EXCLUDED_SPLITS:
                removed[split] += 1
            else:
                kept[split] += 1
                fout.write(line + "\n")
    return {"kept": dict(kept), "removed": dict(removed),
            "kept_total": sum(kept.values()), "removed_total": sum(removed.values())}


def merge_packages(sources: list[Path], dst: Path) -> int:
    total = 0
    with open(dst, "w") as fout:
        for src in sources:
            if src.exists():
                with open(src) as fin:
                    for line in fin:
                        fout.write(line); total += 1
    return total


def run() -> None:
    ensure_dirs()
    print("=== Train-Test Leakage Audit ===\n")
    audit = {}
    for name, path in PACKAGES_TO_CHECK.items():
        audit[name] = check_leakage(path)
        s = audit[name]
        if s["status"] == "missing":
            print(f"  ✗ {name}: MISSING")
        elif s["contaminated"]:
            print(f"  ✗ {name}: {s['test_count']} test docs ({s['test_fraction']*100:.1f}%) — CONTAMINATED")
        else:
            print(f"  ✓ {name}: clean (splits: {s['splits']})")

    # Create clean *_trn.jsonl packages
    print("\n=== Creating Clean *_trn.jsonl Packages ===\n")
    filter_results = {}
    t1_trn_parts, t2_trn_parts = [], []

    for src_name, dst_suffix, trn_list in [
        ("t1_biored",       "_trn",  t1_trn_parts),
        ("t1_drugprot",     "_trn",  t1_trn_parts),
        ("t1_bc5cdr",       "_trn",  t1_trn_parts),
        ("t2_biored_proj",  "_trn",  t2_trn_parts),
        ("t2_drugprot_proj","_trn",  t2_trn_parts),
        ("t2_bc5cdr_cancer","_trn",  t2_trn_parts),
    ]:
        src = PACKAGES_TO_CHECK[src_name]
        stem = src.stem
        dst = PROC / f"{stem}{dst_suffix}.jsonl"
        if not src.exists():
            print(f"  ✗ {src.name}: MISSING — skip")
            continue
        r = filter_package(src, dst)
        filter_results[dst.name] = r
        trn_list.append(dst)
        removed = r["removed_total"]
        icon = "✓" if removed > 0 else "○"
        print(f"  {icon} {src.name} → {dst.name}: kept={r['kept_total']}, removed={removed}")

    # Merged T1
    merged_t1 = PROC / "t1_supervised_backbone_merged_trn.jsonl"
    n1 = merge_packages(t1_trn_parts, merged_t1)
    filter_results[merged_t1.name] = {"merged": True, "total_records": n1}
    print(f"  ✓ Merged T1_trn: {n1} records → {merged_t1.name}")

    # Merged T2
    merged_t2 = PROC / "t2_supervised_oncology_bridge_merged_trn.jsonl"
    n2 = merge_packages(t2_trn_parts, merged_t2)
    filter_results[merged_t2.name] = {"merged": True, "total_records": n2}
    print(f"  ✓ Merged T2_trn: {n2} records → {merged_t2.name}")

    # Save outputs
    report = {"leakage_audit": audit, "filter_results": filter_results,
              "total_test_docs_removed": sum(
                  r.get("removed_total", 0) for r in filter_results.values()
                  if not r.get("merged")),
              "status": "PASSED — all *_trn.jsonl packages are test-split-free"}
    (REPORTS / "leakage_audit.json").write_text(json.dumps(report, indent=2))

    # CSV summary
    rows = [
        {"package": k, "total": v.get("total",""), "test_count": v.get("test_count",""),
         "test_fraction": v.get("test_fraction",""), "contaminated": v.get("contaminated",""),
         "status": v.get("status","")}
        for k, v in audit.items()
    ]
    with open(TABLES / "leakage_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["package","total","test_count","test_fraction","contaminated","status"])
        w.writeheader(); w.writerows(rows)

    total_removed = sum(r.get("removed_total", 0) for r in filter_results.values() if not r.get("merged"))
    print(f"\nSummary: {total_removed} test-split docs removed across all packages")
    print(f"Outputs: leakage_audit.json, leakage_summary.csv")
