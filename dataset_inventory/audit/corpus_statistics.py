"""
Audit 02: Detailed corpus statistics from packaged JSONL files.

Computes entity-type and relation-type distributions, split counts,
and per-corpus relation pair-type breakdowns.

Outputs:
  - reports/tables/corpus_relation_stats.csv
  - reports/tables/corpus_entity_stats.csv
  - reports/tables/corpus_split_counts.csv
  - reports/tables/entity_pair_distribution.csv
  - reports/corpus_statistics_summary.json
"""
from __future__ import annotations

import csv, json
from collections import Counter
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROC, REPORTS, TABLES, ensure_dirs


CLEAN_PACKAGES = {
    "biored_T1":   PROC / "t1_biored_trn.jsonl",
    "bc5cdr_T1":   PROC / "t1_bc5cdr_trn.jsonl",
    "drugprot_T1": PROC / "t1_drugprot_trn.jsonl",
    "biored_T2_mesh":   PROC / "t2_biored_mesh.jsonl",
    "bc5cdr_T2_mesh":   PROC / "t2_bc5cdr_mesh.jsonl",
    "drugprot_T2_mesh": PROC / "t2_drugprot_mesh.jsonl",
    "civic_T3a":        PROC / "t3_civic_semantic_priors.jsonl",
    "civicmine_T3b":    PROC / "t3_civicmine_weak_sentences.jsonl",
    "cancermine_T3c":   PROC / "t3_cancermine_priors.jsonl",
    "lungpubmed_T4":    PROC / "t4_unlabeled_domain_adaptation.jsonl",
}


def analyse_package(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing"}
    splits: Counter = Counter()
    relation_families: Counter = Counter()
    source_labels: Counter = Counter()
    entity_families: Counter = Counter()
    pair_types: Counter = Counter()
    n_docs = 0
    n_relations = 0
    n_entities = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_docs += 1
            splits[rec.get("source_split", "unknown")] += 1

            ent_map = {e["entity_id"]: e.get("mapped_label", "?")
                       for e in rec.get("entities", [])}
            for e in rec.get("entities", []):
                entity_families[e.get("mapped_label", "?")] += 1
                n_entities += 1

            for rel in rec.get("relations", []):
                relation_families[rel.get("relation_family", "?")] += 1
                source_labels[rel.get("source_label", "?")] += 1
                h = ent_map.get(rel.get("head_entity_id", ""), "?")
                t = ent_map.get(rel.get("tail_entity_id", ""), "?")
                pair = "-".join(sorted([h, t]))
                pair_types[pair] += 1
                n_relations += 1

    size_mb = round(path.stat().st_size / 1e6, 1)
    test_docs = splits.get("test", 0)
    return {
        "status": "ok",
        "n_docs": n_docs,
        "n_relations": n_relations,
        "n_entities": n_entities,
        "size_mb": size_mb,
        "test_docs": test_docs,
        "test_contamination": test_docs > 0,
        "splits": dict(splits),
        "relation_families": dict(relation_families.most_common()),
        "source_labels": dict(source_labels.most_common(10)),
        "entity_families": dict(entity_families.most_common()),
        "entity_pair_types": dict(pair_types.most_common(10)),
    }


def run() -> None:
    ensure_dirs()
    stats: dict = {}

    print("=== Corpus Statistics from Packaged JSONL ===\n")
    for pkg_name, pkg_path in CLEAN_PACKAGES.items():
        print(f"  Analysing {pkg_name}...")
        stats[pkg_name] = analyse_package(pkg_path)

    # Summary JSON
    summary = REPORTS / "corpus_statistics_summary.json"
    summary.write_text(json.dumps(stats, indent=2))

    # CSV 1: relation stats per package
    rel_rows = []
    for pkg, s in stats.items():
        if s["status"] == "missing":
            continue
        for fam, cnt in s.get("relation_families", {}).items():
            rel_rows.append({
                "package": pkg,
                "relation_family": fam,
                "count": cnt,
                "fraction": round(cnt / max(s["n_relations"], 1), 4),
            })
    csv_out = TABLES / "corpus_relation_stats.csv"
    if rel_rows:
        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["package", "relation_family", "count", "fraction"])
            w.writeheader(); w.writerows(rel_rows)

    # CSV 2: entity stats
    ent_rows = []
    for pkg, s in stats.items():
        if s["status"] == "missing": continue
        for fam, cnt in s.get("entity_families", {}).items():
            ent_rows.append({"package": pkg, "entity_family": fam, "count": cnt})
    csv_out2 = TABLES / "corpus_entity_stats.csv"
    if ent_rows:
        with open(csv_out2, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["package", "entity_family", "count"])
            w.writeheader(); w.writerows(ent_rows)

    # CSV 3: split counts
    split_rows = []
    for pkg, s in stats.items():
        if s["status"] == "missing": continue
        for split, cnt in s.get("splits", {}).items():
            split_rows.append({
                "package": pkg, "source_split": split, "count": cnt,
                "is_test": split == "test",
            })
    csv_out3 = TABLES / "corpus_split_counts.csv"
    if split_rows:
        with open(csv_out3, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["package", "source_split", "count", "is_test"])
            w.writeheader(); w.writerows(split_rows)

    # CSV 4: entity pair type distribution (BioRED only — most informative)
    biored_pairs = stats.get("biored_T1", {}).get("entity_pair_types", {})
    pair_rows = [{"entity_pair": k, "count": v} for k, v in sorted(biored_pairs.items(), key=lambda x: -x[1])]
    csv_out4 = TABLES / "entity_pair_distribution.csv"
    if pair_rows:
        with open(csv_out4, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["entity_pair", "count"])
            w.writeheader(); w.writerows(pair_rows)

    # Print overview
    print()
    print(f"{'Package':<25} {'Docs':>6} {'Relations':>10} {'Entities':>10} {'Test?':>6} {'MB':>6}")
    print("-" * 70)
    for pkg, s in stats.items():
        if s["status"] == "missing":
            print(f"  {pkg:<23} MISSING")
            continue
        tc = "YES" if s["test_contamination"] else "no"
        print(f"  {pkg:<23} {s['n_docs']:>6} {s['n_relations']:>10} {s['n_entities']:>10} {tc:>6} {s['size_mb']:>6}")

    print(f"\nOutputs: corpus_statistics_summary.json + 4 CSV tables")
