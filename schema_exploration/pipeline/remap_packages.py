"""
Schema package remapping pipeline.

Generates training JSONL packages for SC1 and SC3 by remapping BioRED
relation labels using entity-pair-type and mechanism information.
SC0 uses the existing *_trn.jsonl packages without remapping.

Outputs (all in ~/projects/project_1/training_data_generation/data/processed/):
  SC1: t1_biored_trn_Spair.jsonl, t1_drugprot_trn_Spair.jsonl, t1_bc5cdr_trn_Spair.jsonl
       t1_supervised_backbone_merged_Spair.jsonl
       t2_biored_mesh_Spair.jsonl, t2_bc5cdr_mesh_Spair.jsonl, t2_drugprot_mesh_Spair.jsonl
       t2_supervised_oncology_bridge_mesh_merged_Spair.jsonl
       t1_biored_test_Spair.jsonl  (BioRED test with SC1 gold labels)

  SC3: same structure with _Smech suffix; PART-OF → DGR_STRUCTURAL
"""
from __future__ import annotations

import json, shutil
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROC, REPORTS, TABLES, DATA_OUT, ensure_dirs
from definitions.schema_definitions import sc1_label, sc3_label


def remap_record(rec: dict, label_fn) -> dict:
    ent_map = {e["entity_id"]: e.get("mapped_label", "?")
               for e in rec.get("entities", [])}
    src_ds = rec.get("source_dataset", "")
    new_rels = []
    for rel in rec.get("relations", []):
        h = ent_map.get(rel.get("head_entity_id", ""), "?")
        t = ent_map.get(rel.get("tail_entity_id", ""), "?")
        new_lbl = label_fn(h, t, rel.get("source_label", ""), src_ds)
        new_rel = dict(rel)
        new_rel["mapped_label_sc0"] = rel.get("mapped_label", "")
        new_rel["mapped_label"]    = new_lbl
        new_rel["relation_family"] = new_lbl
        new_rels.append(new_rel)
    out = dict(rec)
    out["relations"] = new_rels
    return out


def process_jsonl(src: Path, dst: Path, label_fn, remap: bool = True) -> dict:
    if not remap:
        shutil.copy2(src, dst)
        with open(dst) as f:
            n = sum(1 for _ in f)
        return {"copied": True, "records": n}
    before: Counter = Counter()
    after: Counter  = Counter()
    n = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            for rel in rec.get("relations", []):
                before[rel.get("relation_family", "?")] += 1
            rec = remap_record(rec, label_fn)
            for rel in rec.get("relations", []):
                after[rel.get("relation_family", "?")] += 1
            fout.write(json.dumps(rec) + "\n")
            n += 1
    return {"records": n, "before": dict(before), "after": dict(after)}


def merge(parts: list[Path], dst: Path) -> int:
    total = 0
    with open(dst, "w") as fout:
        for p in parts:
            if p.exists():
                with open(p) as fin:
                    for l in fin: fout.write(l); total += 1
    return total


def build_schema_packages(schema_name: str, label_fn, suffix: str,
                          reuse_biored_from: str | None = None) -> dict:
    """
    Generate all T1 and T2 packages for a given schema.

    reuse_biored_from: if set (e.g. "_Spair"), reuse existing BioRED packages
      from that suffix instead of remapping (for SC3 which has identical BioRED
      labels to SC1 — avoids disk duplication).
    """
    report = {}

    # ---- T1 packages ----
    if reuse_biored_from is not None:
        # BioRED identical to another schema — create symlink-style copy
        for name in ["biored"]:
            src_existing = PROC / f"t1_{name}_trn{reuse_biored_from}.jsonl"
            dst = PROC / f"t1_{name}_trn{suffix}.jsonl"
            if src_existing.exists() and not dst.exists():
                shutil.copy2(src_existing, dst)
            report[f"t1_{name}"] = {"reused_from": str(reuse_biored_from), "records": sum(1 for _ in open(dst))}

        t1_biored_part = PROC / f"t1_biored_trn{suffix}.jsonl"
    else:
        t1_biored_part = PROC / f"t1_biored_trn{suffix}.jsonl"
        r = process_jsonl(PROC / "t1_biored_trn.jsonl", t1_biored_part, label_fn, True)
        report["t1_biored"] = r

    t1_drugprot_part = PROC / f"t1_drugprot_trn{suffix}.jsonl"
    r = process_jsonl(PROC / "t1_drugprot_trn.jsonl", t1_drugprot_part, label_fn, True)
    report["t1_drugprot"] = r

    t1_bc5cdr_part = PROC / f"t1_bc5cdr_trn{suffix}.jsonl"
    r = process_jsonl(PROC / "t1_bc5cdr_trn.jsonl", t1_bc5cdr_part, label_fn, False)
    report["t1_bc5cdr"] = r

    t1_parts = [t1_biored_part, t1_drugprot_part, t1_bc5cdr_part]
    merged_t1 = PROC / f"t1_supervised_backbone_merged{suffix}.jsonl"
    n1 = merge(t1_parts, merged_t1)
    report["t1_merged"] = {"records": n1}

    # ---- T2 MeSH packages ----
    for name in ["biored", "bc5cdr"]:
        src = PROC / f"t2_{name}_mesh.jsonl"
        if not src.exists(): continue
        if reuse_biored_from is not None and name == "biored":
            dst = PROC / f"t2_{name}_mesh{suffix}.jsonl"
            src_existing = PROC / f"t2_{name}_mesh{reuse_biored_from}.jsonl"
            if src_existing.exists() and not dst.exists():
                shutil.copy2(src_existing, dst)
            report[f"t2_{name}"] = {"reused_from": str(reuse_biored_from), "records": sum(1 for _ in open(dst))}
        else:
            dst = PROC / f"t2_{name}_mesh{suffix}.jsonl"
            r = process_jsonl(src, dst, label_fn, name == "biored")
            report[f"t2_{name}"] = r

    t2_drugprot_src = PROC / "t2_drugprot_mesh.jsonl"
    if t2_drugprot_src.exists():
        t2_dp_dst = PROC / f"t2_drugprot_mesh{suffix}.jsonl"
        r = process_jsonl(t2_drugprot_src, t2_dp_dst, label_fn, True)
        report["t2_drugprot"] = r

    t2_parts = [PROC / f"t2_{n}_mesh{suffix}.jsonl" for n in ["biored", "drugprot", "bc5cdr"]]
    merged_t2 = PROC / f"t2_supervised_oncology_bridge_mesh_merged{suffix}.jsonl"
    n2 = merge(t2_parts, merged_t2)
    report["t2_merged"] = {"records": n2}

    # ---- BioRED test with schema gold labels ----
    if reuse_biored_from is not None:
        # reuse existing test gold (BioRED labels identical)
        src_test = PROC / f"t1_biored_test{reuse_biored_from}.jsonl"
        dst_test = PROC / f"t1_biored_test{suffix}.jsonl"
        if src_test.exists() and not dst_test.exists():
            shutil.copy2(src_test, dst_test)
        test_labels: Counter = Counter()
        with open(dst_test) as f:
            for l in f:
                for rel in json.loads(l).get("relations", []):
                    test_labels[rel.get("relation_family", "?")] += 1
    else:
        test_out = PROC / f"t1_biored_test{suffix}.jsonl"
        test_labels = Counter()
        with open(PROC / "t1_biored.jsonl") as fin, open(test_out, "w") as fout:
            for line in fin:
                rec = json.loads(line)
                if rec.get("source_split") == "test":
                    rec = remap_record(rec, label_fn)
                    fout.write(json.dumps(rec) + "\n")
                    for rel in rec.get("relations", []):
                        test_labels[rel.get("relation_family", "?")] += 1
    report["biored_test_gold"] = dict(test_labels)
    return report


def run() -> None:
    ensure_dirs()
    import json as _json
    from definitions.schema_definitions import sc0_label
    all_reports = {}

    print("=== Schema Package Remapping ===\n")

    # SC0 also needs remapping to collapse VARIANT_GENE (4 instances) to ASSOC_GENERAL
    # The existing *_trn.jsonl packages have VARIANT_GENE in BioRED (from original S2_current)
    # SC0 remapping applies sc0_label which now funnels Conversion to ASSOC_GENERAL

    for schema_name, label_fn, suffix, reuse_biored in [
        ("S_flat", sc0_label, "_Sflat", None),    # SC0: remap everything
        ("S_pair", sc1_label, "_Spair",  None),    # SC1: remap everything
        ("S_mech", sc3_label, "_Smech", "_Spair"),   # SC3: reuse SC1 BioRED (identical); only DrugProt differs
    ]:
        print(f"  Building packages for {schema_name} (suffix={suffix})...")
        r = build_schema_packages(schema_name, label_fn, suffix, reuse_biored)
        all_reports[schema_name] = r

        # Print T1 label distribution
        t1_biored = r.get("t1_biored", {})
        if "after" in t1_biored:
            print(f"    BioRED T1 labels ({schema_name}):")
            for lbl, cnt in sorted(t1_biored["after"].items(), key=lambda x: -x[1]):
                print(f"      {lbl}: {cnt}")

        t2_biored = r.get("t2_biored", {})
        if "after" in t2_biored:
            print(f"    BioRED T2 MeSH labels ({schema_name}):")
            for lbl, cnt in sorted(t2_biored["after"].items(), key=lambda x: -x[1]):
                print(f"      {lbl}: {cnt}")

        print(f"    BioRED test gold ({schema_name}):")
        for lbl, cnt in sorted(r.get("biored_test_gold", {}).items(), key=lambda x: -x[1]):
            print(f"      {lbl}: {cnt}")
        print()

    (REPORTS / "remapping_report.json").write_text(_json.dumps(all_reports, indent=2))
    print(f"  Outputs: remapping_report.json + *_Spair.jsonl, *_Smech.jsonl packages")
