"""PMID-level diagnostics: corpus overlap, annotation conflicts, train/eval leakage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from datasets import load_dataset

from .config import (
    EVALUABLE_INVENTORY_CSV,
    EXCLUDED_PMIDS_JSON,
    OUTPUT_DIR,
    PMID_CONFLICT_EXAMPLES_CSV,
    PMID_CONFLICTS_CSV,
    PMID_LEAKAGE_CSV,
    PMID_OVERLAP_CSV,
    TRAINING_PMIDS_CLEAN_JSON,
    CORPORA,
)
from .entity_normalization import (
    civic_pair_type,
    entity_surface,
    normalize_entity_text,
    normalize_entity_type,
    ordered_pair_key,
)


def _load_training_docs(corpus_key: str) -> dict[str, dict]:
    """PMID -> document for training-relevant splits (train + validation)."""
    spec = CORPORA[corpus_key]
    dataset = load_dataset(spec["hf_id"], spec["config"], trust_remote_code=True)
    docs: dict[str, dict] = {}
    for split in spec["train_splits"]:
        if split not in dataset:
            continue
        for doc in dataset[split]:
            pmid = str(doc["document_id"])
            docs[pmid] = {"split": split, "doc": doc, "corpus": corpus_key}
    return docs


def _positive_pairs(doc: dict) -> set[tuple[str, str, str]]:
    entities = {e["id"]: e for e in doc.get("entities") or []}
    positive: set[tuple[str, str, str]] = set()
    for rel in doc.get("relations") or []:
        e1 = entities.get(rel.get("arg1_id"))
        e2 = entities.get(rel.get("arg2_id"))
        if not e1 or not e2:
            continue
        key = ordered_pair_key(e1["type"], entity_surface(e1), e2["type"], entity_surface(e2))
        if key:
            positive.add(key)
    return positive


def _entity_index(doc: dict) -> dict[tuple[str, str], set[str]]:
    """Map (civic_type, norm_text) -> raw surface forms in document."""
    index: dict[tuple[str, str], set[str]] = {}
    for ent in doc.get("entities") or []:
        ctype = normalize_entity_type(ent.get("type", ""))
        if not ctype:
            continue
        norm = normalize_entity_text(entity_surface(ent))
        if not norm:
            continue
        index.setdefault((ctype, norm), set()).add(entity_surface(ent))
    return index


def _co_occurring_pairs(doc: dict) -> set[tuple[str, str, str]]:
    """All CIViC-relevant entity pairs co-occurring in a document (binary framing universe)."""
    by_type: dict[str, list[tuple[str, str]]] = {}
    for ent in doc.get("entities") or []:
        ctype = normalize_entity_type(ent.get("type", ""))
        if not ctype:
            continue
        norm = normalize_entity_text(entity_surface(ent))
        if norm:
            by_type.setdefault(ctype, []).append((norm, entity_surface(ent)))

    pairs: set[tuple[str, str, str]] = set()
    types = list(by_type.keys())
    for i, ta in enumerate(types):
        for tb in types[i:]:
            for norm_a, surf_a in by_type[ta]:
                for norm_b, surf_b in by_type[tb]:
                    if ta == tb and norm_a >= norm_b:
                        continue
                    # Find raw types for ordered_pair_key
                    raw_a = next(e for e in doc["entities"] if normalize_entity_text(entity_surface(e)) == norm_a)
                    raw_b = next(e for e in doc["entities"] if normalize_entity_text(entity_surface(e)) == norm_b)
                    key = ordered_pair_key(raw_a["type"], surf_a, raw_b["type"], surf_b)
                    if key:
                        pairs.add(key)
    return pairs


def _shared_coannotated_pairs(
    bio_doc: dict, drug_doc: dict
) -> set[tuple[str, str, str]]:
    """Pairs where both entities appear in BOTH corpora for the same PMID."""
    bio_idx = _entity_index(bio_doc)
    drug_idx = _entity_index(drug_doc)
    shared: set[tuple[str, str, str]] = set()
    for key in _co_occurring_pairs(bio_doc) | _co_occurring_pairs(drug_doc):
        pt, e1, e2 = key
        t1, t2 = pt.split("-")
        if (t1, e1) in bio_idx and (t2, e2) in bio_idx and (t1, e1) in drug_idx and (t2, e2) in drug_idx:
            shared.add(key)
    return shared


def _load_eval_pmids() -> dict[str, Any]:
    """Evaluation PMIDs from step-00 abstract-grounded inventory."""
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    match_mod = __import__(
        "00_civic_feasibility.matching",
        fromlist=["entity_mentioned"],
    )
    entity_mentioned = match_mod.entity_mentioned
    evidence_path = repo.parent / "projects" / "project_1" / "data" / "00_civic_feasibility" / "evidence_items.json"
    if not evidence_path.exists():
        from _paths import OUTPUT_ROOT
        evidence_path = OUTPUT_ROOT / "data" / "00_civic_feasibility" / "evidence_items.json"
    records = json.loads(evidence_path.read_text(encoding="utf-8"))
    abstracts = {}
    for item in records:
        src = item.get("source") or {}
        pmid = str(src.get("citationId") or "")
        abstract = src.get("abstract") or ""
        if pmid and abstract:
            abstracts[pmid] = abstract

    inventory = pd.read_csv(EVALUABLE_INVENTORY_CSV)
    pmids: set[str] = set()
    n_targets = 0
    for _, r in inventory.iterrows():
        if not r.get("is_evaluable_target"):
            continue
        pmid = str(r.get("pmid") or "")
        abstract = abstracts.get(pmid, "")
        head = str(r.get("head_entity") or "")
        tail = str(r.get("tail_entity") or "")
        ht = str(r.get("head_type") or "")
        tt = str(r.get("tail_type") or "")
        if pmid and abstract and entity_mentioned(abstract, head, ht) and entity_mentioned(abstract, tail, tt):
            pmids.add(pmid)
            n_targets += 1
    pmid_list = sorted(pmids)
    return {
        "unique_pmids": pmid_list,
        "n_unique_pmids": len(pmid_list),
        "n_ranking_targets": n_targets,
        "source": str(EVALUABLE_INVENTORY_CSV),
    }


def _count_relations_on_pmids(docs: dict[str, dict], pmids: set[str]) -> dict[str, int]:
    rels = 0
    docs_n = 0
    civic_rels = 0
    for pmid in pmids:
        if pmid not in docs:
            continue
        doc = docs[pmid]["doc"]
        docs_n += 1
        for rel in doc.get("relations") or []:
            rels += 1
            entities = {e["id"]: e for e in doc.get("entities") or []}
            e1 = entities.get(rel.get("arg1_id"))
            e2 = entities.get(rel.get("arg2_id"))
            if e1 and e2 and civic_pair_type(e1["type"], e2["type"]):
                civic_rels += 1
    return {"documents": docs_n, "all_relations": rels, "civic_pair_relations": civic_rels}


def run_pmid_diagnostics() -> dict[str, Any]:
    print("\n=== PMID diagnostics (D-overlap / D-conflict / D-leakage) ===")

    bio_docs = _load_training_docs("biored")
    drug_docs = _load_training_docs("drugprot")
    bio_pmids = set(bio_docs)
    drug_pmids = set(drug_docs)
    overlap = bio_pmids & drug_pmids
    union = bio_pmids | drug_pmids
    jaccard = len(overlap) / len(union) if union else 0.0

    overlap_row = {
        "biored_pmids": len(bio_pmids),
        "drugprot_pmids": len(drug_pmids),
        "intersection": len(overlap),
        "union": len(union),
        "jaccard": round(jaccard, 6),
        "biored_only": len(bio_pmids - drug_pmids),
        "drugprot_only": len(drug_pmids - bio_pmids),
        "training_splits": "train+validation",
    }
    pd.DataFrame([overlap_row]).to_csv(PMID_OVERLAP_CSV, index=False)
    print(
        f"  D-overlap: BioRED={len(bio_pmids)} DrugProt={len(drug_pmids)} "
        f"intersection={len(overlap)} Jaccard={jaccard:.4f}"
    )

    conflict_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    for pmid in sorted(overlap):
        bio_doc = bio_docs[pmid]["doc"]
        drug_doc = drug_docs[pmid]["doc"]
        bio_pos = _positive_pairs(bio_doc)
        drug_pos = _positive_pairs(drug_doc)
        shared = _shared_coannotated_pairs(bio_doc, drug_doc)
        n_conflict = 0
        for key in shared:
            b = key in bio_pos
            d = key in drug_pos
            if b == d:
                continue
            n_conflict += 1
            pt, e1, e2 = key
            row = {
                "pmid": pmid,
                "pair_type": pt,
                "entity_1": e1,
                "entity_2": e2,
                "biored_binary": "positive" if b else "negative",
                "drugprot_binary": "positive" if d else "negative",
            }
            conflict_rows.append(row)
            if len(example_rows) < 10:
                example_rows.append(row)
        print(
            f"    PMID {pmid}: co-annotated pairs={len(shared)} conflicts={n_conflict} "
            f"(BioRED rels={len(bio_pos)} DrugProt rels={len(drug_pos)})"
        )

    n_co = sum(
        len(_shared_coannotated_pairs(bio_docs[p]["doc"], drug_docs[p]["doc"])) for p in overlap
    )
    conflict_summary = {
        "overlapping_pmids": len(overlap),
        "co_annotated_pairs": n_co,
        "conflict_count": len(conflict_rows),
        "conflict_rate": round(len(conflict_rows) / n_co, 6) if n_co else 0.0,
        "entity_matching": "normalised surface text + CIViC entity/pair type (see entity_normalization.py)",
    }
    pd.DataFrame([conflict_summary]).to_csv(PMID_CONFLICTS_CSV, index=False)
    pd.DataFrame(conflict_rows).to_csv(PMID_CONFLICT_EXAMPLES_CSV, index=False)
    print(
        f"  D-conflict: co-annotated={n_co} conflicts={len(conflict_rows)} "
        f"rate={conflict_summary['conflict_rate']:.4f}"
    )

    eval_info = _load_eval_pmids()
    eval_pmids = set(eval_info["unique_pmids"])
    leak_bio = bio_pmids & eval_pmids
    leak_drug = drug_pmids & eval_pmids
    leak_combined = (bio_pmids | drug_pmids) & eval_pmids

    leak_rows = [
        {
            "corpus": "BioRED",
            "training_pmids": len(bio_pmids),
            "eval_unique_pmids": eval_info["n_unique_pmids"],
            "overlap_count": len(leak_bio),
            "leaked_pmids": ";".join(sorted(leak_bio)) if leak_bio else "",
        },
        {
            "corpus": "DrugProt",
            "training_pmids": len(drug_pmids),
            "eval_unique_pmids": eval_info["n_unique_pmids"],
            "overlap_count": len(leak_drug),
            "leaked_pmids": ";".join(sorted(leak_drug)) if leak_drug else "",
        },
        {
            "corpus": "combined",
            "training_pmids": len(union),
            "eval_unique_pmids": eval_info["n_unique_pmids"],
            "overlap_count": len(leak_combined),
            "leaked_pmids": ";".join(sorted(leak_combined)) if leak_combined else "",
        },
    ]
    pd.DataFrame(leak_rows).to_csv(PMID_LEAKAGE_CSV, index=False)
    print(
        f"  D-leakage: eval PMIDs={eval_info['n_unique_pmids']} "
        f"(backing {eval_info['n_ranking_targets']} abstract-grounded targets) | "
        f"overlap BioRED={len(leak_bio)} DrugProt={len(leak_drug)} combined={len(leak_combined)}"
    )

    leaked = sorted(leak_combined)
    leakage_clean = len(leaked) == 0
    removal_stats = {}
    if leaked:
        leaked_set = set(leaked)
        removal_stats = {
            "biored": _count_relations_on_pmids(bio_docs, leaked_set),
            "drugprot": _count_relations_on_pmids(drug_docs, leaked_set),
        }
        exclusion = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "excluded_pmids": [
                {
                    "pmid": p,
                    "reason": "train/eval PMID overlap with CIViC abstract-grounded evaluation set",
                }
                for p in leaked
            ],
            "source_eval": eval_info["source"],
            "relations_removed_if_excluded": removal_stats,
        }
        EXCLUDED_PMIDS_JSON.write_text(json.dumps(exclusion, indent=2), encoding="utf-8")
        print(f"  LEAKAGE DETECTED: {len(leaked)} PMIDs -> {EXCLUDED_PMIDS_JSON}")
        for pmid in leaked:
            print(f"    - {pmid}")
    else:
        if EXCLUDED_PMIDS_JSON.exists():
            EXCLUDED_PMIDS_JSON.unlink()

    clean = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_splits": "train+validation",
        "leakage_free": leakage_clean,
        "excluded_pmids": leaked,
        "biored_training_pmids": sorted(bio_pmids - set(leaked)),
        "drugprot_training_pmids": sorted(drug_pmids - set(leaked)),
    }
    TRAINING_PMIDS_CLEAN_JSON.write_text(json.dumps(clean, indent=2), encoding="utf-8")

    if n_co == 0:
        risk = "negligible (no co-annotated entity pairs on overlapping PMIDs)"
    elif conflict_summary["conflict_rate"] < 0.05:
        risk = "negligible"
    elif conflict_summary["conflict_rate"] < 0.15:
        risk = "moderate"
    else:
        risk = "serious"

    result = {
        "overlap": overlap_row,
        "conflict": conflict_summary,
        "conflict_examples": example_rows,
        "leakage": {
            "eval_unique_pmids": eval_info["n_unique_pmids"],
            "eval_ranking_targets": eval_info["n_ranking_targets"],
            "overlap_biored": len(leak_bio),
            "overlap_drugprot": len(leak_drug),
            "overlap_combined": len(leak_combined),
            "leaked_pmids": leaked,
            "leakage_free": leakage_clean,
            "relations_removed_if_excluded": removal_stats,
        },
        "conflict_risk": risk,
    }
    (OUTPUT_DIR / "pmid_diagnostics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
