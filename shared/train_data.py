"""Build binary relation-presence training examples from BioRED + DrugProt."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from importlib import import_module

_en = import_module("01_corpus_relevance.entity_normalization")

from .constants import (
    LEAKED_PMIDS,
    MAX_TRAIN_EXAMPLES,
    NEGATIVES_PER_POSITIVE,
    SAMPLING_SEED,
    TRAIN_PAIR_TYPES,
)
from .marker_insert import bigbio_doc_text, format_marked_pair
from .paths import upstream_paths


def _examples_from_doc(doc: dict[str, Any], corpus: str, rng: random.Random) -> list[dict[str, Any]]:
    text = bigbio_doc_text(doc)
    if not text:
        return []
    entities = {e["id"]: e for e in doc.get("entities") or []}
    asserted: set[tuple[str, str, str]] = set()
    positives: list[dict[str, Any]] = []
    for rel in doc.get("relations") or []:
        e1 = entities.get(rel.get("arg1_id"))
        e2 = entities.get(rel.get("arg2_id"))
        if not e1 or not e2:
            continue
        pt = _en.civic_pair_type(e1.get("type", ""), e2.get("type", ""))
        if pt not in TRAIN_PAIR_TYPES:
            continue
        if _en.normalize_entity_type(e1["type"]) == "gene":
            head_ent, tail_ent = e1, e2
        elif _en.normalize_entity_type(e2["type"]) == "gene":
            head_ent, tail_ent = e2, e1
        else:
            head_ent, tail_ent = e1, e2
        key = (head_ent["id"], tail_ent["id"], pt)
        if key in asserted:
            continue
        asserted.add(key)
        marked, _method, span_meta = format_marked_pair(text, head_ent, tail_ent)
        positives.append({
            "text": marked,
            "label": 1,
            "corpus": corpus,
            "pair_type": pt,
            "pmid": str(doc.get("document_id", "")),
            **span_meta,
        })
    by_type: dict[str, list] = {}
    for ent in entities.values():
        ctype = _en.normalize_entity_type(ent.get("type", ""))
        if ctype:
            by_type.setdefault(ctype, []).append(ent)
    negatives: list[dict] = []
    for pos in positives:
        pt = pos["pair_type"]
        gene_t, other_t = pt.split("-")
        genes, others = by_type.get(gene_t, []), by_type.get(other_t, [])
        if not genes or not others:
            continue
        added, tries = 0, 0
        while added < NEGATIVES_PER_POSITIVE and tries < 50:
            tries += 1
            g, o = rng.choice(genes), rng.choice(others)
            if g["id"] == o["id"]:
                continue
            head_ent, tail_ent = (g, o) if gene_t == "gene" else (o, g)
            key = (head_ent["id"], tail_ent["id"], pt)
            if key in asserted:
                continue
            asserted.add(key)
            marked, _method, span_meta = format_marked_pair(text, head_ent, tail_ent)
            negatives.append({
                "text": marked,
                "label": 0,
                "corpus": corpus,
                "pair_type": pt,
                "pmid": str(doc.get("document_id", "")),
                **span_meta,
            })
            added += 1
    return positives + negatives


def _load_excluded_pmids(excluded_pmids_json: Path) -> set[str]:
    if excluded_pmids_json.exists():
        data = json.loads(excluded_pmids_json.read_text(encoding="utf-8"))
        return {str(x["pmid"]) if isinstance(x, dict) else str(x) for x in data.get("excluded_pmids", [])}
    return set()


def _assert_no_leaked_pmids(examples: list[dict], excluded_pmids_json: Path) -> None:
    pmids_in_data = {str(ex.get("pmid", "")) for ex in examples}
    leaked = pmids_in_data & LEAKED_PMIDS
    if leaked:
        raise RuntimeError(f"BLOCKING: leaked PMIDs in training data: {sorted(leaked)}")
    excluded = _load_excluded_pmids(excluded_pmids_json)
    if excluded and (pmids_in_data & excluded):
        raise RuntimeError(f"Training data contains excluded PMIDs: {sorted(pmids_in_data & excluded)}")


def build_train_val_examples(cache_dir: Path, *, force: bool = False) -> tuple[list[dict], list[dict]]:
    paths = upstream_paths()
    train_cache = cache_dir / "train_examples_train.jsonl"
    val_cache = cache_dir / "train_examples_val.jsonl"
    if train_cache.exists() and val_cache.exists() and not force:
        train_rows = [json.loads(l) for l in train_cache.read_text().splitlines() if l.strip()]
        val_rows = [json.loads(l) for l in val_cache.read_text().splitlines() if l.strip()]
        _assert_no_leaked_pmids(train_rows + val_rows, paths["excluded_pmids_json"])
        return train_rows, val_rows
    excluded = _load_excluded_pmids(paths["excluded_pmids_json"])
    rng = random.Random(SAMPLING_SEED)
    train_examples, val_examples = [], []
    for corpus, split in [("biored", "train"), ("drugprot", "train"), ("biored", "validation"), ("drugprot", "validation")]:
        ds = load_dataset(
            "bigbio/biored" if corpus == "biored" else "bigbio/drugprot",
            "biored_bigbio_kb" if corpus == "biored" else "drugprot_bigbio_kb",
            trust_remote_code=True,
        )
        target = train_examples if split == "train" else val_examples
        for doc in ds[split]:
            if str(doc.get("document_id", "")) in excluded:
                continue
            target.extend(_examples_from_doc(doc, corpus, rng))
    rng.shuffle(train_examples)
    train_examples = train_examples[:MAX_TRAIN_EXAMPLES]
    _assert_no_leaked_pmids(train_examples + val_examples, paths["excluded_pmids_json"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    for p, rows in [(train_cache, train_examples), (val_cache, val_examples)]:
        with p.open("w") as f:
            for ex in rows:
                f.write(json.dumps(ex) + "\n")
    print(f"Built train={len(train_examples)} val={len(val_examples)}; leak check passed", flush=True)
    return train_examples, val_examples
