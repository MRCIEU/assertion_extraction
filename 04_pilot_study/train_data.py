"""Build binary relation-presence training examples from BioRED + DrugProt."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset

# Reuse step-01 entity normalization
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from importlib import import_module

_en = import_module("01_corpus_relevance.entity_normalization")

from .config import (
    EXCLUDED_PMIDS_JSON,
    NEGATIVES_PER_POSITIVE,
    SAMPLING_SEED,
    TRAIN_CACHE_JSON,
    TRAIN_PAIR_TYPES,
    MAX_TRAIN_EXAMPLES,
    TRAINING_PMIDS_CLEAN_JSON,
)


def _doc_text(doc: dict[str, Any]) -> str:
    parts: list[str] = []
    for passage in doc.get("passages") or []:
        text = passage.get("text")
        if isinstance(text, list):
            parts.extend(str(t) for t in text if t)
        elif text:
            parts.append(str(text))
    return " ".join(parts).strip()


def _entity_surface(entity: dict[str, Any]) -> str:
    text = entity.get("text") or [""]
    if isinstance(text, list):
        return str(text[0]) if text else ""
    return str(text)


def _pair_type_for_entities(e1: dict, e2: dict) -> str | None:
    return _en.civic_pair_type(e1.get("type", ""), e2.get("type", ""))


def _format_input(text: str, head: str, tail: str) -> str:
    """Entity-marked abstract (same framing as CIViC eval)."""
    if head and head in text and tail and tail in text:
        marked = text.replace(head, f"[E1]{head}[/E1]", 1)
        marked = marked.replace(tail, f"[E2]{tail}[/E2]", 1)
        return marked
    return f"[E1]{head}[/E1] {text} [E2]{tail}[/E2]"


def _examples_from_doc(
    doc: dict[str, Any],
    corpus: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    text = _doc_text(doc)
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
        pt = _pair_type_for_entities(e1, e2)
        if pt not in TRAIN_PAIR_TYPES:
            continue
        head_ent, tail_ent = (e1, e2) if pt == _pair_type_for_entities(e1, e2) else (e1, e2)
        # Orient gene first for gene-* pairs
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
        head = _entity_surface(head_ent)
        tail = _entity_surface(tail_ent)
        positives.append(
            {
                "text": _format_input(text, head, tail),
                "label": 1,
                "corpus": corpus,
                "pair_type": pt,
                "pmid": str(doc.get("document_id", "")),
            }
        )

    # Natural negatives: co-occurring same-type pairs not in asserted set
    by_type: dict[str, list[dict]] = {}
    for ent in entities.values():
        ctype = _en.normalize_entity_type(ent.get("type", ""))
        if ctype:
            by_type.setdefault(ctype, []).append(ent)

    negatives: list[dict[str, Any]] = []
    for pos in positives:
        pt = pos["pair_type"]
        gene_t, other_t = pt.split("-")
        genes = by_type.get(gene_t, [])
        others = by_type.get(other_t, [])
        if not genes or not others:
            continue
        tries = 0
        added = 0
        while added < NEGATIVES_PER_POSITIVE and tries < 50:
            tries += 1
            g = rng.choice(genes)
            o = rng.choice(others)
            if g["id"] == o["id"]:
                continue
            head_ent, tail_ent = (g, o) if gene_t == "gene" else (o, g)
            key = (head_ent["id"], tail_ent["id"], pt)
            if key in asserted:
                continue
            asserted.add(key)
            head = _entity_surface(head_ent)
            tail = _entity_surface(tail_ent)
            negatives.append(
                {
                    "text": _format_input(text, head, tail),
                    "label": 0,
                    "corpus": corpus,
                    "pair_type": pt,
                    "pmid": str(doc.get("document_id", "")),
                }
            )
            added += 1

    return positives + negatives


def _load_excluded_pmids() -> set[str]:
    if EXCLUDED_PMIDS_JSON.exists():
        data = json.loads(EXCLUDED_PMIDS_JSON.read_text(encoding="utf-8"))
        return {str(x["pmid"]) if isinstance(x, dict) else str(x) for x in data.get("excluded_pmids", [])}
    if TRAINING_PMIDS_CLEAN_JSON.exists():
        return set()
    return set()


def _assert_no_leaked_pmids(examples: list[dict]) -> None:
    excluded = _load_excluded_pmids()
    if not excluded:
        return
    leaked = {ex.get("pmid") for ex in examples if str(ex.get("pmid", "")) in excluded}
    if leaked:
        raise RuntimeError(
            f"Training data contains excluded evaluation PMIDs: {sorted(leaked)}. "
            f"See {EXCLUDED_PMIDS_JSON}"
        )


def build_train_examples(force: bool = False) -> list[dict[str, Any]]:
    if TRAIN_CACHE_JSON.exists() and not force:
        rows = [json.loads(line) for line in TRAIN_CACHE_JSON.read_text(encoding="utf-8").splitlines() if line.strip()]
        _assert_no_leaked_pmids(rows)
        print(f"Loaded {len(rows)} cached training examples")
        return rows

    excluded = _load_excluded_pmids()
    rng = random.Random(SAMPLING_SEED)
    examples: list[dict[str, Any]] = []

    print("Loading BioRED train+validation splits...")
    biored = load_dataset("bigbio/biored", "biored_bigbio_kb", trust_remote_code=True)
    for split in ("train", "validation"):
        for doc in biored[split]:
            if str(doc.get("document_id", "")) in excluded:
                continue
            examples.extend(_examples_from_doc(doc, "biored", rng))

    print("Loading DrugProt train+validation splits...")
    drugprot = load_dataset("bigbio/drugprot", "drugprot_bigbio_kb", trust_remote_code=True)
    for split in ("train", "validation"):
        for doc in drugprot[split]:
            if str(doc.get("document_id", "")) in excluded:
                continue
            examples.extend(_examples_from_doc(doc, "drugprot", rng))

    rng.shuffle(examples)
    if len(examples) > MAX_TRAIN_EXAMPLES:
        examples = examples[:MAX_TRAIN_EXAMPLES]

    _assert_no_leaked_pmids(examples)

    with TRAIN_CACHE_JSON.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    n_pos = sum(1 for e in examples if e["label"] == 1)
    print(f"Built {len(examples)} training examples ({n_pos} pos / {len(examples)-n_pos} neg)")
    return examples
