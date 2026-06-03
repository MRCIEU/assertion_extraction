"""Build binary relation-presence training examples from BioRED + DrugProt."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from importlib import import_module

_en = import_module("01_corpus_relevance.entity_normalization")

from .config import (
    EXCLUDED_PMIDS_JSON,
    LEAKED_PMIDS,
    MAX_TRAIN_EXAMPLES,
    NEGATIVES_PER_POSITIVE,
    SAMPLING_SEED,
    TRAIN_CACHE_TRAIN,
    TRAIN_CACHE_VAL,
    TRAIN_PAIR_TYPES,
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
    if head and head in text and tail and tail in text:
        marked = text.replace(head, f"[E1]{head}[/E1]", 1)
        marked = marked.replace(tail, f"[E2]{tail}[/E2]", 1)
        return marked
    return f"[E1]{head}[/E1] {text} [E2]{tail}[/E2]"


def _examples_from_doc(doc: dict[str, Any], corpus: str, rng: random.Random) -> list[dict[str, Any]]:
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
    return set()


def _assert_no_leaked_pmids(examples: list[dict]) -> None:
    """Blocking assertion: the three known leaked PMIDs must be absent."""
    pmids_in_data = {str(ex.get("pmid", "")) for ex in examples}
    leaked = pmids_in_data & LEAKED_PMIDS
    if leaked:
        raise RuntimeError(
            f"BLOCKING: training data contains leaked evaluation PMIDs {sorted(leaked)}. "
            f"Expected absent: {sorted(LEAKED_PMIDS)}. See {EXCLUDED_PMIDS_JSON}"
        )

    excluded = _load_excluded_pmids()
    if excluded:
        other_leaked = pmids_in_data & excluded
        if other_leaked:
            raise RuntimeError(
                f"Training data contains excluded evaluation PMIDs: {sorted(other_leaked)}. "
                f"See {EXCLUDED_PMIDS_JSON}"
            )


def _load_corpus_split(corpus: str, split: str, excluded: set[str], rng: random.Random) -> list[dict]:
    if corpus == "biored":
        ds = load_dataset("bigbio/biored", "biored_bigbio_kb", trust_remote_code=True)
    else:
        ds = load_dataset("bigbio/drugprot", "drugprot_bigbio_kb", trust_remote_code=True)

    examples: list[dict] = []
    for doc in ds[split]:
        if str(doc.get("document_id", "")) in excluded:
            continue
        examples.extend(_examples_from_doc(doc, corpus, rng))
    return examples


def build_train_val_examples(force: bool = False) -> tuple[list[dict], list[dict]]:
    if TRAIN_CACHE_TRAIN.exists() and TRAIN_CACHE_VAL.exists() and not force:
        train_rows = [
            json.loads(line)
            for line in TRAIN_CACHE_TRAIN.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        val_rows = [
            json.loads(line)
            for line in TRAIN_CACHE_VAL.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        _assert_no_leaked_pmids(train_rows + val_rows)
        print(f"Loaded cached train={len(train_rows)} val={len(val_rows)} examples")
        return train_rows, val_rows

    excluded = _load_excluded_pmids()
    rng = random.Random(SAMPLING_SEED)

    print("Building train split (BioRED train + DrugProt train)...")
    train_examples: list[dict] = []
    train_examples.extend(_load_corpus_split("biored", "train", excluded, rng))
    train_examples.extend(_load_corpus_split("drugprot", "train", excluded, rng))

    print("Building validation split (BioRED validation + DrugProt validation)...")
    val_examples: list[dict] = []
    val_examples.extend(_load_corpus_split("biored", "validation", excluded, rng))
    val_examples.extend(_load_corpus_split("drugprot", "validation", excluded, rng))

    rng.shuffle(train_examples)
    if len(train_examples) > MAX_TRAIN_EXAMPLES:
        train_examples = train_examples[:MAX_TRAIN_EXAMPLES]

    _assert_no_leaked_pmids(train_examples + val_examples)

    for path, rows in ((TRAIN_CACHE_TRAIN, train_examples), (TRAIN_CACHE_VAL, val_examples)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for ex in rows:
                f.write(json.dumps(ex) + "\n")

    n_pos = sum(1 for e in train_examples if e["label"] == 1)
    print(f"Built train={len(train_examples)} ({n_pos} pos) val={len(val_examples)} examples")
    print(f"Leak check passed: none of {sorted(LEAKED_PMIDS)} in training data")
    return train_examples, val_examples
