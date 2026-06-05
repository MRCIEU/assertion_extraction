"""Self-measured BioRED test presence-F1 (benchmark axis)."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from importlib import import_module

_en = import_module("01_corpus_relevance.entity_normalization")

from .constants import INFER_BATCH_SIZE, LEAKED_PMIDS, MAX_SEQ_LENGTH, NEGATIVES_PER_POSITIVE, SAMPLING_SEED, TRAIN_PAIR_TYPES
from .train_core import RelationDataset, require_gpu


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


def _format_input(text: str, head: str, tail: str) -> str:
    if head and head in text and tail and tail in text:
        marked = text.replace(head, f"[E1]{head}[/E1]", 1)
        marked = marked.replace(tail, f"[E2]{tail}[/E2]", 1)
        return marked
    return f"[E1]{head}[/E1] {text} [E2]{tail}[/E2]"


def _examples_from_doc(doc: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
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
        head = _entity_surface(head_ent)
        tail = _entity_surface(tail_ent)
        positives.append(
            {"text": _format_input(text, head, tail), "label": 1, "pair_type": pt}
        )

    by_type: dict[str, list] = {}
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
            negatives.append({"text": _format_input(text, head, tail), "label": 0})
            added += 1

    return positives + negatives


def build_biored_test_examples(seed: int = SAMPLING_SEED) -> list[dict]:
    print("[benchmark] building BioRED test examples...", flush=True)
    rng = random.Random(seed)
    biored = load_dataset("bigbio/biored", "biored_bigbio_kb", trust_remote_code=True)
    examples: list[dict] = []
    for doc in biored["test"]:
        pmid = str(doc.get("document_id", ""))
        if pmid in LEAKED_PMIDS:
            continue
        examples.extend(_examples_from_doc(doc, rng))
    print(f"[benchmark] built {len(examples)} test examples (leaked PMIDs excluded)", flush=True)
    return examples


def evaluate_checkpoint_benchmark_f1(ckpt_dir: Path, test_examples: list[dict] | None = None) -> dict:
    if test_examples is None:
        test_examples = build_biored_test_examples()

    print(
        f"[benchmark] evaluating {ckpt_dir} on {len(test_examples)} BioRED test examples",
        flush=True,
    )
    device = require_gpu()
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    model.to(device)
    model.eval()

    ds = RelationDataset(test_examples, tokenizer, MAX_SEQ_LENGTH)
    loader = DataLoader(ds, batch_size=INFER_BATCH_SIZE, shuffle=False)
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)

    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            pred = logits.argmax(dim=-1).cpu().numpy()
            preds.extend(pred.tolist())
            labels.extend(batch["labels"].cpu().numpy().tolist())
            if batch_idx == 1 or batch_idx == n_batches or batch_idx % log_every == 0:
                print(
                    f"[benchmark] batch {batch_idx}/{n_batches} "
                    f"examples={len(labels)}/{len(test_examples)}",
                    flush=True,
                )

    f1 = float(f1_score(labels, preds, average="binary", zero_division=0))
    prec, rec, _, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    print(
        f"[benchmark] done f1={f1:.4f} precision={float(prec):.4f} recall={float(rec):.4f} "
        f"n_positives={int(sum(labels))}",
        flush=True,
    )
    return {
        "benchmark_f1": f1,
        "benchmark_precision": float(prec),
        "benchmark_recall": float(rec),
        "n_test_examples": len(test_examples),
        "n_positives": int(sum(labels)),
    }
