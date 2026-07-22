"""Self-measured BioRED test presence-F1 (benchmark axis)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from importlib import import_module

_en = import_module("01_corpus_relevance.entity_normalization")

from .constants import INFER_BATCH_SIZE, LEAKED_PMIDS, MAX_SEQ_LENGTH, NEGATIVES_PER_POSITIVE, SAMPLING_SEED, TRAIN_PAIR_TYPES
from .marker_insert import bigbio_doc_text, format_marked_pair
from .train_core import RelationDataset, require_gpu


def _examples_from_doc(doc: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
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
        marked, _method, _meta = format_marked_pair(text, head_ent, tail_ent)
        positives.append({"text": marked, "label": 1, "pair_type": pt})

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
            marked, _method, _meta = format_marked_pair(text, head_ent, tail_ent)
            negatives.append({"text": marked, "label": 0, "pair_type": pt})
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


def evaluate_model_benchmark_f1(
    model,
    tokenizer,
    test_examples: list[dict],
    *,
    label: str = "model",
    device=None,
    save_probs_path: Path | str | None = None,
) -> dict:
    """BioRED test presence F1 for an in-memory sequence-classification model."""
    if device is None:
        device = require_gpu()
    print(
        f"[benchmark] evaluating {label} on {len(test_examples)} BioRED test examples",
        flush=True,
    )
    model.to(device)
    model.eval()

    ds = RelationDataset(test_examples, tokenizer, MAX_SEQ_LENGTH)
    loader = DataLoader(ds, batch_size=INFER_BATCH_SIZE, shuffle=False)
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)

    preds: list[int] = []
    labels: list[int] = []
    probs_pos: list[float] = []
    prob_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            pred = logits.argmax(dim=-1).cpu().numpy()
            preds.extend(pred.tolist())
            labels.extend(batch["labels"].cpu().numpy().tolist())
            probs_pos.extend(prob.tolist())
            if save_probs_path is not None:
                start = len(preds) - len(pred)
                for i, ex_idx in enumerate(range(start, len(preds))):
                    ex = test_examples[ex_idx]
                    prob_rows.append(
                        {
                            "example_idx": ex_idx,
                            "label": int(labels[ex_idx]),
                            "pred": int(preds[ex_idx]),
                            "prob_positive": float(probs_pos[ex_idx]),
                            "pair_type": ex.get("pair_type"),
                        }
                    )
            if batch_idx == 1 or batch_idx == n_batches or batch_idx % log_every == 0:
                print(
                    f"[benchmark] batch {batch_idx}/{n_batches} "
                    f"examples={len(labels)}/{len(test_examples)}",
                    flush=True,
                )

    f1 = float(f1_score(labels, preds, average="binary", zero_division=0))
    prec, rec, _, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    y = np.array(labels, dtype=int)
    p = np.array(probs_pos, dtype=float)
    auc_pr = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float(y.mean())
    auc_roc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    if save_probs_path is not None:
        out = Path(save_probs_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r) for r in prob_rows) + "\n", encoding="utf-8")
    print(
        f"[benchmark] done f1={f1:.4f} precision={float(prec):.4f} recall={float(rec):.4f} "
        f"auc_pr={auc_pr:.4f} auc_roc={auc_roc:.4f} n_positives={int(sum(labels))}",
        flush=True,
    )
    return {
        "benchmark_f1": f1,
        "benchmark_precision": float(prec),
        "benchmark_recall": float(rec),
        "benchmark_auc_pr": auc_pr,
        "benchmark_auc_roc": auc_roc,
        "n_test_examples": len(test_examples),
        "n_positives": int(sum(labels)),
        "probs_path": str(save_probs_path) if save_probs_path else None,
    }


def evaluate_checkpoint_benchmark_f1(ckpt_dir: Path, test_examples: list[dict] | None = None) -> dict:
    if test_examples is None:
        test_examples = build_biored_test_examples()

    device = require_gpu()
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    return evaluate_model_benchmark_f1(model, tokenizer, test_examples, label=str(ckpt_dir), device=device)
