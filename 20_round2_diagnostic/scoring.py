"""Inference-only scoring of existing checkpoints (CPU if no GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from importlib import import_module

_r1 = import_module("10_round1_benchmark_kb.config")
_r1bench = import_module("10_round1_benchmark_kb.benchmark_eval")
_r1inf = import_module("10_round1_benchmark_kb.inference")
_r1dist = import_module("10_round1_benchmark_kb.distance_analysis")
_r1rank = import_module("10_round1_benchmark_kb.metrics_ranking")
_r1pool = import_module("10_round1_benchmark_kb.pool_loader")
_r1fmt = import_module("10_round1_benchmark_kb.input_format")

INFER_BATCH_SIZE = _r1.INFER_BATCH_SIZE
MAX_SEQ_LENGTH = _r1.MAX_SEQ_LENGTH
PAIR_TYPES = _r1.PAIR_TYPES


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("  (using CPU for checkpoint scoring)")
    return torch.device("cpu")


def score_candidates_at_checkpoint(ckpt_dir: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    device = _device()
    texts = [_r1fmt.format_eval_input(row) for _, row in candidates.iterrows()]
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    model.to(device)
    model.eval()
    probs: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), INFER_BATCH_SIZE):
            batch = texts[i : i + INFER_BATCH_SIZE]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
    out = candidates.copy()
    out["score"] = probs
    return out


def benchmark_f1_at_checkpoint(
    ckpt_dir: Path,
    spec,
    test_examples: list[dict] | None = None,
) -> float:
    if test_examples is None:
        test_examples = _r1bench.build_biored_test_examples()
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    model.to(device)
    model.eval()
    RelationDataset = import_module("10_round1_benchmark_kb.train").RelationDataset
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score

    ds = RelationDataset(test_examples, tokenizer, MAX_SEQ_LENGTH)
    loader = DataLoader(ds, batch_size=INFER_BATCH_SIZE, shuffle=False)
    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            pred = logits.argmax(dim=-1).cpu().numpy()
            preds.extend(pred.tolist())
            labels.extend(batch["labels"].cpu().numpy().tolist())
    return float(f1_score(labels, preds, average="binary", zero_division=0))


def kb_metrics_from_scores(
    scores: pd.DataFrame,
    pool: pd.DataFrame,
    label: str,
) -> dict[str, float]:
    merged = scores.merge(pool[["candidate_id", "subset"]], on="candidate_id", how="inner")
    out: dict[str, float] = {}
    for pt in PAIR_TYPES:
        pt_sub = merged[merged["pair_type"] == pt]
        if not pt_sub.empty:
            out[f"kb_mrr_{pt.replace('-', '_')}"] = _r1rank.compute_mrr(pt_sub)
    out["kb_mrr_overall"] = _r1rank.compute_mrr(merged)
    out["kb_mrr_hard_cross_sentence"] = _r1rank.compute_mrr(
        merged[merged["subset"] == "hard_cross_sentence"]
    )
    out["kb_mrr_easy_co_sentence"] = _r1rank.compute_mrr(
        merged[merged["subset"] == "easy_co_sentence"]
    )
    return out
