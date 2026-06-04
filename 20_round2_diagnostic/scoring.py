"""Inference-only scoring of existing checkpoints."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shared.benchmark_eval import evaluate_checkpoint_benchmark_f1
from shared.constants import INFER_BATCH_SIZE, MAX_SEQ_LENGTH, PAIR_TYPES
from shared.input_format import format_eval_input
from shared.metrics_ranking import compute_mrr


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("  (using CPU for checkpoint scoring)")
    return torch.device("cpu")


def score_candidates_at_checkpoint(ckpt_dir: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    device = _device()
    texts = [format_eval_input(row) for _, row in candidates.iterrows()]
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


def benchmark_f1_at_checkpoint(ckpt_dir: Path, test_examples: list[dict] | None = None) -> float:
    from shared.benchmark_eval import build_biored_test_examples

    if test_examples is None:
        test_examples = build_biored_test_examples()
    return float(evaluate_checkpoint_benchmark_f1(ckpt_dir, test_examples)["benchmark_f1"])


def kb_metrics_from_scores(scores: pd.DataFrame, pool: pd.DataFrame) -> dict[str, float]:
    merged = scores.merge(pool[["candidate_id", "subset"]], on="candidate_id", how="inner")
    out: dict[str, float] = {}
    for pt in PAIR_TYPES:
        pt_sub = merged[merged["pair_type"] == pt]
        if not pt_sub.empty:
            out[f"kb_mrr_{pt.replace('-', '_')}"] = compute_mrr(pt_sub)
    out["kb_mrr_overall"] = compute_mrr(merged)
    out["kb_mrr_hard"] = compute_mrr(merged[merged["subset"] == "hard_cross_sentence"])
    out["kb_mrr_easy"] = compute_mrr(merged[merged["subset"] == "easy_co_sentence"])
    return out
