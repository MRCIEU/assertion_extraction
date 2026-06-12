"""Inference-only scoring of per-epoch checkpoints (benchmark + KB axes paired)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shared.benchmark_eval import build_biored_test_examples, evaluate_checkpoint_benchmark_f1
from shared.constants import INFER_BATCH_SIZE, MAX_SEQ_LENGTH, PAIR_TYPES, RECALL_K_VALUES
from shared.input_format import format_eval_input
from shared.metrics_ranking import compute_mrr, compute_recall_at_k


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


def benchmark_f1_at_checkpoint(
    ckpt_dir: Path,
    test_examples: list[dict] | None = None,
) -> float:
    if test_examples is None:
        test_examples = build_biored_test_examples()
    return float(evaluate_checkpoint_benchmark_f1(ckpt_dir, test_examples)["benchmark_f1"])


def _subset_metrics(scores: pd.DataFrame, pool: pd.DataFrame, subset: str | None) -> dict[str, float]:
    merged = scores.merge(pool[["candidate_id", "subset"]], on="candidate_id", how="inner")
    if subset is not None:
        merged = merged[merged["subset"] == subset]
    if merged.empty:
        return {}
    out: dict[str, float] = {"mrr": compute_mrr(merged)}
    for k in RECALL_K_VALUES:
        out[f"recall_at_{k}"] = compute_recall_at_k(merged, k)
    return out


def kb_metrics_from_scores(scores: pd.DataFrame, pool: pd.DataFrame) -> dict[str, float]:
    """Full KB metrics: overall, pair types, easy/hard subsets, pair×subset crosses, recall@k."""
    merged = scores.merge(pool[["candidate_id", "subset"]], on="candidate_id", how="inner")
    out: dict[str, float] = {}

    for pt in PAIR_TYPES:
        pt_sub = merged[merged["pair_type"] == pt]
        if pt_sub.empty:
            continue
        key = pt.replace("-", "_")
        out[f"kb_mrr_{key}"] = compute_mrr(pt_sub)
        for k in RECALL_K_VALUES:
            out[f"kb_recall{k}_{key}"] = compute_recall_at_k(pt_sub, k)
        for label, subset in [("hard", "hard_cross_sentence"), ("easy", "easy_co_sentence")]:
            cross = pt_sub[pt_sub["subset"] == subset]
            if cross.empty:
                continue
            out[f"kb_mrr_{key}_{label}"] = compute_mrr(cross)
            for rk in RECALL_K_VALUES:
                out[f"kb_recall{rk}_{key}_{label}"] = compute_recall_at_k(cross, rk)

    for label, subset in [("overall", None), ("hard", "hard_cross_sentence"), ("easy", "easy_co_sentence")]:
        m = _subset_metrics(scores, pool, subset)
        if not m:
            continue
        suffix = "" if label == "overall" else f"_{label}"
        out[f"kb_mrr{suffix}"] = m["mrr"]
        for k in RECALL_K_VALUES:
            out[f"kb_recall{k}{suffix}"] = m[f"recall_at_{k}"]

    return out


def score_checkpoint_full(
    ckpt_dir: Path,
    candidates: pd.DataFrame,
    pool: pd.DataFrame,
    test_examples: list[dict] | None = None,
) -> dict[str, float]:
    """Paired benchmark F1 + KB metrics at one checkpoint."""
    bench = benchmark_f1_at_checkpoint(ckpt_dir, test_examples)
    scores = score_candidates_at_checkpoint(ckpt_dir, candidates)
    kb = kb_metrics_from_scores(scores, pool)
    return {"benchmark_f1": bench, **kb}
