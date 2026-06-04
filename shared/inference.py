"""Score frozen CIViC pool from any checkpoint path."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .constants import INFER_BATCH_SIZE, MAX_SEQ_LENGTH
from .input_format import format_eval_input
from .pool_loader import load_primary_candidates
from .train_core import require_gpu


def score_checkpoint_path(
    ckpt_dir: Path,
    candidates: pd.DataFrame | None = None,
    *,
    model_id: str = "unknown",
    seed: int = -1,
    run_id: str | None = None,
) -> pd.DataFrame:
    if candidates is None:
        candidates = load_primary_candidates()
    device = require_gpu()
    texts = [format_eval_input(row) for _, row in candidates.iterrows()]
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
    probs: list[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), INFER_BATCH_SIZE):
            enc = tokenizer(texts[i : i + INFER_BATCH_SIZE], truncation=True, padding=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            p = torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
    out = candidates.copy()
    out["model_id"] = model_id
    out["seed"] = seed
    out["run_id"] = run_id or f"{model_id}_seed_{seed}"
    out["score"] = probs
    return out


def write_scores_jsonl(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), default=str) + "\n")
