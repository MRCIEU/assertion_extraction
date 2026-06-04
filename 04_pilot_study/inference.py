"""Score frozen CIViC pool with minimally trained checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import (
    CHECKPOINT_DIR,
    INFER_BATCH_SIZE,
    MAX_SEQ_LENGTH,
    MODELS,
    SCORES_DIR,
    TRAIN_SEEDS,
    ModelSpec,
)
from .input_format import format_eval_input
from .pool_loader import load_primary_candidates
from .train import checkpoint_path


def _require_gpu() -> torch.device:
    if not torch.cuda.is_available():
        print("ERROR: GPU required for inference.", file=sys.stderr)
        sys.exit(1)
    return torch.device("cuda")


def _score_path(spec: ModelSpec) -> Path:
    return SCORES_DIR / f"04_scores_{spec.model_id}.jsonl"


def _predict_probs(model, tokenizer, texts: list[str], device) -> list[float]:
    probs: list[float] = []
    model.eval()
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
    return probs


def score_model(spec: ModelSpec, candidates: pd.DataFrame | None = None, force: bool = False) -> pd.DataFrame:
    out_path = _score_path(spec)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"  Using cached scores: {out_path}")
        return pd.read_json(out_path, lines=True)

    if candidates is None:
        candidates = load_primary_candidates()

    device = _require_gpu()
    texts = [format_eval_input(row) for _, row in candidates.iterrows()]

    seed_probs: list[np.ndarray] = []
    for seed in TRAIN_SEEDS:
        ckpt = checkpoint_path(spec, seed)
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint {ckpt}; run training first.")
        print(f"  Scoring with {spec.model_id} seed={seed}")
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt)
        model.to(device)
        seed_probs.append(np.array(_predict_probs(model, tokenizer, texts, device)))

    mean_probs = np.mean(seed_probs, axis=0)
    out = candidates.copy()
    out["model_id"] = spec.model_id
    out["score"] = mean_probs

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in out.iterrows():
            f.write(
                json.dumps(
                    {
                        "candidate_id": row["candidate_id"],
                        "pmid": row["pmid"],
                        "pair_type": row["pair_type"],
                        "label_civic_curated_positive": bool(row["label_civic_curated_positive"]),
                        "model_id": spec.model_id,
                        "score": float(row["score"]),
                    }
                )
                + "\n"
            )

    print(f"  mean P(present)={mean_probs.mean():.3f} std={mean_probs.std():.3f} -> {out_path}")
    return out


def score_all_models(candidates: pd.DataFrame | None = None, force: bool = False) -> pd.DataFrame:
    if candidates is None:
        candidates = load_primary_candidates()
    parts = []
    for spec in MODELS:
        parts.append(score_model(spec, candidates, force=force))
    return pd.concat(parts, ignore_index=True)


def load_all_scores() -> pd.DataFrame:
    parts = []
    for spec in MODELS:
        path = _score_path(spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing scores: {path}")
        parts.append(pd.read_json(path, lines=True))
    return pd.concat(parts, ignore_index=True)
