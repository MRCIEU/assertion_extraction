"""Score frozen CIViC pool per checkpoint (one seed at a time)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import INFER_BATCH_SIZE, MAX_SEQ_LENGTH, SCORES_DIR, ModelSpec
from .input_format import format_eval_input
from .pool_loader import load_primary_candidates
from .train import checkpoint_path


def _require_gpu() -> torch.device:
    if not torch.cuda.is_available():
        print("ERROR: GPU required for inference.", file=sys.stderr)
        sys.exit(1)
    return torch.device("cuda")


def score_path(spec: ModelSpec, seed: int) -> Path:
    return SCORES_DIR / spec.model_id / f"seed_{seed}.jsonl"


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


def score_checkpoint(
    spec: ModelSpec,
    seed: int,
    candidates: pd.DataFrame | None = None,
    force: bool = False,
) -> pd.DataFrame:
    out_path = score_path(spec, seed)
    if out_path.exists() and not force:
        return pd.read_json(out_path, lines=True)

    if candidates is None:
        candidates = load_primary_candidates()

    ckpt = checkpoint_path(spec, seed)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint {ckpt}")

    device = _require_gpu()
    texts = [format_eval_input(row) for _, row in candidates.iterrows()]
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    model.to(device)
    probs = _predict_probs(model, tokenizer, texts, device)

    out = candidates.copy()
    out["model_id"] = spec.model_id
    out["seed"] = seed
    out["run_id"] = f"{spec.model_id}_seed_{seed}"
    out["score"] = probs

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
                        "seed": int(seed),
                        "run_id": row["run_id"],
                        "score": float(row["score"]),
                    }
                )
                + "\n"
            )

    print(
        f"  scored {spec.model_id} seed={seed}: mean P(present)={np.mean(probs):.3f} -> {out_path}"
    )
    return out


def load_all_scores() -> pd.DataFrame:
    from .config import MODELS, TRAIN_SEEDS

    parts = []
    for spec in MODELS:
        for seed in TRAIN_SEEDS:
            path = score_path(spec, seed)
            if path.exists():
                parts.append(pd.read_json(path, lines=True))
    if not parts:
        raise FileNotFoundError("No score files found; run scoring first.")
    return pd.concat(parts, ignore_index=True)
