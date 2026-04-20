# -*- coding: utf-8 -*-
"""
Load project fine-tuned HuggingFace sequence-classification checkpoints and run
batched predictions. Lives only under knowledge_grounded_evidence_audit.

Checkpoint dict schema:
  - model_name, label2id, model_state_dict

All comments in English.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model_from_checkpoint(
    checkpoint_path: Path | str,
    device: torch.device,
    *,
    state_dict_strict: bool = True,
) -> Tuple[Any, Any, Dict[str, int], List[str], Path]:
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    required = ("model_name", "label2id", "model_state_dict")
    for k in required:
        if k not in ckpt:
            raise KeyError(f"Checkpoint missing key {k!r}: {checkpoint_path}")

    model_name = ckpt["model_name"]
    label2id: Dict[str, int] = dict(ckpt["label2id"])
    num_labels = len(label2id)
    if num_labels < 2:
        raise ValueError(f"Invalid label2id size {num_labels}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    incompatible = model.load_state_dict(ckpt["model_state_dict"], strict=state_dict_strict)
    if state_dict_strict:
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Strict load failed for {checkpoint_path}: "
                f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
            )

    model.to(device)
    model.eval()

    by_id = {v: k for k, v in label2id.items()}
    labels_ordered = [by_id[i] for i in range(num_labels)]

    return model, tokenizer, label2id, labels_ordered, checkpoint_path


@torch.no_grad()
def predict_labels(
    model: Any,
    tokenizer: Any,
    label2id: Dict[str, int],
    rows: Sequence[Dict[str, Any]],
    *,
    device: torch.device,
    max_length: int = 384,
    batch_size: int = 8,
) -> Tuple[List[str], List[float]]:
    id2label = {v: k for k, v in label2id.items()}
    texts = [r["text"] for r in rows]
    preds: List[str] = []
    confs: List[float] = []

    model.eval()
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        prob = torch.softmax(logits, dim=-1)
        conf, pred_idx = prob.max(dim=-1)
        for j in range(len(batch_texts)):
            pid = int(pred_idx[j].item())
            preds.append(id2label[pid])
            confs.append(float(conf[j].item()))

    return preds, confs
