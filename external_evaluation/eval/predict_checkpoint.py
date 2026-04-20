# -*- coding: utf-8 -*-
"""Load checkpoints and run sequence-classification inference."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging as tf_logging

logger = logging.getLogger(__name__)


def _quiet_hf_init() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    tf_logging.set_verbosity_error()


def load_model_from_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    *,
    state_dict_strict: bool = True,
) -> tuple[Any, Any, dict[str, int], list[str], dict[str, Any]]:
    """
    Load fine-tuned ``AutoModelForSequenceClassification`` from ``best.pt`` / ``last.pt``.

    Always applies ``load_state_dict(..., strict=False)`` first so missing/unexpected keys are
    recorded; if ``state_dict_strict`` is True and any keys are missing or unexpected, raises
    ``RuntimeError`` (no silent drop of classifier weights).
    """
    _quiet_hf_init()
    ckpt_path = Path(ckpt_path)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in ("label2id", "model_state_dict", "model_name"):
        if k not in blob:
            raise KeyError(f"Checkpoint {ckpt_path} missing {k!r}")

    label2id: dict[str, int] = dict(blob["label2id"])
    id2label = {v: k for k, v in label2id.items()}
    labels = [id2label[i] for i in range(len(id2label))]
    model_name = str(blob["model_name"] or "")
    if not model_name:
        raise ValueError("Checkpoint model_name is empty")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    incompatible = model.load_state_dict(blob["model_state_dict"], strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    load_report: dict[str, Any] = {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "strict_enforced": state_dict_strict,
        "classifier_weight_norm": float(model.classifier.weight.detach().float().norm().item()),
        "num_labels_model": model.config.num_labels,
        "num_labels_checkpoint": len(label2id),
    }
    if missing or unexpected:
        msg = f"state_dict mismatch for {ckpt_path}: missing={missing!r} unexpected={unexpected!r}"
        if state_dict_strict:
            raise RuntimeError(msg)
        logger.warning(msg)

    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer, label2id, labels, load_report


def predict_labels(
    model,
    tokenizer,
    label2id: dict[str, int],
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    max_length: int = 384,
    batch_size: int = 8,
) -> tuple[list[str], list[float]]:
    id2label = {i: lab for lab, i in label2id.items()}
    texts = [r["text"] for r in rows]
    preds: list[str] = []
    confs: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = tokenizer(
                chunk,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            prob = torch.softmax(logits, dim=-1)
            conf, idx = prob.max(dim=-1)
            for j in range(len(chunk)):
                preds.append(id2label[int(idx[j].item())])
                confs.append(float(conf[j].item()))
    return preds, confs


def filter_rows_for_vocab(rows: list[dict[str, Any]], label2id: dict[str, int]) -> tuple[list[dict[str, Any]], int]:
    kept, skipped = [], 0
    for r in rows:
        if r["label"] not in label2id:
            skipped += 1
            continue
        kept.append(r)
    return kept, skipped
