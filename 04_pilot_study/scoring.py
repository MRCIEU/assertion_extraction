"""Relation-presence probability from model logits."""

from __future__ import annotations

import numpy as np
import torch

from .config import ModelSpec


def present_probability(logits: torch.Tensor, spec: ModelSpec, id2label: dict) -> np.ndarray:
    """
    Map multi-class RE logits to a single relation-presence probability.
    Uniform rule: P(present) = 1 - P(explicit no-relation class) when available,
    else P(positive class) for binary Association models.
    """
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    labels = {int(k): str(v).lower() for k, v in id2label.items()}

    no_rel_idx = None
    for idx, name in labels.items():
        if name in {"no_relation", "none", "null", "o", "other", "0"}:
            no_rel_idx = idx
            break

    if spec.model_id == "distilbert_biored":
        assoc_idx = next((i for i, n in labels.items() if "association" in n), 1)
        return probs[:, assoc_idx]

    if spec.marker_style == "nli":
        # cross-encoder/nli-roberta-base: 0=contradiction, 1=entailment, 2=neutral
        entail_idx = next((i for i, n in labels.items() if "entail" in n), 1)
        return probs[:, entail_idx]

    if no_rel_idx is not None:
        return 1.0 - probs[:, no_rel_idx]

    # Fallback: sum all non-zero labels (SemEval numeric labels except 0)
    present = probs.copy()
    if 0 in labels:
        present[:, 0] = 0.0
    return present.sum(axis=1)
