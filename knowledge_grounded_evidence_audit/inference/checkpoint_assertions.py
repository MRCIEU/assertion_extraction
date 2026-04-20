# -*- coding: utf-8 -*-
"""
Build trainer-format pair strings and run checkpoint predictions for kg audit.

Pair format matches project external eval: ``head [ENT] tail [SEP] sentence``.

All comments in English.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Set, Tuple

import torch

from inference.predict_checkpoint import predict_labels

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _trainer_text(head: str, tail: str, sentence: str) -> str:
    return f"{head.strip()} [ENT] {tail.strip()} [SEP] {sentence.strip()}"


def _label_to_relation_family(pred: str) -> str:
    if pred == "DRUG_GENE_REGULATION":
        return "drug_gene"
    if pred == "DRUG_DISEASE":
        return "drug_disease"
    if pred == "VARIANT_GENE":
        return "variant_disease"
    if pred == "ASSOCIATION_GENERAL":
        return "gene_disease"
    if pred == "__NEGATIVE__":
        return "negative"
    return "unknown"


def extract_neural_pair_assertions(
    pmid: str,
    title: str,
    abstract: str,
    genes: Set[str],
    drugs: Set[str],
    model_id: str,
    model: Any,
    tokenizer: Any,
    label2id: Dict[str, int],
    device: torch.device,
    *,
    max_pairs_per_document: int = 96,
    batch_size: int = 8,
    max_length: int = 384,
) -> List[Dict[str, Any]]:
    """
    Returns dict rows with keys matching RawAssertion needs (caller builds dataclass).
    Skips predicted __NEGATIVE__.
    """
    text = f"{title.strip()}. {abstract}".strip()
    sentences = [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) > 20]

    pair_specs: List[Tuple[str, str, str, str, str]] = []
    for sent in sentences:
        if len(pair_specs) >= max_pairs_per_document:
            break
        for g in genes:
            if not re.search(rf"\b{re.escape(g)}\b", sent, re.I):
                continue
            for d in drugs:
                if len(d) < 4:
                    continue
                if not re.search(rf"\b{re.escape(d)}\b", sent, re.I):
                    continue
                pair_specs.append((g, d, sent, "GENE", "DRUG"))
                if len(pair_specs) >= max_pairs_per_document:
                    break
            if len(pair_specs) >= max_pairs_per_document:
                break
    if not pair_specs:
        return []

    rows_in = [{"text": _trainer_text(g, d, s)} for g, d, s, _, _ in pair_specs]
    preds, confs = predict_labels(
        model,
        tokenizer,
        label2id,
        rows_in,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )

    out: List[Dict[str, Any]] = []
    for (g, d, sent, _ht, _tt), pred, conf in zip(pair_specs, preds, confs):
        if pred == "__NEGATIVE__":
            continue
        fam = _label_to_relation_family(pred)
        aid = hashlib.sha1(
            f"{pmid}|{model_id}|{g}|{d}|{pred}|{sent[:80]}".encode()
        ).hexdigest()[:16]
        out.append(
            {
                "assertion_id": aid,
                "model_id": model_id,
                "doc_pmid": pmid,
                "sentence": sent[:800],
                "relation_family": fam,
                "entity_a": {"type": "gene", "text": g, "normalized": g},
                "entity_b": {"type": "drug", "text": d, "normalized": d},
                "confidence": round(float(conf), 4),
                "provenance": [
                    "checkpoint_inference",
                    f"predicted_label:{pred}",
                ],
                "mapped_label_s2": pred,
            }
        )
    return out
