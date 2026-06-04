"""Encoder definitions for sweep and full-matrix training."""

from __future__ import annotations

from dataclasses import dataclass

SWEEP_LEARNING_RATES = (5e-6, 1e-5, 2e-5, 3e-5)
SWEEP_WARMUP_SETTINGS: tuple[tuple[str, float], ...] = (
    ("none", 0.0),
    ("warmup_10pct", 0.10),
)
SWEEP_SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    short_name: str
    hf_name: str
    architecture: str


MODELS: list[ModelSpec] = [
    ModelSpec("pubmedbert_base", "PubMedBERT-base", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", "domain BERT"),
    ModelSpec("biomedbert_base", "BioMedBERT-base", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", "domain BERT"),
    ModelSpec("biolinkbert_base", "BioLinkBERT-base", "michiyasunaga/BioLinkBERT-base", "domain BERT"),
    ModelSpec("biobert_base", "BioBERT-base", "dmis-lab/biobert-base-cased-v1.2", "domain BERT"),
    ModelSpec("scibert_base", "SciBERT", "allenai/scibert_scivocab_uncased", "domain BERT"),
    ModelSpec("roberta_base", "RoBERTa-base", "roberta-base", "general RoBERTa"),
    ModelSpec("bert_base", "BERT-base", "bert-base-uncased", "general BERT"),
    ModelSpec("distilbert_base", "DistilBERT-base", "distilbert-base-uncased", "lightweight"),
    ModelSpec("deberta_base", "DeBERTa-base", "microsoft/deberta-base", "general DeBERTa"),
]

MODEL_BY_ID = {m.model_id: m for m in MODELS}
SWEEP_MODEL_IDS = ("pubmedbert_base", "roberta_base", "distilbert_base", "deberta_base")
SWEEP_MODELS = [MODEL_BY_ID[m] for m in SWEEP_MODEL_IDS]
