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
    ModelSpec("bluebert_base", "BlueBERT-base", "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12", "domain BERT"),
    ModelSpec("biolinkbert_base", "BioLinkBERT-base", "michiyasunaga/BioLinkBERT-base", "domain BERT"),
    ModelSpec("biobert_base", "BioBERT-base", "dmis-lab/biobert-base-cased-v1.2", "domain BERT"),
    ModelSpec("scibert_base", "SciBERT", "allenai/scibert_scivocab_uncased", "domain BERT"),
    ModelSpec("roberta_base", "RoBERTa-base", "roberta-base", "general RoBERTa"),
    ModelSpec("bert_base", "BERT-base", "bert-base-uncased", "general BERT"),
    ModelSpec("distilbert_base", "DistilBERT-base", "distilbert-base-uncased", "lightweight"),
    ModelSpec("deberta_base", "DeBERTa-base", "microsoft/deberta-base", "general DeBERTa"),
]

MODEL_BY_ID = {m.model_id: m for m in MODELS}
KNOWN_MODEL_IDS = frozenset(MODEL_BY_ID)
SWEEP_MODEL_IDS = ("pubmedbert_base", "roberta_base", "distilbert_base", "deberta_base")
SWEEP_MODELS = [MODEL_BY_ID[m] for m in SWEEP_MODEL_IDS]

# Legacy on-disk / cache keys. Only aliases listed here are accepted; unknown ids raise.
_CHECKPOINT_MODEL_ALIASES = {"biomedbert_base": "bluebert_base"}


def require_model_id(model_id: str) -> str:
    """Resolve a model_id and fail loudly on anything outside the nine-encoder set.

    ``biomedbert_base`` is accepted only as a legacy alias for ``bluebert_base``.
    Returning the input unchanged for unknown ids is forbidden — that path silently
    dropped BlueBERT from AUROC / B3.1 when cache keys and score dirs disagreed.
    """
    resolved = _CHECKPOINT_MODEL_ALIASES.get(model_id, model_id)
    if resolved not in KNOWN_MODEL_IDS:
        raise ValueError(
            f"Unknown model_id {model_id!r} (resolved to {resolved!r}); "
            f"known={sorted(KNOWN_MODEL_IDS)}"
        )
    return resolved
