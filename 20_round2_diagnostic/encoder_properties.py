"""Documented encoder properties for exploratory correlation (Part 2)."""

from __future__ import annotations

from dataclasses import dataclass

from shared.models import MODELS


@dataclass(frozen=True)
class EncoderProperty:
    model_id: str
    short_name: str
    params_millions: float
    biomedical_pretrain: bool
    property_source: str


# Parameter counts: HuggingFace model cards (BERT-family ~110M, RoBERTa-base 125M,
# DistilBERT 66M, DeBERTa-base ~100M). Biomedical flag: PubMed/scientific domain
# pretraining vs general-domain pretraining per model card / publication.
ENCODER_PROPERTIES: list[EncoderProperty] = [
    EncoderProperty(
        "pubmedbert_base",
        "PubMedBERT-base",
        110.0,
        True,
        "HF microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract; PubMed abstracts",
    ),
    EncoderProperty(
        "biomedbert_base",
        "BioMedBERT-base",
        110.0,
        True,
        "HF microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract; PubMed-derived",
    ),
    EncoderProperty(
        "biolinkbert_base",
        "BioLinkBERT-base",
        110.0,
        True,
        "HF michiyasunaga/BioLinkBERT-base; PubMed + link prediction",
    ),
    EncoderProperty(
        "biobert_base",
        "BioBERT-base",
        110.0,
        True,
        "HF dmis-lab/biobert-base-cased-v1.2; PubMed + PMC",
    ),
    EncoderProperty(
        "scibert_base",
        "SciBERT",
        110.0,
        True,
        "HF allenai/scibert_scivocab_uncased; Semantic Scholar biomedical+scientific",
    ),
    EncoderProperty(
        "roberta_base",
        "RoBERTa-base",
        125.0,
        False,
        "HF roberta-base; BookCorpus + English Wikipedia",
    ),
    EncoderProperty(
        "bert_base",
        "BERT-base",
        110.0,
        False,
        "HF bert-base-uncased; BookCorpus + English Wikipedia",
    ),
    EncoderProperty(
        "distilbert_base",
        "DistilBERT-base",
        66.0,
        False,
        "HF distilbert-base-uncased; distilled from general BERT",
    ),
    EncoderProperty(
        "deberta_base",
        "DeBERTa-base",
        100.0,
        False,
        "HF microsoft/deberta-base; general English NLU pretraining",
    ),
]

PROPERTIES_BY_ID = {p.model_id: p for p in ENCODER_PROPERTIES}


def properties_dataframe():
    import pandas as pd

    rows = []
    for p in ENCODER_PROPERTIES:
        rows.append(
            {
                "model_id": p.model_id,
                "short_name": p.short_name,
                "params_millions": p.params_millions,
                "biomedical_pretrain": int(p.biomedical_pretrain),
                "property_source": p.property_source,
            }
        )
    return pd.DataFrame(rows)
