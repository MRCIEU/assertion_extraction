"""
Cancer surface lexicon — recovered from compiled .pyc (2026-04-15).

The lexicon is a case-insensitive substring regex matching 25 cancer-related
surface forms. It was used for the keyword-based T2 oncology projection
(filtering BioRED/DrugProt/BC5CDR documents for oncology-facing content).

Source: Originally in oncology_projection/utils/lexicon.py; recovered from
Python 3.10 .pyc bytecode.

Precision on spot-check examples: 1.00 (13 true positives, 0 false positives)
Recall on spot-check examples: 0.92 (1 false negative: 'oncogene')
"""
from __future__ import annotations
import re

CANCER_REGEX = (
    r"cancer|carcinoma|carcinomas|tumor|tumour|tumors|tumours|oncolog|neoplasm|"
    r"melanoma|lymphoma|leukemia|leukaemia|sarcoma|glioma|nsclc|sclc|"
    r"adenocarcinoma|mesothelioma|myeloma|blastoma|hepatocellular|"
    r"cholangiocarcinoma|seminoma|medulloblastoma"
)

CANCER_TERMS = [t.strip() for t in CANCER_REGEX.split("|") if t.strip()]

_PATTERN = re.compile(CANCER_REGEX, re.IGNORECASE)


def is_cancer_like(text: str) -> bool:
    """Return True if text contains any cancer-related surface form."""
    return bool(_PATTERN.search(text))


KNOWN_GAPS = [
    {"term": "astrocytoma",   "note": "Not in lexicon; no covering prefix", "severity": "medium"},
    {"term": "meningioma",    "note": "Not in lexicon; common CNS tumour",   "severity": "low"},
    {"term": "hepatoma",      "note": "Simple 'hepatoma' missed; HCC via 'hepatocellular'", "severity": "low"},
    {"term": "oncogene",      "note": "No cancer surface form (false negative on spot-check)", "severity": "low"},
    {"term": "glioblastoma",  "note": "Covered via 'blastoma' suffix",       "severity": "none"},
    {"term": "retinoblastoma","note": "Covered via 'blastoma'",               "severity": "none"},
]

PAPER_DESCRIPTION = (
    "A case-insensitive surface regular expression matching 25 cancer-related "
    "strings (cancer, carcinoma, tumor/tumour, oncolog*, neoplasm, melanoma, "
    "lymphoma, leukemia/leukaemia, sarcoma, glioma, NSCLC, SCLC, adenocarcinoma, "
    "mesothelioma, myeloma, blastoma, hepatocellular, cholangiocarcinoma, "
    "seminoma, medulloblastoma) applied case-insensitively to disease entity "
    "text (BioRED, BC5CDR) or abstract text (DrugProt) to identify "
    "oncology-facing documents."
)
