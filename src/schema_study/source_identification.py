"""Checks for a later source-span identifier. This module does not call a model."""

from __future__ import annotations

LOCALISATION_MODEL = {
    "model_id": "Qwen/Qwen3-8B",
    "revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "task": "return verbatim source spans or abstain",
    "status": "specified_not_executed",
    "gpu_authorised_here": False,
}


def recover_quotation(document: str, quotation: str | None) -> dict:
    """Accept a quotation only when it occurs in the original document."""
    if quotation is None or quotation == "":
        return {"status": "abstained", "char_start": None, "char_end": None}
    start = document.find(quotation)
    if start < 0:
        return {"status": "rejected_not_in_source", "char_start": None, "char_end": None}
    return {"status": "recovered", "char_start": start, "char_end": start + len(quotation)}
