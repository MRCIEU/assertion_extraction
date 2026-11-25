"""Source finding requests. This module does not call a model.

A repeated quotation is not resolved by taking the first match. A model
response is accepted only when every quotation occurs at an identified span.
"""

from __future__ import annotations

import hashlib
import json

LOCALISATION_MODEL = {
    "model_id": "Qwen/Qwen3-8B",
    "revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "task": "return verbatim source spans or abstain",
    "status": "specified_not_executed",
    "gpu_authorised_here": False,
}
INSTRUCTION_VERSION = "source_finding_instruction_v1"
TASK_INSTRUCTION = (
    "Identify every passage that states the target proposition about the named entities. "
    "Return the verbatim quotation and the character offsets in the supplied document. "
    "If the same words occur more than once, return each occurrence that states the proposition, "
    "with its offsets. Do not choose the first copy by default. "
    "A finding may be discontinuous or may cross a sentence boundary. "
    "If the document does not state the proposition, abstain. "
    "Do not infer a relation that the text does not report."
)


def recover_quotation(document: str, quotation: str | None, occurrence: int | None = None) -> dict:
    """Accept a quotation only at an identified occurrence."""
    if quotation is None or quotation == "":
        return {"status": "abstained", "char_start": None, "char_end": None, "n_occurrences": 0}
    starts = []
    cursor = 0
    while True:
        found = document.find(quotation, cursor)
        if found < 0:
            break
        starts.append(found)
        cursor = found + 1
    if not starts:
        return {"status": "rejected_not_in_source", "char_start": None, "char_end": None, "n_occurrences": 0}
    if len(starts) > 1 and occurrence is None:
        return {
            "status": "ambiguous_repeated_quotation",
            "char_start": None,
            "char_end": None,
            "n_occurrences": len(starts),
            "occurrences": starts,
        }
    chosen = starts[0] if occurrence is None else starts[occurrence] if 0 <= occurrence < len(starts) else None
    if chosen is None:
        return {"status": "rejected_occurrence", "char_start": None, "char_end": None, "n_occurrences": len(starts)}
    return {
        "status": "recovered",
        "char_start": chosen,
        "char_end": chosen + len(quotation),
        "n_occurrences": len(starts),
        "occurrence": 0 if occurrence is None else occurrence,
    }


def finding_identifier(document_id: str, candidate_id: str, proposition: str, spans: list) -> str:
    payload = json.dumps(
        {"document_id": document_id, "candidate_id": candidate_id, "proposition": proposition, "spans": spans},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_request(
    *,
    document_id: str,
    document_text: str,
    candidate_id: str,
    argument_1: str,
    argument_2: str,
    proposition: str | None,
    mode: str,
) -> dict:
    """One localisation request. Relation-independent mode does not include a proposition."""
    if mode not in {"relation_conditioned", "relation_independent"}:
        raise ValueError(mode)
    return {
        "instruction_version": INSTRUCTION_VERSION,
        "instruction": TASK_INSTRUCTION,
        "model": LOCALISATION_MODEL,
        "mode": mode,
        "document_id": document_id,
        "candidate_id": candidate_id,
        "argument_1": argument_1,
        "argument_2": argument_2,
        "proposition": proposition if mode == "relation_conditioned" else None,
        "document_characters": len(document_text),
        "executed": False,
    }


def validate_model_response(document: str, response: dict) -> dict:
    """Check a structured response. Unknown statuses and unverified quotations are rejected."""
    status = response.get("status")
    if status in {"abstained", "unassessable"}:
        return {"status": status, "findings": [], "accepted": True}
    if status != "findings":
        return {"status": "invalid_output", "accepted": False, "reason": "unknown_status"}
    findings = []
    for item in response.get("findings") or []:
        spans = item.get("spans") or []
        if not spans:
            return {"status": "invalid_output", "accepted": False, "reason": "finding_without_spans"}
        checked = []
        for span in spans:
            if len(span) != 2 or not all(isinstance(value, int) for value in span):
                return {"status": "invalid_output", "accepted": False, "reason": "span_not_a_pair"}
            start, end = span
            if start < 0 or end > len(document) or start >= end:
                return {"status": "invalid_output", "accepted": False, "reason": "span_outside_document"}
            quotation = item.get("quotation")
            if quotation is not None and document[start:end] != quotation and quotation not in document[start:end]:
                return {"status": "invalid_output", "accepted": False, "reason": "quotation_does_not_match_span"}
            checked.append([start, end])
        findings.append({"spans": checked, "cross_sentence": bool(item.get("cross_sentence"))})
    return {"status": "findings", "findings": findings, "accepted": True, "n_findings": len(findings)}
