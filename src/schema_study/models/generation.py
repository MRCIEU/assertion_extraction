"""Prompt generation, resume, and parsing.

Tests supply a scripted generator. The Qwen loader is not used unless the
execution permit allows it.
"""

from __future__ import annotations

from schema_study.models.schedule import assert_execution_allowed
from schema_study.predictions import parse_prediction_text
from schema_study.schemas import Schema

STUDY_GENERATION = {
    "temperature": 0,
    "top_p": 1,
    "max_new_tokens": 256,
    "enable_thinking": False,
}


class UnsupportedGenerationSetting(ValueError):
    pass


def check_generation_settings(requested: dict, supported: set[str] | None = None) -> dict:
    allowed = supported or set(STUDY_GENERATION)
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise UnsupportedGenerationSetting(
            f"Unsupported settings are not silently replaced: {unknown}"
        )
    if requested.get("enable_thinking") not in (None, False):
        raise UnsupportedGenerationSetting("enable_thinking must stay false")
    merged = dict(STUDY_GENERATION)
    merged.update(requested)
    return merged


def render_prompt(candidate_id: str, label_block: str, text: str) -> str:
    return (
        "Return only a JSON object with candidate_id and labels.\n"
        f"Candidate id: {candidate_id}\n"
        f"Positive labels:\n{label_block}\n"
        f"Text:\n{text}\n"
    )


def completed_ids(ledger: list[dict]) -> set[str]:
    return {row["request_id"] for row in ledger if row.get("status") == "completed"}


def generate_requests(
    requests: list[dict],
    generator,
    schema: Schema,
    level: str,
    ledger: list[dict] | None = None,
) -> dict:
    """Parse scripted or future model answers. Completed request ids are skipped."""
    done = completed_ids(ledger or [])
    skipped = sum(1 for request in requests if request["request_id"] in done)
    attempts = []
    adopted = []
    for request in requests:
        request_id = request["request_id"]
        if request_id in done:
            continue
        answer = generator.generate(request["prompt"])
        parsed = parse_prediction_text(
            answer,
            expected_candidate_id=request["candidate_id"],
            schema=schema,
            level=level,
        )
        record = {
            "request_id": request_id,
            "candidate_id": request["candidate_id"],
            "status": "completed",
            "output_status": parsed["output_status"],
            "predicted_labels": parsed["predicted_labels"],
            "illegal_labels": parsed["illegal_labels"],
            "raw_answer": answer,
        }
        attempts.append(record)
        adopted.append(record)
        done.add(request_id)
    return {"attempts": attempts, "adopted": adopted, "skipped_completed": skipped}


def execute_qwen_generation(config: dict, *, requested: bool) -> dict:
    assert_execution_allowed(requested, "qwen generation")
    from schema_study.models.qwen_backend import load_qwen

    return load_qwen(config["model_id"], config["revision"], config["settings"])
