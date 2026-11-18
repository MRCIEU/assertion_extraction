"""Formal scores read gold only from the candidate table.

A prediction file may contain a gold field. That field is ignored. Missing
predictions stay in the scored population.
"""

from __future__ import annotations

from collections import Counter

from schema_study.evaluation import candidate_label_counts, summarize_counts
from schema_study.schemas import Schema, map_fine_ids

SCORING_SPLITS = {
    "biored": {"train", "dev", "test"},
    "bc8": {"bc8"},
    "drugprot": {"train", "dev", "eval"},
}


class FormalScoreError(ValueError):
    """The prediction table cannot be scored against this candidate table."""


def fine_ids_for_names(schema: Schema, names: list[str]) -> list[str]:
    by_name = {display: stable for stable, display in schema.fine_names.items()}
    unknown = [name for name in names if name not in by_name and name not in schema.fine_ids]
    if unknown:
        raise FormalScoreError(f"Gold names are outside the fine schema: {unknown}")
    resolved = []
    for name in names:
        resolved.append(name if name in schema.fine_ids else by_name[name])
    return resolved


def gold_groups(schema: Schema, gold_names: list[str], level: str) -> list[str]:
    """Map fine gold names onto one predicate level. Coarse duplicates collapse."""
    if not gold_names:
        return []
    fine_ids = fine_ids_for_names(schema, list(gold_names))
    order = [label.stable_id for label in schema.level_labels(level)]
    mapped = map_fine_ids(schema, fine_ids, level)
    return [label_id for label_id in order if label_id in mapped]


def scoring_population(candidates: list[dict], *, allowed_splits: set[str]) -> list[dict]:
    """Return included candidates in the requested splits.

    Duplicate candidate ids are a hard error. Unlabeled candidates are absent
    when their split is not allowed.
    """
    seen = Counter(row["candidate_id"] for row in candidates)
    duplicated = sorted(candidate_id for candidate_id, count in seen.items() if count > 1)
    if duplicated:
        raise FormalScoreError(f"Candidate table has duplicate candidate ids: {duplicated[:5]}")
    selected = []
    for row in candidates:
        if row["split"] not in allowed_splits:
            continue
        if not row.get("included_in_scoring", True):
            continue
        selected.append(row)
    return selected


def score_predictions(
    candidates: list[dict],
    predictions: list[dict],
    schema: Schema,
    level: str,
    *,
    allowed_splits: set[str],
    data_version: str,
    input_version: str,
    prediction_version: str,
    mapping_version: str,
) -> dict:
    """Score one prediction table. Gold inside the prediction rows is not read."""
    if not data_version:
        raise FormalScoreError("Formal scoring requires a candidate data version.")
    population = scoring_population(candidates, allowed_splits=allowed_splits)
    by_id = {row["candidate_id"]: row for row in population}
    predicted_ids = [row.get("candidate_id") for row in predictions]
    if any(not candidate_id for candidate_id in predicted_ids):
        raise FormalScoreError("A prediction row has no candidate id.")
    duplicate_predictions = sorted(
        candidate_id for candidate_id, count in Counter(predicted_ids).items() if count > 1
    )
    if duplicate_predictions:
        raise FormalScoreError(f"Prediction table repeats candidate ids: {duplicate_predictions[:5]}")
    extra = sorted(set(predicted_ids) - set(by_id))
    if extra:
        raise FormalScoreError(f"Predictions contain candidate ids outside the scoring population: {extra[:5]}")
    prediction_by_id = {row["candidate_id"]: row for row in predictions}
    scored = []
    for candidate_id, candidate in by_id.items():
        prediction = prediction_by_id.get(candidate_id)
        if prediction is None:
            status = "missing"
            predicted = []
            illegal = []
        else:
            if prediction.get("document_id") not in (None, "", candidate["document_id"]):
                raise FormalScoreError(f"{candidate_id} document does not match the candidate table.")
            if prediction.get("split") not in (None, "", candidate["split"]):
                raise FormalScoreError(f"{candidate_id} split does not match the candidate table.")
            status = prediction["output_status"]
            predicted = list(prediction.get("predicted_labels") or [])
            illegal = list(prediction.get("illegal_labels") or [])
        gold = gold_groups(schema, list(candidate.get("gold_labels") or []), level)
        counts = candidate_label_counts(gold, predicted, status, illegal)
        scored.append(
            {
                "candidate_id": candidate_id,
                "document_id": candidate["document_id"],
                "split": candidate["split"],
                "gold": gold,
                "predicted": predicted if status == "valid" or status == "partial_illegal" else [],
                "status": status,
                "illegal_labels": illegal,
                **counts,
            }
        )
    labels = [label.stable_id for label in schema.level_labels(level)]
    summary = summarize_counts(scored, labels)
    summary.update(
        {
            "data_version": data_version,
            "input_version": input_version,
            "prediction_version": prediction_version,
            "mapping_version": mapping_version,
            "predicate_level": level,
            "allowed_splits": sorted(allowed_splits),
            "n_missing_predictions": sum(1 for row in scored if row["status"] == "missing"),
            "gold_source": "candidate_table",
            "prediction_gold_fields_ignored": True,
            "synthetic_predictions": False,
        }
    )
    return {"rows": scored, "summary": summary}
