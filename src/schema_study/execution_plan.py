"""Formal execution order. This file does not authorize any run."""

from __future__ import annotations

from schema_study.experiments import plan_experiments

BIORED_PREDICTION_TARGETS = (
    ("biored", "test"),
    ("bc8", "bc8"),
)
DRUGPROT_PREDICTION_TARGETS = (("drugprot", "eval"),)


def execution_rows(controls: list[dict] | None = None) -> list[dict]:
    """Train, then best-checkpoint prediction, then rollup, then scoring.

    Rollup rows point at a completed P4 prediction, not only at a finished
    training run. Formal rows stay unauthorized.
    """
    planned = plan_experiments(controls)
    trains = [row for row in planned if row["route"] == "direct_train"]
    rows = []
    predictions = []
    for train in trains:
        train_row = _stage(train, "train", source_id="", target_dataset=train["dataset"], target_split="train")
        train_row["requires_model_execution"] = True
        rows.append(train_row)
        targets = BIORED_PREDICTION_TARGETS if train["dataset"] == "biored" else DRUGPROT_PREDICTION_TARGETS
        for dataset, split in targets:
            prediction = _stage(
                train,
                "predict",
                source_id=train_row["execution_id"],
                target_dataset=dataset,
                target_split=split,
            )
            prediction["requires_model_execution"] = True
            prediction["checkpoint"] = "best"
            predictions.append(prediction)
            rows.append(prediction)
            if train["predicate_level"] == "P4":
                for level in ("P3", "P2", "P1"):
                    rollup = _stage(
                        train,
                        "rollup",
                        source_id=prediction["execution_id"],
                        target_dataset=dataset,
                        target_split=split,
                        predicate_level=level,
                    )
                    rollup["requires_model_execution"] = False
                    rollup["source_stage"] = "prediction"
                    rows.append(rollup)
                    rows.append(_score(rollup))
            rows.append(_score(prediction))
    return rows


def _stage(train: dict, stage: str, source_id: str, target_dataset: str, target_split: str, predicate_level: str | None = None) -> dict:
    level = predicate_level or train["predicate_level"]
    execution_id = f"{stage}|{train['plan_id']}|{target_dataset}|{target_split}|{level}"
    return {
        "execution_id": execution_id,
        "stage": stage,
        "source_id": source_id,
        "source_stage": "train" if stage == "predict" else "",
        "train_plan_id": train["plan_id"],
        "dataset": train["dataset"],
        "entity_level": train["entity_level"],
        "predicate_level": level,
        "model_id": train["model_id"],
        "seed": train["seed"],
        "checkpoint": "best" if stage != "train" else "",
        "target_dataset": target_dataset,
        "target_split": target_split,
        "output_path": f"studies/schema_granularity_v1/runs/formal/{execution_id}",
        "integrity_check": "candidate_ids_match_scoring_population_and_gold_comes_from_candidate_table",
        "downstream": "rollup_then_score" if stage == "predict" and train["predicate_level"] == "P4" else "score",
        "requires_model_execution": stage in {"train", "predict"},
        "execution_status": "planned",
        "execution_authorized": False,
        "study_phase": "formal",
    }


def _score(source: dict) -> dict:
    row = dict(source)
    row["execution_id"] = "score|" + source["execution_id"]
    row["stage"] = "score"
    row["source_id"] = source["execution_id"]
    row["source_stage"] = source["stage"]
    row["requires_model_execution"] = False
    row["checkpoint"] = source.get("checkpoint", "")
    row["output_path"] = source["output_path"] + "/evaluation"
    row["downstream"] = ""
    row["execution_authorized"] = False
    row["study_phase"] = "formal"
    row["execution_status"] = "planned"
    return row
