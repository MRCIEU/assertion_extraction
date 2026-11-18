"""Task scores, information retention, polarity, and document bootstrap.

Seed metrics are completed separately and then averaged. Predictions are not
pooled across seeds before the score is calculated.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from schema_study.schemas import Schema, group_direction, map_fine_ids

USABLE_STATUSES = {"valid", "partial_illegal"}
BOOTSTRAP_DRAWS = 2000
# Round 2 sets the formal bootstrap seed to 100. Review sampling stays on 20261115.
BOOTSTRAP_SEED = 100
PREVIOUS_BOOTSTRAP_SEED = 20261115


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def f1_from_counts(tp: float, fp: float, fn: float) -> float:
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 0.0
    return (2 * tp) / denominator


def candidate_label_counts(
    gold: Iterable[str],
    predicted: Iterable[str],
    status: str,
    illegal_labels: Iterable[str] | None = None,
) -> dict:
    """TP/FP/FN for one candidate, including the separate invalid-output FP."""
    gold_set = set(gold)
    illegal = list(dict.fromkeys(illegal_labels or []))
    if status == "valid":
        pred_set = set(predicted)
        extra_fp = 0
    elif status == "partial_illegal":
        pred_set = set(predicted)
        extra_fp = len(illegal)
    elif status == "rejected":
        pred_set = set()
        extra_fp = 1
    elif status == "missing":
        pred_set = set()
        extra_fp = 0
    else:
        raise ValueError(f"Unknown output status {status}")
    class_tp = gold_set & pred_set
    class_fp = pred_set - gold_set
    class_fn = gold_set - pred_set
    return {
        "tp": len(class_tp),
        "fp": len(class_fp) + extra_fp,
        "fn": len(class_fn),
        "class_tp": class_tp,
        "class_fp": class_fp,
        "class_fn": class_fn,
        "extra_fp": extra_fp,
        "exact": status == "valid" and pred_set == gold_set,
        "usable": status in USABLE_STATUSES,
        "failed": status in {"missing", "rejected"},
    }


def summarize_counts(rows: Sequence[Mapping], positive_labels: Sequence[str]) -> dict:
    tp = fp = fn = extra_fp = exact = usable = failed = 0
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)
    for row in rows:
        counted = candidate_label_counts(
            row.get("gold", ()),
            row.get("predicted", ()),
            row["status"],
            row.get("illegal_labels", ()),
        )
        tp += counted["tp"]
        fp += counted["fp"]
        fn += counted["fn"]
        extra_fp += counted["extra_fp"]
        exact += int(counted["exact"])
        usable += int(counted["usable"])
        failed += int(counted["failed"])
        for label in counted["class_tp"]:
            class_tp[label] += 1
        for label in counted["class_fp"]:
            class_fp[label] += 1
        for label in counted["class_fn"]:
            class_fn[label] += 1
    n = len(rows)
    per_class = []
    f1_values = []
    for label in positive_labels:
        one_tp = class_tp[label]
        one_fp = class_fp[label]
        one_fn = class_fn[label]
        one_f1 = f1_from_counts(one_tp, one_fp, one_fn)
        f1_values.append(one_f1)
        per_class.append(
            {
                "label": label,
                "tp": one_tp,
                "fp": one_fp,
                "fn": one_fn,
                "support": one_tp + one_fn,
                "precision": ratio(one_tp, one_tp + one_fp),
                "recall": ratio(one_tp, one_tp + one_fn),
                "f1": one_f1,
                "precision_status": "defined" if one_tp + one_fp else "undefined",
                "recall_status": "defined" if one_tp + one_fn else "undefined",
            }
        )
    return {
        "n_candidates": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "extra_fp": extra_fp,
        "class_fp_sum": sum(item["fp"] for item in per_class),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": f1_from_counts(tp, fp, fn),
        "precision_status": "defined" if tp + fp else "undefined",
        "recall_status": "defined" if tp + fn else "undefined",
        "macro_f1": (sum(f1_values) / len(f1_values)) if f1_values else 0.0,
        "exact_match": (exact / n) if n else None,
        "output_coverage": (usable / n) if n else None,
        "failure_count": failed,
        "per_class": per_class,
    }


def average_seed_metrics(per_seed: Sequence[Mapping]) -> dict:
    """Arithmetic mean of already computed seed metrics.

    This is intentionally different from scoring the pooled predictions.
    A null precision or recall on any seed makes that mean null.
    """
    if not per_seed:
        raise ValueError("No seed metrics to average")
    numeric = ("f1", "macro_f1", "exact_match", "output_coverage", "tp", "fp", "fn", "failure_count")
    nullable = ("precision", "recall")
    averaged = {"n_seeds": len(per_seed), "aggregation": "mean_of_seed_metrics"}
    for key in numeric:
        averaged[key] = sum(item[key] for item in per_seed) / len(per_seed)
    for key in nullable:
        values = [item[key] for item in per_seed]
        averaged[key] = None if any(value is None for value in values) else sum(values) / len(values)
    averaged["seeds"] = list(per_seed)
    return averaged


def score_pooled_predictions(rows: Sequence[Mapping], positive_labels: Sequence[str]) -> dict:
    """Score rows as one population. Used only to show that pooling is not the seed mean."""
    return summarize_counts(rows, positive_labels)


def smoothed_fine_probabilities(counts: Mapping[str, int], fine_ids: Sequence[str]) -> dict[str, float]:
    total = sum(counts.get(fine_id, 0) + 1 for fine_id in fine_ids)
    return {fine_id: (counts.get(fine_id, 0) + 1) / total for fine_id in fine_ids}


def information_content(probability: float) -> float:
    if probability <= 0.0 or probability > 1.0:
        raise ValueError(f"Probability outside (0, 1]: {probability}")
    return -math.log(probability)


def retention_for_group(
    fine_id: str,
    members: Sequence[str],
    probabilities: Mapping[str, float],
    *,
    level: str,
    n_fine: int,
) -> dict[str, float]:
    """Frequency and frequency-free retention of one fine label in one group.

    P1 has no category distinction under either score. Detection is reported
    separately by the task metric.
    """
    if level == "P1" or fine_id not in set(members):
        return {"frequency": 0.0, "uniform": 0.0}
    group_probability = sum(probabilities[member] for member in members)
    frequency = information_content(group_probability) / information_content(probabilities[fine_id])
    uniform = 1.0 - (math.log(len(set(members))) / math.log(n_fine))
    return {"frequency": frequency, "uniform": uniform}


def information_retention(
    relations: Sequence[Mapping],
    schema: Schema,
    level: str,
    train_fine_counts: Mapping[str, int],
) -> dict:
    """Average retention over every positive fine gold relation, including misses.

    Each relation supplies its fine id, the candidate output status, and the
    predicted group ids at `level`. DrugProt takes the highest covering group.
    """
    probabilities = smoothed_fine_probabilities(train_fine_counts, schema.fine_ids)
    groups = {label.stable_id: label.members for label in schema.level_labels(level)}
    ceilings = {"frequency": [], "uniform": []}
    observed = {"frequency": [], "uniform": []}
    for relation in relations:
        fine_id = relation["fine_id"]
        parent = next(group_id for group_id, members in groups.items() if fine_id in members)
        ceiling = retention_for_group(
            fine_id, groups[parent], probabilities, level=level, n_fine=len(schema.fine_ids)
        )
        ceilings["frequency"].append(ceiling["frequency"])
        ceilings["uniform"].append(ceiling["uniform"])
        best = {"frequency": 0.0, "uniform": 0.0}
        if level != "P1" and relation["status"] in USABLE_STATUSES:
            for group_id in relation.get("predicted_groups", ()):
                score = retention_for_group(
                    fine_id,
                    groups[group_id],
                    probabilities,
                    level=level,
                    n_fine=len(schema.fine_ids),
                )
                if score["frequency"] > best["frequency"]:
                    best = score
        observed["frequency"].append(best["frequency"])
        observed["uniform"].append(best["uniform"])
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0
    result = {"n_gold_relations": len(relations), "level": level}
    for name in ("frequency", "uniform"):
        ceiling = _mean(ceilings[name])
        actual = _mean(observed[name])
        result[name] = {
            "ceiling_C": ceiling,
            "retained_R": actual,
            "schema_loss_1_minus_C": 1.0 - ceiling,
            "prediction_loss_C_minus_R": ceiling - actual,
            "total_loss_1_minus_R": 1.0 - actual,
        }
    return result


def map_gold_relations_to_level(fine_labels: Sequence[str], schema: Schema, level: str) -> list[str]:
    return sorted(map_fine_ids(schema, set(fine_labels), level))


def polarity_summary(rows: Sequence[Mapping], schema: Schema, level: str) -> dict:
    """Direction outcomes for gold relations that have an explicit polarity.

    A gold candidate that already contains both directions is counted once as
    a bidirectional case. Its label set is compared as a set.
    """
    counts = {
        "retained": 0,
        "reversed": 0,
        "coarsened_nonpolar": 0,
        "missed": 0,
        "conflict": 0,
        "bidirectional_set_match": 0,
        "bidirectional_mismatch": 0,
        "bidirectional_missed": 0,
        "bidirectional_conflict": 0,
    }
    for row in rows:
        fine_ids = list(row["fine_ids"])
        directions = [schema.directions[fine_id] for fine_id in fine_ids]
        polar = [direction for direction in directions if direction in {"positive", "negative"}]
        if not polar:
            continue
        predicted_directions = set()
        if row["status"] in USABLE_STATUSES:
            for group_id in row.get("predicted_groups", ()):
                predicted_directions.add(group_direction(schema, level, group_id))
        if "positive" in polar and "negative" in polar:
            _add_bidirectional(counts, row["status"], polar, predicted_directions)
            continue
        gold_direction = polar[0]
        counts[_single_polarity_outcome(row["status"], gold_direction, predicted_directions)] += 1
    counts["n_cases"] = sum(counts.values())
    return counts


def _single_polarity_outcome(status: str, gold_direction: str, predicted: set[str]) -> str:
    if status == "rejected":
        return "conflict"
    if status == "missing" or not predicted:
        return "missed"
    opposite = "negative" if gold_direction == "positive" else "positive"
    has_gold = gold_direction in predicted
    has_opposite = opposite in predicted
    if has_gold and has_opposite:
        return "conflict"
    if has_opposite:
        return "reversed"
    if has_gold:
        return "retained"
    return "coarsened_nonpolar"


def _add_bidirectional(counts: dict, status: str, gold: list[str], predicted: set[str]) -> None:
    if status == "rejected":
        counts["bidirectional_conflict"] += 1
        return
    gold_set = set(gold)
    pred_polar = {item for item in predicted if item in {"positive", "negative"}}
    if status == "missing" or not pred_polar:
        counts["bidirectional_missed"] += 1
    elif pred_polar == gold_set:
        counts["bidirectional_set_match"] += 1
    else:
        counts["bidirectional_mismatch"] += 1


def formal_bootstrap_from_config(config: Mapping) -> dict:
    """Read the formal bootstrap settings. Callers must pass this result explicitly.

    The function default on the resampling helpers is not the formal source.
    """
    block = config["bootstrap"]
    seed = int(block["seed"])
    draws = int(block["draws"])
    interval = str(block["interval"])
    if draws != 2000 or interval != "percentile_95":
        raise ValueError("Formal bootstrap draws and interval are fixed by the study decision.")
    return {"seed": seed, "n_draws": draws, "interval": interval}


def load_formal_bootstrap(path: Path) -> dict:
    import yaml

    return formal_bootstrap_from_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def document_resample_indices(n_documents: int, n_draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> list[list[int]]:
    """Draw documents with replacement. One matrix is shared by a comparison."""
    if n_documents <= 0:
        raise ValueError("Bootstrap needs at least one document")
    rng = random.Random(seed)
    return [[rng.randrange(n_documents) for _ in range(n_documents)] for _ in range(n_draws)]


def apply_document_indices(rows: Sequence[Mapping], documents: Sequence[str], indices: Sequence[int]) -> list[Mapping]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["document_id"]].append(row)
    selected = []
    for index in indices:
        selected.extend(grouped[documents[index]])
    return selected


def bootstrap_paired_mean_difference(
    rows_by_seed_and_condition: Mapping[str, Mapping[str, Sequence[Mapping]]],
    documents: Sequence[str],
    positive_labels: Sequence[str],
    *,
    condition_a: str,
    condition_b: str,
    metric: str = "f1",
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Resample documents once, recompute each seed, average seeds, then subtract.

    The interval is conditional on the already trained or saved outputs. Seeds
    are not redrawn.
    """
    indices = document_resample_indices(len(documents), n_draws, seed)
    observed = _paired_mean(rows_by_seed_and_condition, documents, list(range(len(documents))), positive_labels, condition_a, condition_b, metric)
    draws = [
        _paired_mean(rows_by_seed_and_condition, documents, draw, positive_labels, condition_a, condition_b, metric)
        for draw in indices
    ]
    ordered = sorted(draws)
    lower = ordered[int(0.025 * (len(ordered) - 1))]
    upper = ordered[int(0.975 * (len(ordered) - 1))]
    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "metric": metric,
        "observed_mean_difference": observed,
        "interval_low": lower,
        "interval_high": upper,
        "n_draws": n_draws,
        "seed": seed,
        "interval": "percentile_95",
        "conditioning": "fixed_trained_or_saved_outputs",
    }


def _paired_mean(rows_by_seed, documents, indices, positive_labels, condition_a, condition_b, metric) -> float:
    differences = []
    for seed, conditions in rows_by_seed.items():
        sample_a = apply_document_indices(conditions[condition_a], documents, indices)
        sample_b = apply_document_indices(conditions[condition_b], documents, indices)
        score_a = summarize_counts(sample_a, positive_labels)[metric]
        score_b = summarize_counts(sample_b, positive_labels)[metric]
        if score_a is None or score_b is None:
            raise ValueError(f"Metric {metric} is undefined for seed {seed}")
        differences.append(score_a - score_b)
    return sum(differences) / len(differences)


def threshold_grid() -> list[float]:
    return [round(step / 100, 2) for step in range(5, 100, 5)]


def select_common_threshold(gold_sets: Sequence[set[str]], probabilities: Sequence[Mapping[str, float]], labels: Sequence[str]) -> dict:
    """One threshold for every DrugProt label. Ties keep the higher threshold."""
    best = None
    for threshold in threshold_grid():
        rows = []
        for gold, probs in zip(gold_sets, probabilities):
            predicted = {label for label in labels if probs.get(label, 0.0) >= threshold}
            rows.append({"gold": gold, "predicted": predicted, "status": "valid", "illegal_labels": []})
        score = summarize_counts(rows, labels)
        candidate = (score["f1"], threshold, score)
        if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] > best[1]):
            best = candidate
    assert best is not None
    return {"threshold": best[1], "f1": best[0], "metrics": best[2]}


def select_epoch(epoch_scores: Sequence[Mapping]) -> dict:
    """Highest positive micro-F1. Ties keep the earlier epoch.

    DrugProt passes the F1 of the already selected threshold for that epoch.
    """
    best = None
    for item in epoch_scores:
        candidate = (item["f1"], -item["epoch"], item)
        if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] > best[1]):
            best = candidate
    if best is None:
        raise ValueError("No epoch scores")
    return best[2]


def binary_cross_entropy_with_logits(logits: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]) -> float:
    """Mean BCE-with-logits over examples and labels. No training is performed."""
    total = 0.0
    count = 0
    for row_logits, row_targets in zip(logits, targets):
        for logit, target in zip(row_logits, row_targets):
            # max(logit, 0) - logit * target + log(1 + exp(-abs(logit)))
            total += max(logit, 0.0) - logit * target + math.log1p(math.exp(-abs(logit)))
            count += 1
    if count == 0:
        raise ValueError("Empty BCE inputs")
    return total / count


def softmax_cross_entropy(logits: Sequence[Sequence[float]], target_index: Sequence[int]) -> float:
    total = 0.0
    for row, target in zip(logits, target_index):
        maximum = max(row)
        shifted = [value - maximum for value in row]
        log_z = math.log(sum(math.exp(value) for value in shifted))
        total += -(shifted[target] - log_z)
    return total / len(target_index)
