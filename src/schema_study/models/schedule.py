"""Training schedule and the round-2 execution gate.

Importing this module does not import PyTorch or load a model.
"""

from __future__ import annotations

STUDY_HYPERPARAMETERS = {
    "learning_rate": 2e-5,
    "max_epochs": 10,
    "effective_batch_size": 16,
    "warmup_ratio": 0.1,
    "betas": (0.9, 0.999),
    "epsilon": 1e-8,
    "weight_decay": 0.01,
    "gradient_clip_norm": 1.0,
    "classifier_dropout": 0.1,
}
# This round can validate the code paths. It cannot enable weight loading.
ROUND_MODEL_EXECUTION_PERMITTED = False
CANDIDATE_TRAINING_PRECISION = "bf16"
CANDIDATE_BIOMEDBERT_PREDICTION_PRECISION = "fp32"
CANDIDATE_QWEN_GENERATION_PRECISION = "bf16"
CANDIDATE_QWEN_QUANTIZATION = "none"
PRECISION_STATUS = "candidate_pending_gpu_smoke_test"


class ModelExecutionBlocked(RuntimeError):
    """Parameter checks passed, but model execution is not permitted."""


def validate_training_config(config: dict) -> list[str]:
    """Return configuration errors. This does not construct a model."""
    errors = []
    for key, expected in STUDY_HYPERPARAMETERS.items():
        if key not in config:
            errors.append(f"missing {key}")
            continue
        actual = config[key]
        if key == "betas":
            if tuple(actual) != tuple(expected):
                errors.append(f"betas is {actual!r}")
            continue
        if actual != expected and not _close(actual, expected):
            errors.append(f"{key} is {actual!r}, study value is {expected!r}")
    micro = int(config.get("micro_batch_size", STUDY_HYPERPARAMETERS["effective_batch_size"]))
    effective = int(config.get("effective_batch_size", 0) or 0)
    if micro <= 0 or effective % micro != 0:
        errors.append("micro_batch_size must divide the effective batch size of 16")
    if config.get("class_weights") or config.get("novelty_head") or config.get("extra_negative_sampling"):
        errors.append("class weights, novelty, and extra negative sampling are disabled")
    return errors


def _close(left, right) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def accumulation_steps(micro_batch_size: int, effective_batch_size: int = 16) -> int:
    if micro_batch_size <= 0 or effective_batch_size % micro_batch_size != 0:
        raise ValueError("micro-batch size must divide the effective batch size")
    return effective_batch_size // micro_batch_size


def learning_rate_at_step(step: int, total_steps: int, base_learning_rate: float, warmup_ratio: float) -> float:
    """Linear warmup over the first fraction of steps, then linear decay to zero."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step < warmup_steps:
        return base_learning_rate * float(step + 1) / float(warmup_steps)
    remain = total_steps - warmup_steps
    if remain <= 0:
        return 0.0
    progress = float(step - warmup_steps + 1) / float(remain)
    return base_learning_rate * max(0.0, 1.0 - progress)


def decay_parameter_names(parameter_names: list[str]) -> dict[str, list[str]]:
    """Bias and normalization parameters do not use weight decay."""
    decay = []
    no_decay = []
    for name in parameter_names:
        lowered = name.lower()
        if lowered.endswith("bias") or "norm" in lowered:
            no_decay.append(name)
        else:
            decay.append(name)
    return {"decay": decay, "no_decay": no_decay}


def assert_execution_allowed(requested: bool, action: str) -> None:
    """Separate the caller's request from this round's execution permit."""
    if not requested or not ROUND_MODEL_EXECUTION_PERMITTED:
        raise ModelExecutionBlocked(
            f"{action} was not executed. requested={requested} "
            f"round_permit={ROUND_MODEL_EXECUTION_PERMITTED}"
        )


def plan_optimizer_groups(
    n_examples: int,
    micro_batch_size: int,
    effective_batch_size: int = 16,
) -> list[dict]:
    """Group micro-batches into optimizer updates.

    A full group has `effective_batch_size` examples. The final group is kept
    even when it is smaller. No example is dropped or repeated.
    """
    if n_examples < 0 or micro_batch_size <= 0 or effective_batch_size <= 0:
        raise ValueError("Batch sizes must be positive and the example count cannot be negative.")
    groups = []
    cursor = 0
    while cursor < n_examples:
        micros = []
        count = 0
        while cursor < n_examples and count < effective_batch_size:
            width = min(micro_batch_size, n_examples - cursor, effective_batch_size - count)
            micros.append({"start": cursor, "count": width})
            cursor += width
            count += width
        groups.append({"n_examples": count, "micro_batches": micros, "is_tail": count < effective_batch_size})
    return groups


def normalized_group_loss(micro_totals: list[float], micro_counts: list[int]) -> float:
    """Normalize by the examples in this group, not by an equal mean of micro-batches."""
    if len(micro_totals) != len(micro_counts) or not micro_counts or any(count <= 0 for count in micro_counts):
        raise ValueError("Each micro-batch needs a positive example count.")
    total_count = sum(micro_counts)
    return sum(micro_totals) / total_count


def run_accumulated_epoch(n_examples: int, micro_batch_size: int, step, effective_batch_size: int = 16) -> dict:
    """Call `step(do_optimizer_step, micro_count)` once per micro-batch.

    The final short group is updated. After the epoch the accumulator is empty.
    """
    groups = plan_optimizer_groups(n_examples, micro_batch_size, effective_batch_size)
    forwards = 0
    optimizer_steps = 0
    seen = []
    for group in groups:
        for offset, micro in enumerate(group["micro_batches"]):
            forwards += 1
            do_step = offset == len(group["micro_batches"]) - 1
            step(do_step, micro["count"])
            seen.extend(range(micro["start"], micro["start"] + micro["count"]))
            optimizer_steps += int(do_step)
    return {
        "forwards": forwards,
        "optimizer_steps": optimizer_steps,
        "effective_batch_size": effective_batch_size,
        "micro_batch_size": micro_batch_size,
        "group_sizes": [group["n_examples"] for group in groups],
        "example_indices": seen,
        "pending_examples": 0,
    }
