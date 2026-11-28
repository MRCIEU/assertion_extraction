"""Explicit checkpoint roles. A rolling file stores the current adapter and the selected best adapter separately."""

from __future__ import annotations


def role_weights(saved: dict, role: str) -> dict:
    """Return the tensor dictionary for one role. The embedded best is never used as the final step."""
    if role == "last":
        adapter = saved.get("adapter")
        if not isinstance(adapter, dict) or not adapter:
            raise ValueError("The final-step checkpoint has no adapter for its optimiser step.")
        return adapter
    if role == "best":
        if saved.get("role") == "best":
            adapter = saved.get("adapter")
            if not isinstance(adapter, dict) or not adapter:
                raise ValueError("The best-checkpoint file has no adapter.")
            return adapter
        embedded = saved.get("best_adapter")
        if isinstance(embedded, dict) and embedded:
            return embedded
        adapter = saved.get("adapter")
        if isinstance(adapter, dict) and adapter and "best_adapter" not in saved:
            return adapter
        raise ValueError("This checkpoint does not contain a distinct selected-best adapter.")
    if role == "periodic":
        adapter = saved.get("adapter")
        if not isinstance(adapter, dict) or not adapter:
            raise ValueError("The periodic checkpoint has no adapter for its optimiser step.")
        return adapter
    raise ValueError(f"Unknown checkpoint role {role}.")


def saved_for_role(saved: dict, role: str) -> dict:
    """A restoration payload whose adapter field is the role being scored."""
    payload = dict(saved)
    payload["adapter"] = role_weights(saved, role)
    payload["checkpoint_role"] = role
    return payload


def selected_checkpoint(scores: dict) -> str | None:
    """The higher finite development score. A missing or sentinel score selects nothing."""
    if set(scores) != {"best", "last"}:
        return None
    for value in scores.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if value < 0 or value != value or value in {float("inf"), float("-inf")}:
            return None
    return "best" if scores["best"] >= scores["last"] else "last"


def release_evidence_complete(payload: dict) -> bool:
    """File names are not release evidence. Every required check must have been recorded as true."""
    required = {
        "model_identity",
        "sharding",
        "effective_batch",
        "checkpoint_restored",
        "optimiser_state_present",
        "development_evaluation",
        "selected_checkpoint",
        "adapter_export",
        "adapter_restored",
        "inference_diagnostic",
    }
    checks = payload.get("checks") or {}
    if not isinstance(checks, dict):
        return False
    return payload.get("verified") is True and required <= {key for key, value in checks.items() if value is True}
