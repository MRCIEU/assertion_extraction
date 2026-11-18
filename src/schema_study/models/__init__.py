"""Model execution stays behind an explicit permit.

Importing this package does not import PyTorch, vLLM, or a remote client.
"""

from schema_study.models.schedule import ModelExecutionBlocked, ROUND_MODEL_EXECUTION_PERMITTED

__all__ = ["ModelExecutionBlocked", "ROUND_MODEL_EXECUTION_PERMITTED"]
