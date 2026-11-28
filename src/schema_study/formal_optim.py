"""Differentiable training operations and a CPU continuation fixture.

Importing this module imports PyTorch. It does not load BiomedBERT or Qwen
weights, and a passing fixture is not an empirical extraction result.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from schema_study.learning_conditions import LORA_TARGET_MODULES

HEADS = ("P0", "P1", "P2", "P3")
IGNORE_INDEX = -100


def assistant_supervision(input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_tokens: list[int]) -> torch.Tensor:
    """Ignore the prompt and padding positions. A completion token is kept even when its id equals the pad id."""
    if input_ids.shape != attention_mask.shape:
        raise ValueError("The token ids and attention mask must have the same shape.")
    if len(prompt_tokens) != input_ids.shape[0]:
        raise ValueError("Each row needs its own prompt length.")
    labels = input_ids.clone()
    for row, prompt in enumerate(prompt_tokens):
        if prompt < 0 or prompt > input_ids.shape[1]:
            raise ValueError("A prompt length is outside the sequence.")
        labels[row, :prompt] = IGNORE_INDEX
    labels = labels.masked_fill(attention_mask == 0, IGNORE_INDEX)
    if not bool((labels != IGNORE_INDEX).any()):
        raise ValueError("The supervised span is empty.")
    return labels


def equal_mean_head_loss(losses: dict[str, torch.Tensor]) -> torch.Tensor:
    """Equal mean of four head losses. Each head loss is already normalised by its own labels."""
    missing = [name for name in HEADS if name not in losses]
    if missing:
        raise KeyError(", ".join(missing))
    return torch.stack([losses[name] for name in HEADS]).mean()


def map_native_distribution(native_probabilities: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
    """Sum native probabilities into each coarser label, including the negative outcome."""
    if membership.ndim != 2 or native_probabilities.shape[-1] != membership.shape[1]:
        raise ValueError("Membership columns must match the native label dimension.")
    return native_probabilities @ membership.transpose(0, 1)


def consistency_kl(coarse_log_probabilities: torch.Tensor, mapped_native: torch.Tensor, detach_target: bool = True) -> torch.Tensor:
    """The mapped native distribution is a detached target. Gradients stay on the direct coarse head."""
    target = mapped_native.detach() if detach_target else mapped_native
    return F.kl_div(coarse_log_probabilities, target, reduction="batchmean")


def parent_child_penalty(parent_probability: torch.Tensor, child_probability: torch.Tensor) -> torch.Tensor:
    """Squared hinge requiring each parent probability to be at least its child. This is not a noisy-OR identity."""
    return F.relu(child_probability - parent_probability).square().mean()


def lora_parameter_names(named_parameters: list[tuple[str, nn.Parameter]]) -> list[str]:
    """Names of parameters that belong to the verified attention and MLP projections."""
    selected = []
    for name, parameter in named_parameters:
        if parameter.requires_grad and any(module in name.split(".") for module in LORA_TARGET_MODULES):
            selected.append(name)
    if not selected:
        raise ValueError("No trainable parameter matched the LoRA target modules.")
    return selected


class _SharedEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.heads = nn.ModuleDict({name: nn.Linear(8, 2 + index) for index, name in enumerate(HEADS)})

    def forward(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.embed(token_ids).mean(dim=1)
        return {name: layer(hidden) for name, layer in self.heads.items()}


def _capture(model, optimizer, scheduler, step: int) -> dict:
    return {
        "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "step": step,
    }


def _restore(model, optimizer, scheduler, saved: dict) -> None:
    model.load_state_dict(saved["model_state"])
    optimizer.load_state_dict(saved["optimizer_state"])
    scheduler.load_state_dict(saved["scheduler_state"])
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch"])


def _batch(step_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = random.Random(step_seed)
    token_ids = torch.tensor([[generator.randrange(16) for _column in range(4)] for _row in range(2)])
    target = torch.tensor([generator.randrange(2) for _row in range(2)])
    return token_ids, target


def _objective(outputs: dict[str, torch.Tensor], target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    losses = {name: F.cross_entropy(logits[:, :2], target) for name, logits in outputs.items()}
    task = equal_mean_head_loss(losses)
    native = torch.softmax(outputs["P3"][:, :2], dim=-1)
    membership = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mapped = map_native_distribution(native, membership)
    coarse = F.log_softmax(outputs["P0"][:, :2], dim=-1)
    consistency = consistency_kl(coarse, mapped, detach_target=True)
    parent = torch.sigmoid(outputs["P0"][:, 1])
    child = torch.sigmoid(outputs["P3"][:, 1])
    penalty = parent_child_penalty(parent, child)
    total = task + 0.1 * consistency + penalty
    gradients = {}
    return total, gradients


def run_continuation_check(steps: int = 4, interrupt_at: int = 2) -> dict:
    """Compare an uninterrupted fixture with one that stops and resumes from a real checkpoint."""
    if not 0 < interrupt_at < steps:
        raise ValueError("The interruption must fall inside the run.")

    torch.manual_seed(13)
    initial = _SharedEncoder().state_dict()

    def execute(saved: dict | None, count: int, completed: int) -> tuple[_SharedEncoder, dict, list[float]]:
        model = _SharedEncoder()
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=steps)
        if saved is not None:
            _restore(model, optimizer, scheduler, saved)
        losses = []
        state = saved
        for offset in range(count):
            step = completed + offset
            token_ids, target = _batch(step)
            total, _unused = _objective(model(token_ids), target)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(total.detach()))
            state = _capture(model, optimizer, scheduler, step + 1)
        return model, state, losses

    torch.manual_seed(13)
    random.seed(13)
    np.random.seed(13)
    uninterrupted, final_state, uninterrupted_losses = execute(None, steps, 0)
    interrupted_model, midpoint, first_losses = execute(None, interrupt_at, 0)
    resumed, resumed_state, second_losses = execute(midpoint, steps - interrupt_at, interrupt_at)
    same_weights = all(
        torch.allclose(uninterrupted.state_dict()[name], resumed.state_dict()[name], atol=1e-6)
        for name in uninterrupted.state_dict()
    )
    losses_match = all(abs(left - right) <= 1e-5 for left, right in zip(uninterrupted_losses, first_losses + second_losses))
    first_batch, first_target = _batch(0)
    probe = _SharedEncoder()
    probe.load_state_dict(interrupted_model.state_dict())
    total, _unused = _objective(probe(first_batch), first_target)
    probe.zero_grad(set_to_none=True)
    total.backward()
    head_gradients = {
        name: float(parameter.grad.detach().abs().sum())
        for name, parameter in probe.named_parameters()
        if parameter.grad is not None and name.startswith("heads.")
    }
    shared = float(probe.embed.weight.grad.detach().abs().sum())
    return {
        "uninterrupted_steps": final_state["step"],
        "resumed_steps": resumed_state["step"],
        "losses_match": losses_match,
        "weights_match": same_weights,
        "shared_encoder_gradient": shared,
        "head_gradients": head_gradients,
        "all_heads_receive_gradient": all(value > 0 for value in head_gradients.values()) and shared > 0,
        "empirical_qwen_result": False,
    }


def save_and_load_checkpoint(directory: Path) -> dict:
    """Write a checkpoint through a temporary file and read it back."""
    model = _SharedEncoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, total_iters=1)
    saved = _capture(model, optimizer, scheduler, 1)
    destination = directory / "checkpoint.pt"
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(saved, temporary)
    temporary.replace(destination)
    loaded = torch.load(destination, map_location="cpu", weights_only=False)
    restored = _SharedEncoder()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.LinearLR(restored_optimizer, start_factor=1.0, total_iters=1)
    _restore(restored, restored_optimizer, restored_scheduler, loaded)
    return {"step": loaded["step"], "weights_match": all(torch.equal(model.state_dict()[name], restored.state_dict()[name]) for name in model.state_dict())}
