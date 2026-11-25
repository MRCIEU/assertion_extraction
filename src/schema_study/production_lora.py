"""Production LoRA optimisation loop.

The CPU fixture calls `fit`. It does not use a separate demonstration trainer.
Loading Qwen weights remains behind the approval gate and is not done here.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from schema_study.formal_optim import IGNORE_INDEX, assistant_supervision
from schema_study.learning_conditions import LORA_TARGET_MODULES


class FixtureCausalModel(nn.Module):
    """A small causal stand-in with the same projection names as Qwen."""

    def __init__(self, vocab: int = 32, hidden: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        for name in LORA_TARGET_MODULES:
            setattr(self, name, nn.Linear(hidden, hidden, bias=False))
        self.output = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.embed(input_ids)
        for name in LORA_TARGET_MODULES:
            hidden = getattr(self, name)(hidden)
        return self.output(hidden)


def attach_lora(model: nn.Module, rank: int = 16, alpha: int = 32, dropout: float = 0.05) -> list[str]:
    """Inject a PEFT LoRA adapter and return the trainable parameter names."""
    from peft import LoraConfig, inject_adapter_in_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(LORA_TARGET_MODULES),
    )
    inject_adapter_in_model(config, model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("PEFT did not leave only LoRA parameters trainable.")
    if not any(module in name for name in trainable for module in LORA_TARGET_MODULES):
        raise RuntimeError("The LoRA adapter is not attached to the projection modules.")
    return trainable


def _batch(step: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(step)
    token_ids = torch.randint(0, 32, (2, 6), generator=generator)
    attention = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    return {"input_ids": token_ids, "attention_mask": attention, "prompt_tokens": [2, 2]}


def _loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    labels = assistant_supervision(batch["input_ids"], batch["attention_mask"], batch["prompt_tokens"])
    supervised = (labels != IGNORE_INDEX).sum(dim=1)
    if int(supervised.min()) == 0:
        raise ValueError("A row in the batch has no supervised assistant tokens.")
    logits = model(batch["input_ids"], batch["attention_mask"])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=IGNORE_INDEX)


def _capture(model: nn.Module, optimizer, scheduler, step: int, position: int) -> dict:
    return {
        "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "position": position,
        "torch": torch.get_rng_state(),
    }


def _restore(model: nn.Module, optimizer, scheduler, saved: dict) -> None:
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    for group, saved_group in zip(optimizer.param_groups, saved["optimizer"]["param_groups"]):
        group["lr"] = saved_group["lr"]
    torch.set_rng_state(saved["torch"])


def fit(model: nn.Module, steps: int, resume: dict | None = None, learning_rate: float = 1e-2, total_steps: int | None = None) -> dict:
    """The production update loop. `total_steps` is the full run, so a segment does not rebuild the schedule."""
    if resume is None:
        attach_lora(model, rank=4, alpha=8, dropout=0.0)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=max(total_steps or steps, 1))
    step = 0
    position = 0
    if resume is not None:
        _restore(model, optimizer, scheduler, resume)
        step = int(resume["step"])
        position = int(resume["position"])
    losses = []
    learning_rates = []
    while step < steps:
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        batch = _batch(position)
        loss = _loss(model, batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Loss is not finite.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0 and not any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0 for parameter in trainable):
            raise RuntimeError("LoRA parameters received no gradient.")
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        step += 1
        position += 1
    return {
        "losses": losses,
        "learning_rates": learning_rates,
        "checkpoint": _capture(model, optimizer, scheduler, step, position),
        "empirical_qwen_result": False,
    }


def _fresh() -> FixtureCausalModel:
    torch.manual_seed(13)
    return FixtureCausalModel()


def compare_interrupted_run(steps: int = 4, interrupt_at: int = 2) -> dict:
    """Run the production `fit` uninterrupted, and again with a resume from its own checkpoint."""
    uninterrupted = fit(_fresh(), steps, total_steps=steps)
    paused = _fresh()
    partial = fit(paused, interrupt_at, total_steps=steps)
    resumed = fit(paused, steps, resume=partial["checkpoint"], total_steps=steps)
    weights_match = all(
        torch.allclose(uninterrupted["checkpoint"]["model"][name], resumed["checkpoint"]["model"][name], atol=1e-5)
        for name in uninterrupted["checkpoint"]["model"]
    )
    losses_match = all(
        abs(left - right) <= 1e-5
        for left, right in zip(uninterrupted["losses"], partial["losses"] + resumed["losses"])
    )
    return {
        "weights_match": weights_match,
        "steps": resumed["checkpoint"]["step"],
        "position": resumed["checkpoint"]["position"],
        "losses_match": losses_match,
        "empirical_qwen_result": False,
    }


def save_checkpoint(directory: Path, package: dict) -> Path:
    destination = directory / "adapter_checkpoint.pt"
    temporary = destination.with_suffix(".pt.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, temporary)
    temporary.replace(destination)
    return destination
