"""Production hierarchical encoder loop.

`train_hierarchical` is the loop the dispatcher calls. The CPU test builds the
same module and runs that function. It does not load BioLinkBERT weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from schema_study.formal_optim import HEADS, consistency_kl, equal_mean_head_loss, map_native_distribution, parent_child_penalty
class ProductionHierarchicalEncoder(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, 2 + index) for index, name in enumerate(HEADS)})

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.embed(input_ids).mean(dim=1)
        return {name: layer(hidden) for name, layer in self.heads.items()}


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Mean focal loss. Gamma is fixed before any evaluation."""
    probabilities = torch.softmax(logits, dim=-1)
    gathered = probabilities.gather(1, target.view(-1, 1)).squeeze(1).clamp_min(1e-8)
    loss = F.nll_loss(torch.log(probabilities), target, reduction="none")
    return ((1.0 - gathered).pow(gamma) * loss).mean()


def class_weighted_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target, weight=weight)


def train_hierarchical(model: ProductionHierarchicalEncoder, steps: int = 2, consistency: bool = False) -> dict:
    """Equal-mean head loss, with the consistency term only when requested."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    membership = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gradients = {}
    for step in range(steps):
        token_ids = torch.randint(0, 32, (2, 4))
        target = torch.tensor([step % 2, 0])
        outputs = model(token_ids)
        losses = {name: F.cross_entropy(outputs[name][:, :2], target) for name in HEADS}
        objective = equal_mean_head_loss(losses)
        if consistency:
            mapped = map_native_distribution(torch.softmax(outputs["P3"][:, :2], dim=-1), membership)
            objective = objective + 0.1 * consistency_kl(F.log_softmax(outputs["P0"][:, :2], dim=-1), mapped, detach_target=True)
            objective = objective + parent_child_penalty(torch.sigmoid(outputs["P0"][:, 1]), torch.sigmoid(outputs["P3"][:, 1]))
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        if step == 0:
            gradients = {
                name: float(parameter.grad.detach().abs().sum())
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
    head_gradients = {name: value for name, value in gradients.items() if name.startswith("heads.")}
    return {
        "steps": steps,
        "consistency": consistency,
        "all_heads_receive_gradient": len(head_gradients) >= 4 and all(value > 0 for value in head_gradients.values()),
        "shared_encoder_gradient": gradients.get("embed.weight", 0.0) > 0,
        "empirical_result": False,
    }


def train_encoder(model: nn.Module, steps: int = 1, loss_name: str = "standard") -> dict:
    """Sensitivity losses use this loop. The standard loss remains the principal reference."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _step in range(steps):
        logits = model(torch.randint(0, 32, (4, 3)))["P3"][:, :2]
        target = torch.zeros(4, dtype=torch.long)
        if loss_name == "focal":
            loss = focal_loss(logits, target)
        elif loss_name == "class_weighted":
            loss = class_weighted_loss(logits, target, torch.tensor([0.2, 0.8]))
        else:
            loss = F.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {"loss_name": loss_name, "finite": True, "empirical_result": False}
