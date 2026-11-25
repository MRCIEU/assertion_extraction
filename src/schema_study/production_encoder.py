"""Hierarchical and sensitivity losses for the biomedical encoders.

The small encoder is only a CPU fixture. A production task loads BiomedBERT or
BioLinkBERT through `load_pretrained_encoder` and uses schema head widths.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from schema_study.formal_optim import HEADS, consistency_kl, equal_mean_head_loss, map_native_distribution, parent_child_penalty
from schema_study.production_assets import encoder_layout
from schema_study.production_records import class_weight_from_counts, head_sizes, parent_child_edges
from schema_study.schemas import default_schema_path, load_schema


class TinyEncoder(nn.Module):
    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.embed(input_ids)
        if attention_mask is None:
            return hidden.mean(dim=1)
        weights = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


class SchemaHierarchicalModel(nn.Module):
    """Four heads whose widths are the schema label counts."""

    def __init__(self, encoder: nn.Module, hidden: int, sizes: dict[str, int], corpus: str) -> None:
        super().__init__()
        self.encoder = encoder
        self.corpus = corpus
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, sizes[name]) for name in HEADS})

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.encoder(input_ids, attention_mask)
        if hasattr(hidden, "last_hidden_state"):
            hidden = hidden.last_hidden_state[:, 0]
        elif hidden.ndim == 3:
            hidden = hidden[:, 0]
        return {name: layer(hidden) for name, layer in self.heads.items()}


def ProductionHierarchicalEncoder(corpus: str = "biored", hidden: int = 8, vocab: int = 32) -> SchemaHierarchicalModel:
    """CPU fixture with the real schema dimensions. It does not load BiomedBERT."""
    schema = _schema(corpus)
    return SchemaHierarchicalModel(TinyEncoder(vocab, hidden), hidden, head_sizes(schema, corpus), corpus)


def _schema(corpus: str):
    code = Path("/home/b5ac/freddieyu.b5ac/project_1")
    name = "drugprot" if corpus == "drugprot" else "biored"
    return load_schema(default_schema_path(code, name))


def load_pretrained_encoder(model_name: str):
    """Load the selected encoder. BioLinkBERT is not routed through the BiomedBERT directory."""
    layout = encoder_layout(model_name)
    if model_name == "biolinkbert" and "BiomedNLP" in str(layout["model"]):
        raise RuntimeError("BioLinkBERT resolved to the BiomedBERT directory.")
    from transformers import AutoModel

    return AutoModel.from_pretrained(layout["model"], local_files_only=True)


def prepare_marker_encoder(model_name: str, seed: int = 13):
    """Load the encoder and add the same marker embeddings used by the principal BiomedBERT trainer."""
    from schema_study.experiments import seed_streams
    from schema_study.inputs import SPECIAL_MARKERS, load_biomedbert_tokenizer

    tokenizer = load_biomedbert_tokenizer(str(encoder_layout(model_name)["tokenizer"]))
    encoder = load_pretrained_encoder(model_name)
    before = encoder.get_input_embeddings().weight.shape[0]
    encoder.resize_token_embeddings(len(tokenizer))
    after = encoder.get_input_embeddings().weight.shape[0]
    if after - before != len(SPECIAL_MARKERS):
        raise RuntimeError(f"Embedding resize mismatch: {before} -> {after}")
    streams = seed_streams(int(seed), "hierarchical", "shared")
    generator = torch.Generator()
    generator.manual_seed(streams["marker_embedding_init"] % (2**31 - 1))
    with torch.no_grad():
        rows = torch.randn((len(SPECIAL_MARKERS), encoder.config.hidden_size), generator=generator)
        encoder.get_input_embeddings().weight[-len(SPECIAL_MARKERS):].copy_(rows * encoder.config.initializer_range)
    return encoder, tokenizer


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Single-label focal loss. DrugProt must use the multi-label form."""
    if target.ndim != 1:
        raise ValueError("Categorical focal loss is not a DrugProt objective.")
    probabilities = torch.softmax(logits, dim=-1)
    gathered = probabilities.gather(1, target.view(-1, 1)).squeeze(1).clamp_min(1e-8)
    loss = F.nll_loss(torch.log(probabilities), target, reduction="none")
    return ((1.0 - gathered).pow(gamma) * loss).mean()


def multilabel_focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability_true = probabilities * target + (1.0 - probabilities) * (1.0 - target)
    return ((1.0 - probability_true).pow(gamma) * loss).mean()


def class_weighted_loss(logits: torch.Tensor, target: torch.Tensor, counts: list[int]) -> tuple[torch.Tensor, list[float]]:
    weights = class_weight_from_counts(counts)
    if target.ndim == 1:
        loss = F.cross_entropy(logits, target, weight=torch.tensor(weights, dtype=logits.dtype))
    else:
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(weights, dtype=logits.dtype))
    return loss, weights


def hierarchical_objective(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], corpus: str, schema, consistency: bool) -> tuple[torch.Tensor, dict]:
    """Equal mean of the four head losses, plus one corpus-specific consistency term."""
    losses = {}
    if corpus == "drugprot":
        for name in HEADS:
            losses[name] = F.binary_cross_entropy_with_logits(outputs[name], targets[name])
    else:
        for name in HEADS:
            losses[name] = F.cross_entropy(outputs[name], targets[name])
    objective = equal_mean_head_loss(losses)
    detail = {"normalisation": "equal_mean_of_per_head_mean_losses", "consistency_weight": 0.0}
    if not consistency:
        return objective, detail
    detail["consistency_weight"] = 0.1
    if corpus == "drugprot":
        penalties = []
        for edge in parent_child_edges(schema):
            parent = torch.sigmoid(outputs[edge["parent_level"]][:, edge["parent_index"]])
            child = torch.sigmoid(outputs[edge["child_level"]][:, edge["child_index"]])
            penalties.append(parent_child_penalty(parent, child))
        objective = objective + 0.1 * torch.stack(penalties).mean()
        detail["consistency"] = "drugprot_squared_hinge"
    else:
        native = torch.softmax(outputs["P3"], dim=-1)
        terms = []
        for coarse, historical in (("P0", "P1"), ("P1", "P2"), ("P2", "P3")):
            from schema_study.production_plan import membership_rows

            membership = torch.tensor(membership_rows(schema, historical, "P4"), dtype=native.dtype)
            mapped = map_native_distribution(native, membership)
            terms.append(consistency_kl(F.log_softmax(outputs[coarse], dim=-1), mapped, detach_target=True))
        objective = objective + 0.1 * torch.stack(terms).mean()
        detail["consistency"] = "biored_detached_mapped_native_kl"
    return objective, detail


def _batch(model: SchemaHierarchicalModel, step: int, schema) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(1000 + step)
    token_ids = torch.randint(0, 32, (2, 4), generator=generator)
    targets = {}
    if model.corpus == "drugprot":
        for name, layer in model.heads.items():
            targets[name] = torch.zeros(2, layer.out_features)
            targets[name][:, 0] = 1
    else:
        for name, layer in model.heads.items():
            targets[name] = torch.zeros(2, dtype=torch.long)
    return token_ids, targets


def _write_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_hierarchical(model: SchemaHierarchicalModel | None = None, steps: int = 2, consistency: bool = False, corpus: str = "biored", batches=None, score_fn=None, total_steps: int | None = None, output_dir=None, epoch_updates: int | None = None) -> dict:
    """Train the hierarchical objective. A supplied batch iterator is the production population; the default is a schema-width fixture."""
    from schema_study.production_runtime import training_status

    if model is None:
        model = ProductionHierarchicalEncoder(corpus)
    schema = _schema(model.corpus)
    production = batches is not None
    if production:
        if not torch.cuda.is_available():
            raise RuntimeError("Hierarchical training requires CUDA.")
        model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    gradients = {}
    detail = {}
    best = None
    best_state = None
    horizon = int(total_steps or steps)
    completed = 0
    if output_dir is not None and production:
        last_path = Path(output_dir) / "checkpoints/last/model.pt"
        if last_path.is_file():
            saved = torch.load(last_path, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["state_dict"])
            model.to("cuda")
            optimizer.load_state_dict(saved["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to("cuda")
            completed = int(saved.get("step") or 0)
            best = saved.get("best")
    produced = _production_batches(model, schema, steps) if batches is None else batches
    step = completed
    seen = 0
    for item in produced:
        if step >= steps:
            break
        if seen < completed:
            seen += 1
            continue
        if isinstance(item, dict):
            token_ids, targets = item["input_ids"], item["targets"]
            device = next(model.parameters()).device
            token_ids = token_ids.to(device)
            mask = item.get("attention_mask")
            if mask is not None:
                mask = mask.to(device)
            targets = {name: value.to(device) for name, value in targets.items()}
            outputs = model(token_ids, mask)
        else:
            token_ids, targets = item
            outputs = model(token_ids)
        objective, detail = hierarchical_objective(outputs, targets, model.corpus, schema, consistency)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        step += 1
        seen += 1
        if step == 1 or (completed and step == completed + 1):
            gradients = {
                name: float(parameter.grad.detach().abs().sum())
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            if output_dir is not None and torch.isfinite(objective) and not (Path(output_dir) / "scientific_status.json").is_file():
                from schema_study.production_runtime import publish_scientific_status

                publish_scientific_status(Path(output_dir), {
                    "technical_preflight": True,
                    "useful_computation_started": time.time(),
                    "model_loaded": True,
                    "device": str(next(model.parameters()).device),
                    "optimiser_steps": 1,
                    "loss_finite": True,
                    "checkpoint_validated": False,
                    "monitoring_guarantees_completion": False,
                })
        selection_due = step % epoch_updates == 0 if epoch_updates else step % max(1, horizon) == 0
        if score_fn is not None and selection_due:
            metric = float(score_fn(model))
            if best is None or metric > best["development_micro_f1"]:
                best = {"step": step, "development_micro_f1": metric}
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                if output_dir is not None and production:
                    _write_checkpoint(Path(output_dir) / "checkpoints/best/model.pt", {
                        "state_dict": best_state,
                        "best": best,
                        "corpus": model.corpus,
                    })
        if output_dir is not None and production and (step % 50 == 0 or step == steps):
            _write_checkpoint(Path(output_dir) / "checkpoints/last/model.pt", {
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best": best,
                "corpus": model.corpus,
            })
    head_gradients = {name: value for name, value in gradients.items() if name.startswith("heads.")}
    status = training_status(step, horizon, best is not None) if production else "fixture"
    checkpoint_path = Path(output_dir) / "checkpoints/best/model.pt" if output_dir is not None else None
    if output_dir is not None and production:
        from schema_study.production_runtime import publish_scientific_status

        started = time.time()
        existing = Path(output_dir) / "scientific_status.json"
        if existing.is_file():
            prior = json.loads(existing.read_text(encoding="utf-8"))
            if prior.get("useful_computation_started") is not None:
                started = prior["useful_computation_started"]
        publish_scientific_status(Path(output_dir), {
            "scientific": status,
            "preflight_passed": step >= 1,
            "technical_preflight": step >= 1,
            "useful_computation_started": started,
            "selection_recorded": best is not None,
            "checkpoint_validated": checkpoint_path is not None and checkpoint_path.is_file(),
            "monitoring_guarantees_completion": False,
            "steps": step,
            "total_steps": horizon,
        })
    return {
        "steps": step,
        "consistency": consistency,
        "corpus": model.corpus,
        "head_sizes": {name: int(layer.out_features) for name, layer in model.heads.items()},
        "all_heads_receive_gradient": len({name.split(".")[1] for name in head_gradients}) == 4 and all(value > 0 for value in head_gradients.values()),
        "shared_encoder_gradient": any(name.startswith("encoder.") and value > 0 for name, value in gradients.items()),
        "detail": detail,
        "status": status,
        "completed": status == "training_completed",
        "best": best,
        "empirical_result": False,
    }


def _production_batches(model: SchemaHierarchicalModel, schema, steps: int):
    for step in range(steps):
        yield _batch(model, step, schema)


def train_encoder(model: nn.Module, steps: int = 1, loss_name: str = "standard", corpus: str = "biored", counts: list[int] | None = None) -> dict:
    """Sensitivity losses. Class weights are derived from counts, not written in as constants."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    recorded = None
    finite = True
    for _step in range(steps):
        outputs = model(torch.randint(0, 32, (4, 3)))
        logits = outputs["P3"]
        if corpus == "drugprot":
            target = torch.zeros(logits.shape[0], logits.shape[-1])
            target[:, 0] = 1
            if loss_name == "focal":
                loss = multilabel_focal_loss(logits, target)
            elif loss_name == "class_weighted":
                width = logits.shape[-1]
                loss, recorded = class_weighted_loss(logits, target, counts or [index + 1 for index in range(width)])
            else:
                loss = F.binary_cross_entropy_with_logits(logits, target)
        else:
            target = torch.zeros(logits.shape[0], dtype=torch.long)
            if loss_name == "focal":
                loss = focal_loss(logits, target)
            elif loss_name == "class_weighted":
                width = logits.shape[-1]
                loss, recorded = class_weighted_loss(logits, target, counts or [1] * width)
            else:
                loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            finite = False
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {"loss_name": loss_name, "finite": finite, "class_weights": recorded, "empirical_result": False}
