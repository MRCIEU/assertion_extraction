"""Load the recorded BiomedBERT revision and fine-tune.

`run_biomedbert_training` is reached only after the round execution permit.
Importing this module does not import PyTorch.
"""

from __future__ import annotations


def run_biomedbert_training(config: dict, examples: list[dict], tokenizer) -> dict:
    from pathlib import Path

    import torch
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Dropout, Linear
    from transformers import AutoModel, get_linear_schedule_with_warmup

    loaded = AutoModel.from_pretrained(config["model_path"], revision=config.get("revision"))
    encoder = loaded.bert if hasattr(loaded, "bert") else loaded
    encoder.resize_token_embeddings(len(tokenizer))
    hidden = int(encoder.config.hidden_size)
    dropout = Dropout(float(config["classifier_dropout"]))
    head = Linear(hidden, int(config["n_labels"]))
    softmax = config["loss"] == "softmax_cross_entropy"
    criterion = CrossEntropyLoss() if softmax else BCEWithLogitsLoss()
    named = list(encoder.named_parameters()) + [
        ("classifier." + name, value) for name, value in head.named_parameters()
    ]
    decay = [value for name, value in named if "bias" not in name.lower() and "norm" not in name.lower()]
    no_decay = [value for name, value in named if "bias" in name.lower() or "norm" in name.lower()]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(config["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(config["learning_rate"]),
        betas=tuple(config["betas"]),
        eps=float(config["epsilon"]),
    )
    updates = int(config["total_update_steps"])
    warmup = max(1, int(updates * float(config["warmup_ratio"])))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, max(1, updates))
    micro = int(config["micro_batch_size"])
    group = int(config["effective_batch_size"]) // micro
    encoder.train()
    head.train()
    optimizer.zero_grad(set_to_none=True)
    seen = 0
    for example in examples:
        encoded = tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        output = encoder(**{key: value for key, value in encoded.items() if key in {"input_ids", "attention_mask", "token_type_ids"}})
        logits = head(dropout(output.last_hidden_state[:, 0]))
        target = torch.tensor(example["target"])
        if softmax:
            loss = criterion(logits, target.long().view(-1))
        else:
            loss = criterion(logits, target.float().view(1, -1))
        (loss / group).backward()
        seen += 1
        if seen % group == 0:
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()),
                float(config["gradient_clip_norm"]),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    destination = Path(config["checkpoint_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"encoder": encoder.state_dict(), "classifier": head.state_dict(), "examples_seen": seen},
        destination / "last.pt",
    )
    return {"examples_seen": seen, "loss": criterion.__class__.__name__, "checkpoint": str(destination / "last.pt")}


def load_biomedbert_checkpoint(path: str):
    """Resume a fine-tuned checkpoint. Not called in this round."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)
