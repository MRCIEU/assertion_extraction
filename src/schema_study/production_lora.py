"""Production LoRA training.

`fit` is the training loop used for a Qwen run and for the CPU fixture. The
fixture supplies a small causal model and an explicit batch list. It does not
draw random tokens inside the loop, and it does not change the production
rank, alpha, dropout or learning rate.
"""

from __future__ import annotations

import json
import random
import signal
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from schema_study.formal_optim import IGNORE_INDEX, assistant_supervision
from schema_study.learning_conditions import LORA_TARGET_MODULES, lora_configuration
from schema_study.models.schedule import learning_rate_at_step
from schema_study.production_assets import GENERATIVE

PRODUCTION = lora_configuration()


class FixtureCausalModel(nn.Module):
    """Small causal stand-in. Production tasks must not construct this class."""

    def __init__(self, vocab: int = 32, hidden: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        for name in LORA_TARGET_MODULES:
            setattr(self, name, nn.Linear(hidden, hidden, bias=False))
        self.output = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.embed(input_ids)
        for name in LORA_TARGET_MODULES:
            hidden = getattr(self, name)(hidden)
        return self.output(hidden)


def production_configuration(**overrides) -> dict:
    """The agreed BF16 LoRA configuration. Overrides are explicit and are not the defaults."""
    configuration = dict(PRODUCTION)
    configuration.update(overrides)
    if configuration["precision"] != "bf16" or configuration["quantization"] != "none":
        raise ValueError("The production configuration is BF16 LoRA, not QLoRA.")
    if configuration["rank"] != 16 or configuration["alpha"] != 32 or configuration["dropout"] != 0.05:
        if not overrides:
            raise ValueError("Production defaults must stay at rank 16, alpha 32 and dropout 0.05.")
    if configuration["learning_rate"] != 1e-4 and not overrides:
        raise ValueError("The production learning rate is 1e-4.")
    return configuration


def accumulation_steps(micro_batch: int, world_size: int, effective_batch: int = 16) -> int:
    if micro_batch <= 0 or world_size <= 0 or effective_batch % (micro_batch * world_size) != 0:
        raise ValueError("The micro-batch and world size must divide the global effective batch.")
    return effective_batch // (micro_batch * world_size)


def world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    return 1


def causal_next_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Predict each token from the preceding position. Labels are not compared in place."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Logits and labels must describe the same batch and length.")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    supervised = (shift_labels != IGNORE_INDEX).sum(dim=1)
    if int(supervised.min()) == 0:
        raise ValueError("A row has no supervised token under the next-token alignment.")
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def sequence_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Use a causal LM loss when the model returns one. Do not shift that loss again."""
    labels = batch["labels"]
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels)
    if hasattr(output, "loss") and output.loss is not None:
        return output.loss
    logits = output if torch.is_tensor(output) else output.logits
    return causal_next_token_loss(logits, labels)


def collate_rows(rows: list[dict]) -> dict[str, torch.Tensor]:
    """Pad on the right and build assistant labels. Every row must supervise at least one token."""
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full((len(rows), width), int(rows[0].get("pad_id", 0)), dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    prompt_tokens = []
    for index, row in enumerate(rows):
        tokens = torch.tensor(row["input_ids"], dtype=torch.long)
        input_ids[index, : len(tokens)] = tokens
        attention[index, : len(tokens)] = 1
        prompt_tokens.append(int(row["prompt_tokens"]))
    labels = assistant_supervision(input_ids, attention, prompt_tokens)
    shifted = labels[:, 1:]
    if int((shifted != IGNORE_INDEX).sum(dim=1).min()) == 0:
        raise ValueError("A row has no valid supervised completion.")
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels, "prompt_tokens": prompt_tokens}


def batches_from_records(records: list[dict], tokenizer, micro_batch: int):
    """Tokenise prepared conversations and yield production batches."""
    bucket = []
    for record in records:
        rendered = render_conversation(tokenizer, record["user"], record["assistant"])
        bucket.append(rendered)
        if len(bucket) == micro_batch:
            yield collate_rows(bucket)
            bucket = []
    if bucket:
        yield collate_rows(bucket)


def render_conversation(tokenizer, user: str, assistant: str) -> dict:
    """Tokenise one conversation with the model's own chat template."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("The assistant completion did not add a supervised token.")
    return {"input_ids": full_ids, "prompt_tokens": len(prompt_ids), "pad_id": tokenizer.pad_token_id or 0}


def attach_lora(model: nn.Module, configuration: dict | None = None) -> list[str]:
    """Attach a PEFT adapter. Only parameters whose names contain lora_ stay trainable."""
    from peft import LoraConfig, inject_adapter_in_model

    configuration = configuration or production_configuration()
    config = LoraConfig(
        r=int(configuration["rank"]),
        lora_alpha=int(configuration["alpha"]),
        lora_dropout=float(configuration["dropout"]),
        bias="none",
        target_modules=list(configuration.get("target_modules") or LORA_TARGET_MODULES),
    )
    inject_adapter_in_model(config, model)
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name
        if parameter.requires_grad:
            trainable.append(name)
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("PEFT did not leave only LoRA parameters trainable.")
    missing = [module for module in LORA_TARGET_MODULES if not any(module in name for name in trainable)]
    if missing:
        raise RuntimeError("The LoRA adapter is missing projection modules: " + ", ".join(missing))
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad and "lora_" not in name]
    if not frozen:
        raise RuntimeError("The base model has no frozen parameters.")
    return trainable


def load_causal_lm(model_name: str):
    """Load a pinned causal LM for an authorised run. No device map, and no fixture substitution."""
    if model_name not in {"qwen8b", "qwen32b", "gpt_oss_20b"}:
        raise KeyError(model_name)
    if model_name == "gpt_oss_20b":
        from schema_study.production_assets import architecture_status

        status = architecture_status(model_name)
        if status.get("weight_loading") != "requires_authorised_gpu_preflight":
            raise RuntimeError("GPT-OSS is not cleared for loading.")
    spec = GENERATIVE[model_name]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True, revision=spec["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["path"], local_files_only=True, revision=spec["revision"], torch_dtype=torch.bfloat16, device_map=None,
    )
    return model, tokenizer


def _rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _adapter_state(model: nn.Module) -> dict:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if "lora_" in name}


def save_checkpoint(directory: Path, package: dict) -> Path:
    destination = directory / "adapter_checkpoint.pt"
    temporary = destination.with_suffix(".pt.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, temporary)
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        raise RuntimeError("The checkpoint was not published.")
    temporary.replace(destination)
    return destination


def _package(model, optimizer, step: int, position: int, configuration: dict, fingerprints: dict, best: dict | None) -> dict:
    if any("lora_" not in key for key in _adapter_state(model)):
        raise RuntimeError("A frozen base weight entered the adapter checkpoint.")
    return {
        "adapter": _adapter_state(model),
        "configuration": configuration,
        "optimizer": optimizer.state_dict(),
        "scheduler": {
            "name": "linear_decay_after_warmup",
            "step": step,
            "total_steps": int(configuration["total_steps"]),
            "warmup_ratio": configuration["warmup_ratio"],
            "base_learning_rate": configuration["learning_rate"],
        },
        "step": step,
        "position": position,
        "epoch": int(configuration.get("epoch", 0)),
        "rng": _rng_state(),
        "fingerprints": fingerprints,
        "best": best,
        "base_revision": configuration.get("base_revision"),
        "tokenizer_revision": configuration.get("tokenizer_revision"),
        "full_base_weights_saved": False,
    }


def restore_into_new_model(saved: dict, model: nn.Module, optimizer) -> tuple[int, int]:
    """Load an adapter file into a newly constructed model. The old object is not reused."""
    adapter = saved["adapter"]
    if any("lora_" not in name for name in adapter):
        raise RuntimeError("The checkpoint contains a parameter that is not an adapter weight.")
    current = model.state_dict()
    current.update(adapter)
    model.load_state_dict(current)
    optimizer.load_state_dict(saved["optimizer"])
    _restore_rng(saved["rng"])
    return int(saved["step"]), int(saved["position"])


class BatchCursor:
    """Sequential batches. A resumed run skips completed optimiser positions and does not repeat them."""

    def __init__(self, batches) -> None:
        self.random = isinstance(batches, list)
        self.batches = batches
        self.iterator = None if self.random else iter(batches)
        self.index = 0

    def next(self) -> dict:
        if self.random:
            if self.index >= len(self.batches):
                raise IndexError("The data iterator ended before the optimiser step was complete.")
            item = self.batches[self.index]
        else:
            try:
                item = next(self.iterator)
            except StopIteration as exc:
                raise IndexError("The data iterator ended before the optimiser step was complete.") from exc
        self.index += 1
        return item

    def skip(self, position: int) -> None:
        while self.index < position:
            self.next()


def fit(
    model: nn.Module,
    batches: list[dict],
    *,
    configuration: dict | None = None,
    steps: int,
    total_steps: int | None = None,
    resume: dict | None = None,
    output_dir: Path | None = None,
    fingerprints: dict | None = None,
    fresh: nn.Module | None = None,
) -> dict:
    """Optimise adapter parameters. `steps` is the optimiser-step horizon of this segment."""
    configuration = dict(configuration or production_configuration())
    total = int(total_steps or steps)
    configuration["total_steps"] = total
    micro = int(configuration.get("micro_batch_size", 1))
    effective = int(configuration["effective_batch_size"])
    accumulate = accumulation_steps(micro, int(configuration.get("world_size", world_size())), effective)
    if resume is None:
        attach_lora(model, configuration)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(configuration["learning_rate"]))
    step = 0
    position = 0
    best = None
    if resume is not None:
        if fresh is None:
            raise ValueError("Continuation must restore a newly constructed model.")
        model = fresh
        if not any(parameter.requires_grad and "lora_" in name for name, parameter in model.named_parameters()):
            attach_lora(model, configuration)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=float(configuration["learning_rate"]))
        step, position = restore_into_new_model(resume, model, optimizer)
    losses = []
    learning_rates = []
    consumed = []
    cursor = BatchCursor(batches)
    cursor.skip(position)
    stop = {"requested": False}

    def _checkpoint() -> None:
        if output_dir is not None:
            package = _package(model, optimizer, step, position, configuration, fingerprints or {}, best)
            save_checkpoint(output_dir, package)

    def _handle(signum, frame) -> None:
        stop["requested"] = True
        _checkpoint()

    previous = signal.getsignal(signal.SIGUSR1)
    signal.signal(signal.SIGUSR1, _handle)
    try:
        while step < steps:
            if stop["requested"]:
                break
            learning_rate = learning_rate_at_step(step, total, float(configuration["learning_rate"]), float(configuration["warmup_ratio"]))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            learning_rates.append(learning_rate)
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            for _offset in range(accumulate):
                batch = cursor.next()
                position = cursor.index
                consumed.append([int(value) for value in batch["input_ids"].reshape(-1)[:8].tolist()])
                loss = sequence_loss(model, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError("Loss is not finite.")
                (loss / accumulate).backward()
                accumulated += float(loss.detach())
            if step == 0 and not any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0 for parameter in trainable):
                raise RuntimeError("LoRA parameters received no gradient.")
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(accumulated / accumulate)
            step += 1
            best = {"step": step, "loss": losses[-1]}
    finally:
        signal.signal(signal.SIGUSR1, previous)
    package = _package(model, optimizer, step, position, configuration, fingerprints or {}, best)
    if output_dir is not None:
        save_checkpoint(output_dir, package)
    return {
        "losses": losses,
        "learning_rates": learning_rates,
        "checkpoint": package,
        "consumed": consumed,
        "steps": step,
        "position": position,
        "accumulation": accumulate,
        "empirical_qwen_result": False,
        "model": model,
    }


def _fixture_batches(count: int) -> list[dict]:
    """Explicit fixture rows. The production loop does not call this function."""
    rows = []
    for index in range(count):
        token_ids = [1, 2, 3, 4, 5 + (index % 3), 6]
        rows.append(collate_rows([
            {"input_ids": token_ids, "prompt_tokens": 2, "pad_id": 0},
            {"input_ids": token_ids[:-1] + [7, 8], "prompt_tokens": 2, "pad_id": 0},
        ]))
    return rows


def compare_interrupted_run(directory: Path | None = None, steps: int = 4, interrupt_at: int = 2) -> dict:
    """Compare an uninterrupted run with a fresh model restored from the checkpoint file."""
    import tempfile

    own = directory is None
    if directory is None:
        directory = Path(tempfile.mkdtemp(prefix="lora-continuation-"))
    configuration = production_configuration(micro_batch_size=2, world_size=1)
    configuration["effective_batch_size"] = 16
    accumulate = accumulation_steps(2, 1, 16)
    batches = _fixture_batches(steps * accumulate)
    torch.manual_seed(13)
    random.seed(13)
    np.random.seed(13)
    uninterrupted = fit(FixtureCausalModel(), batches, configuration=configuration, steps=steps, total_steps=steps, output_dir=directory / "full")
    torch.manual_seed(13)
    random.seed(13)
    np.random.seed(13)
    partial = fit(FixtureCausalModel(), batches, configuration=configuration, steps=interrupt_at, total_steps=steps, output_dir=directory / "partial")
    saved = torch.load(directory / "partial" / "adapter_checkpoint.pt", map_location="cpu", weights_only=False)
    torch.manual_seed(13)
    resumed_model = FixtureCausalModel()
    resumed = fit(
        resumed_model, batches, configuration=configuration, steps=steps, total_steps=steps,
        resume=saved, fresh=resumed_model, output_dir=directory / "resumed",
    )
    left = uninterrupted["checkpoint"]["adapter"]
    right = resumed["checkpoint"]["adapter"]
    weights_match = left.keys() == right.keys() and all(torch.allclose(left[name], right[name], atol=1e-5) for name in left)
    losses_match = all(abs(a - b) <= 1e-5 for a, b in zip(uninterrupted["losses"], partial["losses"] + resumed["losses"]))
    consumed_match = uninterrupted["consumed"] == partial["consumed"] + resumed["consumed"]
    if own:
        pass
    return {
        "weights_match": weights_match,
        "losses_match": losses_match,
        "consumed_match": consumed_match,
        "steps": resumed["steps"],
        "position": resumed["position"],
        "accumulation": resumed["accumulation"],
        "full_base_weights_saved": saved["full_base_weights_saved"],
        "fresh_model": True,
        "empirical_qwen_result": False,
        "checkpoint": str(directory / "partial" / "adapter_checkpoint.pt"),
    }


def gpu_placements() -> dict:
    """Training placements. Inference device maps are not used for distributed training."""
    return {
        "qwen8b": {"precision": "bf16", "strategy": "distributed_data_parallel", "gpus": [1, 2, 4], "default_gpus": 1},
        "qwen32b": {"precision": "bf16", "strategy": "distributed_data_parallel", "gpus": [1, 2, 4], "default_gpus": 2},
        "quantization": "none",
    }


def publish_prediction(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)
