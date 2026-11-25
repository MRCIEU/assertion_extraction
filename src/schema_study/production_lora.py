"""Production LoRA training.

`fit` is the training loop used for a Qwen run and for the CPU fixture. The
fixture supplies a small causal model and an explicit batch list. It does not
draw random tokens inside the loop, and it does not change the production
rank, alpha, dropout or learning rate.
"""

from __future__ import annotations

import json
import math
import random
import signal
import time
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
    """Tokenise one conversation. The prompt must be a token prefix, or the assistant mask must mark the completion."""
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    prompt_ids = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=True, add_generation_prompt=True, enable_thinking=False,
    ))
    full_ids = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False))
    boundary = None
    if full_ids[: len(prompt_ids)] == prompt_ids and len(full_ids) > len(prompt_ids):
        boundary = len(prompt_ids)
        method = "prefix"
    else:
        try:
            masked = tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True, enable_thinking=False,
            )
            mask = list(masked["assistant_masks"])
            full_ids = list(masked["input_ids"])
        except (TypeError, KeyError, ValueError):
            mask = []
        if mask and any(mask) and len(mask) == len(full_ids):
            boundary = mask.index(1)
            method = "assistant_mask"
        else:
            raise ValueError("The rendered prompt is not a prefix of the conversation and no assistant mask is available.")
    if boundary is None or boundary >= len(full_ids):
        raise ValueError("The assistant completion did not add a supervised token.")
    return {"input_ids": full_ids, "prompt_tokens": boundary, "pad_id": int(tokenizer.pad_token_id or 0), "boundary": method}


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
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        import torch.distributed as dist

        local = {"rank": dist.get_rank(), "python": state["python"], "numpy": state["numpy"], "torch": state["torch"]}
        if "cuda" in state:
            local["cuda"] = state["cuda"]
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        state["distributed"] = gathered
    return state


def _restore_rng(state: dict) -> None:
    chosen = state
    if state.get("distributed") and torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        chosen = next(item for item in state["distributed"] if item["rank"] == rank)
    random.setstate(chosen["python"])
    np.random.set_state(chosen["numpy"])
    torch.set_rng_state(chosen["torch"])
    if "cuda" in chosen and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(chosen["cuda"])


def _adapter_state(model: nn.Module) -> dict:
    """Collect adapter tensors. FSDP gathers a full state dict on rank 0 before the file is written."""
    if type(model).__name__ == "FullyShardedDataParallel":
        from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel, StateDictType

        config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
            state = model.state_dict()
    else:
        module = model.module if hasattr(model, "module") else model
        state = module.state_dict()
    return {name: value.detach().cpu().clone() for name, value in state.items() if "lora_" in name}


def save_checkpoint(directory: Path, package: dict) -> Path:
    """Publish one checkpoint after every rank has supplied its state. A second writer is rejected."""
    import os

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:
        torch.distributed.barrier()
    if int(os.environ.get("RANK", "0")) != 0:
        if distributed:
            torch.distributed.barrier()
        return directory / "adapter_checkpoint.pt"
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".writer.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("Another process is writing this checkpoint.") from exc
    destination = directory / "adapter_checkpoint.pt"
    temporary = directory / f".adapter_checkpoint.{os.getpid()}.tmp"
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        torch.save(package, temporary)
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError("The checkpoint was not published.")
        temporary.replace(destination)
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)
        if temporary.exists():
            temporary.unlink()
    if distributed:
        torch.distributed.barrier()
    return destination


def _is_fsdp(model: nn.Module) -> bool:
    return type(model).__name__ == "FullyShardedDataParallel"


def _optimizer_state(model: nn.Module, optimizer) -> dict:
    """Collect a full optimiser state. FSDP uses the process group rather than a local shard file."""
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig, FullyShardedDataParallel, StateDictType

        with FullyShardedDataParallel.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            return FullyShardedDataParallel.optim_state_dict(model, optimizer)
    return optimizer.state_dict()


def _load_optimizer_state(model: nn.Module, optimizer, state: dict) -> None:
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig, FullyShardedDataParallel, StateDictType

        with FullyShardedDataParallel.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            prepared = FullyShardedDataParallel.optim_state_dict_to_load(model, optimizer, state)
        optimizer.load_state_dict(prepared)
        return
    optimizer.load_state_dict(state)


def _package(model, optimizer, step: int, position: int, configuration: dict, fingerprints: dict, best: dict | None) -> dict:
    import os

    adapter = _adapter_state(model)
    if any("lora_" not in key for key in adapter):
        raise RuntimeError("A frozen base weight entered the adapter checkpoint.")
    recorded = dict(fingerprints or {})
    if configuration.get("base_revision"):
        recorded["base_revision"] = configuration["base_revision"]
    if configuration.get("tokenizer_revision"):
        recorded["tokenizer_revision"] = configuration["tokenizer_revision"]
    return {
        "adapter": adapter,
        "configuration": configuration,
        "optimizer": _optimizer_state(model, optimizer),
        "scheduler": {
            "name": "linear_decay_after_warmup",
            "step": step,
            "total_steps": int(configuration["total_steps"]),
            "warmup_ratio": configuration["warmup_ratio"],
            "base_learning_rate": configuration["learning_rate"],
        },
        "step": step,
        "position": position,
        "rank_position": {"rank": int(os.environ.get("RANK", "0")), "position": position},
        "data_partition": configuration.get("data_partition"),
        "epoch": int(configuration.get("epoch", 0)),
        "rng": _rng_state(),
        "fingerprints": recorded,
        "best": best,
        "base_revision": configuration.get("base_revision"),
        "tokenizer_revision": configuration.get("tokenizer_revision"),
        "full_base_weights_saved": False,
    }


def restore_into_new_model(saved: dict, model: nn.Module, optimizer, expected: dict | None = None, load_optimizer: bool = True) -> tuple[int, int]:
    """Load an adapter file into a newly constructed model. Missing adapter tensors are not filled with a new initialisation."""
    adapter = saved["adapter"]
    if any("lora_" not in name for name in adapter):
        raise RuntimeError("The checkpoint contains a parameter that is not an adapter weight.")
    required = [name for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_" in name]
    missing = [name for name in required if name not in adapter]
    if missing:
        raise RuntimeError("The checkpoint is missing adapter parameters: " + ", ".join(missing[:5]))
    if expected:
        for key in ("input", "configuration", "base_revision", "tokenizer_revision"):
            if expected.get(key) is not None and saved.get("fingerprints", {}).get(key) != expected.get(key):
                raise RuntimeError(f"The checkpoint {key} fingerprint does not match the task.")
        if expected.get("total_steps") is not None and int(saved.get("scheduler", {}).get("total_steps", -1)) != int(expected["total_steps"]):
            raise RuntimeError("The checkpoint belongs to a different training horizon.")
    current = model.state_dict()
    current.update({name: adapter[name] for name in required})
    model.load_state_dict(current)
    if load_optimizer:
        _load_optimizer_state(model, optimizer, saved["optimizer"])
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


def _apply_device_policy(model: nn.Module, configuration: dict) -> tuple[torch.device, dict]:
    """Place a production model on CUDA. FSDP shards from CPU memory and does not gather the full model first."""
    import os

    from schema_study.production_runtime import require_production_device

    policy = configuration.get("device_policy", "cpu_fixture")
    initialised = world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1
    record = require_production_device(policy, torch.cuda.is_available(), initialised if policy in {"ddp", "fsdp"} else 1)
    if policy == "cpu_fixture":
        return torch.device("cpu"), {**record, "model": model}
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local)
    if policy == "single_gpu":
        model = model.to(device)
    elif policy == "ddp":
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
    elif policy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, ShardingStrategy

        precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        model = FullyShardedDataParallel(
            model, device_id=device, mixed_precision=precision, sharding_strategy=ShardingStrategy.FULL_SHARD,
        )
    record["requested_gpus"] = int(configuration.get("requested_gpus", initialised))
    record["initialised_world"] = initialised if policy in {"ddp", "fsdp"} else 1
    return device, {**record, "model": model}


def export_peft_adapter(directory: Path, adapter: dict, configuration: dict) -> Path:
    """Write the selected adapter in the PEFT directory layout, separate from the resumable checkpoint."""
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "peft_type": "LORA",
        "r": configuration.get("rank"),
        "lora_alpha": configuration.get("alpha"),
        "lora_dropout": configuration.get("dropout"),
        "target_modules": configuration.get("target_modules"),
        "bias": "none",
        "base_model_name_or_path": configuration.get("base_revision"),
    }
    (directory / "adapter_config.json").write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    torch.save({name: tensor for name, tensor in adapter.items() if "lora_" in name}, directory / "adapter_model.bin")
    if not (directory / "adapter_config.json").is_file() or (directory / "adapter_model.bin").stat().st_size <= 0:
        raise RuntimeError("The inference adapter export was not published.")
    return directory


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
    score_fn=None,
    stop_after: int | None = None,
) -> dict:
    """Optimise adapter parameters. `steps` is the optimiser-step horizon of this segment."""
    configuration = dict(configuration or production_configuration())
    total = int(total_steps or steps)
    configuration["total_steps"] = total
    micro = int(configuration.get("micro_batch_size", 1))
    effective = int(configuration["effective_batch_size"])
    accumulate = accumulation_steps(micro, int(configuration.get("world_size", world_size())), effective)
    step = 0
    position = 0
    best = None
    if configuration.get("require_fingerprints"):
        for key in ("input", "configuration"):
            if not (fingerprints or {}).get(key):
                raise RuntimeError(f"The {key} fingerprint is missing. A null fingerprint is not compatibility.")
    if resume is None:
        attach_lora(model, configuration)
    else:
        if fresh is None:
            raise ValueError("Continuation must restore a newly constructed model.")
        model = fresh
        if not any(parameter.requires_grad and "lora_" in name for name, parameter in model.named_parameters()):
            attach_lora(model, configuration)
        step, position = restore_into_new_model(
            resume, model, torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=float(configuration["learning_rate"])),
            expected={**(fingerprints or {}), "total_steps": total}, load_optimizer=False,
        )
        saved_best = resume.get("best") or {}
        if saved_best.get("development_micro_f1") is not None:
            best = {
                "step": saved_best.get("step"),
                "development_micro_f1": saved_best.get("development_micro_f1"),
                "adapter": resume.get("best_adapter"),
            }
    device, placement_record = _apply_device_policy(model, configuration)
    model = placement_record["model"]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(configuration["learning_rate"]))
    if resume is not None:
        _load_optimizer_state(model, optimizer, resume["optimizer"])
    losses = []
    learning_rates = []
    consumed = []
    cursor = BatchCursor(batches)
    cursor.skip(position)
    stop = {"requested": False}

    def _handle(signum, frame) -> None:
        stop["requested"] = True

    previous = signal.getsignal(signal.SIGUSR1)
    signal.signal(signal.SIGUSR1, _handle)
    try:
        while step < steps:
            if stop["requested"] or (stop_after is not None and step >= stop_after):
                break
            learning_rate = learning_rate_at_step(step, total, float(configuration["learning_rate"]), float(configuration["warmup_ratio"]))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            learning_rates.append(learning_rate)
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            for _offset in range(accumulate):
                batch = cursor.next()
                if device.type == "cuda":
                    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
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
            if step == 1 and output_dir is not None and not configuration.get("skip_preflight_checkpoint"):
                technical = _package(model, optimizer, step, position, configuration, fingerprints or {}, None)
                save_checkpoint(output_dir, technical)
                reloaded = torch.load(output_dir / "adapter_checkpoint.pt", map_location="cpu", weights_only=False)
                if reloaded.get("full_base_weights_saved") is not False or not reloaded.get("adapter"):
                    raise RuntimeError("The first checkpoint is not an adapter checkpoint.")
                if any(not torch.isfinite(tensor).all() for tensor in reloaded["adapter"].values()):
                    raise RuntimeError("The first checkpoint contains a non-finite adapter value.")
                from schema_study.production_runtime import publish_scientific_status

                publish_scientific_status(output_dir, {
                    "technical_preflight": True,
                    "useful_computation_started": time.time(),
                    "model_loaded": True,
                    "device": str(device),
                    "optimiser_steps": 1,
                    "loss_finite": True,
                    "checkpoint_validated": True,
                    "monitoring_guarantees_completion": False,
                })
            interval = int(configuration.get("eval_interval") or total)
            if score_fn is not None and step % interval == 0:
                model.eval()
                metric = float(score_fn(model))
                model.train()
                if best is None or metric > float(best.get("development_micro_f1", -1)):
                    best = {"step": step, "development_micro_f1": metric, "adapter": _adapter_state(model)}
    finally:
        signal.signal(signal.SIGUSR1, previous)
    from schema_study.production_runtime import training_status

    selection_recorded = bool(best and "development_micro_f1" in best)
    status = training_status(step, total, selection_recorded)
    public_best = None if not best else {key: value for key, value in best.items() if key != "adapter"}
    package = _package(model, optimizer, step, position, configuration, fingerprints or {}, public_best)
    package["status"] = status
    package["device"] = str(device)
    package["placement"] = {key: value for key, value in placement_record.items() if key != "model"}
    if best and best.get("adapter"):
        package["best_adapter"] = best["adapter"]
    if output_dir is not None:
        save_checkpoint(output_dir, package)
        if best and best.get("adapter"):
            save_checkpoint(output_dir / "best", {**package, "adapter": best["adapter"], "role": "best"})
            export_peft_adapter(output_dir / "best_adapter", best["adapter"], configuration)
        from schema_study.production_runtime import publish_scientific_status

        started = time.time()
        existing = output_dir / "scientific_status.json"
        if existing.is_file():
            prior = json.loads(existing.read_text(encoding="utf-8"))
            if prior.get("useful_computation_started") is not None:
                started = prior["useful_computation_started"]
        publish_scientific_status(output_dir, {
            "scientific": status,
            "preflight_passed": step >= 1 and all(math.isfinite(value) for value in losses),
            "technical_preflight": step >= 1,
            "useful_computation_started": started,
            "selection_recorded": selection_recorded,
            "checkpoint_validated": (output_dir / "adapter_checkpoint.pt").is_file(),
            "monitoring_guarantees_completion": False,
            "steps": step,
            "total_steps": total,
        })
    return {
        "losses": losses,
        "learning_rates": learning_rates,
        "checkpoint": package,
        "consumed": consumed,
        "steps": step,
        "position": position,
        "accumulation": accumulate,
        "status": status,
        "completed": status == "training_completed",
        "best": {key: value for key, value in (best or {}).items() if key != "adapter"},
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


def prepare_process(requested_gpus: int) -> dict:
    """Bind this process to its rank. A requested GPU count is accepted only when it is the world size."""
    import os

    from schema_study.production_runtime import assert_runtime_world

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    assert_runtime_world(requested_gpus, world)
    if world > 1 and not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU training requires an initialised process group. This process will not treat the GPU request as a world size.")
        torch.distributed.init_process_group("nccl")
    if torch.cuda.is_available():
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    return {"rank": rank, "world": world, "device": str(device), "requested_gpus": requested_gpus}


def gpu_placements() -> dict:
    """Training placements. Inference device maps are not used for distributed training."""
    return {
        "qwen8b": {"precision": "bf16", "strategy": "single_process", "multi_gpu": "distributed_data_parallel", "gpus": [1, 2, 4], "default_gpus": 1},
        "qwen32b": {"precision": "bf16", "strategy": "fsdp_full_shard", "one_gpu": "single_process", "gpus": [1, 2, 4], "default_gpus": 2},
        "quantization": "none",
    }


def _gloo_rank(rank: int, world: int, directory: str, port: int) -> None:
    """CPU process-group check. This is not a GH200 or NCCL execution."""
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    import torch.distributed as dist

    from schema_study.production_runtime import rank0_generation

    dist.init_process_group("gloo", rank=rank, world_size=world)
    calls = {"n": 0}

    def generate():
        calls["n"] += 1
        return rank

    produced = rank0_generation(rank, generate, dist.barrier)
    value = torch.tensor([float(produced if produced is not None else -1)])
    dist.broadcast(value, src=0)
    Path(directory, f"rank{rank}.json").write_text(json.dumps({"generated": calls["n"], "broadcast": int(value.item())}) + "\n", encoding="utf-8")
    dist.destroy_process_group()


def cpu_collective_probe(directory: Path) -> dict:
    """Two CPU ranks enter the same barrier and receive rank 0's value."""
    import multiprocessing

    directory.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_gloo_rank, args=(rank, 2, str(directory), 29541)) for rank in (0, 1)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(120)
    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError("The CPU collective probe did not finish on every rank.")
    ranks = [json.loads(Path(directory, f"rank{rank}.json").read_text(encoding="utf-8")) for rank in (0, 1)]
    return {"ranks": ranks, "gh200": False, "backend": "gloo"}


def publish_prediction(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)
