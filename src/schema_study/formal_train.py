"""Formal BiomedBERT training and FP32 prediction for one frozen condition.

This module is the only BiomedBERT entry for the formal run. It does not
enable the project-wide execution switch.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from schema_study.evaluation import select_common_threshold
from schema_study.experiments import seed_streams, training_hyperparameters
from schema_study.formal_eval import score_predictions
from schema_study.inputs import SPECIAL_MARKERS, load_biomedbert_tokenizer
from schema_study.process_record import process_directory, write_process_json
from schema_study.schemas import load_schema

CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
ROOT = STORAGE / "studies/schema_granularity_v1/runs/formal/formal_20261115"
MODEL_DIR = STORAGE / "models/microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/e1354b7a3a09615f6aba48dfad4b7a613eef7062/model_safetensors"
TOKENIZER_DIR = STORAGE / "models/microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/e1354b7a3a09615f6aba48dfad4b7a613eef7062/tokenizer"
MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
REVISION = "e1354b7a3a09615f6aba48dfad4b7a613eef7062"
CONTEXTS = {
    "biored": STORAGE / "data/inputs/biored/target_anchored_v1__biomedbert_e1354b7a_proposed/contexts.jsonl",
    "bc8": STORAGE / "data/inputs/bc8/target_anchored_v1__biomedbert_e1354b7a_proposed/contexts.jsonl",
    "drugprot": STORAGE / "data/inputs/drugprot/target_anchored_v1__biomedbert_e1354b7a/contexts.jsonl",
}
CANDIDATES = {
    "biored": STORAGE / "data/prepared/biored/biored_candidates_v2/candidates.parquet",
    "bc8": STORAGE / "data/prepared/bc8/bc8_candidates_v2/candidates.parquet",
    "drugprot": STORAGE / "data/prepared/drugprot/drugprot_candidates_v1/candidates.parquet",
}
VERSIONS = {"biored": "biored_candidates_v2", "bc8": "bc8_candidates_v2", "drugprot": "drugprot_candidates_v1"}


def _output(dataset: str, entity: str, predicate: str, seed: int) -> Path:
    path = ROOT / "train" / dataset / entity / predicate / f"seed{seed}"
    if ROOT not in path.parents and path != ROOT:
        raise RuntimeError("Refusing to write outside the formal run.")
    return path


def _candidates(dataset: str, splits: set[str]) -> dict[str, dict]:
    table = ds.dataset(CANDIDATES[dataset], format="parquet")
    found = {}
    columns = ["candidate_id", "document_id", "split", "gold_labels"]
    for batch in table.to_batches(columns=columns):
        for row in batch.to_pylist():
            if row["split"] not in splits:
                continue
            found[row["candidate_id"]] = {
                "candidate_id": row["candidate_id"],
                "document_id": str(row["document_id"]),
                "split": row["split"],
                "gold_labels": list(row["gold_labels"] or []),
            }
    return found


def _texts(dataset: str, entity: str, wanted: set[str], split_names: set[str]) -> dict[str, str]:
    found = {}
    with CONTEXTS[dataset].open(encoding="utf-8") as handle:
        for line in handle:
            if not any(f'"split": "{name}"' in line for name in split_names):
                continue
            row = json.loads(line)
            if row.get("split") not in split_names or row["candidate_id"] not in wanted:
                continue
            found[row["candidate_id"]] = row["views"][entity]
            if len(found) == len(wanted):
                break
    return found


def _encode(tokenizer, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    lengths: list[int] = []
    pieces: list[list[int]] = []
    for start in range(0, len(rows), 4096):
        chunk = rows[start:start + 4096]
        encoded = tokenizer([row["text"] for row in chunk], truncation=False, padding=False)
        for ids in encoded["input_ids"]:
            pieces.append(ids)
            lengths.append(len(ids))
    too_long = [rows[index]["candidate_id"] for index, length in enumerate(lengths) if length > 512]
    if too_long:
        raise RuntimeError(f"{len(too_long)} accepted contexts exceed 512 tokens. They were not truncated.")
    for row in rows:
        row.pop("text", None)
    width = max(lengths) if lengths else 0
    ids = np.zeros((len(rows), width), dtype=np.uint16)
    for index, tokens in enumerate(pieces):
        ids[index, :len(tokens)] = np.asarray(tokens, dtype=np.uint16)
    return ids, np.asarray(lengths, dtype=np.int32)


def _build_rows(dataset: str, entity: str, predicate: str, splits: set[str]) -> tuple[list[dict], list[str], object]:
    from schema_study.formal_eval import gold_groups

    schema_name = "biored" if dataset in {"biored", "bc8"} else "drugprot"
    schema = load_schema(CODE / "configs/schemas" / f"{schema_name}.yaml")
    population = _candidates(dataset, splits)
    texts = _texts(dataset, entity, set(population), splits)
    missing = sorted(set(population) - set(texts))
    if missing:
        raise RuntimeError(f"{len(missing)} candidates have no accepted {entity} context.")
    classes = [label.stable_id for label in schema.level_labels(predicate)]
    if dataset != "drugprot":
        classes.append(schema.no_relation_id)
    rows = []
    multiple = []
    for candidate_id in sorted(population):
        item = population[candidate_id]
        mapped = gold_groups(schema, item["gold_labels"], predicate)
        if dataset != "drugprot" and len(mapped) > 1:
            multiple.append(candidate_id)
        item = {**item, "text": texts[candidate_id], "mapped_gold": mapped}
        rows.append(item)
    if multiple:
        raise RuntimeError(f"{len(multiple)} BioRED candidates map to more than one label at {predicate}.")
    return rows, classes, schema


def _targets(dataset: str, rows: list[dict], classes: list[str]) -> np.ndarray:
    if dataset == "drugprot":
        target = np.zeros((len(rows), len(classes)), dtype=np.float32)
        index = {label: position for position, label in enumerate(classes)}
        for row_index, row in enumerate(rows):
            for label in row["mapped_gold"]:
                target[row_index, index[label]] = 1.0
        return target
    target = np.empty(len(rows), dtype=np.int64)
    index = {label: position for position, label in enumerate(classes)}
    no_relation = classes[-1]
    for row_index, row in enumerate(rows):
        label = row["mapped_gold"][0] if row["mapped_gold"] else no_relation
        target[row_index] = index[label]
    return target


def _capture(torch) -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore(state: dict, torch) -> None:
    random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _model(tokenizer, classes: list[str], seed: int, entity: str, predicate: str, torch):
    from torch.nn import Dropout, Linear
    from transformers import AutoModel

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stopping: this GPU does not support bf16.")
    loaded = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True, use_safetensors=True)
    encoder = loaded.bert if hasattr(loaded, "bert") else loaded
    before = encoder.get_input_embeddings().weight.shape[0]
    encoder.resize_token_embeddings(len(tokenizer))
    after = encoder.get_input_embeddings().weight.shape[0]
    if after - before != len(SPECIAL_MARKERS):
        raise RuntimeError(f"Embedding resize mismatch: {before} -> {after}")
    streams = seed_streams(seed, entity, predicate)
    generator = torch.Generator()
    generator.manual_seed(streams["marker_embedding_init"] % (2**31 - 1))
    with torch.no_grad():
        rows = torch.randn((len(SPECIAL_MARKERS), encoder.config.hidden_size), generator=generator)
        encoder.get_input_embeddings().weight[-len(SPECIAL_MARKERS):].copy_(rows * encoder.config.initializer_range)
    dropout = Dropout(0.1)
    head = Linear(encoder.config.hidden_size, len(classes))
    generator.manual_seed(streams["classifier_head"] % (2**31 - 1))
    with torch.no_grad():
        head.weight.copy_(torch.randn(head.weight.shape, generator=generator) * encoder.config.initializer_range)
        head.bias.zero_()
    encoder.to("cuda")
    head.to("cuda")
    dropout.to("cuda")
    return encoder, dropout, head, streams


def _optimizer(encoder, head, torch):
    from transformers import get_linear_schedule_with_warmup

    hyperparameters = training_hyperparameters()
    named = list(encoder.named_parameters()) + [("classifier." + name, value) for name, value in head.named_parameters()]
    decay = [value for name, value in named if "bias" not in name.lower() and "norm" not in name.lower()]
    no_decay = [value for name, value in named if "bias" in name.lower() or "norm" in name.lower()]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": hyperparameters["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=hyperparameters["learning_rate"],
        betas=tuple(hyperparameters["betas"]),
        eps=hyperparameters["epsilon"],
    )
    return optimizer, hyperparameters


def _batch_tensors(ids: np.ndarray, lengths: np.ndarray, indexes: list[int], torch):
    width = int(lengths[indexes].max())
    chosen = ids[indexes, :width]
    tokens = torch.from_numpy(chosen.astype(np.int64, copy=False)).to("cuda")
    mask = (torch.arange(width, device="cuda").unsqueeze(0) < torch.tensor(lengths[indexes], device="cuda").unsqueeze(1)).long()
    token_type = torch.zeros_like(tokens)
    return {"input_ids": tokens, "attention_mask": mask, "token_type_ids": token_type}


def _logits(encoder, dropout, head, batch, torch, train: bool):
    output = encoder(**batch).last_hidden_state[:, 0]
    if train:
        output = dropout(output)
    return head(output)


def _score_split(encoder, dropout, head, ids, lengths, rows, classes, dataset, schema, predicate, threshold, torch) -> tuple[float, float | None, list[list[str]]]:
    encoder.eval()
    dropout.eval()
    head.eval()
    predicted: list[list[str]] = []
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            indexes = list(range(start, min(start + 32, len(rows))))
            batch = _batch_tensors(ids, lengths, indexes, torch)
            logits = _logits(encoder, dropout, head, batch, torch, False).float()
            if dataset == "drugprot":
                probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            else:
                probabilities_tensor = torch.softmax(logits, dim=-1)
                probabilities.extend(probabilities_tensor.cpu().tolist())
                chosen = probabilities_tensor.argmax(dim=-1).tolist()
                for index in chosen:
                    label = classes[index]
                    predicted.append([] if label == schema.no_relation_id else [label])
    selected_threshold = threshold
    if dataset == "drugprot":
        gold_sets = [set(row["mapped_gold"]) for row in rows]
        maps = [{classes[index]: row[index] for index in range(len(classes))} for row in probabilities]
        if threshold is None:
            selected = select_common_threshold(gold_sets, maps, classes)
            selected_threshold = selected["threshold"]
            metric = selected["f1"]
        else:
            selected_threshold = threshold
            metric = None
        predicted = [[label for label, score in item.items() if score >= selected_threshold] for item in maps]
        if metric is None:
            from schema_study.evaluation import summarize_counts
            scored_rows = [{"gold": gold, "predicted": labels, "status": "valid", "illegal_labels": []} for gold, labels in zip(gold_sets, predicted)]
            metric = summarize_counts(scored_rows, classes)["f1"]
        return metric, selected_threshold, predicted, maps
    predictions = [
        {"candidate_id": row["candidate_id"], "document_id": row["document_id"], "split": row["split"], "output_status": "valid", "predicted_labels": labels, "illegal_labels": []}
        for row, labels in zip(rows, predicted)
    ]
    scored = score_predictions(rows, predictions, schema, predicate, allowed_splits={row["split"] for row in rows}, data_version=VERSIONS[dataset], input_version="target_anchored_v1", prediction_version="formal", mapping_version=schema.schema_id)
    return scored["summary"]["f1"], None, predicted, probabilities


def train(dataset: str, entity: str, predicate: str, seed: int) -> None:
    import torch
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
    from transformers import get_linear_schedule_with_warmup

    output = _output(dataset, entity, predicate, seed)
    output.mkdir(parents=True, exist_ok=True)
    last_path = output / "checkpoints/last/checkpoint.pt"
    if last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("completed_epochs", 0) >= 10:
            _record(output, "train", {"model_called": False, "reason": "training_already_complete", "calls": 0})
            return
    tokenizer = load_biomedbert_tokenizer(str(TOKENIZER_DIR))
    observed = [tokenizer.convert_tokens_to_ids(marker) for marker in SPECIAL_MARKERS]
    if observed != list(range(30522, 30522 + len(SPECIAL_MARKERS))):
        raise RuntimeError("Marker ids differ from the accepted tokenizer.")
    rows, classes, schema = _build_rows(dataset, entity, predicate, {"train", "dev"})
    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    train_ids, train_lengths = _encode(tokenizer, train_rows)
    dev_ids, dev_lengths = _encode(tokenizer, dev_rows)
    train_target = _targets(dataset, train_rows, classes)
    encoder, dropout, head, streams = _model(tokenizer, classes, seed, entity, predicate, torch)
    optimizer, hyperparameters = _optimizer(encoder, head, torch)
    steps_per_epoch = (len(train_rows) + 15) // 16
    total_steps = steps_per_epoch * hyperparameters["max_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(total_steps * hyperparameters["warmup_ratio"])), total_steps)
    criterion = BCEWithLogitsLoss() if dataset == "drugprot" else CrossEntropyLoss()
    order_rng = random.Random(streams["data_order"])
    book = {"completed_epochs": 0, "optimizer_updates": 0, "best_score": None, "best_epoch": None, "threshold": None}
    if last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(saved["encoder"])
        dropout.load_state_dict(saved["dropout"])
        head.load_state_dict(saved["classifier"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        _restore(saved["rng"], torch)
        order_rng.setstate(saved["order_rng_state"])
        book.update({key: saved[key] for key in ("completed_epochs", "optimizer_updates", "best_score", "best_epoch", "threshold")})
    history = output / "training_history.jsonl"
    while book["completed_epochs"] < hyperparameters["max_epochs"]:
        epoch = book["completed_epochs"] + 1
        order = list(range(len(train_rows)))
        order_rng.shuffle(order)
        encoder.train()
        dropout.train()
        head.train()
        started = time.perf_counter()
        losses = []
        for start in range(0, len(order), 16):
            indexes = order[start:start + 16]
            batch = _batch_tensors(train_ids, train_lengths, indexes, torch)
            target = torch.as_tensor(train_target[indexes], device="cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = _logits(encoder, dropout, head, batch, torch, True)
                loss = criterion(logits, target if dataset == "drugprot" else target.view(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("Loss is not finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), hyperparameters["gradient_clip_norm"])
            optimizer.step()
            scheduler.step()
            book["optimizer_updates"] += 1
            losses.append(float(loss.detach().float()))
        metric, threshold, _labels, _probabilities = _score_split(encoder, dropout, head, dev_ids, dev_lengths, dev_rows, classes, dataset, schema, predicate, None, torch)
        book["completed_epochs"] = epoch
        improved = book["best_score"] is None or metric > book["best_score"] or (metric == book["best_score"] and epoch < book["best_epoch"])
        if improved:
            book["best_score"] = metric
            book["best_epoch"] = epoch
            book["threshold"] = threshold
        package = _package(encoder, dropout, head, optimizer, scheduler, book, order_rng, torch)
        _save(output / "checkpoints/last/checkpoint.pt", package)
        if improved:
            _save(output / "checkpoints/best/checkpoint.pt", package)
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "epoch": epoch, "updates": book["optimizer_updates"], "mean_loss": sum(losses) / len(losses),
                "dev_f1": metric, "threshold": threshold, "seconds": time.perf_counter() - started, "best_epoch": book["best_epoch"],
            }) + "\n")
        print(f"epoch {epoch} dev_f1 {metric:.6f} seconds {time.perf_counter() - started:.1f}", flush=True)
    _record(output, "train", {"model_called": True, "epochs": book["completed_epochs"], "updates": book["optimizer_updates"], "best_epoch": book["best_epoch"], "best_score": book["best_score"]})


def _package(encoder, dropout, head, optimizer, scheduler, book, order_rng, torch) -> dict:
    return {
        "encoder": {key: value.detach().cpu() for key, value in encoder.state_dict().items()},
        "dropout": {key: value.detach().cpu() for key, value in dropout.state_dict().items()},
        "classifier": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "completed_epochs": book["completed_epochs"],
        "optimizer_updates": book["optimizer_updates"],
        "best_score": book["best_score"],
        "best_epoch": book["best_epoch"],
        "threshold": book["threshold"],
        "rng": _capture(torch),
        "order_rng_state": order_rng.getstate(),
        "amp": {"enabled": True, "dtype": "bf16"},
        "prediction_precision": "fp32",
    }


def _save(path: Path, package: dict) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(package, temporary)
    temporary.replace(path)


def _record(output: Path, kind: str, payload: dict) -> None:
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    attempt = kind
    suffix = 1
    while (output / "processes" / f"{job_id}_{attempt}").exists():
        suffix += 1
        attempt = f"{kind}-{suffix}"
    directory = process_directory(output, job_id, attempt)
    payload = {**payload, "slurm_job_id": job_id, "attempt_id": attempt}
    write_process_json(directory, "status.json", payload)
    write_process_json(directory, "resource_usage.json", payload)


def predict(dataset: str, entity: str, predicate: str, seed: int) -> None:
    import torch

    output = _output(dataset, entity, predicate, seed)
    prediction_path = output / "predictions.jsonl"
    if prediction_path.is_file() and (output / "prediction_complete.json").is_file():
        _record(output, "predict", {"model_called": False, "reason": "predictions_already_complete"})
        return
    tokenizer = load_biomedbert_tokenizer(str(TOKENIZER_DIR))
    schema_name = "biored" if dataset != "drugprot" else "drugprot"
    schema = load_schema(CODE / "configs/schemas" / f"{schema_name}.yaml")
    classes = [label.stable_id for label in schema.level_labels(predicate)]
    if dataset != "drugprot":
        classes.append(schema.no_relation_id)
    encoder, dropout, head, _streams = _model(tokenizer, classes, seed, entity, predicate, torch)
    saved = torch.load(output / "checkpoints/best/checkpoint.pt", map_location="cpu", weights_only=False)
    encoder.load_state_dict(saved["encoder"])
    dropout.load_state_dict(saved["dropout"])
    head.load_state_dict(saved["classifier"])
    targets = [("biored", "test"), ("bc8", "bc8")] if dataset == "biored" else [("drugprot", "eval")]
    with prediction_path.open("w", encoding="utf-8") as handle:
        for target_dataset, split in targets:
            rows, _target_classes, target_schema = _build_rows(target_dataset, entity, predicate, {split})
            if [label.stable_id for label in target_schema.level_labels(predicate)] + ([] if dataset == "drugprot" else [target_schema.no_relation_id]) != classes:
                raise RuntimeError("Prediction label space does not match the trained head.")
            ids, lengths = _encode(tokenizer, rows)
            _metric, _threshold, labels, probabilities = _score_split(
                encoder, dropout, head, ids, lengths, rows, classes, "drugprot" if dataset == "drugprot" else "biored",
                target_schema, predicate, saved["threshold"], torch,
            )
            for row, predicted, scores in zip(rows, labels, probabilities):
                handle.write(json.dumps({
                    "candidate_id": row["candidate_id"],
                    "document_id": row["document_id"],
                    "dataset": target_dataset,
                    "split": split,
                    "entity_level": entity,
                    "predicate_level": predicate,
                    "seed": seed,
                    "output_status": "valid",
                    "predicted_labels": predicted,
                    "label_probabilities": scores,
                    "illegal_labels": [],
                    "threshold": saved["threshold"],
                    "checkpoint": "best",
                    "best_epoch": saved["best_epoch"],
                }, ensure_ascii=False) + "\n")
    (output / "prediction_complete.json").write_text(json.dumps({"rows_written": True, "checkpoint": "best", "best_epoch": saved["best_epoch"]}) + "\n", encoding="utf-8")
    _record(output, "predict", {"model_called": True, "checkpoint": "best", "best_epoch": saved["best_epoch"]})


def apply_programme() -> str:
    """Point a new run at the matched E0–E7 inputs without moving the frozen 2026-11-15 run."""
    global ROOT
    programme = os.environ.get("SCHEMA_STUDY_PROGRAMME", "formal_20261115")
    if programme == "formal_matched_e0_e7":
        ROOT = STORAGE / "studies/schema_granularity_v1/runs/formal/formal_matched_e0_e7"
        CONTEXTS["biored"] = STORAGE / "data/inputs/biored/target_anchored_v2__biomedbert_e1354b7a_matched_e0_e7/contexts.jsonl"
        CONTEXTS["bc8"] = STORAGE / "data/inputs/bc8/target_anchored_v2__biomedbert_e1354b7a_matched_e0_e7/contexts.jsonl"
    return programme


def main() -> None:
    import sys
    from schema_study.paths import get_paths
    from schema_study.production_plan import gpu_authorised

    if not gpu_authorised(get_paths().study, get_paths().code_root):
        raise SystemExit(3)
    apply_programme()
    if len(sys.argv) != 6 or sys.argv[1] not in {"train", "predict"}:
        raise SystemExit("Usage: formal_train train|predict dataset entity predicate seed")
    action, dataset, entity, predicate, seed = sys.argv[1:]
    if action == "train":
        train(dataset, entity, predicate, int(seed))
    else:
        predict(dataset, entity, predicate, int(seed))


if __name__ == "__main__":
    main()
