"""BiomedBERT smoke training, resume, and FP32 prediction.

Torch is imported only inside a stage function, after the smoke permit is checked.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from schema_study.experiments import seed_streams, training_hyperparameters
from schema_study.inputs import SPECIAL_MARKERS, load_biomedbert_tokenizer
from schema_study.models.schedule import plan_optimizer_groups
from schema_study.schemas import load_schema
from schema_study.smoke import SMOKE_MODELS, SMOKE_TRAIN_SEED, assert_smoke_config

CODE_ROOT = Path("/home/b5ac/freddieyu.b5ac/project_1")
STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
MODEL_DIR = STORAGE / "models/microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/e1354b7a3a09615f6aba48dfad4b7a613eef7062/model_safetensors"
TOKENIZER_DIR = STORAGE / "models/microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/e1354b7a3a09615f6aba48dfad4b7a613eef7062/tokenizer"
SMOKE_ROOT = STORAGE / "studies/schema_granularity_v1/runs/smoke/round3_20261115"
MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


def task_config(task: str) -> dict:
    if task == "biored":
        dataset, entity, predicate, loss = "biored", "E2", "P4", "softmax_cross_entropy"
        version = "biored_candidates_v2"
    elif task == "drugprot":
        dataset, entity, predicate, loss = "drugprot", "native", "P4", "bce_with_logits"
        version = "drugprot_candidates_v1"
    else:
        raise ValueError(task)
    output = SMOKE_ROOT / f"{task}_biomedbert"
    return {
        "phase": "smoke",
        "study_phase": "smoke",
        "included_in_formal_analysis": False,
        "formal_execution_authorized": False,
        "model_id": MODEL_ID,
        "revision": SMOKE_MODELS[MODEL_ID],
        "splits": ["train", "dev"],
        "output_dir": str(output),
        "max_updates": 32,
        "dataset": dataset,
        "entity_level": entity,
        "predicate_level": predicate,
        "loss": loss,
        "data_version": version,
        "train_seed": SMOKE_TRAIN_SEED,
        "micro_batch_size": 8,
        "effective_batch_size": 16,
        "epochs": 2,
        "updates_per_epoch": 16,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_examples(task: str) -> tuple[list[dict], list[dict]]:
    samples = SMOKE_ROOT / "samples"
    texts = {row["candidate_id"]: row["text"] for row in _read_jsonl(samples / f"{task}_texts.jsonl")}
    rows = []
    for line in (samples / "smoke_candidates.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        corpus, split, _role, document_id, candidate_id, gold_status, gold_labels = line.split("\t")
        if corpus != task:
            continue
        rows.append(
            {
                "candidate_id": candidate_id,
                "document_id": document_id,
                "split": split,
                "gold_status": gold_status,
                "gold_labels": [label for label in gold_labels.split("|") if label],
                "text": texts[candidate_id],
            }
        )
    return [row for row in rows if row["split"] == "train"], [row for row in rows if row["split"] == "dev"]


def _classes(task: str):
    schema = load_schema(CODE_ROOT / "configs" / "schemas" / f"{task}.yaml")
    positive = list(schema.fine_ids)
    if task == "biored":
        return schema, positive + [schema.no_relation_id]
    return schema, positive


def _target(task: str, labels: list[str], classes: list[str], schema):
    import torch

    names = {display: stable for stable, display in schema.fine_names.items()}
    if task == "biored":
        if not labels:
            index = classes.index(schema.no_relation_id)
        else:
            index = classes.index(names[labels[0]])
        return torch.tensor(index, dtype=torch.long)
    target = torch.zeros(len(classes), dtype=torch.float)
    for label in labels:
        target[classes.index(names[label])] = 1.0
    return target


def _capture_rng(torch):
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict, torch) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _build(config: dict, tokenizer, classes: list[str]):
    import torch
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Dropout, Linear
    from transformers import AutoModel, get_linear_schedule_with_warmup

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stopping: this GPU does not support bf16. Precision was not changed.")
    loaded = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True, use_safetensors=True)
    encoder = loaded.bert if hasattr(loaded, "bert") else loaded
    before = encoder.get_input_embeddings().weight.shape[0]
    encoder.resize_token_embeddings(len(tokenizer))
    after = encoder.get_input_embeddings().weight.shape[0]
    if after != len(tokenizer) or after - before != len(SPECIAL_MARKERS):
        raise RuntimeError(f"Embedding resize mismatch: {before} -> {after}, tokenizer {len(tokenizer)}")
    streams = seed_streams(config["train_seed"], config["entity_level"], config["predicate_level"])
    generator = torch.Generator()
    generator.manual_seed(streams["marker_embedding_init"] % (2**31 - 1))
    with torch.no_grad():
        rows = torch.randn((len(SPECIAL_MARKERS), encoder.config.hidden_size), generator=generator)
        encoder.get_input_embeddings().weight[-len(SPECIAL_MARKERS):].copy_(rows * encoder.config.initializer_range)
    head = Linear(encoder.config.hidden_size, len(classes))
    generator.manual_seed(streams["classifier_head"] % (2**31 - 1))
    with torch.no_grad():
        head.weight.copy_(torch.randn(head.weight.shape, generator=generator) * encoder.config.initializer_range)
        head.bias.zero_()
    hyperparameters = training_hyperparameters()
    encoder.to("cuda")
    head.to("cuda")
    named = list(encoder.named_parameters()) + [("classifier." + name, value) for name, value in head.named_parameters()]
    decay = [value for name, value in named if "bias" not in name.lower() and "norm" not in name.lower()]
    no_decay = [value for name, value in named if "bias" in name.lower() or "norm" in name.lower()]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": hyperparameters["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=hyperparameters["learning_rate"],
        betas=tuple(hyperparameters["betas"]),
        eps=hyperparameters["epsilon"],
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(32 * hyperparameters["warmup_ratio"])), 32)
    criterion = CrossEntropyLoss() if config["loss"] == "softmax_cross_entropy" else BCEWithLogitsLoss()
    return encoder, head, optimizer, scheduler, criterion, streams


def _forward(encoder, head, tokenizer, batch, torch):
    encoded = tokenizer(
        [row["text"] for row in batch],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded = {key: value.to("cuda") for key, value in encoded.items() if key in {"input_ids", "attention_mask", "token_type_ids"}}
    output = encoder(**encoded)
    return head(output.last_hidden_state[:, 0])


def _predict(encoder, head, tokenizer, rows, classes, task, schema, torch):
    encoder.eval()
    head.eval()
    probabilities = []
    labels = []
    with torch.no_grad():
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8]
            logits = _forward(encoder, head, tokenizer, batch, torch).float()
            if task == "biored":
                score = torch.softmax(logits, dim=-1)
                chosen = score.argmax(dim=-1).tolist()
                for index in chosen:
                    label = classes[index]
                    labels.append([] if label == schema.no_relation_id else [label])
                probabilities.extend(score.cpu().tolist())
            else:
                score = torch.sigmoid(logits)
                probabilities.extend(score.cpu().tolist())
    return probabilities, labels


def _package(encoder, head, optimizer, scheduler, book: dict, rng, config, torch) -> dict:
    return {
        "encoder": {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()},
        "classifier": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "completed_epochs": book["completed_epochs"],
        "optimizer_updates": book["optimizer_updates"],
        "best_score": book["best_score"],
        "best_epoch": book["best_epoch"],
        "threshold": book["threshold"],
        "rng": rng,
        "data_order_seed": book["data_order_seed"],
        "order_rng_state": book.get("order_rng_state"),
        "amp": {"enabled": True, "dtype": "bf16"},
        "training_config": config,
        "data_version": config["data_version"],
        "dev_probabilities": book.get("dev_probabilities"),
        "dev_labels": book.get("dev_labels"),
    }


def run_stage(task: str, stage: str) -> None:
    config = task_config(task)
    assert_smoke_config(config)
    import torch

    output = Path(config["output_dir"])
    for name in ("checkpoints/best", "checkpoints/last", "predictions", "evaluation", "logs"):
        (output / name).mkdir(parents=True, exist_ok=True)
    tokenizer = load_biomedbert_tokenizer(TOKENIZER_DIR)
    expected = list(range(30522, 30522 + len(SPECIAL_MARKERS)))
    observed = [tokenizer.convert_tokens_to_ids(marker) for marker in SPECIAL_MARKERS]
    if observed != expected:
        raise RuntimeError("Marker ids differ from the accepted tokenizer.")
    schema, classes = _classes(task)
    train_rows, dev_rows = load_examples(task)
    if len(train_rows) != 256 or len(dev_rows) != 128:
        raise RuntimeError("Smoke example counts are not 256 train and 128 dev.")
    encoder, head, optimizer, scheduler, criterion, streams = _build(config, tokenizer, classes)
    order_rng = random.Random(streams["data_order"])
    book = {"optimizer_updates": 0, "best_score": None, "best_epoch": None, "threshold": None, "data_order_seed": streams["data_order"]}
    if stage == "epoch2":
        saved = torch.load(output / "checkpoints/last/checkpoint.pt", map_location="cpu", weights_only=False)
        if saved["optimizer_updates"] != 16:
            raise RuntimeError(f"Resume expected 16 updates, found {saved['optimizer_updates']}.")
        encoder.load_state_dict(saved["encoder"])
        head.load_state_dict(saved["classifier"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        _restore_rng(saved["rng"], torch)
        order_rng.setstate(saved["order_rng_state"])
        book.update(saved)
        (output / "resume_check.json").write_text(json.dumps({
            "loaded_updates": saved["optimizer_updates"],
            "loaded_epoch": saved["completed_epochs"],
            "scheduler_last_epoch": scheduler.last_epoch,
            "best_epoch": saved["best_epoch"],
            "threshold": saved["threshold"],
        }, indent=2), encoding="utf-8")
    probe = encoder.get_input_embeddings().weight.detach().float().clone()
    hyperparameters = training_hyperparameters()
    epoch = 1 if stage == "epoch1" else 2
    order = list(train_rows)
    order_rng.shuffle(order)
    groups = plan_optimizer_groups(len(order), 8, 16)
    if [group["n_examples"] for group in groups] != [16] * 16:
        raise RuntimeError("Smoke epoch did not form sixteen full updates.")
    encoder.train()
    head.train()
    losses = []
    for group in groups:
        optimizer.zero_grad(set_to_none=True)
        for micro in group["micro_batches"]:
            batch = order[micro["start"]:micro["start"] + micro["count"]]
            targets = torch.stack([_target(task, row["gold_labels"], classes, schema) for row in batch]).to("cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = _forward(encoder, head, tokenizer, batch, torch)
                loss = criterion(logits, targets if task == "drugprot" else targets.view(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("Loss is not finite.")
            (loss * (micro["count"] / group["n_examples"])).backward()
            losses.append(float(loss.detach().float()))
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), hyperparameters["gradient_clip_norm"])
        optimizer.step()
        scheduler.step()
        book["optimizer_updates"] += 1
        if book["optimizer_updates"] > 32:
            raise RuntimeError("Smoke update cap exceeded.")
    changed = torch.max(torch.abs(encoder.get_input_embeddings().weight.detach().float() - probe)).item()
    if changed != changed or changed == 0.0:
        raise RuntimeError("Embedding weights did not change by a finite amount.")
    probabilities, predicted = _predict(encoder, head, tokenizer, dev_rows, classes, task, schema, torch)
    threshold = None
    if task == "drugprot":
        from schema_study.evaluation import select_common_threshold

        names = {display: stable for stable, display in schema.fine_names.items()}
        gold_sets = [{names[label] for label in row["gold_labels"]} for row in dev_rows]
        probability_maps = [{classes[index]: row[index] for index in range(len(classes))} for row in probabilities]
        selected = select_common_threshold(gold_sets, probability_maps, classes)
        threshold = selected["threshold"]
        predicted = [[label for label, score in item.items() if score >= threshold] for item in probability_maps]
        score_value = selected["f1"]
    else:
        from schema_study.formal_eval import score_predictions

        predictions = [
            {"candidate_id": row["candidate_id"], "document_id": row["document_id"], "split": "dev", "output_status": "valid", "predicted_labels": labels, "illegal_labels": []}
            for row, labels in zip(dev_rows, predicted)
        ]
        scored = score_predictions(dev_rows, predictions, schema, "P4", allowed_splits={"dev"}, data_version=config["data_version"], input_version="smoke", prediction_version="smoke", mapping_version=schema.schema_id)
        score_value = scored["summary"]["f1"]
    book.update({
        "completed_epochs": epoch,
        "threshold": threshold,
        "dev_probabilities": probabilities,
        "dev_labels": predicted,
        "data_order_seed": streams["data_order"],
        "order_rng_state": order_rng.getstate(),
    })
    if book["best_score"] is None or score_value > book["best_score"] or (score_value == book["best_score"] and epoch < book["best_epoch"]):
        book["best_score"] = score_value
        book["best_epoch"] = epoch
        torch.save(_package(encoder, head, optimizer, scheduler, book, _capture_rng(torch), config, torch), output / "checkpoints/best/checkpoint.pt")
    torch.save(_package(encoder, head, optimizer, scheduler, book, _capture_rng(torch), config, torch), output / "checkpoints/last/checkpoint.pt")
    (output / "training_history.tsv").open("a", encoding="utf-8").write(f"{epoch}\t{book['optimizer_updates']}\t{sum(losses)/len(losses)}\t{score_value}\t{threshold}\n")
    usage = {
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "updates": book["optimizer_updates"],
        "parameter_abs_change": changed,
        "bf16": True,
        "stage": stage,
    }
    text = json.dumps(usage, indent=2)
    (output / "resource_usage.json").write_text(text, encoding="utf-8")
    (output / f"resource_usage_{stage}.json").write_text(text, encoding="utf-8")


def predict_best(task: str) -> None:
    config = task_config(task)
    assert_smoke_config(config)
    import torch

    output = Path(config["output_dir"])
    tokenizer = load_biomedbert_tokenizer(TOKENIZER_DIR)
    schema, classes = _classes(task)
    _train_rows, dev_rows = load_examples(task)
    encoder, head, _optimizer, _scheduler, _criterion, _streams = _build(config, tokenizer, classes)
    saved = torch.load(output / "checkpoints/best/checkpoint.pt", map_location="cpu", weights_only=False)
    encoder.load_state_dict(saved["encoder"])
    head.load_state_dict(saved["classifier"])
    probabilities, predicted = _predict(encoder, head, tokenizer, dev_rows, classes, task, schema, torch)
    if task == "drugprot":
        threshold = saved["threshold"]
        predicted = [[classes[index] for index, score in enumerate(row) if score >= threshold] for row in probabilities]
    reference = saved["dev_probabilities"]
    worst = 0.0
    mismatches = 0
    for left, right in zip(reference, probabilities):
        for actual, expected in zip(left, right):
            difference = abs(actual - expected)
            worst = max(worst, difference)
            if difference > 1e-6 + 1e-5 * abs(expected):
                mismatches += 1
    tolerant = mismatches == 0
    from schema_study.formal_eval import score_predictions
    from schema_study.predictions import rollup_predicted_labels

    predictions = [
        {"candidate_id": row["candidate_id"], "document_id": row["document_id"], "split": "dev", "output_status": "valid", "predicted_labels": labels, "illegal_labels": []}
        for row, labels in zip(dev_rows, predicted)
    ]
    p4 = score_predictions(dev_rows, predictions, schema, "P4", allowed_splits={"dev"}, data_version=config["data_version"], input_version="smoke", prediction_version="smoke_fp32", mapping_version=schema.schema_id)
    rolled = []
    for row in predictions:
        labels = rollup_predicted_labels(schema, row["predicted_labels"], source_level="P4", target_level="P2")
        rolled.append({**row, "predicted_labels": labels})
    p2 = score_predictions(dev_rows, rolled, schema, "P2", allowed_splits={"dev"}, data_version=config["data_version"], input_version="smoke", prediction_version="smoke_rollup_p2", mapping_version=schema.schema_id)
    report = {"max_abs_probability_difference": worst, "mismatched_probabilities": mismatches, "within_tolerance": tolerant, "p4": p4["summary"], "p2_rollup": p2["summary"], "synthetic": False, "study_phase": "smoke", "included_in_formal_analysis": False}
    (output / "evaluation/smoke_scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not tolerant:
        raise RuntimeError(f"FP32 reload differs from the saved development probabilities by {worst}.")


if __name__ == "__main__":
    import sys

    task, stage = sys.argv[1], sys.argv[2]
    if stage == "predict":
        predict_best(task)
    else:
        run_stage(task, stage)

