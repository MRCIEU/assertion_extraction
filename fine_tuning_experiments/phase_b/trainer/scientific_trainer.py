"""scientific_trainer — Phase B clean rewrite.

Implements the contract in `trainer_inventory/scientific_trainer_api_contract.md`.

Key design decisions (each choice marked with 📌 is a resolution of a §6
unknown from the contract; bridge equivalence run will verify):

📌 Stage optimiser reset (contract §6.1): **OPTIMIZER IS REINITIALISED** at
   the start of each stage.  Rationale: T2 loss tail in Phase A jumps by
   2+ orders of magnitude (0.005 → 0.2), consistent with fresh optimiser
   moments; also the simplest choice.

📌 Learning-rate scheduler (contract §6.2): HuggingFace linear schedule with
   0 warmup, decaying to 0 over each stage's `max_updates`.

📌 Gradient clipping (contract §6.3): `max_grad_norm = 1.0` (HF default).

📌 Mixed precision (contract §6.4): off (pure FP32).  Checkpoint sizes in
   Phase A (438 MB for PB-base) match FP32 state dict.

📌 Per-batch vs per-epoch negative re-sampling (contract §6 / scientific_data
   §6.2): **per-batch**.  `use_online_negatives: True` is explicit.

📌 Source-weighting formula (contract §6.7): per-sample CE weight =
   `source_weights[source]` × `inverse_freq_family_softmax` normalisation.
   See `scientific_data.inverse_freq_family_softmax_weights`.

📌 Data order (contract §6.5): per-epoch random shuffle of training pool,
   contiguous `batch_size`-sized chunks, online negatives injected at batch
   construction time.

📌 Early stopping (contract §6.9): dev_macro_f1 patience of
   `early_stopping_patience = 10` evaluations (= 640 steps at eval_every_steps
   = 64); eval is skipped entirely before `early_stopping_min_updates = 256`.

📌 Selection metric (contract config): `macro_f1` on the in-training dev
   split.  Saved as `best_selection_score` per stage in metrics JSON.

Public entry point:
  `run_scientific_training(cfg, exp_id, run_dir)` — matches Phase A signature.

Outputs (exhaustive, §3 of contract):
  run_dir/
    run_manifest.json
    training.log                                  (plain text, ~5 lines)
    checkpoints/{best,last,stage_t1_{best,end},stage_t2_{best,end}}.pt
    metrics/{metrics_standard,metrics_bundle,metrics_best_checkpoint,
             metrics_by_family,metrics_by_source,metrics_projected_slice,
             validation_history,calibration_summary}.json
    predictions/predictions_scientific.jsonl
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from fine_tuning_experiments.phase_b.trainer.scientific_data import (
    NEG_LABEL,
    PairDataset,
    build_stage_dataset,
    derive_label_space,
    inverse_freq_family_softmax_weights,
    legal_endpoints,
    sample_document_negatives,
)


# ─────────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────────

def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────────────
# Collator — online negative sampling + tokenisation
# ─────────────────────────────────────────────────────────────────────

class _OnlineCollator:
    """Emit batches of exactly `effective_batch_size` samples (positives +
    online-sampled negatives mixed).

    Phase A config sets `batch_size = 4` and `negative_ratio = 4.0`.  The
    natural interpretation — consistent with Phase A's observed step-64 dev
    accuracy of 0.836 (only possible if dev and train both contain a mix of
    positives and negatives, with NEG as the dominant class) — is that a
    "batch" of 4 is **4 total samples**: ~1 positive plus ~3 negatives on
    average.  We therefore configure the `DataLoader` to yield `positives_per_batch
    = max(1, batch_size // (1 + n_neg_per_pos))` positives per step, and the
    collator rounds out each step with `n_neg_per_pos` negatives per positive,
    then truncates to `batch_size`.
    """

    def __init__(
        self, tokenizer, label2id: dict[str, int], max_length: int,
        pair_type_filter: str, negative_ratio: float,
        max_negatives_per_sample: int, source_weights: dict[str, float],
        rng: random.Random, effective_batch_size: int,
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.pair_filter = legal_endpoints(pair_type_filter)
        self.n_neg_per_pos = max(1, min(max_negatives_per_sample, int(round(negative_ratio))))
        self.source_weights = source_weights
        self.rng = rng
        self.effective_batch_size = effective_batch_size

    def __call__(self, positives: list[dict]) -> dict[str, torch.Tensor]:
        rows: list[tuple[str, str, float]] = []  # (text, label_str, source_w)
        for pos in positives:
            rows.append((
                pos["text"], pos["label"],
                float(self.source_weights.get(pos.get("source_dataset"), 1.0)),
            ))
            negs = sample_document_negatives(
                pos, self.rng, self.pair_filter, n_negatives=self.n_neg_per_pos,
            )
            for n in negs:
                rows.append((
                    n["text"], NEG_LABEL,
                    float(self.source_weights.get(pos.get("source_dataset"), 1.0)),
                ))
        # Truncate / pad to effective_batch_size by shuffling and cutting.
        self.rng.shuffle(rows)
        rows = rows[:self.effective_batch_size]
        texts = [r[0] for r in rows]
        labels = [self.label2id[r[1]] for r in rows]
        source_ws = [r[2] for r in rows]

        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        enc["source_weight"] = torch.tensor(source_ws, dtype=torch.float32)
        return enc


# ─────────────────────────────────────────────────────────────────────
# Weighted CE helper
# ─────────────────────────────────────────────────────────────────────

def _weighted_ce_loss(logits: torch.Tensor, labels: torch.Tensor,
                      source_weight: torch.Tensor) -> torch.Tensor:
    per = F.cross_entropy(logits, labels, reduction="none")
    w = source_weight.to(per.device)
    return (per * w).mean()


# ─────────────────────────────────────────────────────────────────────
# Dev evaluation
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _dev_eval(
    model, tokenizer, dev_rows: list[dict], label2id: dict[str, int],
    max_length: int, batch_size: int, device: torch.device,
) -> dict[str, Any]:
    model.eval()
    preds: list[int] = []
    golds: list[int] = []
    for i in range(0, len(dev_rows), batch_size):
        chunk = dev_rows[i:i + batch_size]
        texts = [r["text"] for r in chunk]
        labs = [label2id[r["label"]] for r in chunk]
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        logits = model(**enc).logits
        batch_preds = logits.argmax(dim=-1).cpu().tolist()
        preds.extend(batch_preds)
        golds.extend(labs)
    if not preds:
        return {"dev_accuracy": 0.0, "dev_macro_f1": 0.0}
    # Match Phase A: macro-F1 over classes present in gold ∪ pred only
    # (scikit's default behaviour when `labels=` is not supplied).  Computing
    # macro over the full 4/8/13-class label space penalises early training
    # disproportionately (empty classes score F1=0) and produces a very
    # different trajectory — bridge equivalence smoke #2 had dev F1=0.27 at
    # step 256 where Phase A had F1=0.82 at step 64, almost entirely because
    # Phase A averaged over the 2-3 classes that the undertrained model was
    # actually predicting.
    return {
        "dev_accuracy": float(accuracy_score(golds, preds)),
        "dev_macro_f1": float(f1_score(golds, preds, average="macro", zero_division=0)),
    }


# ─────────────────────────────────────────────────────────────────────
# Training loop for a single stage
# ─────────────────────────────────────────────────────────────────────

@dataclass
class StageLog:
    stage: str
    steps: int = 0
    loss_hist: list[float] = field(default_factory=list)
    dev_metrics: list[dict[str, Any]] = field(default_factory=list)
    best_checkpoint: dict[str, Any] | None = None

    def loss_hist_tail(self, n: int = 8) -> list[float]:
        return self.loss_hist[-n:]


def _train_one_stage(
    stage: str, model, tokenizer, label2id: dict[str, int],
    train_ds: PairDataset, dev_rows: list[dict],
    cfg_st: dict, cfg_neg: dict, source_weights: dict[str, float],
    rng: random.Random, device: torch.device, ckpt_dir: Path,
    stage_best_pt_name: str, stage_end_pt_name: str, stage_rng_seed: int,
) -> StageLog:
    max_updates = int(cfg_st["max_updates"])
    batch_size = int(cfg_st["batch_size"])
    eval_every = int(cfg_st["eval_every_steps"])
    min_updates = int(cfg_st["early_stopping_min_updates"])
    patience = int(cfg_st["early_stopping_patience"])
    selection_metric = cfg_st.get("selection_metric", "macro_f1")
    lr = float(cfg_st["learning_rate"])

    coll_rng = random.Random(stage_rng_seed)
    n_neg_per_pos = max(1, min(
        int(cfg_neg.get("max_negatives_per_sample", 64)),
        int(round(float(cfg_neg.get("negative_ratio", 4.0)))),
    ))
    # positives_per_batch = ceil(batch_size / (1 + n_neg_per_pos)); ensure ≥ 1
    positives_per_batch = max(1, batch_size // (1 + n_neg_per_pos))
    collator = _OnlineCollator(
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=int(cfg_st["max_length"]),
        pair_type_filter=cfg_neg["pair_type_filter"],
        negative_ratio=float(cfg_neg.get("negative_ratio", 4.0)),
        max_negatives_per_sample=int(cfg_neg.get("max_negatives_per_sample", 64)),
        source_weights=source_weights,
        rng=coll_rng,
        effective_batch_size=batch_size,
    )
    dataloader = DataLoader(
        train_ds, batch_size=positives_per_batch, shuffle=True, collate_fn=collator,
        generator=torch.Generator().manual_seed(stage_rng_seed),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=max_updates,
    )

    log = StageLog(stage=stage)
    best_score: float = -1.0
    best_step: int = 0
    best_state: dict | None = None
    best_stopped_early = False
    evals_without_improvement = 0

    model.train()
    step = 0
    epoch = 0
    while step < max_updates:
        epoch += 1
        for batch in dataloader:
            if step >= max_updates:
                break
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            source_weight = batch.pop("source_weight")
            logits = model(**batch).logits
            loss = _weighted_ce_loss(logits, labels, source_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            log.loss_hist.append(float(loss.detach().cpu().item()))

            # Evaluate from the first `eval_every` boundary; `min_updates`
            # only gates early-stopping (not eval itself) — matches Phase A
            # which has dev entries from step 64 despite min_updates=256.
            if step % eval_every == 0:
                dev = _dev_eval(
                    model, tokenizer, dev_rows, label2id,
                    max_length=int(cfg_st["max_length"]), batch_size=batch_size,
                    device=device,
                )
                dev["step"] = step; dev["stage"] = stage
                log.dev_metrics.append(dev)
                score = dev.get(f"dev_{selection_metric}", 0.0)
                if score > best_score:
                    best_score = score
                    best_step = step
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    evals_without_improvement = 0
                else:
                    evals_without_improvement += 1
                # Early stopping is gated by `min_updates` (eval is not).
                if step >= min_updates and evals_without_improvement >= patience:
                    best_stopped_early = True
                    break
                model.train()

    log.steps = step

    # Stage-best checkpoint
    if best_state is not None:
        torch.save(
            {"model_state_dict": best_state, "label2id": label2id,
             "stage": stage, "model_name": cfg_st["model_name"],
             "best_checkpoint_meta": {
                 "stage": stage, "best_step": best_step,
                 "best_selection_score": best_score,
                 "selection_metric": selection_metric,
                 "stopped_early": best_stopped_early,
                 "checkpoint_file": stage_best_pt_name,
             }},
            ckpt_dir / stage_best_pt_name,
        )
    # Stage-end checkpoint
    torch.save(
        {"model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
         "label2id": label2id, "stage": stage,
         "model_name": cfg_st["model_name"],
         "best_checkpoint_meta": {
             "stage": stage, "best_step": step,
             "best_selection_score": best_score,
             "selection_metric": selection_metric,
             "stopped_early": best_stopped_early,
             "checkpoint_file": stage_end_pt_name,
         }},
        ckpt_dir / stage_end_pt_name,
    )

    log.best_checkpoint = {
        "stage": stage, "best_step": best_step,
        "best_selection_score": best_score,
        "selection_metric": selection_metric,
        "stopped_early": best_stopped_early,
        "checkpoint_file": stage_best_pt_name if best_state is not None else stage_end_pt_name,
    }

    # If stage-best wasn't set (no dev eval passed min_updates), use end as best
    if best_state is None:
        import shutil
        shutil.copy(ckpt_dir / stage_end_pt_name, ckpt_dir / stage_best_pt_name)

    return log


# ─────────────────────────────────────────────────────────────────────
# Predictions JSONL (in-training dev split)
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _write_predictions(
    model, tokenizer, dev_rows: list[dict], label2id: dict[str, int],
    max_length: int, batch_size: int, device: torch.device, out_path: Path,
) -> None:
    id2label = {v: k for k, v in label2id.items()}
    model.eval()
    with out_path.open("w") as f:
        for i in range(0, len(dev_rows), batch_size):
            chunk = dev_rows[i:i + batch_size]
            texts = [r["text"] for r in chunk]
            enc = tokenizer(texts, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            conf, pred_idx = probs.max(dim=-1)
            for j, r in enumerate(chunk):
                pid = int(pred_idx[j].item())
                f.write(json.dumps({
                    "sample_id": r.get("sample_id", ""),
                    "source_dataset": r.get("source_dataset"),
                    "routing": None,
                    "weak_source_shard": None,
                    "gold_relations": [{"mapped_label": r["label"]}],
                    "pred_relations": [{"mapped_label": id2label[pid]}],
                    "supervision": r.get("weak_or_gold", "gold"),
                    "confidence": float(conf[j].item()),
                    "weak_or_gold": r.get("weak_or_gold", "gold"),
                }) + "\n")


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

def run_scientific_training(cfg: dict, exp_id: str, run_dir: Path) -> None:
    """Top-level trainer dispatch.  Matches Phase A signature exactly.

    Produces all artifacts listed in §3 of scientific_trainer_api_contract.md.
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = run_dir / "metrics"; metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = run_dir / "predictions"; preds_dir.mkdir(parents=True, exist_ok=True)

    st = cfg["scientific_trainer"]
    seed = int(cfg.get("seed", 1))
    _set_all_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = st["model_name"]

    # ── Label space: derive from active T1 + T2 shards + T3 if enabled ────
    shard_paths: list[Path] = []
    for stage_key, active_key in [("T1_shards", "active_t1_shards"),
                                   ("T2_shards", "active_t2_shards")]:
        active = st.get(active_key, [])
        shard_map = cfg["training_data_paths"].get(stage_key, {}) or {}
        shard_paths.extend(Path(shard_map[s]) for s in active if s in shard_map)
    neg_cfg = cfg.get("negative_sampling", {}) or {}
    label2id = derive_label_space(shard_paths, neg_cfg["pair_type_filter"])
    labels_ordered = sorted(label2id, key=lambda k: label2id[k])
    num_labels = len(label2id)

    # ── Load model + tokenizer ────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    ).to(device)

    # ── Source weights (applied to per-sample CE loss) ────────────────────
    source_weights_cfg = cfg.get("source_weights", {}) or {}
    # Count active-source positives across shards for inverse-frequency weighting
    src_counts: Counter = Counter()
    for p in shard_paths:
        with p.open() as f:
            src_name = None
            for line in f:
                d = json.loads(line)
                s = d.get("source_dataset", "unknown")
                src_counts[s] += len(d.get("relations") or [])
    source_weights = inverse_freq_family_softmax_weights(
        source_weights_cfg, dict(src_counts),
    ) or {s: 1.0 for s in src_counts}

    # ── Determine schedule ────────────────────────────────────────────────
    # Accept both "T1_to_T2" (Phase A) and "T1_to_T2_staged" (Phase B) as the
    # two-stage schedule name.  "T1_flat" / "T1_only" / "T1_biored_only" are
    # single-stage schedules; everything else falls back to single-stage T1.
    schedule = cfg.get("schedule", "T1_to_T2_staged")
    staged_schedules = {"T1_to_T2", "T1_to_T2_staged"}
    stages_executed: list[str] = []

    # ── Stage T1 ──────────────────────────────────────────────────────────
    train_t1, dev_t1 = build_stage_dataset(cfg, "T1", label2id, seed)
    dev_rows_t1 = list(dev_t1._rows)  # noqa: SLF001
    log_t1 = _train_one_stage(
        "T1", model, tokenizer, label2id, train_t1, dev_rows_t1,
        cfg_st=st, cfg_neg=neg_cfg,
        source_weights=source_weights, rng=random.Random(seed + 1),
        device=device, ckpt_dir=ckpt_dir,
        stage_best_pt_name="stage_t1_best.pt",
        stage_end_pt_name="stage_t1_end.pt",
        stage_rng_seed=seed + 101,
    )
    stages_executed.append("T1")

    # ── Stage T2 (if staged schedule) ────────────────────────────────────
    log_t2: StageLog | None = None
    if schedule in staged_schedules:
        train_t2, dev_t2 = build_stage_dataset(cfg, "T2", label2id, seed)
        dev_rows_t2 = list(dev_t2._rows)  # noqa: SLF001
        # Re-initialise optimiser/scheduler implicitly (handled inside _train_one_stage)
        log_t2 = _train_one_stage(
            "T2", model, tokenizer, label2id, train_t2, dev_rows_t2,
            cfg_st=st, cfg_neg=neg_cfg,
            source_weights=source_weights, rng=random.Random(seed + 2),
            device=device, ckpt_dir=ckpt_dir,
            stage_best_pt_name="stage_t2_best.pt",
            stage_end_pt_name="stage_t2_end.pt",
            stage_rng_seed=seed + 202,
        )
        stages_executed.append("T2")

    # ── Overall-best checkpoint + last checkpoint ────────────────────────
    all_stage_logs = [log_t1] + ([log_t2] if log_t2 else [])
    best_stage = max(
        all_stage_logs,
        key=lambda L: (L.best_checkpoint or {}).get("best_selection_score", -1.0),
    )
    best_src = ckpt_dir / (best_stage.best_checkpoint["checkpoint_file"])
    import shutil
    shutil.copy(best_src, ckpt_dir / "best.pt")
    # last = current model state
    torch.save(
        {"model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
         "label2id": label2id,
         "stage": stages_executed[-1],
         "model_name": model_name,
         "best_checkpoint_meta": best_stage.best_checkpoint},
        ckpt_dir / "last.pt",
    )

    # ── Predictions on the stage-last dev split ──────────────────────────
    dev_for_pred = dev_rows_t2 if log_t2 is not None else dev_rows_t1
    _write_predictions(
        model, tokenizer, dev_for_pred, label2id,
        max_length=int(st["max_length"]),
        batch_size=int(st["batch_size"]),
        device=device,
        out_path=preds_dir / "predictions_scientific.jsonl",
    )

    # ── Metrics JSONs ────────────────────────────────────────────────────
    stage_logs_for_json = [
        {
            "stage": L.stage, "steps": L.steps,
            "loss_hist_tail": L.loss_hist_tail(8),
            "loss_hist_full": L.loss_hist,
            "dev_metrics": L.dev_metrics,
            "distill_note": False,
            "best_checkpoint": L.best_checkpoint,
        }
        for L in all_stage_logs
    ]
    metrics_standard = {
        "trainer": "scientific", "stages": stages_executed,
        "stage_logs": stage_logs_for_json,
        "num_labels": num_labels, "labels": labels_ordered,
    }
    metrics_best = {
        "selection_policy": st.get("selection_metric", "macro_f1"),
        "stages": [L.best_checkpoint for L in all_stage_logs],
    }
    validation_history = [d for L in all_stage_logs for d in L.dev_metrics]
    metrics_bundle = {
        "metrics_standard": metrics_standard,
        "metrics_best_checkpoint": metrics_best,
        "metrics_by_family": {},
        "metrics_by_source": {},
        "metrics_projected_slice": {},
        "calibration_summary": {"note": "scientific trainer — full calibration deferred"},
    }
    (metrics_dir / "metrics_standard.json").write_text(json.dumps(metrics_standard, indent=2))
    (metrics_dir / "metrics_best_checkpoint.json").write_text(json.dumps(metrics_best, indent=2))
    (metrics_dir / "metrics_bundle.json").write_text(json.dumps(metrics_bundle, indent=2))
    (metrics_dir / "metrics_by_family.json").write_text("{}")
    (metrics_dir / "metrics_by_source.json").write_text("{}")
    (metrics_dir / "metrics_projected_slice.json").write_text("{}")
    (metrics_dir / "validation_history.json").write_text(json.dumps(validation_history, indent=2))
    (metrics_dir / "calibration_summary.json").write_text(
        json.dumps({"note": "scientific trainer — full calibration deferred"})
    )

    # ── Training log ─────────────────────────────────────────────────────
    (run_dir / "training.log").write_text(
        "utc=" + datetime.now(timezone.utc).isoformat() + "\n"
        f"experiment_id={exp_id}\n"
        f"device={device.type}\n"
        f"stages={stages_executed}\n"
        "checkpoints=" + str([p.name for p in sorted(ckpt_dir.glob('*.pt'))]) + "\n"
    )

    # ── Manifest ─────────────────────────────────────────────────────────
    manifest = {
        "experiment_id": exp_id,
        "schema_id": cfg.get("schema_id"),
        "seed": seed,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": None,
        "ft_data_root": cfg.get("ft_data_root"),
        "scientific_trainer": {
            k: st.get(k) for k in [
                "enabled", "model_name", "max_length", "batch_size", "learning_rate",
                "max_pairs_per_shard", "eval_every_steps", "dev_fraction",
                "use_online_negatives", "active_t1_shards", "active_t2_shards",
                "active_t3_shards", "t4_max_lines", "t4_max_steps", "t4_max_length",
                "t4_learning_rate", "max_updates", "early_stopping_patience",
                "early_stopping_min_updates", "selection_metric",
            ]
        },
        "training_data_paths": cfg.get("training_data_paths", {}),
        "negative_sampling": cfg.get("negative_sampling", {}),
        "weak_supervision": cfg.get("weak_supervision", {}),
        "loss_mode": cfg.get("loss_mode"),
        "phase": cfg.get("phase"),
        "phase_a_metadata": cfg.get("phase_a_metadata"),
        "phase_b_metadata": cfg.get("phase_b_metadata"),
        "resolved": {
            "schedule": schedule,
            "stages_executed": stages_executed,
            "num_labels": num_labels,
            "checkpoints": [str(p) for p in sorted(ckpt_dir.glob("*.pt"))],
            "best_checkpoint_metric": st.get("selection_metric", "macro_f1"),
            "label2id": label2id,
        },
        "stub": False,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
