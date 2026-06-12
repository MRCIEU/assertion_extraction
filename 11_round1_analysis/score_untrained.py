"""Stage 1b: untrained-floor baselines (pretrained encoder + random classification head)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shared.benchmark_eval import build_biored_test_examples, evaluate_model_benchmark_f1
from shared.inference import score_model_on_candidates, write_scores_jsonl
from shared.metrics_calibration import calibration_for_scores
from shared.metrics_ranking import metrics_by_pair_type
from shared.models import MODELS, MODEL_BY_ID
from shared.pool_loader import load_primary_candidates
from shared.train_core import require_gpu

from .config import SCORES_DIR, UNTRAINED_COMPLETE, UNTRAINED_HEAD_SEED

UNTRAINED_PREFIX = "untrained"


def _log(msg: str) -> None:
    print(msg, flush=True)


def untrained_model_id(model_id: str) -> str:
    return f"{UNTRAINED_PREFIX}_{model_id}"


def untrained_score_jsonl(model_id: str) -> Path:
    return SCORES_DIR / untrained_model_id(model_id) / "scores.jsonl"


def untrained_marker_path(model_id: str) -> Path:
    return SCORES_DIR / untrained_model_id(model_id) / UNTRAINED_COMPLETE


def is_untrained_scored(model_id: str) -> bool:
    return untrained_marker_path(model_id).exists()


def count_untrained_scored(model_ids: list[str] | None = None) -> int:
    specs = MODELS if model_ids is None else [MODEL_BY_ID[m] for m in model_ids]
    return sum(1 for s in specs if is_untrained_scored(s.model_id))


def _load_untrained_model(hf_name: str, device: torch.device):
    """Pretrained weights with a freshly initialised binary classification head."""
    torch.manual_seed(UNTRAINED_HEAD_SEED)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_name, num_labels=2)
    model.to(device)
    model.eval()
    return model, tokenizer


def score_untrained_one(model_id: str, candidates: pd.DataFrame, test_examples: list[dict], force: bool = False) -> dict:
    spec = MODEL_BY_ID[model_id]
    marker = untrained_marker_path(model_id)
    out_path = untrained_score_jsonl(model_id)

    if marker.exists() and not force and out_path.exists():
        return json.loads(marker.read_text(encoding="utf-8"))

    device = require_gpu()
    run_id = untrained_model_id(model_id)
    _log(f"  UNTRAINED START {model_id} hf={spec.hf_name}")

    model, tokenizer = _load_untrained_model(spec.hf_name, device)
    scores = score_model_on_candidates(
        model,
        tokenizer,
        candidates,
        model_id=run_id,
        seed=-1,
        run_id=run_id,
    )
    write_scores_jsonl(scores, out_path)

    bench = evaluate_model_benchmark_f1(model, tokenizer, test_examples)
    kb_rows = metrics_by_pair_type(scores, run_id)
    kb_map = {row["pair_type"]: row for _, row in kb_rows.iterrows()}
    cal = calibration_for_scores(scores, run_id)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    payload = {
        "model_id": run_id,
        "base_model_id": model_id,
        "short_name": spec.short_name,
        "hf_name": spec.hf_name,
        "untrained_floor": True,
        "head_init_seed": UNTRAINED_HEAD_SEED,
        "benchmark_f1": float(bench["benchmark_f1"]),
        "benchmark_precision": float(bench["benchmark_precision"]),
        "benchmark_recall": float(bench["benchmark_recall"]),
        "kb_mrr_gene_drug": float(kb_map.get("gene-drug", {}).get("mrr", 0)),
        "kb_mrr_gene_disease": float(kb_map.get("gene-disease", {}).get("mrr", 0)),
        "kb_mrr_overall": float(kb_rows["mrr"].mean()) if not kb_rows.empty else 0.0,
        "ece": float(cal["ece"]),
        "scores_path": str(out_path),
        "n_candidates_scored": int(len(scores)),
        "n_test_examples": int(bench["n_test_examples"]),
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _log(
        f"  UNTRAINED DONE {model_id}: bench={payload['benchmark_f1']:.3f} "
        f"KB gd={payload['kb_mrr_gene_drug']:.3f} gdis={payload['kb_mrr_gene_disease']:.3f}"
    )
    return payload


def load_untrained_summary() -> pd.DataFrame:
    rows: list[dict] = []
    for spec in MODELS:
        mp = untrained_marker_path(spec.model_id)
        if mp.exists():
            rows.append(json.loads(mp.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def score_all_untrained(*, force: bool = False, model_ids: list[str] | None = None) -> pd.DataFrame:
    specs = MODELS if model_ids is None else [MODEL_BY_ID[m] for m in model_ids]
    _log(
        f"\n=== Round 1 untrained-floor scoring (stage 1b, GPU) ===\n"
        f"Encoders: {[s.model_id for s in specs]}\n"
        f"Already scored: {count_untrained_scored([s.model_id for s in specs])}/{len(specs)}\n"
    )

    candidates = load_primary_candidates()
    test_examples = build_biored_test_examples()
    _log(f"Frozen pool: {len(candidates)} candidates; BioRED test examples: {len(test_examples)}")

    rows: list[dict] = []
    n_skip, n_done = 0, 0
    for spec in specs:
        if is_untrained_scored(spec.model_id) and not force:
            rows.append(json.loads(untrained_marker_path(spec.model_id).read_text(encoding="utf-8")))
            n_skip += 1
            _log(f"  skip (scored): {spec.model_id}")
            continue
        try:
            row = score_untrained_one(spec.model_id, candidates, test_examples, force=force)
            rows.append(row)
            n_done += 1
        except Exception as exc:
            _log(f"  FAIL {spec.model_id}: {exc}")
            raise

    df = pd.DataFrame(rows)
    _log(
        f"\n=== Untrained-floor scoring complete ===\n"
        f"  Newly scored: {n_done}, skipped: {n_skip}, "
        f"markers on disk: {count_untrained_scored()}/{len(MODELS)}\n"
    )
    return df
