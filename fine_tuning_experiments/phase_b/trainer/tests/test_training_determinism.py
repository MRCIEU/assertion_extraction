"""Guard against silent loss of training-loop determinism.

The full end-to-end scientific trainer depends on four RNG streams being
seeded from a single integer:

    random.seed(S)   numpy.random.seed(S)   torch.manual_seed(S)   os.environ[PYTHONHASHSEED] = str(S)

If any of those is dropped or a new RNG is introduced without seeding (the
typical regression), the same config + seed will produce different dev
F1 scores across runs, invalidating Phase B's seed-matched evaluation.

This test does not run the full trainer (too expensive for CI).  Instead
it pins two cheap invariants that together guarantee determinism of every
training-loop stochastic step:

  1. After `_set_all_seeds(S)`, four consecutive RNG draws
     (random, numpy, torch_cpu, torch_cuda-if-avail) are bit-identical
     on two invocations.
  2. `_OnlineCollator` fed the same positives + the same coll_rng seed
     produces bit-identical batches across two invocations (this covers
     shuffle + negative sampling + tokenization).

Run directly:
    python3.11 -m fine_tuning_experiments.phase_b.trainer.tests.test_training_determinism
"""
from __future__ import annotations

import random
import sys

import numpy as np
import torch

from fine_tuning_experiments.phase_b.trainer.scientific_trainer import (
    _OnlineCollator,
    _set_all_seeds,
)
from fine_tuning_experiments.phase_b.trainer.scientific_data import legal_endpoints


def _draw_once(seed: int) -> tuple:
    _set_all_seeds(seed)
    r = random.random()
    n = float(np.random.rand())
    t = torch.rand(3).tolist()
    c = torch.cuda.FloatTensor(3).normal_().tolist() if torch.cuda.is_available() else None
    return r, n, t, c


def _make_positives() -> list[dict]:
    # Shared entity inventory + two gold relations for two positives.
    ent = {
        "g1": {"text": "BRCA1", "mapped_label": "GENE"},
        "g2": {"text": "TP53", "mapped_label": "GENE"},
        "d1": {"text": "breast cancer", "mapped_label": "DISEASE"},
        "dr1": {"text": "tamoxifen", "mapped_label": "DRUG"},
        "dr2": {"text": "metformin", "mapped_label": "DRUG"},
    }
    id_list = list(ent)
    gold_pairs = {("g1", "d1"), ("dr1", "g1")}
    doc_text = "BRCA1 TP53 breast cancer tamoxifen metformin context."

    def make(h_id: str, t_id: str, label: str, sid: str) -> dict:
        return {
            "text": f"{ent[h_id]['text']} [ENT] {ent[t_id]['text']} [SEP] {doc_text}",
            "label": label,
            "head_label": ent[h_id]["mapped_label"],
            "tail_label": ent[t_id]["mapped_label"],
            "source_dataset": "synthetic",
            "sample_id": sid,
            "doc_id": "doc_syn",
            "_ent_by_id": ent,
            "_id_list": id_list,
            "_gold_pairs": gold_pairs,
            "_doc_text": doc_text,
        }

    return [
        make("g1", "d1", "GENE_DISEASE", "syn_001"),
        make("dr1", "g1", "DRUG_GENE_REGULATION", "syn_002"),
    ]


def _run_collator(seed: int) -> dict:
    label2id = {
        "ASSOCIATION_GENERAL": 0, "DRUG_DISEASE": 1, "DRUG_GENE_REGULATION": 2,
        "GENE_DISEASE": 3, "GENE_GENE_ASSOC": 4, "VARIANT_DISEASE": 5,
        "DRUG_VARIANT_ASSOC": 6, "__NEGATIVE__": 7,
    }

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")

    _set_all_seeds(seed)
    rng = random.Random(seed + 101)
    pair_filter = "spair_legal_endpoints"
    _ = legal_endpoints(pair_filter)  # validate name
    coll = _OnlineCollator(
        tokenizer=tok, label2id=label2id, max_length=64,
        pair_type_filter=pair_filter,
        negative_ratio=4.0, max_negatives_per_sample=64,
        source_weights={"synthetic": 1.0}, rng=rng, effective_batch_size=4,
    )
    positives = _make_positives()
    out = coll(positives)
    return {
        "input_ids": out["input_ids"].tolist(),
        "attention_mask": out["attention_mask"].tolist(),
        "labels": out["labels"].tolist(),
        "source_weight": out["source_weight"].tolist(),
    }


def run() -> int:
    failures: list[str] = []

    # ── Invariant 1: seeded RNG streams reproduce ─────────────────────
    a = _draw_once(42)
    b = _draw_once(42)
    if a != b:
        failures.append(f"RNG streams diverged under seed=42: {a} vs {b}")
    # Different seed -> different
    c = _draw_once(43)
    if a == c:
        failures.append("seeds 42 and 43 produced identical draws (RNG unseeded?)")

    # ── Invariant 2: collator produces byte-identical batches ─────────
    try:
        ba = _run_collator(7)
        bb = _run_collator(7)
    except Exception as exc:
        failures.append(f"collator run failed: {exc}")
        ba = bb = None

    if ba is not None and ba != bb:
        # Find the first diverging field for a readable error
        for key in ("labels", "source_weight", "input_ids", "attention_mask"):
            if ba[key] != bb[key]:
                failures.append(f"collator diverged at field {key!r}: first run != second run")
                break
    elif ba is not None:
        print(f"[ok] collator produced identical batches "
              f"(len={len(ba['labels'])}, labels={ba['labels']})")

    # Different seed -> different batch
    if ba is not None:
        try:
            bc = _run_collator(8)
            if ba == bc:
                failures.append("collator seeds 7 and 8 produced identical batches")
        except Exception as exc:
            failures.append(f"collator seed-8 run failed: {exc}")

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: RNG streams + collator are deterministic under fixed seeds.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
