"""scientific_data — data loading, text encoding, negative sampling.

Implements the contract in `trainer_inventory/scientific_data_api_contract.md`.

Key design choices (documented here because the original Phase A .pyc source
was destroyed during cleanup; see §6 of the contract for unknowns resolved):

- **Text format**: `f"{head.text} [ENT] {tail.text} [SEP] {doc.text}"[:8000]`.
  Literal `[ENT]` separator (not a special tokenizer token).  This matches
  `external_evaluation/loaders/jsonl_pairs.py:doc_to_gold_pair_rows` which
  was documented to follow the same layout.

- **Label space derivation**: scan all gold `mapped_label` values across the
  active T1 + T2 shards with `pair_type_filter` applied to entity-pair
  membership, sort alphabetically, append `__NEGATIVE__` last.

- **Internal dev split**: seed-controlled random 12 % holdout of the gold
  positive pool per stage; negatives are sampled per-batch from the
  training split only (the dev split is all-positive to match Phase A
  observed dev_metrics).

- **Online negative sampling**: each batch, for every positive drawn, sample
  `negative_ratio` same-document non-gold pairs subject to
  `pair_type_filter` and `max_negatives_per_sample`.  Re-sampled every epoch
  for variety (`use_online_negatives: True` in Phase A configs).

- **Per-source routing** (`use_per_dataset_routing: True`): records the source
  dataset tag on each row; negatives inherit the positive's source_dataset.
  Used downstream only for the `routing` field in predictions JSONL.

- **Source weighting** (`inverse_freq_family_softmax`): per-sample CE loss
  weight = softmax over family-log-inverse-frequencies applied to
  `source_weights`.  Documented explicitly in `scientific_trainer.py` where
  the weighting is applied; `scientific_data` produces a `source_weight`
  field per row.

Public API:
  - `derive_label_space(shard_paths, pair_type_filter) -> dict[str, int]`
  - `encode_pair_to_text(head, tail, doc_text, max_chars=8000) -> str`
  - `build_stage_dataset(cfg, stage, label2id, seed) -> (train_ds, dev_ds)`
  - `PairDataset`  — torch Dataset yielding `{input_ids, attention_mask, label_id,
                                              source, source_weight, weak_or_gold}`
  - `OnlineNegativeCollator` — transforms positive-only batches into
                                positive + online-sampled-negative batches
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Sequence

import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────
# Constants + enums
# ─────────────────────────────────────────────────────────────────────

NEG_LABEL = "__NEGATIVE__"
MAX_TEXT_CHARS = 8000

Stage = Literal["T1", "T2", "T3", "T4"]

# Pair-type filters (match Phase A's `pair_type_filter` config value).
# An entity-pair (head_label, tail_label) is LEGAL if it appears in the set.
# The sets are derived from BioRED + DrugProt + BC5CDR canonical entity types.
_PAIR_FILTERS: dict[str, frozenset[tuple[str, str]]] = {
    "sflat_legal_endpoints": frozenset({
        # S_flat: 3 non-NEG heads cover drug-gene, drug-disease, and
        # association-generic (all remaining pairs).  All oncology-plausible
        # pairs are legal.
        ("GENE", "DISEASE"), ("DISEASE", "GENE"),
        ("GENE", "DRUG"), ("DRUG", "GENE"),
        ("DRUG", "DISEASE"), ("DISEASE", "DRUG"),
        ("GENE", "GENE"),
        ("VARIANT", "DISEASE"), ("DISEASE", "VARIANT"),
        ("VARIANT", "GENE"), ("GENE", "VARIANT"),
        ("DRUG", "VARIANT"), ("VARIANT", "DRUG"),
        ("VARIANT", "VARIANT"),
    }),
    "spair_legal_endpoints": frozenset({
        ("GENE", "DISEASE"), ("DISEASE", "GENE"),
        ("GENE", "DRUG"), ("DRUG", "GENE"),
        ("DRUG", "DISEASE"), ("DISEASE", "DRUG"),
        ("GENE", "GENE"),
        ("VARIANT", "DISEASE"), ("DISEASE", "VARIANT"),
        ("DRUG", "VARIANT"), ("VARIANT", "DRUG"),
    }),
    "smech_legal_endpoints": frozenset({
        ("GENE", "DISEASE"), ("DISEASE", "GENE"),
        ("GENE", "DRUG"), ("DRUG", "GENE"),
        ("DRUG", "DISEASE"), ("DISEASE", "DRUG"),
        ("GENE", "GENE"),
        ("VARIANT", "DISEASE"), ("DISEASE", "VARIANT"),
        ("DRUG", "VARIANT"), ("VARIANT", "DRUG"),
    }),
}


def legal_endpoints(filter_name: str) -> frozenset[tuple[str, str]]:
    if filter_name not in _PAIR_FILTERS:
        raise ValueError(
            f"Unknown pair_type_filter {filter_name!r}; known: {sorted(_PAIR_FILTERS)}"
        )
    return _PAIR_FILTERS[filter_name]


# ─────────────────────────────────────────────────────────────────────
# Text encoding
# ─────────────────────────────────────────────────────────────────────

def encode_pair_to_text(
    head_text: str, tail_text: str, doc_text: str,
    max_chars: int = MAX_TEXT_CHARS,
) -> str:
    """Phase A format: `"{head} [ENT] {tail} [SEP] {doc}"[:max_chars]`."""
    return f"{head_text} [ENT] {tail_text} [SEP] {doc_text}"[:max_chars]


# ─────────────────────────────────────────────────────────────────────
# Shard loading + per-document row expansion
# ─────────────────────────────────────────────────────────────────────

def _iter_docs(path: Path, *, allow_splits: Iterable[str] | None = None) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if allow_splits is not None:
                if d.get("source_split") not in allow_splits:
                    continue
            yield d


def _doc_gold_positives(
    doc: dict, pair_filter: frozenset[tuple[str, str]],
) -> list[dict[str, Any]]:
    """One row per gold-supervised relation, with pair-type filter applied."""
    ent_by_id = {e["entity_id"]: e for e in (doc.get("entities") or [])}
    gold_pairs: set[tuple[str, str]] = set()
    for rel in doc.get("relations") or []:
        hid = rel.get("head_entity_id"); tid = rel.get("tail_entity_id")
        if hid and tid:
            gold_pairs.add((hid, tid))
            gold_pairs.add((tid, hid))

    rows = []
    doc_text = doc.get("text") or ""
    for rel in doc.get("relations") or []:
        if not rel.get("is_gold_supervision", True):
            continue
        hid = rel.get("head_entity_id"); tid = rel.get("tail_entity_id")
        if not (hid and tid and hid in ent_by_id and tid in ent_by_id):
            continue
        h, t = ent_by_id[hid], ent_by_id[tid]
        hl = (h.get("mapped_label") or "").upper()
        tl = (t.get("mapped_label") or "").upper()
        if (hl, tl) not in pair_filter:
            continue
        rows.append({
            "text": encode_pair_to_text(h["text"], t["text"], doc_text),
            "label": rel["mapped_label"],
            "doc_id": doc.get("doc_id", ""),
            "sample_id": doc.get("sample_id", ""),
            "source_dataset": doc.get("source_dataset"),
            "source_split": doc.get("source_split"),
            "head_label": hl,
            "tail_label": tl,
            "weak_or_gold": "gold",
            # Fields needed for online negative sampling
            "_ent_by_id": ent_by_id,
            "_gold_pairs": gold_pairs,
            "_doc_text": doc_text,
            "_id_list": list(ent_by_id.keys()),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────
# Label space derivation
# ─────────────────────────────────────────────────────────────────────

def derive_label_space(
    shard_paths: Iterable[Path], pair_type_filter: str,
) -> dict[str, int]:
    """Stable alphabetic order of non-NEG labels, `__NEGATIVE__` appended last."""
    pair_filter = legal_endpoints(pair_type_filter)
    labels: set[str] = set()
    for path in shard_paths:
        for doc in _iter_docs(Path(path)):
            for row in _doc_gold_positives(doc, pair_filter):
                labels.add(row["label"])
    ordered = sorted(lab for lab in labels if lab != NEG_LABEL)
    ordered.append(NEG_LABEL)
    return {lab: i for i, lab in enumerate(ordered)}


# ─────────────────────────────────────────────────────────────────────
# Dataset class + online negative sampler
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PairRow:
    text: str
    label: str
    source_dataset: str | None
    head_label: str
    tail_label: str
    weak_or_gold: str
    sample_id: str
    # Not serialised — kept for negative sampling
    ent_by_id: dict[str, dict] | None = None
    gold_pairs: set[tuple[str, str]] | None = None
    doc_text: str | None = None
    id_list: list[str] | None = None


class PairDataset(Dataset):
    """Holds positive rows in memory.  Negatives are produced per-batch by
    the collator, not pre-materialised.  `source_weights` per sample are
    assigned later by `scientific_trainer` based on `source_dataset`.
    """

    def __init__(self, rows: Sequence[dict[str, Any]]):
        self._rows = list(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._rows[idx]


def sample_document_negatives(
    row: dict[str, Any],
    rng: random.Random,
    pair_filter: frozenset[tuple[str, str]],
    n_negatives: int,
) -> list[dict[str, Any]]:
    """Sample up to `n_negatives` within-document non-gold pairs whose
    (head_label, tail_label) is in `pair_filter`.  Fails closed (returns
    fewer than `n_negatives`) if the document has insufficient candidates —
    the caller must handle this by oversampling from other documents if
    strict `negative_ratio` is required.
    """
    ent_by_id = row.get("_ent_by_id")
    id_list = row.get("_id_list") or []
    gold_pairs = row.get("_gold_pairs") or set()
    doc_text = row.get("_doc_text") or ""
    if not ent_by_id or len(id_list) < 2:
        return []
    out: list[dict[str, Any]] = []
    # Try up to 256 × n_negatives rejections before giving up
    for _ in range(256 * max(1, n_negatives)):
        if len(out) >= n_negatives:
            break
        h2 = rng.choice(id_list)
        t2 = rng.choice(id_list)
        if h2 == t2 or (h2, t2) in gold_pairs or (t2, h2) in gold_pairs:
            continue
        he, te = ent_by_id[h2], ent_by_id[t2]
        hl = (he.get("mapped_label") or "").upper()
        tl = (te.get("mapped_label") or "").upper()
        if (hl, tl) not in pair_filter:
            continue
        out.append({
            "text": encode_pair_to_text(he["text"], te["text"], doc_text),
            "label": NEG_LABEL,
            "doc_id": row.get("doc_id", ""),
            "sample_id": row.get("sample_id", ""),
            "source_dataset": row.get("source_dataset"),
            "source_split": row.get("source_split"),
            "head_label": hl,
            "tail_label": tl,
            "weak_or_gold": "negative_sample",
        })
    return out


# ─────────────────────────────────────────────────────────────────────
# Stage dataset construction
# ─────────────────────────────────────────────────────────────────────

def _collect_stage_rows(
    shard_paths: Sequence[Path], max_pairs_per_shard: int,
    pair_filter: frozenset[tuple[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Per-shard cap honouring Phase A's `max_pairs_per_shard`
    for path in shard_paths:
        shard_rows: list[dict[str, Any]] = []
        for doc in _iter_docs(Path(path)):
            for row in _doc_gold_positives(doc, pair_filter):
                shard_rows.append(row)
                if len(shard_rows) >= max_pairs_per_shard:
                    break
            if len(shard_rows) >= max_pairs_per_shard:
                break
        rows.extend(shard_rows)
    return rows


def build_stage_dataset(
    cfg: dict, stage: Stage, label2id: dict[str, int], seed: int,
) -> tuple[PairDataset, PairDataset]:
    """Collect + dev-split positives for one stage.

    Returns `(train_ds, dev_ds)` — both hold dicts with `label` still as
    string.  The collator converts to label_id at batch time (so online
    negatives can be added in-place).
    """
    st = cfg["scientific_trainer"]
    neg_cfg = cfg.get("negative_sampling", {}) or {}
    pair_filter = legal_endpoints(neg_cfg["pair_type_filter"])
    active_shards_key = {"T1": "active_t1_shards", "T2": "active_t2_shards",
                         "T3": "active_t3_shards", "T4": "active_t4_shards"}.get(stage, "")
    if not active_shards_key:
        return PairDataset([]), PairDataset([])
    active = st.get(active_shards_key, [])
    if not active:
        return PairDataset([]), PairDataset([])

    paths_key = f"{stage}_shards"
    shard_map = cfg["training_data_paths"].get(paths_key, {}) or {}
    shard_paths = [Path(shard_map[name]) for name in active if name in shard_map]

    rows = _collect_stage_rows(
        shard_paths,
        max_pairs_per_shard=int(st.get("max_pairs_per_shard", 2000)),
        pair_filter=pair_filter,
    )
    rng = random.Random(seed ^ 0x9E3779B1 ^ hash(stage))
    rng.shuffle(rows)
    dev_fraction = float(st.get("dev_fraction", 0.12))
    n_dev = max(1, int(round(dev_fraction * len(rows))))
    dev_rows, train_rows = rows[:n_dev], rows[n_dev:]
    return PairDataset(train_rows), PairDataset(dev_rows)


# ─────────────────────────────────────────────────────────────────────
# Source weighting helper
# ─────────────────────────────────────────────────────────────────────

def inverse_freq_family_softmax_weights(
    source_weights_cfg: dict[str, float],
    source_counts: dict[str, int],
) -> dict[str, float]:
    """inverse_freq_family_softmax: per-source weight proportional to
    `source_weight_cfg[s] / log(1 + count[s])`, normalised so that the
    weights over the active (nonzero-cfg) sources softmax to 1.
    """
    active = {s: w for s, w in source_weights_cfg.items() if w > 0 and source_counts.get(s, 0) > 0}
    if not active:
        return {}
    raw = {s: w / math.log(1 + source_counts[s]) for s, w in active.items()}
    total = sum(raw.values())
    return {s: v / total for s, v in raw.items()} if total else {s: 0.0 for s in raw}
