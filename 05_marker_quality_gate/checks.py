"""Marker and span quality checks on repaired training and evaluation data."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_en = import_module("01_corpus_relevance.entity_normalization")
_matching = import_module("03_candidate_pool.matching")
split_sentences = _matching.split_sentences

from shared.benchmark_eval import build_biored_test_examples
from shared.constants import TRAIN_PAIR_TYPES
from shared.input_format import format_eval_input_with_method
from shared.marker_insert import bigbio_doc_text, bigbio_entity_span, format_marked_pair
from shared.pool_loader import load_primary_candidates
from shared.train_data import build_train_val_examples

from .config import (
    BASELINE_CIVIC_POS_EASY,
    BASELINE_CIVIC_POS_HARD,
    BASELINE_TRAIN_HEAD_MARKER_MISMATCH,
    BASELINE_TRAIN_SAME_SENTENCE_NATIVE,
    BASELINE_TRAIN_SAME_SENTENCE_STRING,
    SAME_SENT_TOLERANCE,
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    value: float | int | str
    detail: str
    before: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "detail": self.detail,
            "before": self.before,
        }


def _sentence_index(text: str, start: int, end: int) -> int | None:
    cursor = 0
    for idx, sentence in enumerate(split_sentences(text)):
        pos = text.find(sentence, cursor)
        if pos == -1:
            continue
        sent_end = pos + len(sentence)
        cursor = sent_end
        if start >= pos and end <= sent_end:
            return idx
    return None


def _same_sentence(text: str, head: tuple[int, int], tail: tuple[int, int]) -> bool | None:
    hi = _sentence_index(text, head[0], head[1])
    ti = _sentence_index(text, tail[0], tail[1])
    if hi is None or ti is None:
        return None
    return hi == ti


def _string_first_same_sentence(text: str, head_surf: str, tail_surf: str) -> bool | None:
    if not head_surf or not tail_surf or head_surf not in text or tail_surf not in text:
        return None
    h0 = text.index(head_surf)
    t0 = text.index(tail_surf)
    return _same_sentence(text, (h0, h0 + len(head_surf)), (t0, t0 + len(tail_surf)))


def _count_multi_mention(text: str, surface: str) -> int:
    if not surface:
        return 0
    return text.count(surface)


def _scan_corpus_positives() -> pd.DataFrame:
    rows: list[dict] = []
    for corpus, split in [
        ("biored", "train"),
        ("drugprot", "train"),
        ("biored", "validation"),
        ("drugprot", "validation"),
    ]:
        ds = load_dataset(
            "bigbio/biored" if corpus == "biored" else "bigbio/drugprot",
            "biored_bigbio_kb" if corpus == "biored" else "drugprot_bigbio_kb",
            trust_remote_code=True,
        )
        for doc in ds[split]:
            text = bigbio_doc_text(doc)
            if not text:
                continue
            entities = {e["id"]: e for e in doc.get("entities") or []}
            seen: set[tuple[str, str, str]] = set()
            for rel in doc.get("relations") or []:
                e1 = entities.get(rel.get("arg1_id"))
                e2 = entities.get(rel.get("arg2_id"))
                if not e1 or not e2:
                    continue
                pt = _en.civic_pair_type(e1.get("type", ""), e2.get("type", ""))
                if pt not in TRAIN_PAIR_TYPES:
                    continue
                if _en.normalize_entity_type(e1["type"]) == "gene":
                    head_ent, tail_ent = e1, e2
                elif _en.normalize_entity_type(e2["type"]) == "gene":
                    head_ent, tail_ent = e2, e1
                else:
                    head_ent, tail_ent = e1, e2
                key = (head_ent["id"], tail_ent["id"], pt)
                if key in seen:
                    continue
                seen.add(key)
                head_span = bigbio_entity_span(head_ent)
                tail_span = bigbio_entity_span(tail_ent)
                if not head_span or not tail_span:
                    continue
                marked, method, meta = format_marked_pair(text, head_ent, tail_ent)
                same_native = _same_sentence(text, (head_span[0], head_span[1]), (tail_span[0], tail_span[1]))
                hs, ts = head_span[2], tail_span[2]
                same_string = _string_first_same_sentence(text, hs, ts)
                rows.append(
                    {
                        "corpus": corpus,
                        "split": split,
                        "pair_type": pt,
                        "marker_method": method,
                        "same_sentence_native": same_native,
                        "same_sentence_string_first": same_string,
                        "head_multi_mention": _count_multi_mention(text, hs) > 1,
                        "tail_multi_mention": _count_multi_mention(text, ts) > 1,
                        "marker_on_native_offset": method == "offset",
                        "head_offset": head_span[0],
                        "head_end": head_span[1],
                        "tail_offset": tail_span[0],
                        "tail_end": tail_span[1],
                        "marked_text": marked,
                        "plain_text": text,
                        **meta,
                    }
                )
    return pd.DataFrame(rows)


def _marker_wraps_native_span(row: pd.Series) -> bool:
    if row["marker_method"] != "offset":
        return False
    text = row["plain_text"]
    marked = row["marked_text"]
    h0, h1 = int(row["head_offset"]), int(row["head_end"])
    t0, t1 = int(row["tail_offset"]), int(row["tail_end"])
    expected_head = text[h0:h1]
    expected_tail = text[t0:t1]
    return (
        f"[E1]{expected_head}[/E1]" in marked
        and f"[E2]{expected_tail}[/E2]" in marked
    )


def check_training_corpus() -> tuple[list[CheckResult], pd.DataFrame]:
    df = _scan_corpus_positives()
    n = len(df)
    offset_rate = float(df["marker_on_native_offset"].mean()) if n else 0.0
    wrap_rate = float(df.apply(_marker_wraps_native_span, axis=1).mean()) if n else 0.0
    same_native = df["same_sentence_native"].dropna()
    same_rate = float(same_native.mean()) if len(same_native) else 0.0
    multi_rate = float((df["head_multi_mention"] | df["tail_multi_mention"]).mean()) if n else 0.0
    multi_with_offset = float(
        df.loc[df["head_multi_mention"] | df["tail_multi_mention"], "marker_on_native_offset"].mean()
    ) if n else 0.0

    checks = [
        CheckResult(
            name="training_offset_insertion_rate",
            passed=offset_rate >= 0.999,
            value=round(offset_rate, 4),
            detail=f"{offset_rate:.1%} of positives use native offset insertion ({n} relations)",
            before=f"{1 - BASELINE_TRAIN_HEAD_MARKER_MISMATCH:.1%} head alignment under string-match",
        ),
        CheckResult(
            name="training_marker_wraps_annotated_span",
            passed=wrap_rate >= 0.999,
            value=round(wrap_rate, 4),
            detail=f"{wrap_rate:.1%} of marked examples wrap the annotated entity spans",
            before=f"~{BASELINE_TRAIN_HEAD_MARKER_MISMATCH:.1%} head markers on wrong mention",
        ),
        CheckResult(
            name="training_same_sentence_native_rate",
            passed=abs(same_rate - BASELINE_TRAIN_SAME_SENTENCE_NATIVE) <= SAME_SENT_TOLERANCE,
            value=round(same_rate, 4),
            detail=f"{same_rate:.1%} same-sentence under native offsets (target ~{BASELINE_TRAIN_SAME_SENTENCE_NATIVE:.1%})",
            before=f"{BASELINE_TRAIN_SAME_SENTENCE_STRING:.1%} same-sentence under string-match",
        ),
        CheckResult(
            name="training_multi_mention_offset_coverage",
            passed=multi_with_offset >= 0.999 if multi_rate > 0 else True,
            value=round(multi_with_offset, 4),
            detail=f"{multi_rate:.1%} positives have multi-mention entities; {multi_with_offset:.1%} of those use native offsets",
            before="string-match used first occurrence, not annotated mention",
        ),
    ]
    return checks, df


def check_benchmark_examples() -> tuple[list[CheckResult], pd.DataFrame]:
    examples = build_biored_test_examples()
    rows = []
    for ex in examples:
        if ex.get("label") != 1:
            continue
        rows.append({"label": ex["label"], "pair_type": ex.get("pair_type")})
    pos_n = len(rows)
    # Re-scan test split directly for method coverage
    biored = load_dataset("bigbio/biored", "biored_bigbio_kb", trust_remote_code=True)
    methods = []
    for doc in biored["test"]:
        text = bigbio_doc_text(doc)
        entities = {e["id"]: e for e in doc.get("entities") or []}
        seen: set[tuple[str, str, str]] = set()
        for rel in doc.get("relations") or []:
            e1 = entities.get(rel.get("arg1_id"))
            e2 = entities.get(rel.get("arg2_id"))
            if not e1 or not e2:
                continue
            pt = _en.civic_pair_type(e1.get("type", ""), e2.get("type", ""))
            if pt not in TRAIN_PAIR_TYPES:
                continue
            if _en.normalize_entity_type(e1["type"]) == "gene":
                head_ent, tail_ent = e1, e2
            elif _en.normalize_entity_type(e2["type"]) == "gene":
                head_ent, tail_ent = e2, e1
            else:
                head_ent, tail_ent = e1, e2
            key = (head_ent["id"], tail_ent["id"], pt)
            if key in seen:
                continue
            seen.add(key)
            _marked, method, _meta = format_marked_pair(text, head_ent, tail_ent)
            methods.append(method)
    offset_rate = float(sum(m == "offset" for m in methods) / len(methods)) if methods else 0.0
    checks = [
        CheckResult(
            name="benchmark_positive_examples_built",
            passed=pos_n > 0 and len(examples) > pos_n,
            value=pos_n,
            detail=f"{pos_n} BioRED test positives in {len(examples)} benchmark examples",
            before=None,
        ),
        CheckResult(
            name="benchmark_offset_insertion_rate",
            passed=offset_rate >= 0.999,
            value=round(offset_rate, 4),
            detail=f"{offset_rate:.1%} of BioRED test positives use offset insertion",
            before="string-match (same bug as training)",
        ),
    ]
    return checks, pd.DataFrame(rows)


def check_civic_pool() -> tuple[list[CheckResult], pd.DataFrame]:
    pool = load_primary_candidates()
    methods: list[str] = []
    subsets: list[str] = []
    _dist = import_module("shared.distance_analysis")
    enriched = _dist.enrich_with_proximity(pool)
    pos = enriched[enriched["label_civic_curated_positive"]]

    for _, row in pool.iterrows():
        _marked, method = format_eval_input_with_method(row)
        methods.append(method)

    method_counts = pd.Series(methods).value_counts()
    offset_rate = float(method_counts.get("offset", 0) / len(methods)) if len(methods) else 0.0
    fallback_rate = 1.0 - offset_rate

    for _, row in pos.iterrows():
        subsets.append(str(row.get("subset", "unknown")))

    sub_counts = pd.Series(subsets).value_counts(normalize=True) if subsets else pd.Series(dtype=float)
    easy_rate = float(sub_counts.get("easy_co_sentence", 0.0))
    hard_rate = float(sub_counts.get("hard_cross_sentence", 0.0))

    checks = [
        CheckResult(
            name="civic_offset_insertion_rate",
            passed=offset_rate >= 0.95,
            value=round(offset_rate, 4),
            detail=f"{offset_rate:.1%} of pool candidates use PubTator offsets ({fallback_rate:.1%} fallback)",
            before="already offset-first; unchanged pool",
        ),
        CheckResult(
            name="civic_positive_easy_hard_distribution",
            passed=True,
            value=round(easy_rate, 4),
            detail=(
                f"positive candidates: {easy_rate:.1%} easy, {hard_rate:.1%} hard "
                f"(prior {BASELINE_CIVIC_POS_EASY:.1%}/{BASELINE_CIVIC_POS_HARD:.1%})"
            ),
            before=f"{BASELINE_CIVIC_POS_EASY:.1%} easy / {BASELINE_CIVIC_POS_HARD:.1%} hard",
        ),
    ]
    return checks, pd.DataFrame({"method": methods})


def check_shared_path() -> list[CheckResult]:
    from shared import benchmark_eval, input_format, train_data

    bench_src = Path(benchmark_eval.__file__).read_text(encoding="utf-8")
    input_src = Path(input_format.__file__).read_text(encoding="utf-8")
    train_src = Path(train_data.__file__).read_text(encoding="utf-8")

    uses_shared = (
        "from .marker_insert import" in bench_src
        and "from .marker_insert import" in input_src
        and "from .marker_insert import" in train_src
        and "format_marked_pair" in train_src
        and "format_marked_pair" in bench_src
        and "insert_entity_markers" in input_src
    )
    no_local_replace = ".replace(" not in train_src and ".replace(" not in bench_src

    return [
        CheckResult(
            name="shared_marker_insertion_path",
            passed=uses_shared and no_local_replace,
            value="marker_insert.py",
            detail="train, benchmark, and CIViC eval import shared.marker_insert",
            before="duplicated string.replace(..., 1) in train and benchmark",
        ),
    ]


def check_train_cache(cache_dir: Path) -> list[CheckResult]:
    train_path = cache_dir / "train_examples_train.jsonl"
    val_path = cache_dir / "train_examples_val.jsonl"
    ok = train_path.exists() and val_path.exists()
    n_train = sum(1 for _ in train_path.open()) if ok else 0
    n_val = sum(1 for _ in val_path.open()) if ok else 0
    offset_methods = 0
    total = 0
    if ok:
        for path in (train_path, val_path):
            for line in path.open():
                if not line.strip():
                    continue
                row = json.loads(line)
                total += 1
                if row.get("marker_method") == "offset":
                    offset_methods += 1
    rate = offset_methods / total if total else 0.0
    return [
        CheckResult(
            name="train_cache_rebuilt",
            passed=ok and n_train > 0,
            value=n_train,
            detail=f"train cache {n_train} train + {n_val} val examples at {cache_dir}",
            before=None,
        ),
        CheckResult(
            name="train_cache_offset_rate",
            passed=rate >= 0.999,
            value=round(rate, 4),
            detail=f"{rate:.1%} of cached examples record marker_method=offset",
            before=None,
        ),
    ]


def run_all_checks(*, rebuild_cache: bool = True) -> dict[str, Any]:
    from .config import FOLDER10_TRAIN_CACHE, TRAIN_CACHE_DIR

    all_checks: list[CheckResult] = []

    if rebuild_cache:
        build_train_val_examples(TRAIN_CACHE_DIR, force=True)
        build_train_val_examples(FOLDER10_TRAIN_CACHE, force=True)

    train_checks, train_df = check_training_corpus()
    bench_checks, bench_df = check_benchmark_examples()
    civic_checks, civic_df = check_civic_pool()
    path_checks = check_shared_path()
    cache_checks = check_train_cache(TRAIN_CACHE_DIR)
    cache_checks10 = check_train_cache(FOLDER10_TRAIN_CACHE)
    cache_checks10[0].name = "folder10_train_cache_rebuilt"
    cache_checks10[1].name = "folder10_train_cache_offset_rate"

    all_checks.extend(
        train_checks + bench_checks + civic_checks + path_checks + cache_checks + cache_checks10
    )

    overall_pass = all(c.passed for c in all_checks)

    return {
        "overall_pass": overall_pass,
        "checks": [c.as_dict() for c in all_checks],
        "training_positives_df_rows": len(train_df),
        "training_same_sentence_rate": float(train_df["same_sentence_native"].dropna().mean())
        if len(train_df)
        else None,
        "civic_offset_rate": float((civic_df["method"] == "offset").mean()) if len(civic_df) else None,
    }
