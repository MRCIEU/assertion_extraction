"""Token lengths of the Qwen prompts built from the prepared source windows.

The 8B and 32B tokenizers are the same file. This module loads that tokenizer
and does not load model weights. A BiomedBERT length is not reused as a Qwen length.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from schema_study.formal_prepare import BUDGET, CARDINALITY, PROMPT, _fill
from schema_study.learning_conditions import HISTORICAL_PREDICATES, QWEN32_REVISION, QWEN8_REVISION
from schema_study.paths import get_paths
from schema_study.records import publish_file
from schema_study.schemas import default_schema_path, load_schema

STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
TOKENIZER_DIR = STORAGE / "models/Qwen__Qwen3-8B" / QWEN8_REVISION
TOKENIZER_32 = STORAGE / "models/Qwen__Qwen3-32B" / QWEN32_REVISION
CAPACITY = 40960
BATCH = 128
INPUTS = {
    "biored": STORAGE / "data/inputs/biored/target_anchored_v2__biomedbert_e1354b7a_matched_e0_e7/contexts.jsonl",
    "bc8": STORAGE / "data/inputs/bc8/target_anchored_v2__biomedbert_e1354b7a_matched_e0_e7/contexts.jsonl",
    "drugprot": STORAGE / "data/inputs/drugprot/target_anchored_v1__biomedbert_e1354b7a/contexts.jsonl",
}


def _percentile(values: list[int], fraction: float) -> int:
    index = round((len(values) - 1) * fraction)
    return values[index]


def _summary(values: list[int]) -> dict:
    ordered = sorted(values)
    counts = Counter(ordered)
    over = sum(1 for value in ordered if value + BUDGET > CAPACITY)
    return {
        "requests": len(ordered),
        "min": ordered[0],
        "median": _percentile(ordered, 0.5),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
        "max_plus_generation_allowance": ordered[-1] + BUDGET,
        "over_capacity_with_generation_allowance": over,
        "histogram": {str(length): count for length, count in sorted(counts.items())},
    }


def measure_corpus(corpus: str, tokenizer, start: int = 0, end: int | None = None) -> dict:
    schema_name = "biored" if corpus == "bc8" else corpus
    code = get_paths().code_root
    schema = load_schema(default_schema_path(code, schema_name))
    blocks = {}
    for research, historical in HISTORICAL_PREDICATES.items():
        names = schema.positive_display_names(historical)
        blocks[(corpus, historical)] = "\n".join(f"- {name}" for name in names)
    grouped: dict[tuple, list[int]] = {}
    pending: list[dict] = []
    seen = 0
    measured = 0
    skipped_unlabeled = 0

    def flush() -> None:
        if not pending:
            return
        messages = [[{"role": "user", "content": item["prompt"]}] for item in pending]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        encoded = tokenizer(rendered, truncation=False, add_special_tokens=False)
        for item, token_ids in zip(pending, encoded["input_ids"]):
            grouped.setdefault((item["split"], item["entity"], item["predicate"]), []).append(len(token_ids))
        pending.clear()

    with INPUTS[corpus].open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            split = str(row.get("split") or "")
            if corpus == "drugprot" and split == "unlabeled":
                skipped_unlabeled += 1
                continue
            if seen < start:
                seen += 1
                continue
            if end is not None and seen >= end:
                break
            seen += 1
            measured += 1
            views = row.get("views") or {}
            for entity, text in views.items():
                for research, historical in HISTORICAL_PREDICATES.items():
                    pending.append(
                        {
                            "prompt": _fill(corpus, row["candidate_id"], historical, text, blocks),
                            "split": split,
                            "entity": entity,
                            "predicate": research,
                        }
                    )
                    if len(pending) >= BATCH:
                        flush()
            if seen % 5000 == 0:
                print(f"{corpus} {seen}", flush=True)
    flush()
    conditions = []
    for (split, entity, predicate), values in sorted(grouped.items()):
        conditions.append({"split": split, "entity_level": entity, "predicate_level": predicate, **_summary(values)})
    return {
        "corpus": corpus,
        "contexts": measured,
        "range_start": start,
        "range_end": end,
        "skipped_unlabeled": skipped_unlabeled,
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "cardinality_rule": CARDINALITY[corpus],
        "generation_allowance": BUDGET,
        "position_limit": CAPACITY,
        "weights_loaded": False,
        "truncation": False,
        "conditions": conditions,
        "prompts_over_capacity": sum(row["over_capacity_with_generation_allowance"] for row in conditions),
    }


def main() -> int:
    digest_8 = hashlib.sha256((TOKENIZER_DIR / "tokenizer.json").read_bytes()).hexdigest()
    digest_32 = hashlib.sha256((TOKENIZER_32 / "tokenizer.json").read_bytes()).hexdigest()
    if digest_8 != digest_32:
        raise SystemExit("The Qwen3-8B and Qwen3-32B tokenizers differ; measure them separately.")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    destination = get_paths().study / "reports" / "experimental_preparation"
    destination.mkdir(parents=True, exist_ok=True)
    requested = [item for item in sys.argv[1:] if item in INPUTS]
    corpora = tuple(requested) or ("biored", "bc8", "drugprot")
    numbers = [int(item) for item in sys.argv[1:] if item.isdigit()]
    start = numbers[0] if numbers else 0
    end = numbers[1] if len(numbers) > 1 else None
    suffix = "" if start == 0 and end is None else f"_{start}_{'end' if end is None else end}"
    for corpus in corpora:
        print(f"Measuring {corpus} {start}:{end}", flush=True)
        payload = {
            "tokenizer_sha256": digest_8,
            "tokenizer_shared_by_qwen3_8b_and_qwen3_32b": True,
            "qwen8b_revision": QWEN8_REVISION,
            "qwen32b_revision": QWEN32_REVISION,
            "corpus": measure_corpus(corpus, tokenizer, start, end),
        }
        publish_file(
            destination / f"qwen_lengths_{corpus}{suffix}.json",
            lambda temporary, body=payload: temporary.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8"),
        )
        print(json.dumps({"corpus": corpus, "over": payload["corpus"]["prompts_over_capacity"], "contexts": payload["corpus"]["contexts"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
