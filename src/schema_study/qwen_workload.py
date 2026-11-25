"""Token lengths of complete Qwen conversations, including demonstrations and assistant text.

Zero-shot lengths are not reused. This module loads the tokenizer and does not
load model weights.
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from schema_study.formal_prepare import BUDGET, _fill
from schema_study.learning_conditions import HISTORICAL_PREDICATES, QWEN8_REVISION
from schema_study.paths import get_paths
from schema_study.production_runtime import context_sources
from schema_study.records import publish_file
from schema_study.schemas import default_schema_path, load_schema

STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
TOKENIZER = STORAGE / "models/Qwen__Qwen3-8B" / QWEN8_REVISION
CAPACITY = 40960
PREPARATION = STORAGE / "studies/schema_granularity_v1/reports/experimental_preparation"
CONTEXTS = {
    "biored": STORAGE / "data/inputs/biored/target_anchored_v2__biomedbert_e1354b7a_matched_e0_e7/contexts.jsonl",
    "bc8": STORAGE / "data/inputs/bc8/target_anchored_v2__biomedbert_e1354b7a_matched_e0_e7/contexts.jsonl",
    "drugprot": STORAGE / "data/inputs/drugprot/target_anchored_v1__biomedbert_e1354b7a/contexts.jsonl",
}


def _summary(values: list[int], supervised: list[int] | None = None) -> dict:
    ordered = sorted(values)
    def percentile(fraction: float) -> int:
        return ordered[round((len(ordered) - 1) * fraction)]
    payload = {
        "requests": len(ordered),
        "total_tokens": sum(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": percentile(0.5),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "over_capacity_with_generation_allowance": sum(1 for value in ordered if value + BUDGET > CAPACITY),
    }
    if supervised is not None:
        payload["supervised_tokens"] = sum(supervised)
        payload["mean_supervised_tokens"] = sum(supervised) / len(supervised)
    return payload


def _tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)


def _texts(corpus: str, entities: tuple[str, ...]) -> dict[str, dict[str, str]]:
    found = {}
    for source in context_sources(corpus):
        with CONTEXTS[source].open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                found.setdefault(row["candidate_id"], {}).update({entity: row["views"][entity] for entity in entities})
    return found


def _blocks(corpus: str) -> dict:
    schema = load_schema(default_schema_path(get_paths().code_root, "biored" if corpus == "bc8" else corpus))
    blocks = {}
    for research, historical in HISTORICAL_PREDICATES.items():
        names = schema.positive_display_names(historical)
        blocks[historical] = "\n".join(f"- {name}" for name in names)
    return blocks


def measure_few_shot(corpus: str = "biored", entities: tuple[str, ...] = ("E0", "E2", "E7"), predicates: tuple[str, ...] = ("P0", "P3"), skip: frozenset[str] = frozenset()) -> dict:
    """Retrieved demonstrations keep their identities. The target window is the matched source text."""
    tokenizer = _tokenizer()
    texts = _texts(corpus, entities)
    blocks = _blocks(corpus)
    grouped: dict[tuple[str, str], list[int]] = {}
    missing = 0
    with (PREPARATION / f"few_shot_{corpus}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            target = json.loads(line)
            if target["candidate_id"] not in texts:
                missing += 1
                continue
            for entity in entities:
                if f"{entity}" and any(f"{entity}/{research}" in skip for research in predicates):
                    if all(f"{entity}/{research}" in skip for research in predicates):
                        continue
                if any(identifier not in texts for identifier in target["demonstrations"]):
                    missing += 1
                    continue
                target_text = texts[target["candidate_id"]][entity]
                for research in predicates:
                    if f"{entity}/{research}" in skip:
                        continue
                    historical = HISTORICAL_PREDICATES[research]
                    label_block = {(corpus, historical): blocks[historical]}
                    demonstrations = [
                        _fill(corpus, identifier, historical, texts[identifier][entity], label_block)
                        for identifier in target["demonstrations"]
                    ]
                    content = "\n\n".join(demonstrations + [_fill(corpus, target["candidate_id"], historical, target_text, label_block)])
                    rendered = tokenizer.apply_chat_template(
                        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False,
                    )
                    length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
                    grouped.setdefault((entity, research), []).append(length)
    return {
        "corpus": corpus,
        "kind": "retrieved_five_shot",
        "missing": missing,
        "by_condition": {f"{entity}/{predicate}": _summary(values) for (entity, predicate), values in sorted(grouped.items())},
        "zero_shot_lengths_reused": False,
    }


def measure_sft(corpus: str = "biored", entities: tuple[str, ...] | None = None) -> dict:
    """Supervised length includes the assistant label object. The candidate identifier is not inside it."""
    tokenizer = _tokenizer()
    entities = entities or tuple(f"E{level}" for level in range(8))
    texts = _texts(corpus, entities)
    blocks = _blocks(corpus)
    grouped: dict[tuple[str, str], list[int]] = {}
    supervised: dict[tuple[str, str], list[int]] = {}
    with (PREPARATION / f"sft_{corpus}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            entity = row["entity_level"]
            research = row["predicate_level"]
            historical = HISTORICAL_PREDICATES[research]
            source = texts.get(row["candidate_id"], {}).get(entity)
            if source is None:
                continue
            prompt = _fill(corpus, row["candidate_id"], historical, source, {(corpus, historical): blocks[historical]})
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False,
            )
            full_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": json.dumps({"labels": row["assistant"]["labels"]}, ensure_ascii=False)}],
                tokenize=True, add_generation_prompt=False, enable_thinking=False,
            )
            key = (entity, research)
            grouped.setdefault(key, []).append(len(full_ids))
            supervised.setdefault(key, []).append(len(full_ids) - len(prompt_ids))
    return {
        "corpus": corpus,
        "kind": "independent_lora_sft",
        "by_condition": {
            f"{entity}/{predicate}": _summary(grouped[(entity, predicate)], supervised[(entity, predicate)])
            for entity, predicate in sorted(grouped)
        },
        "zero_shot_lengths_reused": False,
    }


def measure_remaining() -> dict:
    """Tokenise the workloads that are not already in the BioRED file. That file is not rewritten."""
    done = json.loads((PREPARATION / "qwen_workload_biored.json").read_text(encoding="utf-8"))
    skip = frozenset(done["few_shot"]["by_condition"])
    destination = get_paths().assert_output(PREPARATION / "qwen_workload_remaining.json")
    payload = {
        "zero_shot_lengths_reused": False,
        "biored_independent_sft_preserved": str(PREPARATION / "qwen_workload_biored.json"),
    }

    def publish(section: str, value: dict) -> None:
        payload[section] = value
        publish_file(destination, lambda temporary: temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8"))
        print(json.dumps({"published": section, "conditions": len(value.get("by_condition", {}))}), flush=True)

    publish("biored_few_shot_remaining", measure_few_shot("biored", tuple(f"E{level}" for level in range(8)), ("P0", "P1", "P2", "P3"), skip))
    publish("bc8_few_shot", measure_few_shot("bc8", tuple(f"E{level}" for level in range(8)), ("P0", "P1", "P2", "P3")))
    publish("drugprot_few_shot", measure_few_shot("drugprot", ("native",), ("P0", "P1", "P2", "P3")))
    publish("drugprot_sft", measure_sft("drugprot", ("native",)))
    return payload


def repair_bc8_few_shot() -> dict:
    """Replace the BC8 section that looked for demonstrations only inside the BC8 file."""
    destination = get_paths().assert_output(PREPARATION / "qwen_workload_remaining.json")
    payload = json.loads(destination.read_text(encoding="utf-8")) if destination.is_file() else {"zero_shot_lengths_reused": False}
    measured = measure_few_shot("bc8", tuple(f"E{level}" for level in range(8)), ("P0", "P1", "P2", "P3"))
    measured["demonstration_sources"] = list(context_sources("bc8"))
    payload["bc8_few_shot"] = measured
    publish_file(destination, lambda temporary: temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8"))
    print(json.dumps({"published": "bc8_few_shot", "conditions": len(measured["by_condition"]), "missing": measured["missing"]}), flush=True)
    return payload


def main() -> int:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "remaining":
        measure_remaining()
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "repair-bc8":
        repair_bc8_few_shot()
        return 0
    payload = {"few_shot": measure_few_shot(), "sft": measure_sft()}
    destination = get_paths().assert_output(PREPARATION / "qwen_workload_biored.json")
    publish_file(destination, lambda temporary: temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8"))
    print(json.dumps({
        "few_shot_conditions": len(payload["few_shot"]["by_condition"]),
        "sft_conditions": len(payload["sft"]["by_condition"]),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
