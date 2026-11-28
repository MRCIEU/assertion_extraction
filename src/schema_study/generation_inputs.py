"""Write rendered generation requests for the matched programme.

The prompt text is produced with the Qwen chat template. Qwen3-8B and Qwen3-32B
share that tokenizer, so one rendered string is stored and both models reference
it. This module does not load a model and does not write into formal_20261115.
"""

from __future__ import annotations

import json
from pathlib import Path

from schema_study.batch_plan import prompt_sha256
from schema_study.formal_prepare import _fill
from schema_study.learning_conditions import HISTORICAL_PREDICATES
from schema_study.paths import get_paths
from schema_study.qwen_workload import CONTEXTS, PREPARATION, _blocks, _texts, _tokenizer

STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
OUTPUT = STORAGE / "studies/schema_granularity_v1/runs/formal/formal_matched_e0_e7/generation_inputs"
EVALUATION_SPLIT = {"biored": "test", "bc8": "bc8", "drugprot": "eval"}
SHARD_SIZE = 256


def _request(dataset: str, row: dict, entity: str, predicate: str, prompt: str, rendered: str) -> dict:
    return {
        "request_id": f"{dataset}:{row['candidate_id']}:{entity}:{predicate}",
        "candidate_id": row["candidate_id"],
        "document_id": row.get("document_id"),
        "dataset": dataset,
        "split": row.get("split"),
        "entity_level": entity,
        "predicate_level": predicate,
        "prompt": prompt,
        "rendered": rendered,
        "rendered_qwen32b": rendered,
        "rendered_sha256": prompt_sha256(rendered),
        "shared_qwen3_tokenizer": True,
        "decoding": "greedy",
        "max_new_tokens": 256,
    }


class _ConditionWriter:
    """Append one condition. The completion marker is written only after the last request."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.preserved = (directory / "COMPLETE").is_file()
        self.count = 0
        if not self.preserved:
            directory.mkdir(parents=True, exist_ok=True)
            for old in directory.glob("shard_*.jsonl"):
                old.unlink()

    def add(self, row: dict) -> None:
        if self.preserved:
            return
        path = self.directory / f"shard_{self.count // SHARD_SIZE:05d}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        self.count += 1

    def finish(self) -> dict:
        if self.preserved:
            return {"directory": str(self.directory), "preserved": True}
        shards = sorted(path.name for path in self.directory.glob("shard_*.jsonl"))
        (self.directory / "COMPLETE").write_text(json.dumps({"requests": self.count, "shards": shards}) + "\n", encoding="utf-8")
        return {"directory": str(self.directory), "preserved": False, "requests": self.count, "shards": len(shards)}


def write_zero_shot(dataset: str, *, output: Path | None = None, entities: tuple[str, ...] | None = None, contexts: dict | None = None) -> dict:
    """Render evaluation prompts. Training rows are not generation requests."""
    destination_root = output or OUTPUT
    tokenizer = _tokenizer()
    blocks = _blocks("biored" if dataset == "bc8" else dataset)
    target_split = EVALUATION_SPLIT[dataset]
    if entities is None:
        entities = ("native",) if dataset == "drugprot" else tuple(f"E{level}" for level in range(8))
    context_map = contexts or CONTEXTS
    predicates = ("P0", "P1", "P2", "P3")
    writers = {(entity, predicate): _ConditionWriter(destination_root / "zero_shot" / dataset / entity / predicate) for entity in entities for predicate in predicates}
    split_counts: dict[str, int] = {}
    with context_map[dataset].open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split_counts[row.get("split", "")] = split_counts.get(row.get("split", ""), 0) + 1
            if row.get("split") != target_split:
                continue
            for entity in entities:
                source = (row.get("views") or {}).get(entity)
                if not source:
                    continue
                for predicate in predicates:
                    historical = HISTORICAL_PREDICATES[predicate]
                    prompt = _fill(dataset, row["candidate_id"], historical, source, {(dataset, historical): blocks[historical]})
                    rendered = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False,
                    )
                    writers[(entity, predicate)].add(_request(dataset, row, entity, predicate, prompt, rendered))
    written = [writers[key].finish() for key in sorted(writers)]
    summary = {"dataset": dataset, "split": target_split, "split_counts": split_counts, "conditions": written, "prompt_source": "zero_shot"}
    destination = destination_root / "zero_shot" / dataset / "inventory.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    return summary


def write_few_shot(dataset: str, *, output: Path | None = None, entities: tuple[str, ...] | None = None, contexts: dict | None = None) -> dict:
    """Render retrieved five-shot prompts. A missing demonstration is recorded and not omitted silently."""
    destination_root = output or OUTPUT
    tokenizer = _tokenizer()
    if entities is None:
        entities = ("native",) if dataset == "drugprot" else tuple(f"E{level}" for level in range(8))
    predicates = ("P0", "P1", "P2", "P3")
    texts = _texts(dataset, entities, contexts)
    blocks = _blocks("biored" if dataset == "bc8" else dataset)
    writers = {(entity, predicate): _ConditionWriter(destination_root / "few_shot" / dataset / entity / predicate) for entity in entities for predicate in predicates}
    eligible = missing_target = missing_demonstration = 0
    with (PREPARATION / f"few_shot_{dataset}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            target = json.loads(line)
            eligible += 1
            if target["candidate_id"] not in texts:
                missing_target += 1
                continue
            unresolved = [identifier for identifier in target["demonstrations"] if identifier not in texts]
            if unresolved:
                missing_demonstration += 1
                continue
            for entity in entities:
                if entity not in texts[target["candidate_id"]]:
                    missing_target += 1
                    continue
                for predicate in predicates:
                    historical = HISTORICAL_PREDICATES[predicate]
                    label_block = {(dataset, historical): blocks[historical]}
                    demonstrations = [
                        _fill(dataset, identifier, historical, texts[identifier][entity], label_block)
                        for identifier in target["demonstrations"]
                    ]
                    prompt = "\n\n".join(demonstrations + [_fill(dataset, target["candidate_id"], historical, texts[target["candidate_id"]][entity], label_block)])
                    rendered = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False,
                    )
                    writers[(entity, predicate)].add(_request(dataset, target, entity, predicate, prompt, rendered))
    written = [writers[key].finish() for key in sorted(writers)]
    summary = {
        "dataset": dataset,
        "eligible_targets": eligible,
        "missing_target_contexts": missing_target,
        "missing_demonstration_contexts": missing_demonstration,
        "conditions": written,
        "omitted_unresolved_demonstrations": False,
    }
    destination = destination_root / "few_shot" / dataset / "inventory.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else ""
    dataset = sys.argv[2] if len(sys.argv) > 2 else ""
    get_paths().assert_output(OUTPUT / "inventory.json")
    if command == "zero-shot" and dataset in EVALUATION_SPLIT:
        print(json.dumps({"wrote": command, "dataset": dataset, "conditions": len(write_zero_shot(dataset)["conditions"])}), flush=True)
        return 0
    if command == "few-shot" and dataset in EVALUATION_SPLIT:
        print(json.dumps({"wrote": command, "dataset": dataset, "eligible": write_few_shot(dataset)["eligible_targets"]}), flush=True)
        return 0
    print("Usage: generation_inputs zero-shot|few-shot biored|bc8|drugprot", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
