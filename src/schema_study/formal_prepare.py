"""Measure every formal V1 prompt and write generation shards. No model call and no gold."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from schema_study.schemas import load_schema

CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
OUT = STORAGE / "studies/schema_granularity_v1/runs/formal/formal_20261115"
PROMPT = (CODE / "prompts/relation_extraction_v1.txt").read_text(encoding="utf-8")
MODELS = {
    "qwen8b": STORAGE / "models/Qwen__Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218",
    "qwen32b": STORAGE / "models/Qwen__Qwen3-32B/9216db5781bf21249d130ec9da846c4624c16137",
}
SOURCES = (
    ("biored", "test", STORAGE / "data/inputs/biored/target_anchored_v1__biomedbert_e1354b7a_proposed/contexts.jsonl", ("E0", "E1", "E2"), 7047),
    ("bc8", "bc8", STORAGE / "data/inputs/bc8/target_anchored_v1__biomedbert_e1354b7a_proposed/contexts.jsonl", ("E0", "E1", "E2"), 28734),
    ("drugprot", "eval", STORAGE / "data/inputs/drugprot/target_anchored_v1__biomedbert_e1354b7a/contexts.jsonl", ("native",), 116887),
)
CARDINALITY = {
    "biored": "This BioRED candidate allows at most one positive label.",
    "bc8": "This BioRED candidate allows at most one positive label.",
    "drugprot": "This DrugProt candidate may include more than one positive label.",
}
LEVELS = ("P1", "P2", "P3", "P4")
LIMIT = 40960
BUDGET = 256
SHARD = 8192
BATCH = 256


def _fill(dataset: str, candidate_id: str, level: str, text: str, blocks: dict) -> str:
    shown = "biored" if dataset == "bc8" else dataset
    return (
        PROMPT.replace("{{candidate_id}}", candidate_id)
        .replace("{{dataset}}", shown)
        .replace("{{cardinality_rule}}", CARDINALITY[dataset])
        .replace("{{label_block}}", blocks[(dataset, level)])
        .replace("{{text}}", text)
    )


def _percentile(values: list[int], fraction: float) -> int:
    index = round((len(values) - 1) * fraction)
    return sorted(values)[index]


class ShardWriter:
    def __init__(self, root: Path, dataset: str, view: str, level: str) -> None:
        self.directory = root / dataset / view / level
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.view = view
        self.level = level
        self.count = 0
        self.shard_index = 0
        self.buffer: list[str] = []
        self.shards: list[dict] = []
        self._open()

    def _open(self) -> None:
        self.path = self.directory / f"shard_{self.shard_index:05d}.jsonl"
        self.handle = self.path.open("w", encoding="utf-8")
        self.shard_start = self.count

    def add(self, row: dict) -> None:
        self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.count += 1
        if self.count - self.shard_start == SHARD:
            self._close_shard()
            self.shard_index += 1
            self._open()

    def _close_shard(self) -> None:
        self.handle.close()
        size = self.count - self.shard_start
        if size == 0:
            self.path.unlink(missing_ok=True)
            return
        if size:
            self.shards.append({
                "dataset": self.dataset,
                "entity_level": self.view,
                "predicate_level": self.level,
                "shard_index": self.shard_index,
                "path": str(self.path),
                "requests": size,
                "first_request_offset": self.shard_start,
            })

    def close(self) -> None:
        if not self.handle.closed:
            self._close_shard()


def main() -> None:
    if (OUT / "prepare/length_summary.json").exists():
        raise SystemExit("V1 length summary already exists and will not be rewritten.")
    tokenizers = {name: AutoTokenizer.from_pretrained(path, local_files_only=True) for name, path in MODELS.items()}
    capacities = {name: int(json.loads((path / "config.json").read_text(encoding="utf-8"))["max_position_embeddings"]) for name, path in MODELS.items()}
    schemas = {
        "biored": load_schema(CODE / "configs/schemas/biored.yaml"),
        "bc8": load_schema(CODE / "configs/schemas/biored.yaml"),
        "drugprot": load_schema(CODE / "configs/schemas/drugprot.yaml"),
    }
    blocks = {(dataset, level): "\n".join(f"- {name}" for name in schema.positive_display_names(level)) for dataset, schema in schemas.items() for level in LEVELS}
    lengths: dict[tuple, list[int]] = {}
    longest: dict[tuple, dict] = {}
    writers: dict[tuple, ShardWriter] = {}
    seen = {}
    identical_render = True
    over = {name: 0 for name in MODELS}
    for dataset, split, path, views, expected in SOURCES:
        count = 0
        with path.open(encoding="utf-8") as handle:
            pending: list[dict] = []

            def flush() -> None:
                nonlocal identical_render
                if not pending:
                    return
                messages = [[{"role": "user", "content": item["prompt"]}] for item in pending]
                rendered = {
                    name: tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    for name, tokenizer in tokenizers.items()
                }
                counts = {}
                for name, tokenizer in tokenizers.items():
                    encoded = tokenizer(rendered[name], truncation=False, add_special_tokens=False)
                    counts[name] = [len(ids) for ids in encoded["input_ids"]]
                for index, item in enumerate(pending):
                    for name in tokenizers:
                        count_value = counts[name][index]
                        if count_value + BUDGET > capacities[name]:
                            over[name] += 1
                        key = (name, dataset, item["view"], item["level"])
                        bucket = lengths.setdefault(key, [])
                        bucket.append(count_value)
                        current = longest.get(key)
                        if current is None or count_value > current["prompt_tokens"]:
                            longest[key] = {"candidate_id": item["candidate_id"], "document_id": item["document_id"], "prompt_tokens": count_value}
                    text_8b = rendered["qwen8b"][index]
                    text_32b = rendered["qwen32b"][index]
                    if text_8b != text_32b:
                        identical_render = False
                    digest = hashlib.sha256(text_8b.encode("utf-8")).hexdigest()
                    row = {
                        "request_id": f"{dataset}|{item['view']}|{item['level']}|{item['candidate_id']}",
                        "candidate_id": item["candidate_id"],
                        "document_id": item["document_id"],
                        "dataset": dataset,
                        "split": split,
                        "entity_level": item["view"],
                        "predicate_level": item["level"],
                        "rendered": text_8b,
                        "rendered_sha256": digest,
                        "tokens_qwen8b": counts["qwen8b"][index],
                        "tokens_qwen32b": counts["qwen32b"][index],
                    }
                    if text_8b != text_32b:
                        row["rendered_qwen32b"] = text_32b
                        row["rendered_qwen32b_sha256"] = hashlib.sha256(text_32b.encode("utf-8")).hexdigest()
                    key = (dataset, item["view"], item["level"])
                    writer = writers.get(key)
                    if writer is None:
                        writer = ShardWriter(OUT / "generation/shards", dataset, item["view"], item["level"])
                        writers[key] = writer
                    writer.add(row)
                pending.clear()

            for line in handle:
                if f'"split": "{split}"' not in line:
                    continue
                row = json.loads(line)
                if row.get("split") != split:
                    continue
                count += 1
                for view in views:
                    text = row["views"][view]
                    for level in LEVELS:
                        pending.append({
                            "prompt": _fill(dataset, row["candidate_id"], level, text, blocks),
                            "view": view,
                            "level": level,
                            "candidate_id": row["candidate_id"],
                            "document_id": str(row["document_id"]),
                        })
                        if len(pending) >= BATCH:
                            flush()
                if count % 20000 == 0:
                    print(f"{dataset} {count}", flush=True)
            flush()
        seen[dataset] = count
        if count != expected:
            raise SystemExit(f"{dataset} has {count} {split} contexts, expected {expected}.")
    for writer in writers.values():
        writer.close()
    conditions = []
    for key, values in sorted(lengths.items()):
        model, dataset, view, level = key
        conditions.append({
            "model": model,
            "dataset": dataset,
            "entity_level": view,
            "predicate_level": level,
            "requests": len(values),
            "min": min(values),
            "median": _percentile(values, 0.5),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
            "max": max(values),
            "max_plus_256": max(values) + BUDGET,
            "over_position_limit": sum(value + BUDGET > capacities[model] for value in values),
            "longest": longest[key],
        })
    per_model = {name: sum(row["requests"] for row in conditions if row["model"] == name) for name in MODELS}
    payload = {
        "prompt_path": str(CODE / "prompts/relation_extraction_v1.txt"),
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "prompt_version": "v1",
        "gold_used": False,
        "model_called": False,
        "truncation": False,
        "position_limit": capacities,
        "output_budget": BUDGET,
        "candidates_seen": seen,
        "requests_per_model": per_model,
        "main_matrix_requests": per_model["qwen8b"] + per_model["qwen32b"],
        "rendered_strings_identical_across_models": identical_render,
        "prompts_over_capacity": over,
        "accepted": all(value == 0 for value in over.values()),
        "conditions": conditions,
        "shards": [shard for writer in writers.values() for shard in writer.shards],
        "controls_included": False,
    }
    prepare = OUT / "prepare"
    prepare.mkdir(parents=True, exist_ok=True)
    (prepare / "length_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"seen": seen, "requests_per_model": per_model, "over": over, "identical_render": identical_render, "max": max(row["max"] for row in conditions)}, indent=2))


if __name__ == "__main__":
    main()
