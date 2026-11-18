"""Stream formal-evaluation prompts and record token lengths. No model and no gold."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from transformers import AutoTokenizer

from schema_study.schemas import load_schema

CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
OUT = STORAGE / "studies/schema_granularity_v1/runs/backend_validation/round5_20261115/lengths"
PROMPT = (CODE / "prompts/relation_extraction.txt").read_text(encoding="utf-8")
MODELS = {
    "qwen8b": STORAGE / "models/Qwen__Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218",
    "qwen32b": STORAGE / "models/Qwen__Qwen3-32B/9216db5781bf21249d130ec9da846c4624c16137",
}
SOURCES = (
    ("biored", "test", STORAGE / "data/inputs/biored/target_anchored_v1__biomedbert_e1354b7a_proposed/contexts.jsonl", ("E0", "E1", "E2")),
    ("bc8", "bc8", STORAGE / "data/inputs/bc8/target_anchored_v1__biomedbert_e1354b7a_proposed/contexts.jsonl", ("E0", "E1", "E2")),
    ("drugprot", "eval", STORAGE / "data/inputs/drugprot/target_anchored_v1__biomedbert_e1354b7a/contexts.jsonl", ("native",)),
)
OLD_LIMIT = 1243
OUTPUT_BUDGET = 256


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _summarize(values: list[int], longest: dict, capacity: int) -> dict:
    return {
        "requests": len(values),
        "min": min(values),
        "median": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
        "max_plus_output_budget": max(values) + OUTPUT_BUDGET,
        "over_1243": sum(value > OLD_LIMIT for value in values),
        "prompt_over_model_capacity": sum(value > capacity for value in values),
        "prompt_plus_output_over_model_capacity": sum(value + OUTPUT_BUDGET > capacity for value in values),
        "longest": longest,
    }


def main() -> None:
    tokenizers = {
        name: AutoTokenizer.from_pretrained(path, local_files_only=True)
        for name, path in MODELS.items()
    }
    capacities = {}
    for name, path in MODELS.items():
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        capacities[name] = int(config["max_position_embeddings"])
    schemas = {
        "biored": load_schema(CODE / "configs/schemas/biored.yaml"),
        "bc8": load_schema(CODE / "configs/schemas/biored.yaml"),
        "drugprot": load_schema(CODE / "configs/schemas/drugprot.yaml"),
    }
    levels = ("P1", "P2", "P3", "P4")
    blocks = {
        (dataset, level): "\n".join(f"- {label}" for label in schema.positive_display_names(level))
        for dataset, schema in schemas.items()
        for level in levels
    }
    lengths: dict[tuple, list[int]] = {}
    longest: dict[tuple, dict] = {}
    seen = {source[0]: 0 for source in SOURCES}
    for dataset, split, path, views in SOURCES:
        schema_key = dataset
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if f'"split": "{split}"' not in line:
                    continue
                row = json.loads(line)
                if row.get("split") != split:
                    continue
                seen[dataset] += 1
                for view in views:
                    text = row["views"][view]
                    for level in levels:
                        prompt = (
                            PROMPT.replace("{{candidate_id}}", row["candidate_id"])
                            .replace("{{dataset}}", "biored" if dataset == "bc8" else dataset)
                            .replace("{{label_block}}", blocks[(schema_key, level)])
                            .replace("{{text}}", text)
                        )
                        messages = [{"role": "user", "content": prompt}]
                        for model, tokenizer in tokenizers.items():
                            rendered = tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=False,
                            )
                            count = len(tokenizer(rendered, truncation=False, add_special_tokens=False)["input_ids"])
                            key = (model, dataset, view, level)
                            bucket = lengths.setdefault(key, [])
                            bucket.append(count)
                            current = longest.get(key)
                            if current is None or count > current["prompt_tokens"]:
                                longest[key] = {
                                    "candidate_id": row["candidate_id"],
                                    "document_id": row["document_id"],
                                    "split": split,
                                    "source": str(path),
                                    "prompt_tokens": count,
                                }
                if seen[dataset] % 20000 == 0:
                    print(f"{dataset} {seen[dataset]}", flush=True)
    rows = []
    for key, values in sorted(lengths.items()):
        model, dataset, view, level = key
        summary = _summarize(values, longest[key], capacities[model])
        summary.update({"model": model, "dataset": dataset, "entity_level": view, "predicate_level": level, "control": False})
        rows.append(summary)
    total = sum(row["requests"] for row in rows if row["model"] == "qwen8b")
    per_model = {model: sum(row["requests"] for row in rows if row["model"] == model) for model in tokenizers}
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "output_budget": OUTPUT_BUDGET,
        "old_round4_total_limit": OLD_LIMIT,
        "model_capacity": capacities,
        "capacity_check": "Tokenizer truncation is false. Transformers generate has no max_model_len argument. A prompt is over capacity when its token count, or that count plus 256, exceeds config.max_position_embeddings.",
        "candidates_seen": seen,
        "requests_per_model": per_model,
        "main_matrix_requests_one_model_sum_should_match_total_over_models": total,
        "main_matrix_requests": total,
        "conditions": rows,
        "controls": {
            "status": "draft_pending_confirmation",
            "included_in_length_scan": False,
            "reason": "Display names are not frozen. Corpus, entity level, and model are specified as BioRED E2 Qwen3-32B. The evaluation split is not separately frozen.",
        },
        "gold_used": False,
        "model_called": False,
    }
    (OUT / "length_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"seen": seen, "requests_per_model": per_model, "max": max(row["max"] for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
