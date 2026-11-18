"""Render and measure the frozen round-6 prompts. This does not call a model."""

from __future__ import annotations

import json
from pathlib import Path

from schema_study.batch_plan import freeze_manifest, write_manifest_once
from schema_study.generation_ledger import read_jsonl

CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
OUT = STORAGE / "studies/schema_granularity_v1/runs/task_suitability/round6_20261115"
MODELS = {
    "qwen8b": STORAGE / "models/Qwen__Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218",
    "qwen32b": STORAGE / "models/Qwen__Qwen3-32B/9216db5781bf21249d130ec9da846c4624c16137",
}
LIMIT = 40960


def _render(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main() -> None:
    requests = read_jsonl(OUT / "samples/requests.jsonl")
    if any("rendered_v0_qwen8b" in row for row in requests):
        raise SystemExit("Rendered prompts are already frozen.")
    from transformers import AutoTokenizer

    tokenizers = {
        name: AutoTokenizer.from_pretrained(path, local_files_only=True)
        for name, path in MODELS.items()
    }
    summary = {}
    for version, field in (("v0", "prompt_v0"), ("v1", "prompt_v1")):
        summary[version] = {}
        for model_name, tokenizer in tokenizers.items():
            counts = []
            for row in requests:
                rendered = _render(tokenizer, row[field])
                count = len(tokenizer(rendered, truncation=False)["input_ids"])
                if count + 256 > LIMIT:
                    raise SystemExit(f"{row['request_id']} {version} {model_name} exceeds the position limit.")
                row[f"rendered_{version}_{model_name}"] = rendered
                row[f"tokens_{version}_{model_name}"] = count
                counts.append(count)
            counts.sort()
            summary[version][model_name] = {
                "min": counts[0],
                "median": counts[len(counts) // 2],
                "max": counts[-1],
                "max_plus_256": counts[-1] + 256,
                "over_40960": 0,
            }
    with (OUT / "samples/requests.jsonl").open("w", encoding="utf-8") as handle:
        for row in requests:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for version in ("v0", "v1"):
        field = f"rendered_{version}_qwen8b"
        if any(row[field] != row[f"rendered_{version}_qwen32b"] for row in requests):
            for model_name in MODELS:
                manifest = freeze_manifest(requests, 16, f"rendered_{version}_{model_name}")
                write_manifest_once(OUT / "plan" / f"batch_manifest_{version}_{model_name}.json", manifest)
        else:
            manifest = freeze_manifest(requests, 16, field)
            for member, row in zip(
                (member for batch in manifest["batches"] for member in batch["members"]),
                requests,
            ):
                member["unpadded_tokens"] = row[f"tokens_{version}_qwen8b"]
            write_manifest_once(OUT / "plan" / f"batch_manifest_{version}.json", manifest)
    plan_path = OUT / "plan/development_check_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["prompt_lengths"] = summary
    plan["position_limit"] = LIMIT
    plan["truncation"] = False
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
