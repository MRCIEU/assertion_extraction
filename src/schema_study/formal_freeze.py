"""Write the formal-run freeze after the V1 length scan has passed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from schema_study.experiments import training_hyperparameters

CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
ROOT = STORAGE / "studies/schema_granularity_v1/runs/formal/formal_20261115"
REPORTS = STORAGE / "studies/schema_granularity_v1/reports/formal/formal_20261115"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    summary_path = ROOT / "prepare/length_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("accepted"):
        raise SystemExit("V1 length scan did not pass. Formal jobs stay stopped.")
    if summary["prompt_sha256"] != _sha(CODE / "prompts/relation_extraction_v1.txt"):
        raise SystemExit("V1 prompt hash does not match the length scan.")
    code_files = [
        CODE / "prompts/relation_extraction_v1.txt",
        CODE / "src/schema_study/formal_train.py",
        CODE / "src/schema_study/formal_generate.py",
        CODE / "src/schema_study/formal_prepare.py",
        CODE / "src/schema_study/predictions.py",
        CODE / "src/schema_study/formal_eval.py",
        CODE / "src/schema_study/evaluation.py",
        CODE / "configs/schemas/biored.yaml",
        CODE / "configs/schemas/drugprot.yaml",
        CODE / "scripts/slurm/gpu_formal_train.sbatch",
        CODE / "scripts/slurm/gpu_formal_generate.sbatch",
    ]
    data_files = [
        STORAGE / "data/prepared/biored/biored_candidates_v2/candidates.parquet",
        STORAGE / "data/prepared/bc8/bc8_candidates_v2/candidates.parquet",
        STORAGE / "data/prepared/drugprot/drugprot_candidates_v1/candidates.parquet",
        STORAGE / "data/inputs/biored/target_anchored_v1__biomedbert_e1354b7a_proposed/contexts.jsonl",
        STORAGE / "data/inputs/bc8/target_anchored_v1__biomedbert_e1354b7a_proposed/contexts.jsonl",
        STORAGE / "data/inputs/drugprot/target_anchored_v1__biomedbert_e1354b7a/contexts.jsonl",
    ]
    record = {
        "freeze_name": "formal_execution_freeze",
        "not_an_independent_preregistration": True,
        "run_id": "formal_20261115",
        "authorization": "Researcher authorized the V1 main matrix, 48 BiomedBERT trainings, and both Qwen models.",
        "prompt_version": "v1",
        "prompt_selection_reason": "The development wording was clearer. Selection is not a claim of stable development-sample gain.",
        "v0_not_run": True,
        "controls_included": False,
        "human_review_complete": False,
        "semantic_validated": False,
        "api_requests_made": False,
        "prior_tuning_use_confirmed": False,
        "code_hashes": {str(path): _sha(path) for path in code_files},
        "data_hashes": {str(path): _sha(path) for path in data_files},
        "models": {
            "biomedbert": {"model_id": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "revision": "e1354b7a3a09615f6aba48dfad4b7a613eef7062"},
            "qwen8b": {"model_id": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218"},
            "qwen32b": {"model_id": "Qwen/Qwen3-32B", "revision": "9216db5781bf21249d130ec9da846c4624c16137"},
        },
        "training": training_hyperparameters(),
        "training_precision_note": "The original agreement left precision pending. The accepted smoke and this authorization use bf16 autocast for training and FP32 prediction.",
        "seeds": [13, 42, 2026],
        "selection": {
            "biored": "dev positive micro-F1, earlier epoch on a tie",
            "drugprot": "one threshold from 0.05 to 0.95 step 0.05, higher threshold on a tie, then earlier epoch",
            "rollup": "map saved P4 labels, do not search a new threshold",
        },
        "generation": {
            "backend": "transformers",
            "dtype": "bf16",
            "batch_size": 16,
            "do_sample": False,
            "enable_thinking": False,
            "max_new_tokens": 256,
            "padding_side": "left",
            "truncation": False,
            "attention_mask": "tokenizer mask passed explicitly",
            "position_limit": summary["position_limit"],
            "shard_requests": 8192,
        },
        "bootstrap": {"seed": 100, "replicates": 2000, "unit": "document", "interval": "percentile_95", "paired": True},
        "length_scan": {
            "accepted": True,
            "requests_per_model": summary["requests_per_model"],
            "main_matrix_requests": summary["main_matrix_requests"],
            "prompts_over_capacity": summary["prompts_over_capacity"],
            "summary": str(summary_path),
        },
        "counts": {
            "biomedbert_trainings": 48,
            "direct_finetune_evaluations": 84,
            "qwen_prompt_conditions": 32,
            "qwen_direct_generation_units": 56,
            "qwen_unique_requests": summary["main_matrix_requests"],
        },
        "known_limits": [
            "DrugProt prior tuning is unconfirmed, so results are not claimed as an untouched confirmation.",
            "Five control prompts are not in this run.",
            "No human semantic audit is done.",
            "Bootstrap intervals reflect document resampling, not all training uncertainty.",
        ],
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=2) + "\n"
    (ROOT / "formal_freeze_record.json").write_text(text, encoding="utf-8")
    (REPORTS / "formal_freeze_record.json").write_text(text, encoding="utf-8")
    print(json.dumps({"frozen": True, "requests": summary["main_matrix_requests"], "shards": len(summary["shards"])}, indent=2))


if __name__ == "__main__":
    main()
