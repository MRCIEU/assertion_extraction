"""Build mesh_ancestor_v1 contexts and prompts. The stored v2 files are not overwritten."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from schema_study.contexts import build_context_tables

STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
STUDY = STORAGE / "studies/schema_granularity_v1"
CONCEPTS = STUDY / "reports/representation_corrected/concepts.jsonl"
FROZEN = "77fa5b41a6e18c277f987dedd74ea28833b8afa4c823b1396c935f5d7ee850a5"
VERSION = "target_anchored_mesh_ancestor_v1__biomedbert_e1354b7a"
TOKENIZER = STORAGE / "models/microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/e1354b7a3a09615f6aba48dfad4b7a613eef7062/tokenizer"
HISTORICAL = {
    "biored": STORAGE / "data/inputs/biored/target_anchored_v2__biomedbert_e1354b7a_matched_e0_e7/contexts.jsonl",
    "bc8": STORAGE / "data/inputs/bc8/target_anchored_v2__biomedbert_e1354b7a_matched_e0_e7/contexts.jsonl",
}
PROMPT_ROOT = STUDY / "runs/formal/formal_matched_e0_e7/generation_inputs/mesh_ancestor_v1"
TYPE_MARKER = re.compile(r"<<TYPE:[^>]*>>")
LEVELS = tuple(f"E{level}" for level in range(9))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _display() -> dict[tuple[str, str], dict]:
    found = {}
    for line in CONCEPTS.open(encoding="utf-8"):
        row = json.loads(line)
        found[(row["corpus"], row["concept_id"])] = row["displayed"]
    return found


def _mention_map(corpus: str, displayed: dict) -> dict[str, dict]:
    prepared = STORAGE / "data/prepared" / corpus / f"{corpus}_candidates_v2" / "entities.jsonl"
    mapping = {}
    for line in prepared.open(encoding="utf-8"):
        row = json.loads(line)
        shown = displayed.get((corpus, str(row.get("concept_id") or "")))
        if shown:
            mapping[row["mention_id"]] = shown
    return mapping


def _compare(corpus: str, new_path: Path) -> dict:
    historical = {}
    with HISTORICAL[corpus].open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            historical[row["candidate_id"]] = row
    changed = {"train": 0, "dev": 0, "test": 0, "bc8": 0, "other": 0}
    seen = anomalies = over = source_diff = 0
    with new_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            seen += 1
            if row.get("anomaly"):
                anomalies += 1
            lengths = row.get("encoded_length_by_view") or {}
            if lengths and max(lengths.values()) > 512:
                over += 1
            views = row.get("views") or {}
            bare = {TYPE_MARKER.sub("", text) for text in views.values()}
            if len(bare) > 1:
                source_diff += 1
            previous = historical.get(row["candidate_id"])
            if previous and previous.get("fragments") != row.get("fragments"):
                changed[row.get("split") or "other"] = changed.get(row.get("split") or "other", 0) + 1
    return {
        "contexts": seen,
        "anomalies": anomalies,
        "over_512": over,
        "source_text_differs_across_levels": source_diff,
        "changed_windows": changed,
        "sha256": _sha(new_path),
    }


def _validation(root: Path) -> None:
    for directory in root.glob("*/*/*/*"):
        if not directory.is_dir():
            continue
        complete = directory / "COMPLETE"
        if complete.is_file():
            payload = json.loads(complete.read_text(encoding="utf-8"))
            (directory / "VALIDATION.json").write_text(json.dumps({"ready": True, "requests": payload.get("requests", 0)}) + "\n", encoding="utf-8")


def main() -> None:
    concept_sha = _sha(CONCEPTS)
    if concept_sha != FROZEN:
        raise SystemExit(f"Concept fingerprint {concept_sha} is not the frozen mapping.")
    displayed = _display()
    code = Path(__import__("os").environ.get("SCHEMA_STUDY_CODE_ROOT") or "/home/b5ac/freddieyu.b5ac/project_1")
    views = {level: {} for level in LEVELS}
    summaries = {}
    for corpus in ("biored", "bc8"):
        output = STORAGE / "data/inputs" / corpus / VERSION
        build_context_tables(
            STORAGE / "data/prepared" / corpus / f"{corpus}_candidates_v2",
            output,
            corpus=corpus,
            tokenizer_dir=TOKENIZER,
            code_root=code,
            workers=8,
            input_version=VERSION,
            views_override=views,
            display_by_mention=_mention_map(corpus, displayed),
        )
        summaries[corpus] = _compare(corpus, output / "contexts.jsonl")
        print(json.dumps({"corpus": corpus, **summaries[corpus]}), flush=True)
    contexts = {
        "biored": STORAGE / "data/inputs/biored" / VERSION / "contexts.jsonl",
        "bc8": STORAGE / "data/inputs/bc8" / VERSION / "contexts.jsonl",
    }
    manifest = {
        "representation_version": "mesh_ancestor_v1",
        "concepts_sha256": concept_sha,
        "contexts_ready": False,
        "prompts_ready": False,
        "acceptance": {"contexts_passed": False, "prompts_passed": False, "validator": None},
        "contexts": summaries,
        "rerun_training_e0_e2": any(summaries[corpus]["changed_windows"].get(split, 0) for corpus in summaries for split in ("train", "dev")),
        "rerun_inference_e0_e2": any(summaries[corpus]["changed_windows"].get(split, 0) for corpus in summaries for split in ("test", "bc8")),
        "common_window": "one fragment set must fit the longest E0-E8 rendering inside 512 tokens",
    }
    destination = STUDY / "reports/representation_corrected/input_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    entities = LEVELS
    from schema_study.generation_inputs import write_few_shot, write_zero_shot

    for corpus in ("biored", "bc8"):
        write_zero_shot(corpus, output=PROMPT_ROOT, entities=entities, contexts=contexts)
        write_few_shot(corpus, output=PROMPT_ROOT, entities=entities, contexts=contexts)
    _validation(PROMPT_ROOT)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"built": True, "accepted": False, "rerun_training_e0_e2": manifest["rerun_training_e0_e2"]}), flush=True)


if __name__ == "__main__":
    main()
