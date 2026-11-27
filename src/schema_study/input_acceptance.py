"""Acceptance tests for corrected contexts. A finished build script is not acceptance."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

TYPE_MARKER = re.compile(r"<<TYPE:[^>]*>>")
TARGET_MARKER = re.compile(r"<<T>>")
LEVELS = tuple(f"E{i}" for i in range(9))
EXPECTED_COUNTS = {
    "biored": {"all": 35136, "train": 21755, "dev": 6334, "test": 7047},
    "bc8": {"all": 28734},
}
FROZEN_CONCEPT_SHA256 = "77fa5b41a6e18c277f987dedd74ea28833b8afa4c823b1396c935f5d7ee850a5"


def assess_context_record(row: dict, levels: tuple[str, ...] = LEVELS) -> list[str]:
    """Reasons a single context fails. An empty list is a controlled record."""
    reasons = []
    views = row.get("views") or {}
    missing = [level for level in levels if not isinstance(views.get(level), str) or not views.get(level)]
    if missing:
        reasons.append("missing_view:" + ",".join(missing))
    present = [level for level in levels if isinstance(views.get(level), str)]
    bare = {TYPE_MARKER.sub("", views[level]) for level in present}
    if len(bare) > 1:
        reasons.append("source_text_differs_across_levels")
    if row.get("anomaly"):
        reasons.append("anomaly:" + str(row.get("anomaly")))
    lengths = row.get("encoded_length_by_view") or {}
    if any(int(lengths.get(level) or 0) > 512 for level in levels):
        reasons.append("over_512")
    if not row.get("candidate_id"):
        reasons.append("missing_candidate_id")
    if not row.get("document_id"):
        reasons.append("missing_document_id")
    if row.get("split") not in {"train", "dev", "test"}:
        reasons.append("bad_split")
    if sum(1 for level in present if len(TARGET_MARKER.findall(views[level])) < 2) and present:
        reasons.append("target_markers_missing")
    if not row.get("fragments"):
        reasons.append("missing_source_window")
    return reasons


def assess_population(rows: list[dict], expected_ids: set[str], levels: tuple[str, ...] = LEVELS) -> dict:
    """Identity-set acceptance. Matching counts without matching identities is a failure."""
    seen = []
    row_failures = 0
    failure_examples = []
    for row in rows:
        seen.append(row.get("candidate_id"))
        reasons = assess_context_record(row, levels)
        if reasons:
            row_failures += 1
            if len(failure_examples) < 8:
                failure_examples.append({"candidate_id": row.get("candidate_id"), "reasons": reasons})
    observed = [identifier for identifier in seen if identifier]
    duplicate = len(observed) - len(set(observed))
    missing = sorted(expected_ids - set(observed))
    unexpected = sorted(set(observed) - expected_ids)
    passed = not row_failures and duplicate == 0 and not missing and not unexpected and len(observed) == len(expected_ids)
    return {
        "passed": passed,
        "rows": len(rows),
        "duplicate_candidates": duplicate,
        "missing_candidates": len(missing),
        "unexpected_candidates": len(unexpected),
        "invalid_rows": row_failures,
        "examples": failure_examples,
        "missing_example": missing[:5],
        "unexpected_example": unexpected[:5],
    }


def contexts_accepted(manifest: dict) -> bool:
    acceptance = manifest.get("acceptance") or {}
    return (
        manifest.get("concepts_sha256") == FROZEN_CONCEPT_SHA256
        and acceptance.get("contexts_passed") is True
        and acceptance.get("validator") == "input_acceptance"
    )


def prompts_accepted(manifest: dict) -> bool:
    acceptance = manifest.get("acceptance") or {}
    return contexts_accepted(manifest) and acceptance.get("prompts_passed") is True


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _mention_display(corpus: str) -> dict:
    storage = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
    concepts = storage / "studies/schema_granularity_v1/reports/representation_corrected/concepts.jsonl"
    displayed = {}
    for line in concepts.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("corpus") == corpus:
            displayed[str(row.get("concept_id") or "")] = row.get("displayed") or {}
    mapping = {}
    entities = storage / "data/prepared" / corpus / f"{corpus}_candidates_v2" / "entities.jsonl"
    if not entities.is_file():
        return {}
    for line in entities.open(encoding="utf-8"):
        row = json.loads(line)
        shown = displayed.get(str(row.get("concept_id") or ""))
        if shown and row.get("mention_id"):
            mapping[row["mention_id"]] = shown
    return mapping


def _expected_ids(historical: Path) -> set[str]:
    found = set()
    with historical.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                found.add(json.loads(line)["candidate_id"])
    return found


def validate_context_file(path: Path, expected_ids: set[str], corpus: str, mention_display: dict | None = None) -> dict:
    """Stream one generated context file. This does not trust the builder's own ready flag."""
    rows = []
    splits: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            splits[row.get("split") or "missing"] = splits.get(row.get("split") or "missing", 0) + 1
    population = assess_population(rows, expected_ids)
    expected = EXPECTED_COUNTS[corpus]
    count_reasons = []
    if population["rows"] != expected["all"]:
        count_reasons.append(f"count:{population['rows']}!={expected['all']}")
    for split, count in expected.items():
        if split == "all":
            continue
        if splits.get(split) != count:
            count_reasons.append(f"split:{split}:{splits.get(split)}!={count}")
    type_failures = 0
    if mention_display:
        for row in rows:
            if type_marker_reasons(row, mention_display):
                type_failures += 1
    population["type_failures"] = type_failures
    population["passed"] = bool(population["passed"] and not count_reasons and type_failures == 0)
    population["count_reasons"] = count_reasons
    population["splits"] = splits
    population["sha256"] = _sha(path)
    return population


def publish_acceptance(destination: Path, payload: dict) -> None:
    """Atomic publication. contexts_ready follows the acceptance result and nothing else."""
    body = dict(payload)
    acceptance = body.setdefault("acceptance", {})
    body["contexts_ready"] = acceptance.get("contexts_passed") is True
    body["prompts_ready"] = acceptance.get("prompts_passed") is True
    body["concepts_sha256"] = FROZEN_CONCEPT_SHA256
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def reuse_decision(task: dict, record: dict | None) -> dict | None:
    """Exact predecessor, exact family, and an existing artifact. A loose field match is not reuse."""
    if not record or not task.get("replaces"):
        return None
    if record.get("predecessor") != task.get("replaces"):
        return None
    if record.get("source_family") != "principal_training":
        return None
    if record.get("model") != task.get("model"):
        return None
    if record.get("training_tokens_match") is not True or record.get("development_tokens_match") is not True:
        return None
    artifact = Path(str(record.get("artifact") or ""))
    if artifact.is_dir():
        if not (artifact / "adapter_config.json").is_file() and not (artifact / "checkpoint.pt").is_file():
            return None
    elif not artifact.is_file():
        return None
    if record.get("selection_protocol") == "unrecoverable":
        return None
    return {
        "state": "completed",
        "scientific": "reused_historical_equivalent",
        "replaces": task["replaces"],
        "artifact": str(artifact),
        "artifact_sha256": record.get("artifact_sha256"),
        "source_task": task["replaces"],
        "release_evaluation": record.get("evaluation_tokens_match") is not True,
        "selection_protocol": record.get("selection_protocol"),
    }


def classify_equivalence(train_dev_changed: bool, model: str, selection_protocol: str) -> str:
    """One condition's disposition. A global window flag is not an argument."""
    if train_dev_changed:
        return "retrain"
    if str(model).startswith("qwen") and selection_protocol != "left_pad_online":
        return "retrain"
    if str(model).startswith("qwen"):
        return "reuse_when_complete"
    return "reuse"


def type_marker_reasons(row: dict, mention_display: dict, levels: tuple[str, ...] = LEVELS) -> list[str]:
    """The displayed label must appear in that level's rendered view."""
    reasons = []
    views = row.get("views") or {}
    for mention in row.get("marked_mention_ids") or []:
        shown = mention_display.get(mention) or {}
        for level in levels:
            label = shown.get(level)
            if not label:
                continue
            if f"<<TYPE:{label}>>" not in (views.get(level) or ""):
                reasons.append(f"type:{level}")
                return reasons
    return reasons


def _window_changes(new_path: Path, old_path: Path) -> dict[str, dict[str, int]]:
    """Per-level counts of E0–E2 rendered strings that differ from the stored contexts."""
    import hashlib

    levels = ("E0", "E1", "E2")

    def digest(text: object) -> str | None:
        if not isinstance(text, str):
            return None
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    historical = {}
    with old_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            historical[row["candidate_id"]] = (
                row.get("split"),
                {level: digest((row.get("views") or {}).get(level)) for level in levels},
            )
    changed = {level: {"train": 0, "dev": 0, "test": 0} for level in levels}
    with new_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            previous = historical.get(row.get("candidate_id"))
            if previous is None:
                continue
            split, old_views = previous
            bucket = split if split in {"train", "dev", "test"} else "test"
            views = row.get("views") or {}
            for level in levels:
                if old_views[level] != digest(views.get(level)):
                    changed[level][bucket] += 1
    return changed


def _left_pad(revision: str | None) -> bool:
    import subprocess

    if not revision:
        return False
    anchor = "6480077b0d34eb6cf41e8a23914812351563db0c"
    code = Path("/home/b5ac/freddieyu.b5ac/project_1")
    result = subprocess.run(
        ["git", "-C", str(code), "merge-base", "--is-ancestor", anchor, revision],
        check=False, capture_output=True,
    )
    return result.returncode == 0


def _cell_dispositions() -> dict:
    """Classify each corrected E0–E2 training condition. E3–E8 are always retrained."""
    from schema_study.production_plan import _historical_principal, corrected_principal_tasks, task_directory
    from schema_study.production_runtime import PROGRAMME_ROOT, load_controller_state

    storage = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
    version = "target_anchored_mesh_ancestor_v1__biomedbert_e1354b7a"
    changes = _window_changes(
        storage / "data/inputs/biored" / version / "contexts.jsonl",
        storage / "data/inputs/biored/target_anchored_v2__biomedbert_e1354b7a_matched_e0_e7/contexts.jsonl",
    )
    state = load_controller_state(PROGRAMME_ROOT / "controller_state.json")
    left_pad_revisions: dict[str, bool] = {}
    disposition = {}
    equivalence = {}
    for task in corrected_principal_tasks():
        if task.get("entity") not in {"E0", "E1", "E2"}:
            continue
        predecessor = _historical_principal(task["model"], "biored", task["entity"], task["predicate"], task["seed"])
        revision = (state.get(predecessor["task_id"]) or {}).get("manifest_revision")
        if task["model"] == "biomedbert":
            protocol = "encoder_loss"
        else:
            if revision not in left_pad_revisions:
                left_pad_revisions[revision] = _left_pad(revision)
            protocol = "left_pad_online" if left_pad_revisions[revision] else "unrecoverable"
        level_changes = changes[task["entity"]]
        changed = level_changes["train"] + level_changes["dev"] > 0
        label = classify_equivalence(changed, task["model"], protocol)
        artifact = Path(task_directory(predecessor)) / ("checkpoints/best/checkpoint.pt" if task["model"] == "biomedbert" else "best_adapter")
        finished = (state.get(predecessor["task_id"]) or {}).get("state") == "training_completed"
        if label == "reuse_when_complete" and finished and not artifact.exists():
            label = "retrain"
        if label == "reuse" and not artifact.exists():
            label = "retrain"
        disposition[task["task_id"]] = label
        if label == "reuse" or (label == "reuse_when_complete" and finished and artifact.exists()):
            equivalence[task["task_id"]] = {
                "predecessor": predecessor["task_id"],
                "source_family": "principal_training",
                "model": task["model"],
                "training_tokens_match": not changed,
                "development_tokens_match": not changed,
                "evaluation_tokens_match": level_changes["test"] == 0,
                "selection_protocol": protocol,
                "artifact": str(artifact),
            }
    return {"cell_disposition": disposition, "equivalence": equivalence, "window_changes": changes}


def validate_publication(task: dict, workdir: Path) -> dict:
    """Read the generated files and publish acceptance only when the checks pass."""
    from schema_study.production_runtime import publish_scientific_status

    storage = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
    study = storage / "studies/schema_granularity_v1"
    version = "target_anchored_mesh_ancestor_v1__biomedbert_e1354b7a"
    destination = study / "reports/representation_corrected/input_manifest.json"
    workdir.mkdir(parents=True, exist_ok=True)
    results = {}
    passed = True
    for corpus in ("biored", "bc8"):
        generated = storage / "data/inputs" / corpus / version / "contexts.jsonl"
        historical = storage / "data/inputs" / corpus / f"target_anchored_v2__biomedbert_e1354b7a_matched_e0_e7/contexts.jsonl"
        if not generated.is_file():
            results[corpus] = {"passed": False, "reason": "context_file_missing"}
            passed = False
            continue
        outcome = validate_context_file(generated, _expected_ids(historical), corpus, _mention_display(corpus))
        results[corpus] = {key: value for key, value in outcome.items() if key != "examples"}
        results[corpus]["examples"] = outcome["examples"]
        passed = passed and outcome["passed"]
    disposition = _cell_dispositions() if passed else {}
    payload = {
        "representation_version": "mesh_ancestor_v1",
        "concepts_sha256": FROZEN_CONCEPT_SHA256,
        "acceptance": {
            "validator": "input_acceptance",
            "contexts_passed": passed,
            "prompts_passed": False,
            "results": results,
        },
        "contexts": results,
        "cell_disposition": disposition.get("cell_disposition") or {},
        "equivalence": disposition.get("equivalence") or {},
    }
    publish_acceptance(destination, payload)
    status = "inference_completed" if passed else "invalid"
    publish_scientific_status(workdir, {"scientific": status, "predictions_usable": False, "contexts_passed": passed})
    return {"completed": True, "status": status}


def prepare_contexts(task: dict, workdir: Path) -> dict:
    """Build corrected contexts only when the generated file is absent. Do not overwrite a running build."""
    import os
    import subprocess

    from schema_study.production_runtime import publish_scientific_status

    storage = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
    version = "target_anchored_mesh_ancestor_v1__biomedbert_e1354b7a"
    existing = [storage / "data/inputs" / corpus / version / "contexts.jsonl" for corpus in ("biored", "bc8")]
    workdir.mkdir(parents=True, exist_ok=True)
    if all(path.is_file() and path.stat().st_size > 0 for path in existing):
        publish_scientific_status(workdir, {"scientific": "inference_completed", "action": "files_already_present"})
        return {"completed": True, "status": "inference_completed"}
    code = Path(os.environ.get("SCHEMA_STUDY_CODE_ROOT") or "/home/b5ac/freddieyu.b5ac/project_1")
    subprocess.run(["python", str(code / "scripts/build_mesh_ancestor_inputs.py")], check=True, cwd=code)
    publish_scientific_status(workdir, {"scientific": "inference_completed", "action": "built"})
    return {"completed": True, "status": "inference_completed"}
