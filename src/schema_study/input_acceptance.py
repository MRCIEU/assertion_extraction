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
    "bc8": {"all": 28734, "bc8": 28734},
}
ALLOWED_SPLITS = {"train", "dev", "test", "bc8"}
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
    for level in levels:
        value = lengths.get(level) if isinstance(lengths, dict) else None
        if value is None:
            reasons.append("encoded_length_missing:" + level)
            break
        try:
            number = int(value)
        except (TypeError, ValueError):
            reasons.append("encoded_length_missing:" + level)
            break
        if number <= 0 or number > 512:
            reasons.append("encoded_length:" + level)
            break
    if not row.get("candidate_id"):
        reasons.append("missing_candidate_id")
    if not row.get("document_id"):
        reasons.append("missing_document_id")
    if row.get("split") not in ALLOWED_SPLITS:
        reasons.append("bad_split")
    if any("<<H>>" not in views[level] or "<<T>>" not in views[level] for level in present):
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


def validate_context_file(
    path: Path,
    expected_ids: set[str],
    corpus: str,
    mention_display: dict | None = None,
    mention_text: dict | None = None,
    authority: dict | None = None,
) -> dict:
    """Stream one generated context file. This does not trust the builder's own ready flag."""
    rows = []
    splits: dict[str, int] = {}
    identity_failures = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            splits[row.get("split") or "missing"] = splits.get(row.get("split") or "missing", 0) + 1
            if authority is not None:
                record = authority.get(row.get("candidate_id"))
                if record is None or record.get("document_id") != row.get("document_id") or record.get("split") != row.get("split"):
                    identity_failures += 1
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
    for row in rows:
        if type_marker_reasons(row, mention_display or {}, mention_text):
            type_failures += 1
    population["type_failures"] = type_failures
    population["identity_failures"] = identity_failures
    population["passed"] = bool(population["passed"] and not count_reasons and type_failures == 0 and identity_failures == 0)
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
    from schema_study.programme_closure import EQUIVALENCE_METHODS, artifact_valid

    if record.get("rendered_input_match") is not True:
        return None
    if record.get("equivalence_method") not in EQUIVALENCE_METHODS:
        return None
    artifact = Path(str(record.get("artifact") or ""))
    if not artifact_valid(artifact):
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
        "release_evaluation": record.get("evaluation_rendered_match") is not True,
        "selection_protocol": record.get("selection_protocol"),
        "equivalence_method": record.get("equivalence_method"),
    }


def classify_equivalence(train_dev_changed: bool, model: str, selection_protocol: str) -> str:
    """Rendered-view identity is an encoder input. It is not a Qwen token sequence."""
    if train_dev_changed or str(model).startswith("qwen") or selection_protocol == "unrecoverable":
        return "retrain"
    return "reuse"


def type_marker_reasons(row: dict, mention_display: dict, mention_text: dict | None = None, levels: tuple[str, ...] = LEVELS) -> list[str]:
    """The displayed label must wrap the target mention. A label elsewhere in the window is not enough."""
    if not mention_display:
        return ["mention_map_missing"]
    views = row.get("views") or {}
    mentions = row.get("anchor_mention_ids") or row.get("marked_mention_ids") or []
    if len(mentions) < 2:
        return ["anchor_mentions"]
    chosen = mentions[:2]
    if mention_text is not None and any(not mention_text.get(mention) for mention in chosen):
        return ["mention_text_missing"]

    def occurrences(view: str, role: str) -> list[tuple[str | None, str]]:
        found = []
        cursor = 0
        token = f"<<{role}>>"
        while True:
            opened = view.find(token, cursor)
            if opened < 0:
                return found
            rest = view[opened + len(token):]
            label = None
            if rest.startswith("<<TYPE:"):
                end = rest.find(">>")
                if end < 0:
                    return found
                label = rest[len("<<TYPE:"):end]
                rest = rest[end + 2:]
            closed = rest.find(f"<</{role}>>")
            if closed < 0:
                return found
            found.append((label, rest[:closed]))
            cursor = opened + len(token)

    for level in levels:
        view = views.get(level) or ""
        if mention_text is None:
            for mention in chosen:
                label = (mention_display.get(mention) or {}).get(level)
                if label and f"<<TYPE:{label}>>" not in view:
                    return [f"type:{level}"]
            continue
        parsed = [item for role in ("H", "T") for item in occurrences(view, role)]
        if not parsed:
            return [f"type:{level}"]
        for mention in chosen:
            label = (mention_display.get(mention) or {}).get(level)
            body_text = mention_text[mention]
            if not any(body_text in body and (got == label if label else got is None) for got, body in parsed):
                return [f"type:{level}"]
    return []


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
        predecessor_state = (state.get(predecessor["task_id"]) or {}).get("state")
        method = "encoder_rendered_view" if task["model"] == "biomedbert" and not changed else None
        if label == "reuse" and predecessor_state in {"terminal_failure", "retryable_failure"} :
            label = "retrain"
        disposition[task["task_id"]] = label
        equivalence[task["task_id"]] = {
            "predecessor": predecessor["task_id"],
            "source_family": "principal_training",
            "model": task["model"],
            "rendered_input_match": not changed,
            "evaluation_rendered_match": level_changes["test"] == 0,
            "equivalence_method": method,
            "selection_protocol": protocol,
            "artifact": str(artifact),
        }
    return {"cell_disposition": disposition, "equivalence": equivalence, "window_changes": changes}


def _concept_sha256() -> str:
    path = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/reports/representation_corrected/concepts.jsonl")
    if not path.is_file():
        return ""
    return _sha(path)


def _candidate_authority(corpus: str) -> dict[str, dict]:
    """Approved prepared candidates. A historical context file is not this authority."""
    import pyarrow.parquet as pq

    name = "biored_candidates_v2" if corpus == "biored" else "bc8_candidates_v2"
    path = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/data/prepared") / corpus / name / "candidates.parquet"
    table = pq.read_table(path, columns=["candidate_id", "document_id", "split"])
    rows = table.to_pydict()
    return {
        rows["candidate_id"][index]: {"document_id": rows["document_id"][index], "split": rows["split"][index]}
        for index in range(table.num_rows)
    }


def _mention_text(corpus: str) -> dict:
    storage = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
    path = storage / "data/prepared" / corpus / f"{corpus}_candidates_v2" / "entities.jsonl"
    mapping = {}
    if not path.is_file():
        return mapping
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("mention_id") and row.get("text"):
            mapping[row["mention_id"]] = row["text"]
    return mapping


def context_files_complete() -> bool:
    """Both corrected context files contain the approved row counts. A non-empty file is not complete."""
    storage = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/data/inputs")
    version = "target_anchored_mesh_ancestor_v1__biomedbert_e1354b7a"
    for corpus, expected in (("biored", 35136), ("bc8", 28734)):
        path = storage / corpus / version / "contexts.jsonl"
        if not path.is_file():
            return False
        count = sum(1 for line in path.open(encoding="utf-8") if line.strip())
        if count != expected:
            return False
    return True


def validate_publication(task: dict, workdir: Path) -> dict:
    """Validate one branch. Context acceptance does not reset prompt acceptance."""
    if task.get("role") == "mesh_ancestor_v1_prompts":
        return _validate_prompts(task, workdir)
    return _validate_contexts(task, workdir)


def _manifest_path() -> Path:
    return Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/reports/representation_corrected/input_manifest.json")


def _load_manifest(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_contexts(task: dict, workdir: Path) -> dict:
    from schema_study.programme_closure import merge_acceptance
    from schema_study.production_runtime import publish_scientific_status

    storage = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
    version = "target_anchored_mesh_ancestor_v1__biomedbert_e1354b7a"
    destination = _manifest_path()
    workdir.mkdir(parents=True, exist_ok=True)
    measured = _concept_sha256()
    results = {}
    passed = measured == FROZEN_CONCEPT_SHA256
    if measured != FROZEN_CONCEPT_SHA256:
        results["concepts"] = {"passed": False, "measured": measured}
    for corpus in ("biored", "bc8"):
        generated = storage / "data/inputs" / corpus / version / "contexts.jsonl"
        if not generated.is_file():
            results[corpus] = {"passed": False, "reason": "context_file_missing"}
            passed = False
            continue
        authority = _candidate_authority(corpus)
        outcome = validate_context_file(
            generated, set(authority), corpus, _mention_display(corpus), _mention_text(corpus), authority,
        )
        results[corpus] = {key: value for key, value in outcome.items() if key != "examples"}
        results[corpus]["examples"] = outcome["examples"]
        passed = passed and outcome["passed"]
    disposition = _cell_dispositions() if passed else {}
    existing = _load_manifest(destination)
    payload = merge_acceptance(existing, contexts={"passed": passed, "results": results, "concepts_sha256": measured})
    payload["representation_version"] = "mesh_ancestor_v1"
    payload["contexts"] = results
    if disposition:
        payload["cell_disposition"] = disposition.get("cell_disposition") or {}
        payload["equivalence"] = disposition.get("equivalence") or {}
        payload["window_changes"] = disposition.get("window_changes") or {}
    publish_acceptance(destination, payload)
    status = "inference_completed" if passed else "invalid"
    publish_scientific_status(workdir, {"scientific": status, "predictions_usable": False, "contexts_passed": passed, "population_complete": passed})
    return {"completed": True, "status": status}


def _validate_prompts(task: dict, workdir: Path) -> dict:
    """Read shard bytes. A context acceptance record cannot set this branch."""
    from schema_study.programme_closure import assess_prompt_requests, merge_acceptance
    from schema_study.production_runtime import publish_scientific_status

    root = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/runs/formal/formal_matched_e0_e7/generation_inputs/mesh_ancestor_v1")
    workdir.mkdir(parents=True, exist_ok=True)
    failures = []
    checked = 0
    expected_by_corpus = {}
    version = "target_anchored_mesh_ancestor_v1__biomedbert_e1354b7a"
    storage = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/data/inputs")
    for corpus, split in (("biored", "test"), ("bc8", "bc8")):
        ids = set()
        path = storage / corpus / version / "contexts.jsonl"
        if path.is_file():
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") == split:
                    ids.add(row["candidate_id"])
        expected_by_corpus[corpus] = ids
    for source in ("zero_shot", "few_shot"):
        for corpus in ("biored", "bc8"):
            for entity in LEVELS:
                for predicate in ("P0", "P1", "P2", "P3"):
                    directory = root / source / corpus / entity / predicate
                    rows = []
                    if directory.is_dir():
                        for shard in sorted(directory.glob("shard_*.jsonl")):
                            for line in shard.open(encoding="utf-8"):
                                if line.strip():
                                    rows.append(json.loads(line))
                    outcome = assess_prompt_requests(rows, expected_by_corpus[corpus], entity, predicate)
                    checked += 1
                    if not outcome["passed"]:
                        failures.append({"directory": str(directory), **outcome})
    passed = checked == 144 and not failures
    destination = _manifest_path()
    existing = _load_manifest(destination)
    payload = merge_acceptance(existing, prompts={"passed": passed, "conditions": checked, "failures": failures[:8]})
    publish_acceptance(destination, payload)
    status = "inference_completed" if passed else "invalid"
    publish_scientific_status(workdir, {"scientific": status, "predictions_usable": False, "prompts_passed": passed, "population_complete": passed})
    return {"completed": True, "status": status, "prompts_passed": passed}


def prepare_contexts(task: dict, workdir: Path) -> dict:
    """Rebuild only an incomplete context file. A non-empty partial file is not finished."""
    import os
    import subprocess

    from schema_study.production_runtime import publish_scientific_status

    workdir.mkdir(parents=True, exist_ok=True)
    if task.get("role") == "mesh_ancestor_v1_prompts":
        return _prepare_prompts(workdir)
    if context_files_complete():
        publish_scientific_status(workdir, {"scientific": "inference_completed", "action": "complete_files_preserved"})
        return {"completed": True, "status": "inference_completed"}
    code = Path(os.environ.get("SCHEMA_STUDY_CODE_ROOT") or "/home/b5ac/freddieyu.b5ac/project_1")
    subprocess.run(["python", str(code / "scripts/build_mesh_ancestor_inputs.py")], check=True, cwd=code)
    complete = context_files_complete()
    publish_scientific_status(workdir, {"scientific": "inference_completed" if complete else "invalid", "action": "built", "population_complete": complete})
    return {"completed": True, "status": "inference_completed" if complete else "invalid"}


def _prepare_prompts(workdir: Path) -> dict:
    from schema_study.generation_inputs import write_few_shot, write_zero_shot
    from schema_study.production_runtime import publish_scientific_status

    version = "target_anchored_mesh_ancestor_v1__biomedbert_e1354b7a"
    storage = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/data/inputs")
    contexts = {corpus: storage / corpus / version / "contexts.jsonl" for corpus in ("biored", "bc8")}
    if not all(path.is_file() for path in contexts.values()):
        publish_scientific_status(workdir, {"scientific": "invalid", "reason": "contexts_absent"})
        return {"completed": True, "status": "invalid"}
    root = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/runs/formal/formal_matched_e0_e7/generation_inputs/mesh_ancestor_v1")
    entities = LEVELS
    for corpus in ("biored", "bc8"):
        write_zero_shot(corpus, output=root, entities=entities, contexts=contexts)
        write_few_shot(corpus, output=root, entities=entities, contexts=contexts)
    publish_scientific_status(workdir, {"scientific": "inference_completed", "action": "prompts_written", "predictions_usable": False})
    return {"completed": True, "status": "inference_completed"}
