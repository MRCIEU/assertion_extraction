"""Transitions the controller can make without another development session.

The functions here are the acceptance and reuse decisions. They do not treat a
filename, a scheduler state, or a builder's own flag as scientific acceptance.
"""

from __future__ import annotations

import json
from pathlib import Path

from schema_study.batch_plan import prompt_sha256

REQUIRED_COUNTS = {
    "principal_training": 360,
    "principal_prompts": 304,
    "principal_evaluations": 684,
}
EQUIVALENCE_METHODS = {"encoder_rendered_view"}


def expected_cells() -> dict[str, list[dict]]:
    """The frozen matrix. This inventory is not derived from the operational manifest."""
    entities = tuple(f"E{level}" for level in range(9))
    predicates = ("P0", "P1", "P2", "P3")
    seeds = (13, 42, 2026)
    supervised = ("biomedbert", "qwen8b", "qwen32b")
    training = []
    evaluations = []
    for model in supervised:
        for entity in entities:
            for predicate in predicates:
                for seed in seeds:
                    training.append({"model": model, "dataset": "biored", "entity": entity, "predicate": predicate, "seed": seed})
                    for corpus in ("biored", "bc8"):
                        evaluations.append({"model": model, "dataset": corpus, "entity": entity, "predicate": predicate, "seed": seed, "kind": "supervised_evaluation"})
        for predicate in predicates:
            for seed in seeds:
                training.append({"model": model, "dataset": "drugprot", "entity": "native", "predicate": predicate, "seed": seed})
                evaluations.append({"model": model, "dataset": "drugprot", "entity": "native", "predicate": predicate, "seed": seed, "kind": "supervised_evaluation"})
    prompts = []
    for model in ("qwen8b", "qwen32b"):
        for dataset in ("biored", "bc8"):
            for entity in entities:
                for predicate in predicates:
                    for strategy in ("zero_shot", "few_shot"):
                        prompts.append({"model": model, "dataset": dataset, "entity": entity, "predicate": predicate, "strategy": strategy})
        for predicate in predicates:
            for strategy in ("zero_shot", "few_shot"):
                prompts.append({"model": model, "dataset": "drugprot", "entity": "native", "predicate": predicate, "strategy": strategy})
    cells = {"training": training, "prompts": prompts, "evaluations": evaluations}
    counts = {f"principal_{name}" if name != "evaluations" else "principal_evaluations": len(rows) for name, rows in (("training", training), ("prompts", prompts), ("evaluations", evaluations))}
    if counts != REQUIRED_COUNTS:
        raise RuntimeError(f"Canonical cells {counts} do not match the frozen denominators {REQUIRED_COUNTS}.")
    return cells


def merge_acceptance(existing: dict | None, *, contexts: dict | None = None, prompts: dict | None = None, model_inputs: dict | None = None) -> dict:
    """Update one acceptance branch. A context run must not clear prompt acceptance, or the reverse."""
    body = dict(existing or {})
    acceptance = dict(body.get("acceptance") or {})
    if contexts is not None:
        acceptance["contexts_passed"] = contexts.get("passed") is True
        acceptance["context_results"] = {key: contexts[key] for key in contexts if key != "rows"}
        acceptance["validator"] = "input_acceptance"
        if contexts.get("concepts_sha256"):
            body["concepts_sha256"] = contexts["concepts_sha256"]
    if prompts is not None:
        acceptance["prompts_passed"] = prompts.get("passed") is True
        acceptance["prompt_results"] = prompts
        acceptance["prompt_validator"] = "input_acceptance"
    if model_inputs is not None:
        acceptance["model_inputs_passed"] = model_inputs.get("passed") is True
        acceptance["model_input_fingerprints"] = model_inputs.get("fingerprints")
    body["acceptance"] = acceptance
    body["contexts_ready"] = acceptance.get("contexts_passed") is True
    body["prompts_ready"] = acceptance.get("prompts_passed") is True
    return body


def artifact_valid(path: Path) -> bool:
    """A path is an artifact only when the weights and, for an adapter, the configuration are present."""
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0 and path.suffix == ".pt"
    if not path.is_dir():
        return False
    config = path / "adapter_config.json"
    weights = path / "adapter_model.safetensors"
    if not weights.is_file():
        weights = path / "adapter_model.bin"
    return config.is_file() and weights.is_file() and weights.stat().st_size > 0


def resolve_equivalence(disposition: str | None, predecessor_state: str | None, artifact_ok: bool, evidence_ok: bool) -> str:
    """The next automatic state. A running predecessor is a wait, not a permanent hold."""
    if disposition == "retrain" or not evidence_ok:
        return "retrain"
    if disposition not in {"reuse", "reuse_when_complete"}:
        return "unclassified"
    if predecessor_state in {"running", "pending", "submitted", "valid_continuation_required"}:
        return "wait"
    if predecessor_state == "training_completed" and artifact_ok:
        return "reuse"
    return "retrain"


def assess_prediction_rows(rows: list[dict], expected_ids: set[str], corpus: str) -> dict:
    """Identity-set acceptance of one final ledger. A positive row count is not a population."""
    observed = []
    wrong_corpus = 0
    for row in rows:
        if row.get("dataset") not in {None, corpus}:
            continue
        identifier = row.get("candidate_id")
        if identifier:
            observed.append(identifier)
    duplicate = len(observed) - len(set(observed))
    missing = len(expected_ids - set(observed))
    unexpected = len(set(observed) - expected_ids)
    return {
        "passed": wrong_corpus == 0 and duplicate == 0 and missing == 0 and unexpected == 0 and len(observed) == len(expected_ids) and len(expected_ids) > 0,
        "rows": len(rows),
        "duplicate_candidates": duplicate,
        "missing_candidates": missing,
        "unexpected_candidates": unexpected,
        "wrong_corpus": wrong_corpus,
    }


def publish_ledger_acceptance(directory: Path, corpus: str, rows: list[dict], expected_ids: set[str]) -> dict:
    """Write the identity-set record. The controller later checks that the ledger bytes still match."""
    import hashlib
    import json

    outcome = assess_prediction_rows(rows, expected_ids, corpus)
    ledger = directory / "predictions.jsonl"
    digest = hashlib.sha256()
    if ledger.is_file():
        with ledger.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    record = {**outcome, "corpus": corpus, "ledger_sha256": digest.hexdigest(), "expected_count": len(expected_ids)}
    destination = directory / "population_acceptance.json"
    body = {}
    if destination.is_file():
        body = json.loads(destination.read_text(encoding="utf-8"))
    body[corpus] = record
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return record


def _selected_checkpoint_fingerprint(training_output: str | None, workdir: Path) -> str | None:
    """Hash one selected checkpoint. Repeated ledgers from the same model reuse the hash."""
    import hashlib
    import json

    if not training_output:
        return None
    root = Path(training_output)
    path = next((
        candidate for candidate in (
            root / "checkpoints/best/checkpoint.pt",
            root / "best_adapter" / "adapter_model.safetensors",
            root / "best" / "adapter_checkpoint.pt",
        )
        if candidate.is_file() and candidate.stat().st_size > 0
    ), None)
    if path is None:
        return None
    cache_path = workdir / "checkpoint_fingerprints.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    stamp = f"{path.stat().st_size}:{path.stat().st_mtime_ns}"
    key = str(path)
    cached = cache.get(key) or {}
    if cached.get("stamp") == stamp and cached.get("sha256"):
        return cached["sha256"]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    cache[key] = {"stamp": stamp, "sha256": digest.hexdigest()}
    temporary = cache_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(cache) + "\n", encoding="utf-8")
    temporary.replace(cache_path)
    return cache[key]["sha256"]


def accept_completed_ledgers(task: dict, workdir: Path) -> dict:
    """Validate finished ledgers in place. This does not load a model or repeat inference."""
    from schema_study.input_acceptance import _candidate_authority
    from schema_study.production_plan import execution_manifest
    from schema_study.production_runtime import publish_scientific_status

    import json

    del task
    workdir.mkdir(parents=True, exist_ok=True)
    from schema_study.learning_conditions import HISTORICAL_PREDICATES
    from schema_study.schemas import default_schema_path, load_schema
    from schema_study.paths import get_paths

    authorities = {}
    schemas = {}
    outputs = {item["task_id"]: item.get("output") for item in execution_manifest()}
    written = 0
    rejected = 0
    provenance_written = 0
    for item in execution_manifest():
        if item.get("family") not in {"zero_shot", "few_shot", "adapter_inference", "final_evaluation"}:
            continue
        ledger = Path(item.get("output") or "") / "predictions.jsonl"
        if not ledger.is_file():
            continue
        corpus = str(item.get("dataset") or "")
        directory = Path(item["output"])
        population_ok = ledger_acceptance_valid(directory, corpus)
        provenance_ok = scientific_provenance_valid(directory, corpus)
        if population_ok and provenance_ok:
            continue
        stat = ledger.stat()
        if not population_ok:
            if corpus not in authorities:
                authorities[corpus] = _candidate_authority(corpus)
            split = {"biored": "test", "bc8": "bc8", "drugprot": "eval"}[corpus]
            expected = {identifier for identifier, row in authorities[corpus].items() if row.get("split") == split}
            outcome = _assess_ledger_file(ledger, expected, corpus)
            publish_ledger_acceptance(directory, corpus, [], expected)
            record_path = directory / "population_acceptance.json"
            body = json.loads(record_path.read_text(encoding="utf-8"))
            body[corpus] = {
                **body.get(corpus, {}),
                **outcome,
                "corpus": corpus,
                "expected_count": len(expected),
                "ledger_sha256": body.get(corpus, {}).get("ledger_sha256"),
                "ledger_size": stat.st_size,
                "ledger_mtime_ns": stat.st_mtime_ns,
            }
            temporary = record_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
            temporary.replace(record_path)
            written += 1
            rejected += 0 if outcome["passed"] else 1
        elif not (directory / "population_acceptance.json").is_file():
            continue
        else:
            record_path = directory / "population_acceptance.json"
            body = json.loads(record_path.read_text(encoding="utf-8"))
            cell = body.get(corpus) if isinstance(body.get(corpus), dict) else {}
            if cell.get("ledger_size") != stat.st_size or cell.get("ledger_mtime_ns") != stat.st_mtime_ns:
                cell["ledger_size"] = stat.st_size
                cell["ledger_mtime_ns"] = stat.st_mtime_ns
                body[corpus] = cell
                temporary = record_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
                temporary.replace(record_path)
        if provenance_ok:
            continue
        if corpus not in schemas:
            schemas[corpus] = load_schema(default_schema_path(get_paths().code_root, "drugprot" if corpus == "drugprot" else "biored"))
        level = item.get("schema_predicate") or HISTORICAL_PREDICATES.get(item.get("predicate"), item.get("predicate"))
        try:
            labels = schemas[corpus].level_labels(level)
        except KeyError:
            labels = ()
        legal = {label.stable_id for label in labels} | {label.display_name for label in labels}
        legal.update({schemas[corpus].no_relation_id, schemas[corpus].no_relation_name})
        def _matching_rows():
            with ledger.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("dataset") in {None, corpus}:
                        yield row

        provenance = assess_scientific_provenance(_matching_rows(), item, legal)
        fingerprint = _selected_checkpoint_fingerprint(outputs.get(item.get("trains_from")), workdir)
        provenance.update({
            "corpus": corpus,
            "ledger_size": stat.st_size,
            "ledger_mtime_ns": stat.st_mtime_ns,
            "checkpoint_fingerprint": fingerprint,
            "checkpoint_role": "selected",
        })
        if item.get("family") in {"adapter_inference", "final_evaluation"} and not fingerprint:
            provenance["provenance_passed"] = False
            provenance["reason"] = "checkpoint_unresolved"
        destination = directory / "scientific_provenance.json"
        existing = json.loads(destination.read_text(encoding="utf-8")) if destination.is_file() else {}
        existing[corpus] = provenance
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
        provenance_written += 1
    publish_scientific_status(workdir, {
        "scientific": "inference_completed",
        "ledgers_validated": written,
        "ledgers_rejected": rejected,
        "provenance_written": provenance_written,
    })
    return {"completed": True, "status": "inference_completed", "ledgers_validated": written, "ledgers_rejected": rejected, "provenance_written": provenance_written}


def _assess_ledger_file(path: Path, expected_ids: set[str], corpus: str) -> dict:
    observed = []
    contract_failures = 0
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") not in {None, corpus}:
                continue
            rows += 1
            status = row.get("output_status") or row.get("status")
            if status not in {None, "valid", "valid_empty"}:
                contract_failures += 1
            identifier = row.get("candidate_id")
            if identifier:
                observed.append(identifier)
    duplicate = len(observed) - len(set(observed))
    missing = len(expected_ids - set(observed))
    unexpected = len(set(observed) - expected_ids)
    contract = rows > 0 and contract_failures == rows
    passed = not contract and duplicate == 0 and missing == 0 and unexpected == 0 and len(observed) == len(expected_ids) and len(expected_ids) > 0
    return {
        "passed": passed,
        "rows": rows,
        "duplicate_candidates": duplicate,
        "missing_candidates": missing,
        "unexpected_candidates": unexpected,
        "contract_failures": contract_failures,
        "reason": "output_contract" if contract else ("identity" if not passed else "accepted"),
    }


def ledger_acceptance_valid(directory: Path, corpus: str) -> bool:
    """A controller tick may trust an acceptance record only while the ledger bytes are unchanged."""
    import hashlib
    import json

    record_path = directory / "population_acceptance.json"
    ledger = directory / "predictions.jsonl"
    if not record_path.is_file() or not ledger.is_file():
        return False
    body = json.loads(record_path.read_text(encoding="utf-8"))
    record = body.get(corpus) if isinstance(body.get(corpus), dict) else body
    if record.get("passed") is not True or record.get("corpus") != corpus:
        return False
    stat = ledger.stat()
    if record.get("ledger_size") != stat.st_size or record.get("ledger_mtime_ns") != stat.st_mtime_ns:
        digest = hashlib.sha256()
        with ledger.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        if record.get("ledger_sha256") != digest.hexdigest():
            return False
    return int(record.get("expected_count") or 0) > 0 and record.get("missing_candidates") == 0


def assess_scientific_provenance(rows: list[dict], task: dict, legal_ids: set[str]) -> dict:
    """Population identity is not scientific acceptance. A missing output status is not a valid prediction."""
    missing_status = 0
    illegal = 0
    checked = 0
    missing_prompt = 0
    generative = task.get("family") in {"zero_shot", "few_shot", "adapter_inference"}
    for row in rows:
        if row.get("dataset") not in {None, task.get("dataset")}:
            continue
        checked += 1
        status = row.get("output_status") or row.get("status")
        if status not in {"valid", "valid_empty", "invalid"}:
            missing_status += 1
            continue
        labels = row.get("predicted_labels")
        if not isinstance(labels, list):
            illegal += 1
            continue
        if status == "valid" and (not labels or any(label not in legal_ids for label in labels) or row.get("illegal_labels")):
            illegal += 1
        elif status == "valid_empty" and labels:
            illegal += 1
        if generative and not row.get("rendered_sha256"):
            missing_prompt += 1
    return {
        "provenance_passed": checked > 0 and missing_status == 0 and illegal == 0 and missing_prompt == 0,
        "rows": checked,
        "missing_output_status": missing_status,
        "illegal_predictions": illegal,
        "missing_prompt_fingerprint": missing_prompt,
        "model": task.get("model"),
        "representation_version": task.get("representation_version"),
        "predicate": task.get("predicate"),
        "schema_predicate": task.get("schema_predicate"),
        "trains_from": task.get("trains_from"),
        "parser": "interpret_generation" if generative else "encoder_threshold",
        "inference_backend": "transformers" if task.get("family") == "adapter_inference" else task.get("family"),
    }


def scientific_provenance_valid(directory: Path, corpus: str) -> bool:
    """Trust the provenance record only while the ledger bytes are unchanged. This does not reread the ledger."""
    import json

    record_path = directory / "scientific_provenance.json"
    ledger = directory / "predictions.jsonl"
    if not record_path.is_file() or not ledger.is_file():
        return False
    body = json.loads(record_path.read_text(encoding="utf-8"))
    record = body.get(corpus) if isinstance(body.get(corpus), dict) else None
    if not isinstance(record, dict) or record.get("provenance_passed") is not True or record.get("corpus") != corpus:
        return False
    stat = ledger.stat()
    return record.get("ledger_size") == stat.st_size and record.get("ledger_mtime_ns") == stat.st_mtime_ns


def continuation_is_verified(restored_step: int, saved_step: int, optimizer_loaded: bool, scheduler_step: int, production_requests: int) -> bool:
    """A weight load without the optimiser, or a toy string, is not a training resume."""
    return restored_step == saved_step and optimizer_loaded and scheduler_step == saved_step and production_requests >= 1


def assess_prompt_requests(rows: list[dict], expected_ids: set[str], entity: str, predicate: str) -> dict:
    """One shard population. A COMPLETE marker is not consulted."""
    seen = []
    bad = 0
    for row in rows:
        seen.append(row.get("candidate_id"))
        reasons = []
        if row.get("entity_level") != entity or row.get("predicate_level") != predicate:
            reasons.append("level")
        rendered = row.get("rendered")
        if not isinstance(rendered, str) or not rendered:
            reasons.append("rendered")
        elif row.get("rendered_sha256") and prompt_sha256(rendered) != row.get("rendered_sha256"):
            reasons.append("checksum")
        if row.get("request_id") != f"{row.get('dataset')}:{row.get('candidate_id')}:{entity}:{predicate}":
            reasons.append("request_id")
        if reasons:
            bad += 1
    observed = [identifier for identifier in seen if identifier]
    duplicate = len(observed) - len(set(observed))
    missing = len(expected_ids - set(observed))
    unexpected = len(set(observed) - expected_ids)
    return {
        "passed": bad == 0 and duplicate == 0 and missing == 0 and unexpected == 0 and len(observed) == len(expected_ids),
        "duplicate_requests": duplicate,
        "missing_requests": missing,
        "unexpected_requests": unexpected,
        "invalid_requests": bad,
    }
