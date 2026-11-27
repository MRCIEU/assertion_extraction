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


def accept_completed_ledgers(task: dict, workdir: Path) -> dict:
    """Validate finished ledgers in place. This does not load a model or repeat inference."""
    from schema_study.input_acceptance import _candidate_authority
    from schema_study.production_plan import execution_manifest
    from schema_study.production_runtime import publish_scientific_status

    import json

    del task
    workdir.mkdir(parents=True, exist_ok=True)
    authorities = {}
    written = 0
    rejected = 0
    for item in execution_manifest():
        if item.get("family") not in {"zero_shot", "few_shot", "adapter_inference", "final_evaluation"}:
            continue
        ledger = Path(item.get("output") or "") / "predictions.jsonl"
        if not ledger.is_file():
            continue
        corpus = str(item.get("dataset") or "")
        if ledger_acceptance_valid(Path(item["output"]), corpus):
            continue
        if corpus not in authorities:
            authorities[corpus] = _candidate_authority(corpus)
        split = {"biored": "test", "bc8": "bc8", "drugprot": "eval"}[corpus]
        expected = {identifier for identifier, row in authorities[corpus].items() if row.get("split") == split}
        outcome = _assess_ledger_file(ledger, expected, corpus)
        publish_ledger_acceptance(Path(item["output"]), corpus, [], expected)
        record_path = Path(item["output"]) / "population_acceptance.json"
        body = json.loads(record_path.read_text(encoding="utf-8"))
        body[corpus] = {**body.get(corpus, {}), **outcome, "corpus": corpus, "expected_count": len(expected), "ledger_sha256": body.get(corpus, {}).get("ledger_sha256")}
        temporary = record_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        temporary.replace(record_path)
        written += 1
        rejected += 0 if outcome["passed"] else 1
    publish_scientific_status(workdir, {"scientific": "inference_completed", "ledgers_validated": written, "ledgers_rejected": rejected})
    return {"completed": True, "status": "inference_completed", "ledgers_validated": written, "ledgers_rejected": rejected}


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
    digest = hashlib.sha256()
    with ledger.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return record.get("ledger_sha256") == digest.hexdigest() and int(record.get("expected_count") or 0) > 0 and record.get("missing_candidates") == 0


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
