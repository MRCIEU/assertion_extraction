"""Command line for preparation, validation, planning, and blocked model runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from schema_study.data import load_drugprot_directory, parse_pubtator, prepare_drugprot, prepare_pubtator_corpus
from schema_study.experiments import (
    ModelExecutionBlocked,
    blocked_generate,
    blocked_predict,
    blocked_train,
    generation_settings,
    plan_counts,
    plan_experiments,
    training_hyperparameters,
    write_experiment_table,
)
from schema_study.paths import get_paths, load_settings
from schema_study.records import environment_record, write_json, write_tsv
from schema_study.schemas import assert_partition, composed_equals_direct, default_schema_path, load_schema, partition_errors, random_merge_versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="schema-study")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-environment")
    sub.add_parser("validate-schema")
    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("--corpus", required=True, choices=["biored", "bc8", "drugprot"])
    build = sub.add_parser("build-inputs")
    build.add_argument("--corpus", required=True, choices=["biored", "bc8", "drugprot"])
    build.add_argument("--limit", type=int, default=None)
    build.add_argument("--workers", type=int, default=1)
    build.add_argument("--splits", default=None, help="Comma-separated split names.")
    build.add_argument("--shard-index", type=int, default=0)
    build.add_argument("--shard-count", type=int, default=1)
    build.add_argument("--input-version", default=None)
    build.add_argument("--include-unlabeled", action="store_true")
    sub.add_parser("plan-experiments")
    sub.add_parser("review-materials")
    sub.add_parser("cpu-validate")
    parse_cmd = sub.add_parser("parse")
    parse_cmd.add_argument("--input", required=True)
    parse_cmd.add_argument("--output", required=True)
    parse_cmd.add_argument("--dataset", required=True, choices=["biored", "drugprot"])
    parse_cmd.add_argument("--level", required=True)
    rollup_cmd = sub.add_parser("rollup")
    rollup_cmd.add_argument("--input", required=True)
    rollup_cmd.add_argument("--output", required=True)
    rollup_cmd.add_argument("--dataset", required=True, choices=["biored", "drugprot"])
    rollup_cmd.add_argument("--source-level", required=True)
    rollup_cmd.add_argument("--target-level", required=True)
    evaluate_cmd = sub.add_parser("evaluate")
    evaluate_cmd.add_argument("--predictions", required=True)
    evaluate_cmd.add_argument("--output", required=True)
    evaluate_cmd.add_argument("--dataset", required=True, choices=["biored", "drugprot"])
    evaluate_cmd.add_argument("--level", required=True)
    sub.add_parser("measure-granularity")
    sub.add_parser("prepare-source-references")
    sub.add_parser("prepare-representation-readiness")
    sub.add_parser("prepare-formal-protocol")
    sub.add_parser("prepare-study-records")
    sub.add_parser("prepare-experimental-inputs")
    matched = sub.add_parser("build-matched-inputs")
    matched.add_argument("--corpus", required=True, choices=["biored", "bc8"])
    matched.add_argument("--shard-index", type=int, required=True)
    matched.add_argument("--shard-count", type=int, required=True)
    matched.add_argument("--workers", type=int, default=8)
    merge = sub.add_parser("merge-matched-inputs")
    merge.add_argument("--corpus", required=True, choices=["biored", "bc8"])
    merge.add_argument("--shard-count", type=int, default=8)
    sub.add_parser("measure-qwen-lengths")
    sub.add_parser("finalise-experimental-inputs")
    report_cmd = sub.add_parser("report")
    report_cmd.add_argument("--output", required=True)
    report_cmd.add_argument("--synthetic", action="store_true")
    for name in ("train", "predict", "generate"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "check-environment":
            print(json.dumps(environment_record(), indent=2, sort_keys=True))
            return 0
        if args.command == "validate-schema":
            return _validate_schema()
        if args.command == "prepare-data":
            return _prepare(args.corpus)
        if args.command == "plan-experiments":
            return _plan()
        if args.command == "build-inputs":
            return _build_inputs(
                args.corpus,
                args.limit,
                args.workers,
                args.splits,
                args.shard_index,
                args.shard_count,
                args.input_version,
                args.include_unlabeled,
            )
        if args.command == "review-materials":
            return _review_materials()
        if args.command == "cpu-validate":
            return _cpu_validate()
        if args.command == "parse":
            return _parse(args.input, args.output, args.dataset, args.level)
        if args.command == "rollup":
            return _rollup(args.input, args.output, args.dataset, args.source_level, args.target_level)
        if args.command == "evaluate":
            return _evaluate(args.predictions, args.output, args.dataset, args.level)
        if args.command == "measure-granularity":
            from schema_study.measure_granularity import measure

            return measure()
        if args.command == "prepare-source-references":
            from schema_study.source_references import prepare

            return prepare()
        if args.command == "prepare-representation-readiness":
            from schema_study.representation_readiness import prepare

            return prepare()
        if args.command == "prepare-formal-protocol":
            from schema_study.formal_protocol import prepare

            return prepare()
        if args.command == "prepare-study-records":
            from schema_study.study_records import prepare

            return prepare()
        if args.command == "prepare-experimental-inputs":
            from schema_study.experimental_preparation import prepare

            return prepare()
        if args.command == "build-matched-inputs":
            from schema_study.experimental_preparation import build_matched_shard

            return build_matched_shard(args.corpus, args.shard_index, args.shard_count, args.workers)
        if args.command == "merge-matched-inputs":
            from schema_study.experimental_preparation import merge_matched_inputs

            print(json.dumps(merge_matched_inputs(args.corpus, args.shard_count), indent=2))
            return 0
        if args.command == "measure-qwen-lengths":
            from schema_study.qwen_lengths import main as measure_qwen_lengths

            return measure_qwen_lengths()
        if args.command == "finalise-experimental-inputs":
            from schema_study.experimental_preparation import finalise

            return finalise()
        if args.command == "report":
            return _report(args.output, args.synthetic)
        if args.command == "train":
            return _model_command("training", args.config, args.dry_run)
        if args.command == "predict":
            return _model_command("prediction", args.config, args.dry_run)
        if args.command == "generate":
            return _model_command("generation", args.config, args.dry_run)
    except ModelExecutionBlocked as exc:
        print(str(exc))
        return 3
    parser.error(f"Unhandled command {args.command}")
    return 2


def _read_yaml(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _validate_schema() -> int:
    paths = get_paths()
    failed = False
    for corpus in ("biored", "drugprot"):
        schema = load_schema(default_schema_path(paths.code_root, corpus))
        errors = partition_errors(schema)
        composed = composed_equals_direct(schema)
        print(f"{schema.schema_id}: errors={errors or 'none'} p4_p3_p2_equals_p4_p2={composed}")
        failed = failed or bool(errors) or not composed
        assert_partition(schema)
    biored = load_schema(default_schema_path(paths.code_root, "biored"))
    versions = random_merge_versions(list(biored.fine_ids))
    print(f"random_merge_versions={len(versions)} unique={len({frozenset(frozenset(group) for group in version) for version in versions})}")
    return 1 if failed else 0


def _prepare(corpus: str) -> int:
    paths = get_paths()
    if corpus == "biored":
        release = paths.raw / "BioRED" / "BIORED_2022-04-07_sha256-c3032230bd89" / "extracted" / "BioRED"
        splits = {"Train.PubTator": "train", "Dev.PubTator": "dev", "Test.PubTator": "test"}
        documents = []
        split_map = {}
        sources = []
        for name, split in splits.items():
            path = release / name
            sources.append(path)
            for document in parse_pubtator(path.read_text(encoding="utf-8")):
                documents.append(document)
                split_map[document["document_id"]] = split
        summary = prepare_pubtator_corpus(
            documents,
            load_schema(default_schema_path(paths.code_root, "biored")),
            corpus="biored",
            release_id="BIORED_2022-04-07_sha256-c3032230bd89",
            source_files=sources,
            output_dir=paths.prepared / "biored" / "biored_candidates_v2",
            split_of=split_map.get,
        )
    elif corpus == "bc8":
        path = (
            paths.raw
            / "BioRED_BC8"
            / "BC8_BIOREDIRECT_2025-01-22_sha256-29aa0b2060f9"
            / "extracted"
            / "datasets"
            / "bioredirect"
            / "bioredirect_bc8_test.pubtator"
        )
        documents = parse_pubtator(path.read_text(encoding="utf-8"))
        summary = prepare_pubtator_corpus(
            documents,
            load_schema(default_schema_path(paths.code_root, "biored")),
            corpus="bc8",
            release_id="BC8_BIOREDIRECT_2025-01-22_sha256-29aa0b2060f9",
            source_files=[path],
            output_dir=paths.prepared / "bc8" / "bc8_candidates_v2",
            split_of=lambda _document_id: "bc8",
        )
    else:
        root = paths.raw / "DrugProt"
        releases = sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
        if not releases:
            print(f"DrugProt raw directory is missing: {root}")
            return 2
        release = releases[-1]
        training = load_drugprot_directory(_drugprot_subdir(release, ["training"]))
        evaluation = load_drugprot_directory(_drugprot_subdir(release, ["development"]))
        unlabeled_dir = _drugprot_subdir(release, ["test-background"], required=False)
        unlabeled = load_drugprot_directory(unlabeled_dir) if unlabeled_dir else None
        sources = training["files"] + evaluation["files"] + (unlabeled["files"] if unlabeled else [])
        summary = prepare_drugprot(
            training,
            evaluation,
            load_schema(default_schema_path(paths.code_root, "drugprot")),
            release_id=release.name,
            source_files=sources,
            output_dir=paths.prepared / "drugprot" / "drugprot_candidates_v1",
            unlabeled=unlabeled,
        )
    print(json.dumps({key: summary[key] for key in summary if key != "separator"}, indent=2, default=str))
    return 0 if summary["validation_status"] != "fail" else 1


def _drugprot_subdir(release: Path, names: list[str], required: bool = True) -> Path | None:
    wanted = {name.lower() for name in names}
    matches = [path for path in release.rglob("*") if path.is_dir() and path.name.lower() in wanted]
    if len(matches) != 1:
        if required:
            raise FileNotFoundError(f"Could not find one DrugProt directory named {sorted(wanted)} under {release}")
        return None
    return matches[0]


def _model_command(action: str, config_path: str, dry_run: bool) -> int:
    from schema_study.models.runner import dry_run_training, execute_training
    from schema_study.models.schedule import ModelExecutionBlocked as Blocked
    from schema_study.models.schedule import assert_execution_allowed

    config = _read_yaml(config_path)
    checked = dry_run_training(config)
    if not checked["ok"]:
        print(json.dumps(checked, indent=2))
        return 1
    if dry_run:
        print(json.dumps({"action": action, **checked}, indent=2))
        return 0
    try:
        if action == "training":
            execute_training(config, [], requested=True)
        else:
            assert_execution_allowed(True, action)
    except Blocked as exc:
        print(str(exc))
        return 3
    return 3


def _plan() -> int:
    paths = get_paths()
    controls = _control_rows(paths.code_root)
    rows = plan_experiments(controls)
    destination = paths.study / "experiment_plan_v2.tsv"
    paths.assert_output(destination)
    write_experiment_table(destination, rows)
    counts = plan_counts(rows)
    write_json(paths.study / "experiment_plan_counts.json", {**counts, "hyperparameters": training_hyperparameters(), "generation": generation_settings(), "model_execution": False})
    print(json.dumps(counts, indent=2))
    return 0


def _build_inputs(
    corpus: str,
    limit: int | None,
    workers: int,
    splits: str | None,
    shard_index: int,
    shard_count: int,
    input_version: str | None,
    include_unlabeled: bool,
) -> int:
    from schema_study.contexts import INPUT_VERSION, build_context_tables

    paths = get_paths()
    prepared = {
        "biored": paths.prepared / "biored" / "biored_candidates_v2",
        "bc8": paths.prepared / "bc8" / "bc8_candidates_v2",
        "drugprot": paths.prepared / "drugprot" / "drugprot_candidates_v1",
    }[corpus]
    if not (prepared / "candidates.parquet").is_file():
        print(f"Prepared candidates are missing: {prepared}")
        return 2
    selected = set(splits.split(",")) if splits else None
    if corpus == "drugprot" and selected is None and not include_unlabeled:
        selected = {"train", "dev", "eval"}
    version = input_version
    if version is None and corpus == "drugprot" and limit is None:
        version = "target_anchored_v1__biomedbert_e1354b7a"
    destination = paths.inputs / corpus / (version or INPUT_VERSION)
    if shard_count > 1:
        destination = destination / "shards" / f"shard_{shard_index:02d}"
    if limit is not None:
        destination = paths.tmp / "input_timing" / corpus
    paths.assert_output(destination)
    summary = build_context_tables(
        prepared,
        destination,
        corpus=corpus,
        tokenizer_dir=Path(load_settings(paths.code_root)["biomedbert_tokenizer_dir"]),
        code_root=paths.code_root,
        limit=limit,
        workers=workers,
        splits=selected,
        shard_index=shard_index,
        shard_count=shard_count,
        input_version=version,
    )
    print(json.dumps(summary, indent=2))
    return 0 if not summary["anomalies"] else 1


def _review_materials() -> int:
    from schema_study.e3 import write_empty_template
    from schema_study.review import (
        blind_package_row,
        development_and_check,
        stratum_of,
        write_empty_answers,
        write_sampling_bundle,
    )
    from schema_study.records import read_jsonl

    paths = get_paths()
    sampling = paths.study / "review" / "sampling"
    packages = paths.study / "review" / "reviewer_packages"
    paths.assert_output(sampling)
    biored = _candidate_rows(paths.prepared / "biored" / "biored_candidates_v2" / "candidates.parquet")
    bc8 = _candidate_rows(paths.prepared / "bc8" / "bc8_candidates_v2" / "candidates.parquet")
    test_rows = [_review_record(row) for row in biored if row["split"] == "test" and row["included_in_scoring"]]
    bc8_rows = [_review_record(row) for row in bc8 if row["included_in_scoring"]]
    train_rows = [_review_record(row) for row in biored if row["split"] == "train" and row["included_in_scoring"]]
    print(json.dumps({"biored_test": write_sampling_bundle(sampling, "biored_test", test_rows), "bc8": write_sampling_bundle(sampling, "bc8", bc8_rows)}, indent=2))
    developed = development_and_check(train_rows)
    write_empty_answers(sampling / "biored_train_development_answers_empty.tsv", "biored_train_development", developed["development"]["selected"], "reviewer_a")
    write_empty_answers(sampling / "biored_train_check_answers_empty.tsv", "biored_train_check", developed["check"]["selected"], "reviewer_a")
    documents = {row["document_id"]: row for row in read_jsonl(paths.prepared / "biored" / "biored_candidates_v2" / "documents.jsonl")}
    mentions = {row["mention_id"]: row for row in read_jsonl(paths.prepared / "biored" / "biored_candidates_v2" / "entities.jsonl")}
    by_id = {row["candidate_id"]: row for row in biored}
    blinded = []
    for item in developed["development"]["selected"]:
        source = by_id[item["candidate_id"]]
        blinded.append(
            blind_package_row(
                item,
                documents[source["document_id"]]["text"],
                _mention_text(source["mention_ids_1"], mentions),
                _mention_text(source["mention_ids_2"], mentions),
            )
        )
    packages.mkdir(parents=True, exist_ok=True)
    write_tsv(packages / "biored_train_development_blind.tsv", blinded, ["candidate_id", "document_id", "text", "target_1", "target_2"])
    write_empty_template(paths.study / "extensions" / "e3_feasibility" / "mapping_template.tsv")
    (paths.study / "review" / "received_answers").mkdir(parents=True, exist_ok=True)
    (paths.study / "review" / "received_answers" / "README.txt").write_text(
        "No review answers have been received. Do not write semantic judgments into the empty tables.\n",
        encoding="utf-8",
    )
    return 0


def _candidate_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _review_record(row: dict) -> dict:
    from schema_study.review import stratum_of

    labels = list(row["gold_labels"] or [])
    return {
        "candidate_id": row["candidate_id"],
        "document_id": row["document_id"],
        "stratum": stratum_of(labels),
    }


def _mention_text(mention_ids, mentions: dict) -> str:
    seen = []
    for mention_id in list(mention_ids or []):
        text = mentions.get(mention_id, {}).get("text", "")
        if text and text not in seen:
            seen.append(text)
    return " | ".join(seen)


def _cpu_validate() -> int:
    """Score hand-written predictions. The numbers are not model performance."""
    from schema_study.evaluation import candidate_label_counts, information_retention, summarize_counts
    from schema_study.predictions import parse_prediction_text, rollup_predicted_labels

    paths = get_paths()
    schema = load_schema(default_schema_path(paths.code_root, "biored"))
    names = schema.positive_display_names("P4")
    by_name = schema.by_display("P4")
    examples = [
        ("c1", '{"candidate_id":"c1","labels":["Positive_Correlation"]}', ["Positive_Correlation"]),
        ("c2", '{"candidate_id":"c2","labels":["Association"]}', ["Positive_Correlation"]),
        ("c3", '{"candidate_id":"c3","labels":[]}', []),
        ("c4", "", ["Bind"]),
        ("c5", '{"candidate_id":"c5","labels":["NotARealRelation"]}', ["Association"]),
        ("c6", '{"candidate_id":"c6","labels":["Bind","Association"]}', ["Bind"]),
    ]
    scored = []
    for candidate_id, text, gold_names in examples:
        parsed = parse_prediction_text(text, expected_candidate_id=candidate_id, schema=schema, level="P4")
        gold = [by_name[name].stable_id for name in gold_names]
        counts = candidate_label_counts(gold, parsed["predicted_labels"], parsed["output_status"], parsed["illegal_labels"])
        scored.append({"candidate_id": candidate_id, "status": parsed["output_status"], "gold": gold, "predicted": parsed["predicted_labels"], "illegal_labels": parsed["illegal_labels"], **{key: counts[key] for key in ("tp", "fp", "fn", "extra_fp", "exact")}})
    summary = summarize_counts(scored, [by_name[name].stable_id for name in names])
    comparison = by_name["Comparison"].stable_id
    conversion = by_name["Conversion"].stable_id
    rolled = rollup_predicted_labels(schema, [comparison, conversion], source_level="P4", target_level="P2")
    other = schema.by_display("P2")["Other_Relation"].stable_id
    retention = information_retention(
        [{"fine_id": comparison, "status": "valid", "predicted_groups": rolled}],
        schema,
        "P2",
        {fine_id: 1 for fine_id in schema.fine_ids},
    )
    real = _candidate_rows(paths.prepared / "biored" / "biored_candidates_v2" / "candidates.parquet")[:3]
    destination = paths.tmp / "cpu_validation_synthetic"
    paths.assert_output(destination)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "synthetic": True,
        "model_executed": False,
        "weights_loaded": False,
        "api_called": False,
        "uses_real_candidate_ids_only_as_structure": [row["candidate_id"] for row in real],
        "examples": scored,
        "summary": {key: summary[key] for key in ("n_candidates", "tp", "fp", "fn", "extra_fp", "f1", "output_coverage", "failure_count", "exact_match")},
        "rollup_comparison_and_conversion_to_p2": rolled,
        "rollup_is_single_other_relation": rolled == [other],
        "retention_comparison_predicted_as_other": retention,
    }
    write_json(destination / "synthetic_pipeline.json", payload)
    if rolled != [other]:
        print(json.dumps(payload, indent=2, default=str))
        return 1
    print(json.dumps({"synthetic": True, "rollup": rolled, "f1": summary["f1"], "coverage": summary["output_coverage"], "failure_count": summary["failure_count"]}, indent=2))
    return 0


def _parse(source: str, destination: str, dataset: str, level: str) -> int:
    from schema_study.predictions import parse_prediction_text, unified_row
    from schema_study.records import read_jsonl, write_jsonl

    paths = get_paths()
    paths.assert_output(destination)
    schema = load_schema(default_schema_path(paths.code_root, dataset))
    rows = []
    for item in read_jsonl(source):
        parsed = parse_prediction_text(
            item.get("text"),
            expected_candidate_id=item["candidate_id"],
            schema=schema,
            level=level,
        )
        rows.append(
            unified_row(
                run_id=item.get("run_id", "unspecified"),
                dataset=dataset,
                split=item.get("split", ""),
                document_id=item.get("document_id", ""),
                candidate_id=item["candidate_id"],
                entity_level=item.get("entity_level", ""),
                predicate_level=level,
                route=item.get("route", "direct"),
                model_id=item.get("model_id", ""),
                seed=item.get("seed", ""),
                input_id=item.get("input_id", ""),
                predicted_labels=parsed["predicted_labels"],
                output_status=parsed["output_status"],
                probabilities=None,
                source_prediction_id=item.get("source_prediction_id", ""),
                request_id=item.get("request_id", ""),
                illegal_labels=parsed["illegal_labels"],
            )
        )
    write_jsonl(Path(destination), rows)
    print(json.dumps({"n": len(rows), "synthetic_input": True}, indent=2))
    return 0


def _rollup(source: str, destination: str, dataset: str, source_level: str, target_level: str) -> int:
    from schema_study.predictions import rollup_predicted_labels
    from schema_study.records import read_jsonl, write_jsonl

    paths = get_paths()
    paths.assert_output(destination)
    schema = load_schema(default_schema_path(paths.code_root, dataset))
    rows = []
    for item in read_jsonl(source):
        rolled = dict(item)
        rolled["predicted_labels"] = rollup_predicted_labels(
            schema,
            item["predicted_labels"],
            source_level=source_level,
            target_level=target_level,
        )
        rolled["predicate_level"] = target_level
        rolled["route"] = "rollup"
        rolled["source_predicate_level"] = source_level
        rows.append(rolled)
    write_jsonl(Path(destination), rows)
    print(json.dumps({"n": len(rows)}, indent=2))
    return 0


def _evaluate(predictions: str, destination: str, dataset: str, level: str) -> int:
    from schema_study.evaluation import summarize_counts
    from schema_study.records import read_jsonl

    paths = get_paths()
    paths.assert_output(destination)
    schema = load_schema(default_schema_path(paths.code_root, dataset))
    rows = []
    for item in read_jsonl(predictions):
        rows.append(
            {
                "gold": item.get("gold_labels", []),
                "predicted": item.get("predicted_labels", []),
                "status": item["output_status"],
                "illegal_labels": item.get("illegal_labels", []),
            }
        )
    summary = summarize_counts(rows, [label.stable_id for label in schema.level_labels(level)])
    summary["synthetic"] = True
    summary["model_executed"] = False
    write_json(Path(destination), summary)
    print(json.dumps({"n": summary["n_candidates"], "f1": summary["f1"], "synthetic": True}, indent=2))
    return 0


def _report(destination: str, synthetic: bool) -> int:
    from schema_study.reporting import plot_loss_decomposition

    paths = get_paths()
    paths.assert_output(destination)
    if not synthetic:
        write_json(
            Path(destination),
            {
                "formal_figure_written": False,
                "reason": "No trained or prompted outputs are available. A formal figure is not invented.",
            },
        )
        print(json.dumps({"formal_figure_written": False}, indent=2))
        return 0
    plot_loss_decomposition(0.25, 0.15, Path(destination), synthetic=True)
    print(json.dumps({"synthetic": True, "composite_score": False}, indent=2))
    return 0


def _control_rows(code_root: Path) -> list[dict]:
    with (code_root / "configs" / "schemas" / "biored_controls.yaml").open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    rows = [{"control_id": item["control_id"], "predicate_level": item["base_level"]} for item in payload["rename"]]
    for item in payload["random_merge"]["versions"]:
        rows.append({"control_id": item["control_id"], "predicate_level": "P_control"})
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
