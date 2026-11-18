"""Build target_anchored_v1 inputs from a prepared candidate table.

The BiomedBERT tokenizer only measures the shared 512-token budget. This module
does not load model weights. The resulting input version stays proposed until
the tokenizer revision is confirmed.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from schema_study.inputs import (
    SPECIAL_MARKERS,
    STRATEGY,
    BiomedBertEncoder,
    MentionView,
    build_target_anchored_context,
    load_biomedbert_tokenizer,
)
from schema_study.records import digest_file, read_jsonl, write_json, write_jsonl, write_tsv
from schema_study.validation import ValidationReport

INPUT_VERSION = "target_anchored_v1__biomedbert_e1354b7a_proposed"
TOKENIZER_STATUS = "proposed_not_frozen"


def entity_views(code_root: Path, corpus: str) -> dict:
    if corpus == "drugprot":
        return {"native": {"CHEMICAL": "CHEMICAL", "GENE": "GENE"}}
    path = code_root / "configs" / "schemas" / "entities.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        "E0": None,
        "E1": dict(raw["entity_levels"]["E1"]["merge"]),
        "E2": dict(raw["entity_levels"]["E2"].get("merge") or {}),
    }


_STATE: dict = {}


def _init_worker(tokenizer_dir: str) -> None:
    tokenizer = load_biomedbert_tokenizer(tokenizer_dir, SPECIAL_MARKERS)
    _STATE["encoder"] = BiomedBertEncoder(tokenizer)


def _render_document(task) -> list[dict]:
    text, document_id, items, views = task
    rendered = []
    for candidate_id, split, mention_payload in items:
        mentions = [MentionView(*item) for item in mention_payload]
        record = build_target_anchored_context(text, mentions, _STATE["encoder"], views)
        record["candidate_id"] = candidate_id
        record["document_id"] = document_id
        record["split"] = split
        rendered.append(record)
    return rendered


def _missing_record(candidate: dict) -> dict:
    return {
        "candidate_id": candidate["candidate_id"],
        "document_id": candidate["document_id"],
        "split": candidate["split"],
        "mode": "anomaly",
        "anomaly": "missing_document_or_mention",
        "fragments": [],
        "views": {},
        "encoded_length_by_view": {},
        "max_encoded_length": None,
    }


def build_context_tables(
    prepared_dir: Path,
    output_dir: Path,
    *,
    corpus: str,
    tokenizer_dir: Path,
    code_root: Path,
    limit: int | None = None,
    workers: int = 1,
    splits: set[str] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    input_version: str | None = None,
    candidate_ids: set[str] | None = None,
) -> dict:
    """Write contexts for prepared candidates. `limit` is only for a timed sample."""
    report = ValidationReport(name=f"{corpus}_inputs")
    documents = {row["document_id"]: row for row in read_jsonl(prepared_dir / "documents.jsonl")}
    mentions = {row["mention_id"]: row for row in read_jsonl(prepared_dir / "entities.jsonl")}
    views = entity_views(code_root, corpus)
    import pyarrow.parquet as pq

    candidate_path = prepared_dir / "candidates.parquet"
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be inside shard_count")
    eligible_docs = sorted(
        document_id
        for document_id, document in documents.items()
        if splits is None or document.get("split") in splits
    )
    if shard_count > 1:
        eligible_docs = eligible_docs[shard_index::shard_count]
    eligible = set(eligible_docs)
    if candidate_ids is not None and len(candidate_ids) <= 5000:
        frame = []
        source = pq.ParquetFile(candidate_path)
        for batch in source.iter_batches(batch_size=20000):
            for row in batch.to_pylist():
                if row["candidate_id"] in candidate_ids and row["document_id"] in eligible:
                    frame.append(row)
    elif limit is None:
        import pyarrow.dataset as ds

        dataset = ds.dataset(str(candidate_path), format="parquet")
        filter_expression = ds.field("document_id").isin(eligible_docs) if eligible_docs else None
        frame = dataset.to_table(filter=filter_expression).to_pylist() if eligible_docs else []
    else:
        frame = []
        source = pq.ParquetFile(candidate_path)
        for batch in source.iter_batches(batch_size=1000):
            frame.extend(batch.to_pylist())
            if len(frame) >= limit:
                frame = frame[:limit]
                break
        frame = [row for row in frame if row["document_id"] in eligible]
    if candidate_ids is not None:
        frame = [row for row in frame if row["candidate_id"] in candidate_ids]
    version = input_version or INPUT_VERSION
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list] = {}
    missing_rows = []
    for candidate in frame:
        mention_views = _mentions_for_candidate(candidate, mentions, report)
        document = documents.get(candidate["document_id"])
        if document is None or not mention_views:
            missing_rows.append(_missing_record(candidate))
            report.add("error", "missing_document_or_mention", "Candidate could not be attached to its document or mentions.")
            continue
        grouped.setdefault(candidate["document_id"], []).append(
            (
                candidate["candidate_id"],
                candidate["split"],
                [(item.mention_id, item.start, item.end, item.role, item.entity_type) for item in mention_views],
                document["text"],
            )
        )
    tasks = []
    for document_id, items in grouped.items():
        text = items[0][3]
        tasks.append((text, document_id, [(item[0], item[1], item[2]) for item in items], views))
    if workers > 1 and len(tasks) > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(workers, initializer=_init_worker, initargs=(str(tokenizer_dir),)) as pool:
            rendered_groups = pool.map(_render_document, tasks)
    else:
        _init_worker(str(tokenizer_dir))
        rendered_groups = [_render_document(task) for task in tasks]
    rows = missing_rows + [record for group in rendered_groups for record in group]
    view_rows = []
    modes = Counter()
    anomalies = Counter()
    for record in rows:
        record["input_id"] = f"{version}:{record['candidate_id']}"
        record["tokenizer_status"] = "accepted" if "proposed" not in version else TOKENIZER_STATUS
        modes[record["mode"]] += 1
        if record.get("anomaly"):
            anomalies[record["anomaly"]] += 1
        for level, length in record.get("encoded_length_by_view", {}).items():
            view_rows.append(
                {
                    "candidate_id": record["candidate_id"],
                    "input_id": record["input_id"],
                    "entity_level": level,
                    "encoded_length": length,
                }
            )
    write_jsonl(output_dir / "contexts.jsonl", rows)
    _write_view_parquet(output_dir / "entity_views.parquet", view_rows)
    coverage = [
        {"split": split, "mode": mode, "n": count}
        for (split, mode), count in sorted(Counter((row["split"], row["mode"]) for row in rows).items())
    ]
    write_tsv(output_dir / "coverage_summary.tsv", coverage, ["split", "mode", "n"])
    summary = {
        "corpus": corpus,
        "input_version": version,
        "tokenizer_status": "accepted" if "proposed" not in version else TOKENIZER_STATUS,
        "weights_loaded": False,
        "n_contexts": len(rows),
        "limit": limit,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "modes": dict(modes),
        "anomalies": dict(anomalies),
        "full_release_built": limit is None and shard_count == 1 and candidate_ids is None,
    }
    report_body = report.as_dict()
    report_body["summary"] = summary
    write_json(output_dir / "validation_report.json", report_body)
    vocab = tokenizer_dir / "vocab.txt"
    write_json(
        output_dir / "provenance.json",
        {
            "strategy": STRATEGY,
        "input_version": version,
        "tokenizer_dir": str(tokenizer_dir),
        "tokenizer_vocab_sha256": digest_file(vocab) if vocab.is_file() else None,
        "tokenizer_status": "accepted" if "proposed" not in version else TOKENIZER_STATUS,
        "weights_loaded": False,
        "prepared_dir": str(prepared_dir),
        "views": list(views),
        "budget": 512,
        "limit": limit,
        "splits": sorted(splits) if splits else None,
        "shard_index": shard_index,
        "shard_count": shard_count,
        },
    )
    return summary


def _mentions_for_candidate(candidate: dict, mentions: dict, report: ValidationReport) -> list[MentionView]:
    views = []
    pairs = (
        (candidate["mention_ids_1"], candidate["argument_1_role"]),
        (candidate["mention_ids_2"], candidate["argument_2_role"]),
    )
    for mention_ids, role in pairs:
        for mention_id in list(mention_ids or []):
            mention = mentions.get(mention_id)
            if mention is None:
                report.add("error", "missing_mention", "A candidate mention id is not in the entity table.")
                continue
            views.append(
                MentionView(
                    mention_id,
                    int(mention["char_start"]),
                    int(mention["char_end"]),
                    role,
                    mention["entity_type"],
                )
            )
    return views


def _write_view_parquet(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        table = pa.table(
            {
                "candidate_id": pa.array([], type=pa.string()),
                "input_id": pa.array([], type=pa.string()),
                "entity_level": pa.array([], type=pa.string()),
                "encoded_length": pa.array([], type=pa.int64()),
            }
        )
    else:
        table = pa.table(
            {
                "candidate_id": [row["candidate_id"] for row in rows],
                "input_id": [row["input_id"] for row in rows],
                "entity_level": [row["entity_level"] for row in rows],
                "encoded_length": [row["encoded_length"] for row in rows],
            }
        )
    pq.write_table(table, path)
