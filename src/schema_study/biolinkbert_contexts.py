"""Build the BioLinkBERT context collection and verify hierarchical encoder inputs.

The principal BiomedBERT windows stay in place. A window is rewritten only when
the BioLinkBERT tokenizer exceeds 512 tokens. Source text is not truncated.
"""

from __future__ import annotations

import json
from pathlib import Path

from schema_study.contexts import build_context_tables, entity_views
from schema_study.inputs import SPECIAL_MARKERS, load_biomedbert_tokenizer
from schema_study.paths import get_paths
from schema_study.production_assets import encoder_layout

STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
RUN = STORAGE / "studies/schema_granularity_v1/runs/formal/formal_matched_e0_e7"
SOURCES = {
    "biored": STORAGE / "data/inputs/biored/target_anchored_v2__biomedbert_e1354b7a_matched_e0_e7/contexts.jsonl",
    "bc8": STORAGE / "data/inputs/bc8/target_anchored_v2__biomedbert_e1354b7a_matched_e0_e7/contexts.jsonl",
    "drugprot": STORAGE / "data/inputs/drugprot/target_anchored_v1__biomedbert_e1354b7a/contexts.jsonl",
}
DESTINATIONS = {
    "biored": "target_anchored_v2__biolinkbert_b71f5d70_matched_e0_e7",
    "bc8": "target_anchored_v2__biolinkbert_b71f5d70_matched_e0_e7",
    "drugprot": "target_anchored_v1__biolinkbert_b71f5d70",
}


def _lengths(tokenizer, views: dict) -> dict[str, int]:
    measured = {}
    for level, text in views.items():
        measured[level] = len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
    return measured


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def verify_hierarchical_inputs() -> dict:
    """The hierarchical BiomedBERT path must use the marker-aware principal tokenizer."""
    tokenizer = load_biomedbert_tokenizer(str(encoder_layout("biomedbert")["tokenizer"]))
    marker_ids = [tokenizer.convert_tokens_to_ids(marker) for marker in SPECIAL_MARKERS]
    if any(marker_id == tokenizer.unk_token_id for marker_id in marker_ids):
        raise RuntimeError("A context marker is not a single BiomedBERT token.")
    failures = []
    overflow_count = 0
    checked = 0
    maximum = 0
    for corpus, path in SOURCES.items():
        for record in _read_jsonl(path):
            measured = _lengths(tokenizer, record.get("views") or {})
            checked += 1
            longest = max(measured.values()) if measured else 0
            maximum = max(maximum, longest)
            if longest > 512:
                overflow_count += 1
                if len(failures) < 20:
                    failures.append({"candidate_id": record["candidate_id"], "corpus": corpus, "lengths": measured})
    report = {
        "tokenizer": "principal BiomedBERT tokenizer with the study markers added as special tokens",
        "marker_ids": marker_ids,
        "records_checked": checked,
        "maximum_length": maximum,
        "overflow_count": overflow_count,
        "ready": overflow_count == 0 and maximum <= 512,
        "failures": failures,
        "new_context_version": False,
        "source_truncated": False,
    }
    destination = RUN / "hierarchical_input_compatibility.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return report


def _views_for(corpus: str, paths) -> dict:
    if corpus == "drugprot":
        return entity_views(paths.code_root, corpus)
    from schema_study.experimental_preparation import _load_extended_concepts, _matched_views

    _concepts, levels = _load_extended_concepts(paths)
    return _matched_views(paths, levels)


def rebuild_biolinkbert(corpus: str) -> dict:
    paths = get_paths()
    source = SOURCES[corpus]
    destination = STORAGE / "data/inputs" / corpus / DESTINATIONS[corpus]
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer = load_biomedbert_tokenizer(str(encoder_layout("biolinkbert")["tokenizer"]))
    lengths = {}
    overflow = []
    population = 0
    for record in _read_jsonl(source):
        population += 1
        measured = _lengths(tokenizer, record.get("views") or {})
        lengths[record["candidate_id"]] = measured
        if any(value > 512 for value in measured.values()):
            overflow.append(record["candidate_id"])
    rebuilt = {}
    if overflow:
        from schema_study.experimental_preparation import _display_by_mention, _load_extended_concepts
        from schema_study.formal_protocol import CURRENT_VERSIONS

        concepts, _levels = _load_extended_concepts(paths)
        build_context_tables(
            paths.prepared / corpus / CURRENT_VERSIONS[corpus],
            destination / "rebuilt",
            corpus=corpus,
            tokenizer_dir=encoder_layout("biolinkbert")["tokenizer"],
            code_root=paths.code_root,
            workers=8,
            input_version=DESTINATIONS[corpus],
            candidate_ids=set(overflow),
            views_override=_views_for(corpus, paths),
            display_by_mention=_display_by_mention(paths, corpus, concepts),
        )
        for record in _read_jsonl(destination / "rebuilt" / "contexts.jsonl"):
            if record.get("mode") == "anomaly" or not record.get("views"):
                raise RuntimeError(f"{record.get('candidate_id')} could not be rebuilt without dropping its source.")
            record["encoded_length_by_view"] = _lengths(tokenizer, record["views"])
            if any(value > 512 for value in record["encoded_length_by_view"].values()):
                raise RuntimeError(f"{record['candidate_id']} still exceeds 512 after the BioLinkBERT window rebuild.")
            rebuilt[record["candidate_id"]] = record
        missing = [candidate_id for candidate_id in overflow if candidate_id not in rebuilt]
        if missing:
            raise RuntimeError(f"{len(missing)} overflow candidates were not rebuilt.")
    changes = destination / "window_changes.jsonl"
    output = destination / "contexts.jsonl"
    temporary = destination / "contexts.jsonl.partial"
    written = 0
    with temporary.open("w", encoding="utf-8") as handle, changes.open("w", encoding="utf-8") as change_handle:
        for record in _read_jsonl(source):
            candidate_id = record["candidate_id"]
            if candidate_id in rebuilt:
                replacement = rebuilt[candidate_id]
                change_handle.write(json.dumps({
                    "candidate_id": candidate_id,
                    "corpus": corpus,
                    "split": record.get("split"),
                    "previous_fragments": record.get("fragments"),
                    "new_fragments": replacement.get("fragments"),
                    "previous_encoded_length_by_view": record.get("encoded_length_by_view"),
                    "biolinkbert_previous_lengths": lengths[candidate_id],
                    "biolinkbert_new_lengths": replacement["encoded_length_by_view"],
                    "reason": "The unchanged window exceeded 512 BioLinkBERT tokens.",
                }, sort_keys=True) + "\n")
                record = replacement
            else:
                record["encoded_length_by_view"] = lengths[candidate_id]
                record["max_encoded_length"] = max(lengths[candidate_id].values()) if lengths[candidate_id] else None
            if any(value > 512 for value in record["encoded_length_by_view"].values()):
                raise RuntimeError(f"{candidate_id} exceeds 512 in the BioLinkBERT collection.")
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            written += 1
    if written != population:
        raise RuntimeError(f"{corpus} changed population size from {population} to {written}.")
    temporary.replace(output)
    report = {
        "corpus": corpus,
        "input_version": DESTINATIONS[corpus],
        "population": written,
        "windows_retained": written - len(overflow),
        "windows_rebuilt": len(overflow),
        "source_truncated": False,
        "candidates_removed": 0,
        "unchanged_window_comparison": "Existing BiomedBERT results remain comparable on the retained windows.",
        "strict_common_window_requirement": "A full-population comparison on identical source windows would need a separate BiomedBERT training run under the rebuilt windows. That run is not part of this recovery and the principal BiomedBERT experiment is unchanged.",
        "ready": True,
    }
    ready = destination / "READY.json"
    ready_temporary = ready.with_suffix(".json.tmp")
    ready_temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ready_temporary.replace(ready)
    return report


def main() -> None:
    hierarchical = verify_hierarchical_inputs()
    print(json.dumps({"hierarchical_ready": hierarchical["ready"], "maximum_length": hierarchical["maximum_length"]}), flush=True)
    if not hierarchical["ready"]:
        raise SystemExit("Hierarchical inputs still exceed the encoder window.")
    for corpus in ("biored", "bc8", "drugprot"):
        print(json.dumps(rebuild_biolinkbert(corpus)), flush=True)


if __name__ == "__main__":
    main()
