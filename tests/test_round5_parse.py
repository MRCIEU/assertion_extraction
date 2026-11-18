"""Parser and prompt checks for empty outputs and left-padding masks."""

from __future__ import annotations

import json
from pathlib import Path

from schema_study.predictions import parse_prediction_text
from schema_study.schemas import load_schema

ROOT = Path("/home/b5ac/freddieyu.b5ac/project_1")
BIORED = load_schema(ROOT / "configs/schemas/biored.yaml")
DRUG = load_schema(ROOT / "configs/schemas/drugprot.yaml")
PROMPT = (ROOT / "prompts/relation_extraction.txt").read_text(encoding="utf-8")


def test_prompt_allows_an_empty_label_list():
    lowered = PROMPT.lower()
    assert "return an empty labels list" in lowered
    assert "at most one positive label" in lowered
    assert "more than one positive label" in lowered
    for phrase in ("must choose", "must select", "exactly one", "choose one"):
        assert phrase not in lowered


def test_empty_and_cardinality_contract():
    empty = parse_prediction_text('{"candidate_id":"c","labels":[]}', expected_candidate_id="c", schema=BIORED, level="P4")
    assert empty["output_status"] == "valid" and empty["predicted_labels"] == []
    one = parse_prediction_text('{"candidate_id":"c","labels":["Association"]}', expected_candidate_id="c", schema=BIORED, level="P4")
    assert one["output_status"] == "valid" and one["predicted_labels"] == ["BR03_ASSOCIATION"]
    several = parse_prediction_text('{"candidate_id":"c","labels":["Association","Bind"]}', expected_candidate_id="c", schema=BIORED, level="P4")
    assert several["output_status"] == "rejected" and several["predicted_labels"] == []
    multi = parse_prediction_text('{"candidate_id":"c","labels":["INHIBITOR","AGONIST"]}', expected_candidate_id="c", schema=DRUG, level="P4")
    assert multi["output_status"] == "valid"
    assert multi["predicted_labels"] == ["DP09_INHIBITOR", "DP02_AGONIST"]
    illegal = parse_prediction_text('{"candidate_id":"c","labels":["NotARealRelation"]}', expected_candidate_id="c", schema=BIORED, level="P4")
    assert illegal["output_status"] == "rejected" and illegal["predicted_labels"] == []
    mixed = parse_prediction_text('{"candidate_id":"c","labels":["INHIBITOR","NotARealRelation"]}', expected_candidate_id="c", schema=DRUG, level="P4")
    assert mixed["output_status"] == "partial_illegal" and mixed["predicted_labels"] == ["DP09_INHIBITOR"]
    missing = parse_prediction_text("   ", expected_candidate_id="c", schema=BIORED, level="P4")
    assert missing["output_status"] == "missing"


def test_prompt_example_is_not_parsed_as_the_answer():
    answer = '{"candidate_id":"c","labels":[]}'
    combined = PROMPT + "\n" + answer
    parsed_combined = parse_prediction_text(combined, expected_candidate_id="c", schema=BIORED, level="P4")
    parsed_answer = parse_prediction_text(answer, expected_candidate_id="c", schema=BIORED, level="P4")
    assert parsed_combined["output_status"] == "missing"
    assert parsed_answer["output_status"] == "valid" and parsed_answer["predicted_labels"] == []


def test_left_pad_mask_does_not_drop_a_real_eos_token():
    model = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/models/Qwen__Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218")
    if not model.exists():
        return
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id
    texts = [f"{tokenizer.pad_token} keep this sentence intact", "x"]
    single = [tokenizer(text, truncation=False, add_special_tokens=False) for text in texts]
    batched = tokenizer(texts, padding=True, truncation=False, add_special_tokens=False)
    for index, original in enumerate(single):
        mask = batched["attention_mask"][index]
        ids = batched["input_ids"][index]
        content = [token for token, keep in zip(ids, mask) if keep == 1]
        assert content == original["input_ids"]
        leading = len(ids) - len(content)
        assert all(token == pad_id and keep == 0 for token, keep in zip(ids[:leading], mask[:leading]))
    long_mask = batched["attention_mask"][0]
    long_ids = batched["input_ids"][0]
    long_content = [token for token, keep in zip(long_ids, long_mask) if keep == 1]
    assert pad_id in long_content
    assert long_mask[long_content.index(pad_id)] == 1
    short_mask = batched["attention_mask"][1]
    assert short_mask[0] == 0 and short_mask[-1] == 1
