import json
import re
from typing import get_origin, get_args, List, Type
from pydantic import parse_obj_as


def clean_json_text(text: str) -> str:
    """
    Preprocess text from LLM output to improve JSON validity.
    Fix common issues like:
    - extra markdown formatting
    - missing commas
    - semicolon instead of comma
    """
    text = text.strip().replace("```json", "").replace("```", "").strip()

    # Fix common JSON syntax errors
    text = text.replace(";}", "}").replace("; ]", "]")
    text = re.sub(r'}\s*{', '}, {', text)         # {...}{...} -> {...}, {...}
    text = re.sub(r'"\s+"', '", "', text)         # "..." "..." -> "..., ..."
    text = re.sub(r'(?<=[\w\"]) +(?=\")', '', text)  # remove stray spaces before quoted key
    text = re.sub(r'[\)\]]\s*$', '}', text)

    return text


def extract_first_json(text: str) -> str:
    """
    Attempt to extract the first valid JSON block.
    Supports full object, array, or object with 'result' key.
    """
    text = clean_json_text(text)

    # Direct JSON parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            if not parsed:
                raise ValueError("Parsed JSON is empty list []")
            return json.dumps(parsed)
        if isinstance(parsed, dict):
            if "result" in parsed and isinstance(parsed["result"], list):
                if not parsed["result"]:
                    raise ValueError("Parsed JSON 'result' key is empty []")
                return json.dumps(parsed["result"])
            if not parsed:
                raise ValueError("Parsed JSON is empty dict {}")
            return json.dumps(parsed)
    except Exception:
        pass

    # Fallback regex: look for JSON array
    match = re.search(r'\[\s*\{[\s\S]*?\}\s*\]', text)
    if match:
        return match.group(0)

    # Fallback regex: look for single dict
    match = re.search(r'\{[\s\S]*?\}', text)
    if match:
        return match.group(0)

    raise ValueError(f"No valid JSON object found. Sample: {text[:80]}...")


def validate_doc(raw: str, target_class: Type):
    """
    Parse JSON into target Pydantic object (single or list).
    """
    data = json.loads(raw)

    if data in [[], {}, None] or (isinstance(data, dict) and all(v in [None, "N/A", ""] for v in data.values())):
        raise ValueError("Parsed JSON is empty or invalid content")

    if get_origin(target_class) == list:
        inner_type = get_args(target_class)[0]
        if isinstance(data, dict):
            data = [data]
        return parse_obj_as(List[inner_type], data)

    if hasattr(target_class, "model_validate"):
        return target_class.model_validate(data)
    return target_class.parse_obj(data)


def parse_or_fix(
    raw: str,
    client,
    messages: list[dict],
    target_class,
    max_retry: int = 2,
    log_failures: bool = True,
):
    """
    Try to parse LLM output. Retry if invalid. Log final failure if any.
    """
    last_err = None
    for attempt in range(max_retry + 1):
        try:
            clean = extract_first_json(raw)
            return validate_doc(clean, target_class)
        except Exception as e:
            last_err = e
            print(f"[Retry {attempt+1}] Invalid JSON: {e}")
            print("-> Raw output:", raw[:300], "..." if len(raw) > 300 else "")

            if attempt == max_retry:
                break

            # Ask LLM to fix the output
            raw = client.run(
                messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Invalid JSON ({e}). Please return ONLY valid JSON."}
                ]
            )

    if log_failures:
        with open("parse_failures.jsonl", "a", encoding="utf-8") as fw:
            fw.write(json.dumps({
                "error": str(last_err),
                "raw": raw
            }, ensure_ascii=False) + "\n")

    raise ValueError(f"Final JSON parse failed: {last_err}")