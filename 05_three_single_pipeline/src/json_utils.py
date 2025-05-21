import json
import re
from typing import get_origin, get_args, List, Type
from pydantic import parse_obj_as


def clean_json_text(text: str) -> str:
    """
    Preprocess text from LLM output to improve JSON validity.
    Removes markdown and fixes common formatting bugs.
    """
    text = text.strip().replace("```json", "").replace("```", "").strip()
    text = text.replace(";}", "}").replace("; ]", "]")
    text = re.sub(r'}\s*{', '}, {', text)
    text = re.sub(r'"\s+"', '", "', text)
    text = re.sub(r'(?<=[\w\"]) +(?=\")', '', text)
    text = re.sub(r'[\)\]]\s*$', '}', text)
    return text


def extract_first_json(text: str) -> str:
    text = clean_json_text(text)

    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            if "result" in parsed:
                if isinstance(parsed["result"], list) and parsed["result"]:
                    return json.dumps(parsed["result"][0])
                if isinstance(parsed["result"], dict):
                    return json.dumps(parsed["result"])
            return json.dumps(parsed)

        if isinstance(parsed, list) and parsed:
            return json.dumps(parsed[0])

    except Exception:
        pass

    match = re.search(r'\{[\s\S]*?\}', text)
    if match:
        return match.group(0)

    raise ValueError(f"No valid JSON object found. Sample: {text[:100]}...")

def validate_doc(raw: str, target_class: Type):
    data = json.loads(raw)

    if data in [None, {}, []] or (
        isinstance(data, dict) and all(v in [None, "", "N/A"] for v in data.values())
    ):
        raise ValueError("Parsed JSON is empty or invalid content")

    if get_origin(target_class) == list:
        inner = get_args(target_class)[0]
        if isinstance(data, dict):
            raise ValueError("Expected list, got dict")
        return parse_obj_as(List[inner], data)

    return target_class.model_validate(data) if hasattr(target_class, "model_validate") else target_class.parse_obj(data)

def parse_or_fix(raw: str, client, messages, target_class, max_retry: int = 2, log_failures: bool = True):
    last_err = None
    for attempt in range(max_retry + 1):
        try:
            if not raw.strip():
                raise ValueError("Empty model output")

            clean = extract_first_json(raw)
            return validate_doc(clean, target_class)

        except Exception as e:
            last_err = e
            print(f"[Retry {attempt+1}] Invalid JSON: {e.__class__.__name__} – {e}")
            print("-> Raw output:", raw[:300], "..." if len(raw) > 300 else "")

            if attempt == max_retry:
                break

            raw = client.run(messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Invalid JSON ({e}). Please return ONLY valid JSON object."}
            ])

    if log_failures:
        Path("parse_failures.jsonl").parent.mkdir(parents=True, exist_ok=True)
        with open("parse_failures.jsonl", "a", encoding="utf-8") as fw:
            fw.write(json.dumps({
                "error": str(last_err),
                "raw": raw
            }, ensure_ascii=False) + "\n")

    raise ValueError(f"Final JSON parse failed: {last_err}")