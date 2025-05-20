import re
from pathlib import Path
from typing import List, Union

def split_sents(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    # Normalize spacing
    clean_text = re.sub(r'\s+', ' ', text.strip())

    # Sentence segmentation
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', clean_text)
    return [s.strip() for s in parts if s.strip()]

def ensure_dir_exists(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def estimate_openai_cost(model: str, total_tokens: int) -> float:
    pricing = {
        "gpt-4o": 0.005 / 1000,
        "gpt-4o-mini": 0.003 / 1000,
        "gpt-3.5-turbo": 0.001 / 1000,
    }
    rate = pricing.get(model, 0.002 / 1000)
    return round(total_tokens * rate, 6)
