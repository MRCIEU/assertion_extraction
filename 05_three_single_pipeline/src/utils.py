import re
from pathlib import Path
from typing import List, Union
import csv
import time
from datetime import datetime

API_STATS = {}

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


def estimate_cost(model: str, total_tokens: int) -> float:
    pricing = {
        "gpt-4o": 0.005 / 1000,
        "gpt-4o-mini": 0.003 / 1000,
        "claude-3-haiku-20240307": 0.002 / 1000,
        "llama3-8b": 0.0015 / 1000,
    }
    return round(total_tokens * pricing.get(model, 0.002 / 1000), 6)

def estimate_tokens(text: str, model="gpt-4") -> int:
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)

def start_timer(model: str, stage: str):
    key = (model, stage)
    API_STATS[key] = {
        "start_time": time.time(),
        "calls": 0,
        "tokens": 0
    }

def log_tokens(model: str, stage: str, total_tokens: int):
    key = (model, stage)
    if key not in API_STATS:
        start_timer(model, stage)
    API_STATS[key]["calls"] += 1
    API_STATS[key]["tokens"] += total_tokens

def save_stats_to_csv(stage: str, path="logs/summary.csv"):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    rows = []

    for (model, stg), stat in API_STATS.items():
        if stg != stage:
            continue
        duration = round(time.time() - stat["start_time"], 2)
        cost = estimate_cost(model, stat["tokens"])
        rows.append({
            "model": model,
            "stage": stg,
            "calls": stat["calls"],
            "total_tokens": stat["tokens"],
            "duration_sec": duration,
            "cost_usd": cost,
            "timestamp": datetime.now().isoformat()
        })

    if not rows:
        print(f"[WARN] No API stats to log for stage '{stage}'")
        return

    file_exists = Path(path).exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)