import json
import time
from pathlib import Path
from datetime import datetime

LOG_PATH = "logs/api_calls.jsonl"
Path("logs").mkdir(exist_ok=True)

def log_api_call(model, task_id, prompt_tokens, response_tokens, duration_sec, cost_usd=None):
    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "task_id": task_id,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": prompt_tokens + response_tokens,
        "duration_sec": round(duration_sec, 2),
        "cost_usd": round(cost_usd, 6) if cost_usd else None
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log) + "\n")