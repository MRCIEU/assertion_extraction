import os
import csv
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

BASE_NAME = "API_log"
EXT = ".csv"
MAX_MB = 50 # MB

FIELDS = ["timestamp", "model", "task_id", "prompt_tokens", "response_tokens", "duration_sec", "cost_usd"]

def _get_next_logfile() -> Path:
    """
    Automatically select the currently available log file, and create a numbered file if it exceeds the size limit.
    """
    index = 0
    while True:
        name = f"{BASE_NAME}_{index}{EXT}" if index > 0 else f"{BASE_NAME}{EXT}"
        path = LOG_DIR / name
        if not path.exists() or path.stat().st_size < MAX_MB * 1024 * 1024:
            return path
        index += 1

def log_api_call(model, task_id, prompt_tokens, response_tokens, duration_sec, cost_usd):
    log_path = _get_next_logfile()
    row = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "task_id": task_id,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "duration_sec": round(duration_sec, 2),
        "cost_usd": round(cost_usd, 6) if cost_usd is not None else ""
    }
    file_exists = log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists or os.stat(log_path).st_size == 0:
            writer.writeheader()
        writer.writerow(row)