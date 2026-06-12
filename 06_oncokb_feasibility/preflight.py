"""Preflight checks for step 06."""

from __future__ import annotations

import json
from pathlib import Path

import requests

from .config import ONCOKB_BASE_URL, TRAINING_PMIDS_JSON
from .oncokb_client import OncoKBClient


def check_training_pmids(path: Path = TRAINING_PMIDS_JSON) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Training PMID list missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    biored = set(data.get("biored_training_pmids") or [])
    drugprot = set(data.get("drugprot_training_pmids") or [])
    union = biored | drugprot
    return {
        "path": str(path),
        "biored_n": len(biored),
        "drugprot_n": len(drugprot),
        "union_n": len(union),
        "excluded_pmids": list(data.get("excluded_pmids") or []),
    }


def check_api_reachability() -> dict:
    response = requests.get(f"{ONCOKB_BASE_URL}/info", timeout=30)
    response.raise_for_status()
    return {"reachable": True, "status_code": response.status_code}


def run_preflight() -> dict:
    print("=== Step 06 preflight ===")
    training = check_training_pmids()
    print(f"  training PMIDs present: {training['union_n']} (BioRED {training['biored_n']}, DrugProt {training['drugprot_n']})")

    reach = check_api_reachability()
    print(f"  OncoKB API reachable: {reach['reachable']} (HTTP {reach['status_code']})")

    client = OncoKBClient()
    status, body, _ = client.get("info")
    if status != 200:
        raise RuntimeError(f"Authenticated /info failed with HTTP {status}")
    print(f"  authenticated API access: OK ({client.access_mode})")

    return {
        "training_pmids": training,
        "api_reachability": reach,
        "access_mode": client.access_mode,
        "api_info_status": status,
        "data_version": body.get("dataVersion") if isinstance(body, dict) else None,
    }
