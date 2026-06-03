"""Fetch and cache PubTator3 biocjson annotations."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from .config import (
    BATCH_SIZE,
    PUBTATOR_API,
    PUBTATOR_CACHE_JSON,
    PUBTATOR_METADATA_JSON,
    REQUEST_INTERVAL_S,
)


def _fetch_batch(pmids: list[str]) -> list[dict[str, Any]]:
    query = ",".join(pmids)
    url = f"{PUBTATOR_API}?pmids={query}"
    req = urllib.request.Request(url, headers={"User-Agent": "project_1/03_candidate_pool"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read())
    return payload.get("PubTator3") or []


def fetch_pubtator_annotations(
    pmids: list[str],
    force: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """
    Return {pmid: biocjson_document} and fetch metadata.
    Uses on-disk cache unless force=True or PMIDs are missing.
    """
    pmids = sorted({str(p) for p in pmids if p})
    cached: dict[str, dict[str, Any]] = {}
    if PUBTATOR_CACHE_JSON.exists() and not force:
        cached = json.loads(PUBTATOR_CACHE_JSON.read_text(encoding="utf-8"))

    missing = [p for p in pmids if p not in cached]
    print(f"PubTator3: {len(pmids)} PMIDs requested, {len(cached)} cached, {len(missing)} to fetch")

    fetched_at = datetime.now(timezone.utc).isoformat()
    n_requests = 0
    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i : i + BATCH_SIZE]
        if not batch:
            continue
        if n_requests > 0:
            time.sleep(REQUEST_INTERVAL_S)
        try:
            docs = _fetch_batch(batch)
        except urllib.error.HTTPError as exc:
            if len(batch) > 1:
                # Fallback: smaller batches on failure
                for pmid in batch:
                    time.sleep(REQUEST_INTERVAL_S)
                    docs = _fetch_batch([pmid])
                    for doc in docs:
                        cached[str(doc.get("pmid") or doc.get("id", "")).replace("PMID:", "")] = doc
                    n_requests += 1
                continue
            raise exc
        for doc in docs:
            pmid = str(doc.get("pmid") or doc.get("id", "")).replace("PMID:", "")
            if pmid:
                cached[pmid] = doc
        n_requests += 1
        print(f"  fetched batch {i // BATCH_SIZE + 1}/{(len(missing) + BATCH_SIZE - 1) // BATCH_SIZE} "
              f"({len(batch)} PMIDs)")

    # Keep only requested PMIDs in cache write
    out = {p: cached[p] for p in pmids if p in cached}
    PUBTATOR_CACHE_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    metadata = {
        "api_endpoint": PUBTATOR_API,
        "format": "biocjson",
        "fetch_timestamp": fetched_at,
        "n_pmids_requested": len(pmids),
        "n_pmids_cached": len(out),
        "n_api_requests": n_requests,
        "batch_size": BATCH_SIZE,
        "request_interval_s": REQUEST_INTERVAL_S,
        "source": "PubTator3 precomputed annotations (NCBI)",
    }
    PUBTATOR_METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out, metadata
