"""Pull CIViC entities via GraphQL and cache locally."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .civic_client import CivicGraphQLClient
from .config import DATA_DIR, PAGE_SIZE

EVIDENCE_QUERY = """
query EvidenceItems($first: Int, $after: String, $status: EvidenceStatusFilter) {
  evidenceItems(first: $first, after: $after, status: $status) {
    totalCount
    pageInfo { hasNextPage endCursor }
    nodes {
      id name status description
      evidenceType evidenceDirection significance evidenceLevel evidenceRating
      variantOrigin therapyInteractionType
      source {
        id name sourceType citationId title abstract pmcId
        publicationYear publicationMonth publicationDay journal
      }
      molecularProfile {
        id name
        variants {
          id name
          feature { id name featureType }
        }
      }
      disease { id name doid }
      therapies { id name ncitId }
      phenotypes { id name }
      assertions {
        id name status assertionType assertionDirection significance
      }
    }
  }
}
"""

ASSERTION_QUERY = """
query Assertions($first: Int, $after: String, $status: EvidenceStatusFilter) {
  assertions(first: $first, after: $after, status: $status) {
    totalCount
    pageInfo { hasNextPage endCursor }
    nodes {
      id name status summary description
      assertionType assertionDirection significance
      evidenceItemsCount variantOrigin therapyInteractionType
      molecularProfile {
        id name
        variants { id name feature { id name featureType } }
      }
      disease { id name }
      therapies { id name ncitId }
      evidenceItems {
        id
        source { id sourceType citationId }
      }
    }
  }
}
"""

MOLECULAR_PROFILE_QUERY = """
query MolecularProfiles($first: Int, $after: String) {
  molecularProfiles(first: $first, after: $after) {
    totalCount
    pageInfo { hasNextPage endCursor }
    nodes {
      id name description
      variants {
        id name
        feature { id name featureType }
      }
    }
  }
}
"""

VARIANT_QUERY = """
query Variants($first: Int, $after: String) {
  variants(first: $first, after: $after) {
    totalCount
    pageInfo { hasNextPage endCursor }
    nodes {
      id name
      feature { id name featureType }
    }
  }
}
"""

FEATURE_QUERY = """
query BrowseFeatures($first: Int, $after: String) {
  browseFeatures(first: $first, after: $after) {
    totalCount
    pageInfo { hasNextPage endCursor }
    nodes {
      id name fullName featureInstanceType
    }
  }
}
"""

DATA_RELEASE_QUERY = """
query DataReleaseMeta {
  dataReleases {
    name
    acceptedEvidenceTsv { filename path }
    acceptedAssertionTsv { filename path }
  }
  timepointStats {
    evidenceItems { allTime newThisMonth newThisYear }
    assertions { allTime newThisMonth newThisYear }
  }
}
"""


def _save_json(records: list[dict], path: Path) -> None:
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _save_parquet(records: list[dict], path: Path) -> None:
    pd.DataFrame(records).to_parquet(path, index=False)


def fetch_all(force: bool = False) -> dict:
    """Fetch accepted CIViC records and write cache files."""
    cache_marker = DATA_DIR / "fetch_metadata.json"
    if cache_marker.exists() and not force:
        metadata = json.loads(cache_marker.read_text(encoding="utf-8"))
        print(f"Cache exists ({metadata['fetch_timestamp']}); use force=True to refresh.")
        return metadata

    client = CivicGraphQLClient()
    fetched_at = datetime.now(timezone.utc).isoformat()

    print("Fetching data release metadata...")
    release_meta = client.execute(DATA_RELEASE_QUERY)

    print("Fetching accepted evidence items...")
    evidence_items, evidence_total = client.paginate_connection(
        EVIDENCE_QUERY,
        ["evidenceItems"],
        {"status": "ACCEPTED"},
        PAGE_SIZE,
    )

    print("Fetching accepted assertions...")
    assertions, assertion_total = client.paginate_connection(
        ASSERTION_QUERY,
        ["assertions"],
        {"status": "ACCEPTED"},
        PAGE_SIZE,
    )

    print("Fetching molecular profiles...")
    molecular_profiles, mp_total = client.paginate_connection(
        MOLECULAR_PROFILE_QUERY,
        ["molecularProfiles"],
        {},
        PAGE_SIZE,
    )

    print("Fetching variants...")
    variants, variant_total = client.paginate_connection(
        VARIANT_QUERY,
        ["variants"],
        {},
        PAGE_SIZE,
    )

    print("Fetching features (browseFeatures)...")
    features, feature_total = client.paginate_connection(
        FEATURE_QUERY,
        ["browseFeatures"],
        {},
        PAGE_SIZE,
    )

    _save_json(evidence_items, DATA_DIR / "evidence_items.json")
    _save_parquet(evidence_items, DATA_DIR / "evidence_items.parquet")
    _save_json(assertions, DATA_DIR / "assertions.json")
    _save_parquet(assertions, DATA_DIR / "assertions.parquet")
    _save_json(molecular_profiles, DATA_DIR / "molecular_profiles.json")
    _save_json(variants, DATA_DIR / "variants.json")
    _save_json(features, DATA_DIR / "features.json")
    _save_json(release_meta, DATA_DIR / "release_meta.json")

    metadata = {
        "fetch_timestamp": fetched_at,
        "api_endpoint": "https://civicdb.org/api/graphql",
        "data_releases": release_meta.get("dataReleases", []),
        "timepoint_stats": release_meta.get("timepointStats", {}),
        "counts": {
            "accepted_evidence_items": evidence_total,
            "accepted_assertions": assertion_total,
            "molecular_profiles": mp_total,
            "variants": variant_total,
            "features": feature_total,
        },
    }
    cache_marker.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== Fetch summary ===")
    for key, value in metadata["counts"].items():
        print(f"  {key}: {value}")
    print(f"  fetch_timestamp: {fetched_at}")

    return metadata


if __name__ == "__main__":
    fetch_all(force=False)
