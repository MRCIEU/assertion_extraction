"""Regression tests for `derive_label_space`.

Each schema in the factorial has a fixed number of classifier output units.
If `derive_label_space` ever returns a different count we silently corrupt
Phase A and Phase B (the classifier head reshapes, logits no longer align
with `schema_expected_label`, KB metrics become uninterpretable).

Run directly:
    python3.11 -m fine_tuning_experiments.phase_b.trainer.tests.test_label_space
"""
from __future__ import annotations

import sys
from pathlib import Path

from fine_tuning_experiments.phase_b.trainer.scientific_data import (
    NEG_LABEL,
    derive_label_space,
)

SHARD_ROOT = Path(
    "/lus/lfs1aip2/projects/b5ac/project_1/training_data_generation/data/processed"
)

EXPECTED = {
    "Sflat": {
        "n_labels": 4,
        "must_contain": {"ASSOCIATION_GENERAL", "DRUG_DISEASE",
                         "DRUG_GENE_REGULATION", NEG_LABEL},
        "pair_filter": "sflat_legal_endpoints",
        "shards": ["t1_biored_trn_Sflat.jsonl",
                   "t1_drugprot_trn_Sflat.jsonl",
                   "t1_bc5cdr_trn_Sflat.jsonl"],
    },
    "Spair": {
        "n_labels": 8,
        "must_contain": {"ASSOCIATION_GENERAL", NEG_LABEL},
        "pair_filter": "spair_legal_endpoints",
        "shards": ["t1_biored_trn_Spair.jsonl",
                   "t1_drugprot_trn_Spair.jsonl",
                   "t1_bc5cdr_trn_Spair.jsonl"],
    },
    "Smech": {
        "n_labels": 13,
        "must_contain": {NEG_LABEL},
        "pair_filter": "smech_legal_endpoints",
        "shards": ["t1_biored_trn_Smech.jsonl",
                   "t1_drugprot_trn_Smech.jsonl",
                   "t1_bc5cdr_trn_Smech.jsonl"],
    },
}


def run() -> int:
    failures: list[str] = []
    for schema, spec in EXPECTED.items():
        shard_paths = [SHARD_ROOT / s for s in spec["shards"]]
        missing = [str(p) for p in shard_paths if not p.exists()]
        if missing:
            failures.append(f"{schema}: missing shards {missing}")
            continue
        label2id = derive_label_space(shard_paths, spec["pair_filter"])
        n = len(label2id)
        labs = set(label2id)

        if n != spec["n_labels"]:
            failures.append(
                f"{schema}: got {n} labels, expected {spec['n_labels']}: {sorted(labs)}"
            )
        missing_req = spec["must_contain"] - labs
        if missing_req:
            failures.append(
                f"{schema}: missing required labels {sorted(missing_req)}"
            )
        if label2id.get(NEG_LABEL) != n - 1:
            failures.append(
                f"{schema}: {NEG_LABEL} not at last index ({label2id.get(NEG_LABEL)} vs {n - 1})"
            )
        print(f"[{schema}] n_labels={n} (ok={n == spec['n_labels']}) "
              f"required={len(spec['must_contain'])}/{len(spec['must_contain'])} "
              f"labels={sorted(labs)}")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS: all schemas match expected label counts.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
