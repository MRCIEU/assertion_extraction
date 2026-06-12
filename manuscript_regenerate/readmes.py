"""Write minimal README files for each step code folder."""

from __future__ import annotations

from pathlib import Path

from .paths import REPO, STEPS

README_BODIES: dict[str, str] = {
    "00": """# Step 00 — CIViC feasibility

Pulls accepted CIViC evidence via GraphQL and inventories PubMed-backed two-entity targets for relation presence ranking.

**Result:** 4856 accepted evidence items; 4674 evaluable two-entity targets; 2074 abstract-grounded pairs (both entities in the abstract). Step 02 freezes 1812 gene-drug and gene-disease targets; 262 variant pairs are excluded from ranking.
""",
    "01": """# Step 01 — Corpus alignment and CIViC relevance

Maps BioRED, DrugProt, and BC5CDR onto CIViC pair types, audits PMID leakage, and quantifies oncology-intersection training volume.

**Result:** BioRED admissible 4/4 pair types; DrugProt partial 1/4; BC5CDR 0/4. Three leaked PMIDs must be excluded before training. BioRED oncology intersection: 1086 gene-disease relations under the conservative all-criteria rule.
""",
    "02": """# Step 02 — Ranking evaluation protocol

Freezes evaluable ranking targets and defines metrics only (no model scores).

**Result:** 1812 abstract-grounded gene-drug and gene-disease positives across 915 PMIDs (1230 gene-drug; 582 gene-disease). The step-00 inventory contains 2074 abstract-grounded pairs; 262 variant pairs are not evaluable. Metrics: MRR, Recall@k, AUC-PR. Trivial baselines run in step 03.
""",
    "03": """# Step 03 — Candidate pool

Builds PubTator3 candidate pools for the 1812 frozen targets from step 02.

**Result:** 18911 primary-scope candidates; 1590/1812 matched recall (87.8%). Entity-type granularity gaps (PubTator Chemical versus CIViC drug) inflate pools common-mode across encoders. Mean pool size ~10.3; distance ranker MRR 0.489 versus random 0.322.
""",
    "04": """# Step 04 — Pilot study

Minimal-training pilot (pre-fix pipeline) on three encoders with step-03 pool scoring.

**Result:** PubMedBERT MRR 0.469 versus random 0.322 and distance ranker 0.489 on the full pool; reference benchmark F1 0.893 from literature axis in 04_pilot_study_benchmark_vs_kb.csv. Hard-subset ranking beats the distance ranker. Not comparable to post-fix step-10 matrix.
""",
    "05": """# Step 05 — Marker quality gate

Verifies offset-based entity marker insertion across training, benchmark, and CIViC evaluation; rebuilds train caches.

**Result:** Offset gate passed. Training offset insertion 100%. Shared marker_insert path for train, benchmark, and pool evaluation. Rebuilt caches for steps 10, 11, and 20.
""",
    "10": """# Step 10 — Recipe sweep and training matrix

Selects a stable recipe and trains nine encoders by eight seeds at benchmark-only monitoring; saves per-epoch checkpoints for step 20.

**Result:** DeBERTa gate failure at 3e-5/warmup; confirmed recipe 5e-6/none. Matrix mean benchmark F1 spread ~0.72–0.75 across nine encoders. KB scoring happens in folders 11 and 20.
""",
    "20": """# Step 20 — Training dynamics diagnostic

Scores per-epoch checkpoints from the step-10 matrix on both benchmark and CIViC ranking axes; adjudicates mechanistic erosion versus static pool mismatch.

**Result:** 498 epoch checkpoints; 65 pairable seeds. Pooled hard-subset KB delta -0.0016; gene-disease -0.0569 (48/65 fall); gene-drug +0.0080. Verdict: mixed_gene_disease_signal.
""",
}

STEP_DIRS: dict[str, str] = {
    "00": "00_civic_feasibility",
    "01": "01_corpus_relevance",
    "02": "02_evaluation_protocol",
    "03": "03_candidate_pool",
    "04": "04_pilot_study",
    "05": "05_marker_quality_gate",
    "10": "10_recipe_sweep_and_training",
    "20": "20_round2_diagnostic",
}


def write_readme(step_key: str, repo: Path | None = None) -> Path:
    repo = repo or REPO
    body = README_BODIES[step_key]
    folder = STEP_DIRS.get(step_key, STEPS[step_key])
    path = repo / folder / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return path


def write_all_readmes(repo: Path | None = None) -> list[Path]:
    return [write_readme(key, repo=repo) for key in README_BODIES]
