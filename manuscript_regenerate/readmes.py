"""Write minimal README files for each step code folder."""

from __future__ import annotations

from pathlib import Path

from .paths import REPO, STEPS

README_BODIES: dict[str, str] = {
    "00": """# Step 00 — CIViC feasibility

Pulls accepted CIViC evidence and inventories PubMed-backed two-entity targets for relation presence ranking.

Method: GraphQL fetch of accepted evidence, abstract-grounding check, entity-pair inventory.

Results: 4856 accepted evidence items; 4674 evaluable two-entity targets; 2074 abstract-grounded pairs. Step 02 freezes 1812 gene-drug and gene-disease targets; 262 variant pairs are excluded from ranking.
""",
    "01": """# Step 01 — Corpus alignment and CIViC relevance

Maps training corpora onto CIViC pair types, audits PMID leakage, and quantifies oncology-intersection training volume.

Method: Alignment matrix across BioRED, DrugProt, and BC5CDR; PMID overlap audit; oncology criteria on gene-drug and gene-disease training relations.

Results: BioRED admissible 4/4 pair types; DrugProt partial 1/4; BC5CDR 0/4. Three leaked PMIDs excluded before training. BioRED oncology intersection: 1086 gene-disease relations under the all-criteria rule.
""",
    "02": """# Step 02 — Ranking evaluation protocol

Freezes evaluable ranking targets and defines metrics only.

Method: Freeze abstract-grounded gene-drug and gene-disease positives from step 00; define MRR, Recall@k, and AUC-PR.

Results: 1812 targets across 915 PMIDs (1230 gene-drug; 582 gene-disease). 262 variant pairs excluded. Trivial baselines run in step 03.
""",
    "03": """# Step 03 — Candidate pool

Builds PubTator3 candidate pools for the 1812 frozen targets.

Method: Per-abstract pool construction with frozen matching rules; trivial ranking baselines on the primary pool.

Results: 18911 primary candidates; 1590 matched and 222 missed recall (87.7%). Distance ranker MRR 0.489 versus random 0.322. Entity-type granularity gaps inflate pools common-mode across encoders.
""",
    "04": """# Step 04 — Pilot study

Minimal-training pilot on three encoders with step-03 pool scoring under the pre-fix pipeline.

Method: Short training run on BioRED plus DrugProt; score frozen pool at best checkpoint.

Results: PubMedBERT MRR 0.469 versus random 0.322 and distance ranker 0.489; reference benchmark F1 0.893. Hard-subset ranking beats the distance ranker. Not comparable to post-fix step-10 matrix.
""",
    "05": """# Step 05 — Marker quality gate

Verifies offset-based entity marker insertion and rebuilds train caches.

Method: Compare native-offset insertion against prior string-match insertion on training, benchmark, and pool evaluation paths.

Results: Offset gate passed. Training offset insertion 100%. Rebuilt caches for steps 10, 11, and 20.
""",
    "10": """# Step 10 — Recipe sweep and training matrix

Selects a stable recipe and trains nine encoders by eight seeds.

Method: Learning-rate and warmup sweep with DeBERTa health gate; full matrix at confirmed recipe with per-epoch checkpoints.

Results: DeBERTa gate failure at 3e-5/warmup; confirmed recipe 5e-6/none. Matrix benchmark F1 spread 0.025 across nine encoders (means roughly 0.72 to 0.75).
""",
    "11": """# Step 11 — Round-one encoder comparison

Compares nine encoders on in-distribution benchmark and out-of-distribution CIViC ranking at a single checkpoint.

Method: Score seventy-two fine-tuned runs plus nine untrained-floor references on both axes; variance decomposition, association bootstrap, and calibration diagnostics.

Results: Benchmark spread 0.025; variance shares 36/64 benchmark, 23/77 gene-drug KB, 13/87 gene-disease KB. Fine-tuned KB means 0.676 gene-drug and 0.625 gene-disease. Seed-level benchmark–KB association negative on both pair types.
""",
    "20": """# Step 20 — Training dynamics diagnostic

Scores per-epoch checkpoints on benchmark and CIViC ranking to test within-model training effects.

Method: Pairwise comparison from epoch 1 to best validation-F1 checkpoint, split by gene-drug and gene-disease.

Results: 498 epoch checkpoints; 65 pairable seeds. Pooled hard-subset KB delta -0.0016; gene-disease -0.0569 (48/65 fall); gene-drug +0.0080. Verdict: gene_disease_biomed_pretraining_erosion (within-model biomed-pretraining erosion; regular encoder heterogeneity).
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
    "11": "11_round1_analysis",
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
    skip = {"20"}  # Step 20 README is owned by 20_round2_diagnostic/report.py
    return [write_readme(key, repo=repo) for key in README_BODIES if key not in skip]
