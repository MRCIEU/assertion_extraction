"""Write minimal README files for each step (reviewed location: reports/<step>/)."""

from __future__ import annotations

from pathlib import Path

from .paths import OUTPUT_ROOT, REPO, STEPS, step_paths

README_BODIES: dict[str, str] = {
    "00": """# Step 00 — CIViC feasibility

Pulls accepted CIViC evidence and inventories PubMed-backed two-entity targets for relation presence ranking.

Method: GraphQL fetch of accepted evidence, abstract-grounding check, entity-pair inventory.

| Metric | Count |
| --- | ---: |
| Accepted evidence items | 4856 |
| Evaluable two-entity targets | 4674 |
| Abstract-grounded pairs | 2074 |
| Frozen gene-drug + gene-disease (step 02) | 1812 |
| Variant pairs excluded | 262 |
""",
    "01": """# Step 01 — Corpus alignment and CIViC relevance

Maps training corpora onto CIViC pair types, audits PMID leakage, and quantifies oncology-intersection training volume.

Method: Alignment matrix across BioRED, DrugProt, and BC5CDR; PMID overlap audit; oncology criteria on gene-drug and gene-disease training relations.

| Corpus | CIViC relevance | Admissible pair types | Leakage |
| --- | ---: | ---: | --- |
| BioRED | high | 4/4 | 0 PMIDs |
| DrugProt | partial | 1/4 | — |
| BC5CDR | low | 0/4 | — |
| Combined training-evaluation overlap | — | — | 3 PMIDs excluded |

BioRED oncology intersection (gene-disease, all three criteria): 1086 relations.
""",
    "02": """# Step 02 — Ranking evaluation protocol

Freezes evaluable ranking targets and defines metrics only.

Method: Freeze abstract-grounded gene-drug and gene-disease positives from step 00; define MRR, Recall@k, and AUC-PR.

| Target set | Count |
| --- | ---: |
| Frozen ranking targets | 1812 |
| PMIDs | 915 |
| Gene-drug | 1230 |
| Gene-disease | 582 |
| Variant pairs excluded | 262 |
""",
    "03": """# Step 03 — Candidate pool

Builds PubTator3 candidate pools for the 1812 frozen targets.

Method: Per-abstract pool construction with frozen matching rules; trivial ranking baselines on the primary pool.

| Pool metric | Value |
| --- | ---: |
| Primary candidates | 18911 |
| Matched recall | 1590 (87.7%) |
| Missed recall | 222 |
| Random MRR | 0.322 |
| Distance ranker MRR | 0.489 |
""",
    "04": """# Step 04 — Pilot study

Minimal-training pilot on three encoders with step-03 pool scoring under the pre-fix pipeline.

Method: Short training run on BioRED plus DrugProt; score frozen pool at best checkpoint.

| Model / baseline | MRR |
| --- | ---: |
| Random | 0.322 |
| Distance ranker | 0.489 |
| PubMedBERT-base (pilot) | 0.469 |
| Reference benchmark F1 (PubMedBERT) | 0.893 |

Not comparable to post-fix step-10 matrix.
""",
    "05": """# Step 05 — Marker quality gate

Verifies offset-based entity marker insertion and rebuilds train caches.

Method: Compare native-offset insertion against prior string-match insertion on training, benchmark, and pool evaluation paths.

| Check | Result |
| --- | ---: |
| Offset gate | passed |
| Training offset insertion | 100% |
| Downstream caches rebuilt | steps 10, 11, 20 |
""",
    "10": """# Step 10 — Recipe sweep and training matrix

Selects a stable recipe and trains nine encoders by eight seeds.

Method: Learning-rate and warmup sweep with DeBERTa health gate; full matrix at confirmed recipe with per-epoch checkpoints.

| Recipe outcome | Value |
| --- | ---: |
| DeBERTa gate failure | 3e-5 / warmup |
| Confirmed recipe | 5e-6 / none |
| Matrix benchmark F1 spread | 0.025 |
| Epoch checkpoints (for step 20) | 498 |
""",
    "11": """# Step 11 — Round-one encoder comparison

Compares nine encoders on in-distribution benchmark and out-of-distribution CIViC ranking at a single checkpoint.

Method: Score seventy-two fine-tuned runs plus nine untrained-floor references on both axes; variance decomposition and seed-level association bootstrap.

| Axis | Between / within encoder variance | Fine-tuned mean |
| --- | --- | ---: |
| Benchmark F1 | 36% / 64% | spread 0.025 |
| KB MRR gene-drug | 23% / 77% | 0.676 |
| KB MRR gene-disease | 13% / 87% | 0.625 |

Seed-level benchmark–KB Spearman: gene-drug negative; gene-disease negative (interval-heavy).
""",
    "20": """# Step 20 — Training dynamics diagnostic

Scores per-epoch checkpoints on benchmark and CIViC ranking to test within-model training effects.

Method: Pairwise comparison from epoch 1 to best validation-F1 checkpoint, split by gene-drug and gene-disease; mundane-explanation and qualitative deepening.

| Metric | Value |
| --- | ---: |
| Epoch checkpoints | 498 |
| Pairable seeds | 65 |
| Pooled hard KB delta | -0.0016 |
| Gene-drug KB delta | +0.0080 |
| Gene-disease KB delta | -0.0569 (48/65 fall) |
| Gene-disease-hard bootstrap P(negative) | 99.1% |

Pushing the in-distribution benchmark erodes out-of-distribution gene-disease ranking for biomedically pretrained encoders in a regular, predictable pattern; gene-drug stays flat or positive.
""",
}


def write_readme(step_key: str, repo: Path | None = None) -> Path:
    repo = repo or REPO
    body = README_BODIES[step_key]
    step_name = STEPS[step_key]
    paths = step_paths(step_name)
    report_path = paths["reports"] / "README.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(body.strip() + "\n", encoding="utf-8")
    # Optional mirror under code folder for discoverability.
    folder = step_name
    code_path = repo / folder / "README.md"
    code_path.parent.mkdir(parents=True, exist_ok=True)
    code_path.write_text(body.strip() + "\n", encoding="utf-8")
    return report_path


def write_all_readmes(repo: Path | None = None) -> list[Path]:
    return [write_readme(key, repo=repo) for key in README_BODIES]
