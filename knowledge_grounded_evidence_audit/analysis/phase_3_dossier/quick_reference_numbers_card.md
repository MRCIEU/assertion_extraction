# Quick reference — headline numbers (1 page)

Abbreviated lookup; authoritative detail in `MANUSCRIPT_EVIDENCE_DOSSIER.md` + `provenance_map.md`.

| Claim | Number | CI / SD | Source code |
|-------|--------|---------|-------------|
| R_B pre-registered 9-cell | **0.214** | **[0.028, 0.990]** | F |
| Variance share schedule (KB) | **59.6%** | — | F |
| Variance share schedule (BioRED ex-NEG) | **8.2%** | — | F |
| Multi-corpus BC5CDR Δ (PB anchor) | **+0.139** (d≈**2.04**) | **[+0.108, +0.166]** | F |
| Schema S_pair vs S_family (KB) | **+0.117** (d≈**0.55**) | **[+0.045, +0.194]** | E |
| Rank inversion rate (ρ=0.03) | **0.50** | **[0.14, 0.83]** cluster; CP **[0.26, 0.74]** | H |
| Drop-7 R_B (155 targets) | **0.217** | **[0.028, 1.015]** | A1 |
| Drop-7 rank inversion | **0.500** | [see H] | A1 |
| Per-cell KB CI width at n=100 | **0.063** | — | A1 |
| Per-cell KB CI width at n=150 | **0.021** | — | A1 |
| Bias mechanism Δ (disagreement-7) | **+0.762** | **[+0.712, +0.812]** | A1 |
| Bias mechanism Δ (random-7) | **+0.362** | **[+0.291, +0.433]** | A1 |
| Calibration ECE max-softmax | **0.277** | — | A1 |
| Calibration ECE pmass_B | **0.209** | — | A1 |
| T* temperature (pooled) | **2.10** | — | A1 |
| CIViCmine strict coverage | **25.3%** (**41/162**) | — | L |
| CIViCmine PMID-only | **48.8%** (**79/162**) | — | L |
| CIViCmine acc strict-41 | **0.951** | — | L |
| PB-T2 KB on same 41 | **0.855** | — | L |
| Always-DGR baseline 162 | **0.951** (**154/162**) | — | trivial |
| GPT-4o-mini zero-shot 162 / 41 | **0.988** / **1.000** | — | M |
| GPT-4o-mini 6-shot 162 / 41 | **0.920** / **0.927** | — | M |
| GPT-4o-mini 6-shot+rationale 162 / 41 | **0.926** / **0.951** | — | M |
| T1F-2048 KB mean (PB) | **0.477** | — | alpha |
| T1F-4096 KB mean (PB) | **0.619** | — | alpha |
| T2 KB mean (PB) | **0.756** | — | alpha |
| Δ_compute (paired mean) | **+0.142** | **[−0.017, +0.299]** boot | alpha |
| Δ_content (paired mean) | **+0.137** | **[+0.036, +0.240]** boot | alpha |
| α̂ (mean ratio) | **0.509** | **[−0.087, +0.868]** boot | alpha |
| PB 4-schedule R_B extension | **0.762** | **[0.011, 6.38]** | rbext |
| Augmented 10-cell R_B | **0.237** | **[0.039, 1.08]** | rbext |
| κ(heuristic, LLM Opus) | **0.561** | **[0.321, 0.790]** | author_iaa |
| κ(heuristic, author) | **0.434** | **[0.214, 0.654]** | author_iaa |
| κ(LLM, author) | **0.835** | **[0.628, 1.000]** | author_iaa |
| Fleiss 3-way | **0.603** | — | author_iaa |
| Author NEG on 7 IAA targets | **7/7** w/ LLM | — | author_iaa |
| Within-cell SD KB (PB T1F-4096, pstdev) | **0.169** | — | seedCSV |
| ICC(1,1) KB argmax (9×20 design) | **0.67** | — | paper† |

† **`paper`** = `report/project/sections/03_methods.tex`; independent recomputation from CSV **`[PROVENANCE UNKNOWN]`** in this dossier.

**A1** rows: values per Phase 3 brief — **`[PROVENANCE UNKNOWN]`** until `phase_c_robustness/outputs/` artefacts are pinned in repo (see `MANUSCRIPT_EVIDENCE_DOSSIER.md` §2).
