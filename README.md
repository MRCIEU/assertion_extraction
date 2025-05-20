# Assertion Extraction from Biomedical Abstracts

This repository contains all code, configuration, and modules for an experimental pipeline to extract scientific assertions (subject–predicate–object triples) from biomedical literature. The core focus is on evaluating and comparing different large language models (LLMs) for multi-stage information extraction.

---

## Project Overview

The pipeline is divided into three main stages:

1. **Finding Detection** – Identify which sentences in an abstract contain scientific findings.
2. **Completion / Resolution** – Convert finding sentences into standalone factual statements.
3. **Assertion Extraction** – Convert standalone statements into structured triples: (subject, predicate, object), with optional conditions.

---

## Directory Structure

| Folder / File                      | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `01_sentence_classification/`     | Sentence-level classification experiments for finding detection.            |
| `01.1_binary_classification/`     | Old binary classifier experiments (e.g. SciBERT vs Logistic) – *deprecated*. |
| `02_extract_results_CITATIONS/`   | Tools to extract and prepare PubMed citation abstracts (CSV files, PMIDs). |
| `02.1_exmine_results_tag_API/`    | Evaluation of OpenAI API tagging outputs (e.g., result/finding matching).   |
| `03_extract_article_summary/`     | Early experiments in summarization / simplification of article content.    |
| `04_classification_model/`        | Transformer-based sentence classification models (e.g., BioBERT). – *deprecated*          |
| `05_three_single_pipeline/`       | **Main pipeline experiment. Includes finding → completion → assertion**.   |
| `.gitignore`                      | Specifies excluded file types (e.g. `.npy`, `.pt`).                         |
| `environment.yml`                 | Conda environment file with all required dependencies.                      |
| `README.md`                       | This project overview and usage instructions.                              |

---

## Current Main Experiment: `05_three_single_pipeline/`

This folder includes the three sequential LLM pipelines:

- `finding_pipeline.py` – Run GPT/Claude/LLaMA to identify finding sentences.
- `completion_pipeline.py` – Expand sentences into standalone factual units.
- `assertion_pipeline.py` – Extract subject–predicate–object triples.

Additional utilities:
- `schemas.py`, `json_utils.py`, `prompts.py` – shared dataclass schemas, JSON parsing, and prompt templates.
- `config.py` – API keys and model configuration (use environment variables in production).

- `eval/` – evaluation outputs and summary statistics.

---

## 🛠️ Environment Setup

```bash
conda env create -f environment.yml
conda activate hf-hpc
