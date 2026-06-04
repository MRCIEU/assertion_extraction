# Round 1 — Benchmark rank vs KB ranking and calibration

First main-experiment round. Fixed BioRED + DrugProt presence training; only encoder and seed vary.

## Question

Does a model's self-measured BioRED benchmark score predict its ranking quality and calibration on the frozen CIViC candidate pool? Either alignment or divergence is a valid finding.

## Fixed training strategy (settled by a small sweep)

| Setting | Value |
|---------|--------|
| Learning rate | 2e-5 |
| Warmup | none |
| Checkpoint | best **validation F1** (not val_loss) |
| Early stopping | max 10 epochs, patience 3 |

## Design

- **9 encoders:** PubMedBERT, BioMedBERT, BioLinkBERT, BioBERT, SciBERT, RoBERTa, BERT, DistilBERT, DeBERTa
- **8 seeds** per encoder (42–49) → 72 runs
- **Primary analysis:** 70 clean runs (DeBERTa seeds 45 and 49 excluded from all metrics)
- Blocking leak check: PMIDs 16434489, 18794803, 23430109 absent from training

## Key numbers

| Quantity | Value |
|----------|--------|
| Benchmark F1 range (9 encoders) | 0.725 – 0.785 (spread 0.060) |
| Gene-drug KB MRR range | 0.623 – 0.680 (DeBERTa high end on clean seeds) |
| DeBERTa benchmark F1 (6 seeds) | 0.739 |
| DeBERTa benchmark F1 (8 seeds, sensitivity) | 0.554 |
| Benchmark vs ECE Spearman (primary, clean seeds) | -0.783 |
| Benchmark vs ECE Spearman (all 8 seeds per encoder) | -0.867 |