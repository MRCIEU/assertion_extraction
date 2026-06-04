# Encoder recipe check

DeBERTa-only grid: learning rate and warmup, seed 42, same data and benchmark as Round 1. No KB scoring.

## Key numbers (seed 42)

| Recipe | Best epoch | Benchmark F1 |
|--------|------------|----------------|
| lr 1e-5, no warmup | 5 | **0.774** |
| lr 1e-5, 10% warmup | 7 | 0.765 |
| lr 2e-5, 10% warmup | 9 | 0.738 |
| lr 2e-5, no warmup (Round-1 recipe) | 2 | 0.721 |

- Learning rate (mean over warmup): 1e-5 → **0.769**, 2e-5 → **0.730** (delta about −0.04)
- Warmup (mean over lr): none → **0.747**, 10% warmup → **0.752** (delta about +0.005)
- Round-1 eight encoders: **0.725–0.785**; DeBERTa clean six-seed mean **0.739**; eight-seed mean with collapsed seeds **0.554**

Single-seed evidence only (seed 42)
