# Formal execution readiness

Formal GPU execution is not approved. The manifest command writes an inventory and does not call `sbatch`.

## Environment

Non-interactive jobs source `/home/b5ac/freddieyu.b5ac/miniforge3/etc/profile.d/conda.sh` and then run `conda activate hf-hpc`. The resolved interpreter is `/home/b5ac/freddieyu.b5ac/miniforge3/envs/hf-hpc/bin/python`.

Recorded package versions in that environment: Python site packages give PyTorch 2.9.1, Transformers 4.57.5, PEFT 0.14.0, Accelerate 1.12.0, vLLM 0.13.0 and Jinja2 3.1.6. spaCy is absent. Installing spaCy into this environment would change a shared stack that already contains vLLM, so it has not been installed. The completed scispaCy comparison remains in its separate record. This login node has no NVIDIA driver, so CUDA execution was not attempted here.

## Accounting

`workq_qos` allows 256 running jobs, 512 pending or running jobs, and a wall-time maximum of 24 hours. `scontrol` reports `workq` itself as having unlimited partition time; the QoS is the operative limit. The user association is `brics.b5ac` with QoS `normal`.

`sshare` reports a fair-share value. It does not report remaining node-hour credits. Those credits are shown in the Awards Portal, which is not available from Slurm. The full matrix is therefore not submitted.

A recent CPU request for 8 CPUs was allocated `cpu=72,billing=72` and no GPU. On Isambard-AI, 72 CPU cores are one quarter of a node, so the job is charged 0.25 node-hours per wall-clock hour, the same minimum as one GPU. CPU-only work on this service is not free of GPU-node allocation.

One historical DrugProt BiomedBERT run recorded about 1,320 to 1,345 seconds per epoch and 28,212 updates, which matches 451,387 training candidates at an effective batch of 16. That observation is not a forecast for BioRED or Qwen. Per-job GPU elapsed times were not returned by `sacct` for the earlier formal job names, so the previous aggregate of 392,433 GPU seconds is not scaled up to the new matrix.

## Training inventory

The designed supervised matrix has 324 runs: 288 BioRED runs across three trainable systems and 36 DrugProt runs. BC8 adds evaluation only. The matched BioRED training windows changed, so the 96 BiomedBERT BioRED runs and all 216 Qwen LoRA runs are new. Twelve historical DrugProt BiomedBERT checkpoints remain, one for each native predicate and seed, and they are not repeated. The resulting new principal training count is 312.

Supplementary configurations, including BioLinkBERT revision `b71f5d70f063d1c8f1124070ce86f1ee463ca1fe` and the public DeBERTa base verifier revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`, are listed separately. Their weights have not been downloaded. The verifier label map is entailment, neutral, contradiction. The previously selected xsmall verifier still returns HTTP 401 to anonymous access.

## What remains blocked

Assistant-only loss masking, distributed accumulation, hierarchical losses, schema sampling, curriculum order and nested subsets are implemented and tested with fixtures. They have not been run on a GPU. Few-shot and supervised token lengths have not been measured; the existing Qwen lengths are zero-shot. Human certainty and support judgements remain unfilled. No formal job will be submitted until `formal_execution_approval.json` is written in the study directory and the remaining node-hour balance is known.
