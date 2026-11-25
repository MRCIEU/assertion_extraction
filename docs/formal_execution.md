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

Public checkpoints that resolve without an access token, and whose weights are not yet local, are BioLinkBERT-base `b71f5d70f063d1c8f1124070ce86f1ee463ca1fe` (Apache-2.0), DeBERTa-v3-base-mnli-fever-anli `6f5cf0a2b59cabb106aca4c287eed12e357e90eb` (MIT), GPT-OSS-20B `6cee5e81ee83917806bbde320786a8fb61efebee` (Apache-2.0) and Qwen3.6-27B `1cda14dd71a3f47115850eb790f416f165da1cb5` (Apache-2.0). They are not described as execution-ready until the files are stored under the project model directory. The twelve historical DrugProt BiomedBERT conditions each have a non-empty best checkpoint and a completion record, so they are the reused encoder runs.

A CPU fixture in `hf-hpc` trained a four-head encoder for four steps, resumed from a real checkpoint at step two, and matched the uninterrupted weights. Every head and the shared embedding received a gradient. This is not a Qwen result. Few-shot and supervised token-length distributions have not yet been measured.

Resolved tasks keep `executable` false until GPU preflight and approval. Each task names an executor: BiomedBERT and BioLinkBERT use `formal_train.train`, Qwen LoRA including shared-schema and curriculum uses `production_lora.fit`, and the hierarchical and encoder-sensitivity losses use `production_encoder`. The approval digest is the hash of those resolved task definitions and the git revision of the checkout, not a count summary. Proposed Slurm jobs are queued; they are not treated as already running. The running-job ceiling remains 256 and the queued-plus-running ceiling remains 512.

BioLinkBERT-base and the DeBERTa base verifier are stored locally. On the SciFact development rationale task the verifier scored 444 assessable pairs, with accuracy 0.541; eight pairs exceeded the 512-token context and were left unassessable. That figure is not biological truth. GPT-OSS-20B (41 GB) loads as `GptOssConfig` under Transformers 4.57.5, and `GptOssForCausalLM` is present, but it has not been executed. Qwen3.6-27B (56 GB) is `qwen3_5`, which this Transformers release does not recognise. Its weights stay on disk and its tasks stay blocked; the environment is not being upgraded to force recognition.

## Production path

`production_plan.py` enumerates the agreed conditions. With the twelve completed DrugProt BiomedBERT checkpoints reused, the new principal training count is 312 of 324 designed runs. BioLinkBERT has 48 runs, each hierarchical comparison has 12, schema-conditioned LoRA has 24, and the Qwen3-8B curriculum has 27. Sensitivity runs are counted separately and do not repeat the standard-loss or full-population baselines.

The multi-task loss used by the trainer is an equal mean of the four predicate-head losses and stays on the autograd graph. DrugProt consistency uses a squared hinge that requires each parent probability to be at least its child. The mapped native distribution is a detached target for the BioRED divergence.

`formal_train.py` refuses to start unless `formal_execution_approval.json` names the manifest checksum, the repository commit and account `brics.b5ac`. When `SCHEMA_STUDY_PROGRAMME=formal_matched_e0_e7`, BioRED and BC8 read the matched input collection and write to a new run directory. The frozen November run is left in place. GPU batch scripts activate `hf-hpc`. No approval file has been written.

## What remains blocked

Assistant-only loss masking, distributed accumulation, hierarchical losses, schema sampling, curriculum order and nested subsets are implemented and tested with fixtures. They have not been run on a GPU. Few-shot and supervised token lengths have not been measured; the existing Qwen lengths are zero-shot. Human certainty and support judgements remain unfilled. No formal job will be submitted until `formal_execution_approval.json` is written in the study directory and the remaining node-hour balance is known.
