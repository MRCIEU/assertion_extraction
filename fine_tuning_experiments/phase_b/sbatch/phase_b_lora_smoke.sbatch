#!/bin/bash -l
#SBATCH -J PB_LORA_SMOKE
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/LORA_SMOKE_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/LORA_SMOKE_%j.err

# End-to-end LoRA smoke for v2 trainer re-implementation (post-2026-04-24
# source-tree deletion).  Runs a short (max_updates=128) LoRA training
# then verifies best.pt is in the FT-compatible merged schema and that
# eval_one_run.py loads it without peft.

set -euo pipefail
export PROJECT_1_DATA_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/..
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
export HF_HOME=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/../hf_hub_cache
export TRANSFORMERS_CACHE=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/../hf_hub_cache
cd /home/b5ac/freddieyu.b5ac/project_1

EXP_ID=PB_PB_LR_T2_s99
CFG=/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/phase_b/configs/${EXP_ID}.yaml
RUN_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b_smoke

mkdir -p "$RUN_ROOT" "/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b"
# Always start from a clean smoke dir so we exercise the full write path.
rm -rf "$RUN_ROOT/$EXP_ID"

echo "[$(date -u +%FT%TZ)] Starting LoRA smoke ${EXP_ID}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

/usr/bin/python3.11 -m fine_tuning_experiments.train.run_experiment \
    --experiment-id "${EXP_ID}" \
    --config-path "${CFG}" \
    --run-root "${RUN_ROOT}"

echo ""
echo "=== Post-training verification ==="
ls -la "$RUN_ROOT/$EXP_ID/checkpoints/"

# Verify best.pt contains the FT-compatible merged schema + lora_meta with v2 tag.
/usr/bin/python3.11 - <<'PY'
import torch
p = "/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b_smoke/PB_PB_LR_T2_s99/checkpoints/best.pt"
c = torch.load(p, map_location="cpu", weights_only=False)

assert "model_state_dict" in c, "missing model_state_dict"
assert "label2id" in c, "missing label2id"
assert "lora_meta" in c, "missing lora_meta"
assert c["lora_meta"]["enabled"] is True, "lora_meta.enabled not True"
assert c["lora_meta"]["merged_at_save"] is True, "lora_meta.merged_at_save not True"
assert c["lora_meta"]["r"] == 16, f"r={c['lora_meta']['r']} != 16"
assert c["lora_meta"]["alpha"] == 32, f"alpha={c['lora_meta']['alpha']} != 32"
assert c["lora_meta"]["trainer_version"].startswith("v2"), (
    f"trainer_version={c['lora_meta']['trainer_version']!r}"
)

sd = c["model_state_dict"]
peft_keys = [k for k in sd if "lora_" in k.lower() or "adapter" in k.lower()]
assert not peft_keys, f"residual adapter keys: {peft_keys[:5]}"
classifier_keys = [k for k in sd if "classifier" in k]
assert any(k.endswith("weight") for k in classifier_keys)
assert any(k.endswith("bias") for k in classifier_keys)
print(f"best.pt schema OK  n_keys={len(sd)}  lora_meta={c['lora_meta']}")
PY

# Verify eval_one_run can load it (full eval — exercises the BiomedBERT
# path end-to-end).
/usr/bin/python3.11 -m fine_tuning_experiments.phase_b.eval.eval_one_run \
    --run-dir "$RUN_ROOT/$EXP_ID" \
    --overwrite

echo ""
echo "=== Post-eval verification ==="
ls -la "$RUN_ROOT/$EXP_ID/eval/"
/usr/bin/python3.11 - <<'PY'
import json
p = "/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b_smoke/PB_PB_LR_T2_s99/eval/phase_b_eval.json"
r = json.load(open(p))
assert r["update_key"] == "LR", r["update_key"]
assert r["update_regime"] == "lora", r["update_regime"]
assert r["schedule_key"] == "T2", r["schedule_key"]
assert r["schema_key"] == "Spair", r["schema_key"]
assert "biored_test" in r and "bc5cdr_test" in r and "kb_surface" in r
bio_f1 = r["biored_test"]["macro_f1"]
bc_f1 = r["bc5cdr_test"]["macro_f1"]
kb = r["kb_surface"]["kb_surface_mean"]
print(f"eval OK  biored_macro_f1={bio_f1:.4f}  bc5cdr_macro_f1={bc_f1:.4f}  kb_surface_mean={kb:.4f}")
PY

echo ""
echo "[$(date -u +%FT%TZ)] LoRA smoke PASSED"
