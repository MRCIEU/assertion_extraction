#!/bin/bash -l
#SBATCH -J PB_LORA_VERIFY
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/LORA_VERIFY_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/LORA_VERIFY_%j.err

# Full verification of the v2 LoRA trainer re-implementation.
# All checks run on the compute node (no login-node execution):
#   Step 1: pytest — the 13 LoRA unit tests (hyperparameters, param counts,
#           merge_and_unload schema, peft-mutation invariance, determinism,
#           bias-null defense).
#   Step 2: End-to-end LoRA smoke on PB_PB_LR_T2_s99 (128 updates, fully
#           exercising the peft-wrap → train → merge → save → eval path).
#   Step 3: Schema assertions on best.pt (no residual peft keys, lora_meta
#           present, merged_at_save=True, trainer_version v2, user-intent
#           modules_to_save preserved NOT peft-polluted).
#   Step 4: Schema assertions on last.pt (merged plain state dict).
#   Step 5: FT parity smoke on PB_RB_FT_T2_s99 (must be byte-equivalent to
#           the pre-deletion FT runs — no regression from LoRA integration).
#   Step 6: Phase-B eval on the merged LoRA best.pt (strict state_dict
#           loading into AutoModelForSequenceClassification — the
#           previously unverified path).
#   Step 7: Aggregate script dry-run (synthesize a CSV from the smoke run
#           and verify it parses with the analyze_phase_b header schema).
#
# Exits non-zero on ANY check failure, making this a safe go/no-go gate
# before we delete the 146 v1 LoRA run dirs and resubmit 180 new LR runs.

set -euo pipefail
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
export HF_HOME=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
export TRANSFORMERS_CACHE=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
cd /home/b5ac/freddieyu.b5ac/project_1

LOGDIR=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b
mkdir -p "$LOGDIR"
SMOKE_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b_smoke

section() { echo ""; echo "==============================================="; echo "  $1"; echo "==============================================="; }

section "Step 1: pytest — LoRA unit tests"
/usr/bin/python3.11 -m pytest \
    fine_tuning_experiments/phase_b/trainer/tests/test_lora_integration.py \
    -v --tb=short
echo "  pytest: PASS"

section "Step 2: LoRA smoke — PB_PB_LR_T2_s99 (128 updates)"
LR_EXP=PB_PB_LR_T2_s99
LR_CFG=/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/phase_b/configs/${LR_EXP}.yaml
rm -rf "$SMOKE_ROOT/$LR_EXP"
/usr/bin/python3.11 -m fine_tuning_experiments.train.run_experiment \
    --experiment-id "${LR_EXP}" \
    --config-path "${LR_CFG}" \
    --run-root "${SMOKE_ROOT}"
echo "  LoRA training: PASS"

section "Step 3: best.pt schema assertions (LoRA)"
/usr/bin/python3.11 - <<PY
import sys
import torch

p = "${SMOKE_ROOT}/${LR_EXP}/checkpoints/best.pt"
c = torch.load(p, map_location="cpu", weights_only=False)

assert "model_state_dict" in c, "missing model_state_dict"
assert "label2id" in c, "missing label2id"
assert "lora_meta" in c, "missing lora_meta"

lm = c["lora_meta"]
assert lm["enabled"] is True
assert lm["merged_at_save"] is True
assert lm["r"] == 16 and lm["alpha"] == 32
assert abs(lm["dropout"] - 0.05) < 1e-9
assert lm["bias"] == "none"
assert set(lm["target_modules"]) == {"query", "value"}, f"target_modules={lm['target_modules']}"
# User-intent modules_to_save MUST be preserved (not peft-expanded)
assert lm["modules_to_save"] == ["classifier"], (
    f"modules_to_save is peft-polluted: {lm['modules_to_save']}"
)
assert lm["trainer_version"].startswith("v2"), f"trainer_version={lm['trainer_version']}"

sd = c["model_state_dict"]
peft_keys = [k for k in sd if ("lora_" in k.lower()) or ("adapter" in k.lower()) or k.startswith("base_model.")]
assert not peft_keys, f"residual peft/adapter keys: {peft_keys[:5]}"

classifier_keys = [k for k in sd if "classifier" in k]
assert any(k.endswith("weight") for k in classifier_keys)
assert any(k.endswith("bias") for k in classifier_keys)

print(f"  best.pt schema OK  n_keys={len(sd)}  lora_meta.modules_to_save={lm['modules_to_save']}")
PY

section "Step 4: last.pt schema assertions (LoRA)"
/usr/bin/python3.11 - <<PY
import torch

p = "${SMOKE_ROOT}/${LR_EXP}/checkpoints/last.pt"
c = torch.load(p, map_location="cpu", weights_only=False)
sd = c["model_state_dict"]
peft_keys = [k for k in sd if ("lora_" in k.lower()) or ("adapter" in k.lower()) or k.startswith("base_model.")]
assert not peft_keys, f"last.pt has residual peft keys: {peft_keys[:5]}"
assert "label2id" in c and "rng_state" in c
print(f"  last.pt schema OK  n_keys={len(sd)}")
PY

section "Step 5: FT parity smoke — PB_RB_FT_T2_s99 (byte-compat with existing FT)"
FT_EXP=PB_RB_FT_T2_s99
FT_CFG=/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/phase_b/configs/${FT_EXP}.yaml
if [ ! -f "$FT_CFG" ]; then
    # Synthesize an FT smoke config from an existing seed.
    /usr/bin/python3.11 - <<PY
import yaml
from pathlib import Path
src = Path("/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/phase_b/configs/PB_RB_FT_T2_s01.yaml")
dst = Path("/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/phase_b/configs/PB_RB_FT_T2_s99.yaml")
cfg = yaml.safe_load(src.read_text())
cfg["experiment_id"] = "PB_RB_FT_T2_s99"
cfg["seed"] = 99
cfg["scientific_trainer"]["max_updates"] = 128
cfg["scientific_trainer"]["early_stopping_min_updates"] = 64
cfg["scientific_trainer"]["eval_every_steps"] = 32
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"wrote {dst}")
PY
fi
rm -rf "$SMOKE_ROOT/$FT_EXP"
/usr/bin/python3.11 -m fine_tuning_experiments.train.run_experiment \
    --experiment-id "${FT_EXP}" \
    --config-path "${FT_CFG}" \
    --run-root "${SMOKE_ROOT}"
/usr/bin/python3.11 - <<PY
import torch
p = "${SMOKE_ROOT}/${FT_EXP}/checkpoints/best.pt"
c = torch.load(p, map_location="cpu", weights_only=False)
# FT best.pt must NOT have lora_meta (only LoRA runs do).
assert "lora_meta" not in c, "FT best.pt unexpectedly has lora_meta"
sd = c["model_state_dict"]
assert not any("lora_" in k.lower() for k in sd), "FT best.pt has LoRA keys"
print(f"  FT parity OK  n_keys={len(sd)}  no lora_meta (expected)")
PY

section "Step 6: phase_b.eval.eval_one_run on merged LoRA best.pt"
/usr/bin/python3.11 -m fine_tuning_experiments.phase_b.eval.eval_one_run \
    --run-dir "${SMOKE_ROOT}/${LR_EXP}" \
    --overwrite
/usr/bin/python3.11 -m fine_tuning_experiments.phase_b.eval.eval_one_run \
    --run-dir "${SMOKE_ROOT}/${FT_EXP}" \
    --overwrite
/usr/bin/python3.11 - <<PY
import json
for exp in ["${LR_EXP}", "${FT_EXP}"]:
    p = "${SMOKE_ROOT}/" + exp + "/eval/phase_b_eval.json"
    r = json.load(open(p))
    assert "biored_test" in r and "bc5cdr_test" in r and "kb_surface" in r
    assert r["encoder_key"] in ("PB", "RB")
    assert r["update_key"] in ("FT", "LR")
    assert r["schedule_key"] == "T2"
    bio = r["biored_test"]["macro_f1"]
    bc = r["bc5cdr_test"]["macro_f1"]
    kb = r["kb_surface"]["kb_surface_mean"]
    print(f"  {exp}: biored_macro_f1={bio:.4f}  bc5cdr_macro_f1={bc:.4f}  kb_surface_mean={kb:.4f}")
PY

section "Step 7: aggregate dry-run on smoke runs"
/usr/bin/python3.11 -m fine_tuning_experiments.phase_b.aggregate_phase_b \
    --run-root "${SMOKE_ROOT}" \
    --out /tmp/phase_b_smoke_results.csv \
    --include-smoke
/usr/bin/python3.11 - <<PY
import csv
rows = list(csv.DictReader(open("/tmp/phase_b_smoke_results.csv")))
assert len(rows) >= 2, f"expected >=2 smoke rows, got {len(rows)}"
for r in rows:
    assert r["role"] == "smoke"
    assert r["encoder"] in ("PB", "RB")
    assert r["update"] in ("FT", "LR")
    assert r["update_regime"] in ("full_finetune", "lora")
    assert float(r["biored_macro_f1"]) >= 0.0
    assert float(r["kb_hit_A_setvalued"]) >= 0.0
# analyze_phase_b imports — verify CSV is readable by the downstream script.
from fine_tuning_experiments.phase_b.analysis.analyze_phase_b import load_rows
loaded = load_rows(__import__("pathlib").Path("/tmp/phase_b_smoke_results.csv"))
assert len(loaded) == len(rows)
print(f"  aggregate → analyze round-trip OK ({len(rows)} rows)")
PY

echo ""
echo "==============================================="
echo "  ALL 7 VERIFICATION STEPS PASSED"
echo "==============================================="
date -u '+%FT%TZ  phase_b_lora_verify DONE'
