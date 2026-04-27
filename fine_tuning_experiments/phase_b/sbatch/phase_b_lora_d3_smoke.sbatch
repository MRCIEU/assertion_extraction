#!/bin/bash -l
#SBATCH -J PB_LR_D3
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/PB_LR_D3_smoke_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/PB_LR_D3_smoke_%j.err

# Phase B — Amendment B.9 D3 budget probe.
# Single LoRA smoke at the *pre-lock* learning rate (2.0e-05) with
# max_updates doubled from 2,048 to 4,096.  All other LoRA hyperparameters
# are unchanged from the locked pre-registration:
#   r=16, alpha=32, dropout=0.05, target=Q/V, modules_to_save=classifier,
#   bias=none, batch=4 (× 4 negatives = 16 effective), max_length=384,
#   AdamW, linear schedule from 2e-5 to 0, eval_every=64,
#   early_stopping_patience=10, early_stopping_min_updates=256,
#   selection_metric=macro_f1.
#
# This is the *single falsifier* required by amendment B.9(e):
#   - If escapes  (dev_f1 > 0.20 by some step ≤ 1024 AND
#                 dev_f1 > 0.30 at step 4096 AND
#                 ≥ 1 dev_f1_excluding_negative > 0):
#     → Submit the 180 LoRA bulk retrain at LR=2e-5 max_updates=4096.
#   - Else: drop LoRA arm (D1), record in B.10.
#
# Re-uses experiment_id PB_PB_LR_T1B_s99; the prior LR=3e-4 incarnation
# of this ID has been moved to
#   runs/phase_b_degenerate_lr_archive/lr3e4_postB8_20260427T101524Z/
# per amendment B.9(g).  The run-root is empty for this ID before
# this job starts (verified manually).
#
# Safety:
#   - set -euo pipefail
#   - defensive check that config has LR=2.0e-05 and max_updates=4096
#   - non-degeneracy check is the gating step (not eval); if the model
#     collapsed, we do not run the (pointless) Phase B eval.
#   - exit codes: 0 (ESCAPED), 14 (still degenerate, falsifies D3),
#                  1 (config error / wiring failure), 2 (eval missing).

set -euo pipefail
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
export HF_HOME=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
export TRANSFORMERS_CACHE=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
cd /home/b5ac/freddieyu.b5ac/project_1

EXP_ID=PB_PB_LR_T1B_s99
CFG=/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/phase_b/configs/${EXP_ID}.yaml
RUN_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b
RUN_DIR=${RUN_ROOT}/${EXP_ID}

# ---- preflight ----
echo "[$(date -u +%FT%TZ)] D3 smoke preflight"
echo "  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

if [[ ! -f "$CFG" ]]; then
    echo "ERROR: config not found: $CFG" >&2
    exit 1
fi

cfg_lr=$(grep -E '^[[:space:]]+learning_rate:' "$CFG" | head -1 | awk '{print $2}')
cfg_mu=$(grep -E '^[[:space:]]+max_updates:'   "$CFG" | head -1 | awk '{print $2}')
if [[ "$cfg_lr" != "2.0e-05" ]]; then
    echo "ERROR: D3 requires LR=2.0e-05, config has LR=${cfg_lr}" >&2
    exit 1
fi
if [[ "$cfg_mu" != "4096" ]]; then
    echo "ERROR: D3 requires max_updates=4096, config has ${cfg_mu}" >&2
    exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
    echo "ERROR: run-dir already exists: $RUN_DIR" >&2
    echo "       (D3 expects a clean slate; archive any prior incarnation first)" >&2
    exit 1
fi

echo "  config:        $CFG"
echo "  LR:            $cfg_lr"
echo "  max_updates:   $cfg_mu"
echo "  run_dir:       $RUN_DIR"
echo "  trainer commit: $(cd /home/b5ac/freddieyu.b5ac/project_1 && git rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
echo ""

# ---- [1/3] train ----
echo "=== [1/3] Train LoRA (pre-lock LR=2e-5) for 4,096 steps ==="
/usr/bin/python3.11 -m fine_tuning_experiments.phase_b.trainer.run_experiment \
    --experiment-id "${EXP_ID}" \
    --config-path "${CFG}" \
    --run-root "${RUN_ROOT}"

if [[ ! -f "$RUN_DIR/checkpoints/best.pt" ]]; then
    echo "ERROR: train completed but best.pt missing" >&2
    exit 2
fi

# ---- [2/3] non-degeneracy check (THE FALSIFIER) ----
echo ""
echo "=== [2/3] D3 non-degeneracy check ==="
/usr/bin/python3.11 - <<'PY'
import json, sys
from pathlib import Path

RUN = Path("/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b/PB_PB_LR_T1B_s99")
vh = RUN / "metrics" / "validation_history.json"
if not vh.exists():
    print(f"ERROR: validation_history.json missing at {vh}", file=sys.stderr); sys.exit(2)

h = json.loads(vh.read_text())
if isinstance(h, dict): h = h.get("entries", [])
if not h:
    print("ERROR: validation history empty", file=sys.stderr); sys.exit(2)

print(f"  entries: {len(h)}")

# Print every entry compactly
for e in h:
    s   = e.get("step")
    f1  = e.get("dev_macro_f1") or e.get("macro_f1") or 0.0
    fen = e.get("dev_macro_f1_excluding_negative")
    fen_s = f"{fen:.4f}" if isinstance(fen, (int, float)) else "None"
    print(f"    step={s:>5}  dev_f1={f1:.6f}  excl_neg={fen_s}")

f1s = [(e.get("step"), e.get("dev_macro_f1") or e.get("macro_f1") or 0.0) for e in h]
fens = [e.get("dev_macro_f1_excluding_negative") for e in h
        if isinstance(e.get("dev_macro_f1_excluding_negative"), (int, float))]
final_step, final_f1 = f1s[-1]

# D3 decision criteria (B.9(e)):
#   (i)  dev_f1 > 0.20 by some step <= 1024
#   (ii) dev_f1 > 0.30 at the final step (4096)
#   (iii) at least one positive macro_f1_excluding_negative
crit_i  = any(s <= 1024 and f > 0.20 for s, f in f1s)
crit_ii = final_f1 > 0.30
crit_iii = any((v is not None) and (v > 0.0) for v in fens)

print()
print(f"  crit (i)  dev_f1 > 0.20 by step ≤ 1024:        {'PASS' if crit_i  else 'FAIL'}")
print(f"  crit (ii) dev_f1 > 0.30 at final step ({final_step}): {'PASS' if crit_ii else 'FAIL'}  (final={final_f1:.4f})")
print(f"  crit (iii) any dev_f1_excluding_negative > 0:  {'PASS' if crit_iii else 'FAIL'}")

# Also check bit-identicality of dev_f1 (the original collapse signature)
unique_f1 = len({round(f, 12) for _, f in f1s})
spread = max(f for _, f in f1s) - min(f for _, f in f1s)
print(f"  diagnostic: unique dev_f1 values = {unique_f1};  spread = {spread:.6e}")

# Verdict
escaped = crit_i and crit_ii and crit_iii
if escaped:
    print()
    print("  D3 VERDICT: ESCAPED  --  proceed to bulk retrain (180 LoRA + 2 PL_FT)")
    sys.exit(0)
else:
    print()
    print("  D3 VERDICT: STILL DEGENERATE  --  falsifies budget hypothesis; drop LoRA (D1)")
    sys.exit(14)
PY
status=$?

if [[ $status -eq 14 ]]; then
    echo ""
    echo "[$(date -u +%FT%TZ)] D3 SMOKE FAILED (still degenerate)."
    echo "  Skipping eval (no scientific value on a degenerate run)."
    exit 14
elif [[ $status -ne 0 ]]; then
    echo "[$(date -u +%FT%TZ)] D3 smoke check raised an unexpected error (code $status)" >&2
    exit $status
fi

# ---- [3/3] eval (only on escape) ----
echo ""
echo "=== [3/3] Phase B eval pipeline ==="
/usr/bin/python3.11 -m fine_tuning_experiments.phase_b.eval.eval_one_run \
    --run-dir "${RUN_DIR}" --overwrite

if [[ ! -s "$RUN_DIR/eval/phase_b_eval.json" ]]; then
    echo "ERROR: eval file missing or empty after run" >&2
    exit 2
fi

echo ""
echo "[$(date -u +%FT%TZ)] D3 SMOKE PASSED — bulk retrain is approved."
echo "  Next: scripts/update_lora_max_updates_to_4096.py + phase_b_LR_retrain_d3.sbatch"
