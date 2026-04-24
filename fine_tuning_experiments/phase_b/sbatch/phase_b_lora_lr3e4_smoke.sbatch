#!/bin/bash -l
#SBATCH -J PB_LR3e4_smk
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/PB_LR3e4_smoke_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/PB_LR3e4_smoke_%j.err

# Amendment B.8 smoke test: verify that LoRA (r=16, alpha=32, Q/V, classifier mts)
# at learning_rate=3.0e-4 escapes the majority-class basin that collapsed at LR=2e-5.
#
# Success criterion: dev_macro_f1 must exceed 0.20 at some step <= 512 AND show
# non-trivial variation across eval checkpoints (not bit-identical like LR=2e-5).

set -euo pipefail
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
export HF_HOME=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
export TRANSFORMERS_CACHE=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
cd /home/b5ac/freddieyu.b5ac/project_1

EXP_ID=PB_PB_LR_T1B_s99
CFG=fine_tuning_experiments/phase_b/configs/${EXP_ID}.yaml
RUN_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b
RUN_DIR=${RUN_ROOT}/${EXP_ID}

echo "[$(date -u +%FT%TZ)] host=$(hostname)"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader || true

# Remove any prior smoke artifacts (so this is clean)
if [ -d "$RUN_DIR" ]; then
    echo "  removing prior smoke artifacts at $RUN_DIR"
    rm -rf "$RUN_DIR"
fi

echo ""
echo "=== [1/3] Train at LR=3.0e-04 ==="
/usr/bin/python3.11 -m fine_tuning_experiments.phase_b.trainer.run_experiment \
    --experiment-id "${EXP_ID}" \
    --config-path "${CFG}" \
    --run-root "${RUN_ROOT}"

if [ ! -f "$RUN_DIR/checkpoints/best.pt" ]; then
    echo "ERROR: training produced no best.pt" >&2
    exit 10
fi

echo ""
echo "=== [2/3] Non-degeneracy check (dev F1 history) ==="
/usr/bin/python3.11 - <<'PY'
import json, sys
from pathlib import Path
rundir = Path("/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b/PB_PB_LR_T1B_s99")
vh = rundir / "metrics" / "validation_history.json"
if not vh.exists():
    print("  validation_history.json missing", file=sys.stderr); sys.exit(11)
hist = json.loads(vh.read_text())
if not isinstance(hist, list) or not hist: hist = hist.get("entries", [])
print(f"  eval entries: {len(hist)}")
f1s = []
for i, e in enumerate(hist):
    step = e.get("step") or e.get("global_step")
    f1   = e.get("dev_macro_f1") or e.get("macro_f1")
    f1e  = e.get("dev_macro_f1_excluding_negative") or e.get("macro_f1_excluding_negative")
    f1s.append((step, f1, f1e))
    if i < 16 or i >= len(hist)-4:
        print(f"    step={step:>4}  dev_f1={f1}  dev_f1_excl_neg={f1e}")

# Criterion 1: at least one eval at step<=512 with dev_f1 > 0.20
early = [f1 for (s, f1, _) in f1s if s is not None and s <= 512 and f1 is not None]
if not early:
    print("  no evals <= step 512", file=sys.stderr); sys.exit(12)
max_early = max(early)
print(f"\n  max dev_f1 in first 512 steps: {max_early:.4f}")

# Criterion 2: variation across first 8 eval points
first8 = [f1 for (s,f1,_) in f1s[:8] if f1 is not None]
if len(first8) >= 2:
    unique = len(set(first8))
    spread = max(first8) - min(first8)
    print(f"  first-8 f1 values: {first8}")
    print(f"    unique={unique}, spread={spread:.4e}")
else:
    print("  not enough early evals"); sys.exit(13)

# Criterion 3: macro_f1_excluding_negative > 0 at some point (i.e., at least one
# non-negative class recovers non-zero F1 at any eval point) — stronger check
any_nonzero_excl_neg = any(
    f1e is not None and f1e > 0.0 for (_, _, f1e) in f1s
)

status = []
if max_early > 0.20: status.append("MAX_F1_EARLY_PASS")
else:                status.append(f"MAX_F1_EARLY_FAIL({max_early:.3f})")
if unique > 1 and spread > 1e-4: status.append("VARIATION_PASS")
else:                            status.append(f"VARIATION_FAIL(unique={unique},spread={spread:.2e})")
if any_nonzero_excl_neg: status.append("EXCL_NEG_POSITIVE_PASS")
else:                    status.append("EXCL_NEG_POSITIVE_FAIL(still_degenerate)")

print("\n  STATUS: " + " | ".join(status))

fail = [s for s in status if "FAIL" in s]
if fail:
    print("\nSMOKE FAILED", file=sys.stderr); sys.exit(14)
print("\nSMOKE PASSED")
PY

echo ""
echo "=== [3/3] Full evaluation (phase_b_eval.json) ==="
/usr/bin/python3.11 -m fine_tuning_experiments.phase_b.eval.eval_one_run \
    --run-dir "$RUN_DIR" --overwrite

/usr/bin/python3.11 - <<'PY'
import json
from pathlib import Path
p = Path("/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b/PB_PB_LR_T1B_s99/eval/phase_b_eval.json")
d = json.loads(p.read_text())
for split in ("biored_test", "bc5cdr_test"):
    if split in d:
        s = d[split]
        print(f"  {split}: macro_f1={s.get('macro_f1'):.4f}  macro_f1_excluding_negative={s.get('macro_f1_excluding_negative'):.4f}")
        for lab, v in sorted(s.get("per_label",{}).items()):
            print(f"     {lab:>22}  f1={v.get('f1',0):.4f}")
PY

echo ""
echo "[$(date -u +%FT%TZ)] LoRA LR=3e-4 smoke test COMPLETE"
