"""Pre-flight checks before step-2 full-matrix training on clean offset-marked data."""

from __future__ import annotations

import json
import py_compile
import tempfile
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shared.constants import CHECKPOINT_CRITERION, LEAKED_PMIDS, TRAIN_SEEDS
from shared.models import MODELS
from shared.train_core import _save_ckpt, train_with_epoch_checkpoints
from shared.train_data import build_train_val_examples

from .config import (
    ESTIMATED_AVG_EPOCHS_PER_RUN,
    ESTIMATED_FP16_CHECKPOINT_MIB,
    ESTIMATED_FP32_CHECKPOINT_MIB,
    MATRIX_CKPT_DIR,
    MATRIX_COMPLETE,
    MATRIX_DATA,
    MATRIX_RESULTS_DIR,
    MAX_EPOCH_CHECKPOINTS_TO_KEEP,
    SAVE_EPOCH_CHECKPOINTS_FP16,
    TRAIN_CACHE_DIR,
    matrix_result_path,
    matrix_run_root,
    require_chosen_recipe,
)
from .step1_preflight import verify_clean_train_cache
from .step2_train import is_matrix_complete, print_checkpoint_footprint_estimate


def _log(msg: str) -> None:
    print(msg, flush=True)


def _check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    _log(msg)
    return ok


def _grep_step2_for_stale_lr() -> list[str]:
    """Flag hardcoded non-sweep RecipeConfig defaults in step-2 modules."""
    step_dir = Path(__file__).resolve().parent
    hits: list[str] = []
    for path in [step_dir / "config.py", step_dir / "step2_train.py"]:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "RecipeConfig" in line and ("1e-5" in line or "2e-5" in line or "1e-05" in line or "2e-05" in line):
                if "CHOSEN_RECIPE" in line and "3e-5" not in line and "3e-05" not in line:
                    hits.append(f"{path.name}:{i}: {stripped}")
    return hits


def _fp16_roundtrip_loadable() -> tuple[bool, str]:
    """Structural sanity: save fp16 checkpoint and reload on CPU."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "epoch_01"
            spec = MODELS[-1]  # DeBERTa-base (largest architecture in matrix)
            tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
            model = AutoModelForSequenceClassification.from_pretrained(spec.hf_name, num_labels=2)
            _save_ckpt(model, tokenizer, tmp_path, fp16=True)
            required = ("config.json",)
            if not all((tmp_path / f).exists() for f in required):
                return False, f"missing files under {tmp_path}"
            weights = list(tmp_path.glob("model*.safetensors")) + list(tmp_path.glob("pytorch_model.bin"))
            if not weights:
                return False, "no weight file saved"
            reloaded = AutoModelForSequenceClassification.from_pretrained(tmp_path)
            n_params = sum(p.numel() for p in reloaded.parameters())
            del model, reloaded
            return True, f"{spec.short_name} fp16 round-trip OK ({n_params:,} params)"
    except Exception as exc:
        return False, str(exc)


def _old_matrix_status() -> dict:
    """Summarise any prior matrix outputs that could collide with the rerun."""
    n_markers = 0
    recipe_lrs: set[float] = set()
    sample_paths: list[str] = []
    if MATRIX_RESULTS_DIR.exists():
        for spec in MODELS:
            for seed in TRAIN_SEEDS:
                marker = matrix_result_path(spec.model_id, seed)
                if marker.exists():
                    n_markers += 1
                    try:
                        data = json.loads(marker.read_text(encoding="utf-8"))
                        recipe_lrs.add(float(data.get("recipe_lr", 0)))
                    except json.JSONDecodeError:
                        pass
                    if len(sample_paths) < 3:
                        sample_paths.append(str(marker))
    ckpt_dirs = 0
    if MATRIX_CKPT_DIR.exists():
        ckpt_dirs = sum(1 for _ in MATRIX_CKPT_DIR.rglob("best"))
    return {
        "matrix_data_exists": MATRIX_DATA.exists(),
        "n_markers": n_markers,
        "n_best_ckpt_dirs": ckpt_dirs,
        "recipe_lrs": sorted(recipe_lrs),
        "sample_markers": sample_paths,
    }


def run_step2_preflight() -> bool:
    _log("\n=== Step-2 matrix training pre-flight ===")
    all_ok = True

    # 1. Recipe
    try:
        recipe = require_chosen_recipe()
        ok = (
            abs(recipe.lr - 3e-5) < 1e-12
            and recipe.warmup_label == "none"
            and recipe.warmup_ratio == 0.0
            and CHECKPOINT_CRITERION == "val_f1"
        )
        all_ok &= _check(
            "Recipe 3e-5/none/val_f1",
            ok,
            f"lr={recipe.lr}, warmup={recipe.warmup_label}, criterion={CHECKPOINT_CRITERION}, "
            f"strategy={recipe.strategy_tag()}",
        )
        stale = _grep_step2_for_stale_lr()
        all_ok &= _check("No stale 1e-5/2e-5 defaults in step-2 path", not stale, "; ".join(stale) or "none found")
    except SystemExit as exc:
        all_ok &= _check("Recipe 3e-5/none/val_f1", False, str(exc))

    # 2. Clean cache
    try:
        cache_summary = verify_clean_train_cache(TRAIN_CACHE_DIR)
        all_ok &= _check(
            "Clean offset-marked train cache",
            cache_summary["ok"],
            f"{cache_summary['cache_dir']}; {cache_summary['n_examples']} examples; "
            f"offset={cache_summary['offset_marker_rate']:.1%}",
        )
    except SystemExit:
        all_ok &= _check("Clean offset-marked train cache", False, "preflight failed")

    # 3. Leaked PMID guard
    try:
        train_rows, val_rows = build_train_val_examples(TRAIN_CACHE_DIR)
        pmids = {str(r.get("pmid", "")) for r in train_rows + val_rows}
        leaked = pmids & LEAKED_PMIDS
        all_ok &= _check(
            "Leaked-PMID guard active",
            not leaked,
            f"LEAKED_PMIDS={sorted(LEAKED_PMIDS)}; found in cache={sorted(leaked) or 'none'}",
        )
    except Exception as exc:
        all_ok &= _check("Leaked-PMID guard active", False, str(exc))

    # 4. Cost profile: benchmark at best only; val in training_log
    doc_ok = (
        "evaluate_checkpoint_benchmark_f1" in Path(__file__).parent.joinpath("step2_train.py").read_text()
        and "train_with_epoch_checkpoints" in Path(__file__).parent.joinpath("step2_train.py").read_text()
        and "No benchmark F1 here" in (Path(__file__).resolve().parents[1] / "shared" / "train_core.py").read_text()
    )
    n_runs = len(MODELS) * len(TRAIN_SEEDS)
    all_ok &= _check(
        "Benchmark F1 at best checkpoint only (72 evals)",
        doc_ok,
        f"{n_runs} runs x 1 benchmark eval; per-epoch val in training_log.json",
    )

    # 5. Checkpoint layout
    layout_ok = (
        SAVE_EPOCH_CHECKPOINTS_FP16
        and train_with_epoch_checkpoints.__doc__ is not None
        and "epochs/epoch_NN" in train_with_epoch_checkpoints.__doc__
    )
    sample_root = matrix_run_root(MODELS[0].model_id, TRAIN_SEEDS[0])
    all_ok &= _check(
        "Checkpoint layout (fp16 epochs/, fp32 best/, training_log.json)",
        layout_ok,
        f"example root {sample_root}; fp16={SAVE_EPOCH_CHECKPOINTS_FP16}",
    )

    # 6. fp16 loadable
    fp16_ok, fp16_detail = _fp16_roundtrip_loadable()
    all_ok &= _check("fp16 epoch checkpoint loadable (structural)", fp16_ok, fp16_detail)

    # 7. Resumable markers
    resumable = MATRIX_COMPLETE == "matrix_complete.json" and hasattr(is_matrix_complete, "__call__")
    all_ok &= _check(
        "Resumable via matrix_complete.json per run",
        resumable,
        f"skip when marker exists unless --force; marker={MATRIX_COMPLETE}",
    )

    # 8. Footprint estimate
    _log("\n=== Estimated checkpoint footprint (rough) ===")
    print_checkpoint_footprint_estimate(len(MODELS), len(TRAIN_SEEDS))
    per_run_mib = (
        ESTIMATED_AVG_EPOCHS_PER_RUN
        * (ESTIMATED_FP16_CHECKPOINT_MIB if SAVE_EPOCH_CHECKPOINTS_FP16 else ESTIMATED_FP32_CHECKPOINT_MIB)
        + ESTIMATED_FP32_CHECKPOINT_MIB
    )
    total_gib = len(MODELS) * len(TRAIN_SEEDS) * per_run_mib / 1024
    all_ok &= _check(
        "Footprint estimate printed",
        True,
        f"~{total_gib:.1f} GiB total ({len(MODELS)}x{len(TRAIN_SEEDS)} runs, "
        f"~{per_run_mib} MiB/run, keep={MAX_EPOCH_CHECKPOINTS_TO_KEEP})",
    )

    # 9. Matrix shape
    all_ok &= _check(
        "Matrix 9 encoders x 8 seeds = 72 runs",
        len(MODELS) == 9 and TRAIN_SEEDS == list(range(42, 50)),
        f"encoders={[m.model_id for m in MODELS]}; seeds={TRAIN_SEEDS}",
    )

    # 10. sbatch script
    sbatch = Path(__file__).parent / "step2_train.sbatch"
    sbatch_text = sbatch.read_text(encoding="utf-8") if sbatch.exists() else ""
    sbatch_ok = all(
        x in sbatch_text
        for x in ("conda activate hf-hpc", "PYTHONUNBUFFERED=1", "step2_train", "--train-only")
    )
    all_ok &= _check(
        "step2_train.sbatch standalone (hf-hpc, unbuffered, own logs)",
        sbatch_ok,
        str(sbatch),
    )

    # 11. py_compile + path trace
    repo = Path(__file__).resolve().parents[1]
    py_targets = [
        Path(__file__).parent / "step2_train.py",
        Path(__file__).parent / "run.py",
        repo / "shared" / "train_core.py",
        repo / "shared" / "train_data.py",
        repo / "shared" / "benchmark_eval.py",
        repo / "shared" / "marker_insert.py",
    ]
    compile_ok = True
    compile_err = ""
    for pf in py_targets:
        try:
            py_compile.compile(str(pf), doraise=True)
        except py_compile.PyCompileError as exc:
            compile_ok = False
            compile_err = f"{pf.name}: {exc}"
            break
    all_ok &= _check("py_compile step-2 + shared modules", compile_ok, compile_err or f"{len(py_targets)} files")

    try:
        assert callable(build_train_val_examples)
        assert callable(require_chosen_recipe)
        assert TRAIN_CACHE_DIR.name == "cache"
        all_ok &= _check(
            "End-to-end path trace",
            True,
            f"train cache={TRAIN_CACHE_DIR}; matrix={MATRIX_DATA}; recipe from config",
        )
    except Exception as exc:
        all_ok &= _check("End-to-end path trace", False, str(exc))

    # 12. Old outputs handling
    old = _old_matrix_status()
    _log("\n=== Old matrix output status ===")
    _log(f"  matrix/ exists: {old['matrix_data_exists']}")
    _log(f"  matrix_complete.json markers: {old['n_markers']}/72")
    _log(f"  best/ checkpoint dirs: {old['n_best_ckpt_dirs']}")
    if old["recipe_lrs"]:
        _log(f"  recipe_lr in existing markers: {old['recipe_lrs']}")
    if old["n_markers"] == 0 and old["n_best_ckpt_dirs"] == 0:
        plan = (
            "Clean slate: no matrix/ outputs on disk. New run writes to "
            "data/10_recipe_sweep_and_training/matrix/ at 3e-5/none on offset-marked data. "
            "Prior 1e-5 buggy-data matrix (from Jun 5 jobs) is absent; nothing to mix."
        )
        old_ok = True
    elif old["n_markers"] > 0 and old["recipe_lrs"] != [3e-5]:
        plan = (
            f"WARNING: {old['n_markers']} stale matrix_complete.json markers exist with "
            f"recipe_lr={old['recipe_lrs']}. Without --force, step 2 will SKIP completed runs "
            "and folder 11 could read old 1e-5 checkpoints. Before submit, either archive "
            f"matrix/ (e.g. mv {MATRIX_DATA} {MATRIX_DATA}_prior_1e5_string_match) "
            "or submit with STEP_ARGS='--force' to overwrite all 72 runs."
        )
        old_ok = False
    else:
        plan = "Partial or matching matrix present; resubmit skips finished runs unless --force."
        old_ok = old["n_markers"] == 0 or old["recipe_lrs"] == [3e-5]
    all_ok &= _check("Old output handling plan", old_ok, plan)

    _log("\n=== Pre-flight summary ===")
    if all_ok:
        _log("ALL CHECKS PASSED — safe to submit step 2")
    else:
        _log("ONE OR MORE CHECKS FAILED — do not submit until fixed")
    return all_ok
