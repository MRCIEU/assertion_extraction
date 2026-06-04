"""Run hyperparameter sweep and aggregate results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from importlib import import_module

from .benchmark_eval import add_benchmark_scores
from .config import (
    OUTPUT_DIR,
    SWEEP_MODEL_IDS,
    SWEEP_RESULTS_DIR,
    SweepRun,
    all_runs,
    runs_for_model,
)
from .report import write_sweep_report
from .train import train_sweep_run


def _load_train_val():
    td = import_module("10_round1_benchmark_kb.train_data")
    return td.build_train_val_examples(force=False)


def run_one(run: SweepRun, train_examples, val_examples, test_examples, force: bool = False) -> dict:
    payload = train_sweep_run(run, train_examples, val_examples, force=force)
    if "benchmark_f1_val_loss_ckpt" not in payload:
        payload = add_benchmark_scores(payload, test_examples)
        run.result_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_model(model_id: str, force: bool = False) -> None:
    be = import_module("10_round1_benchmark_kb.benchmark_eval")
    train_examples, val_examples = _load_train_val()
    test_examples = be.build_biored_test_examples()

    for run in runs_for_model(model_id):
        if run.result_path().exists() and not force:
            data = json.loads(run.result_path().read_text(encoding="utf-8"))
            if "benchmark_f1_val_loss_ckpt" not in data:
                data = add_benchmark_scores(data, test_examples)
                run.result_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"  skip (complete): {run.run_id}")
            continue
        payload = run_one(run, train_examples, val_examples, test_examples, force=force)


def aggregate_and_report() -> None:
    from .figures import generate_sweep_figures

    rows = []
    curve_rows = []
    for run in all_runs():
        path = run.result_path()
        if not path.exists():
            print(f"  missing: {run.run_id}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_id": data["run_id"],
                "model_id": data["model_id"],
                "short_name": data["short_name"],
                "lr": data["lr"],
                "warmup_label": data["warmup_label"],
                "best_epoch_by_val_loss": data["best_epoch_by_val_loss"],
                "best_epoch_by_val_f1": data["best_epoch_by_val_f1"],
                "best_val_loss": data["best_val_loss"],
                "best_val_f1": data["best_val_f1"],
                "val_f1_at_loss_epoch": data["val_f1_at_loss_epoch"],
                "selection_disagrees": data["selection_disagrees"],
                "benchmark_f1_val_loss_ckpt": data.get("benchmark_f1_val_loss_ckpt"),
                "benchmark_f1_val_f1_ckpt": data.get("benchmark_f1_val_f1_ckpt"),
                "epochs_run": data["epochs_run"],
            }
        )
        for ep in data.get("epoch_curve", []):
            curve_rows.append({**ep, "run_id": data["run_id"], "model_id": data["model_id"], "lr": data["lr"], "warmup_label": data["warmup_label"]})

    import pandas as pd

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    curves = pd.DataFrame(curve_rows)
    summary.to_csv(OUTPUT_DIR / "sweep_summary.csv", index=False)
    curves.to_csv(OUTPUT_DIR / "sweep_curves.csv", index=False)

    if summary.empty:
        spread = pd.DataFrame()
        spread.to_csv(OUTPUT_DIR / "sweep_benchmark_spread.csv", index=False)
        from .objective_analysis import run_objective_analysis
        results = run_objective_analysis()
        write_sweep_report(results)
        print("\n=== Sweep aggregate: no completed runs yet ===")
        return

    spread_rows = []
    for (lr, warmup_label), sub in summary.groupby(["lr", "warmup_label"]):
        bf1 = sub["benchmark_f1_val_loss_ckpt"].astype(float)
        spread_rows.append(
            {
                "lr": lr,
                "warmup_label": warmup_label,
                "benchmark_f1_min": float(bf1.min()),
                "benchmark_f1_max": float(bf1.max()),
                "benchmark_f1_spread": float(bf1.max() - bf1.min()),
                "benchmark_f1_mean": float(bf1.mean()),
                "mean_best_epoch_loss": float(sub["best_epoch_by_val_loss"].mean()),
                "mean_best_epoch_f1": float(sub["best_epoch_by_val_f1"].mean()),
                "n_disagreements": int(sub["selection_disagrees"].sum()),
            }
        )
    spread = pd.DataFrame(spread_rows).sort_values(["benchmark_f1_spread", "mean_best_epoch_loss"], ascending=[False, False])
    spread.to_csv(OUTPUT_DIR / "sweep_benchmark_spread.csv", index=False)

    from .objective_analysis import run_objective_analysis
    from .figures import generate_objective_figures

    results = run_objective_analysis()
    generate_sweep_figures(summary, curves, spread)
    generate_objective_figures(results)
    write_sweep_report(results)

    print("\n=== Sweep aggregate (legacy val_loss-only spread table) ===")
    if not spread.empty:
        best = spread.iloc[0]
        print(
            f"Widest spread (val_loss ckpt): lr={best['lr']}, warmup={best['warmup_label']} "
            f"spread={best['benchmark_f1_spread']:.3f}"
        )
    print(f"Tables -> {OUTPUT_DIR}")
    print(f"Report -> reports/10_round1_benchmark_kb/sweep_diagnostic.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 1 hyperparameter sweep (no KB)")
    parser.add_argument("--model", choices=SWEEP_MODEL_IDS, default=None)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.analyze_only:
        aggregate_and_report()
        return

    if not args.model:
        raise SystemExit("Specify --model or use --analyze-only")

    print(f"=== Sweep model {args.model} ===")
    run_model(args.model, force=args.force)
    print(f"=== Sweep model {args.model} complete ===")


if __name__ == "__main__":
    main()
