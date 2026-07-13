"""Run all manuscript regeneration writers."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from .figures import REGENERATORS
from .paths import OUTPUT_ROOT, REPO, STEPS, VOCAB, step_paths
from .preflight import check_artifacts, check_style_module
from .provenance_checks import VERIFY
from .readmes import write_all_readmes
from .reports_step00_02 import write_report_00, write_report_01, write_report_02
from .reports_step03_05 import write_report_03, write_report_04, write_report_05
from .reports_step10_20 import write_report_10, write_report_20
from .reports_step11 import write_report_11

REPORT_WRITERS = {
    "00": write_report_00,
    "01": write_report_01,
    "02": write_report_02,
    "03": write_report_03,
    "04": write_report_04,
    "05": write_report_05,
    "10": write_report_10,
    "11": write_report_11,
    "20": write_report_20,
}

FIGURE_BUDGET = {
    "00": 1, "01": 2, "02": 1, "03": 2, "04": 1, "05": 1, "10": 2, "11": 4, "20": 10,
}
TABLE_BUDGET = {
    "00": 2, "01": 3, "02": 2, "03": 3, "04": 2, "05": 1, "10": 2, "11": 0, "20": 1,
}


def count_markdown_tables(text: str) -> int:
    return len(re.findall(r"^\|[^\n]+\|\s*\n\|[-:\s|]+\|", text, flags=re.MULTILINE))


def regenerate_reports(steps: list[str] | None = None) -> list[str]:
    keys = steps or list(REPORT_WRITERS.keys())
    written: list[str] = []
    for key in keys:
        path = REPORT_WRITERS[key](step_paths(STEPS[key]))
        written.append(str(path))
    return written


def regenerate_figures(steps: list[str] | None = None) -> dict[str, list[str]]:
    keys = steps or list(REGENERATORS.keys())
    out: dict[str, list[str]] = {}
    for key in keys:
        out[key] = REGENERATORS[key]()
    return out


def print_step_summary(step_key: str, figure_names: list[str], report_path: Path) -> None:
    if step_key in VERIFY:
        VERIFY[step_key]()
    text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    n_tables = count_markdown_tables(text)
    fig_budget = FIGURE_BUDGET.get(step_key, 99)
    tbl_budget = TABLE_BUDGET.get(step_key, 99)
    print(f"\n--- Step {step_key} summary ---")
    print(f"  Figures regenerated: {len(figure_names)} (budget {fig_budget})")
    print(f"  Figure names: {figure_names}")
    print(f"  Tables in report: {n_tables} (budget <= {tbl_budget})")
    if len(figure_names) > fig_budget:
        print("  WARNING: figure count exceeds budget")


def commit_changes() -> None:
    root = REPO
    paths_to_add = [
        "shared/plot_style.py",
        "shared/provenance.py",
        "manuscript_regenerate/",
        "regenerate_manuscript_sources.py",
        "00_civic_feasibility/README.md",
        "01_corpus_relevance/README.md",
        "02_evaluation_protocol/README.md",
        "03_candidate_pool/README.md",
        "04_pilot_study/README.md",
        "05_marker_quality_gate/README.md",
        "10_recipe_sweep_and_training/README.md",
        "11_round1_analysis/README.md",
        "20_round2_diagnostic/README.md",
    ]
    existing = [p for p in paths_to_add if (root / p).exists()]
    if not existing:
        print("No in-repo files to commit.")
        return
    subprocess.run(["git", "add"] + existing, cwd=root, check=True)
    msg = """Fix shared figure style, rewrite reports 11 and 20, and minimal READMEs.

Apply Okabe-Ito role colours at 300 dpi across steps 00-05, 10, 11, 20. Fix step-11
label overlap and variance-bar ambiguity; resolve duplicate figure numbers. Skip step 06."""
    result = subprocess.run(["git", "commit", "-m", msg], cwd=root, capture_output=True, text=True)
    if result.returncode != 0 and "nothing to commit" in (result.stdout + result.stderr):
        print("Nothing new to commit.")
    elif result.returncode != 0:
        print(result.stderr)
        result.check_returncode()
    else:
        print("\nCommit complete. Push when ready:")
        print("  git push origin main")
    projects = root.parent / "projects" / "project_1"
    print(f"\nReports and figures written under {projects} (outside git repo).")


def regenerate_step20_report() -> None:
    """Refresh step-20 report from report.py (CPU only, existing scores)."""
    import os
    import subprocess
    import sys

    step_dir = REPO / "20_round2_diagnostic"
    env = os.environ.copy()
    env["OUTPUT_ROOT"] = str(OUTPUT_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    print("\n=== Step 20 report assembly (CPU, existing artifacts) ===")
    subprocess.run(
        [sys.executable, "run.py", "--analyze-only", "--skip-stratum-inference"],
        cwd=step_dir,
        env=env,
        check=True,
    )


def main() -> None:
    print("=== Manuscript source regeneration ===")
    print(f"Shared vocabulary: {VOCAB['benchmark']}; {VOCAB['kb']}; {VOCAB['question']}")

    if not check_style_module():
        sys.exit(1)
    if not check_artifacts():
        sys.exit(1)

    print("\n=== Regenerating figures ===")
    figures = regenerate_figures()

    print("\n=== Regenerating reports ===")
    reports = regenerate_reports()

    print("\n=== Regenerating READMEs ===")
    readmes = write_all_readmes()

    regenerate_step20_report()

    for key in sorted(set(list(REPORT_WRITERS.keys()) + list(REGENERATORS.keys()))):
        report_path = step_paths(STEPS[key])["reports"] / "report.md" if key in STEPS else Path()
        if key in STEPS:
            print_step_summary(key, figures.get(key, []), report_path)

    print("\n=== Outputs written ===")
    for p in reports:
        print(f"  report: {p}")
    for p in readmes:
        print(f"  readme: {p}")

    if os.environ.get("SKIP_GIT_COMMIT", "") != "1":
        commit_changes()


if __name__ == "__main__":
    main()
