"""Run all manuscript regeneration writers."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from .figures import REGENERATORS
from .paths import REPO, STEPS, VOCAB, step_paths
from .preflight import check_artifacts, check_style_module
from .provenance_checks import VERIFY
from .readmes import write_all_readmes
from .reports_step00_02 import write_report_00, write_report_01, write_report_02
from .reports_step03_05 import write_report_03, write_report_04, write_report_05
from .reports_step10_20 import write_report_10, write_report_20

REPORT_WRITERS = {
    "00": write_report_00,
    "01": write_report_01,
    "02": write_report_02,
    "03": write_report_03,
    "04": write_report_04,
    "05": write_report_05,
    "10": write_report_10,
    "20": write_report_20,
}

FIGURE_BUDGET = {
    "00": 1, "01": 2, "02": 1, "03": 2, "04": 1, "05": 1, "10": 2, "20": 3,
}
TABLE_BUDGET = {
    "00": 2, "01": 3, "02": 2, "03": 3, "04": 2, "05": 1, "10": 2, "20": 3,
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
    text = report_path.read_text(encoding="utf-8")
    n_tables = count_markdown_tables(text)
    fig_budget = FIGURE_BUDGET[step_key]
    tbl_budget = TABLE_BUDGET[step_key]
    print(f"\n--- Step {step_key} summary ---")
    print(f"  Figures kept: {len(figure_names)}/{fig_budget} budget -> {figure_names}")
    print(f"  Tables in report: {n_tables} (budget <= {tbl_budget})")
    if len(figure_names) > fig_budget:
        print(f"  WARNING: figure count exceeds budget")
    if n_tables > tbl_budget:
        print(f"  WARNING: table count exceeds budget")


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
        "20_round2_diagnostic/README.md",
    ]
    existing = [p for p in paths_to_add if (root / p).exists()]
    if not existing:
        print("No in-repo files to commit.")
        return
    subprocess.run(["git", "add"] + existing, cwd=root, check=True)
    msg = """Regenerate manuscript source reports, figures, and READMEs for steps 00-05, 10, 20.

Introduce shared Okabe-Ito plot style and provenance checks. Rewrite reports as detailed,
traceable source material with budget-limited figures. Skip step 06 OncoKB feasibility."""
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
    print(f"\nReports and figures written under {projects} (outside git repo; not committed).")


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

    for key in REPORT_WRITERS:
        report_path = step_paths(STEPS[key])["reports"] / "report.md"
        print_step_summary(key, figures.get(key, []), report_path)
        VERIFY[key]()

    print("\n=== Outputs written ===")
    for p in reports:
        print(f"  report: {p}")
    for p in readmes:
        print(f"  readme: {p}")

    commit_changes()


if __name__ == "__main__":
    main()
