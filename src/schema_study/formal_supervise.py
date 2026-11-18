"""Restart the formal controller if it stops before the matrix is finished.

When every manifest row is complete, score any remaining outputs and write
the formal reports. This process does not submit a second controller while
one is alive, and it does not start a new study.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/runs/formal/formal_20261115")
CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
PYTHON = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/environments/analysis/bin/python")
LOG = ROOT / "supervisor.log"


def _alive() -> bool:
    lock = ROOT / "launcher.pid"
    if not lock.is_file():
        return False
    pid = lock.read_text(encoding="utf-8").strip()
    return pid.isdigit() and Path(f"/proc/{pid}").exists()


def _remaining() -> bool:
    lines = (ROOT / "execution_manifest.tsv").read_text(encoding="utf-8").splitlines()
    index = lines[0].split("\t").index("status")
    return any(line.split("\t")[index] != "complete" for line in lines[1:])


def _start_launcher() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE / "src")
    env["PYTHONNOUSERSITE"] = "1"
    subprocess.Popen(
        [str(PYTHON), "-m", "schema_study.formal_launch"],
        cwd=str(CODE),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _run(module: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE / "src")
    env["PYTHONNOUSERSITE"] = "1"
    subprocess.run([str(PYTHON), "-m", module], cwd=str(CODE), env=env, check=True)


def main() -> None:
    while True:
        try:
            if _remaining():
                if not _alive():
                    LOG.write_text(LOG.read_text(encoding="utf-8") + "restart\n" if LOG.is_file() else "restart\n", encoding="utf-8")
                    _start_launcher()
            else:
                _run("schema_study.formal_analyze")
                _run("schema_study.formal_report")
                LOG.write_text((LOG.read_text(encoding="utf-8") if LOG.is_file() else "") + "reports_written\n", encoding="utf-8")
                return
        except Exception as exc:
            with LOG.open("a", encoding="utf-8") as handle:
                handle.write(f"error {type(exc).__name__}: {exc}\n")
            return
        time.sleep(120)


if __name__ == "__main__":
    main()
