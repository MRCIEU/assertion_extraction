"""
Disk space utility for fine-tuning experiments.
Reconstructed from compiled bytecode analysis.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
from fine_tuning_experiments.utils.paths import ensure_ft_dirs, ft_root


def record_df_for_ft_root(run_dir: Path | None = None) -> None:
    """Run df -h on the fine-tuning data root and write result to manifests/."""
    target = run_dir or ft_root.root
    try:
        result = subprocess.run(
            ["df", "-h", str(target)],
            capture_output=True, text=True, timeout=10
        )
        out_dir = ft_root.manifests
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "disk_space_df.txt"
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(f"# df -h {target}\n")
            f.write(result.stdout)
            f.write(f"\n# exit_code={result.returncode}\n")
            f.write("\n")
    except Exception:
        pass  # Non-critical; do not fail training for disk space recording
