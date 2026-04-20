"""
Utility paths for the fine-tuning experiments package.
Reconstructed from bytecode analysis (original source not available).
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FtDirs:
    """Standard directory structure under a fine-tuning data root."""
    root: Path
    runs: Path
    logs: Path
    configs: Path
    manifests: Path
    sbatch_out: Path

    def __post_init__(self) -> None:
        for attr in ("runs", "logs", "configs", "manifests", "sbatch_out"):
            getattr(self, attr).mkdir(parents=True, exist_ok=True)


def _default_ft_root() -> Path:
    """Compute the default ft_data_root from environment."""
    data_root = os.environ.get(
        "FT_DATA_ROOT",
        os.environ.get(
            "PROJECT_1_DATA_ROOT",
            str(Path.home() / "projects" / "project_1"),
        ),
    )
    return Path(data_root) / "fine_tuning_experiments"


def ensure_ft_dirs(ft_data_root=None) -> FtDirs:
    """Create and return standard subdirectory structure.

    Args:
        ft_data_root: path to the fine-tuning experiments output root.
                      Defaults to $PROJECT_1_DATA_ROOT/fine_tuning_experiments.
    """
    root = Path(ft_data_root) if ft_data_root is not None else _default_ft_root()
    root.mkdir(parents=True, exist_ok=True)
    return FtDirs(
        root=root,
        runs=root / "runs",
        logs=root / "logs",
        configs=root / "configs",
        manifests=root / "manifests",
        sbatch_out=root / "sbatch_out",
    )


# Module-level singleton — used by disk_space and other utils
ft_root = ensure_ft_dirs()
