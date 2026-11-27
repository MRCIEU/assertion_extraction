"""Code and storage locations.

Study outputs go under the external storage root. The code root stays limited
to source, configuration, prompts, and tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

CODE_ROOT = Path(__file__).resolve().parents[2]
STORAGE_ENV = "SCHEMA_STUDY_STORAGE_ROOT"
DEFAULT_STORAGE = Path("/home/b5ac/freddieyu.b5ac/projects/project_1")
STUDY_ID = "schema_granularity_v1"


def overlaps(storage: Path, code: Path) -> bool:
    """A snapshot stored under the project storage is code, not a data collision."""
    if storage == code or storage.is_relative_to(code):
        return True
    if not code.is_relative_to(storage):
        return False
    relative = code.relative_to(storage)
    return not (len(relative.parts) >= 2 and relative.parts[0] == "execution_snapshots")


@dataclass(frozen=True)
class ProjectPaths:
    code_root: Path
    storage_root: Path

    def assert_output(self, path: Path) -> Path:
        """Reject a destination that resolves into the code checkout."""
        target = Path(path).resolve()
        if overlaps(target, self.code_root.resolve()):
            raise ValueError(f"Refusing to write study output inside the code root: {target}")
        return target

    @property
    def raw(self) -> Path:
        return self.storage_root / "data" / "raw"

    @property
    def prepared(self) -> Path:
        return self.storage_root / "data" / "prepared"

    @property
    def inputs(self) -> Path:
        return self.storage_root / "data" / "inputs"

    @property
    def study(self) -> Path:
        return self.storage_root / "studies" / STUDY_ID

    @property
    def assets_tokenizers(self) -> Path:
        return self.storage_root / "assets" / "tokenizers"

    @property
    def environments(self) -> Path:
        return self.storage_root / "environments"

    @property
    def job_logs(self) -> Path:
        return self.storage_root / "job_logs"

    @property
    def cache(self) -> Path:
        return self.storage_root / "cache"

    @property
    def tmp(self) -> Path:
        return self.storage_root / "tmp"


def load_settings(code_root: Path | None = None) -> dict:
    root = code_root or CODE_ROOT
    local = root / "local" / "settings.yaml"
    example = root / "local" / "settings.example.yaml"
    path = local if local.is_file() else example
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Settings file must be a mapping: {path}")
    return loaded


def get_paths(
    storage_root: str | Path | None = None,
    code_root: Path | None = None,
) -> ProjectPaths:
    code = (code_root or CODE_ROOT).resolve()
    if storage_root is None:
        storage_root = os.environ.get(STORAGE_ENV, "").strip()
    if not storage_root:
        storage_root = load_settings(code).get("storage_root", str(DEFAULT_STORAGE))
    given = Path(str(storage_root)).expanduser()
    if not given.is_absolute():
        raise ValueError(f"Storage root must be absolute, got {given}")
    if not given.is_dir():
        raise ValueError(f"Storage root does not exist: {given}")
    resolved = given.resolve()
    if overlaps(resolved, code):
        raise ValueError(
            "Storage root and code root overlap after resolving symlinks: "
            f"storage={resolved} code={code}"
        )
    return ProjectPaths(code_root=code, storage_root=given)
