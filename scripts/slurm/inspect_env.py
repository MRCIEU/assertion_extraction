"""Report whether named variables exist. Never print a variable value."""

from __future__ import annotations

import os
import sys

EXPECTED = (
    "PATH",
    "HOME",
    "USER",
    "BASH_ENV",
    "ENV",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "SLURM_JOB_ID",
    "SLURM_EXPORT_ENV",
)
CREDENTIAL_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AZURE_OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GH_TOKEN",
    "GITHUB_TOKEN",
)
CREDENTIAL_MARKERS = ("API_KEY", "SECRET", "PASSWORD", "PASSWD", "_TOKEN", "CREDENTIAL")
# Switches the project sets itself. A TOKEN substring here is not a secret.
PROJECT_SETTINGS = {
    "HF_HUB_DISABLE_IMPLICIT_TOKEN",
    "HF_HUB_ENABLE_HF_TRANSFER",
    "TOKENIZERS_PARALLELISM",
}


def credential_like(name: str) -> bool:
    if name in PROJECT_SETTINGS:
        return False
    upper = name.upper()
    if upper.startswith("SLURM_") or upper.startswith("SPANK_"):
        return False
    return any(marker in upper for marker in CREDENTIAL_MARKERS)


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "batch"
    print(f"stage={stage}")
    for name in EXPECTED:
        print(f"{name}_present={'yes' if name in os.environ else 'no'}")
    unexpected = False
    for name in CREDENTIAL_NAMES:
        present = name in os.environ
        print(f"credential_{name}_present={'yes' if present else 'no'}")
        if present:
            unexpected = True
    for name in os.environ:
        if name in CREDENTIAL_NAMES:
            continue
        if credential_like(name):
            unexpected = True
            break
    print(f"unexpected_credential_like_present={'yes' if unexpected else 'no'}")
    if unexpected:
        print("credential_check=stop")
        return 2
    print("credential_check=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
