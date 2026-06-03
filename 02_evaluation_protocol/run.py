#!/usr/bin/env python3
"""Step 02 entry point."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from importlib import import_module


def main() -> None:
    print(f"=== Step 02 start {__import__('datetime').datetime.now().isoformat()} ===")
    build = import_module("02_evaluation_protocol.build_protocol")
    build.build_protocol()
    print("=== Step 02 complete ===")


if __name__ == "__main__":
    main()
