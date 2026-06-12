#!/usr/bin/env python3
"""Step 03 entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from importlib import import_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 03: candidate pool")
    parser.add_argument("--force", action="store_true", help="Re-fetch PubTator3")
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Recompute ranking baselines + report section on existing frozen pool",
    )
    parser.add_argument(
        "--type-alignment-only",
        action="store_true",
        help="Read-only entity-type alignment diagnostic on frozen pool; patch report",
    )
    args = parser.parse_args()

    print(f"=== Step 03 start {__import__('datetime').datetime.now().isoformat()} ===")
    build = import_module("03_candidate_pool.build_pool")
    if args.type_alignment_only:
        build.refresh_entity_type_alignment()
    elif args.recall_diagnostic_only:
        build.refresh_recall_diagnostic()
    elif args.baselines_only:
        build.refresh_baselines_and_report()
    else:
        build.build_candidate_pool(force_fetch=args.force)
    print("=== Step 03 complete ===")


if __name__ == "__main__":
    main()
