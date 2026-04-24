#!/usr/bin/env python3.11
"""Bulk-update Phase B LoRA configs to LR = 3.0e-4 per Appendix B amendment B.8.

Only `scientific_trainer.learning_rate` is changed (2.0e-05 -> 3.0e-04).  All
other fields are preserved byte-identically.  The smoke config
PB_PB_LR_T1B_s99 is already at 3.0e-04 and will be a no-op.

FT configs are not touched — they continue to use the pre-registered 2.0e-05
because amendment B.8 is specifically a LoRA-only change (see (d) in the
amendment).

Idempotent: safe to re-run.
"""
from __future__ import annotations

import sys
from pathlib import Path

CONFIG_DIR = Path("/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/phase_b/configs")
OLD_LR = "learning_rate: 2.0e-05"
NEW_LR = "learning_rate: 3.0e-04"


def main() -> None:
    lr_configs = sorted(CONFIG_DIR.glob("PB_*_LR_*.yaml"))
    print(f"Found {len(lr_configs)} LoRA configs in {CONFIG_DIR}")
    touched = skipped = already = 0
    for p in lr_configs:
        text = p.read_text()
        if OLD_LR in text:
            # The only `learning_rate:` line at indent 2 is the scientific_trainer one;
            # `t4_learning_rate:` has a different key and is not matched.
            text2 = text.replace(OLD_LR, NEW_LR)
            p.write_text(text2)
            touched += 1
        elif NEW_LR in text:
            already += 1
        else:
            print(f"  WARN: no expected LR line found in {p.name}")
            skipped += 1

    print(f"  updated:            {touched}")
    print(f"  already at 3.0e-04: {already}")
    print(f"  skipped (no match): {skipped}")

    # Verify invariant post-update
    mismatched = []
    for p in lr_configs:
        t = p.read_text()
        if OLD_LR in t:
            mismatched.append(p.name)
    if mismatched:
        print(f"FAIL: {len(mismatched)} configs still at 2.0e-05: {mismatched[:5]}")
        sys.exit(1)
    print("OK: all LoRA configs at learning_rate=3.0e-04")


if __name__ == "__main__":
    main()
