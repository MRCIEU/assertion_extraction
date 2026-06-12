"""Provenance verification helpers for manuscript source regeneration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ClaimCheck:
    step: str
    claim: str
    source: str
    status: str  # OK, CORRECTED, SOURCE NOT FOUND


def verify_value(
    step: str,
    claim: str,
    source_path: Path,
    *,
    reader=None,
    key=None,
    expected=None,
    tolerance: float = 1e-6,
) -> ClaimCheck:
    """Verify a numeric claim against a source artifact."""
    if not source_path.exists():
        return ClaimCheck(step, claim, str(source_path), "SOURCE NOT FOUND")
    try:
        if reader is None:
            if source_path.suffix == ".csv":
                df = pd.read_csv(source_path)
                if key is None:
                    val = len(df)
                elif isinstance(key, tuple):
                    val = df.loc[key[0], key[1]]
                else:
                    val = df[key].iloc[0] if key in df.columns else df.iloc[0][key]
            else:
                import json

                data = json.loads(source_path.read_text(encoding="utf-8"))
                val = data if key is None else data[key] if not isinstance(key, list) else data
                for k in key:
                    val = val[k]
        else:
            val = reader(source_path)
        if expected is not None:
            if isinstance(expected, (int, float)) and isinstance(val, (int, float)):
                ok = abs(float(val) - float(expected)) <= tolerance
            else:
                ok = val == expected
            status = "OK" if ok else f"CORRECTED (artifact={val})"
        else:
            status = "OK"
        return ClaimCheck(step, claim, str(source_path), status)
    except Exception as exc:
        return ClaimCheck(step, claim, str(source_path), f"SOURCE NOT FOUND ({exc})")


def print_verification(notes: list[ClaimCheck]) -> None:
    step = notes[0].step if notes else "?"
    print(f"\n=== Provenance verification: step {step} ===")
    for n in notes:
        print(f"  [{n.status}] {n.claim} <- {n.source}")
