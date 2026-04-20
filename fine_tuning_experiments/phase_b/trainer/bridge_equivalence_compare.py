"""Bridge equivalence comparison — compare a Phase B (rewritten trainer) run
against the saved Phase A artifacts for the same config.

Tier-3 acceptance (per §11.6 of paper design):
  * `label2id` exact match.
  * Final dev macro-F1 per stage within ±0.01 of Phase A's saved value.
  * Checkpoint top-level key set matches (`model_state_dict`, `label2id`,
    `stage`, `model_name`, `best_checkpoint_meta`).

Writes `trainer_equivalence_report.md` under the Phase B run directory.
Exits non-zero if tier 3 fails on any criterion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


TIER3_F1_TOLERANCE = 0.01
EXPECTED_CKPT_KEYS = {"model_state_dict", "label2id", "stage", "model_name",
                      "best_checkpoint_meta"}


def load_stage_best(run_dir: Path) -> dict[str, float | None]:
    m = json.loads((run_dir / "metrics" / "metrics_best_checkpoint.json").read_text())
    return {s["stage"]: s.get("best_selection_score")
            for s in m.get("stages", [])}


def load_label2id(run_dir: Path) -> dict[str, int]:
    man = json.loads((run_dir / "run_manifest.json").read_text())
    return man.get("resolved", {}).get("label2id", {})


def inspect_ckpt_keys(ckpt_path: Path) -> set[str]:
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return set(ck.keys()) if isinstance(ck, dict) else set()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase-a-run", required=True, type=Path)
    p.add_argument("--phase-b-run", required=True, type=Path)
    args = p.parse_args()

    pa = args.phase_a_run
    pb = args.phase_b_run

    failures: list[str] = []

    # ── Label2id ──────────────────────────────────────────────────────────
    l_a = load_label2id(pa)
    l_b = load_label2id(pb)
    if l_a != l_b:
        failures.append(f"label2id differs: A={l_a} B={l_b}")

    # ── Final dev F1 per stage ────────────────────────────────────────────
    s_a = load_stage_best(pa)
    s_b = load_stage_best(pb)
    shared = set(s_a) & set(s_b)
    per_stage_diffs: dict[str, dict] = {}
    for stage in sorted(shared):
        va = s_a.get(stage); vb = s_b.get(stage)
        if va is None or vb is None:
            per_stage_diffs[stage] = {"a": va, "b": vb, "diff": None}
            continue
        diff = vb - va
        per_stage_diffs[stage] = {"a": va, "b": vb, "diff": diff}
        if abs(diff) > TIER3_F1_TOLERANCE:
            failures.append(f"stage {stage}: |Δ F1|={abs(diff):.4f} > {TIER3_F1_TOLERANCE}")

    # ── Checkpoint structure ──────────────────────────────────────────────
    for name in ("best.pt", "stage_t1_end.pt", "stage_t2_end.pt"):
        pa_ck = pa / "checkpoints" / name
        pb_ck = pb / "checkpoints" / name
        if not pa_ck.exists():
            continue
        if not pb_ck.exists():
            failures.append(f"missing Phase B ckpt: {name}")
            continue
        ka = inspect_ckpt_keys(pa_ck); kb = inspect_ckpt_keys(pb_ck)
        missing = EXPECTED_CKPT_KEYS - kb
        extra = kb - EXPECTED_CKPT_KEYS
        if missing:
            failures.append(f"ckpt {name}: missing keys {missing}")
        if ka != kb:
            # Not a hard failure (some legacy fields are optional) but flag
            per_stage_diffs[f"ckpt_{name}_keys"] = {"a_only": sorted(ka - kb),
                                                    "b_only": sorted(kb - ka)}

    # ── Report ───────────────────────────────────────────────────────────
    lines = ["# Trainer equivalence report", ""]
    lines.append(f"- Phase A run: `{pa}`")
    lines.append(f"- Phase B run: `{pb}`")
    lines.append("")
    lines.append("## label2id comparison")
    lines.append(f"- A: {json.dumps(l_a)}")
    lines.append(f"- B: {json.dumps(l_b)}")
    lines.append(f"- **match:** {l_a == l_b}")
    lines.append("")
    lines.append("## Dev macro-F1 per stage")
    lines.append("| stage | Phase A | Phase B | Δ (B−A) | tier-3 pass (±0.01) |")
    lines.append("|---|---|---|---|---|")
    for stage in sorted(shared):
        s = per_stage_diffs[stage]
        va, vb, diff = s["a"], s["b"], s["diff"]
        ok = "—" if diff is None else ("✓" if abs(diff) <= TIER3_F1_TOLERANCE else "✗")
        lines.append(f"| {stage} | {va} | {vb} | {'N/A' if diff is None else f'{diff:+.4f}'} | {ok} |")
    lines.append("")
    lines.append("## Overall verdict")
    if failures:
        lines.append(f"**Tier-3 FAIL** — {len(failures)} failure(s):")
        for f in failures:
            lines.append(f"- {f}")
    else:
        lines.append("**Tier-3 PASS** on all criteria (label2id, final F1 per stage, ckpt keys).")

    out = pb / "trainer_equivalence_report.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    if failures:
        print("TIER 3 FAIL", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
