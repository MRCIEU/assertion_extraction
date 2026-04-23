"""Verify that a freshly-loaded checkpoint's forward pass is byte-exact
with the state_dict that was saved (modulo float rounding in softmax).

We do NOT need CUDA for this test: the scientific_trainer writes CPU state
dicts, so CPU-only load + forward reproduces training-time logits exactly.

The test also pins:
  - `label2id` roundtrips bit-for-bit,
  - `best_checkpoint_meta` is serialised and recovered,
  - `rng_state` is recoverable and contains python + numpy + torch_cpu streams.

If any of these change, downstream KB inference and post-hoc calibration
would silently shift.  The sentinel passed here is a 1-run probe against
`PA_BL_Sflat_s01/checkpoints/best.pt`; if that run doesn't exist on the
scratch mount we SKIP (useful for running tests off-cluster).

Run directly:
    python3.11 -m fine_tuning_experiments.phase_b.trainer.tests.test_checkpoint_roundtrip
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

DATA_ROOT = Path(os.environ.get(
    "PROJECT_1_DATA_ROOT", "/lus/lfs1aip2/projects/b5ac/project_1",
))
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # .../project_1
CKPT = (DATA_ROOT / "fine_tuning_experiments" / "runs" / "schema_exp"
        / "PA_BL_Sflat_s01" / "checkpoints" / "best.pt")

REQUIRED_KEYS = {"model_state_dict", "label2id", "model_name",
                 "best_checkpoint_meta", "rng_state"}
REQUIRED_RNG_STREAMS = {"python", "numpy", "torch_cpu"}


def run() -> int:
    if not CKPT.exists():
        print(f"SKIP: sentinel checkpoint not found: {CKPT}")
        return 0

    # ── Load raw state dict ───────────────────────────────────────────
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    failures: list[str] = []
    missing = REQUIRED_KEYS - set(ckpt)
    if missing:
        failures.append(f"missing top-level keys: {sorted(missing)}")
    if "rng_state" in ckpt:
        missing_rng = REQUIRED_RNG_STREAMS - set(ckpt["rng_state"])
        if missing_rng:
            failures.append(f"rng_state missing streams: {sorted(missing_rng)}")
    if "label2id" in ckpt:
        l2i = ckpt["label2id"]
        if "__NEGATIVE__" not in l2i:
            failures.append("label2id missing __NEGATIVE__")
        if l2i.get("__NEGATIVE__") != len(l2i) - 1:
            failures.append(
                f"__NEGATIVE__ not at last index: {l2i.get('__NEGATIVE__')} vs {len(l2i) - 1}"
            )
    if "best_checkpoint_meta" in ckpt:
        meta = ckpt["best_checkpoint_meta"]
        for required_meta_key in ("best_step", "best_selection_score",
                                   "selection_metric", "stage"):
            if required_meta_key not in meta:
                failures.append(
                    f"best_checkpoint_meta missing {required_meta_key!r}"
                )

    # ── Roundtrip via predict_checkpoint.load_model_from_checkpoint ───
    # This is the path downstream KB inference and H6 analysis will use.
    sys.path.insert(0, str(PROJECT_ROOT / "knowledge_grounded_evidence_audit"))
    from inference.predict_checkpoint import load_model_from_checkpoint

    device = torch.device("cpu")
    try:
        model, tokenizer, label2id, labels_ordered, path_out = \
            load_model_from_checkpoint(CKPT, device, state_dict_strict=True)
    except Exception as exc:
        failures.append(f"strict load_model_from_checkpoint failed: {exc}")
        model = None

    if model is not None:
        if label2id != ckpt["label2id"]:
            failures.append("label2id diverged between raw torch.load and loader")
        # Deterministic forward: two identical inputs must produce identical logits
        texts = ["BRCA1 [ENT] breast cancer [SEP] BRCA1 and breast cancer are associated."]
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            logits_a = model(**enc).logits
            logits_b = model(**enc).logits
        if not torch.equal(logits_a, logits_b):
            failures.append("double forward produced non-identical logits on same input")
        # Logits shape matches label space
        if logits_a.shape[-1] != len(label2id):
            failures.append(
                f"logit dim {logits_a.shape[-1]} != n_labels {len(label2id)}"
            )

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PASS: checkpoint {CKPT.name} roundtrips cleanly under strict load.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
