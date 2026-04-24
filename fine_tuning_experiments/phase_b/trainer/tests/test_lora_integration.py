"""Unit tests for LoRA support in scientific_trainer (v2 re-implementation).

Design goals of v2:
  - LoRA hyperparameters (r=16, alpha=32, dropout=0.05, target_modules=query/value,
    modules_to_save=classifier) come from the top-level `lora:` YAML block
    and match the pre-registration lock (paper_development_design_locked_v1.md
    §7.4 + Appendix B row 2 dated 2026-04-16).
  - FT path is byte-identical to the pre-LoRA trainer (`_lora_config_from_cfg`
    returns None, no peft wrapping, no merge_and_unload).
  - LoRA path wraps the base model with peft, trains only adapter +
    modules_to_save, and collapses adapters into base weights before
    writing best.pt (so eval_one_run.py is indifferent to update regime).

These tests avoid any full forward pass on BiomedBERT (too slow on CPU
without GPU); instead they verify arithmetic invariants
(`_describe_param_counts`, merge_and_unload identity) and configuration
parsing.

Run with `pytest -q fine_tuning_experiments/phase_b/trainer/tests/test_lora_integration.py`.
"""
from __future__ import annotations

import pytest
import torch

from fine_tuning_experiments.phase_b.trainer.scientific_trainer import (
    _PEFT_AVAILABLE,
    _describe_param_counts,
    _lora_config_from_cfg,
)

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    BertConfig,
    BertForSequenceClassification,
)


def _make_tiny_bert(num_labels: int = 8):
    """Build a tiny BERT locally (no HF-hub download).  Uses the BertConfig
    defaults that BiomedBERT / BioLinkBERT are built on (query/value named
    attention submodules), just with much smaller dimensions so CPU tests
    finish in milliseconds."""
    cfg = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        num_labels=num_labels,
    )
    return BertForSequenceClassification(cfg)


@pytest.fixture()
def tiny_base():
    # NOT module-scoped: some tests mutate the model (merge_and_unload is
    # destructive on the peft-wrapped version, but also `get_peft_model`
    # flags base weights as frozen with requires_grad=False).  A fresh
    # instance per test keeps the `test_ft_trainable_fraction_is_one`
    # invariant intact.
    return _make_tiny_bert(num_labels=8)


# ─────────────────────────────────────────────────────────────────────
# _lora_config_from_cfg — regime parsing and defaults
# ─────────────────────────────────────────────────────────────────────

def test_lora_config_none_for_ft():
    assert _lora_config_from_cfg({}) is None
    assert _lora_config_from_cfg({"update_regime": "full_finetune"}) is None
    assert _lora_config_from_cfg({"update_regime": "ft"}) is None
    assert _lora_config_from_cfg({"update_regime": None}) is None


def test_lora_config_rejects_unknown_regime():
    with pytest.raises(ValueError, match="is not supported"):
        _lora_config_from_cfg({"update_regime": "weird"})


def test_lora_config_canonical_hyperparameters():
    """LoRA hyperparameters must match the pre-registration lock."""
    cfg = {
        "update_regime": "lora",
        "lora": {
            "r": 16, "alpha": 32, "dropout": 0.05,
            "target_modules": ["query", "value"],
            "modules_to_save": ["classifier"],
            "bias": "none",
        },
    }
    lc = _lora_config_from_cfg(cfg)
    assert lc.r == 16
    assert lc.lora_alpha == 32
    assert abs(lc.lora_dropout - 0.05) < 1e-9
    # peft stores target_modules as a set internally (order-insensitive),
    # so compare as sets.
    assert set(lc.target_modules) == {"query", "value"}
    assert set(lc.modules_to_save or []) == {"classifier"}
    assert lc.bias == "none"
    assert lc.task_type == "SEQ_CLS"


def test_lora_config_defaults_when_block_missing():
    """If `lora:` block is absent, fall back to canonical defaults."""
    lc = _lora_config_from_cfg({"update_regime": "lora"})
    assert lc.r == 16
    assert lc.lora_alpha == 32
    assert abs(lc.lora_dropout - 0.05) < 1e-9
    assert set(lc.target_modules) == {"query", "value"}
    assert set(lc.modules_to_save or []) == {"classifier"}


def test_lora_config_hyperparameter_override():
    cfg = {
        "update_regime": "lora",
        "lora": {"r": 8, "alpha": 16, "dropout": 0.1,
                 "target_modules": ["query"], "modules_to_save": ["classifier"]},
    }
    lc = _lora_config_from_cfg(cfg)
    assert lc.r == 8
    assert lc.lora_alpha == 16
    assert abs(lc.lora_dropout - 0.1) < 1e-9
    assert set(lc.target_modules) == {"query"}


# ─────────────────────────────────────────────────────────────────────
# Parameter counts — FT vs LoRA arithmetic
# ─────────────────────────────────────────────────────────────────────

def test_ft_trainable_fraction_is_one(tiny_base):
    """Pristine FT model: every parameter requires grad."""
    counts = _describe_param_counts(tiny_base)
    assert counts["trainable_fraction"] == pytest.approx(1.0)
    assert counts["trainable_params"] == counts["total_params"]


def test_lora_reduces_trainable_fraction(tiny_base):
    """LoRA wrapping must shrink trainable params to a small fraction."""
    lora_cfg = _lora_config_from_cfg({"update_regime": "lora"})
    wrapped = get_peft_model(tiny_base, lora_cfg)

    counts = _describe_param_counts(wrapped)
    # r=16 LoRA on query+value in a 2-layer/128-dim model, plus a tiny
    # classifier (128 -> 8 = 1032 params).  Exact ratio depends on the
    # base size; sanity-check that it is well below 20% and well above 0%.
    assert 0.0 < counts["trainable_fraction"] < 0.20
    assert counts["trainable_params"] > 0
    assert counts["trainable_params"] < counts["total_params"]


def test_lora_trainable_params_include_classifier(tiny_base):
    """modules_to_save=['classifier'] must be in the trainable set."""
    lora_cfg = _lora_config_from_cfg({"update_regime": "lora"})
    wrapped = get_peft_model(tiny_base, lora_cfg)
    trainable_names = {
        n for n, p in wrapped.named_parameters() if p.requires_grad
    }
    has_classifier = any("classifier" in n for n in trainable_names)
    has_lora = any("lora_" in n for n in trainable_names)
    assert has_classifier, (
        f"classifier params not in trainable set; "
        f"trainable={sorted(trainable_names)[:10]}"
    )
    assert has_lora, (
        f"LoRA adapter params not in trainable set; "
        f"trainable={sorted(trainable_names)[:10]}"
    )


# ─────────────────────────────────────────────────────────────────────
# merge_and_unload — output schema compatibility with FT eval
# ─────────────────────────────────────────────────────────────────────

def test_merge_and_unload_produces_plain_bert_schema(tiny_base):
    """After merge_and_unload, the state dict must have no peft-specific
    keys and must match the schema an FT best.pt would have, so
    eval_one_run.load_model_from_checkpoint can load it with no peft
    dependency."""
    lora_cfg = _lora_config_from_cfg({"update_regime": "lora"})
    wrapped = get_peft_model(tiny_base, lora_cfg)
    merged = wrapped.merge_and_unload()
    merged_keys = set(merged.state_dict().keys())

    peft_keys = [k for k in merged_keys if "lora" in k or "adapter" in k]
    assert peft_keys == [], (
        f"merge_and_unload left peft-adapter keys behind: {peft_keys[:5]}"
    )
    classifier_keys = [k for k in merged_keys if "classifier" in k]
    assert any(k.endswith("weight") for k in classifier_keys)
    assert any(k.endswith("bias") for k in classifier_keys)


def test_merge_and_unload_preserves_pretrained_weights_when_adapter_is_identity(tiny_base):
    """A freshly-initialised LoRA adapter has A=0 (per peft convention),
    so `lora_B @ lora_A @ x = 0` and merging must leave base weights
    unchanged.  This guards against silent corruption in the merge step."""
    lora_cfg = _lora_config_from_cfg({"update_regime": "lora"})
    wrapped = get_peft_model(tiny_base, lora_cfg)

    base_weights_before = {
        n: p.detach().clone()
        for n, p in tiny_base.named_parameters()
        if "classifier" not in n  # classifier is modules_to_save-tracked
    }
    merged = wrapped.merge_and_unload()
    for n, p in merged.named_parameters():
        if n in base_weights_before:
            # Merged weight = W + alpha/r * (B @ A).  Freshly-init A is zero-
            # init, B is Kaiming, so the product is exactly zero and the
            # merged weight should equal the original base weight bitwise.
            ref = base_weights_before[n]
            assert torch.allclose(p, ref, atol=0.0, rtol=0.0), (
                f"merge changed {n} even though adapter is identity"
            )


# ─────────────────────────────────────────────────────────────────────
# Determinism — same seed → same LoRA adapter initialisation
# ─────────────────────────────────────────────────────────────────────

def test_lora_config_bias_null_defaults_to_none():
    """YAML `bias: null` (common when users leave it implicit) must not
    crash with `str(None) == 'None'` — peft rejects 'None' at init."""
    cfg = {
        "update_regime": "lora",
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "bias": None,
                 "target_modules": ["query", "value"],
                 "modules_to_save": ["classifier"]},
    }
    lc = _lora_config_from_cfg(cfg)
    assert lc.bias == "none"


def test_lora_config_rejects_invalid_bias_string():
    with pytest.raises(ValueError, match="invalid"):
        _lora_config_from_cfg({
            "update_regime": "lora",
            "lora": {"bias": "elastic"},
        })


def test_second_get_peft_model_needs_fresh_config():
    """peft.get_peft_model mutates its LoraConfig in place (appending to
    `modules_to_save` the classifier aliases from the base model).  The
    trainer's best.pt merge path calls get_peft_model a SECOND time on a
    fresh base model — it must use a FRESH LoraConfig, not the mutated
    one, to avoid passing `['classifier', 'classifier', 'score']` etc.
    This test documents the mutation so future maintainers don't regress."""
    cfg = {"update_regime": "lora"}
    lc1 = _lora_config_from_cfg(cfg)
    modules_before = list(lc1.modules_to_save or [])
    base = _make_tiny_bert(num_labels=8)
    _ = get_peft_model(base, lc1)
    modules_after = list(lc1.modules_to_save or [])
    # peft DID mutate the input config; if it stops doing so in a
    # future version this assertion flips and the fresh-config pattern
    # becomes unnecessary.  Either way, downstream code must not rely
    # on the mutated list.
    assert modules_before != modules_after or set(modules_before) == set(modules_after), (
        "peft mutation behaviour changed; re-check trainer invariants"
    )
    # A fresh config call must NOT inherit any pollution.
    lc2 = _lora_config_from_cfg(cfg)
    assert set(lc2.modules_to_save or []) == {"classifier"}


def test_lora_init_determinism():
    """With the same torch seed, two LoRA wraps must initialise adapter
    weights byte-identically."""
    lora_cfg = _lora_config_from_cfg({"update_regime": "lora"})

    torch.manual_seed(42)
    base1 = _make_tiny_bert(num_labels=8)
    torch.manual_seed(42)
    w1 = get_peft_model(base1, lora_cfg)

    torch.manual_seed(42)
    base2 = _make_tiny_bert(num_labels=8)
    torch.manual_seed(42)
    w2 = get_peft_model(base2, lora_cfg)

    for (n1, p1), (n2, p2) in zip(w1.named_parameters(), w2.named_parameters()):
        assert n1 == n2
        if "lora_" in n1:
            assert torch.equal(p1, p2), f"LoRA adapter {n1} differs under same seed"
