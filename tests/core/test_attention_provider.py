"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Regression suite for the concrete ``MultiHeadAttentionProvider`` (ROADMAP #7).

The provider is wired to a real ``torch.nn.MultiheadAttention`` surface and
replaces the removed deterministic-random placeholder.  These tests pin its
contract (per-head shape, softmax-normalised rows, fail-closed before any
forward, determinism) and that it drives the GOSNN optimizer's attention
metric instead of being skipped.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.gosnn_optimizer import (
    AttentionProvider,
    GOSNNOptimizer,
    MultiHeadAttentionProvider,
)


class TestMultiHeadAttentionProvider:
    def test_is_attention_provider(self) -> None:
        assert issubclass(MultiHeadAttentionProvider, AttentionProvider)

    def test_get_attention_before_observe_fails_closed(self) -> None:
        """Before any forward pass, the provider raises (model not yet run) —
        the optimizer must then skip the metric rather than score noise."""
        provider = MultiHeadAttentionProvider(d_model=64, num_heads=8, seed=0)
        with pytest.raises(RuntimeError, match="model not yet run"):
            provider.get_attention()

    def test_observe_returns_per_head_attention(self) -> None:
        provider = MultiHeadAttentionProvider(d_model=64, num_heads=8, seed=0)
        seq = np.random.default_rng(0).standard_normal((12, 64))
        attn = provider.observe(seq)
        # Per-head attention: (num_heads, seq_len, seq_len).
        assert attn.shape == (8, 12, 12)
        assert np.all(np.isfinite(attn))
        # get_attention returns the same cached scores.
        np.testing.assert_array_equal(attn, provider.get_attention())

    def test_attention_rows_are_softmax_normalised(self) -> None:
        """Real attention weights: each query row is a probability dist."""
        provider = MultiHeadAttentionProvider(d_model=32, num_heads=4, seed=1)
        seq = np.random.default_rng(1).standard_normal((6, 32))
        attn = provider.observe(seq)
        row_sums = attn.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)
        assert np.all(attn >= 0.0)

    def test_deterministic_under_seed(self) -> None:
        seq = np.random.default_rng(2).standard_normal((8, 64))
        a = MultiHeadAttentionProvider(d_model=64, num_heads=8, seed=7).observe(seq)
        b = MultiHeadAttentionProvider(d_model=64, num_heads=8, seed=7).observe(seq)
        np.testing.assert_allclose(a, b)

    def test_accepts_batched_input(self) -> None:
        provider = MultiHeadAttentionProvider(d_model=64, num_heads=8, seed=0)
        seq = np.random.default_rng(0).standard_normal((3, 9, 64))  # (batch, seq, d)
        attn = provider.observe(seq)
        assert attn.shape == (8, 9, 9)

    def test_d_model_not_divisible_by_heads_raises(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadAttentionProvider(d_model=65, num_heads=8)

    def test_wrong_feature_dim_raises(self) -> None:
        provider = MultiHeadAttentionProvider(d_model=64, num_heads=8, seed=0)
        with pytest.raises(ValueError, match="sequence must be"):
            provider.observe(np.zeros((10, 32)))


class TestProviderDrivesOptimizerMetric:
    def test_real_provider_drives_metric_not_skip(self) -> None:
        """A wired, observed provider makes the optimizer compute the
        attention-overhead metric from real model tensors — the skip-marker
        the placeholder emitted must be absent."""
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
            reset_global_network,
        )

        # 32 heads to match AttentionOptimizer's triadic φ-weighting.
        provider = MultiHeadAttentionProvider(d_model=64, num_heads=32, seed=0)
        provider.observe(np.random.default_rng(0).standard_normal((16, 64)))

        reset_global_network()
        optimizer = GOSNNOptimizer(attention_provider=provider)
        result = optimizer.optimize(GlobalOmniScalarNetwork())

        assert not any(
            "attention overhead metric skipped" in r.lower() for r in result.recommendations
        )

    def test_unobserved_provider_skips_fail_closed(self) -> None:
        """A provider that has not run yet raises RuntimeError at
        get_attention, so the optimizer skips the metric (does not fabricate
        one) — the fail-closed half of the contract."""
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
            reset_global_network,
        )

        provider = MultiHeadAttentionProvider(d_model=64, num_heads=32, seed=0)
        reset_global_network()
        optimizer = GOSNNOptimizer(attention_provider=provider)
        result = optimizer.optimize(GlobalOmniScalarNetwork())
        assert any("attention overhead metric skipped" in r.lower() for r in result.recommendations)
