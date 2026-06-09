# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for unifying the neuro-symbolic blend (Issue #5).

The hardcoded ``0.6*neural + 0.4*symbolic`` static blends (previously duplicated
in models/neurosymbolic.py and core/symbolic_reasoning.py) are replaced by a
single canonical adaptive blend (``adaptive_neurosymbolic_fuse``). These tests
prove:
  * one implementation is used everywhere (both reason() sites delegate to it),
  * sane reduction properties (equal confidence -> even blend), and
  * parity-or-better classification vs. the old static blend on a fixture where
    branch confidence correlates with reliability (the premise that justifies
    confidence-weighted fusion).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from omni_mercury_engine.cognitive.neurosymbolic_fusion import (
    FusionStrategy,
    adaptive_neurosymbolic_fuse,
)
from omni_mercury_engine.core.symbolic_reasoning import SymbolicReasoningEngine
from omni_mercury_engine.models.neurosymbolic import SymbolicReasoningLayer


class TestCanonicalBlend:
    def test_equal_confidence_is_even_blend(self) -> None:
        # With equal (derived) confidence, confidence-weighted reduces to the
        # mean of the two scores.
        fused, _ = adaptive_neurosymbolic_fuse(0.8, 0.4)
        # both decisiveness = 0.6 and 0.2 -> not equal; use explicit confidences
        fused_eq, _ = adaptive_neurosymbolic_fuse(0.8, 0.4, 0.5, 0.5)
        assert abs(fused_eq - 0.6) < 1e-9

    def test_confident_neural_dominates(self) -> None:
        fused, _ = adaptive_neurosymbolic_fuse(0.95, 0.5, 0.9, 0.1)
        assert fused > 0.8  # decisive neural pulls result toward neural

    def test_confident_symbolic_dominates(self) -> None:
        fused, _ = adaptive_neurosymbolic_fuse(0.5, 0.95, 0.1, 0.9)
        assert fused > 0.8

    def test_output_bounded(self) -> None:
        for _ in range(200):
            n, s = np.random.rand(), np.random.rand()
            fused, conf = adaptive_neurosymbolic_fuse(n, s)
            assert 0.0 <= fused <= 1.0
            assert 0.0 <= conf <= 1.0

    def test_gated_strategy_supported(self) -> None:
        fused, conf = adaptive_neurosymbolic_fuse(0.9, 0.2, 0.8, 0.3, strategy=FusionStrategy.GATED)
        assert 0.0 <= fused <= 1.0


class TestParityOrBetter:
    """Adaptive blend classifies at least as well as the old static blend on a
    fixture where confidence correlates with reliability."""

    def _fixture(self, n: int = 4000, seed: int = 0) -> tuple[
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ]:
        rng = np.random.RandomState(seed)
        # Latent truth from two independent evidence sources.
        truth_n = rng.rand(n)
        truth_s = rng.rand(n)
        label = ((truth_n + truth_s) / 2 > 0.5).astype(int)

        # Observed scores: a branch is reliable (low noise) when it is decisive
        # (far from 0.5); unreliable branches are noisy. This is exactly the
        # regime confidence-weighting is designed for.
        def observe(
            truth: np.ndarray[Any, Any],
        ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
            decisiveness = np.abs(truth - 0.5) * 2.0
            noise = rng.normal(0, 0.35 * (1 - decisiveness))
            obs = np.clip(truth + noise, 0.0, 1.0)
            conf = decisiveness
            return obs, conf

        neural, n_conf = observe(truth_n)
        symbolic, s_conf = observe(truth_s)
        return neural, symbolic, n_conf, s_conf, label

    def test_adaptive_not_worse_than_static(self) -> None:
        neural, symbolic, n_conf, s_conf, label = self._fixture()

        static_pred = (0.6 * neural + 0.4 * symbolic > 0.5).astype(int)
        adaptive_scores = np.array(
            [
                adaptive_neurosymbolic_fuse(n, s, nc, sc)[0]
                for n, s, nc, sc in zip(neural, symbolic, n_conf, s_conf)
            ]
        )
        adaptive_pred = (adaptive_scores > 0.5).astype(int)

        static_acc = float((static_pred == label).mean())
        adaptive_acc = float((adaptive_pred == label).mean())

        assert adaptive_acc >= static_acc, (
            f"adaptive accuracy {adaptive_acc:.4f} should be >= static "
            f"{static_acc:.4f} when confidence tracks reliability"
        )


class TestReasonSitesDelegate:
    """Both reason() implementations use the canonical blend, not a literal
    0.6/0.4 mix (regression guard against re-introducing the static blend)."""

    def test_models_reason_uses_adaptive(self) -> None:
        import inspect

        from omni_mercury_engine.models import neurosymbolic as m

        src = inspect.getsource(m.SymbolicReasoningLayer.reason)
        assert "adaptive_neurosymbolic_fuse" in src
        assert "0.6 * neural_score + 0.4" not in src

    def test_core_reason_uses_adaptive(self) -> None:
        import inspect

        from omni_mercury_engine.core import symbolic_reasoning as c

        src = inspect.getsource(c.SymbolicReasoningEngine.reason)
        assert "adaptive_neurosymbolic_fuse" in src
        assert "0.6 * neural_score + 0.4" not in src


class TestNoRulesFiredSemantics:
    def test_core_reason_no_rules_fired_preserves_neural_signal(self) -> None:
        engine = SymbolicReasoningEngine()
        engine.rules = []

        result = engine.reason(np.array([0.8]), {})

        assert result["symbolic_rules_fired"] == []
        assert result["confidence"] >= 0.79
        assert result["final_decision"] == "anomalous"

    def test_models_reason_no_rules_fired_preserves_neural_signal(self) -> None:
        layer = SymbolicReasoningLayer()

        result = layer.reason(0.8, {})

        assert result.rules_fired == []
        assert result.confidence >= 0.79
        assert result.result is True
