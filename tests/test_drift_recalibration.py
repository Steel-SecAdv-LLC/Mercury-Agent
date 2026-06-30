# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for online drift recalibration (issue #3).

Drift previously demoted a grounded verdict to DEFER but left the threshold
stale. enable_online_recalibration wires Gibbs-Candes AdaptiveConformalInference
into detect_with_fusion so the operating threshold tracks score drift.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.core.conformal_prediction import AdaptiveConformalInference


class TestAdaptiveConformalUpdate:
    def test_uncovered_raises_threshold(self) -> None:
        aci = AdaptiveConformalInference(target_coverage=0.9, learning_rate=0.1, initial_threshold=0.5)
        t0 = aci.get_current_threshold()
        # score above threshold -> not covered -> threshold rises
        new_t, covered = aci.update(0.9)
        assert covered is False
        assert new_t > t0

    def test_covered_lowers_threshold(self) -> None:
        aci = AdaptiveConformalInference(target_coverage=0.9, learning_rate=0.1, initial_threshold=0.5)
        t0 = aci.get_current_threshold()
        new_t, covered = aci.update(0.1)  # below threshold -> covered
        assert covered is True
        assert new_t <= t0

    def test_threshold_tracks_score_quantile_under_drift(self) -> None:
        aci = AdaptiveConformalInference(target_coverage=0.8, learning_rate=0.05, initial_threshold=0.5)
        rng = np.random.default_rng(0)
        # Stream from a shifted-up distribution; threshold should climb to track
        # the ~80th percentile of the new score regime.
        for _ in range(2000):
            aci.update(float(np.clip(rng.normal(0.8, 0.1), 0, 1)))
        stats = aci.get_coverage_stats()
        assert abs(stats["empirical_coverage"] - 0.8) < 0.1
        assert aci.get_current_threshold() > 0.5  # rose to track the drift


class TestEngineOnlineRecalibration:
    def _engine(self):
        from omni_mercury_engine.engine import OmniMercuryEngine

        # Legacy auto-fit opt-in keeps this light (no separate fit_fusion needed
        # to exercise the recalibration mechanism).
        return OmniMercuryEngine(mode="fusion", device="cpu", require_explicit_fit=False)

    def test_detect_surfaces_adaptive_threshold(self) -> None:
        engine = self._engine()
        engine.enable_online_recalibration(target_coverage=0.9, learning_rate=0.1, warmup=1)
        rng = np.random.default_rng(0)
        res = engine.detect_with_fusion(rng.normal(0, 1, (16, 6)))
        assert "adaptive_threshold" in res
        assert "adaptive_conformal" in res
        assert res["adaptive_conformal"]["n_updates"] == 16

    def test_threshold_updates_across_calls(self) -> None:
        engine = self._engine()
        engine.enable_online_recalibration(target_coverage=0.9, learning_rate=0.1, warmup=1)
        rng = np.random.default_rng(1)
        engine.detect_with_fusion(rng.normal(0, 1, (16, 6)))
        n1 = engine._adaptive_conformal.n_updates
        engine.detect_with_fusion(rng.normal(0, 1, (16, 6)))
        n2 = engine._adaptive_conformal.n_updates
        assert n2 > n1  # online updates accumulate; threshold is not stale

    def test_disabled_by_default_is_noop(self) -> None:
        engine = self._engine()
        rng = np.random.default_rng(2)
        res = engine.detect_with_fusion(rng.normal(0, 1, (16, 6)))
        assert "adaptive_threshold" not in res
        assert engine._adaptive_conformal is None
