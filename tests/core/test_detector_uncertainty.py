# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Detectors report real decision uncertainty to fusion, never a constant 0.

Regression: ``BaseDetector.get_uncertainty`` returned a hard 0.0 for every
detector built on the shared base, so fusion was told that all of them were
perfectly certain of every verdict, including those sitting exactly on
their decision thresholds.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.core.base import BaseDetector


class TestUncertaintyFromResult:
    def test_boundary_score_is_maximally_uncertain(self) -> None:
        result = {"anomaly_score": 0.5}
        assert BaseDetector.uncertainty_from_result(result, threshold=0.5) == pytest.approx(1.0)

    @pytest.mark.parametrize("score", [0.0, 1.0])
    def test_extreme_scores_are_certain(self, score: float) -> None:
        result = {"anomaly_score": score}
        assert BaseDetector.uncertainty_from_result(result, threshold=0.5) == pytest.approx(0.0)

    def test_margin_scales_linearly(self) -> None:
        result = {"anomaly_score": 0.75}
        assert BaseDetector.uncertainty_from_result(result, threshold=0.5) == pytest.approx(0.5)

    def test_asymmetric_threshold_uses_the_wider_span(self) -> None:
        # threshold 0.8: span is 0.8, so a score of 0.0 is fully certain and
        # 1.0 (margin 0.2/0.8) is not.
        assert BaseDetector.uncertainty_from_result(
            {"anomaly_score": 0.0}, threshold=0.8
        ) == pytest.approx(0.0)
        assert BaseDetector.uncertainty_from_result(
            {"anomaly_score": 1.0}, threshold=0.8
        ) == pytest.approx(0.75)

    def test_batch_result_averages_per_sample_margins(self) -> None:
        result = {"scores": [0.5, 1.0]}
        assert BaseDetector.uncertainty_from_result(result, threshold=0.5) == pytest.approx(0.5)

    def test_detector_reported_uncertainty_wins(self) -> None:
        result = {"anomaly_score": 0.5, "uncertainty": 0.123}
        assert BaseDetector.uncertainty_from_result(result, threshold=0.5) == pytest.approx(0.123)

    def test_empty_scores_are_fully_uncertain(self) -> None:
        assert BaseDetector.uncertainty_from_result({"scores": []}) == pytest.approx(1.0)


class TestDefaultGetUncertainty:
    def test_base_default_is_no_longer_a_constant_zero(self) -> None:
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        rng = np.random.default_rng(7)
        normal = rng.standard_normal((200, 4))
        detector = MercuryAnomalyDetector()
        detector.fit(normal)

        in_distribution = detector.get_uncertainty(normal[:16])
        far_out = detector.get_uncertainty(normal[:16] + 50.0)
        assert 0.0 <= in_distribution <= 1.0
        assert 0.0 <= far_out <= 1.0
        # A gross anomaly is decided with more certainty (a smaller margin
        # uncertainty) than borderline in-distribution data; and at least one
        # of the two must be nonzero, which the old constant could never be.
        assert far_out < in_distribution
        assert in_distribution > 0.0


class TestEngineCollectsUncertainties:
    def test_extraction_reports_per_detector_uncertainty(self) -> None:
        from omni_mercury_engine.engine import OmniMercuryEngine

        engine = OmniMercuryEngine(require_explicit_fit=False)
        rng = np.random.default_rng(11)
        data = rng.standard_normal((64, 6))
        _, scores, _, uncertainties = engine._extract_detector_features(data)
        assert set(uncertainties) == set(scores)
        assert uncertainties, "no detector produced an uncertainty"
        for name, value in uncertainties.items():
            assert 0.0 <= value <= 1.0, name
        assert any(value > 0.0 for value in uncertainties.values())
