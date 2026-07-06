# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the root-cause graph localisation detector."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.rca import RootCauseGraphDetector


def _matrix_with_node_fault(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (200, 6))
    x[120:140, 3] += 8.0  # fault localised on node 3
    return x.astype(np.float64)


class TestContract:
    def test_unfitted_then_fitted(self) -> None:
        det = RootCauseGraphDetector()
        assert det.is_fitted() is False
        det.fit(np.random.default_rng(0).normal(size=(200, 6)))
        assert det.is_fitted() is True

    def test_scores_shape_and_range(self) -> None:
        x = _matrix_with_node_fault()
        scores = np.asarray(RootCauseGraphDetector().fit(x).detect(x)["scores"])
        assert scores.shape == (x.shape[0],)
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0

    def test_extract_features_shape(self) -> None:
        x = _matrix_with_node_fault()
        feats = RootCauseGraphDetector().fit(x).extract_features(x)
        assert feats.shape == (x.shape[0], 1)
        assert np.isfinite(feats).all()

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            RootCauseGraphDetector(damping=1.5)
        with pytest.raises(ValueError):
            RootCauseGraphDetector(walk_iters=0)

    def test_deterministic(self) -> None:
        x = _matrix_with_node_fault(7)
        a = RootCauseGraphDetector().fit(x).detect(x)["scores"]
        b = RootCauseGraphDetector().fit(x).detect(x)["scores"]
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class TestSignal:
    def test_fault_flagged_and_ranked(self) -> None:
        train = np.random.default_rng(1).normal(size=(400, 6))
        x = _matrix_with_node_fault(2)
        det = RootCauseGraphDetector().fit(train)
        scores = np.asarray(det.detect(x)["scores"])
        assert float(scores[120:140].max()) > 0.5
        # Localise a faulted row (row 130): node 3 should rank first.
        ranked = det.rank_root_causes(x[:131])
        assert ranked[0][0] == 3

    def test_low_false_positive_rate(self) -> None:
        train = np.random.default_rng(3).normal(size=(400, 6))
        test = np.random.default_rng(4).normal(size=(400, 6))
        scores = np.asarray(
            RootCauseGraphDetector(calibration_quantile=0.98).fit(train).detect(test)["scores"]
        )
        assert (scores > 0.5).mean() < 0.08
