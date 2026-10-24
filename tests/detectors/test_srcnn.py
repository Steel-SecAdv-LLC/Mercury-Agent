# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the SR-CNN saliency-discriminator detector."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.detectors.srcnn import SRCNNDetector


def _series(seed: int = 0, n: int = 400) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=n)
    return x


class TestContract:
    def test_unfitted_then_fitted(self) -> None:
        det = SRCNNDetector(epochs=3)
        assert det.is_fitted() is False
        det.fit(_series())
        assert det.is_fitted() is True

    def test_scores_shape_and_range(self) -> None:
        x = _series()
        det = SRCNNDetector(epochs=3).fit(x)
        result = det.detect(x)
        scores = np.asarray(result["scores"])
        assert scores.shape == (x.shape[0],)
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0
        assert 0.0 <= float(result["anomaly_score"]) <= 1.0

    def test_extract_features_shape(self) -> None:
        x = _series()
        feats = SRCNNDetector(epochs=3).fit(x).extract_features(x)
        assert feats.shape == (x.shape[0], 1)
        assert np.isfinite(feats).all()

    def test_even_window_forced_odd(self) -> None:
        assert SRCNNDetector(window=32).window == 33

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            SRCNNDetector(window=2)
        with pytest.raises(ValueError):
            SRCNNDetector(inject_ratio=1.5)
        with pytest.raises(ValueError):
            SRCNNDetector(lr=0.0)


class TestSignal:
    def test_injected_spike_scored_high(self) -> None:
        rng = np.random.default_rng(5)
        train = rng.normal(0.0, 1.0, size=500)
        test = rng.normal(0.0, 1.0, size=500)
        test[250] += 10.0  # a clear point anomaly
        det = SRCNNDetector(epochs=60).fit(train)
        scores = np.asarray(det.detect(test)["scores"])
        # the spike neighbourhood should be among the most anomalous points
        assert scores[248:253].max() >= np.quantile(scores, 0.9)


class TestScoreIsNotACalibratedProbability:
    """``pos_weight`` offsets the logit, so 0.5 is not the sigmoid's midpoint.

    The BCE objective weights the rare positive class by
    ``(1 - inject_ratio) / inject_ratio``.  At the optimum
    ``w * p * (1 - s) = (1 - p) * s``, so ``logit(s) = log(w) + logit(p)``: the
    trained logit carries a constant ``+log(w)``.  These pin what that costs
    and what undoing it would cost, since the shipped operating point is a
    choice between the two and nothing in the published pipeline fixes it.
    """

    @staticmethod
    def _stationary(seed: int, n: int = 800, phi: float = 0.6) -> np.ndarray:
        rng = np.random.default_rng(seed)
        series = np.zeros(n)
        for t in range(1, n):
            series[t] = phi * series[t - 1] + rng.normal()
        return series

    def test_the_equivalent_calibrated_cut_is_the_weight_ratio(self) -> None:
        detector = SRCNNDetector()
        weight = (1.0 - detector.inject_ratio) / detector.inject_ratio

        assert weight == pytest.approx(19.0)
        # sigmoid(log w) == w / (1 + w)
        assert 1.0 / (1.0 + np.exp(-np.log(weight))) == pytest.approx(weight / (1.0 + weight))
        assert weight / (1.0 + weight) == pytest.approx(0.95)

    def test_an_anomaly_free_series_is_flagged_at_the_measured_rate(self) -> None:
        """0.2725 over five seeds; the prior-corrected logit flags 0.0000."""
        series = self._stationary(7)
        torch.manual_seed(0)
        np.random.seed(0)
        detector = SRCNNDetector().fit(series[:400])

        held_out = series[400:]
        shipped = np.asarray(detector.detect(held_out)["is_anomaly"], dtype=bool)

        saliency = detector._sr._saliency_map(held_out)
        assert detector._model is not None
        with torch.no_grad():
            logits = detector._model(detector._windows(saliency))
        weight = (1.0 - detector.inject_ratio) / detector.inject_ratio
        corrected = torch.sigmoid(logits - float(np.log(weight))).numpy() > 0.5

        assert shipped.mean() > 0.1, "the shipped cut flags a sizeable share of clean data"
        assert corrected.mean() == 0.0
        # A monotone logit shift cannot reorder anything.
        scores = np.asarray(detector.detect(held_out)["scores"], dtype=float)
        shifted = torch.sigmoid(logits - float(np.log(weight))).numpy().reshape(-1)
        assert np.array_equal(np.argsort(scores, kind="stable"), np.argsort(shifted, kind="stable"))
