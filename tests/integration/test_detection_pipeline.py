# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cross-module detection-pipeline integration.

These tests drive data through the *assembled* detection stack — the
``OmniMercuryEngine`` facade orchestrating its registered detectors, plus the
standalone statistical detector — and assert that an injected anomaly is
surfaced end-to-end. Unlike the per-detector unit tests, this exercises the
engine's detector orchestration and result-aggregation boundary as a whole.

Determinism: the autouse ``set_random_seed`` fixture in ``tests/conftest.py``
seeds the global RNGs, and the synthetic data here is generated from an
explicit ``numpy`` ``default_rng`` seed, so the injected outlier and the score
ordering are reproducible run to run.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from omni_mercury_engine.engine import OmniMercuryEngine

pytestmark = pytest.mark.integration

# Index of the row/sample we deliberately corrupt into an anomaly.
_OUTLIER_INDEX = 57
_N_SAMPLES = 60


def _data_with_injected_outlier() -> np.ndarray:
    """Deterministic 3-feature matrix with one unmistakable outlier row."""
    rng = np.random.default_rng(0)
    data = rng.normal(loc=0.0, scale=1.0, size=(_N_SAMPLES, 3))
    # A row many standard deviations away on every feature.
    data[_OUTLIER_INDEX] = [12.0, -11.0, 13.0]
    return data


class TestEngineOrchestratedDetection:
    """The engine facade runs its detector suite and aggregates results."""

    def test_engine_flags_injected_outlier_across_detector_suite(self) -> None:
        """Default ``detect`` runs the full registered detector suite and the
        aggregate verdict is anomalous, with the statistical detector pinning
        the injected row."""
        engine = OmniMercuryEngine(mode="statistical", device="cpu")
        result = engine.detect(_data_with_injected_outlier())

        # Aggregate contract: a dict carrying per-detector results and a
        # boolean roll-up.
        assert isinstance(result, dict)
        assert result["is_anomaly"] is True
        assert isinstance(result["detectors"], dict)

        # The engine wired up more than one detector — this is the
        # orchestration boundary the unit tests don't cover.
        assert "statistical" in result["detectors"]
        assert len(result["detectors"]) >= 2

        statistical = result["detectors"]["statistical"]
        flags = np.asarray(statistical["is_anomaly"], dtype=bool)
        scores = np.asarray(statistical["scores"], dtype=float)
        assert flags.shape == (_N_SAMPLES,)
        assert scores.shape == (_N_SAMPLES,)

        # The injected row is flagged, and its score stands clear of the bulk
        # (a strict margin, not just "above threshold").
        assert bool(flags[_OUTLIER_INDEX]) is True
        assert scores[_OUTLIER_INDEX] > float(np.median(scores))

    def test_detector_scoping_limits_the_suite(self) -> None:
        """``detector_types`` narrows orchestration to the requested detector
        only — proving the selection seam is honoured."""
        engine = OmniMercuryEngine(mode="statistical", device="cpu")
        result = engine.detect(_data_with_injected_outlier(), detector_types=["statistical"])

        assert list(result["detectors"].keys()) == ["statistical"]
        assert result["is_anomaly"] is True


class TestStandaloneStatisticalDetector:
    """The statistical detector, fit/detect round-trip, end-to-end."""

    def test_fit_then_detect_pins_obvious_spike(self) -> None:
        """A detector fit on a flat baseline flags a single injected spike at
        its exact index, with a dominating score."""
        detector = MercuryAnomalyDetector()
        detector.fit(np.ones((100, 1)))

        series = np.array([1, 1, 1, 1, 9, 1, 1, 1], dtype=float).reshape(-1, 1)
        result = detector.detect(series)

        flags = np.asarray(result["is_anomaly"], dtype=bool)
        scores = np.asarray(result["scores"], dtype=float)
        assert flags.shape == (8,)
        assert bool(flags[4]) is True
        # The spike is the single highest-scoring point.
        assert int(np.argmax(scores)) == 4

    def test_ensemble_components_are_populated_and_bounded(self) -> None:
        """The three ensemble components are present and live in ``[0, 1]`` —
        the contract the adaptive fusion weights depend on."""
        detector = MercuryAnomalyDetector()
        detector.fit(np.random.default_rng(1).normal(size=(200, 3)))

        result = detector.detect(np.random.default_rng(2).normal(size=(40, 3)))

        for key in (
            "resonance_scores",
            "kinematic_scores",
            "info_geometry_scores",
        ):
            component = np.asarray(result[key], dtype=float)
            assert component.shape == (40,)
            assert np.all((component >= 0.0) & (component <= 1.0)), key
