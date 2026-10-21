# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What ``TornadoDetector.predict_tornado`` serves, on the real Moore EF5 sector.

Each test fails on the detector as it was before: masked radar gates (NaN)
reached the trained LSTM, which then returned NaN and detected nothing on a
real EF5 sector; the network's probability was overwritten by an indicator
count; and location, injected noise and the code complexity of a lambda all
raised the decision.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.exceptions import InvalidObservationError
from omni_mercury_engine.detectors.geological import tornado_detector as td
from omni_mercury_engine.ml.hazard_training import tornado_radar

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "hazard_training" / "tornado"
) / "moore_ef5_ktlx_sector.npz"


@pytest.fixture(scope="module")
def sector() -> np.ndarray:
    """KTLX lowest-sweep velocity sector around the 2013-05-20 Moore EF5 report."""
    with np.load(FIXTURE, allow_pickle=False) as z:
        window = np.asarray(z["expected_sector"], dtype=np.float64)
    assert 0.3 < float(np.isfinite(window).mean()) < 1.0  # real, partly masked
    return window


@pytest.fixture(scope="module")
def served() -> td.TornadoDetector:
    """The default-constructed detector, serving the shipped checkpoint."""
    return td.TornadoDetector()


def test_masked_real_sector_is_detected_as_served(
    served: td.TornadoDetector, sector: np.ndarray
) -> None:
    result = served.predict_tornado({"radar_sequence": sector})
    assert np.isfinite(result.confidence)
    assert result.mesocyclone_detected is True
    assert result.tornado_likely is True


def test_served_input_matches_the_training_preprocessing(
    served: td.TornadoDetector, sector: np.ndarray
) -> None:
    as_trained = np.where(np.isfinite(sector), sector, 0.0)
    raw = served.predict_tornado({"radar_sequence": sector})
    cleaned = served.predict_tornado({"radar_sequence": as_trained})
    assert raw.confidence == cleaned.confidence


def test_confidence_is_the_trained_mesocyclone_probability(
    served: td.TornadoDetector, sector: np.ndarray
) -> None:
    result = served.predict_tornado({"radar_sequence": sector})
    assert served._meso_prob_threshold is not None
    assert result.confidence >= served._meso_prob_threshold
    assert result.confidence not in (0.5, 1.0)  # not an indicator count


def test_non_evidence_inputs_carry_no_decision_weight(served: td.TornadoDetector) -> None:
    base = {"atmospheric_data": {"cape": 500.0}}
    extras = {
        **base,
        "location": {"state": "OK"},
        "signal_data": np.random.default_rng(1).normal(0.0, 3.0, 512),
        "observed_data": [1.0, 2.0, 3.0],
    }
    a, b = served.predict_tornado(base), served.predict_tornado(extras)
    assert (a.tornado_likely, a.confidence, a.threat_level) == (
        b.tornado_likely,
        b.confidence,
        b.threat_level,
    )
    assert b.tornado_alley_correlation == 0.8  # still reported


def test_mostly_masked_window_is_refused(served: td.TornadoDetector) -> None:
    window = np.full((61, 64), np.nan)
    window[:5] = 10.0
    with pytest.raises(InvalidObservationError, match="finite gates"):
        served.predict_tornado({"radar_sequence": window})


def test_empty_window_is_refused(served: td.TornadoDetector) -> None:
    with pytest.raises(InvalidObservationError, match="empty"):
        served.predict_tornado({"radar_sequence": np.zeros((0, 64))})


def test_finite_gate_floor_matches_the_training_set() -> None:
    assert td.MIN_FINITE_GATE_FRACTION == tornado_radar.MIN_FINITE_FRACTION


class TestAtmosphericAndPressureGuards:
    """Supplied-but-bad inputs are refused, never graded (regression tests)."""

    def test_nan_cape_is_refused_not_graded_low(self) -> None:
        """A NaN CAPE used to flow through the STP and grade 'low'."""
        from omni_mercury_engine.detectors._hazard_input import InvalidObservationError
        from omni_mercury_engine.detectors.geological.tornado_detector import (
            AtmosphericInstabilityAnalyzer,
        )

        analyzer = AtmosphericInstabilityAnalyzer()
        with pytest.raises(InvalidObservationError, match="cape_j_kg"):
            analyzer.analyze_instability({"cape_j_kg": float("nan")})
        # Absent keys keep the documented quiet defaults.
        quiet = analyzer.analyze_instability({})
        assert quiet["tornado_potential"] == "low"

    def test_pascal_series_is_refused_not_read_as_millibar_crash(self) -> None:
        """A pressure series in Pa used to read as a 1300 mb drop ('slight')."""
        import numpy as np

        from omni_mercury_engine.detectors._hazard_input import InvalidObservationError
        from omni_mercury_engine.detectors.geological.tornado_detector import (
            PressureGradientMonitor,
        )

        monitor = PressureGradientMonitor()
        with pytest.raises(InvalidObservationError, match=r"pressure_mb.*mb"):
            monitor.analyze_pressure({"pressure_mb": np.array([101300.0, 100000.0])})
        with pytest.raises(InvalidObservationError):
            monitor.analyze_pressure({"pressure_mb": np.array([1013.0, float("nan")])})
        ok = monitor.analyze_pressure({"pressure_mb": np.array([1013.0, 1008.0])})
        assert bool(ok["rapid_drop_detected"]) is True  # 5 mb drop > 4 mb threshold
