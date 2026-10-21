# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""What ``TsunamiDetector.predict_tsunami`` serves, on real DART 21419 records.

The records are passed raw, tide included, as a caller would pass them. Each
test fails on the detector as it was before: the tide was reported as the wave
height (a quiet January day read "watch"/"warning" with evacuation zones), a
pure tide, a constant column and a lone spike were all detected, a record in
centimetres read MAJOR, a batch was silently cut to its first row, and
severity and evacuation zones were issued with no tsunami detected.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.exceptions import InvalidObservationError
from omni_mercury_engine.detectors.geological.disaster_detectors import (
    TsunamiDetector,
    TsunamiSeverity,
)
from omni_mercury_engine.ml.hazard_training import tsunami_waveform as tw

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "hazard_training" / "tsunami"
DART_RATE_HZ = 1.0 / tw.SAMPLE_PERIOD_S


def _raw_window(fixture: str, start: dt.datetime, hours: int) -> np.ndarray:
    """Raw DART water-column heights (m) for a gap-free span of the fixture."""
    grid = tw.parse_dart_file(FIXTURES / fixture, "21419", 2011)
    first = tw._slot_of(start, 2011)
    values = grid.values[first : first + hours * tw.SLOTS_PER_HOUR].astype(np.float64)
    assert np.isfinite(values).all()
    return values


@pytest.fixture(scope="module")
def quiet_day() -> np.ndarray:
    """2011-01-02, a quiet day at the Kuril Islands DART buoy (tide only)."""
    return _raw_window(
        "21419t2011_jan01-09_excerpt.txt.gz", dt.datetime(2011, 1, 2, tzinfo=dt.UTC), 24
    )


@pytest.fixture(scope="module")
def detector() -> TsunamiDetector:
    return TsunamiDetector(sampling_rate=DART_RATE_HZ)


def test_quiet_raw_dart_day_is_no_tsunami(detector: TsunamiDetector, quiet_day: np.ndarray) -> None:
    assert np.ptp(quiet_day) > 0.5  # the tide is in the record
    result = detector.predict_tsunami(quiet_day)
    assert result.tsunami_detected is False
    assert result.severity == TsunamiSeverity.NONE.value
    assert result.evacuation_zones == []
    assert result.estimated_wave_height_m < 0.05  # the residual, not the tide


def test_tohoku_wave_height_is_measured_above_the_tide(detector: TsunamiDetector) -> None:
    arrival = dt.datetime(2011, 3, 11, 3, 6, tzinfo=dt.UTC)  # 4 h before the wave
    raw = _raw_window("21419t2011_mar10-12_excerpt.txt.gz", arrival, 24)
    result = detector.predict_tsunami(raw)
    # Same order as the DART-measured amplitude (HazEL 0.54 m), not the tide.
    assert 0.3 < result.estimated_wave_height_m < 0.7
    assert result.waveform_anomaly_score > 10.0


def test_pure_tide_is_no_tsunami() -> None:
    t = np.arange(6 * 60) * 60.0  # 6 h at 1/min, a NOS tsunami-gauge rate
    tide = 0.5 * np.sin(2 * np.pi * t / (12.42 * 3600.0))
    result = TsunamiDetector(sampling_rate=1.0 / 60.0).predict_tsunami(tide)
    assert result.tsunami_detected is False
    assert result.severity == TsunamiSeverity.NONE.value


def test_constant_water_column_has_no_resonance(detector: TsunamiDetector) -> None:
    result = detector.predict_tsunami(np.full(96, 4000.0))
    assert result.tsunami_detected is False
    assert result.resonance_score == 0.0


def test_single_glitch_does_not_evacuate(detector: TsunamiDetector, quiet_day: np.ndarray) -> None:
    glitched = quiet_day.copy()
    glitched[40] += 3.0
    result = detector.predict_tsunami(glitched)
    assert result.tsunami_detected is False
    assert not any("EVACUATE" in action for action in result.warning_actions)
    assert result.estimated_wave_height_m < 0.05


def test_record_in_centimetres_is_refused(detector: TsunamiDetector, quiet_day: np.ndarray) -> None:
    with pytest.raises(InvalidObservationError, match="metres"):
        detector.predict_tsunami(quiet_day * 100.0)


def test_batch_of_records_is_refused(detector: TsunamiDetector, quiet_day: np.ndarray) -> None:
    with pytest.raises(InvalidObservationError, match="one record per call"):
        detector.predict_tsunami(np.stack([quiet_day, quiet_day]))


def test_record_too_short_to_detide_is_refused() -> None:
    with pytest.raises(InvalidObservationError, match="30 min"):
        TsunamiDetector(sampling_rate=1.0 / 60.0).predict_tsunami(np.zeros(10))


def test_detected_wave_gets_an_nws_height_category() -> None:
    rng = np.random.default_rng(3)
    t = np.arange(4096)
    record = rng.normal(0.0, 0.05, t.size) + 1.5 * np.sin(2 * np.pi * t / 600.0) * (t > 2000)
    result = TsunamiDetector(sampling_rate=1.0).predict_tsunami(record)
    assert result.tsunami_detected is True
    assert result.severity == TsunamiSeverity.WARNING.value
    assert result.evacuation_zones


@pytest.mark.parametrize(
    ("height_m", "category"),
    [(0.2, "information"), (0.5, "advisory"), (0.99, "advisory"), (1.2, "warning"), (3.5, "major")],
)
def test_height_categories_follow_nws(height_m: float, category: str) -> None:
    assert TsunamiDetector()._determine_severity(height_m) == category


def test_default_rate_is_the_dart_standard_mode() -> None:
    assert TsunamiDetector().sampling_rate == pytest.approx(DART_RATE_HZ)
