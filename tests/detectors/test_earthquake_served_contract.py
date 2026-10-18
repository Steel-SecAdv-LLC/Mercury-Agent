# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""What ``EarthquakeDetector`` serves, on real STEAD traces and a USGS catalog.

Each test fails on the detector as it was before: one NaN sample produced a
NaN confidence with magnitude class "great" and aftershock probability 0.9,
the learned path served a magnitude from an amplitude-free spectrogram plus
an aftershock "probability" of min(0.9, M/10), a DC offset hid a clear event
from STA/LTA, a batch was cut to its first row, a 40 Hz trace ran through the
100 Hz CNN, and a global catalog told everyone to "Take protective action".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.exceptions import InvalidObservationError
from omni_mercury_engine.data_sources.earth_science import USGSEarthquakeSource
from omni_mercury_engine.detectors.geological import disaster_detectors as dd
from omni_mercury_engine.detectors.geological.disaster_detectors import EarthquakeDetector

ROOT = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="module")
def stead() -> dict[str, np.ndarray]:
    """Four real STEAD vertical traces (2 events, 2 noise), 6000 samples at 100 Hz."""
    with np.load(ROOT / "hazard_training" / "seismic" / "stead_z_fixture.npz") as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def served() -> EarthquakeDetector:
    return EarthquakeDetector()


@pytest.fixture(scope="module")
def physics() -> EarthquakeDetector:
    return EarthquakeDetector(load_shipped_weights=False)


def _event(stead: dict[str, np.ndarray]) -> np.ndarray:
    return stead["z"][int(np.flatnonzero(stead["label"] == 1)[0])].astype(np.float64)


def test_one_nan_sample_is_refused(served: EarthquakeDetector, stead: dict[str, Any]) -> None:
    trace = _event(stead)
    trace[100] = np.nan
    with pytest.raises(InvalidObservationError, match="non-finite"):
        served.predict_earthquake(trace)


def test_learned_path_serves_no_magnitude_or_aftershock_figure(
    served: EarthquakeDetector, stead: dict[str, Any]
) -> None:
    result = served.predict_earthquake(_event(stead))
    assert result.detection_method == "learned_seismic_stead"
    assert result.estimated_magnitude is None
    assert result.magnitude_class == "undetermined"
    assert result.aftershock_probability == 0.0


def test_dc_offset_does_not_hide_an_event(
    physics: EarthquakeDetector, stead: dict[str, Any]
) -> None:
    trace = _event(stead)
    plain = physics.predict_earthquake(trace)
    offset = physics.predict_earthquake(trace + 1000.0 * float(np.std(trace)))
    assert offset.p_wave_detected == plain.p_wave_detected
    assert offset.confidence == pytest.approx(plain.confidence, rel=1e-6)


def test_batch_of_traces_is_refused(served: EarthquakeDetector, stead: dict[str, Any]) -> None:
    with pytest.raises(InvalidObservationError, match="one trace per call"):
        served.predict_earthquake(stead["z"][:2])


def test_off_contract_rate_runs_the_physics_path(stead: dict[str, Any]) -> None:
    detector = EarthquakeDetector(sampling_rate=40.0)
    assert detector._neural_trained is True
    result = detector.predict_earthquake(_event(stead))
    assert result.detection_method == "physics_sta_lta"


def test_global_catalog_does_not_order_protective_action() -> None:
    payload = json.loads((ROOT / "live_wiring" / "usgs_earthquakes.json").read_text())
    source = USGSEarthquakeSource(min_magnitude=4.5)

    class _Response:
        def json(self) -> Any:
            return payload

    async def _get(endpoint: str, params: dict[str, Any] | None = None) -> _Response:
        return _Response()

    source._http_get = _get  # type: ignore[method-assign]
    result = EarthquakeDetector(data_source=source).detect_live()
    assert result.estimated_magnitude == pytest.approx(5.8)
    assert result.warning_actions == ["Monitor official earthquake information"]
    assert result.detection_method == "usgs_catalog"


def test_b_value_recovers_a_known_gutenberg_richter_slope() -> None:
    rng = np.random.default_rng(0)
    # b = 1 above M 2.0, binned at 0.1, with an incomplete tail below.
    complete = 2.0 + rng.exponential(1.0 / np.log(10.0), 3000)
    incomplete = rng.uniform(1.0, 2.0, 300)
    mags = np.round(np.concatenate([complete, incomplete]) * 10.0) / 10.0
    b, ci, mc, n = dd.gutenberg_richter_b(mags)
    assert b is not None and ci is not None and mc is not None
    assert ci[0] <= 1.0 <= ci[1]
    assert 2.0 <= mc <= 2.4
    assert n >= 50


def test_b_value_needs_fifty_events_above_completeness() -> None:
    b, ci, _mc, n = dd.gutenberg_richter_b(np.linspace(4.5, 5.8, 12))
    assert b is None and ci is None and n == 12
