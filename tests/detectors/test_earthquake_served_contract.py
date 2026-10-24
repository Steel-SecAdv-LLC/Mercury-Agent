# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What ``EarthquakeDetector`` serves, on real STEAD traces and a USGS catalog.

Each test fails on the detector as it was before: one NaN sample produced a
NaN confidence with magnitude class "great" and aftershock probability 0.9,
the learned path served a magnitude from an amplitude-free spectrogram plus
an aftershock "probability" of min(0.9, M/10), a DC offset hid a clear event
from STA/LTA, a batch was cut to its first row, a 40 Hz trace ran through the
100 Hz CNN, and a global catalog told everyone to "Take protective action".
A magnitude now comes from the IASPEI standard on a calibrated amplitude, and
an aftershock probability from Reasenberg-Jones given regional parameters. A
new result field no longer shifts the released positional order.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta
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
    return np.asarray(stead["z"][int(np.flatnonzero(stead["label"] == 1)[0])], dtype=np.float64)


def test_one_nan_sample_is_a_gap_and_the_trace_is_graded(
    served: EarthquakeDetector, stead: dict[str, Any]
) -> None:
    """Regression: one NaN sample refused a whole recorded STEAD trace."""
    trace = _event(stead)
    trace[100] = np.nan
    result = served.predict_earthquake(trace)
    assert result.partial is True
    assert f"samples [101, {trace.size}) of {trace.size}" in result.set_aside[0]["reason"]
    assert np.isfinite(result.confidence)


def test_all_nan_trace_is_refused(served: EarthquakeDetector, stead: dict[str, Any]) -> None:
    with pytest.raises(InvalidObservationError, match="no finite sample"):
        served.predict_earthquake(np.full_like(_event(stead), np.nan))


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

    source._http_get = _get  # type: ignore[assignment, method-assign, unused-ignore]
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


def test_iaspei_ml_reproduces_richters_anchor() -> None:
    # Richter: ML 3 is 1 mm on a Wood-Anderson (measured gain 2080) at 100 km.
    assert dd.local_magnitude(1e6 / 2080.0, 100.0) == pytest.approx(3.0, abs=0.01)


def test_calibrated_amplitude_gives_a_magnitude_on_either_path(
    served: EarthquakeDetector, physics: EarthquakeDetector, stead: dict[str, Any]
) -> None:
    info = {"wood_anderson_amplitude_nm": 2.0e5, "hypocentral_distance_km": 40.0}
    expected = dd.local_magnitude(2.0e5, 40.0)
    for det in (served, physics):
        result = det.predict_earthquake(_event(stead), station_info=info)
        assert result.estimated_magnitude == pytest.approx(expected)
        assert result.magnitude_class == "moderate"  # ML 5.1


def test_amplitude_without_distance_is_refused(
    physics: EarthquakeDetector, stead: dict[str, Any]
) -> None:
    with pytest.raises(InvalidObservationError, match="hypocentral_distance_km"):
        physics.predict_earthquake(_event(stead), station_info={"wood_anderson_amplitude_nm": 1e4})


@pytest.mark.parametrize(
    ("mainshock", "expected"),
    [
        # 10^(a + b(M - 5)) * [(c)^(1-p) - (7 + c)^(1-p)] / (p - 1), P = 1 - exp(-N)
        (6.0, 0.5944),
        (7.0, 0.9993),
    ],
)
def test_reasenberg_jones_generic_california(mainshock: float, expected: float) -> None:
    p = dd.reasenberg_jones_probability(mainshock, dd.REASENBERG_JONES_GENERIC_CALIFORNIA)
    assert p == pytest.approx(expected, abs=1e-4)


def test_aftershock_forecast_needs_regional_parameters(stead: dict[str, Any]) -> None:
    info = {"wood_anderson_amplitude_nm": 2.0e5, "hypocentral_distance_km": 40.0}
    plain = EarthquakeDetector(load_shipped_weights=False).predict_earthquake(
        _event(stead), station_info=info
    )
    assert plain.aftershock_probability == 0.0  # not estimated
    regional = EarthquakeDetector(
        load_shipped_weights=False,
        aftershock_parameters=dd.REASENBERG_JONES_GENERIC_CALIFORNIA,
    ).predict_earthquake(_event(stead), station_info=info)
    assert regional.earthquake_detected is True
    assert regional.aftershock_probability == pytest.approx(
        dd.reasenberg_jones_probability(
            dd.local_magnitude(2.0e5, 40.0), dd.REASENBERG_JONES_GENERIC_CALIFORNIA
        )
    )


def test_live_aftershock_window_starts_at_as_of() -> None:
    payload = json.loads((ROOT / "live_wiring" / "usgs_earthquakes.json").read_text())

    class _Response:
        def json(self) -> Any:
            return payload

    async def _get(endpoint: str, params: dict[str, Any] | None = None) -> _Response:
        return _Response()

    def _detector() -> EarthquakeDetector:
        source = USGSEarthquakeSource(min_magnitude=4.5)
        source._http_get = _get  # type: ignore[assignment, method-assign, unused-ignore]
        return EarthquakeDetector(
            data_source=source, aftershock_parameters=dd.REASENBERG_JONES_GENERIC_CALIFORNIA
        )

    first = _detector().detect_live()
    assert first.live_context is not None
    strongest = first.live_context["strongest_event"]
    t0 = datetime.fromisoformat(str(strongest["time"]).replace("Z", "+00:00"))
    at_event = _detector().detect_live(as_of=t0)
    week_later = _detector().detect_live(as_of=t0 + timedelta(days=7))
    assert at_event.aftershock_probability == pytest.approx(
        dd.reasenberg_jones_probability(5.8, dd.REASENBERG_JONES_GENERIC_CALIFORNIA)
    )
    # Omori decay: the following week is less likely than the first.
    assert 0.0 < week_later.aftershock_probability < at_event.aftershock_probability


#: EarthquakePredictionResult fields in their released (origin/main) order.
RELEASED_EARTHQUAKE_FIELDS = [
    "earthquake_detected",
    "confidence",
    "estimated_magnitude",
    "magnitude_class",
    "p_wave_detected",
    "s_wave_detected",
    "p_wave_arrival_time",
    "s_wave_arrival_time",
    "epicenter_distance_km",
    "depth_km",
    "resonance_score",
    "spectral_anomalies",
    "warning_actions",
    "aftershock_probability",
    "source_id",
    "data_provenance",
    "live_context",
    "diagnostics",
]


def test_released_positional_field_order_is_kept() -> None:
    names = [f.name for f in dataclasses.fields(dd.EarthquakePredictionResult)]
    assert names[: len(RELEASED_EARTHQUAKE_FIELDS)] == RELEASED_EARTHQUAKE_FIELDS
    positional = dd.EarthquakePredictionResult(
        True, 0.9, 5.0, "moderate", True, False, 1.0, None, None, None, 0.0, [], [], 0.0, "usgs"
    )
    assert positional.source_id == "usgs"
    assert positional.detection_method == "physics_sta_lta"
