# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What ``VolcanicEruptionDetector.predict_eruption`` serves.

Each test fails on the detector as it was before: the same observation gave a
different confidence depending on what had been scored earlier (HMM state
carried across calls and volcanoes), a bare array returned a silent
all-clear, baseline gas raised the forecast, a forecast eruption could carry
the alert level "normal", VEI / eruption type / time to eruption were read
off precursor strength, WATCH advised evacuation and promised an eruption
within 24 h, ELF "correlation" added to the decision, an inverse-rate fit
the data contradict read "eruption in 0 h", and a new field shifted the
released positional order.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError
from omni_mercury_engine.detectors.geological import volcanic
from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector


def _swarm() -> np.ndarray:
    """A moderate swarm (4% of samples elevated): the forecast runs unsaturated."""
    x = np.random.default_rng(4).normal(0.0, 1.0, 2000)
    x[::25] *= 8.0
    return x


@pytest.fixture()
def physics() -> VolcanicEruptionDetector:
    return VolcanicEruptionDetector(load_shipped_weights=False)


def _unrest() -> dict[str, Any]:
    return {"seismic_sequence": _swarm(), "insar_data": {"vertical_displacement_cm": 6.0}}


def test_same_observation_same_answer_whatever_came_before(
    physics: VolcanicEruptionDetector,
) -> None:
    # Old code: 0.5588, 0.5608, 0.5639, ... on the identical observation.
    confidences = {physics.predict_eruption(_unrest()).confidence for _ in range(6)}
    assert len(confidences) == 1


def test_persisting_hmm_state_is_an_explicit_choice() -> None:
    sequential = VolcanicEruptionDetector(load_shipped_weights=False, persist_hmm_state=True)
    first = sequential.predict_eruption(_unrest())
    for _ in range(5):
        sequential.predict_eruption(_unrest())
    later = sequential.predict_eruption(_unrest())
    assert later.confidence > first.confidence


def test_bare_array_is_refused(physics: VolcanicEruptionDetector) -> None:
    bare_array: Any = np.ones((10, 10))  # deliberately outside the declared type
    with pytest.raises(InputModalityError, match="mapping"):
        physics.predict_eruption(bare_array)


def test_non_finite_thermal_pixel_is_refused(physics: VolcanicEruptionDetector) -> None:
    thermal = {"brightness_temperature_k": np.array([300.0, np.inf])}
    with pytest.raises(InvalidObservationError, match="non-finite"):
        physics.predict_eruption({"thermal_data": thermal})


def test_baseline_gas_adds_nothing(physics: VolcanicEruptionDetector) -> None:
    gas = {"so2_tons_per_day": 100.0, "co2_tons_per_day": 500.0}  # the baselines
    plain = physics.predict_eruption(_unrest())
    with_gas = physics.predict_eruption({**_unrest(), "gas_data": gas})
    assert with_gas.confidence == pytest.approx(plain.confidence)


def test_forecast_eruption_is_never_alert_normal(physics: VolcanicEruptionDetector) -> None:
    result = physics.predict_eruption(
        {
            **_unrest(),
            "gas_data": {"so2_tons_per_day": 900.0, "co2_tons_per_day": 2500.0},
            "insar_data": {"vertical_displacement_cm": 25.0},
        }
    )
    assert result.eruption_imminent is True
    assert result.alert_level in ("watch", "warning")
    assert result.forecast_method == "physics"


def test_size_style_and_timing_are_not_forecast(physics: VolcanicEruptionDetector) -> None:
    result = physics.predict_eruption(
        {**_unrest(), "gas_data": {"so2_tons_per_day": 900.0, "co2_tons_per_day": 2500.0}}
    )
    assert result.vei_estimate is None
    assert result.time_to_eruption_hours is None
    assert result.eruption_type in ("no_eruption", "undetermined")


def test_watch_prepares_but_does_not_evacuate(physics: VolcanicEruptionDetector) -> None:
    result = physics.predict_eruption(
        {**_unrest(), "gas_data": {"so2_tons_per_day": 900.0, "co2_tons_per_day": 2500.0}}
    )
    if result.alert_level == "watch":
        assert not any("Evacuate" in rec for rec in result.evacuation_recommendations)
        assert not any("24 hours" in action for action in result.early_warning_actions)


def test_elf_deviation_carries_no_decision_weight(physics: VolcanicEruptionDetector) -> None:
    elf = np.full(256, 11.0)  # far from 7.83 Hz
    plain = physics.predict_eruption(_unrest())
    with_elf = physics.predict_eruption({**_unrest(), "schumann_elf": elf})
    assert with_elf.schumann_elf_correlation is not None
    assert (with_elf.confidence, with_elf.alert_level) == (plain.confidence, plain.alert_level)


def _accelerating_rate(t_h: np.ndarray, failure_h: float, k: float = 50.0) -> np.ndarray:
    """dR/dt = A R^2 integrates to R = k / (t_f - t): 1/R falls linearly to 0 at t_f."""
    return k / (failure_h - t_h)


def test_failure_forecast_recovers_the_failure_time() -> None:
    t = np.arange(0.0, 96.0, 6.0)  # 16 samples over four days
    forecast = volcanic.failure_forecast(t, _accelerating_rate(t, failure_h=120.0))
    assert forecast is not None
    assert forecast["failure_time_h"] == pytest.approx(120.0)
    assert forecast["time_to_failure_h"] == pytest.approx(120.0 - 90.0)
    assert forecast["r_squared"] == pytest.approx(1.0)


def _accelerating_counts(seed: int, edges: np.ndarray, failure_h: float = 120.0) -> np.ndarray:
    """Poisson counts per bin of the rate 400 / (t_f - t)."""
    expected = 400.0 * np.log((failure_h - edges[:-1]) / (failure_h - edges[1:]))
    return np.random.default_rng(seed).poisson(expected)


def test_inverse_rate_least_squares_fails_on_sparse_counts() -> None:
    edges = np.arange(0.0, 102.0, 6.0)
    counts = _accelerating_counts(0, edges)
    # Bell et al. (2013): 1/count scatter hides the trend (R^2 0.13 here).
    assert volcanic.failure_forecast(edges[:-1], np.maximum(counts, 1)) is None


def test_poisson_failure_forecast_on_counts() -> None:
    edges = np.arange(0.0, 102.0, 6.0)
    errors = []
    for seed in range(20):
        forecast = volcanic.failure_forecast_poisson(edges, _accelerating_counts(seed, edges))
        assert forecast is not None
        assert forecast["likelihood_ratio"] > volcanic.FFM_LR_CRITICAL
        errors.append(abs(forecast["failure_time_h"] - 120.0))
    assert float(np.median(errors)) < 6.0


def test_constant_count_rate_is_not_an_acceleration() -> None:
    edges = np.arange(0.0, 102.0, 6.0)
    accepted = sum(
        volcanic.failure_forecast_poisson(
            edges, np.random.default_rng(1000 + seed).poisson(np.full(16, 30.0))
        )
        is not None
        for seed in range(100)
    )
    assert accepted <= 10  # the 5% test level, with sampling slack


def _rate_still_finite_past_the_fitted_failure() -> tuple[np.ndarray, np.ndarray]:
    """1/R falls, then levels off above zero: the line hits 0 at 40.8 h (R^2 0.90)."""
    t = np.arange(0.0, 54.0, 6.0)
    inverse = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.3, 0.2])
    return t, 1.0 / inverse


def test_failure_time_before_the_last_sample_is_no_forecast() -> None:
    t, rate = _rate_still_finite_past_the_fitted_failure()
    # The rate at 48 h is finite, so a failure at 40.8 h is contradicted.
    assert volcanic.failure_forecast(t, rate) is None


def test_detector_serves_no_time_to_eruption_from_a_contradicted_fit(
    physics: VolcanicEruptionDetector,
) -> None:
    t, rate = _rate_still_finite_past_the_fitted_failure()
    result = physics.predict_eruption({"precursor_rate": {"times_h": t, "rate": rate}})
    assert (result.time_to_eruption_hours, result.time_to_eruption_fit) == (None, None)


#: VolcanicPredictionResult fields in their released (origin/main) order.
RELEASED_VOLCANIC_FIELDS = [
    "eruption_imminent",
    "confidence",
    "alert_level",
    "eruption_type",
    "time_to_eruption_hours",
    "vei_estimate",
    "seismic_swarm_detected",
    "thermal_anomaly_detected",
    "gas_flux_anomaly",
    "deformation_detected",
    "schumann_elf_correlation",
    "hazard_zones",
    "ashfall_forecast",
    "lahar_risk",
    "early_warning_actions",
    "evacuation_recommendations",
    "source_id",
    "data_provenance",
    "live_context",
    "diagnostics",
]


def test_released_positional_field_order_is_kept() -> None:
    names = [f.name for f in dataclasses.fields(volcanic.VolcanicPredictionResult)]
    assert names[: len(RELEASED_VOLCANIC_FIELDS)] == RELEASED_VOLCANIC_FIELDS
    positional = volcanic.VolcanicPredictionResult(
        False, 0.1, "normal", "no_eruption", None, None, True
    )
    assert positional.seismic_swarm_detected is True
    assert positional.forecast_method is None


def test_steady_or_declining_rate_has_no_failure_time() -> None:
    t = np.arange(0.0, 48.0, 6.0)
    assert volcanic.failure_forecast(t, np.full(t.size, 10.0)) is None
    assert volcanic.failure_forecast(t, 50.0 / (t + 10.0)) is None


@pytest.mark.parametrize(
    ("times", "rates"),
    [
        ([0, 6, 12, 18, 24], [5, 6, 0, 9, 12]),  # a zero rate has no inverse
        ([0, 6, 6, 18, 24], [5, 6, 7, 9, 12]),  # times must increase
        ([0, 6, 12, 18], [5, 6, 7, 9]),  # too few samples
    ],
)
def test_failure_forecast_refuses_unusable_series(times: list[float], rates: list[float]) -> None:
    with pytest.raises(InvalidObservationError):
        volcanic.failure_forecast(times, rates)


def test_detector_reports_the_failure_forecast() -> None:
    t = np.arange(0.0, 96.0, 6.0)
    det = VolcanicEruptionDetector(load_shipped_weights=False)
    result = det.predict_eruption(
        {"precursor_rate": {"times_h": t, "rate": _accelerating_rate(t, failure_h=120.0)}}
    )
    assert result.time_to_eruption_hours == pytest.approx(30.0)
    assert result.time_to_eruption_fit is not None
    assert result.time_to_eruption_fit["method"] == "ffm_inverse_rate"
    assert result.vei_estimate is None  # VEI is not forecast from precursors
    edges = np.arange(0.0, 102.0, 6.0)
    counted = det.predict_eruption(
        {"precursor_rate": {"bin_edges_h": edges, "counts": _accelerating_counts(3, edges)}}
    )
    assert counted.time_to_eruption_fit is not None
    assert counted.time_to_eruption_fit["method"] == "ffm_poisson_mle"
    assert counted.time_to_eruption_hours is not None
    assert 0.0 < counted.time_to_eruption_hours < 40.0


@pytest.mark.parametrize(
    ("section", "values"),
    [
        ("gas_data", {"so2_flux_tons_day": 5000.0}),  # misspelt: read as baseline
        ("insar_data", {"vertical_cm": 30.0}),  # misspelt: read as 0 cm
        ("thermal_data", {"max_temperature_k": 900.0}),  # misspelt: read as no pixels
    ],
)
def test_misspelt_section_key_is_refused(
    physics: VolcanicEruptionDetector, section: str, values: dict[str, float]
) -> None:
    """A misspelt measurement used to grade as baseline -- a silent all-clear."""
    with pytest.raises(InvalidObservationError, match=r"nothing usable.*unknown key"):
        physics.predict_eruption({section: values})


@pytest.mark.parametrize(
    ("section", "values"),
    [
        ("gas_data", {}),
        ("thermal_data", {"radiant_heat_mw": 50.0}),  # no brightness raster to grade
        ("insar_data", {"deformation_rate_cm_day": 1.0}),  # rate alone decides nothing
        ("precursor_rate", {"times_h": [0.0, 1.0, 2.0]}),  # half a series
    ],
)
def test_section_without_a_measurement_is_refused(
    physics: VolcanicEruptionDetector, section: str, values: dict[str, Any]
) -> None:
    with pytest.raises(InvalidObservationError, match="no measurement"):
        physics.predict_eruption({section: values})


def test_one_measured_gas_is_graded(physics: VolcanicEruptionDetector) -> None:
    """A single measured flux is enough; the other stays at its baseline."""
    result = physics.predict_eruption({"gas_data": {"so2_tons_per_day": 5000.0}})
    assert result.gas_flux_anomaly


@pytest.mark.parametrize("section", ["thermal_data", "gas_data", "insar_data", "precursor_rate"])
def test_non_mapping_section_is_refused(physics: VolcanicEruptionDetector, section: str) -> None:
    """A list section used to fail deep inside with a bare AttributeError."""
    with pytest.raises(InvalidObservationError, match="must be a mapping"):
        physics.predict_eruption({section: [1.0, 2.0]})


def test_non_mapping_record_is_refused(physics: VolcanicEruptionDetector) -> None:
    not_a_mapping: Any = [1.0, 2.0]  # deliberately outside the declared type
    with pytest.raises(InputModalityError, match="mapping"):
        physics.predict_eruption(not_a_mapping)


@pytest.mark.parametrize(
    ("extra", "where"),
    [
        ({"insar_data": {"vertical_cm": 30.0}}, "insar_data.vertical_cm"),
        ({"insar_data": {"vertical_displacement_cm": float("nan")}}, "insar_data"),
        ({"thermal_data": {"brightness_temperature_k": [300.0, 9000.0]}}, "thermal_data"),
        ({"seismic_sequence": [1.0, float("nan"), 3.0]}, "seismic_sequence"),
        ({"insar_data": [1.0]}, "insar_data"),
        ({"radon_data": {"bq": 3.0}}, "radon_data"),
        (
            {"precursor_rate": {"times_h": [1.0, 2.0, 3.0, 4.0, 5.0], "rate": [1.0, -1.0]}},
            "precursor_rate",
        ),
    ],
)
def test_usable_streams_are_graded_beside_an_unusable_one(
    physics: VolcanicEruptionDetector, extra: dict[str, Any], where: str
) -> None:
    """Regression: one unusable stream refused a measured gas anomaly."""
    result = physics.predict_eruption({"gas_data": {"so2_tons_per_day": 5000.0}, **extra})
    assert result.gas_flux_anomaly
    assert result.partial is True
    assert any(entry["where"] == where for entry in result.set_aside), result.set_aside


def test_clean_record_is_not_partial(physics: VolcanicEruptionDetector) -> None:
    result = physics.predict_eruption({"gas_data": {"so2_tons_per_day": 5000.0}})
    assert result.partial is False and result.set_aside == []


@pytest.mark.parametrize(
    "record", [{}, {"schumann_elf": [7.8, 7.9, 7.8]}, {"metadata": {"name": "x"}}]
)
def test_record_without_a_deciding_stream_is_refused(
    physics: VolcanicEruptionDetector, record: dict[str, Any]
) -> None:
    """Nothing that decides used to grade as a silent "normal"."""
    with pytest.raises(InvalidObservationError, match="no monitoring stream"):
        physics.predict_eruption(record)
