# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""What ``VolcanicEruptionDetector.predict_eruption`` serves.

Each test fails on the detector as it was before: the same observation gave a
different confidence depending on what had been scored earlier (HMM state
carried across calls and volcanoes), a bare array returned a silent
all-clear, baseline gas raised the forecast, a forecast eruption could carry
the alert level "normal", VEI / eruption type / time to eruption were read
off precursor strength, WATCH advised evacuation and promised an eruption
within 24 h, and ELF "correlation" added to the decision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError
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
    with pytest.raises(InputModalityError, match="mapping"):
        physics.predict_eruption(np.ones((10, 10)))  # type: ignore[arg-type]


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
