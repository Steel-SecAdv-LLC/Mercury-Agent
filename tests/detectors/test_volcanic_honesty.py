# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The volcanic detector must forecast from observed physics, never fabricate.

Regression lock for the anti-theater fix: previously ``_forecast_eruption``
synthesised a 128-dim ``randn`` feature vector and fed it to an *untrained*
EruptionForecastModel, so VEI / eruption probability were noise on the common
path. Now the untrained networks are guarded and the forecast is a deterministic
function of the real precursor magnitudes.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector


def _unrest(vertical_cm: float, so2: float, co2: float, radiant_mw: float) -> dict:
    rng = np.random.default_rng(0)
    return {
        "seismic_sequence": np.concatenate([rng.normal(0, 1, 180), np.full(20, 12.0)]),
        "gas_data": {"so2_tons_per_day": so2, "co2_tons_per_day": co2},
        "insar_data": {"vertical_displacement_cm": vertical_cm, "deformation_rate_cm_day": 2.0},
        "thermal_data": {
            "brightness_temperature_k": np.array([300.0, 305.0, 410.0, 420.0]),
            "radiant_heat_mw": radiant_mw,
        },
    }


class TestVolcanicHonesty:
    def test_default_detector_is_untrained(self) -> None:
        assert VolcanicEruptionDetector()._neural_trained is False

    def test_forecast_is_deterministic_no_rng(self) -> None:
        """Identical input → byte-identical forecast (proves no RNG in the path)."""
        det = VolcanicEruptionDetector()
        data = _unrest(25.0, 800.0, 2000.0, 800.0)
        r1 = det.predict_eruption({**data})
        r2 = det.predict_eruption({**data})
        assert (r1.eruption_imminent, r1.confidence, r1.vei_estimate, r1.eruption_type) == (
            r2.eruption_imminent,
            r2.confidence,
            r2.vei_estimate,
            r2.eruption_type,
        )

    def test_untrained_path_ignores_neural_model_weights(self) -> None:
        """With no weights loaded, mutating the NN must not change the forecast.

        The strongest possible proof the untrained network is not consulted:
        randomise its parameters and confirm the result is unchanged.
        """
        det = VolcanicEruptionDetector()
        data = _unrest(25.0, 800.0, 2000.0, 800.0)
        before = det.predict_eruption({**data}).confidence
        with __import__("torch").no_grad():
            for p in det.eruption_model.parameters():
                p.mul_(0).add_(7.0)  # obliterate the weights
        after = det.predict_eruption({**data}).confidence
        assert before == after  # physics path, network irrelevant

    def test_probability_is_monotonic_in_severity(self) -> None:
        """Higher precursor magnitudes yield higher probability (sub-saturation regime)."""
        f = VolcanicEruptionDetector._forecast_eruption_physics
        mild = f({"degassing_index": 1.0, "total_displacement_cm": 2.0})["confidence"]
        moderate = f({"degassing_index": 3.0, "total_displacement_cm": 8.0})["confidence"]
        assert moderate > mild
        assert 0.0 < mild < moderate < 1.0  # neither saturates in this regime

    def test_quiet_volcano_forecasts_no_eruption(self) -> None:
        det = VolcanicEruptionDetector()
        quiet = {
            "gas_data": {"so2_tons_per_day": 100.0, "co2_tons_per_day": 500.0},
            "insar_data": {"vertical_displacement_cm": 0.5, "deformation_rate_cm_day": 0.0},
            "thermal_data": {
                "brightness_temperature_k": np.array([295.0, 296.0, 297.0]),
                "radiant_heat_mw": 0.0,
            },
            "seismic_sequence": np.random.default_rng(1).normal(0, 1, 200),
        }
        r = det.predict_eruption(quiet)
        assert r.eruption_imminent is False
        assert r.vei_estimate in (0, None)

    def test_swarm_physics_flags_amplitude_burst_deterministically(self) -> None:
        rng = np.random.default_rng(2)
        calm = rng.normal(0, 1, 200)
        burst = np.concatenate([rng.normal(0, 1, 150), np.full(50, 15.0)])
        calm_res = VolcanicEruptionDetector._detect_swarm_physics(calm)
        burst_res = VolcanicEruptionDetector._detect_swarm_physics(burst)
        assert burst_res["swarm_detected"] is True
        assert burst_res["confidence"] > calm_res["confidence"]
        # Deterministic: identical input → identical confidence.
        assert (
            VolcanicEruptionDetector._detect_swarm_physics(burst)["confidence"]
            == burst_res["confidence"]
        )

    def test_physics_forecast_noisy_or_bounds(self) -> None:
        """A single critical precursor already implies high probability; empty → 0."""
        assert VolcanicEruptionDetector._forecast_eruption_physics({})["confidence"] == 0.0
        crit = VolcanicEruptionDetector._forecast_eruption_physics({"degassing_index": 5.0})
        assert crit["confidence"] >= 0.99  # degassing_index/5 == 1.0 → noisy-OR saturates
        assert crit["eruption_imminent"] is True
