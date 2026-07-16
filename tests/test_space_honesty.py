# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Space-cluster detectors must derive from observed physics, never fabricate.

Regression locks for the anti-theater fixes:

* ``SolarStormDetector`` previously fed solar wind/IMF into an UNTRAINED
  ``GeomagneticStormPredictor`` whose Kp output drove the G-scale. Kp now comes
  from the deterministic Boyle-index coupling function of the observed solar
  wind speed and IMF until real weights are loaded.
* ``DisasterPrecursorDetector`` previously multiplied an untrained network's
  output by 9.0 to fabricate a Richter magnitude from EM features. No validated
  physics supports that mapping, so the transparent behaviour is to emit NO magnitude
  (``estimated_magnitude=None``) until trained weights exist -- the real
  Schumann/geomagnetic/ionospheric/seismic correlation paths keep working.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from omni_mercury_engine.space.disaster_precursor_detector import DisasterPrecursorDetector
from omni_mercury_engine.space.solar_storm_detector import SolarStormDetector


def _kp(det: SolarStormDetector, v: float, bz: float, by: float = 0.0) -> dict[str, Any]:
    return det._predict_geomagnetic_storm(
        {"solar_wind_speed_km_s": v, "bz_imf_nt": bz, "by_imf_nt": by}
    )


class TestSolarStormHonesty:
    """Boyle-index physics contract on the ``load_shipped_weights=False`` lane.

    The default detector serves the ratified ``solar_storm_geomag`` winner (its
    32-feature predictor is exercised elsewhere); these tests pin the disclosed
    deterministic physics fallback, so they construct the physics configuration
    explicitly.
    """

    def test_default_detector_serves_the_shipped_winner(self) -> None:
        assert SolarStormDetector()._neural_trained is True

    def test_physics_configuration_is_untrained(self) -> None:
        assert SolarStormDetector(load_shipped_weights=False)._neural_trained is False

    def test_untrained_path_ignores_neural_model_weights(self) -> None:
        """Obliterating the NN's weights must not change the Kp — physics path."""
        det = SolarStormDetector(load_shipped_weights=False)
        data = {"magnetosphere_data": {"solar_wind_speed_km_s": 600, "bz_imf_nt": -10}}
        before = det.predict_solar_storm(dict(data)).kp_index
        assert det.geomag_predictor is not None
        with torch.no_grad():
            for p in det.geomag_predictor.parameters():
                p.mul_(0).add_(5.0)
        after = det.predict_solar_storm(dict(data)).kp_index
        assert before == after

    def test_boyle_kp_is_deterministic(self) -> None:
        det = SolarStormDetector(load_shipped_weights=False)
        assert _kp(det, 550, -10)["kp_index"] == _kp(det, 550, -10)["kp_index"]

    def test_boyle_kp_matches_storm_phenomenology(self) -> None:
        """Quiet→G0, extreme driving→G5, and Kp is monotonic in driving strength."""
        det = SolarStormDetector(load_shipped_weights=False)
        quiet = _kp(det, 400, 2.0)
        extreme = _kp(det, 800, -20.0)
        assert quiet["storm_level"] == "none" and quiet["kp_index"] < 2.0
        assert extreme["storm_level"] == "extreme" and extreme["kp_index"] >= 8.5
        kps = [_kp(det, v, bz)["kp_index"] for v, bz in [(450, -5), (550, -10), (700, -15)]]
        assert kps == sorted(kps)

    def test_southward_northward_imf_asymmetry(self) -> None:
        """Southward Bz couples, northward does not — the physics an untrained
        NN cannot know. Same |B|, same speed: southward must yield far higher Kp."""
        det = SolarStormDetector(load_shipped_weights=False)
        south = _kp(det, 800, -20.0)["kp_index"]
        north = _kp(det, 800, +20.0)["kp_index"]
        assert south >= north + 4.0

    def test_geomag_result_declares_physics_method(self) -> None:
        det = SolarStormDetector(load_shipped_weights=False)
        assert _kp(det, 550, -10)["method"] == "physics_boyle_index"


class TestDisasterPrecursorHonesty:
    @staticmethod
    def _payload(rng: np.random.Generator) -> dict[str, object]:
        t = np.arange(1000) / 100.0
        return {
            "elf_signal": rng.normal(0, 1, 1000) + np.sin(2 * np.pi * 7.83 * t),
            "em_features": rng.normal(0, 1, 64).astype(np.float32),
            "seismic_data": np.array([4.5, 5.0, 5.2]),
        }

    def test_no_magnitude_is_fabricated_when_untrained(self) -> None:
        det = DisasterPrecursorDetector()
        result = det.detect_disaster_precursor(self._payload(np.random.default_rng(0)))
        assert result.estimated_magnitude is None

    def test_correlation_paths_still_function(self) -> None:
        """Refusing to fabricate a magnitude must not kill real precursor detection."""
        det = DisasterPrecursorDetector()
        result = det.detect_disaster_precursor(self._payload(np.random.default_rng(0)))
        assert result.risk_level in {"low", "moderate", "high", "critical"}
        assert result.schumann_anomaly is not None

    def test_direct_neural_call_is_guarded(self) -> None:
        det = DisasterPrecursorDetector()
        with pytest.raises(RuntimeError, match=r"untrained"):
            det._predict_earthquake(np.random.default_rng(1).normal(0, 1, 64).astype(np.float32))

    def test_untrained_result_is_independent_of_network_weights(self) -> None:
        det = DisasterPrecursorDetector()
        payload = self._payload(np.random.default_rng(2))
        before = det.detect_disaster_precursor(dict(payload))
        assert det.earthquake_analyzer is not None
        with torch.no_grad():
            for p in det.earthquake_analyzer.parameters():
                p.mul_(0).add_(3.0)
        after = det.detect_disaster_precursor(dict(payload))
        assert before.estimated_magnitude == after.estimated_magnitude is None
        assert before.confidence == after.confidence
