# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hazard adapters: the engine contract over the hazard detectors' own APIs.

Each test drives an adapter with physically meaningful rows and asserts the
wrapped detector's OWN outputs come through — scores on the detectors' cited
scales, decisions from their released logic — and that the batch contract
(schema refusal, non-finite refusal, feature alignment) holds. These fail on
the tree before the adapters existed: the hazard detectors registered in the
manifest without ``extract_features``/``predict``, so the registry's feature
extraction skipped them silently.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import InvalidObservationError
from omni_mercury_engine.detectors.hazard_adapter import (
    FinancialStressAdapter,
    FloodPrecipitationAdapter,
    GICMinuteSeriesAdapter,
    HurricanePressureAdapter,
    SolarWindGeomagAdapter,
    TornadoConditionsAdapter,
    XrayFlareAdapter,
)

QUIET_SOUNDING = [200.0, -50.0, 50.0, 10.0, 1500.0]
SUPERCELL_SOUNDING = [3500.0, -30.0, 450.0, 60.0, 800.0]


class TestBatchContract:
    """The shared row-batch contract, shown on one representative adapter."""

    def test_is_a_base_detector(self) -> None:
        adapter = TornadoConditionsAdapter()
        assert isinstance(adapter, BaseDetector)

    def test_fit_validates_schema_and_trains_nothing(self) -> None:
        adapter = TornadoConditionsAdapter()
        assert adapter.fit(np.array([QUIET_SOUNDING]))._is_fitted is True

    def test_wrong_width_refused(self) -> None:
        with pytest.raises(InvalidObservationError, match="declared columns"):
            TornadoConditionsAdapter().detect(np.zeros((3, 4)))

    def test_non_finite_rows_refused_not_graded(self) -> None:
        bad = np.array([QUIET_SOUNDING, [np.nan] * 5])
        with pytest.raises(InvalidObservationError, match="not measurements"):
            TornadoConditionsAdapter().detect(bad)

    def test_detect_contract_keys_and_alignment(self) -> None:
        result = TornadoConditionsAdapter().detect(np.array([QUIET_SOUNDING, SUPERCELL_SOUNDING]))
        assert set(result) >= {"anomaly_score", "is_anomaly", "scores"}
        assert result["scores"].shape == (2,)
        assert result["anomaly_score"] == result["scores"].max()

    def test_features_align_with_declared_names(self) -> None:
        adapter = TornadoConditionsAdapter()
        features = adapter.extract_features(np.array([QUIET_SOUNDING, SUPERCELL_SOUNDING]))
        assert features.shape == (2, len(adapter.FEATURE_NAMES))
        assert np.all(np.isfinite(features))

    def test_predict_serves_per_row_scores(self) -> None:
        result = TornadoConditionsAdapter().predict(np.array([QUIET_SOUNDING, SUPERCELL_SOUNDING]))
        assert result["anomaly_scores"].shape == (2,)


class TestTornadoConditions:
    def test_supercell_outranks_quiet_with_released_semantics(self) -> None:
        result = TornadoConditionsAdapter().detect(np.array([QUIET_SOUNDING, SUPERCELL_SOUNDING]))
        # The wrapped detector's released decision needs more than the
        # atmospheric path (radar / pressure indicators), so a supercell
        # sounding alone reads confidence 0.25, not tornado_likely.
        assert result["scores"][0] == 0.0
        assert result["scores"][1] == pytest.approx(0.25)
        assert result["is_anomaly"] is False

    def test_features_echo_the_accepted_measurements(self) -> None:
        features = TornadoConditionsAdapter().extract_features(np.array([SUPERCELL_SOUNDING]))
        confidence, likely, cape, helicity = features[0]
        assert (cape, helicity) == (3500.0, 450.0)
        assert likely == 0.0
        assert confidence == pytest.approx(0.25)


class TestHurricanePressure:
    def test_920_mb_is_a_detected_cyclone(self) -> None:
        result = HurricanePressureAdapter().detect(
            np.array([[1013.0, 1013.0, 26.0], [920.0, 1010.0, 30.0]])
        )
        assert result["scores"][0] == 0.0
        assert result["scores"][1] == pytest.approx(0.7)
        assert result["is_anomaly"] is True

    def test_wind_feature_is_the_atkinson_holliday_value(self) -> None:
        features = HurricanePressureAdapter().extract_features(np.array([[920.0, 1010.0, 30.0]]))
        wind_kt = features[0][2]
        assert wind_kt == pytest.approx(6.7 * (1010.0 - 920.0) ** 0.644, rel=1e-6)


class TestFloodPrecipitation:
    def test_heavy_rain_outranks_dry(self) -> None:
        result = FloodPrecipitationAdapter().detect(
            np.array([[0.0, 0.0, 0.1, 0.0], [1.5, 4.0, 6.0, 3.0]])
        )
        assert result["scores"][0] == 0.0
        assert result["scores"][1] > 0.0

    def test_feature_echoes_the_24h_depth(self) -> None:
        features = FloodPrecipitationAdapter().extract_features(np.array([[1.5, 4.0, 6.0, 3.0]]))
        assert features[0][2] == 6.0


class TestFinancialStress:
    def test_crash_and_banking_stress_is_an_imminent_systemic_crisis(self) -> None:
        calm = [0.2, 15.0, 1.0, 80.0, 1.0, 1.5]
        crisis = [-9.0, 75.0, 4.0, 600.0, 9.0, 0.4]
        result = FinancialStressAdapter().detect(np.array([calm, crisis]))
        assert result["scores"][0] == 0.0
        assert result["scores"][1] == pytest.approx(0.8)
        assert result["is_anomaly"] is True
        features = FinancialStressAdapter().extract_features(np.array([crisis]))
        _, crisis_imminent, market_crash, banking_stress = features[0]
        assert (crisis_imminent, market_crash, banking_stress) == (1.0, 1.0, 1.0)


class TestSolarWindGeomag:
    def test_score_is_the_noaa_g_level_over_five(self) -> None:
        result = SolarWindGeomagAdapter().detect(np.array([[380.0, 1.0], [800.0, -25.0]]))
        assert result["scores"][0] == 0.0  # G0
        assert result["scores"][1] * 5 == int(result["scores"][1] * 5) > 0
        assert result["is_anomaly"] is True  # G >= 1 is NOAA's storm

    def test_features_carry_the_predicted_kp(self) -> None:
        features = SolarWindGeomagAdapter().extract_features(np.array([[800.0, -25.0]]))
        kp, g_level, confidence = features[0]
        assert 0.0 <= kp <= 9.0 and g_level >= 1 and 0.0 <= confidence <= 1.0


class TestXrayFlare:
    def test_score_is_the_noaa_r_level_over_five(self) -> None:
        result = XrayFlareAdapter().detect(np.array([[5e-7], [5e-4]]))
        assert result["scores"][0] == 0.0  # B5: below R1
        assert result["scores"][1] == pytest.approx(3 / 5)  # X5 is R3
        assert result["is_anomaly"] is True


class TestGICMinuteSeries:
    @staticmethod
    def _quiet() -> np.ndarray:
        return np.column_stack([20000 + np.linspace(0, 5, 60), 1000 + np.linspace(0, 3, 60)])

    def test_quiet_series_is_low_tier(self) -> None:
        result = GICMinuteSeriesAdapter().detect(self._quiet())
        assert result["anomaly_score"] == 0.0
        assert result["is_anomaly"] is False

    def test_500_nt_per_min_is_the_severe_tier(self) -> None:
        storm = self._quiet()
        storm[30:, 0] += np.cumsum(np.full(30, 500.0))
        adapter = GICMinuteSeriesAdapter()
        result = adapter.detect(storm)
        assert result["anomaly_score"] == 1.0  # severe / 3 tiers
        assert result["is_anomaly"] is True
        features = adapter.extract_features(storm)
        assert features.shape == (60, 3)
        assert features[0][0] == pytest.approx(500.0, rel=1e-2)  # peak dB/dt
        assert np.all(features[:, 2] == 3.0)  # tier repeats over the window

    def test_cadence_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="cadence_s"):
            GICMinuteSeriesAdapter(config={"cadence_s": 0.0})


class TestMutationClosedGaps:
    """Boundary and shape cases the first mutation run proved untested.

    A 45-mutant sweep of hazard_adapter.py against this file measured a
    66.7% kill rate; each test here kills real survivors (decision
    boundaries mutated from >= to >, the market/banking column split, 1-D
    input handling, the cadence guard) and one caught a genuine crash:
    float(None) on the sustained metric for windows under 10 minutes.
    """

    def test_one_dimensional_row_is_one_sample(self) -> None:
        adapter = TornadoConditionsAdapter()
        flat = adapter.detect(np.array(SUPERCELL_SOUNDING))
        square = adapter.detect(np.array([SUPERCELL_SOUNDING]))
        assert flat["scores"].shape == (1,)
        np.testing.assert_array_equal(flat["scores"], square["scores"])

    def test_even_width_one_dimensional_row_is_still_one_sample(self) -> None:
        # Six columns divide evenly, so a wrong reshape would silently make
        # two bogus three-column rows instead of erroring; pin one sample.
        crisis = np.array([-9.0, 75.0, 4.0, 600.0, 9.0, 0.4])
        result = FinancialStressAdapter().detect(crisis)
        assert result["scores"].shape == (1,)
        assert result["scores"][0] == pytest.approx(0.8)

    def test_liquidity_shortage_alone_is_banking_stress_not_crisis(self) -> None:
        calm_market_dry_liquidity = [0.2, 15.0, 1.0, 80.0, 1.0, 0.05]
        adapter = FinancialStressAdapter()
        features = adapter.extract_features(np.array([calm_market_dry_liquidity]))
        confidence, crisis_imminent, market_crash, banking_stress = features[0]
        assert (crisis_imminent, market_crash, banking_stress) == (0.0, 0.0, 1.0)
        assert confidence == pytest.approx(0.8)

    def test_exactly_g1_is_a_storm(self) -> None:
        result = SolarWindGeomagAdapter().detect(np.array([[650.0, -15.0]]))
        assert result["scores"][0] == pytest.approx(1 / 5)  # exactly G1
        assert result["is_anomaly"] is True  # G1 IS a storm (NOAA), not G2+

    def test_m1_flux_is_exactly_r1_and_c9_is_not(self) -> None:
        result = XrayFlareAdapter().detect(np.array([[1e-5], [9e-6]]))
        assert result["scores"][0] == pytest.approx(1 / 5)  # M1.0 is R1
        assert result["scores"][1] == 0.0  # C9 is below R1
        assert result["is_anomaly"] is True

    def test_100_nt_per_min_is_exactly_the_moderate_tier(self) -> None:
        series = np.column_stack([20000 + np.zeros(60), 1000 + np.zeros(60)])
        series[30:, 0] += np.cumsum(np.full(30, 100.0))
        result = GICMinuteSeriesAdapter().detect(series)
        assert result["anomaly_score"] == pytest.approx(1 / 3)  # moderate
        assert result["is_anomaly"] is True  # >= 90 nT/min acts, not > moderate

    def test_second_cadence_is_averaged_and_short_sustained_is_nan(self) -> None:
        quiet_3_minutes = np.column_stack([20000 + np.zeros(180), 1000 + np.zeros(180)])
        adapter = GICMinuteSeriesAdapter(config={"cadence_s": 1.0})
        result = adapter.detect(quiet_3_minutes)
        assert result["anomaly_score"] == 0.0 and result["is_anomaly"] is False
        features = adapter.extract_features(quiet_3_minutes)
        # Regression: float(None) crashed here. The 10-minute rolling metric
        # is unmeasured on a 3-minute window: NaN, never a quiet zero.
        assert np.isfinite(features[0][0]) and np.isnan(features[0][1])
