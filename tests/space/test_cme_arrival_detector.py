# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the CME arrival detector (Gopalswamy ESA + Vršnak DBM).

Physics values are hand-computed from the published model equations and
cross-checked against independent numerical integration; ingestion tests run
against a recorded real DONKI CMEAnalysis fixture (May 2024 Gannon storm CME
sequence) with provenance headers.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from omni_mercury_engine.space.cme_arrival_detector import (
    AU_KM,
    DONKI_R0_KM,
    CMEArrivalDetector,
    CMEKinematics,
    dbm_speed_at_1au_km_s,
    dbm_transit_time_hours,
    gopalswamy_transit_time_hours,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "space_weather"


def _load_fixture(name: str) -> dict:
    with open(FIXTURE_DIR / name) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Gopalswamy empirical shock-arrival model
# ---------------------------------------------------------------------------


class TestGopalswamyModel:
    def test_1000_km_s_hand_computed_transit(self) -> None:
        """u=1000 km/s: a = 1.41 - 3.5 = -2.09 m/s^2.

        Phase 1 (to 0.76 AU = 1.13694e11 m):
            v1 = sqrt(1e12 + 2*(-2.09)*1.13694e11) = 7.24403e5 m/s
            t1 = (v1 - u)/a = 1.31865e5 s
        Phase 2 (0.24 AU at v1): t2 = 3.59035e10 / 7.24403e5 = 4.95628e4 s
        Total: 1.81428e5 s = 50.40 h.
        """
        transit = gopalswamy_transit_time_hours(1000.0)
        assert transit == pytest.approx(50.40, abs=0.05)

    def test_1000_km_s_within_literature_band(self) -> None:
        """Gopalswamy et al. (2001) report ~45-50 h for 1000 km/s CMEs."""
        transit = gopalswamy_transit_time_hours(1000.0)
        assert 45.0 <= transit <= 52.0

    def test_slow_cme_accelerates_hand_computed(self) -> None:
        """u=300 km/s: a = +0.36 m/s^2 (slow CMEs accelerate).

        v1 = sqrt(9e10 + 2*0.36*1.13694e11) = 4.14570e5 m/s
        t1 = (4.14570e5 - 3e5) / 0.36 = 3.18250e5 s
        t2 = 3.59035e10 / 4.14570e5 = 8.66040e4 s
        Total = 4.04854e5 s = 112.46 h — faster than the 138.5 h ballistic
        time at a constant 300 km/s.
        """
        transit = gopalswamy_transit_time_hours(300.0)
        assert transit == pytest.approx(112.46, abs=0.1)
        ballistic_h = AU_KM / 300.0 / 3600.0
        assert transit < ballistic_h

    def test_faster_cme_arrives_earlier(self) -> None:
        assert (
            gopalswamy_transit_time_hours(2000.0)
            < gopalswamy_transit_time_hours(1000.0)
            < gopalswamy_transit_time_hours(500.0)
        )

    @pytest.mark.parametrize("speed", [0.0, -100.0, 50.0, 5000.0, float("nan")])
    def test_out_of_domain_speed_fails_loud(self, speed: float) -> None:
        with pytest.raises(ValueError, match=r"calibration domain|domain"):
            gopalswamy_transit_time_hours(speed)


# ---------------------------------------------------------------------------
# Drag-based model (Vršnak et al. 2013)
# ---------------------------------------------------------------------------


def _dbm_numerical_transit_hours(
    v0_km_s: float, gamma_per_km: float, w_km_s: float, r0_km: float
) -> float:
    """Independent forward-Euler integration of dv/dt = -g|v-w|(v-w)."""
    dt = 10.0  # s
    r, v, t = r0_km, v0_km_s, 0.0
    while r < AU_KM:
        dv = v - w_km_s
        a = -gamma_per_km * abs(dv) * dv  # km/s^2
        v += a * dt
        r += v * dt
        t += dt
        if t > 1e7:  # pragma: no cover - guard against runaway loop
            raise AssertionError("numerical DBM integration did not converge")
    return t / 3600.0


class TestDragBasedModel:
    def test_matches_independent_numerical_integration(self) -> None:
        """Analytic inversion agrees with forward integration to < 0.1 h."""
        for v0, gamma, wind in [
            (1000.0, 0.2e-7, 400.0),
            (1000.0, 2.0e-7, 400.0),
            (600.0, 1.0e-7, 300.0),
            (250.0, 1.0e-7, 450.0),  # accelerating regime (v0 < w)
        ]:
            analytic = dbm_transit_time_hours(v0, gamma, wind)
            numeric = _dbm_numerical_transit_hours(v0, gamma, wind, DONKI_R0_KM)
            assert analytic == pytest.approx(numeric, abs=0.1), (v0, gamma, wind)

    def test_hand_computed_anchor_1000_km_s(self) -> None:
        """v0=1000, gamma=0.2e-7 km^-1, w=400 km/s from 21.5 Rs.

        Solving r(T) = 1 AU with
        r(t) = r0 + 400 t + (1/2e-8) ln(1 + 2e-8 * 600 * t)
        gives T ~ 52.5 h (hand bisection: r(52.5 h) ~ 1.496e8 km).
        """
        transit = dbm_transit_time_hours(1000.0, 0.2e-7, 400.0)
        assert transit == pytest.approx(52.5, abs=0.5)

    def test_equal_speed_wind_is_ballistic(self) -> None:
        expected_h = (AU_KM - DONKI_R0_KM) / 400.0 / 3600.0
        assert dbm_transit_time_hours(400.0, 1.0e-7, 400.0) == pytest.approx(expected_h, rel=1e-9)

    def test_arrival_speed_converges_toward_wind(self) -> None:
        v_arr = dbm_speed_at_1au_km_s(1000.0, 2.0e-7, 400.0)
        assert 400.0 < v_arr < 1000.0
        # Stronger drag pulls the arrival speed closer to the wind speed.
        v_arr_weak = dbm_speed_at_1au_km_s(1000.0, 0.2e-7, 400.0)
        assert v_arr < v_arr_weak

    def test_gamma_outside_documented_bounds_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            dbm_transit_time_hours(1000.0, gamma_per_km=1.0e-5)
        with pytest.raises(ValueError, match="gamma"):
            dbm_transit_time_hours(1000.0, gamma_per_km=0.0)

    def test_wind_outside_bounds_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="wind"):
            dbm_transit_time_hours(1000.0, wind_km_s=50.0)


# ---------------------------------------------------------------------------
# Earth-directedness geometry
# ---------------------------------------------------------------------------


class TestDirectedness:
    def test_disk_center_head_on(self) -> None:
        cls, sep = CMEArrivalDetector.classify_directedness(0.0, 0.0, 40.0)
        assert cls == "head_on"
        assert sep == pytest.approx(0.0, abs=1e-9)

    def test_hand_computed_separation(self) -> None:
        """(-7°, +9°): sep = arccos(cos 7° cos 9°) = 11.39°."""
        sep = CMEArrivalDetector.angular_separation_deg(-7.0, 9.0)
        expected = math.degrees(
            math.acos(math.cos(math.radians(7.0)) * math.cos(math.radians(9.0)))
        )
        assert sep == pytest.approx(expected, abs=1e-9)
        assert sep == pytest.approx(11.39, abs=0.01)

    def test_flank_and_miss_classes(self) -> None:
        assert CMEArrivalDetector.classify_directedness(0.0, 35.0, 40.0)[0] == "flank"
        assert CMEArrivalDetector.classify_directedness(0.0, 50.0, 40.0)[0] == "unlikely"
        assert CMEArrivalDetector.classify_directedness(0.0, 90.0, 40.0)[0] == "miss"


# ---------------------------------------------------------------------------
# End-to-end prediction + DONKI ingestion (recorded real fixture)
# ---------------------------------------------------------------------------


class TestPrediction:
    def test_predict_window_ordering_and_confidence(self) -> None:
        detector = CMEArrivalDetector()
        kin = CMEKinematics(
            speed_km_s=1000.0,
            latitude_deg=0.0,
            longitude_deg=0.0,
            half_angle_deg=45.0,
            time_21_5=datetime(2024, 5, 8, 12, 0, tzinfo=UTC),
            cme_id="test-cme",
        )
        pred = detector.predict(kin)
        assert pred.earliest_arrival_hours < pred.most_probable_arrival_hours
        assert pred.most_probable_arrival_hours < pred.latest_arrival_hours
        assert pred.earliest_arrival < pred.most_probable_arrival < pred.latest_arrival
        assert pred.earth_directed
        assert pred.directedness == "head_on"
        assert 0.0 <= pred.confidence <= 1.0
        assert pred.model_spread_hours == pytest.approx(
            pred.latest_arrival_hours - pred.earliest_arrival_hours
        )
        # Ensemble = 4 DBM corners + DBM typical + ESA model.
        assert len(pred.model_predictions_hours) == 6

    def test_miss_geometry_drops_confidence(self) -> None:
        detector = CMEArrivalDetector()
        base = {
            "speed_km_s": 1000.0,
            "half_angle_deg": 30.0,
            "time_21_5": datetime(2024, 5, 8, 12, 0, tzinfo=UTC),
        }
        head_on = detector.predict(CMEKinematics(latitude_deg=0.0, longitude_deg=0.0, **base))
        miss = detector.predict(CMEKinematics(latitude_deg=0.0, longitude_deg=120.0, **base))
        assert not miss.earth_directed
        assert miss.directedness == "miss"
        assert miss.confidence < head_on.confidence

    def test_donki_fixture_gannon_sequence(self) -> None:
        """Recorded May 2024 DONKI CMEAnalysis records all parse and predict."""
        fixture = _load_fixture("donki_cme_analysis_2024_05.json")
        assert "_provenance" in fixture and "source_url" in fixture["_provenance"]
        records = fixture["data"]
        assert len(records) >= 20

        detector = CMEArrivalDetector()
        predictions = [detector.predict_from_donki(rec) for rec in records]
        assert len(predictions) == len(records)

        # The 2024-05-08T05:36 CME (870 km/s at (-7°, +9°), half-angle 43°)
        # was Earth-directed head-on; separation is hand-computed 11.39°.
        target = next(p for p in predictions if p.cme_id == "2024-05-08T05:36:00-CME-001")
        assert target.directedness == "head_on"
        assert target.earth_directed
        assert target.angular_separation_deg == pytest.approx(11.39, abs=0.01)
        # Its actual shock arrived ~2024-05-10T17:00Z (~55.5 h after
        # time21_5 2024-05-08T09:30Z); the predicted window must cover it.
        actual_arrival = datetime(2024, 5, 10, 17, 0, tzinfo=UTC)
        assert target.earliest_arrival <= actual_arrival <= target.latest_arrival

        # At least a third of the AR13664 sequence was Earth-directed.
        n_directed = sum(1 for p in predictions if p.earth_directed)
        assert n_directed >= len(predictions) // 3

    def test_missing_kinematics_fail_loud(self) -> None:
        detector = CMEArrivalDetector()
        record = {
            "speed": 900.0,
            "latitude": None,
            "longitude": 10.0,
            "halfAngle": 40.0,
            "time21_5": "2024-05-08T09:30Z",
            "associatedCMEID": "broken-record",
        }
        with pytest.raises(ValueError, match=r"latitude.*broken-record|missing required"):
            detector.predict_from_donki(record)

    def test_empty_record_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="missing required kinematics"):
            CMEArrivalDetector().predict_from_donki({})

    def test_zero_speed_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            CMEArrivalDetector().predict(
                CMEKinematics(
                    speed_km_s=0.0,
                    latitude_deg=0.0,
                    longitude_deg=0.0,
                    half_angle_deg=40.0,
                    time_21_5=datetime(2024, 5, 8, tzinfo=UTC),
                )
            )

    def test_naive_datetime_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            CMEArrivalDetector().predict(
                CMEKinematics(
                    speed_km_s=800.0,
                    latitude_deg=0.0,
                    longitude_deg=0.0,
                    half_angle_deg=40.0,
                    time_21_5=datetime(2024, 5, 8),  # naive
                )
            )

    def test_detector_rejects_inconsistent_configuration(self) -> None:
        with pytest.raises(ValueError, match="Typical gamma"):
            CMEArrivalDetector(gamma_typical_per_km=9.0e-7)
        with pytest.raises(ValueError, match="Inverted"):
            CMEArrivalDetector(wind_range_km_s=(500.0, 300.0))
