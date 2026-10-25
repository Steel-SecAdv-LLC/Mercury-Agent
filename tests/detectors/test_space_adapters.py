# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Space-weather adapters: each serves its detector's own verdict, on real data where recorded.

Every adapter is checked against a direct call to the domain detector it
wraps (the adapter must add no judgement of its own), on storm and quiet
inputs, and must refuse malformed input instead of grading it.  Recorded
real observations are used where the repository holds them: the USGS
Boulder magnetometer and DONKI flare / CME / Kp records of the May 2024
(Gannon) storm.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError
from omni_mercury_engine.detectors.adapters import (
    CMEArrivalAdapter,
    GICWindowAdapter,
    ScintillationAdapter,
    SEPProtonWindowAdapter,
    SolarGICCascadeAdapter,
    SolarStormSnapshotAdapter,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "space_weather"
NAN = float("nan")


def _load(name: str) -> Any:
    with open(FIXTURES / name) as fh:
        return json.load(fh)["data"]


@pytest.fixture(scope="module")
def bou() -> dict[str, Any]:
    """Recorded USGS Boulder minute magnetometer, 2024-05-10 (Gannon storm)."""
    data = _load("usgs_geomag_bou_2024_05_10.json")
    series = {v["id"]: v["values"] for v in data["values"]}

    def arr(vals: list[Any]) -> np.ndarray[Any, Any]:
        return np.array([np.nan if v is None else float(v) for v in vals], dtype=np.float64)

    return {"times": data["times"], "bx": arr(series["X"]), "by": arr(series["Y"])}


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


@pytest.mark.parametrize(
    "cls",
    [
        SolarStormSnapshotAdapter,
        CMEArrivalAdapter,
        SEPProtonWindowAdapter,
        GICWindowAdapter,
        ScintillationAdapter,
        SolarGICCascadeAdapter,
    ],
)
def test_every_space_adapter_is_a_base_detector(cls: Any) -> None:
    assert issubclass(cls, BaseDetector)
    for method in ("fit", "detect", "extract_features", "predict"):
        assert callable(getattr(cls, method))


class TestSolarStormSnapshot:
    STORM = [800.0, -25.0, 5.0, NAN, NAN, NAN, NAN, 5e-4, 1500.0]
    QUIET = [380.0, 1.0, NAN, NAN, NAN, NAN, NAN, 5e-7, 0.3]

    def test_storm_and_quiet_match_the_detectors_noaa_levels(self) -> None:
        from omni_mercury_engine.evaluation.hazard_metrics import G_SCALE_NAMES
        from omni_mercury_engine.space.solar_storm_detector import SolarStormDetector

        adapter = SolarStormSnapshotAdapter().fit(np.array([self.QUIET]))
        result = adapter.detect(np.array([self.STORM, self.QUIET]))
        direct = SolarStormDetector().predict_solar_storm(
            {
                "magnetosphere_data": {
                    "solar_wind_speed_km_s": 800.0,
                    "bz_imf_nt": -25.0,
                    "by_imf_nt": 5.0,
                },
                "xray_data": {"flux_long_wm2": 5e-4},
                "proton_flux_pfu": 1500.0,
            }
        )
        top = max(
            G_SCALE_NAMES.index(direct.geomagnetic_storm_level),
            direct.radio_blackout_level,
            direct.radiation_storm_level or 0,
        )
        assert result["scores"][0] == pytest.approx(top / 5.0)
        assert top >= 3  # X-class flare: R3 at least
        assert result["flags"].tolist() == [True, False]
        assert result["scores"][1] == 0.0

    def test_operating_point_decision_is_the_detectors(self) -> None:
        """(600 km/s, Bz -10 nT): Kp ~4.1 is sub-G1, the trained operating point says G1."""
        result = SolarStormSnapshotAdapter().detect(np.array([[600.0, -10.0] + [NAN] * 7]))
        assert result["flags"][0]
        assert result["scores"][0] == pytest.approx(0.2)

    def test_unmeasured_required_or_nonphysical_values_are_refused(self) -> None:
        adapter = SolarStormSnapshotAdapter()
        with pytest.raises(InvalidObservationError, match="non-finite"):
            adapter.detect(np.array([[NAN, -5.0] + [NAN] * 7]))
        with pytest.raises(InvalidObservationError, match="physical range"):
            adapter.detect(np.array([[-400.0, -5.0] + [NAN] * 7]))
        with pytest.raises(InvalidObservationError, match="non-finite"):
            adapter.detect(np.array([[400.0, -5.0, np.inf] + [NAN] * 6]))


class TestCMEArrival:
    def test_recorded_gannon_cme_is_earth_directed(self) -> None:
        record = next(
            r
            for r in _load("donki_cme_analysis_2024_05.json")
            if r.get("associatedCMEID") == "2024-05-08T05:36:00-CME-001" and r.get("isMostAccurate")
        )
        row = [
            float(record["speed"]),
            float(record["latitude"]),
            float(record["longitude"]),
            float(record["halfAngle"]),
            _epoch(record["time21_5"]),
        ]
        result = CMEArrivalAdapter().detect(np.array([row]))
        assert result["flags"][0]
        assert result["scores"][0] >= 2.0 / 3.0  # flank or head-on

    def test_far_side_cme_is_a_miss(self) -> None:
        t = datetime(2024, 5, 8, tzinfo=UTC).timestamp()
        result = CMEArrivalAdapter().detect(np.array([[350.0, 20.0, -80.0, 15.0, t]]))
        assert not result["flags"][0]
        assert result["scores"][0] == 0.0

    def test_decision_is_exactly_score_at_least_flank(self) -> None:
        t = datetime(2024, 5, 8, tzinfo=UTC).timestamp()
        rows = [
            [900.0, lat, lon, 30.0, t] for lat in (-40, -10, 0, 10, 40) for lon in (-60, -20, 0, 25)
        ]
        predicted = CMEArrivalAdapter().predict(np.array(rows, dtype=float))
        np.testing.assert_array_equal(predicted["flags"], predicted["anomaly_scores"] >= 2.0 / 3.0)

    def test_outside_the_empirical_speed_domain_is_refused(self) -> None:
        t = datetime(2024, 5, 8, tzinfo=UTC).timestamp()
        with pytest.raises(InvalidObservationError, match="outside"):
            CMEArrivalAdapter().detect(np.array([[60.0, 0.0, 0.0, 30.0, t]]))


def _sep_window(flux10: list[float], flux100: list[float] | None = None) -> np.ndarray[Any, Any]:
    t0 = datetime(2024, 5, 11, tzinfo=UTC)
    times = [(t0 + timedelta(minutes=5 * i)).timestamp() for i in range(len(flux10))]
    f100 = [NAN] * len(flux10) if flux100 is None else flux100
    return np.column_stack([times, flux10, f100])


class TestSEPProtonWindow:
    STORM = [0.5] * 6 + [15, 40, 120, 400, 900, 1500, 2000, 1800, 1500]

    def test_storm_is_a_noaa_proton_event_at_its_s_level(self) -> None:
        result = SEPProtonWindowAdapter().detect(_sep_window(self.STORM))
        assert result["is_anomaly"]
        assert result["scores"].tolist() == [pytest.approx(3 / 5)]  # 2000 pfu peak: S3
        assert result["metadata"]["results"][0].onset_detected

    def test_a_lone_spike_scores_s1_but_is_not_an_event(self) -> None:
        result = SEPProtonWindowAdapter().detect(_sep_window([0.4, 0.5, 50.0, 0.5, 0.4]))
        assert result["scores"][0] == pytest.approx(1 / 5)
        assert not result["is_anomaly"]

    def test_quiet_flux_is_s0(self) -> None:
        result = SEPProtonWindowAdapter().detect(_sep_window([0.3, 0.4, 0.5, 0.4]))
        assert result["anomaly_score"] == 0.0 and not result["is_anomaly"]

    def test_partly_measured_100_mev_channel_is_refused(self) -> None:
        with pytest.raises(InvalidObservationError, match="partly unmeasured"):
            SEPProtonWindowAdapter().detect(_sep_window([1.0, 2.0, 3.0], [0.1, NAN, 0.1]))

    def test_detector_now_accepts_numpy_arrays(self) -> None:
        """SEPStormDetector.assess raised 'truth value ... ambiguous' on arrays."""
        from omni_mercury_engine.space.sep_storm_detector import SEPStormDetector

        times = [datetime(2024, 5, 11, tzinfo=UTC) + timedelta(minutes=5 * i) for i in range(6)]
        flux = [1.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        from_array = SEPStormDetector().assess(times, np.array(flux), np.array([0.1] * 6))
        from_list = SEPStormDetector().assess(times, flux, [0.1] * 6)
        assert from_array.s_scale == from_list.s_scale and from_array.onset_detected
        with pytest.raises(ValueError, match="Non-numeric"):
            SEPStormDetector().assess(times, [True] * 6)


class TestGICWindow:
    def _window(
        self, bou: dict[str, Any], by: np.ndarray[Any, Any] | None = None
    ) -> np.ndarray[Any, Any]:
        epochs = [_epoch(t) for t in bou["times"]]
        return np.column_stack([epochs, bou["bx"], bou["by"] if by is None else by])

    def test_recorded_gannon_boulder_window_is_the_detectors_moderate(
        self, bou: dict[str, Any]
    ) -> None:
        from omni_mercury_engine.space.gic_detector import GICDetector

        result = GICWindowAdapter({"observatory": "BOU"}).detect(self._window(bou))
        times = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in bou["times"]]
        direct = GICDetector().assess(times, bou["bx"], bou["by"], observatory="BOU")
        assert result["scores"].tolist() == [pytest.approx(1 / 3)]
        assert result["is_anomaly"]
        assessed = result["metadata"]["results"][0]
        assert assessed.risk_level == direct.risk_level == "moderate"
        assert assessed.peak_dbdt_nt_per_min == pytest.approx(253.9, abs=1.0)
        features = GICWindowAdapter().extract_features(self._window(bou))
        assert features.shape == (1, 5)
        assert features[0, 2] == pytest.approx(direct.geoelectric_peak_v_per_km)

    def test_gaps_and_h_only_mode_are_served(self, bou: dict[str, Any]) -> None:
        bx = bou["bx"].copy()
        bx[100:103] = np.nan
        gapped = self._window({**bou, "bx": bx})
        assert GICWindowAdapter().detect(gapped)["metadata"]["results"][0].n_gaps > 0
        h_only = self._window(bou, by=np.full(len(bou["bx"]), np.nan))
        assert GICWindowAdapter().detect(h_only)["metadata"]["results"][0].single_component

    def test_quiet_drift_is_low(self) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=UTC).timestamp()
        window = np.column_stack(
            [t0 + 60.0 * np.arange(60), 20000 + 0.1 * np.arange(60), 100 + 0.05 * np.arange(60)]
        )
        result = GICWindowAdapter().detect(window)
        assert result["anomaly_score"] == 0.0 and not result["is_anomaly"]

    def test_coarse_cadence_is_refused(self) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=UTC).timestamp()
        window = np.column_stack([t0 + 300.0 * np.arange(20), np.full(20, 2e4), np.full(20, 100.0)])
        with pytest.raises(InvalidObservationError, match="cadence"):
            GICWindowAdapter().detect(window)


class TestScintillation:
    def test_strong_and_weak_scintillation(self) -> None:
        rng = np.random.default_rng(0)
        storm = {
            "intensity": rng.exponential(1.0, 3000),
            "phase_rad": rng.normal(0.0, 0.9, 3000),
            "sample_rate_hz": 50.0,
        }
        quiet = {
            "intensity": 1.0 + 0.05 * rng.normal(size=3000),
            "phase_rad": rng.normal(0.0, 0.02, 3000),
            "sample_rate_hz": 50.0,
        }
        result = ScintillationAdapter().detect([storm, quiet])
        assert result["scores"].tolist() == [1.0, 0.0]
        assert result["flags"].tolist() == [True, False]

    def test_phase_without_sample_rate_is_refused(self) -> None:
        with pytest.raises(InvalidObservationError, match="sample_rate_hz"):
            ScintillationAdapter().detect({"phase_rad": np.zeros(100)})

    def test_non_record_payload_is_refused(self) -> None:
        with pytest.raises(InputModalityError, match="record"):
            ScintillationAdapter().detect(np.zeros(10))


class TestSolarGICCascade:
    @pytest.fixture(scope="class")
    def evidence(self, bou: dict[str, Any]) -> dict[str, Any]:
        kp_series = sorted(
            [entry["observedTime"], float(entry["kpIndex"])]
            for storm in _load("donki_gst_2024_05.json")
            for entry in storm["allKpIndex"]
        )
        return {
            "flares": _load("donki_flr_2024_05.json"),
            "cme_analyses": _load("donki_cme_analysis_2024_05.json"),
            "kp_series": kp_series,
            "mag_times": bou["times"],
            "mag_bx_nt": bou["bx"].tolist(),
            "mag_by_nt": bou["by"].tolist(),
            "observatory": "BOU",
        }

    def test_real_gannon_timeline_quiet_then_storm(self, evidence: dict[str, Any]) -> None:
        result = SolarGICCascadeAdapter().detect(
            [
                {**evidence, "now": "2024-05-07T00:00:00Z"},
                {**evidence, "now": "2024-05-10T23:00:00Z"},
            ]
        )
        assert result["scores"].tolist() == [0.0, 1.0]
        assert result["flags"].tolist() == [False, True]
        assert result["metadata"]["results"][1].stage.value == "storm_in_progress"

    def test_naive_time_is_refused(self, evidence: dict[str, Any]) -> None:
        with pytest.raises(InvalidObservationError, match="naive"):
            SolarGICCascadeAdapter().detect({**evidence, "now": datetime(2024, 5, 10)})
