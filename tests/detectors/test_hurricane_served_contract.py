# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""What ``HurricaneDetector.predict_hurricane`` serves, on real ERA5 patches.

Each test fails on the detector as it was before this contract was written:
the category ladder used mph boundaries on knots, a wind-only input in the
trained (T=2) format could never be detected, a category was assigned with no
cyclone detected, white noise in ``signal_data`` flipped the decision, and a
pressure in kPa or an SST in kelvin produced a confident result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.exceptions import InvalidObservationError
from omni_mercury_engine.detectors.geological.hurricane_detector import HurricaneDetector
from omni_mercury_engine.ml.hazard_training import hurricane_wind as hw

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "hazard_training" / "hurricane"
) / "era5_patch_sample.npz"


@pytest.fixture(scope="module")
def patches() -> dict[str, np.ndarray[Any, Any]]:
    """Super Typhoon Mawar (165 kt best track) and a far-from-storm negative."""
    with np.load(FIXTURE, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def served() -> HurricaneDetector:
    """The default-constructed detector, serving the shipped checkpoint."""
    return HurricaneDetector()


def _case(x: np.ndarray[Any, Any]) -> dict[str, Any]:
    return {
        "wind_field": {
            "u": x[:, 0].astype(np.float64),
            "v": x[:, 1].astype(np.float64),
            "grid_spacing_m": hw.GRID_SPACING_M,
        }
    }


def test_wind_only_trained_format_detects_a_real_super_typhoon(
    served: HurricaneDetector, patches: dict[str, np.ndarray[Any, Any]]
) -> None:
    result = served.predict_hurricane(_case(patches["x"][0]))
    assert result.cyclone_detected is True
    assert result.confidence >= 0.5
    assert result.category.startswith("category_")


def test_far_from_storm_patch_is_no_cyclone(
    served: HurricaneDetector, patches: dict[str, np.ndarray[Any, Any]]
) -> None:
    result = served.predict_hurricane(_case(patches["x"][1]))
    assert result.cyclone_detected is False
    assert result.category == "no_cyclone"
    assert result.warning_actions == []
    assert result.evacuation_zones == []


def test_category_follows_the_reported_wind_in_knots(
    served: HurricaneDetector, patches: dict[str, np.ndarray[Any, Any]]
) -> None:
    result = served.predict_hurricane(_case(patches["x"][0]))
    expected = served._classify_category(result.max_wind_speed_kt)
    assert result.category == expected
    if result.max_wind_speed_kt >= 96:
        assert result.category in {"category_3", "category_4", "category_5"}


def test_signal_noise_carries_no_decision_weight(served: HurricaneDetector) -> None:
    base = {"pressure_data": {"central_pressure_mb": 1000.0, "environmental_pressure_mb": 1013.0}}
    noisy = {**base, "signal_data": np.random.default_rng(0).normal(0.0, 1.0, 512)}
    a, b = served.predict_hurricane(base), served.predict_hurricane(noisy)
    assert (a.cyclone_detected, a.confidence) == (b.cyclone_detected, b.confidence)


@pytest.mark.parametrize(
    ("pressure_data", "fragment"),
    [
        ({"central_pressure_mb": 95.0}, "central_pressure_mb"),  # kPa
        ({"central_pressure_mb": 101_325.0}, "central_pressure_mb"),  # Pa
        ({"central_pressure_mb": float("nan")}, "finite"),
        ({"central_pressure_mb": float("-inf")}, "finite"),
    ],
)
def test_pressure_outside_hpa_range_is_refused(
    served: HurricaneDetector, pressure_data: dict[str, float], fragment: str
) -> None:
    with pytest.raises(InvalidObservationError, match=fragment):
        served.predict_hurricane({"pressure_data": pressure_data})


def test_sst_in_kelvin_is_refused(served: HurricaneDetector) -> None:
    with pytest.raises(InvalidObservationError, match="sst_celsius"):
        served.predict_hurricane({"sst_data": {"sst_celsius": 300.15}})


def test_wind_in_knots_is_refused(served: HurricaneDetector) -> None:
    u = np.full((8, 8), 150.0)  # 150 "m/s" is a knot value
    with pytest.raises(InvalidObservationError, match="m/s"):
        served.predict_hurricane({"wind_field": {"u": u, "v": u * 0, "grid_spacing_m": 25e3}})


def test_vorticity_needs_a_stated_grid_spacing() -> None:
    det = HurricaneDetector(load_shipped_weights=False)
    n = 33
    x = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(x, x)
    field = {"u": -yy * 30.0, "v": xx * 30.0}
    assert det._analyze_wind_field(field)["max_relative_vorticity_s1"] == 0.0
    spaced = det._analyze_wind_field({**field, "grid_spacing_m": 2.0 * 50e3 / (n - 1)})
    assert spaced["max_relative_vorticity_s1"] == pytest.approx(2 * 30.0 / 50e3, rel=1e-6)


def test_sequence_vorticity_is_taken_on_the_newest_frame() -> None:
    det = HurricaneDetector(load_shipped_weights=False)
    n = 33
    x = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(x, x)
    calm = np.zeros((n, n))
    seq = {
        "u": np.stack([calm, -yy * 30.0]),
        "v": np.stack([calm, xx * 30.0]),
        "grid_spacing_m": 2.0 * 50e3 / (n - 1),
    }
    assert det._analyze_wind_field(seq)["max_relative_vorticity_s1"] > 1e-3


def test_rapid_intensification_needs_a_time_base() -> None:
    det = HurricaneDetector(load_shipped_weights=False)
    history = [990.0, 985.0, 975.0, 960.0, 945.0]  # 45 hPa over the record
    undated = det.predict_hurricane(
        {"pressure_data": {"central_pressure_mb": 945.0, "pressure_history_mb": history}}
    )
    assert undated.rapid_intensification is False
    assert undated.pressure_change_24h_mb is None
    six_hourly = det.predict_hurricane(
        {
            "pressure_data": {
                "central_pressure_mb": 945.0,
                "pressure_history_mb": history,
                "pressure_history_interval_h": 6.0,
            }
        }
    )
    assert six_hourly.pressure_change_24h_mb == pytest.approx(45.0)
    assert six_hourly.rapid_intensification is True
    twelve_hourly = det.predict_hurricane(
        {
            "pressure_data": {
                "central_pressure_mb": 945.0,
                "pressure_history_mb": history,
                "pressure_history_interval_h": 12.0,
            }
        }
    )
    assert twelve_hourly.pressure_change_24h_mb == pytest.approx(30.0)
    assert twelve_hourly.rapid_intensification is False


#: Hurricane Milton (AL142024), synoptic 6-hourly best-track fixes 2024-10-06
#: 00Z to 10-07 18Z (max wind kt, central pressure hPa), from NHC HURDAT2
#: hurdat2-1851-2025-092326.txt (sha256 75202780ce56c845...).
MILTON_WIND_KT = [35.0, 45.0, 55.0, 70.0, 75.0, 90.0, 120.0, 150.0]
MILTON_PRESSURE_HPA = [1004.0, 999.0, 991.0, 986.0, 981.0, 972.0, 943.0, 908.0]


def test_milton_intensification_is_measured_in_knots() -> None:
    det = HurricaneDetector(load_shipped_weights=False)
    result = det.predict_hurricane(
        {
            "pressure_data": {
                "central_pressure_mb": 908.0,
                "pressure_history_mb": MILTON_PRESSURE_HPA,
                "pressure_history_interval_h": 6.0,
            },
            "intensity_history": {"max_wind_kt": MILTON_WIND_KT, "interval_h": 6.0},
        }
    )
    assert result.intensification_rate_kt_24h == pytest.approx(80.0)  # 70 -> 150 kt
    assert result.pressure_change_24h_mb == pytest.approx(78.0)  # 986 -> 908 hPa
    assert result.rapid_intensification is True


def test_wind_history_alone_flags_rapid_intensification() -> None:
    det = HurricaneDetector(load_shipped_weights=False)
    result = det.predict_hurricane(
        {"intensity_history": {"max_wind_kt": MILTON_WIND_KT, "interval_h": 6.0}}
    )
    assert result.rapid_intensification is True
    steady = det.predict_hurricane(
        {"intensity_history": {"max_wind_kt": [90.0] * 5, "interval_h": 6.0}}
    )
    assert (steady.intensification_rate_kt_24h, steady.rapid_intensification) == (0.0, False)
    short = det.predict_hurricane(
        {"intensity_history": {"max_wind_kt": [70.0, 90.0], "interval_h": 6.0}}
    )
    assert short.intensification_rate_kt_24h is None


@pytest.mark.parametrize(
    "data",
    [
        {"intensity_history": {"max_wind_kt": MILTON_WIND_KT, "interval_h": 7.0}},
        {
            "pressure_data": {
                "central_pressure_mb": 908.0,
                "pressure_history_mb": MILTON_PRESSURE_HPA,
                "pressure_history_interval_h": 7.0,
            }
        },
        {"intensity_history": {"max_wind_kt": MILTON_WIND_KT}},
    ],
)
def test_a_24h_change_needs_a_sample_24h_back(data: dict[str, Any]) -> None:
    # A 7 h interval measured the change over 21 h and called it 24 h.
    with pytest.raises(InvalidObservationError):
        HurricaneDetector(load_shipped_weights=False).predict_hurricane(data)


def test_short_signal_features_do_not_raise() -> None:
    det = HurricaneDetector(load_shipped_weights=False)
    features = det.extract_features(np.ones((4, 64)))
    assert features.shape[-1] == 20
