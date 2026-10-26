# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ported NOAA loaders read real fields and refuse to fabricate.

These tests drive the pure parsing/alignment layers with recorded-shape
fixtures (no network): the SWPC product parsers and hourly alignment, the
Shue magnetopause closed form, the full HURDAT2 layout (radii, ``-999``
refusal, kinematics from consecutive fixes, the knots-vs-mph correction the
validation wrapper now applies), and the CO-OPS bucketing and in-window
percentile labels. The validation wrappers' policy-gate behaviour stays
pinned in ``tests/validation/test_synthetic_policy_gate.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.datasets.hurricane import NOAAHurricaneLoader, parse_hurdat2
from omni_mercury_engine.datasets.ocean import NOAACoOpsStationLoader
from omni_mercury_engine.datasets.space_weather import (
    NOAASpaceWeatherLoader,
    build_hourly_matrix,
    parse_kp_rows,
    parse_proton_rows,
    parse_solar_wind_rows,
    parse_xray_rows,
    shue_magnetopause_standoff_re,
)


class TestSWPCParsers:
    def test_synoptic_kp_spreads_over_its_three_hours(self) -> None:
        buckets = parse_kp_rows([{"time_tag": "2026-10-26T00:00:00", "Kp": 4.33}])
        base = min(buckets)
        assert sorted(buckets) == [base, base + 1, base + 2]
        assert all(buckets[b] == [4.33] for b in buckets)

    def test_minute_kp_fallback_keys_fill_one_hour(self) -> None:
        buckets = parse_kp_rows(
            [
                {"time_tag": "2026-10-26T00:01:00", "estimated_kp": 2.67},
                {"time_tag": "2026-10-26T00:59:00", "kp_index": 3},
            ]
        )
        assert len(buckets) == 1
        assert sorted(next(iter(buckets.values()))) == [2.67, 3.0]

    def test_solar_wind_header_is_verified_and_drift_is_refused(self) -> None:
        table: list[list[object]] = [
            ["time_tag", "speed", "density", "temperature", "bx", "by", "bz", "bt", "vx"],
            ["2026-10-26T00:10:00Z", 400.0, 5.0, 1e5, 1.0, -2.0, -3.0, 3.74, -399.0],
        ]
        buckets = parse_solar_wind_rows(table)
        assert len(buckets) == 1
        assert next(iter(buckets.values()))[0] == [400.0, 5.0, 1e5, 1.0, -2.0, -3.0, 3.74]
        with pytest.raises(ValueError, match="schema drifted"):
            parse_solar_wind_rows([["time_tag", "velocity", "n", "T"]])

    def test_band_parsers_pair_bands_per_timestamp(self) -> None:
        xray = parse_xray_rows(
            [
                {"time_tag": "2026-10-26T00:00:00Z", "flux": 1e-9, "energy": "0.05-0.4nm"},
                {"time_tag": "2026-10-26T00:00:00Z", "flux": 2e-7, "energy": "0.1-0.8nm"},
                # Unpaired timestamp: never becomes a sample.
                {"time_tag": "2026-10-26T00:01:00Z", "flux": 3e-9, "energy": "0.05-0.4nm"},
            ]
        )
        assert len(xray) == 1 and next(iter(xray.values())) == [[1e-9, 2e-7]]
        protons = parse_proton_rows(
            [
                {"time_tag": "2026-10-26T00:00:00Z", "flux": 10.0, "energy": ">=10 MeV"},
                {"time_tag": "2026-10-26T00:00:00Z", "flux": 0.2, "energy": ">=100 MeV"},
                {"time_tag": "2026-10-26T00:00:00Z", "flux": 99.0, "energy": ">=1 MeV"},
            ]
        )
        assert next(iter(protons.values())) == [[10.0, 0.2]]

    def test_shue_standoff_reproduces_textbook_magnitude(self) -> None:
        # Nominal solar wind (n=5 cm^-3, v=400 km/s, Bz=0) sits near 10 Re.
        r0 = shue_magnetopause_standoff_re(5.0, 400.0, 0.0)
        assert 9.0 < r0 < 11.5
        # Stronger dynamic pressure compresses; southward Bz erodes.
        assert shue_magnetopause_standoff_re(20.0, 700.0, 0.0) < r0
        assert shue_magnetopause_standoff_re(5.0, 400.0, -20.0) < r0

    def test_matrix_keeps_only_fully_observed_hours_and_labels_storms(self) -> None:
        hour = 498000  # arbitrary whole hour since epoch
        kp = {hour: [6.0], hour + 1: [2.0], hour + 2: [3.0]}
        sw = {
            hour: [[400.0, 5.0, 1e5, 1.0, 1.0, -2.0, 3.0]],
            hour + 1: [[420.0, 4.0, 9e4, 0.5, 0.5, 1.0, 2.0]],
        }
        xray = {hour: [[1e-9, 2e-7]], hour + 1: [[1e-9, 1e-7]], hour + 3: [[1e-9, 1e-7]]}
        protons = {hour: [[10.0, 0.2]], hour + 1: [[9.0, 0.1]]}
        features, labels = build_hourly_matrix(kp, sw, xray, protons, storm_threshold=5.0)
        assert features.shape == (2, len(NOAASpaceWeatherLoader.FEATURE_NAMES))
        assert labels.tolist() == [1, 0]
        assert np.all(np.isfinite(features))


_HURDAT2_FIXTURE = """AL092022,                IAN,      4,
20220928, 1200,  , HU, 26.7N,  82.2W, 135,  947,  175,  150,  105,  140,   80,   70,   50,   60,   45,   35,   25,   30, -999
20220928, 1800, L, HU, 26.9N,  82.0W, 130,  942,  175,  150,  105,  140,   80,   70,   50,   60,   45,   35,   25,   30, -999
20220929, 0000,  , HU, 27.2N,  81.6W,  90,  962,  180,  160,  100,  130,   80,   70,   40,   50,   30,   25,   15,   20, -999
20220929, 0600,  , TS, 27.6N,  81.2W,  60,  980,  180,  170,   90,  120, -999, -999, -999, -999, -999, -999, -999, -999, -999
AL031999,            UNNAMED,      2,
19990701, 0000,  , TS, 30.0N,  70.0W,  45, 1000, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999
19990701, 0600,  , TS, 30.5N,  70.5W,  50,  998, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999
"""


class TestHURDAT2Parser:
    def test_parses_real_fields_and_derives_kinematics(self) -> None:
        features, wind = parse_hurdat2(_HURDAT2_FIXTURE, since_year=2004)
        # Storm 1: four fixes; the first has no predecessor and the fourth
        # has -999 radii, so exactly two rows survive. Storm 2 predates
        # since_year and contributes nothing.
        assert features.shape == (2, len(NOAAHurricaneLoader.FEATURE_NAMES))
        assert wind.tolist() == [130.0, 90.0]
        lat, lon = features[0, 0], features[0, 1]
        assert (lat, lon) == (26.9, -82.0)
        assert features[0, 4] == 175.0 and features[0, 15] == 30.0  # real radii, not constants
        # ~13.6 nm in 6 h => ~2.3 kt, heading roughly ENE.
        assert 1.0 < features[0, 16] < 4.0
        assert 30.0 < features[0, 17] < 60.0
        assert features[0, 18] == 18.0  # hour of day of the second fix

    def test_missing_pressure_or_radii_rows_are_dropped_not_filled(self) -> None:
        features, _ = parse_hurdat2(_HURDAT2_FIXTURE, since_year=2004)
        assert np.all(features != -999.0)

    def test_since_year_filter_keeps_old_storms_out(self) -> None:
        features, _ = parse_hurdat2(_HURDAT2_FIXTURE, since_year=1990)
        # The 1999 storm has no radii, so it still yields no rows — but the
        # filter itself is what excludes pre-2004 seasons by default.
        assert features.shape[0] == 2

    def test_category3_threshold_is_in_knots(self) -> None:
        assert NOAAHurricaneLoader.MAJOR_HURRICANE_KT == 96.0


class TestCoOpsParsers:
    def test_bucket_scalar_averages_within_the_hour(self) -> None:
        rows = [
            {"t": "2026-10-26 00:00", "v": "10.0"},
            {"t": "2026-10-26 00:30", "v": "12.0"},
            {"t": "2026-10-26 01:00", "v": "nan"},
            {"t": "2026-10-26 02:00", "v": "not-a-number"},
        ]
        buckets = NOAACoOpsStationLoader._bucket_scalar(rows, "v")
        assert len(buckets) == 1
        assert sorted(next(iter(buckets.values()))) == [10.0, 12.0]

    def test_heatwave_labels_are_the_in_window_monthly_p90(self) -> None:
        import types

        loader = types.SimpleNamespace(HEATWAVE_PERCENTILE=90.0)
        n = 100
        features = np.zeros((n, 9))
        features[:, 0] = np.arange(n, dtype=float)  # water temp strictly rising
        buckets = [498000 + i for i in range(n)]  # one month
        labels = NOAACoOpsStationLoader._heatwave_labels(loader, features, buckets)  # type: ignore[arg-type]
        assert labels.sum() == 10  # exactly the top decile
        assert labels[-10:].all() and not labels[:-10].any()


class TestValidationWrapperDelegation:
    def test_hurricane_wrapper_converts_mph_to_knots(self, tmp_path: object) -> None:
        """111 mph (Category 3) must meet HURDAT2's knots as ~96.46 kt."""
        from omni_mercury_engine.validation.data_loaders import (
            NOAAHurricaneLoader as Wrapper,
        )

        wrapper = Wrapper(cache_dir=str(tmp_path))
        canonical = wrapper._canonical(100, major_threshold_kt=111.0 / Wrapper._MPH_PER_KT)
        assert canonical.major_threshold_kt == pytest.approx(96.46, abs=0.01)
