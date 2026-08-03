# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""RTSW/ACE solar-wind merge preference for EnergyLoader.

The RTSW 1-minute product is multi-spacecraft (SOLAR1/SWFO-L1 active, ACE
and IMAP non-active as of 2026) and its feed order is not active-first, so
``drop_duplicates(keep="first")`` was measured keeping the non-active
spacecraft's calibration in 35% of duplicate minutes and once keeping a
fill row (NaN) while discarding a valid measurement at the same minute
(2026-08-03T03:04Z, evidence in the PR). These tests pin the explicit
preference that replaced feed order: most valid measurements first, then
the active spacecraft, then feed order (RTSW before ACE hourly history).
Payloads are constructed to the exact RTSW/SWEPAM shapes and labelled as
constructed, never presented as recorded data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from omni_mercury_engine.loaders.energy_loader import EnergyLoader

if TYPE_CHECKING:
    from pathlib import Path


def _rtsw_row(
    time_tag: str,
    speed: float | None,
    source: str,
    active: bool,
    density: float | None = 5.0,
    temperature: float | None = 100000.0,
) -> dict[str, Any]:
    """One constructed RTSW wind row; None becomes the -9999 fill sentinel."""
    return {
        "time_tag": time_tag,
        "proton_density": -9999.0 if density is None else density,
        "proton_speed": -9999.0 if speed is None else speed,
        "proton_temperature": -9999.0 if temperature is None else temperature,
        "source": source,
        "active": active,
    }


def _loader(tmp_path: Path, rtsw: list[dict[str, Any]], ace: list[dict[str, Any]]) -> EnergyLoader:
    loader = EnergyLoader(cache_dir=tmp_path / "cache")

    def fake_fetch_json(url: str, params: dict[str, Any] | None = None) -> Any:
        if "rtsw" in url:
            return rtsw
        if "swepam" in url:
            return ace
        raise AssertionError(f"unexpected fetch in solar-wind test: {url}")

    loader._fetch_json = fake_fetch_json  # type: ignore[method-assign, unused-ignore]
    return loader


class TestSolarWindMergePreference:
    def test_active_spacecraft_wins_duplicate_minute(self, tmp_path: Path) -> None:
        """Feed order lists ACE first; the active SOLAR1 row must win."""
        rtsw = [
            _rtsw_row("2026-08-03T03:05:00", 400.0, "ACE", active=False),
            _rtsw_row("2026-08-03T03:05:00", 412.0, "SOLAR1", active=True),
        ]
        df = _loader(tmp_path, rtsw, [])._fetch_solar_wind()
        assert len(df) == 1
        assert df["solar_wind_speed"].iloc[0] == pytest.approx(412.0)

    def test_fill_row_never_shadows_valid_measurement(self, tmp_path: Path) -> None:
        """The measured 03:04Z incident: active fill first, valid ACE second."""
        rtsw = [
            _rtsw_row(
                "2026-08-03T03:04:00",
                None,
                "SOLAR1",
                active=True,
                density=None,
                temperature=None,
            ),
            _rtsw_row("2026-08-03T03:04:00", 373.66, "ACE", active=False),
        ]
        df = _loader(tmp_path, rtsw, [])._fetch_solar_wind()
        assert len(df) == 1
        assert df["solar_wind_speed"].iloc[0] == pytest.approx(373.66), (
            "keep='first' used to keep the all-NaN fill row and discard the "
            "valid measurement at the same minute"
        )

    def test_offgrid_seconds_floor_to_minute_grid(self, tmp_path: Path) -> None:
        """IMAP's second-precision tags no longer interleave a third calibration."""
        rtsw = [
            _rtsw_row("2026-08-03T18:36:00", 410.0, "SOLAR1", active=True),
            _rtsw_row("2026-08-03T18:36:08", 395.0, "IMAP", active=False),
        ]
        df = _loader(tmp_path, rtsw, [])._fetch_solar_wind()
        assert len(df) == 1, "18:36:08 must collapse onto the 18:36 grid minute"
        assert df["solar_wind_speed"].iloc[0] == pytest.approx(410.0)

    def test_nonactive_row_fills_minute_with_no_active_coverage(self, tmp_path: Path) -> None:
        """A valid non-active row beats no row at all."""
        rtsw = [
            _rtsw_row("2026-08-03T18:37:08", 395.0, "IMAP", active=False),
        ]
        df = _loader(tmp_path, rtsw, [])._fetch_solar_wind()
        assert len(df) == 1
        assert df["solar_wind_speed"].iloc[0] == pytest.approx(395.0)

    def test_rtsw_beats_ace_hourly_at_overlapping_stamp(self, tmp_path: Path) -> None:
        """The pre-existing RTSW-over-ACE-history preference is preserved."""
        rtsw = [_rtsw_row("2026-08-03T18:00:00", 420.0, "SOLAR1", active=True)]
        ace = [
            {
                "time_tag": "2026-08-03T18:00:00",
                "dens": 6.1,
                "speed": 401.0,
                "temperature": 90000.0,
            }
        ]
        df = _loader(tmp_path, rtsw, ace)._fetch_solar_wind()
        assert len(df) == 1
        assert df["solar_wind_speed"].iloc[0] == pytest.approx(420.0)

    def test_ace_history_still_covers_older_window(self, tmp_path: Path) -> None:
        """ACE hourly rows outside RTSW coverage survive with NaN handling."""
        rtsw = [_rtsw_row("2026-08-03T18:00:00", 420.0, "SOLAR1", active=True)]
        ace = [
            {
                "time_tag": "2026-08-01T05:00:00",
                "dens": -9999.9,
                "speed": 388.0,
                "temperature": 85000.0,
            }
        ]
        df = _loader(tmp_path, rtsw, ace)._fetch_solar_wind()
        assert len(df) == 2
        older = df[df["timestamp"] == df["timestamp"].min()]
        assert older["solar_wind_speed"].iloc[0] == pytest.approx(388.0)
        assert np.isnan(older["solar_wind_density"].iloc[0]), (
            "the -9999.9 fill must become NaN, never a measurement"
        )

    def test_provenance_columns_do_not_leak_into_output(self, tmp_path: Path) -> None:
        rtsw = [_rtsw_row("2026-08-03T18:00:00", 420.0, "SOLAR1", active=True)]
        df = _loader(tmp_path, rtsw, [])._fetch_solar_wind()
        assert list(df.columns) == [
            "timestamp",
            "solar_wind_density",
            "solar_wind_speed",
            "solar_wind_temperature",
        ]
