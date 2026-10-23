# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PM2.5 AQI against EPA's own index, on EPA's recorded daily file.

The fixture is every row of EPA AirData's ``daily_88101_2024`` for two dates
(see ``tests/fixtures/environment/epa_aqs_pm25/README.md``).  EPA publishes
its AQI beside each 24-hour standard value, so the detector is checked
against the regulator's own number, not a re-implementation of it.  The
same file pins the loader defects: each continuous monitor's day appears as
a "1 HOUR" summary beside its "24-HR BLK AVG" standard row, and an
exceptional-event day as both "Included" and "Excluded".
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from omni_mercury_engine.core.exceptions import InvalidObservationError
from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.epa_air import EPAAirQualityLoader
from omni_mercury_engine.detectors.adapters.earth import PM25AirQualityAdapter
from omni_mercury_engine.detectors.meteorological.air_quality import (
    AirQualityDetector,
    pm25_aqi,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "environment"
    / "epa_aqs_pm25"
    / "daily_88101_2024_0705_0909.csv"
)


def _rows() -> list[dict[str, str]]:
    with FIXTURE.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _standard(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """The 24-hour standard value of each monitor-day, as measured."""
    return [r for r in rows if r["Pollutant Standard"] and r["Event Type"] != "Excluded"]


def _monitor_day(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[k] for k in ("State Code", "County Code", "Site Num", "POC", "Date Local"))


class TestAgainstEPA:
    def test_every_standard_row_matches_epa_aqi(self) -> None:
        rows = _standard(_rows())
        assert len(rows) == 2715
        mismatches = [
            (r["Arithmetic Mean"], r["AQI"])
            for r in rows
            if pm25_aqi(float(r["Arithmetic Mean"]))[0] != int(r["AQI"])
        ]
        assert mismatches == []
        # The fixture spans the scale up to "very unhealthy".
        assert max(int(r["AQI"]) for r in rows) == 208
        assert sum(int(r["AQI"]) >= 101 for r in rows) == 59

    def test_excluded_row_carries_the_measured_days_aqi(self) -> None:
        """EPA's AQI on an 'Excluded' row is the AQI of the air as measured."""
        rows = _rows()
        (excluded,) = [r for r in rows if r["Pollutant Standard"] and r["Event Type"] == "Excluded"]
        (included,) = [
            r
            for r in rows
            if r["Pollutant Standard"]
            and r["Event Type"] == "Included"
            and _monitor_day(r) == _monitor_day(excluded)
        ]
        assert pm25_aqi(float(excluded["Arithmetic Mean"]))[0] != int(excluded["AQI"])
        assert pm25_aqi(float(included["Arithmetic Mean"]))[0] == int(excluded["AQI"])


class TestTable2:
    """40 CFR Part 58 Appendix G, Table 2 and section 3.4."""

    @pytest.mark.parametrize(
        ("concentration", "aqi", "category"),
        [
            (0.0, 0, "good"),
            (9.0, 50, "good"),
            (9.1, 51, "moderate"),
            (35.4, 100, "moderate"),
            (35.5, 101, "unhealthy_for_sensitive_groups"),
            (55.4, 150, "unhealthy_for_sensitive_groups"),
            (55.5, 151, "unhealthy"),
            (125.4, 200, "unhealthy"),
            (125.5, 201, "very_unhealthy"),
            (225.4, 300, "very_unhealthy"),
            (225.5, 301, "hazardous"),
            (325.4, 500, "hazardous"),
        ],
    )
    def test_breakpoints_equal_their_index(
        self, concentration: float, aqi: int, category: str
    ) -> None:
        assert pm25_aqi(concentration) == (aqi, category)

    def test_truncation_not_rounding(self) -> None:
        """35.49 truncates to 35.4 (moderate); the loader's '> 35.4' read it as USG."""
        assert pm25_aqi(35.49) == (100, "moderate")
        # Exact decimal truncation: binary floor(2.3 * 10) / 10 would give 2.2.
        assert pm25_aqi(2.3) == pm25_aqi(2.39)
        assert pm25_aqi(2.3) != pm25_aqi(2.2)

    def test_above_500_extends_the_hazardous_segment(self) -> None:
        """Footnote 4; EPA's 2024 file reports 372.1 ug/m3 as AQI 593."""
        assert pm25_aqi(372.1) == (593, "hazardous")

    def test_negative_mean_is_zero_as_epa_reports_it(self) -> None:
        rows = [r for r in _standard(_rows()) if float(r["Arithmetic Mean"]) < 0]
        assert rows and all(r["AQI"] == "0" for r in rows)
        assert all(pm25_aqi(float(r["Arithmetic Mean"]))[0] == 0 for r in rows)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), "high"])
    def test_non_finite_is_refused(self, value: object) -> None:
        with pytest.raises(InvalidObservationError):
            AirQualityDetector().assess(value)


class TestLoader:
    """EPAAirQualityLoader on the recorded file (no network)."""

    @pytest.fixture
    def parsed(self, tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
        loader = EPAAirQualityLoader(
            DatasetConfig(
                name="epa_air_quality",
                data_dir=str(tmp_path / "data"),
                cache_dir=str(tmp_path / "cache"),
            )
        )
        return loader._parse_epa_csv(FIXTURE.read_text())

    def test_one_row_per_monitor_day(self, parsed: tuple[np.ndarray, np.ndarray]) -> None:
        """Every row was kept: 4,602 rows for 2,715 monitor-days."""
        features, _ = parsed
        rows = _rows()
        assert len(rows) == 4602
        assert len({_monitor_day(r) for r in _standard(rows)}) == 2715
        assert features.shape == (2715, 8)

    def test_labels_are_epa_aqi_101_or_more(self, parsed: tuple[np.ndarray, np.ndarray]) -> None:
        """The label was 'mean > 35.4' over every row: 1 HOUR duplicates counted twice."""
        features, labels = parsed
        expected = [int(int(r["AQI"]) >= 101) for r in _standard(_rows())]
        assert labels.tolist() == expected
        assert np.allclose(
            features[:, 0], [float(r["Arithmetic Mean"]) for r in _standard(_rows())]
        )

    def test_blank_mean_is_skipped_not_zero(self, tmp_path: Path) -> None:
        header = FIXTURE.read_text().splitlines()[0]
        good = next(line for line in FIXTURE.read_text().splitlines()[1:] if "PM25 24-hour" in line)
        fields = next(csv.reader([good]))
        blank = list(fields)
        blank[header.split(",").index("Arithmetic Mean")] = ""
        text = "\n".join([header, good, ",".join(f'"{f}"' for f in blank)]) + "\n"
        loader = EPAAirQualityLoader(
            DatasetConfig(name="epa_air_quality", data_dir=str(tmp_path), cache_dir=str(tmp_path))
        )
        features, _ = loader._parse_epa_csv(text)
        assert features.shape[0] == 1

    def test_download_path_reads_the_zip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("daily_88101_2024.csv", FIXTURE.read_text())
        monkeypatch.setattr(
            "omni_mercury_engine.datasets.epa_air.http_get_with_retry",
            lambda url, **kwargs: buf.getvalue(),
        )
        loader = EPAAirQualityLoader(
            DatasetConfig(
                name="epa_air_quality",
                data_dir=str(tmp_path / "data"),
                cache_dir=str(tmp_path / "cache"),
                preprocessing={"year": 2024},
            )
        )
        assert loader.download() is True
        features, labels = loader._load_raw()
        assert features.shape == (2715, 8)
        assert int(labels.sum()) == 59


class TestAdapter:
    def test_recorded_days_through_the_adapter(self) -> None:
        rows = _standard(_rows())
        batch = np.array([[float(r["Arithmetic Mean"])] for r in rows])
        out = PM25AirQualityAdapter().detect(batch)
        assert out["flags"].tolist() == [int(r["AQI"]) >= 101 for r in rows]
        assert np.all((out["scores"] >= 0.0) & (out["scores"] <= 1.0))
        assert np.array_equal(out["scores"] >= 2 / 5, out["flags"])
