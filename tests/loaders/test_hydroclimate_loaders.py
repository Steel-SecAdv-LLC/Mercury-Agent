# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline-deterministic tests for DroughtLoader and HeatwaveLoader (T4b1).

Daily-station payloads are constructed in-test to the exact CSV column
schema the loaders' parsers consume (labelled constructed, never presented
as recorded data). The live paths are exercised by the domain benchmark
(measured 2026-07-09: drought mean AUC 0.5728, heatwave 0.6757) and the
weekly network lane.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from omni_mercury_engine.loaders import label_provenance
from omni_mercury_engine.loaders.drought_loader import DroughtLoader
from omni_mercury_engine.loaders.heatwave_loader import HeatwaveLoader

RNG = np.random.default_rng(7)


def _gsom_csv(start: str = "1950-01", end: str = "2015-12") -> pd.DataFrame:
    """Constructed NCEI GSOM access CSV (metric: PRCP mm, TAVG/TMAX degC)."""
    months = pd.period_range(start, end, freq="M")
    doy = np.array([m.month for m in months], dtype=np.float64)
    seasonal = 20.0 + 10.0 * np.sin(2.0 * np.pi * (doy - 4.0) / 12.0)
    prcp = RNG.gamma(2.0, 40.0, size=len(months))
    return pd.DataFrame(
        {
            "DATE": [str(m) for m in months],
            "PRCP": np.round(prcp, 1).astype(str),
            "TAVG": np.round(seasonal, 1).astype(str),
            "TMAX": np.round(seasonal + 6.0, 1).astype(str),
        }
    )


def _gsod_csv(year: int) -> pd.DataFrame:
    """Constructed NCEI GSOD access CSV (degF, sentinel-aware schema)."""
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    doy = dates.dayofyear.to_numpy(dtype=np.float64)
    tmax_f = 68.0 + 25.0 * np.sin(2.0 * np.pi * (doy - 105.0) / 365.0)
    tmax_f = tmax_f + RNG.normal(0.0, 3.0, size=len(dates))
    return pd.DataFrame(
        {
            "DATE": dates.strftime("%Y-%m-%d"),
            "MAX": np.round(tmax_f, 1).astype(str),
            "MIN": np.round(tmax_f - 18.0, 1).astype(str),
            "TEMP": np.round(tmax_f - 9.0, 1).astype(str),
            "DEWP": np.round(tmax_f - 25.0, 1).astype(str),
            "PRCP": np.full(len(dates), "0.00"),
        }
    )


class TestDroughtLoader:
    def test_contract_and_catalog(self, tmp_path: Path) -> None:
        loader = DroughtLoader(cache_dir=tmp_path)
        assert loader.DOMAIN == "drought"
        assert loader.LABEL_SOURCE == "statistical"
        events = loader.list_events()
        assert events and {"event_id", "name", "date", "description"} <= set(events[0])

    def test_fetch_and_labels_align(self, tmp_path: Path) -> None:
        loader = DroughtLoader(cache_dir=tmp_path)
        loader._fetch_csv = lambda url, dtype=None: _gsom_csv()  # type: ignore[method-assign]
        event_id = "texas_2011"
        df = loader.fetch_historical(event_id)
        assert not df.empty
        assert next(iter(df["datetime"])) <= "1950-12"
        labels = loader.get_ground_truth(event_id)
        feats = loader.engineer_features(df)
        assert labels.shape[0] == feats.shape[0]
        assert set(np.unique(labels)) <= {0, 1}
        assert np.isfinite(feats).all()

    def test_provenance_registry_entry(self) -> None:
        registry = label_provenance.LABEL_PROVENANCE_REGISTRY
        assert "drought_loader.DroughtLoader" in registry
        assert registry["drought_loader.DroughtLoader"][0] == "statistical"


class TestHeatwaveLoader:
    def test_contract_and_catalog(self, tmp_path: Path) -> None:
        loader = HeatwaveLoader(cache_dir=tmp_path)
        assert loader.DOMAIN == "heatwave"
        assert loader.LABEL_SOURCE == "statistical"
        assert loader.list_events()

    def test_fetch_and_labels_align(self, tmp_path: Path) -> None:
        loader = HeatwaveLoader(cache_dir=tmp_path)

        def fake_fetch_csv(url: str, dtype: Any = None) -> pd.DataFrame:
            year = int(Path(url).stem.split("-")[-1]) if "-" in Path(url).stem else 2011
            # GSOD archive URLs end in <year>/<station>.csv — recover the year.
            for token in url.split("/"):
                if token.isdigit() and len(token) == 4:
                    year = int(token)
            return _gsod_csv(year)

        loader._fetch_csv = fake_fetch_csv  # type: ignore[method-assign]
        event_id = "texas_2011"
        df = loader.fetch_historical(event_id)
        assert not df.empty
        labels = loader.get_ground_truth(event_id)
        feats = loader.engineer_features(df)
        assert labels.shape[0] == feats.shape[0]
        assert set(np.unique(labels)) <= {0, 1}
        assert np.isfinite(feats).all()

    def test_empty_station_payload_fails_loud(self, tmp_path: Path) -> None:
        loader = HeatwaveLoader(cache_dir=tmp_path)
        loader._fetch_csv = lambda url, dtype=None: pd.DataFrame()  # type: ignore[method-assign]
        with pytest.raises((ValueError, KeyError)):
            loader.fetch_historical("texas_2011")

    def test_provenance_registry_entry(self) -> None:
        registry = label_provenance.LABEL_PROVENANCE_REGISTRY
        assert "heatwave_loader.HeatwaveLoader" in registry
        assert registry["heatwave_loader.HeatwaveLoader"][0] == "statistical"
