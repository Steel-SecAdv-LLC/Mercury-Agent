"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

Comprehensive tests for all domain loaders.

Tests cover instantiation, interface compliance, list_events() validation,
feature engineering with mock data, mocked HTTP responses, sklearn-free
verification, and BaseDomainLoader helper methods.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from omni_mercury_engine.loaders.base import BaseDomainLoader

# ---------------------------------------------------------------------------
# Loader registry: (class_name, module_path, domain_str, requires_api_key)
# ---------------------------------------------------------------------------
LOADER_REGISTRY: list[tuple[str, str, str, bool]] = [
    ("EarthquakeLoader", "omni_mercury_engine.loaders.earthquake_loader", "earthquake", False),
    ("TsunamiLoader", "omni_mercury_engine.loaders.tsunami_loader", "tsunami", False),
    ("HurricaneLoader", "omni_mercury_engine.loaders.hurricane_loader", "hurricane", False),
    ("TornadoLoader", "omni_mercury_engine.loaders.tornado_loader", "tornado", False),
    ("FloodLoader", "omni_mercury_engine.loaders.flood_loader", "flood", False),
    ("WildfireLoader", "omni_mercury_engine.loaders.wildfire_loader", "wildfire", True),
    ("VolcanicLoader", "omni_mercury_engine.loaders.volcanic_loader", "volcanic", False),
    ("LandslideLoader", "omni_mercury_engine.loaders.landslide_loader", "landslide", False),
    ("SepsisLoader", "omni_mercury_engine.loaders.sepsis_loader", "sepsis", False),
    ("PandemicLoader", "omni_mercury_engine.loaders.pandemic_loader", "pandemic", False),
    ("FinancialLoader", "omni_mercury_engine.loaders.financial_loader", "financial", True),
    ("EnergyLoader", "omni_mercury_engine.loaders.energy_loader", "energy", False),
    ("MarineLoader", "omni_mercury_engine.loaders.marine_loader", "marine", False),
    (
        "NetworkSecurityLoader",
        "omni_mercury_engine.loaders.network_security_loader",
        "network_security",
        False,
    ),
    ("FEMALoader", "omni_mercury_engine.loaders.fema_loader", "fema", False),
]

# Collect all module paths for the sklearn-free check.
_LOADER_MODULE_PATHS = [entry[1] for entry in LOADER_REGISTRY]

# Required keys every dict returned by list_events() must contain.
_REQUIRED_EVENT_KEYS = {"event_id", "name", "date", "description"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_loader_class(module_path: str, class_name: str) -> type:
    """Import a loader class by module path and class name."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls


def _make_loader(module_path: str, class_name: str, tmp_path: Path) -> BaseDomainLoader:
    """Instantiate a loader, providing an API key stub when required and a
    temporary cache directory so tests never pollute the real cache."""
    cls = _import_loader_class(module_path, class_name)
    kwargs: dict[str, Any] = {
        "cache_dir": tmp_path / class_name,
        "max_retries": 0,
        "timeout": 5,
    }
    if cls.REQUIRES_API_KEY:
        kwargs["api_key"] = "TEST_KEY_PLACEHOLDER"
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Parameterized IDs for readable test output
# ---------------------------------------------------------------------------

_LOADER_IDS = [entry[0] for entry in LOADER_REGISTRY]
_LOADER_PARAMS = [
    pytest.param(entry[0], entry[1], entry[2], entry[3], id=entry[0]) for entry in LOADER_REGISTRY
]


# =========================================================================
# 1. Instantiation
# =========================================================================


class TestLoaderInstantiation:
    """Each loader class can be instantiated without network access."""

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_loader_instantiation(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        assert loader is not None
        assert isinstance(loader, BaseDomainLoader)

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_domain_attribute(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        assert domain == loader.DOMAIN

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_requires_api_key_attribute(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        assert loader.REQUIRES_API_KEY is requires_key

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_source_url_not_empty(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        assert loader.SOURCE_URL != "", f"{class_name} has empty SOURCE_URL"

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_cache_dir_created(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        assert loader.cache_dir.exists()


# =========================================================================
# 2. Interface compliance
# =========================================================================


class TestInterfaceCompliance:
    """Each loader implements the full BaseDomainLoader interface."""

    _REQUIRED_METHODS = [
        "fetch_realtime",
        "fetch_historical",
        "list_events",
        "get_ground_truth",
        "engineer_features",
    ]

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_has_required_methods(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        for method_name in self._REQUIRED_METHODS:
            method = getattr(loader, method_name, None)
            assert method is not None, f"{class_name} is missing required method '{method_name}'"
            assert callable(method), f"{class_name}.{method_name} is not callable"

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_is_subclass_of_base(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
    ) -> None:
        cls = _import_loader_class(module_path, class_name)
        assert issubclass(cls, BaseDomainLoader)

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_fetch_realtime_signature(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
    ) -> None:
        cls = _import_loader_class(module_path, class_name)
        sig = inspect.signature(cls.fetch_realtime)
        # Only parameter should be 'self'
        params = list(sig.parameters.keys())
        assert params == [
            "self"
        ], f"{class_name}.fetch_realtime should accept only 'self', got {params}"

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_fetch_historical_signature(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
    ) -> None:
        cls = _import_loader_class(module_path, class_name)
        sig = inspect.signature(cls.fetch_historical)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "event_id" in params

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_list_events_signature(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
    ) -> None:
        cls = _import_loader_class(module_path, class_name)
        sig = inspect.signature(cls.list_events)
        params = list(sig.parameters.keys())
        assert params == ["self"]

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_get_ground_truth_signature(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
    ) -> None:
        cls = _import_loader_class(module_path, class_name)
        sig = inspect.signature(cls.get_ground_truth)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "event_id" in params

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_engineer_features_signature(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
    ) -> None:
        cls = _import_loader_class(module_path, class_name)
        sig = inspect.signature(cls.engineer_features)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "raw_data" in params

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_has_compute_data_hash(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
    ) -> None:
        cls = _import_loader_class(module_path, class_name)
        assert hasattr(cls, "compute_data_hash")
        assert callable(cls.compute_data_hash)

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_has_get_provenance(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
    ) -> None:
        cls = _import_loader_class(module_path, class_name)
        assert hasattr(cls, "get_provenance")
        assert callable(cls.get_provenance)


# =========================================================================
# 3. list_events() returns non-empty list of dicts with required keys
# =========================================================================


class TestListEvents:
    """Validate the event catalog exposed by each loader."""

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_list_events_returns_list(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        events = loader.list_events()
        assert isinstance(events, list), f"{class_name}.list_events() should return a list"

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_list_events_non_empty(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        events = loader.list_events()
        assert len(events) > 0, f"{class_name}.list_events() returned an empty list"

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_list_events_items_are_dicts(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        events = loader.list_events()
        for idx, event in enumerate(events):
            assert isinstance(
                event, dict
            ), f"{class_name}.list_events()[{idx}] is not a dict: {type(event)}"

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_list_events_required_keys(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        events = loader.list_events()
        for idx, event in enumerate(events):
            missing = _REQUIRED_EVENT_KEYS - set(event.keys())
            assert not missing, f"{class_name}.list_events()[{idx}] missing keys: {missing}"

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_list_events_values_are_strings(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        events = loader.list_events()
        for idx, event in enumerate(events):
            for key in _REQUIRED_EVENT_KEYS:
                val = event[key]
                assert isinstance(val, str), (
                    f"{class_name}.list_events()[{idx}]['{key}'] "
                    f"is {type(val).__name__}, expected str"
                )
                assert len(val) > 0, f"{class_name}.list_events()[{idx}]['{key}'] is empty"

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_list_events_unique_ids(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        events = loader.list_events()
        event_ids = [e["event_id"] for e in events]
        assert len(event_ids) == len(
            set(event_ids)
        ), f"{class_name}.list_events() contains duplicate event_ids"


# =========================================================================
# 4. Feature engineering with mock data
# =========================================================================


def _make_generic_numeric_df(n_rows: int = 50, n_cols: int = 5) -> pd.DataFrame:
    """Create a generic DataFrame with numeric columns for testing
    the base engineer_features() implementation."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_rows, n_cols))
    columns = [f"feature_{i}" for i in range(n_cols)]
    return pd.DataFrame(data, columns=columns)


def _make_earthquake_df(n_rows: int = 50) -> pd.DataFrame:
    """Create mock earthquake data matching the USGS GeoJSON schema."""
    rng = np.random.default_rng(42)
    base_time = 1700000000000  # epoch ms
    return pd.DataFrame(
        {
            "time": base_time + np.arange(n_rows) * 60000,
            "latitude": rng.uniform(30, 40, n_rows),
            "longitude": rng.uniform(-120, -110, n_rows),
            "depth": rng.uniform(0, 300, n_rows),
            "magnitude": rng.uniform(1.0, 8.0, n_rows),
            "place": [f"location_{i}" for i in range(n_rows)],
            "event_id": [f"evt_{i}" for i in range(n_rows)],
        }
    )


def _make_tsunami_df(n_rows: int = 100) -> pd.DataFrame:
    """Create mock tsunami BPR data."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2011-03-11", periods=n_rows, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "bpr": 5000.0 + rng.normal(0, 0.05, n_rows),
            "station_id": "21418",
        }
    )


def _make_hurricane_df(n_rows: int = 40) -> pd.DataFrame:
    """Create mock hurricane track data matching IBTrACS schema."""
    rng = np.random.default_rng(42)
    times = pd.date_range("2005-08-23", periods=n_rows, freq="6h")
    return pd.DataFrame(
        {
            "sid": "2005236N23285",
            "season": 2005,
            "name": "KATRINA",
            "iso_time": times.strftime("%Y-%m-%d %H:%M:%S"),
            "lat": np.linspace(23, 30, n_rows) + rng.normal(0, 0.1, n_rows),
            "lon": np.linspace(-85, -90, n_rows) + rng.normal(0, 0.1, n_rows),
            "wind_kt": np.clip(rng.normal(80, 30, n_rows), 20, 165),
            "pressure_mb": np.clip(rng.normal(960, 20, n_rows), 900, 1013),
        }
    )


def _make_tornado_df(n_rows: int = 30) -> pd.DataFrame:
    """Create mock tornado data matching SPC archive schema."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "om": np.arange(n_rows),
            "yr": [2011] * n_rows,
            "mo": [4] * n_rows,
            "dy": rng.integers(25, 29, n_rows),
            "date": [f"2011-04-{d:02d}" for d in rng.integers(25, 29, n_rows)],
            "time": ["18:00:00"] * n_rows,
            "tz": [3] * n_rows,
            "st": ["AL"] * n_rows,
            "stf": [1] * n_rows,
            "stn": [0] * n_rows,
            "mag": rng.integers(0, 6, n_rows),
            "inj": rng.integers(0, 20, n_rows),
            "fat": rng.integers(0, 5, n_rows),
            "loss": rng.uniform(0, 1e6, n_rows),
            "closs": rng.uniform(0, 1e5, n_rows),
            "slat": rng.uniform(33, 35, n_rows),
            "slon": rng.uniform(-88, -86, n_rows),
            "elat": rng.uniform(33, 35, n_rows),
            "elon": rng.uniform(-88, -86, n_rows),
            "len": rng.uniform(0.1, 30, n_rows),
            "wid": rng.integers(10, 2000, n_rows),
            "ns": [1] * n_rows,
            "sn": [0] * n_rows,
            "sg": [1] * n_rows,
            "f1": [0] * n_rows,
            "f2": [0] * n_rows,
            "f3": [0] * n_rows,
            "f4": [0] * n_rows,
            "fc": [0] * n_rows,
        }
    )


def _make_flood_df(n_rows: int = 100) -> pd.DataFrame:
    """Create mock flood gauge data matching USGS schema."""
    rng = np.random.default_rng(42)
    datetimes = pd.date_range("2024-09-25", periods=n_rows, freq="15min")
    return pd.DataFrame(
        {
            "datetime": datetimes,
            "site_id": "03451500",
            "site_name": "French Broad River at Marshall, NC",
            "gauge_height_ft": rng.uniform(3.0, 20.0, n_rows),
            "discharge_cfs": rng.uniform(500, 50000, n_rows),
        }
    )


def _make_wildfire_df(n_rows: int = 40) -> pd.DataFrame:
    """Create mock FIRMS fire detection data."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "latitude": rng.uniform(33.5, 34.5, n_rows),
            "longitude": rng.uniform(-119.0, -117.5, n_rows),
            "bright_ti4": rng.uniform(300, 500, n_rows),
            "frp": rng.uniform(0.5, 200, n_rows),
            "confidence": rng.choice(["low", "nominal", "high"], n_rows),
            "scan": rng.uniform(0.3, 1.0, n_rows),
            "track": rng.uniform(0.3, 1.0, n_rows),
            "acq_date": ["2025-01-07"] * n_rows,
            "acq_time": rng.integers(0, 2400, n_rows),
            "satellite": ["N"] * n_rows,
            "instrument": ["VIIRS"] * n_rows,
            "version": ["2.0NRT"] * n_rows,
        }
    )


def _make_volcanic_df(n_rows: int = 20) -> pd.DataFrame:
    """Create mock volcanic alert data."""
    rng = np.random.default_rng(42)
    levels = ["NORMAL", "ADVISORY", "WATCH", "WARNING"]
    colors = ["GREEN", "YELLOW", "ORANGE", "RED"]
    alert_level_map = {"NORMAL": 0, "ADVISORY": 1, "WATCH": 2, "WARNING": 3}
    color_code_map = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
    chosen_levels = rng.choice(levels, n_rows)
    chosen_colors = rng.choice(colors, n_rows)
    return pd.DataFrame(
        {
            "volcano_name": ["Kilauea"] * n_rows,
            "alert_level": chosen_levels,
            "color_code": chosen_colors,
            "alert_level_numeric": [alert_level_map[lvl] for lvl in chosen_levels],
            "color_code_numeric": [color_code_map[c] for c in chosen_colors],
            "alert_date": pd.date_range("2018-05-01", periods=n_rows, freq="1D").astype(str),
            "latitude": [19.421] * n_rows,
            "longitude": [-155.287] * n_rows,
            "elevation": [1222] * n_rows,
        }
    )


def _make_landslide_df(n_rows: int = 30) -> pd.DataFrame:
    """Create mock NASA COOLR landslide data."""
    rng = np.random.default_rng(42)
    categories = ["landslide", "mudslide", "rockfall", "debris_flow"]
    triggers = ["rain", "earthquake", "construction"]
    sizes = ["small", "medium", "large", "very_large"]
    return pd.DataFrame(
        {
            "event_id": [f"ls_{i}" for i in range(n_rows)],
            "event_date": pd.date_range("2014-03-01", periods=n_rows, freq="1D").astype(str),
            "event_category": rng.choice(categories, n_rows),
            "landslide_trigger": rng.choice(triggers, n_rows),
            "landslide_size": rng.choice(sizes, n_rows),
            "fatality_count": rng.integers(0, 10, n_rows),
            "injury_count": rng.integers(0, 30, n_rows),
            "latitude": rng.uniform(47, 49, n_rows),
            "longitude": rng.uniform(-122, -121, n_rows),
            "country_name": ["United States"] * n_rows,
            "admin_division_name": ["Washington"] * n_rows,
        }
    )


def _make_sepsis_df(n_rows: int = 100) -> pd.DataFrame:
    """Create mock PhysioNet sepsis data."""
    rng = np.random.default_rng(42)
    n_patients = 5
    rows_per_patient = n_rows // n_patients
    dfs = []
    for p in range(n_patients):
        patient_df = pd.DataFrame(
            {
                "HR": rng.uniform(60, 120, rows_per_patient),
                "O2Sat": rng.uniform(90, 100, rows_per_patient),
                "Temp": rng.uniform(36, 39, rows_per_patient),
                "SBP": rng.uniform(90, 160, rows_per_patient),
                "MAP": rng.uniform(60, 110, rows_per_patient),
                "DBP": rng.uniform(50, 90, rows_per_patient),
                "Resp": rng.uniform(10, 30, rows_per_patient),
                "WBC": rng.uniform(4, 20, rows_per_patient),
                "Lactate": rng.uniform(0.5, 4, rows_per_patient),
                "Creatinine": rng.uniform(0.5, 3, rows_per_patient),
                "Platelets": rng.uniform(100, 400, rows_per_patient),
                "Bilirubin_total": rng.uniform(0.1, 5, rows_per_patient),
                "FiO2": rng.uniform(0.21, 1.0, rows_per_patient),
                "pH": rng.uniform(7.2, 7.5, rows_per_patient),
                "PaCO2": rng.uniform(30, 50, rows_per_patient),
                "SaO2": rng.uniform(90, 100, rows_per_patient),
                "BUN": rng.uniform(5, 40, rows_per_patient),
                "Age": [65.0] * rows_per_patient,
                "Gender": [1.0] * rows_per_patient,
                "HospAdmTime": [-1.0] * rows_per_patient,
                "ICULOS": np.arange(1, rows_per_patient + 1, dtype=float),
                "SepsisLabel": rng.integers(0, 2, rows_per_patient),
                "patient_id": [f"p{p:06d}"] * rows_per_patient,
            }
        )
        dfs.append(patient_df)
    return pd.concat(dfs, ignore_index=True)


def _make_pandemic_df(n_rows: int = 120) -> pd.DataFrame:
    """Create mock OWID pandemic data."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-03-01", periods=n_rows, freq="1D")
    cumulative = np.cumsum(rng.integers(100, 5000, n_rows))
    new_cases = np.concatenate([[0], np.diff(cumulative)])
    return pd.DataFrame(
        {
            "date": dates.astype(str),
            "location": ["United States"] * n_rows,
            "new_cases": np.abs(new_cases),
            "new_deaths": rng.integers(0, 200, n_rows),
            "total_cases": cumulative,
            "total_deaths": np.cumsum(rng.integers(0, 200, n_rows)),
            "new_cases_per_million": rng.uniform(0, 50, n_rows),
            "new_deaths_per_million": rng.uniform(0, 5, n_rows),
            "new_cases_smoothed": rng.uniform(100, 5000, n_rows),
            "reproduction_rate": rng.uniform(0.5, 3.0, n_rows),
            "new_tests_per_thousand": rng.uniform(0, 10, n_rows),
            "positive_rate": rng.uniform(0.01, 0.30, n_rows),
            "stringency_index": rng.uniform(0, 100, n_rows),
        }
    )


def _make_financial_df(n_rows: int = 252) -> pd.DataFrame:
    """Create mock FRED financial stress data.

    The FinancialLoader's engineer_features() expects lowercase column names
    matching the standardised FRED series mapping: vix, yield_curve_10y2y,
    high_yield_spread, fed_funds_rate, ted_spread.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2008-01-02", periods=n_rows, freq="B")
    return pd.DataFrame(
        {
            "date": dates.astype(str),
            "vix": rng.uniform(10, 80, n_rows),
            "yield_curve_10y2y": rng.uniform(-1, 3, n_rows),
            "high_yield_spread": rng.uniform(3, 20, n_rows),
            "fed_funds_rate": rng.uniform(0, 5, n_rows),
            "ted_spread": rng.uniform(0, 4, n_rows),
        }
    )


def _make_energy_df(n_rows: int = 100) -> pd.DataFrame:
    """Create mock NOAA SWPC + EIA energy data.

    The EnergyLoader's engineer_features() uses ``_safe_column()`` which
    gracefully returns zeros for missing columns.  Key expected columns
    are ``kp``, ``solar_wind_speed``, ``solar_wind_density``, and
    ``xray_class``.
    """
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-05-10", periods=n_rows, freq="3h").astype(str),
            "kp": rng.uniform(0, 9, n_rows),
            "solar_wind_speed": rng.uniform(300, 800, n_rows),
            "solar_wind_density": rng.uniform(1, 20, n_rows),
            "xray_class": rng.choice([0, 1, 2, 3, 4, 5], n_rows).astype(float),
        }
    )


def _make_marine_df(n_rows: int = 50) -> pd.DataFrame:
    """Create mock OBIS marine occurrence data."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "decimalLatitude": rng.uniform(-30, 30, n_rows),
            "decimalLongitude": rng.uniform(-180, 180, n_rows),
            "depth": rng.uniform(0, 500, n_rows),
            "speciesCount": rng.integers(1, 200, n_rows),
            "year": rng.integers(2015, 2023, n_rows),
            "month": rng.integers(1, 13, n_rows),
        }
    )


def _make_network_security_df(n_rows: int = 100) -> pd.DataFrame:
    """Create mock network security flow data."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "duration": rng.uniform(0, 1000, n_rows),
            "src_bytes": rng.integers(0, 100000, n_rows),
            "dst_bytes": rng.integers(0, 100000, n_rows),
            "count": rng.integers(1, 500, n_rows),
            "srv_count": rng.integers(1, 500, n_rows),
            "serror_rate": rng.uniform(0, 1, n_rows),
            "rerror_rate": rng.uniform(0, 1, n_rows),
            "same_srv_rate": rng.uniform(0, 1, n_rows),
            "diff_srv_rate": rng.uniform(0, 1, n_rows),
            "dst_host_count": rng.integers(1, 255, n_rows),
        }
    )


def _make_fema_df(n_rows: int = 50) -> pd.DataFrame:
    """Create mock FEMA disaster declaration data."""
    rng = np.random.default_rng(42)
    declaration_types = ["DR", "EM", "FM"]
    incident_types = ["Flood", "Hurricane", "Severe Storm(s)", "Fire"]
    return pd.DataFrame(
        {
            "disasterNumber": rng.integers(4000, 5000, n_rows),
            "declarationDate": pd.date_range("2020-01-01", periods=n_rows, freq="7D").astype(str),
            "declarationType": rng.choice(declaration_types, n_rows),
            "incidentType": rng.choice(incident_types, n_rows),
            "state": rng.choice(["TX", "FL", "CA", "NC", "NY"], n_rows),
            "declarationTitle": [f"Disaster {i}" for i in range(n_rows)],
            "ihProgramDeclared": rng.integers(0, 2, n_rows),
            "iaProgramDeclared": rng.integers(0, 2, n_rows),
            "paProgramDeclared": rng.integers(0, 2, n_rows),
            "hmProgramDeclared": rng.integers(0, 2, n_rows),
        }
    )


# Map loader class names to their mock DataFrame factories.
_MOCK_DF_FACTORIES: dict[str, Any] = {
    "EarthquakeLoader": _make_earthquake_df,
    "TsunamiLoader": _make_tsunami_df,
    "HurricaneLoader": _make_hurricane_df,
    "TornadoLoader": _make_tornado_df,
    "FloodLoader": _make_flood_df,
    "WildfireLoader": _make_wildfire_df,
    "VolcanicLoader": _make_volcanic_df,
    "LandslideLoader": _make_landslide_df,
    "SepsisLoader": _make_sepsis_df,
    "PandemicLoader": _make_pandemic_df,
    "FinancialLoader": _make_financial_df,
    "EnergyLoader": _make_energy_df,
    "MarineLoader": _make_marine_df,
    "NetworkSecurityLoader": _make_network_security_df,
    "FEMALoader": _make_fema_df,
}


class TestFeatureEngineering:
    """Test engineer_features() with mock data for every loader."""

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_engineer_features_returns_2d_array(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        factory = _MOCK_DF_FACTORIES.get(class_name, _make_generic_numeric_df)
        mock_df = factory()
        features = loader.engineer_features(mock_df)

        assert isinstance(
            features, np.ndarray
        ), f"{class_name}.engineer_features() should return np.ndarray"
        assert (
            features.ndim == 2
        ), f"{class_name}.engineer_features() should return a 2-D array, got {features.ndim}-D"

    # Loaders that aggregate/bin input rows (output rows != input rows).
    _AGGREGATING_LOADERS = {"MarineLoader"}

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_engineer_features_correct_rows(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        factory = _MOCK_DF_FACTORIES.get(class_name, _make_generic_numeric_df)
        mock_df = factory()
        features = loader.engineer_features(mock_df)

        if class_name in self._AGGREGATING_LOADERS:
            # MarineLoader bins into spatial grid cells so output rows
            # may differ from input rows.  Just verify non-zero output.
            assert (
                features.shape[0] > 0
            ), f"{class_name}.engineer_features() returned 0 rows from {len(mock_df)} input rows"
        else:
            assert features.shape[0] == len(mock_df), (
                f"{class_name}.engineer_features() row count mismatch: "
                f"expected {len(mock_df)}, got {features.shape[0]}"
            )

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_engineer_features_no_nan(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        factory = _MOCK_DF_FACTORIES.get(class_name, _make_generic_numeric_df)
        mock_df = factory()
        features = loader.engineer_features(mock_df)

        nan_count = int(np.isnan(features).sum())
        assert nan_count == 0, f"{class_name}.engineer_features() produced {nan_count} NaN values"

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_engineer_features_no_inf(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        factory = _MOCK_DF_FACTORIES.get(class_name, _make_generic_numeric_df)
        mock_df = factory()
        features = loader.engineer_features(mock_df)

        inf_count = int(np.isinf(features).sum())
        assert (
            inf_count == 0
        ), f"{class_name}.engineer_features() produced {inf_count} infinite values"

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_engineer_features_positive_feature_count(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        factory = _MOCK_DF_FACTORIES.get(class_name, _make_generic_numeric_df)
        mock_df = factory()
        features = loader.engineer_features(mock_df)

        assert features.shape[1] > 0, f"{class_name}.engineer_features() returned 0 columns"

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_engineer_features_empty_df(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        empty_df = pd.DataFrame()
        features = loader.engineer_features(empty_df)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 0

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_engineer_features_dtype_float(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        loader = _make_loader(module_path, class_name, tmp_path)
        factory = _MOCK_DF_FACTORIES.get(class_name, _make_generic_numeric_df)
        mock_df = factory()
        features = loader.engineer_features(mock_df)

        assert np.issubdtype(
            features.dtype, np.floating
        ), f"{class_name}.engineer_features() dtype is {features.dtype}, expected floating-point"


# =========================================================================
# 5. Mocked HTTP responses (no real network calls)
# =========================================================================


class TestMockedNetworkCalls:
    """Verify loaders work with mocked HTTP responses."""

    @patch("omni_mercury_engine.loaders.base.BaseDomainLoader._fetch_url")
    def test_earthquake_fetch_realtime_mocked(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        """Earthquake loader processes mocked USGS GeoJSON correctly."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"time": 1700000000000, "mag": 4.5, "place": "Test"},
                    "geometry": {"type": "Point", "coordinates": [-118.5, 34.0, 10.0]},
                    "id": "test_quake_1",
                },
            ],
        }
        mock_fetch.return_value = json.dumps(geojson).encode("utf-8")

        from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader

        loader = EarthquakeLoader(cache_dir=tmp_path / "eq", max_retries=0)
        df = loader.fetch_realtime()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "magnitude" in df.columns
        mock_fetch.assert_called_once()

    @patch("omni_mercury_engine.loaders.base.BaseDomainLoader._fetch_url")
    def test_tsunami_fetch_realtime_mocked(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        """Tsunami loader handles mocked DART data."""
        # Simulate a DART data file with a header and a few data rows.
        dart_content = (
            "# YY  MM DD hh mm  T     HEIGHT\n"
            "2024  01 15 12 00  1  5000.12\n"
            "2024  01 15 12 01  1  5000.15\n"
            "2024  01 15 12 02  1  5000.11\n"
        )
        mock_fetch.return_value = dart_content.encode("utf-8")

        from omni_mercury_engine.loaders.tsunami_loader import TsunamiLoader

        loader = TsunamiLoader(cache_dir=tmp_path / "ts", max_retries=0)
        df = loader.fetch_realtime()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 3  # At least the 3 rows * number of stations that succeed

    @patch("omni_mercury_engine.loaders.base.BaseDomainLoader._fetch_url")
    def test_fema_fetch_realtime_mocked(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        """FEMA loader processes mocked OpenFEMA JSON."""
        fema_response = {
            "DisasterDeclarationsSummaries": [
                {
                    "disasterNumber": 4999,
                    "declarationDate": "2024-01-01T00:00:00.000Z",
                    "declarationType": "DR",
                    "incidentType": "Flood",
                    "state": "NC",
                    "declarationTitle": "Test Disaster",
                    "ihProgramDeclared": 1,
                    "iaProgramDeclared": 1,
                    "paProgramDeclared": 1,
                    "hmProgramDeclared": 0,
                },
            ],
        }
        mock_fetch.return_value = json.dumps(fema_response).encode("utf-8")

        from omni_mercury_engine.loaders.fema_loader import FEMALoader

        loader = FEMALoader(cache_dir=tmp_path / "fema", max_retries=0)
        df = loader.fetch_realtime()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1

    @patch("omni_mercury_engine.loaders.base.BaseDomainLoader._fetch_url")
    def test_earthquake_fetch_historical_mocked(
        self, mock_fetch: MagicMock, tmp_path: Path
    ) -> None:
        """Earthquake fetch_historical with a valid event_id and mocked response."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "time": 1700000000000 + i * 60000,
                        "mag": 3.0 + i * 0.5,
                        "place": f"loc_{i}",
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [-118.5 + i * 0.01, 34.0 + i * 0.01, 10.0],
                    },
                    "id": f"q_{i}",
                }
                for i in range(10)
            ],
        }
        mock_fetch.return_value = json.dumps(geojson).encode("utf-8")

        from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader

        loader = EarthquakeLoader(cache_dir=tmp_path / "eq2", max_retries=0)
        df = loader.fetch_historical("turkey_syria_2023")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

    @patch("omni_mercury_engine.loaders.base.BaseDomainLoader._fetch_url")
    def test_earthquake_get_ground_truth_mocked(
        self, mock_fetch: MagicMock, tmp_path: Path
    ) -> None:
        """Earthquake get_ground_truth returns binary labels."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "time": 1700000000000 + i * 60000,
                        "mag": 2.5 + i * 0.7,
                        "place": f"loc_{i}",
                    },
                    "geometry": {"type": "Point", "coordinates": [-118.5, 34.0, 10.0]},
                    "id": f"q_{i}",
                }
                for i in range(20)
            ],
        }
        mock_fetch.return_value = json.dumps(geojson).encode("utf-8")

        from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader

        loader = EarthquakeLoader(cache_dir=tmp_path / "eq3", max_retries=0)
        labels = loader.get_ground_truth("turkey_syria_2023")
        assert isinstance(labels, np.ndarray)
        assert labels.ndim == 1
        assert set(labels).issubset({0, 1})

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_fetch_historical_unknown_event_raises(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        """All loaders raise ValueError for an unknown event_id."""
        loader = _make_loader(module_path, class_name, tmp_path)
        with pytest.raises(ValueError, match=r"[Uu]nknown"):
            loader.fetch_historical("nonexistent_event_id_xyz")

    @pytest.mark.parametrize("class_name,module_path,domain,requires_key", _LOADER_PARAMS)
    def test_get_ground_truth_unknown_event_raises(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        """All loaders raise ValueError for an unknown event_id in get_ground_truth."""
        loader = _make_loader(module_path, class_name, tmp_path)
        with pytest.raises(ValueError, match=r"[Uu]nknown"):
            loader.get_ground_truth("nonexistent_event_id_xyz")


# =========================================================================
# 6. No loader imports sklearn
# =========================================================================


class TestNoSklearnImports:
    """Verify that none of the loader modules import sklearn."""

    @pytest.mark.parametrize("module_path", _LOADER_MODULE_PATHS, ids=_LOADER_IDS)
    def test_no_sklearn_in_source(self, module_path: str) -> None:
        """Check the module source code does not import sklearn.

        Comments mentioning sklearn (e.g. 'no sklearn dependency') are
        acceptable; only actual import statements are forbidden.
        """
        mod = importlib.import_module(module_path)
        source_file = inspect.getfile(mod)

        with open(source_file) as f:
            lines = f.readlines()

        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            # Skip comments and docstrings
            if stripped.startswith("#"):
                continue
            # Check for actual import statements referencing sklearn
            if "sklearn" in stripped and (
                stripped.startswith("import ")
                or stripped.startswith("from ")
                or "import sklearn" in stripped
                or "from sklearn" in stripped
            ):
                raise AssertionError(
                    f"{module_path} imports sklearn at line {lineno}: {stripped!r}"
                )

    @pytest.mark.parametrize("module_path", _LOADER_MODULE_PATHS, ids=_LOADER_IDS)
    def test_no_sklearn_in_loaded_modules(self, module_path: str) -> None:
        """Check that importing the loader does not pull in sklearn."""
        # Capture modules before import
        before = set(sys.modules.keys())

        # Force re-import (if already loaded, just check current state)
        importlib.import_module(module_path)

        after = set(sys.modules.keys())
        new_modules = after - before

        sklearn_modules = [m for m in new_modules if m.startswith("sklearn")]
        assert (
            len(sklearn_modules) == 0
        ), f"Importing {module_path} loaded sklearn modules: {sklearn_modules}"

    def test_base_loader_no_sklearn(self) -> None:
        """The base loader itself does not import sklearn."""
        mod = importlib.import_module("omni_mercury_engine.loaders.base")
        source_file = inspect.getfile(mod)

        with open(source_file) as f:
            lines = f.readlines()

        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "sklearn" in stripped and (
                stripped.startswith("import ")
                or stripped.startswith("from ")
                or "import sklearn" in stripped
                or "from sklearn" in stripped
            ):
                raise AssertionError(f"Base loader imports sklearn at line {lineno}: {stripped!r}")


# =========================================================================
# 7. BaseDomainLoader helper methods
# =========================================================================


class TestBaseDomainLoaderHelpers:
    """Test the BaseDomainLoader static/helper methods directly."""

    # -- compute_data_hash --

    def test_compute_data_hash_returns_hex_string(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        result = BaseDomainLoader.compute_data_hash(data)
        assert isinstance(result, str)
        # SHA-256 hex digest is 64 characters
        assert len(result) == 64

    def test_compute_data_hash_deterministic(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        h1 = BaseDomainLoader.compute_data_hash(data)
        h2 = BaseDomainLoader.compute_data_hash(data)
        assert h1 == h2

    def test_compute_data_hash_different_for_different_data(self) -> None:
        data_a = np.array([1.0, 2.0, 3.0])
        data_b = np.array([1.0, 2.0, 4.0])
        h_a = BaseDomainLoader.compute_data_hash(data_a)
        h_b = BaseDomainLoader.compute_data_hash(data_b)
        assert h_a != h_b

    def test_compute_data_hash_matches_manual_sha256(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        expected = hashlib.sha256(data.tobytes()).hexdigest()
        result = BaseDomainLoader.compute_data_hash(data)
        assert result == expected

    def test_compute_data_hash_2d_array(self) -> None:
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = BaseDomainLoader.compute_data_hash(data)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_compute_data_hash_empty_array(self) -> None:
        data = np.array([])
        result = BaseDomainLoader.compute_data_hash(data)
        assert isinstance(result, str)
        assert len(result) == 64

    # -- caching --

    def test_cache_write_and_read(self, tmp_path: Path) -> None:
        """Data written to cache can be read back."""
        from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader

        loader = EarthquakeLoader(cache_dir=tmp_path / "cache_test", max_retries=0)

        test_data = {"key": "value", "numbers": [1, 2, 3]}
        loader._write_cache("test_key", test_data)

        result = loader._read_cache("test_key")
        assert result is not None
        assert result == test_data

    def test_cache_returns_none_for_missing_key(self, tmp_path: Path) -> None:
        from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader

        loader = EarthquakeLoader(cache_dir=tmp_path / "cache_test2", max_retries=0)
        result = loader._read_cache("nonexistent_key")
        assert result is None

    def test_cache_expires_after_ttl(self, tmp_path: Path) -> None:
        """Cached data expires after the configured TTL."""
        from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader

        loader = EarthquakeLoader(cache_dir=tmp_path / "cache_ttl", max_retries=0)
        # Set a very short TTL
        loader.CACHE_TTL = 0

        test_data = {"ttl_test": True}
        loader._write_cache("ttl_key", test_data)

        # Immediately reading after TTL=0 should return None (or the data
        # if the write-then-read happens within the same second). To force
        # expiration, manually adjust the cache file timestamp.
        cache_path = loader._get_cache_path("ttl_key")
        with open(cache_path) as f:
            cached = json.load(f)
        # Backdate the timestamp by 2 seconds
        cached["timestamp"] = time.time() - 2
        with open(cache_path, "w") as f:
            json.dump(cached, f)

        result = loader._read_cache("ttl_key")
        assert result is None

    def test_get_cache_path_deterministic(self, tmp_path: Path) -> None:
        from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader

        loader = EarthquakeLoader(cache_dir=tmp_path / "cache_path", max_retries=0)
        p1 = loader._get_cache_path("same_key")
        p2 = loader._get_cache_path("same_key")
        assert p1 == p2

    def test_get_cache_path_different_keys(self, tmp_path: Path) -> None:
        from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader

        loader = EarthquakeLoader(cache_dir=tmp_path / "cache_diff", max_retries=0)
        p1 = loader._get_cache_path("key_a")
        p2 = loader._get_cache_path("key_b")
        assert p1 != p2

    # -- get_provenance --

    def test_get_provenance_returns_dict(self, tmp_path: Path) -> None:
        from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader

        loader = EarthquakeLoader(cache_dir=tmp_path / "prov", max_retries=0)
        data = np.array([1.0, 2.0, 3.0])
        prov = loader.get_provenance("test_event", data)

        assert isinstance(prov, dict)
        assert "domain" in prov
        assert "event_id" in prov
        assert "timestamp" in prov
        assert "data_hash" in prov
        assert "data_shape" in prov
        assert "git_commit" in prov
        assert "source_url" in prov

    def test_get_provenance_domain_matches(self, tmp_path: Path) -> None:
        from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader

        loader = EarthquakeLoader(cache_dir=tmp_path / "prov2", max_retries=0)
        data = np.array([1.0, 2.0, 3.0])
        prov = loader.get_provenance("test_event", data)

        assert prov["domain"] == "earthquake"
        assert prov["event_id"] == "test_event"

    def test_get_provenance_data_hash_matches(self, tmp_path: Path) -> None:
        from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader

        loader = EarthquakeLoader(cache_dir=tmp_path / "prov3", max_retries=0)
        data = np.array([1.0, 2.0, 3.0])
        prov = loader.get_provenance("test_event", data)

        expected_hash = BaseDomainLoader.compute_data_hash(data)
        assert prov["data_hash"] == expected_hash

    def test_get_provenance_data_shape(self, tmp_path: Path) -> None:
        from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader

        loader = EarthquakeLoader(cache_dir=tmp_path / "prov4", max_retries=0)
        data = np.zeros((10, 5))
        prov = loader.get_provenance("test_event", data)

        assert prov["data_shape"] == [10, 5]

    # -- base engineer_features (default implementation) --

    def test_base_engineer_features_numeric_only(self, tmp_path: Path) -> None:
        """The base engineer_features picks numeric columns and cleans them."""
        from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader

        loader = EarthquakeLoader(cache_dir=tmp_path / "base_eng", max_retries=0)

        # Call the BASE class method directly (not the override)
        df = pd.DataFrame(
            {
                "num_a": [1.0, 2.0, np.nan, 4.0],
                "num_b": [10.0, np.inf, 30.0, 40.0],
                "str_c": ["a", "b", "c", "d"],
            }
        )

        features = BaseDomainLoader.engineer_features(loader, df)
        assert isinstance(features, np.ndarray)
        assert features.shape == (4, 2)  # only 2 numeric columns
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()

    # -- _fetch_url retry and error handling --

    @patch("urllib.request.urlopen")
    def test_fetch_url_raises_on_failure(self, mock_urlopen: MagicMock, tmp_path: Path) -> None:
        """_fetch_url raises ConnectionError after exhausting retries."""
        mock_urlopen.side_effect = Exception("Network error")

        from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader

        loader = EarthquakeLoader(
            cache_dir=tmp_path / "fetch_fail",
            max_retries=0,
            timeout=1,
        )
        with pytest.raises(ConnectionError):
            loader._fetch_url("http://example.com/test")


# =========================================================================
# Network tests (skipped in CI, run with -m network)
# =========================================================================


class TestNetworkLive:
    """Live network tests -- only run when explicitly requested.

    Run with: pytest -m network
    """

    @pytest.mark.network
    @pytest.mark.parametrize(
        "class_name,module_path,domain,requires_key",
        [p for p in _LOADER_PARAMS if not p.values[3]],  # skip loaders requiring API keys
    )
    def test_list_events_live(
        self,
        class_name: str,
        module_path: str,
        domain: str,
        requires_key: bool,
        tmp_path: Path,
    ) -> None:
        """Verify list_events() works with live data (no mocking)."""
        loader = _make_loader(module_path, class_name, tmp_path)
        events = loader.list_events()
        assert isinstance(events, list)
        assert len(events) > 0
        for event in events:
            assert _REQUIRED_EVENT_KEYS.issubset(event.keys())
