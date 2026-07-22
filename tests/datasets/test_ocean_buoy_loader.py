# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mercury Agent - Tests for the NOAA NDBC buoy dataset loader.

Exercises :class:`omni_mercury_engine.datasets.ocean.NOAABuoyLoader` end to
end with NO network access. Every fetch goes through
``omni_mercury_engine.datasets.ocean.http_get_with_retry``, which is
monkeypatched to return a small, deterministic synthetic NOAA NDBC
real-time text payload (space-delimited header + units + data rows).

Covered surface:
- construction / class constants / registry wiring
- ``download`` happy path (parse -> process -> cache), cache-hit short
  circuit, all-stations-fail failure and synthetic-fallback branches
- ``_process_buoy_data`` transforms: missing-code replacement,
  physics-bounds masking, short-gap interpolation, rolling-median and
  column-median imputation, the fully-missing-column 0.0 fallback,
  ``max_samples`` subsampling, single-row rate-of-change edge, and the
  empty/no-feature-column error paths
- ``_create_synthetic_fallback``, ``_load_raw`` (real / synthetic /
  missing), ``load_data``, ``preprocess``, ``get_metadata``,
  ``get_statistics``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

try:  # torch is present under MERCURY_REQUIRES_ML=1; guard for thin lanes.
    import torch

    torch.manual_seed(0)
except Exception:  # pragma: no cover - torch always available in the ML lane
    torch = None  # type: ignore[assignment]

from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
from omni_mercury_engine.datasets.ocean import NOAABuoyLoader

HTTP_TARGET = "omni_mercury_engine.datasets.ocean.http_get_with_retry"

# NOAA NDBC real-time .txt files are two header lines (column names, then
# units) followed by whitespace-delimited observation rows. The loader
# reads with ``sep=r"\s+"`` and ``skiprows=[1]`` (units line dropped).
_HEADER = "#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP"
_UNITS = "#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC"

SEED = 20240115


def build_buoy_payload(n_rows: int = 40, seed: int = SEED, inject_anomaly: bool = True) -> bytes:
    """Return a deterministic NDBC real-time text payload as bytes.

    Values are realistic, within the loader's physics bounds, and avoid the
    NDBC missing-value sentinels (0.0, 99, 999, ...) so the rows survive the
    quality filter. When ``inject_anomaly`` is set, one clearly extreme row
    (still inside the physics bounds) is appended so the anomaly detector is
    guaranteed to flag at least one sample.
    """
    rng = np.random.default_rng(seed)
    lines = [_HEADER, _UNITS]
    for i in range(n_rows):
        wdir = int(rng.uniform(10, 350))
        wspd = round(float(abs(rng.normal(5, 1.5)) + 1.0), 1)
        gst = round(wspd + float(abs(rng.normal(1.5, 0.5))), 1)
        wvht = round(float(abs(rng.normal(1.5, 0.4)) + 0.5), 2)
        dpd = round(float(rng.uniform(6, 11)), 1)
        apd = round(float(rng.uniform(4, 7)), 1)
        mwd = int(rng.uniform(100, 300))
        pres = round(float(rng.normal(1013, 4)), 1)
        atmp = round(float(rng.normal(15, 2)), 1)
        wtmp = round(float(rng.normal(14, 1.5)), 1)
        dewp = round(float(rng.normal(10, 2)), 1)
        lines.append(
            f"2024 01 {(i % 28) + 1:02d} 12 00 {wdir} {wspd} {gst} {wvht} "
            f"{dpd} {apd} {mwd} {pres} {atmp} {wtmp} {dewp}"
        )
    if inject_anomaly:
        lines.append("2024 02 01 12 00 200 25.0 40.0 25.0 20.0 20.0 200 1050.0 40.0 40.0 5.0")
    return ("\n".join(lines) + "\n").encode("utf-8")


def make_config(tmp_path: Any, **preprocessing: Any) -> DatasetConfig:
    """Build a DatasetConfig rooted under a unique tmp dir."""
    return DatasetConfig(
        name="noaa_buoy",
        data_dir=str(tmp_path / "data"),
        cache_dir=str(tmp_path / "cache"),
        preprocessing=preprocessing,
    )


def frame_from_features(rows: dict[str, list[float]]) -> pd.DataFrame:
    """Build a buoy DataFrame from a per-column mapping of the feature cols."""
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Construction and class-level contract
# ---------------------------------------------------------------------------


class TestConstruction:
    """Loader construction and configuration wiring."""

    def test_default_stations_and_anomaly_std(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path))
        # Defaults to the first five configured NDBC stations.
        assert loader.stations == list(NOAABuoyLoader.BUOY_STATIONS.keys())[:5]
        assert loader.anomaly_std == 3.0

    def test_custom_stations_and_anomaly_std(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026", "44025"], anomaly_std=2.0))
        assert loader.stations == ["46026", "44025"]
        assert loader.anomaly_std == 2.0

    def test_initial_state_is_empty(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path))
        assert loader._features is None
        assert loader._raw_labels is None
        assert loader.is_real_data is False

    def test_data_path_uses_dataset_name(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path))
        assert loader.data_path.name == "noaa_buoy"
        assert loader.data_path.exists()


class TestClassConstants:
    """Static surface that downstream code and the registry depend on."""

    def test_dataset_identity(self) -> None:
        assert NOAABuoyLoader.DATASET_NAME == "noaa_buoy"
        assert NOAABuoyLoader.REQUIRES_CREDENTIALS is False
        # Labels are manufactured from signal statistics, not ground truth.
        assert NOAABuoyLoader.LABEL_SOURCE == "statistical"
        assert "ndbc.noaa.gov" in NOAABuoyLoader.DATASET_URL

    def test_feature_columns(self) -> None:
        assert NOAABuoyLoader.FEATURE_COLS == [
            "WVHT",
            "DPD",
            "APD",
            "MWD",
            "WTMP",
            "ATMP",
            "PRES",
            "WSPD",
            "GST",
        ]

    def test_base_url_pattern(self) -> None:
        url = NOAABuoyLoader.BASE_URL.format(station="46026")
        assert url == "https://www.ndbc.noaa.gov/data/realtime2/46026.txt"

    def test_missing_value_sentinels_present(self) -> None:
        for sentinel in (99.0, 999.0, 9999.0, 0.0, -99.0):
            assert sentinel in NOAABuoyLoader.MISSING_VALUES


class TestRegistry:
    """Both registry aliases resolve to this loader."""

    def test_registered_under_both_names(self) -> None:
        from omni_mercury_engine.datasets import DatasetRegistry

        assert DatasetRegistry.get("noaa_buoy") is NOAABuoyLoader
        assert DatasetRegistry.get("ocean-buoy") is NOAABuoyLoader


# ---------------------------------------------------------------------------
# download() — happy path
# ---------------------------------------------------------------------------


class TestDownloadHappyPath:
    """Full parse -> process -> cache flow over the mocked HTTP boundary."""

    def test_download_returns_real_data(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026", "46047"]))
        payload = build_buoy_payload()

        calls: list[str] = []

        def fake_get(url: str, **kw: Any) -> bytes:
            calls.append(url)
            return payload

        with patch(HTTP_TARGET, side_effect=fake_get):
            ok = loader.download()

        assert ok is True
        assert loader.is_real_data is True
        # One fetch per requested station.
        assert calls == [
            NOAABuoyLoader.BASE_URL.format(station="46026"),
            NOAABuoyLoader.BASE_URL.format(station="46047"),
        ]

    def test_download_shapes_and_dtypes(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026", "46047"]))
        payload = build_buoy_payload(n_rows=40)

        with patch(HTTP_TARGET, return_value=payload):
            loader.download()

        feats = loader._features
        labels = loader._raw_labels
        assert feats is not None and labels is not None
        # 41 parsed rows per station (40 normal + 1 injected anomaly) x 2.
        assert feats.shape == (82, 9)
        assert feats.dtype == np.float32
        assert labels.shape == (82,)
        assert labels.dtype == np.int64

    def test_download_detects_some_but_not_all_anomalies(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026"]))
        with patch(HTTP_TARGET, return_value=build_buoy_payload()):
            loader.download()

        labels = loader._raw_labels
        assert labels is not None
        assert set(np.unique(labels)).issubset({0, 1})
        # The injected extreme row guarantees >=1; clean rows keep it < all.
        assert 0.0 < labels.mean() < 1.0

    def test_download_writes_real_cache_file(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026"]))
        with patch(HTTP_TARGET, return_value=build_buoy_payload()):
            loader.download()

        cache_file = loader.data_path / "noaa_buoy_real.npz"
        assert cache_file.exists()

    def test_one_bad_station_is_skipped_others_succeed(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026", "46047"]))
        good = build_buoy_payload()

        def fake_get(url: str, **kw: Any) -> bytes:
            if url.endswith("46026.txt"):
                raise RuntimeError("station 46026 offline")
            return good

        with patch(HTTP_TARGET, side_effect=fake_get):
            ok = loader.download()

        # Per-station failures are swallowed; the remaining station is enough.
        assert ok is True
        assert loader.is_real_data is True
        assert loader._features is not None
        assert loader._features.shape == (41, 9)


# ---------------------------------------------------------------------------
# download() — cache short circuit
# ---------------------------------------------------------------------------


class TestDownloadCacheHit:
    """A pre-existing real cache short circuits before any HTTP call."""

    def test_existing_cache_skips_network(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026"]))
        cache_file = loader.data_path / "noaa_buoy_real.npz"
        np.savez_compressed(
            cache_file,
            features=np.ones((5, 9), dtype=np.float32),
            labels=np.zeros(5, dtype=np.int64),
        )

        def explode(url: str, **kw: Any) -> bytes:
            raise AssertionError("network must not be touched on cache hit")

        with patch(HTTP_TARGET, side_effect=explode):
            ok = loader.download()

        assert ok is True
        assert loader.is_real_data is True


# ---------------------------------------------------------------------------
# download() — failure and synthetic-fallback branches
# ---------------------------------------------------------------------------


class TestDownloadFailure:
    """All-stations-fail behaviour under the strict / permissive contract."""

    def test_all_stations_fail_strict_raises(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026"]))

        def boom(url: str, **kw: Any) -> bytes:
            raise RuntimeError("network down")

        with patch(HTTP_TARGET, side_effect=boom), pytest.raises(DataSourceUnavailableError) as exc:
            loader.download()

        assert exc.value.loader_name == "NOAABuoy"
        assert loader.is_real_data is False

    def test_all_stations_fail_synthetic_fallback(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026"], anomaly_std=3.0))
        loader.config.max_samples = 300

        def boom(url: str, **kw: Any) -> bytes:
            raise RuntimeError("network down")

        with patch(HTTP_TARGET, side_effect=boom):
            ok = loader.download()

        assert ok is True
        # Synthetic data is explicitly flagged as not-real.
        assert loader.is_real_data is False
        assert loader._features is not None
        assert loader._raw_labels is not None
        assert loader._features.shape == (300, 9)
        assert loader._raw_labels.dtype == np.int64
        assert (loader.data_path / "synthetic_noaa_buoy.npz").exists()

    def test_no_feature_columns_strict_raises(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Payload parses fine but exposes none of FEATURE_COLS -> the process
        # step raises inside the outer try, which re-raises as unavailable.
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026"]))
        bad = b"#YY MM DD FOO BAR\n#yr mo dy x y\n2024 01 01 1 2\n2024 01 02 3 4\n"

        with patch(HTTP_TARGET, return_value=bad), pytest.raises(DataSourceUnavailableError):
            loader.download()

    def test_processing_error_synthetic_fallback(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Data downloads and parses, but processing raises (no feature cols)
        # inside the OUTER try -> the outer synthetic-fallback branch fires.
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026"]))
        loader.config.max_samples = 120
        bad = b"#YY MM DD FOO BAR\n#yr mo dy x y\n2024 01 01 1 2\n2024 01 02 3 4\n"

        with patch(HTTP_TARGET, return_value=bad):
            ok = loader.download()

        assert ok is True
        assert loader.is_real_data is False
        assert loader._features is not None
        assert loader._features.shape == (120, 9)


class TestPandasUnavailable:
    """The ``download`` guard when pandas cannot be imported."""

    def test_strict_raises_without_pandas(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import omni_mercury_engine.datasets.ocean as ocean_mod

        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
        monkeypatch.setattr(ocean_mod, "PANDAS_AVAILABLE", False)
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026"]))

        with pytest.raises(DataSourceUnavailableError):
            loader.download()

    def test_synthetic_fallback_without_pandas(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import omni_mercury_engine.datasets.ocean as ocean_mod

        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
        monkeypatch.setattr(ocean_mod, "PANDAS_AVAILABLE", False)
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026"]))
        loader.config.max_samples = 100

        ok = loader.download()
        assert ok is True
        assert loader.is_real_data is False
        assert loader._features is not None
        assert loader._features.shape == (100, 9)


# ---------------------------------------------------------------------------
# _process_buoy_data — transform, imputation, labeling
# ---------------------------------------------------------------------------


class TestProcessBuoyData:
    """Direct unit tests for the missing-value / labeling pipeline."""

    def _clean_frame(self, n: int) -> pd.DataFrame:
        return frame_from_features(
            {
                "WVHT": [1.5 + 0.05 * i for i in range(n)],
                "DPD": [8.0 + 0.2 * (i % 4) for i in range(n)],
                "APD": [5.0 + 0.1 * (i % 3) for i in range(n)],
                "MWD": [200.0 + i for i in range(n)],
                "WTMP": [14.0 + 0.1 * (i % 6) for i in range(n)],
                "ATMP": [15.0 + 0.1 * (i % 5) for i in range(n)],
                "PRES": [1013.0 + 0.5 * (i % 4) for i in range(n)],
                "WSPD": [5.0 + 0.2 * (i % 5) for i in range(n)],
                "GST": [7.0 + 0.2 * (i % 5) for i in range(n)],
            }
        )

    def test_clean_frame_roundtrips(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path))
        feats, labels = loader._process_buoy_data(self._clean_frame(20))
        assert feats.shape == (20, 9)
        assert feats.dtype == np.float32
        assert labels.dtype == np.int64
        assert not np.isnan(feats).any()

    def test_missing_codes_and_imputation_branches(self, tmp_path: Any) -> None:
        # Exercises: short-gap interpolation, long-gap rolling median, and a
        # fully-missing column falling through to the 0.0 column-median floor.
        loader = NOAABuoyLoader(make_config(tmp_path))
        df = self._clean_frame(20)
        df.loc[5, "WVHT"] = 999.0  # single NaN -> linear interpolation
        df.loc[8:12, "DPD"] = 999.0  # 5-long gap -> rolling median fill
        df["APD"] = 999.0  # entire column missing -> 0.0 fallback

        feats, _ = loader._process_buoy_data(df)
        assert not np.isnan(feats).any()
        # APD is FEATURE_COLS index 2; its column-median floor is 0.0.
        assert np.allclose(feats[:, 2], 0.0)

    def test_physics_bounds_out_of_range_values_dropped(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path))
        df = self._clean_frame(15)
        df.loc[3, "WVHT"] = 50.0  # > 30 m physics ceiling -> NaN -> imputed
        df.loc[7, "PRES"] = 500.0  # < 870 hPa physics floor -> NaN -> imputed

        feats, _ = loader._process_buoy_data(df)
        assert not np.isnan(feats).any()
        # The out-of-range spike is imputed away, not preserved.
        assert feats[3, 0] != np.float32(50.0)

    def test_single_row_rate_of_change_edge(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path))
        one = frame_from_features(
            {
                "WVHT": [1.5],
                "DPD": [8.0],
                "APD": [5.0],
                "MWD": [200.0],
                "WTMP": [14.0],
                "ATMP": [15.0],
                "PRES": [1013.0],
                "WSPD": [5.0],
                "GST": [7.0],
            }
        )
        feats, labels = loader._process_buoy_data(one)
        assert feats.shape == (1, 9)
        # A lone sample has no rate-of-change and no spread -> not anomalous.
        assert labels.tolist() == [0]

    def test_max_samples_subsampling(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path))
        loader.config.max_samples = 6
        loader.config.random_seed = 42
        df = frame_from_features(
            {
                c: [float(1 + (i % 7)) + 0.01 * i for i in range(20)]
                for c in NOAABuoyLoader.FEATURE_COLS
            }
        )
        feats, labels = loader._process_buoy_data(df)
        assert feats.shape == (6, 9)
        assert labels.shape == (6,)

    def test_small_frame_skips_rolling_median(self, tmp_path: Any) -> None:
        # With < 12 surviving rows the rolling-median window is < 3, so a
        # long (non-interpolatable) gap falls straight through to the
        # column-median imputation strategy.
        loader = NOAABuoyLoader(make_config(tmp_path))
        df = self._clean_frame(8)
        df.loc[2:6, "MWD"] = 999.0  # 5-long gap in an 8-row frame
        feats, _ = loader._process_buoy_data(df)
        assert feats.shape == (8, 9)
        assert not np.isnan(feats).any()

    def test_no_feature_columns_raises(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path))
        with pytest.raises(ValueError, match="No feature columns"):
            loader._process_buoy_data(pd.DataFrame({"FOO": [1, 2], "BAR": [3, 4]}))

    def test_all_missing_raises(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path))
        all_missing = pd.DataFrame({c: [999.0, 999.0, 999.0] for c in NOAABuoyLoader.FEATURE_COLS})
        with pytest.raises(ValueError, match="All data removed"):
            loader._process_buoy_data(all_missing)


# ---------------------------------------------------------------------------
# _create_synthetic_fallback
# ---------------------------------------------------------------------------


class TestSyntheticFallback:
    """Directly generate the synthetic approximation."""

    def test_default_sample_count(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path, anomaly_std=3.0))
        ok = loader._create_synthetic_fallback()
        assert ok is True
        assert loader.is_real_data is False
        # max_samples unset -> the 5000-sample default branch.
        assert loader._features is not None
        assert loader._raw_labels is not None
        assert loader._features.shape == (5000, 9)
        assert loader._features.dtype == np.float32
        assert loader._raw_labels.dtype == np.int64
        assert (loader.data_path / "synthetic_noaa_buoy.npz").exists()

    def test_respects_max_samples(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path, anomaly_std=2.5))
        loader.config.max_samples = 250
        loader._create_synthetic_fallback()
        assert loader._features is not None
        assert loader._raw_labels is not None
        assert loader._features.shape == (250, 9)
        # Synthetic injects ~5% anomalies, so some labels fire but not all.
        assert 0 < int(loader._raw_labels.sum()) < 250


# ---------------------------------------------------------------------------
# _load_raw
# ---------------------------------------------------------------------------


class TestLoadRaw:
    """Cache resolution precedence: real npz > synthetic npz > FileNotFound."""

    def test_loads_real_cache(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path))
        np.savez_compressed(
            loader.data_path / "noaa_buoy_real.npz",
            features=np.ones((5, 9), dtype=np.float32),
            labels=np.zeros(5, dtype=np.int64),
        )
        feats, labels = loader._load_raw()
        assert feats.shape == (5, 9)
        assert loader.is_real_data is True

    def test_loads_synthetic_cache_when_allowed(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
        loader = NOAABuoyLoader(make_config(tmp_path))
        np.savez_compressed(
            loader.data_path / "synthetic_noaa_buoy.npz",
            features=np.ones((3, 9), dtype=np.float32),
            labels=np.zeros(3, dtype=np.int64),
        )
        feats, _ = loader._load_raw()
        assert feats.shape == (3, 9)
        assert loader.is_real_data is False

    def test_synthetic_cache_ignored_when_denied(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
        loader = NOAABuoyLoader(make_config(tmp_path))
        np.savez_compressed(
            loader.data_path / "synthetic_noaa_buoy.npz",
            features=np.ones((3, 9), dtype=np.float32),
            labels=np.zeros(3, dtype=np.int64),
        )
        with pytest.raises(FileNotFoundError):
            loader._load_raw()

    def test_missing_everything_raises(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path))
        with pytest.raises(FileNotFoundError, match="Run download"):
            loader._load_raw()


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    """The public loading entry point."""

    def test_returns_in_memory_after_download(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026"]))
        with patch(HTTP_TARGET, return_value=build_buoy_payload()):
            loader.download()

        # _features already populated -> no cache re-read, no download.
        def explode(url: str, **kw: Any) -> bytes:
            raise AssertionError("load_data must reuse in-memory arrays")

        with patch(HTTP_TARGET, side_effect=explode):
            feats, labels = loader.load_data()
        assert feats is loader._features
        assert labels is loader._raw_labels

    def test_triggers_download_when_uncached(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No cache and no in-memory data: _load_raw() FileNotFound -> download().
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026"]))

        def boom(url: str, **kw: Any) -> bytes:
            raise RuntimeError("network down")

        with patch(HTTP_TARGET, side_effect=boom):
            feats, labels = loader.load_data()

        # Falls back to synthetic through the download() path.
        assert feats.shape[1] == 9
        assert len(feats) == len(labels)
        assert loader.is_real_data is False


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


class TestPreprocess:
    """Normalisation is finite, float32, and roughly zero-mean."""

    def test_handles_nan_and_inf(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path))
        arr = np.array(
            [[1.0, np.nan, np.inf], [2.0, 3.0, -np.inf], [3.0, 4.0, 5.0]],
            dtype=np.float64,
        )
        out = loader.preprocess(arr)
        assert out.dtype == np.float32
        assert np.isfinite(out).all()

    def test_zero_mean_unit_scale(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026"]))
        with patch(HTTP_TARGET, return_value=build_buoy_payload()):
            loader.download()

        assert loader._features is not None
        out = loader.preprocess(loader._features)
        assert out.dtype == np.float32
        assert np.abs(out.mean()) < 0.5


# ---------------------------------------------------------------------------
# get_metadata / get_statistics
# ---------------------------------------------------------------------------


class TestMetadataAndStatistics:
    """Reporting accessors, including their lazy-load trigger."""

    def test_metadata_after_download(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026"]))
        with patch(HTTP_TARGET, return_value=build_buoy_payload()):
            loader.download()

        meta = loader.get_metadata()
        assert loader._features is not None
        assert meta["name"] == "NOAA NDBC Buoy"
        assert meta["n_features"] == 9
        assert meta["n_samples"] == len(loader._features)
        assert meta["feature_names"] == NOAABuoyLoader.FEATURE_COLS
        assert meta["stations"] == ["46026"]
        assert meta["is_real_data"] is True

    def test_metadata_lazy_loads(self, tmp_path: Any) -> None:
        # _features is None on entry -> get_metadata calls load_data().
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026"]))
        with patch(HTTP_TARGET, return_value=build_buoy_payload()):
            meta = loader.get_metadata()
        assert meta["n_samples"] > 0
        assert meta["n_features"] == 9

    def test_statistics_shapes_and_types(self, tmp_path: Any) -> None:
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026"]))
        with patch(HTTP_TARGET, return_value=build_buoy_payload()):
            stats = loader.get_statistics()

        assert loader._features is not None
        assert stats["n_samples"] == len(loader._features)
        assert stats["n_features"] == 9
        assert isinstance(stats["n_anomalies"], int)
        assert isinstance(stats["anomaly_ratio"], float)
        assert 0.0 <= stats["anomaly_ratio"] <= 1.0
        assert set(stats["feature_means"]).issubset(set(NOAABuoyLoader.FEATURE_COLS))
        assert stats["is_real_data"] is True

    def test_statistics_after_download_no_reload(self, tmp_path: Any) -> None:
        # _features already populated -> get_statistics skips the lazy load.
        loader = NOAABuoyLoader(make_config(tmp_path, stations=["46026"]))
        with patch(HTTP_TARGET, return_value=build_buoy_payload()):
            loader.download()

        def explode(url: str, **kw: Any) -> bytes:
            raise AssertionError("get_statistics must not reload after download")

        with patch(HTTP_TARGET, side_effect=explode):
            stats = loader.get_statistics()
        assert loader._features is not None
        assert loader._raw_labels is not None
        assert stats["n_samples"] == len(loader._features)
        assert stats["n_anomalies"] == int(loader._raw_labels.sum())
