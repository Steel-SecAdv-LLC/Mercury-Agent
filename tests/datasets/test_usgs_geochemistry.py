# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mercury Agent - Tests for USGS NURE-HSSR stream-sediment geochemistry loader.

Covers the real-data parser added on top of the previously-stubbed
``USGSGeochemistryLoader._download_from_usgs``.  The network probe lives
in ``tests/datasets/test_unreachable_loaders_network.py`` (gated by
``MERCURY_NETWORK_TESTS=1``); the tests in this file are offline only and
exercise the parser by feeding it a hand-crafted ZIP that matches the
NURE-HSSR CSV schema.
"""

from __future__ import annotations

import io
import zipfile
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.environmental import USGSGeochemistryLoader
from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError

# Headers in the order the live mrdata.usgs.gov/nure/sediment/nuresed-csv.zip
# uses (2014-12-01 recompile).  We do not need every column the upstream
# publishes — just the eleven the loader extracts plus a few neighbours so
# the schema-drift guard inside the parser is exercised.
_NURE_HEADER = [
    "rec_no",
    "prime_id",
    "replc",
    "samptyp",
    "rec_cnt",
    "latitude",
    "longitude",
    "doelab",
    "laslid",
    "ornlid",
    "srlid",
    "lllid",
    "site",
    "state",
    "ph",
    "as_ppm",
    "pb_ppm",
    "hg_ppm",
    "cd_ppm",
    "cu_ppm",
    "zn_ppm",
    "fe_pct",
    "ca_pct",
]


def _make_zip(rows: list[list[Any]], header: list[str] | None = None) -> bytes:
    """Build an in-memory NURE-style ZIP from a list of row vectors."""
    header = header if header is not None else _NURE_HEADER
    csv_lines = [",".join(header)]
    for row in rows:
        csv_lines.append(",".join("" if v is None else str(v) for v in row))
    csv_bytes = ("\n".join(csv_lines) + "\n").encode("utf-8")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("nuresed.csv", csv_bytes)
        zf.writestr("nurehssr.met", b"NURE-HSSR metadata sidecar\n")
    return buf.getvalue()


def _row(
    *,
    lat: float = 40.0,
    lon: float = -100.0,
    ph: float | str = 6.5,
    as_ppm: float | str = 1.0,
    pb_ppm: float | str = 10.0,
    hg_ppm: float | str = 0.05,
    cd_ppm: float | str = 0.5,
    cu_ppm: float | str = 25.0,
    zn_ppm: float | str = 60.0,
    fe_pct: float | str = 3.5,
    ca_pct: float | str = 1.2,
) -> list[Any]:
    """Build a row matching ``_NURE_HEADER`` column order."""
    return [
        1,  # rec_no
        "P1",  # prime_id
        "",  # replc
        "15",  # samptyp
        1,  # rec_cnt
        lat,
        lon,
        "LASL",  # doelab
        "",  # laslid
        "",  # ornlid
        "",  # srlid
        "",  # lllid
        "",  # site
        "MT",  # state
        ph,
        as_ppm,
        pb_ppm,
        hg_ppm,
        cd_ppm,
        cu_ppm,
        zn_ppm,
        fe_pct,
        ca_pct,
    ]


@pytest.fixture
def config(tmp_path: Any) -> DatasetConfig:
    """Per-test config bounded to a region that includes the test rows."""
    return DatasetConfig(
        name="geochemistry",
        data_dir=str(tmp_path / "data"),
        cache_dir=str(tmp_path / "cache"),
        max_samples=100,
        random_seed=42,
        preprocessing={
            "region": {
                "lat_min": 24,
                "lat_max": 50,
                "lon_min": -125,
                "lon_max": -66,
            },
        },
    )


@pytest.fixture
def loader(config: DatasetConfig) -> USGSGeochemistryLoader:
    return USGSGeochemistryLoader(config)


class TestNUREDownloader:
    """The real-data path that replaces the historical synthetic-only stub."""

    def test_metadata_constants(self, loader: USGSGeochemistryLoader) -> None:
        """Metadata invariants the unreachable-loader harness pins."""
        assert loader.DATASET_NAME == "geochemistry"
        assert loader.DATASET_URL == "https://mrdata.usgs.gov/geochem/"
        assert loader.LICENSE.startswith("Public Domain")
        assert "USGS" in loader.CITATION
        assert loader.REQUIRES_CREDENTIALS is False

    def test_field_map_covers_feature_names(self, loader: USGSGeochemistryLoader) -> None:
        """The NURE field map must cover every published FEATURE_NAMES."""
        assert set(loader._NURE_FIELD_MAP.keys()) == set(loader.FEATURE_NAMES)

    def test_download_writes_real_cache_and_marks_real(
        self, loader: USGSGeochemistryLoader
    ) -> None:
        """Happy-path: a valid NURE ZIP populates the real-data cache."""
        zip_bytes = _make_zip(
            [
                _row(lat=45.0, lon=-100.0, as_ppm=2.0, pb_ppm=20.0, hg_ppm=0.1),
                _row(lat=35.0, lon=-110.0, as_ppm=0.5, pb_ppm=5.0, hg_ppm=0.02),
            ]
        )
        with patch(
            "omni_mercury_engine.datasets.environmental.http_get_with_retry",
            return_value=zip_bytes,
        ):
            ok = loader.download()

        assert ok is True
        assert loader.is_real_data is True
        cache = loader.data_path / "usgs_geochemistry_real.npz"
        assert cache.exists()
        data = np.load(cache)
        assert data["features"].shape == (2, len(loader.FEATURE_NAMES))
        assert data["labels"].shape == (2,)

    def test_below_detection_limit_uses_half_threshold(
        self, loader: USGSGeochemistryLoader
    ) -> None:
        """NURE encodes below-detection as ``-threshold``; loader uses half."""
        # ``-5`` in arsenic means "below 5 ppm"; loader stores 2.5.
        zip_bytes = _make_zip([_row(lat=40.0, lon=-90.0, as_ppm="-5", pb_ppm="-10")])
        with patch(
            "omni_mercury_engine.datasets.environmental.http_get_with_retry",
            return_value=zip_bytes,
        ):
            loader.download()

        features, _ = loader._load_raw()
        # FEATURE_NAMES order: lat, lon, As, Pb, Hg, Cd, Cu, Zn, Fe, Ca, pH
        assert features[0, 2] == pytest.approx(2.5)  # arsenic
        assert features[0, 3] == pytest.approx(5.0)  # lead

    def test_region_filter_drops_out_of_bounds_samples(
        self, loader: USGSGeochemistryLoader
    ) -> None:
        """Continental-US region filter excludes Alaska/Hawaii samples."""
        zip_bytes = _make_zip(
            [
                _row(lat=64.0, lon=-150.0),  # Alaska -> drop
                _row(lat=21.0, lon=-157.0),  # Hawaii -> drop
                _row(lat=40.0, lon=-100.0),  # CONUS -> keep
                _row(lat=35.0, lon=-110.0),  # CONUS -> keep
            ]
        )
        with patch(
            "omni_mercury_engine.datasets.environmental.http_get_with_retry",
            return_value=zip_bytes,
        ):
            loader.download()
        features, _ = loader._load_raw()
        assert features.shape[0] == 2

    def test_max_samples_caps_parsed_rows(self, loader: USGSGeochemistryLoader) -> None:
        """The 235 MB CSV is materialised only up to ``max_samples``."""
        loader.config.max_samples = 3
        zip_bytes = _make_zip([_row(lat=40.0 + i * 0.01) for i in range(10)])
        with patch(
            "omni_mercury_engine.datasets.environmental.http_get_with_retry",
            return_value=zip_bytes,
        ):
            loader.download()
        features, _ = loader._load_raw()
        assert features.shape[0] == 3

    def test_epa_screening_labels_match_synthetic_convention(
        self, loader: USGSGeochemistryLoader
    ) -> None:
        """Anomaly labels follow the same EPA-screening rule as the synthetic path."""
        # Row 0: background — below every screening level.
        # Row 1: arsenic-contaminated — above 0.68 ppm.
        # Row 2: lead-contaminated — above 400 ppm.
        zip_bytes = _make_zip(
            [
                _row(as_ppm=0.1, pb_ppm=5),
                _row(as_ppm=5.0, pb_ppm=5),
                _row(as_ppm=0.1, pb_ppm=500),
            ]
        )
        with patch(
            "omni_mercury_engine.datasets.environmental.http_get_with_retry",
            return_value=zip_bytes,
        ):
            loader.download()
        _, labels = loader._load_raw()
        assert labels.tolist() == [0, 1, 1]

    def test_invalid_ph_becomes_zero(self, loader: USGSGeochemistryLoader) -> None:
        """pH outside [0, 14] is treated as missing."""
        zip_bytes = _make_zip(
            [
                _row(ph=6.5),
                _row(ph=-1.0),  # missing
                _row(ph=15.0),  # missing
                _row(ph=""),  # missing
            ]
        )
        with patch(
            "omni_mercury_engine.datasets.environmental.http_get_with_retry",
            return_value=zip_bytes,
        ):
            loader.download()
        features, _ = loader._load_raw()
        # pH is the last column.
        assert features[0, -1] == pytest.approx(6.5)
        assert features[1, -1] == pytest.approx(0.0)
        assert features[2, -1] == pytest.approx(0.0)
        assert features[3, -1] == pytest.approx(0.0)

    def test_missing_required_column_raises_in_parser(
        self, loader: USGSGeochemistryLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parser refuses to operate on a schema-drifted CSV."""
        # Force-disable synthetic so the failure surfaces as
        # DataSourceUnavailableError rather than the synthetic fallback.
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
        bad_header = [c for c in _NURE_HEADER if c != "as_ppm"]
        bad_row = [v for c, v in zip(_NURE_HEADER, _row(), strict=False) if c != "as_ppm"]
        zip_bytes = _make_zip([bad_row], header=bad_header)
        with (
            patch(
                "omni_mercury_engine.datasets.environmental.http_get_with_retry",
                return_value=zip_bytes,
            ),
            pytest.raises(DataSourceUnavailableError),
        ):
            loader.download()

    def test_empty_zip_falls_back_to_failure(
        self, loader: USGSGeochemistryLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An archive with no CSV member is a parser error, not silent success."""
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("README.txt", b"no csv here\n")
        with (
            patch(
                "omni_mercury_engine.datasets.environmental.http_get_with_retry",
                return_value=buf.getvalue(),
            ),
            pytest.raises(DataSourceUnavailableError),
        ):
            loader.download()

    def test_cached_npz_skips_redownload(self, loader: USGSGeochemistryLoader) -> None:
        """Existing real-data cache short-circuits the network path."""
        cache_file = loader.data_path / "usgs_geochemistry_real.npz"
        loader.data_path.mkdir(parents=True, exist_ok=True)
        n_feat = len(loader.FEATURE_NAMES)
        np.savez_compressed(
            cache_file,
            features=np.zeros((5, n_feat), dtype=np.float32),
            labels=np.zeros(5, dtype=np.int64),
        )
        with patch("omni_mercury_engine.datasets.environmental.http_get_with_retry") as mock_http:
            ok = loader.download()
        assert ok is True
        assert loader.is_real_data is True
        mock_http.assert_not_called()

    def test_network_failure_synthetic_disabled_raises(
        self, loader: USGSGeochemistryLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Network outage + ``MERCURY_ALLOW_SYNTHETIC=0`` -> loud error."""
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
        with (
            patch(
                "omni_mercury_engine.datasets.environmental.http_get_with_retry",
                side_effect=ConnectionError("upstream outage"),
            ),
            pytest.raises(DataSourceUnavailableError),
        ):
            loader.download()

    def test_network_failure_synthetic_enabled_falls_back(
        self, loader: USGSGeochemistryLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Network outage + ``MERCURY_ALLOW_SYNTHETIC=1`` -> synthetic path."""
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
        with patch(
            "omni_mercury_engine.datasets.environmental.http_get_with_retry",
            side_effect=ConnectionError("upstream outage"),
        ):
            ok = loader.download()
        assert ok is True
        # Synthetic path does not flip is_real_data to True.
        assert loader.is_real_data is False
        assert (loader.data_path / "synthetic_geochemistry.npz").exists()
