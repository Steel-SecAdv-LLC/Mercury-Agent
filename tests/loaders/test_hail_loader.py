# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline HailLoader tests over recorded real SPC data.

Fixtures (provenance in tests/fixtures/meteorological/PROVENANCE.json):

* ``spc_hail_archive_subset.csv`` -- 451 verbatim rows of the SPC
  1955-2023 severe-hail archive, filtered by date to the loader's three
  ground-truth event windows (69 / 238 / 144 rows).
* ``spc_storm_reports_20120629.csv`` -- a full recorded SPC filtered
  daily-report file whose hail section (``Time,Size,...``, sizes in
  hundredths of an inch) exercises the realtime parser.

All HTTP is patched to serve these recorded bytes; nothing here touches
the network (the live paths are covered by
``tests/detectors/test_severe_storm_network.py``).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from omni_mercury_engine.loaders.hail_loader import HailLoader
from omni_mercury_engine.loaders.label_provenance import (
    LABEL_PROVENANCE_REGISTRY,
    audit_label_provenance,
    discover_loaders,
)

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "meteorological"
_ARCHIVE_SUBSET = FIXTURE_DIR / "spc_hail_archive_subset.csv"
_DAILY_REPORTS = FIXTURE_DIR / "spc_storm_reports_20120629.csv"

#: (event_id, n_rows, n_significant, max_size_in) measured from the live
#: archive when the fixture was recorded.
_EVENT_EXPECTATIONS = [
    ("vivian_2010", 69, 4, 8.0),
    ("texas_2016", 238, 67, 5.25),
    ("colorado_2017", 144, 15, 2.75),
]


def _archive_subset_df(*_args, **_kwargs) -> pd.DataFrame:
    """Serve the recorded archive subset in place of the zipped download."""
    df = pd.read_csv(_ARCHIVE_SUBSET, low_memory=False)
    return df


@pytest.fixture
def loader(tmp_path: Path) -> HailLoader:
    """HailLoader with an isolated cache dir and archive HTTP patched out."""
    ld = HailLoader(cache_dir=tmp_path)
    patcher = patch.object(HailLoader, "_fetch_csv", side_effect=_archive_subset_df)
    patcher.start()
    yield ld
    patcher.stop()


class TestInterface:
    """Loader contract and honest label-provenance declaration."""

    def test_declarations(self) -> None:
        assert HailLoader.DOMAIN == "hail"
        assert HailLoader.LABEL_SOURCE == "statistical"
        assert HailLoader.REQUIRES_API_KEY is False
        assert "spc.noaa.gov" in HailLoader.SOURCE_URL
        assert len(HailLoader.FEATURE_COLUMNS) == 11

    def test_list_events_shape(self, loader: HailLoader) -> None:
        events = loader.list_events()
        assert len(events) == 3
        for event in events:
            assert {"event_id", "name", "date", "description"} <= set(event)

    def test_registered_in_label_provenance(self) -> None:
        """The frozen audit carries the hail loader as statistical."""
        entry = LABEL_PROVENANCE_REGISTRY.get("hail_loader.HailLoader")
        assert entry is not None
        assert entry[0] == "statistical"

    def test_provenance_audit_clean_for_hail(self) -> None:
        """The provenance gate finds no leak for the hail loader."""
        loaders = {k: v for k, v in discover_loaders().items() if k == "hail_loader.HailLoader"}
        assert loaders, "discover_loaders() must see HailLoader"
        findings = [
            f for f in audit_label_provenance(loaders) if f.loader == "hail_loader.HailLoader"
        ]
        assert findings == [], [str(f) for f in findings]


class TestHistoricalEvents:
    """Event windows against the recorded archive subset."""

    @pytest.mark.parametrize(("event_id", "n_rows", "n_sig", "max_size"), _EVENT_EXPECTATIONS)
    def test_fetch_and_ground_truth(
        self, loader: HailLoader, event_id: str, n_rows: int, n_sig: int, max_size: float
    ) -> None:
        df = loader.fetch_historical(event_id)
        assert len(df) == n_rows
        assert pd.to_numeric(df["mag"]).max() == pytest.approx(max_size)

        labels = loader.get_ground_truth(event_id)
        assert labels.shape == (n_rows,)
        assert int(labels.sum()) == n_sig
        # Labels are exactly the significant-hail threshold on mag.
        mag = pd.to_numeric(df["mag"], errors="coerce").fillna(-1).values
        np.testing.assert_array_equal(labels, (mag >= 2.0).astype(np.int64))

    def test_windows_are_disjoint_and_sorted(self, loader: HailLoader) -> None:
        df = loader.fetch_historical("texas_2016")
        dates = pd.to_datetime(df["date"])
        assert dates.min() >= pd.Timestamp("2016-04-10")
        assert dates.max() <= pd.Timestamp("2016-04-13")
        assert dates.is_monotonic_increasing

    def test_unknown_event_fails_loud(self, loader: HailLoader) -> None:
        with pytest.raises(ValueError, match="Unknown event_id"):
            loader.fetch_historical("hailmageddon_1899")
        with pytest.raises(ValueError, match="Unknown event_id"):
            loader.get_ground_truth("hailmageddon_1899")

    def test_second_fetch_served_from_cache(self, tmp_path: Path) -> None:
        """The per-event slice is disk-cached; the archive fetch runs once."""
        ld = HailLoader(cache_dir=tmp_path)
        with patch.object(HailLoader, "_fetch_csv", side_effect=_archive_subset_df) as mock_fetch:
            ld.fetch_historical("vivian_2010")
            assert mock_fetch.call_count == 1
            fresh = HailLoader(cache_dir=tmp_path)  # no memoized archive
            fresh.fetch_historical("vivian_2010")
            assert mock_fetch.call_count == 1  # served from disk cache


class TestFeatureEngineering:
    """Feature matrix over the recorded events."""

    def test_shape_and_finiteness(self, loader: HailLoader) -> None:
        df = loader.fetch_historical("texas_2016")
        features = loader.engineer_features(df)
        assert features.shape == (len(df), 11)
        assert np.isfinite(features).all()

    def test_feature0_is_hail_size(self, loader: HailLoader) -> None:
        """feature[0] is the mag column -- the declared label circularity."""
        df = loader.fetch_historical("vivian_2010")
        features = loader.engineer_features(df)
        np.testing.assert_allclose(features[:, 0], pd.to_numeric(df["mag"]).fillna(0).values)
        assert features[:, 0].max() == pytest.approx(8.0)  # the Vivian stone

    def test_temporal_cluster_and_geo_anomaly(self, loader: HailLoader) -> None:
        df = loader.fetch_historical("colorado_2017")
        features = loader.engineer_features(df)
        cluster = features[:, 6]
        geo = features[:, 7]
        assert (cluster >= 0).all()
        assert cluster.max() > 0  # the Front Range storm clusters in space-time
        # The May 2017 storm hit the Denver metro, a few hundred km from
        # the hail-alley centroid used by the feature.
        assert 0.0 < geo.min() < 500.0

    def test_empty_frame(self, loader: HailLoader) -> None:
        features = loader.engineer_features(pd.DataFrame())
        assert features.shape == (0, 11)


class TestRealtimeParser:
    """Daily-report hail-section parsing over the recorded 2012-06-29 file."""

    def test_parses_hail_section_with_inch_conversion(self, tmp_path: Path) -> None:
        ld = HailLoader(cache_dir=tmp_path)
        raw = _DAILY_REPORTS.read_bytes()
        with patch.object(HailLoader, "_fetch_url", return_value=raw):
            df = ld.fetch_realtime()
        assert len(df) > 0
        assert {"mag", "slat", "slon"} <= set(df.columns)
        # Recorded sizes are hundredths of an inch (e.g. 100 -> 1.00 in).
        assert df["mag"].min() >= 0.25
        assert df["mag"].max() <= 5.0
        assert (df["mag"] == 1.0).any()

    def test_missing_hail_section_fails_loud(self, tmp_path: Path) -> None:
        ld = HailLoader(cache_dir=tmp_path)
        tornado_only = (
            b"Time,F_Scale,Location,County,State,Lat,Lon,Comments\n"
            b"1200,UNK,X,Y,KS,38.0,-98.0,c\n"
        )
        with patch.object(HailLoader, "_fetch_url", return_value=tornado_only):
            with pytest.raises(ValueError, match="no hail section"):
                ld.fetch_realtime()


class TestProvenanceMetadata:
    """get_provenance ties results to source URL and data hash."""

    def test_provenance_fields(self, loader: HailLoader) -> None:
        df = loader.fetch_historical("vivian_2010")
        features = loader.engineer_features(df)
        prov = loader.get_provenance("vivian_2010", features)
        assert prov["domain"] == "hail"
        assert prov["event_id"] == "vivian_2010"
        assert prov["source_url"] == HailLoader.SOURCE_URL
        assert prov["data_shape"] == [69, 11]
        assert len(prov["data_hash"]) == 64
