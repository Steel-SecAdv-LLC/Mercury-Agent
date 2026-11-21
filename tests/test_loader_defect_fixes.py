# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for verified loader and benchmark-harness defect fixes.

Each test pins behaviour that was wrong before the corresponding fix and
would fail against the pre-fix code.  All data is synthetic / mocked: these
loaders fetch live third-party data in production, so every fetch is patched
and no network call is made.

Coverage:

* H1 -- ``engineer_features`` must not permute tied ``datetime64`` rows
  relative to the label order (stable re-sort) in the flood, FEMA and
  pandemic loaders.
* H3 -- ``compute_auc`` returns ``NaN`` (not ``0.5``) for a single class;
  ``run_domain_benchmark`` excludes undefined-AUC events from ``mean_auc``
  and raises on a feature/label length mismatch.
* H4 -- the earthquake historical query carries a circular spatial filter
  and an inclusive end date.
* H5 -- Ebola ``TimeDim`` years parse into the event window (non-empty
  frame); the OWID "World" filter yields a single aggregate series; mpox is
  marked not-benchmarkable instead of serving COVID data.
* H6 -- the volcanic historical path refuses to serve a current alert as a
  past eruption.
* H7 -- the hurricane loader keeps only synoptic (00/06/12/18Z) main-track
  rows so fixed-step deltas and the RI label measure true 24 h.
* M2 -- flood USGS datetimes parse with ``utc=True`` across a DST boundary.
* M4 -- the tornado loader drops per-state segment duplicates (``sg == 1``).
* M5 -- FEMA warns when the record cap truncates; the trailing-count scan is
  exact (no silent look-back cap).
* M6 -- WHO GHO skips missing ``NumericValue`` years; emergency timestamps
  normalise to UTC before dropping the tz.
* M7 -- the network-security BATADAL path refuses an all-normal set when the
  attack file is missing and fills gaps with the column median, not ``0``.
* Catalogue defects -- every hurricane event carries the storm identifier
  IBTrACS holds for it (seven did not, and returned no track), pinned to a
  committed extract of the archive; hurricane frames cached while only
  ``main`` rows were kept are not read back; ``ebola_2014`` and the pandemic
  stubs are listed as not benchmarkable and return nothing; the tornado
  archive file covers the 2024 event, and an unassigned rating is missing,
  not ``-9``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path


# ======================================================================
# H1 -- stable re-sort keeps features aligned with labels on tied keys
# ======================================================================
class TestH1StableSortAlignment:
    """``engineer_features`` must preserve the label row order on ties.

    ``get_ground_truth`` labels the frame in the order ``fetch_historical``
    cached it; ``engineer_features`` re-sorts that same frame.  On a
    ``datetime64`` key with tied timestamps the default introsort is not
    idempotent, so an unstable re-sort permuted tied rows and misaligned
    feature row *i* against label row *i*.  These tests feed a tied-key
    frame through the real loaders and assert zero misalignment.
    """

    def test_flood_multisite_alignment(self, tmp_path: Path) -> None:
        """Flood: two gauges sharing 15-min stamps stay feature/label aligned."""
        from omni_mercury_engine.loaders.flood_loader import FloodLoader

        times = pd.date_range("2024-09-25", periods=400, freq="15min")
        stage_a, stage_b = 11.0, 8.0
        high, low = stage_a + 5.0, stage_b - 5.0
        a = pd.DataFrame(
            {
                "datetime": times,
                "site_id": "03451500",
                "gauge_height_ft": high,
                "discharge_cfs": 1000.0,
            }
        )
        b = pd.DataFrame(
            {
                "datetime": times,
                "site_id": "03439000",
                "gauge_height_ft": low,
                "discharge_cfs": 10.0,
            }
        )
        raw = pd.concat([a, b], ignore_index=True).sort_values("datetime").reset_index(drop=True)

        loader = FloodLoader(cache_dir=tmp_path / "flood", max_retries=0)
        with patch.object(loader, "fetch_historical", return_value=raw):
            labels = loader.get_ground_truth("helene_2024")
        features = loader.engineer_features(raw)
        gauge_feat = features[:, 0]

        # Label 1 rows are the high (site A) gauge; label 0 rows the low one.
        mismatches = int(
            ((labels == 1) & ~np.isclose(gauge_feat, high)).sum()
            + ((labels == 0) & ~np.isclose(gauge_feat, low)).sum()
        )
        assert mismatches == 0, f"{mismatches} feature/label rows misaligned after re-sort"

    def test_fema_tied_date_alignment(self, tmp_path: Path) -> None:
        """FEMA: declarations clustered on one date stay feature/label aligned."""
        from omni_mercury_engine.loaders.fema_loader import FEMALoader

        rows = []
        rid = 0
        for d in range(30):
            date = f"2024-09-{(d % 28) + 1:02d}T00:00:00.000Z"
            for k in range(12):
                is_dr = k % 2 == 0
                rows.append(
                    {
                        "declarationDate": date,
                        "declarationType": "DR" if is_dr else "EM",
                        "state": "FL",
                        "iaProgramDeclared": 1 if is_dr else 0,
                        "ihProgramDeclared": 0,
                        "paProgramDeclared": 1 if is_dr else 0,
                        "hmProgramDeclared": 0,
                        "incidentType": "Hurricane",
                        "fipsStateCode": 12,
                        "fipsCountyCode": rid % 99,
                        "femaDeclarationString": f"DR-{rid}",
                    }
                )
                rid += 1
        raw = pd.DataFrame(rows).sort_values("declarationDate").reset_index(drop=True)

        loader = FEMALoader(cache_dir=tmp_path / "fema", max_retries=0)
        event_id = next(iter(loader.list_events()))["event_id"]
        with patch.object(loader, "fetch_historical", return_value=raw):
            labels = loader.get_ground_truth(event_id)
        features = loader.engineer_features(raw)
        is_major_feat = features[:, 5]  # feature 6: is_major_disaster (1 for DR)

        # Every anomaly (DR+IA+PA) must sit on a major-disaster feature row.
        mismatches = int(((labels == 1) & (is_major_feat != 1.0)).sum())
        assert mismatches == 0, f"{mismatches} FEMA anomaly rows misaligned after re-sort"

    def test_pandemic_world_interleave_alignment(self, tmp_path: Path) -> None:
        """Pandemic: countries interleaved on shared dates keep stable order."""
        from omni_mercury_engine.loaders.pandemic_loader import PandemicLoader

        dates = pd.date_range("2022-05-01", periods=120, freq="D")
        a = pd.DataFrame({"date": dates, "location": "X", "new_cases": 1.0, "new_deaths": 0.0})
        b = pd.DataFrame({"date": dates, "location": "Y", "new_cases": 5000.0, "new_deaths": 0.0})
        raw = pd.concat([a, b], ignore_index=True)
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.sort_values("date").reset_index(drop=True)

        loader = PandemicLoader(cache_dir=tmp_path / "pand", max_retries=0)
        features = loader.engineer_features(raw)
        assert features.shape[0] == len(raw)

        # The same rows with the ties broken by a per-row time offset: no sort
        # can reorder these, so their features are the row-aligned reference.
        # ``engineer_features`` must return the identical matrix for the tied
        # frame; an unstable re-sort permutes the tied rows and its rolling
        # features then differ. (This test used to assert on a frame it had
        # stable-sorted itself, which passed whatever the loader did.)
        untied = raw.copy()
        untied["date"] = untied["date"] + pd.to_timedelta(untied.groupby("date").cumcount(), "s")
        reference = loader.engineer_features(untied)
        np.testing.assert_allclose(features, reference, rtol=0, atol=1e-9)


# ======================================================================
# H3 -- benchmark honesty
# ======================================================================
class TestH3ComputeAucSentinel:
    """``compute_auc`` returns NaN for a single class and still raises on
    length mismatch."""

    def test_single_class_returns_nan(self) -> None:
        """All-one-class input yields NaN, never the attainable 0.5."""
        from benchmarks.domain_benchmark_base import compute_auc

        scores = np.random.default_rng(0).uniform(0, 1, 50)
        assert np.isnan(compute_auc(np.zeros(50), scores))
        assert np.isnan(compute_auc(np.ones(50), scores))

    def test_two_class_still_computes(self) -> None:
        """A genuine two-class case is unaffected (perfect separation = 1.0)."""
        from benchmarks.domain_benchmark_base import compute_auc

        y = np.array([0, 0, 0, 1, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert compute_auc(y, s) == pytest.approx(1.0)

    def test_length_mismatch_raises(self) -> None:
        """Mismatched lengths raise rather than silently truncating."""
        from benchmarks.domain_benchmark_base import compute_auc

        with pytest.raises(ValueError, match="same length"):
            compute_auc(np.array([0, 1, 0]), np.array([0.1, 0.2]))


_torch_available = False
try:  # the real detector requires torch; guard like test_domain_benchmarks
    import torch  # noqa: F401

    _torch_available = True
except ImportError:
    pass


class _SingleClassLoader:
    """Minimal loader whose ground truth is all-one-class (undefined AUC)."""

    SOURCE_URL = "https://mock.example.com/data"
    LABEL_DERIVED_FEATURES: tuple[str, ...] = ()
    LABEL_FEATURE_DISJOINT = True

    def __init__(self, n_samples: int = 60, gt_value: int = 1) -> None:
        self._n = n_samples
        self._gt_value = gt_value

    def list_events(self) -> list[dict[str, Any]]:
        return [{"event_id": "single_class_event"}, {"event_id": "two_class_event"}]

    def fetch_historical(self, event_id: str) -> np.ndarray:
        return np.random.default_rng(abs(hash(event_id)) % 2**31).standard_normal((self._n, 3))

    def engineer_features(self, raw: np.ndarray) -> np.ndarray:
        return raw

    def engineer_scored_features(self, raw: np.ndarray) -> np.ndarray:
        return raw

    def get_ground_truth(self, event_id: str) -> np.ndarray:
        if event_id == "single_class_event":
            return np.full(self._n, self._gt_value, dtype=np.int64)
        gt = np.zeros(self._n, dtype=np.int64)
        gt[: self._n // 2] = 1
        return gt

    def get_provenance(self, event_id: str, features: np.ndarray) -> dict[str, Any]:
        return {"source": "mock", "event_id": event_id}


class _MismatchLoader(_SingleClassLoader):
    """Loader whose feature count never matches its label count."""

    def list_events(self) -> list[dict[str, Any]]:
        return [{"event_id": "mismatch_event"}]

    def get_ground_truth(self, event_id: str) -> np.ndarray:
        # One fewer label than feature rows -> must raise, not truncate.
        gt = np.zeros(self._n - 5, dtype=np.int64)
        gt[: (self._n - 5) // 2] = 1
        return gt


@pytest.mark.skipif(not _torch_available, reason="torch not available")
class TestH3HarnessExclusion:
    """``run_domain_benchmark`` excludes undefined-AUC events and raises on
    length mismatch."""

    @patch("omni_mercury_engine.detectors.statistical.MercuryAnomalyDetector")
    def test_single_class_excluded_from_mean(self, mock_cls: Any, tmp_path: Path) -> None:
        """A single-class event is reported undefined and left out of mean_auc."""
        from benchmarks.domain_benchmark_base import run_domain_benchmark

        inst = MagicMock()
        inst.fit.return_value = inst
        inst.detect.return_value = {
            "scores": np.linspace(0.0, 1.0, 60),
            "is_anomaly": np.array([False] * 40 + [True] * 20),
        }
        mock_cls.return_value = inst

        loader = _SingleClassLoader()
        result = run_domain_benchmark("h3", loader, output_path=tmp_path / "r.json")

        single = result["events"]["single_class_event"]
        assert single["status"] == "auc_undefined"
        assert single["auc"] is None
        # mean_auc is computed over the two-class event only.
        assert result["summary"]["events_with_defined_auc"] == 1
        assert result["summary"]["events_auc_undefined"] == 1
        assert result["summary"]["mean_auc"] is not None
        assert not np.isnan(result["summary"]["mean_auc"])

    @patch("omni_mercury_engine.detectors.statistical.MercuryAnomalyDetector")
    def test_all_single_class_mean_is_none(self, mock_cls: Any, tmp_path: Path) -> None:
        """If no event has a defined AUC, mean_auc is None, not a number."""
        from benchmarks.domain_benchmark_base import run_domain_benchmark

        inst = MagicMock()
        inst.fit.return_value = inst
        inst.detect.return_value = {
            "scores": np.linspace(0.0, 1.0, 60),
            "is_anomaly": np.array([False] * 60),
        }
        mock_cls.return_value = inst

        loader = _SingleClassLoader()
        loader.list_events = lambda: [{"event_id": "single_class_event"}]  # type: ignore[method-assign]
        result = run_domain_benchmark("h3b", loader, output_path=tmp_path / "r.json")
        assert result["summary"]["mean_auc"] is None
        assert result["summary"]["events_with_defined_auc"] == 0

    @patch("omni_mercury_engine.detectors.statistical.MercuryAnomalyDetector")
    def test_length_mismatch_errors_event(self, mock_cls: Any, tmp_path: Path) -> None:
        """A feature/label length mismatch is surfaced as an error, not truncated."""
        from benchmarks.domain_benchmark_base import run_domain_benchmark

        inst = MagicMock()
        inst.fit.return_value = inst
        inst.detect.return_value = {
            "scores": np.linspace(0.0, 1.0, 60),
            "is_anomaly": np.array([False] * 60),
        }
        mock_cls.return_value = inst

        loader = _MismatchLoader()
        with pytest.raises(SystemExit):
            # The only event errors out, so no event is benchmarked -> exit(1).
            run_domain_benchmark("h3c", loader, output_path=tmp_path / "r.json")


# ======================================================================
# H4 -- earthquake spatial filter + inclusive end
# ======================================================================
class TestH4EarthquakeSpatialQuery:
    """The historical query must bound the catalogue spatially and cover the
    final day."""

    def _capture(self, tmp_path: Path, event_id: str) -> dict[str, str]:
        from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader

        captured: dict[str, dict[str, str]] = {}

        def fake_json(url: str, params: dict[str, str] | None = None) -> dict[str, Any]:
            captured["params"] = params or {}
            return {"type": "FeatureCollection", "features": []}

        loader = EarthquakeLoader(cache_dir=tmp_path / "eq", max_retries=0)
        with patch.object(loader, "_fetch_json", side_effect=fake_json):
            loader.fetch_historical(event_id)
        return captured["params"]

    def test_spatial_filter_present(self, tmp_path: Path) -> None:
        """Query carries latitude/longitude/maxradiuskm (was temporal-only)."""
        params = self._capture(tmp_path, "turkey_syria_2023")
        for key in ("latitude", "longitude", "maxradiuskm"):
            assert key in params, f"missing spatial param {key!r}"
        assert float(params["maxradiuskm"]) > 0

    def test_end_date_inclusive(self, tmp_path: Path) -> None:
        """endtime reaches the end of the final day, not midnight at its start."""
        params = self._capture(tmp_path, "turkey_syria_2023")
        assert params["endtime"].startswith("2023-02-08")
        assert params["endtime"] != "2023-02-08"  # not bare midnight
        assert "T23:59:59" in params["endtime"]

    def test_all_catalog_events_have_coords(self) -> None:
        """Every catalog event defines a real epicenter and radius."""
        from omni_mercury_engine.loaders.earthquake_loader import _EVENT_CATALOG

        for eid, ev in _EVENT_CATALOG.items():
            assert {"latitude", "longitude", "max_radius_km"} <= set(ev), eid
            assert -90 <= ev["latitude"] <= 90
            assert -180 <= ev["longitude"] <= 180


# ======================================================================
# H5 -- pandemic Ebola year parsing, World filter, mpox honesty
# ======================================================================
class TestH5Pandemic:
    """Ebola years parse into the window; World returns one series; mpox is a
    stub."""

    def test_ebola_timedim_year_parsing(self, tmp_path: Path) -> None:
        """Integer TimeDim years yield in-window rows (was an empty frame)."""
        from omni_mercury_engine.loaders.pandemic_loader import _EVENT_CATALOG, PandemicLoader

        loader = PandemicLoader(cache_dir=tmp_path / "pand", max_retries=0)
        gho = {
            "value": [
                {"TimeDim": 2014, "SpatialDim": "GIN", "NumericValue": 2400},
                {"TimeDim": 2014, "SpatialDim": "LBR", "NumericValue": 3600},
            ]
        }
        with patch.object(loader, "_fetch_json", return_value=gho):
            df = loader._fetch_who_gho_event(_EVENT_CATALOG["ebola_2014"])
        assert not df.empty
        assert (df["date"] >= pd.Timestamp("2014-03-01")).all()
        assert (df["date"] <= pd.Timestamp("2014-12-31")).all()

    def test_world_filter_single_series(self, tmp_path: Path) -> None:
        """location=='World' returns only the World aggregate rows."""
        from omni_mercury_engine.loaders.pandemic_loader import PandemicLoader

        owid = pd.DataFrame(
            {
                "date": ["2022-05-01"] * 3 + ["2022-05-02"] * 3,
                "location": ["World", "United States", "Spain"] * 2,
                "new_cases": [100, 50, 30, 110, 55, 33],
            }
        )
        loader = PandemicLoader(cache_dir=tmp_path / "pand2", max_retries=0)
        with patch.object(loader, "_load_owid_data", return_value=owid):
            out = loader._fetch_owid_event(
                {"location": "World", "start": "2022-05-01", "end": "2022-12-31"}
            )
        assert out["location"].unique().tolist() == ["World"]
        assert len(out) == 2

    def test_mpox_is_not_benchmarkable(self, tmp_path: Path) -> None:
        """mpox_2022 is a stub (no COVID-as-mpox) and returns an empty frame."""
        from omni_mercury_engine.loaders.pandemic_loader import _EVENT_CATALOG, PandemicLoader

        assert _EVENT_CATALOG["mpox_2022"]["source"] == "stub"
        loader = PandemicLoader(cache_dir=tmp_path / "pand3", max_retries=0)
        # Must not touch the OWID COVID CSV at all.
        with patch.object(
            loader, "_load_owid_data", side_effect=AssertionError("must not load OWID")
        ):
            df = loader.fetch_historical("mpox_2022")
        assert df.empty

    def test_who_gho_skips_missing_numeric_value(self, tmp_path: Path) -> None:
        """A missing NumericValue is skipped, not coerced to a zero-case year."""
        from omni_mercury_engine.loaders.pandemic_loader import PandemicLoader

        loader = PandemicLoader(cache_dir=tmp_path / "pand4", max_retries=0)
        gho = {
            "value": [
                {"TimeDim": 2017, "SpatialDim": "YEM", "NumericValue": 1200},
                {"TimeDim": 2018, "SpatialDim": "YEM", "NumericValue": None},
            ]
        }
        with patch.object(loader, "_fetch_json", return_value=gho):
            df = loader._fetch_who_gho("WHS3_41", "YEM", "2017-01-01", "2018-12-31")
        # Only 2017 rows survive; no fabricated zero-case 2018 months.
        assert df["date"].dt.year.unique().tolist() == [2017]


# ======================================================================
# H6 -- volcanic historical must not serve a current alert as the past event
# ======================================================================
class TestH6Volcanic:
    """A present-day alert is refused for a past eruption window."""

    def test_current_alert_not_served_as_past_event(self, tmp_path: Path) -> None:
        """Today's Kilauea alert must not be returned as the 2018 eruption."""
        from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
        from omni_mercury_engine.loaders.volcanic_loader import VolcanicLoader

        today = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        alerts = {
            "alerts": [
                {
                    "volcanoName": "Kilauea",
                    "alertLevel": "WATCH",
                    "colorCode": "ORANGE",
                    "alertDate": today,
                    "vnum": "332010",
                }
            ]
        }
        loader = VolcanicLoader(cache_dir=tmp_path / "volc", max_retries=0)
        with (
            patch.object(loader, "_fetch_json", return_value=alerts),
            pytest.raises(DataSourceUnavailableError),
        ):
            loader.fetch_historical("kilauea_2018")

    def test_in_window_alert_accepted(self, tmp_path: Path) -> None:
        """An alert actually dated within the event window is accepted."""
        from omni_mercury_engine.loaders.volcanic_loader import VolcanicLoader

        alerts = {
            "alerts": [
                {
                    "volcanoName": "Kilauea",
                    "alertLevel": "WARNING",
                    "colorCode": "RED",
                    "alertDate": "2018-05-15T00:00:00Z",
                    "vnum": "332010",
                }
            ]
        }
        loader = VolcanicLoader(cache_dir=tmp_path / "volc2", max_retries=0)
        with patch.object(loader, "_fetch_json", return_value=alerts):
            df = loader.fetch_historical("kilauea_2018")
        assert len(df) == 1


# ======================================================================
# H7 -- hurricane synoptic main-track filter
# ======================================================================
class TestH7Hurricane:
    """Off-hour and spur rows are dropped so fixed-step deltas mean 24 h."""

    def _mixed_track(self) -> pd.DataFrame:
        rows = []
        base = pd.Timestamp("2005-08-25 00:00:00")
        winds = [50, 55, 60, 95, 100, 110, 120]
        for i in range(7):
            t = base + pd.Timedelta(hours=6 * i)
            rows.append(
                {
                    "sid": "S1",
                    "season": 2005,
                    "name": "TEST",
                    "iso_time": str(t),
                    "lat": 25.0,
                    "lon": -80.0,
                    "wind_kt": winds[i],
                    "pressure_mb": 990,
                    "nature": "TS",
                    "track_type": "main",
                }
            )
            if i < 6:  # off-hour 3-hourly main row
                rows.append(
                    {
                        "sid": "S1",
                        "season": 2005,
                        "name": "TEST",
                        "iso_time": str(t + pd.Timedelta(hours=3)),
                        "lat": 25.1,
                        "lon": -80.1,
                        "wind_kt": winds[i] + 2,
                        "pressure_mb": 989,
                        "nature": "TS",
                        "track_type": "main",
                    }
                )
            rows.append(  # spur duplicate of the synoptic fix
                {
                    "sid": "S1",
                    "season": 2005,
                    "name": "TEST",
                    "iso_time": str(t),
                    "lat": 25.0,
                    "lon": -80.0,
                    "wind_kt": winds[i],
                    "pressure_mb": 990,
                    "nature": "TS",
                    "track_type": "spur",
                }
            )
        return pd.DataFrame(rows)

    def test_only_synoptic_main_rows_survive(self, tmp_path: Path) -> None:
        """20 mixed rows reduce to 7 synoptic main-track fixes."""
        from omni_mercury_engine.loaders.hurricane_loader import HurricaneLoader

        loader = HurricaneLoader(cache_dir=tmp_path / "hur", max_retries=0)
        raw = self._mixed_track()
        with patch.object(loader, "_fetch_storm_track", return_value=raw):
            df = loader.fetch_historical("katrina_2005")
        assert len(df) == 7
        assert df["track_type"].unique().tolist() == ["main"]
        hours = sorted(pd.to_datetime(df["iso_time"]).dt.hour.unique().tolist())
        assert hours == [0, 6, 12, 18]

    @pytest.mark.parametrize("track_type", ["PROVISIONAL", "US-PROVISIONAL", "main", " Main "])
    def test_provisional_tracks_are_the_storms_own_track(self, track_type: str) -> None:
        """A storm not yet post-analysed is kept; only ``spur`` rows are secondary.

        IBTrACS files current-season storms under ``PROVISIONAL`` /
        ``US-PROVISIONAL`` until they are re-analysed. Keeping only ``main``
        removed every one of them, so the realtime read stopped months back.
        """
        from omni_mercury_engine.loaders.hurricane_loader import HurricaneLoader

        frame = self._mixed_track()
        own_track = frame["track_type"] == "main"
        frame.loc[own_track, "track_type"] = track_type
        kept = HurricaneLoader._filter_synoptic_main_track(frame)
        assert len(kept) == 7
        assert set(kept["track_type"]) == {track_type}

    @pytest.mark.parametrize("spur", ["spur", "US-PROVISIONAL_spur", "PROVISIONAL_spur"])
    def test_every_kind_of_spur_is_dropped(self, spur: str) -> None:
        from omni_mercury_engine.loaders.hurricane_loader import HurricaneLoader

        frame = self._mixed_track()
        frame.loc[frame["track_type"] == "spur", "track_type"] = spur
        kept = HurricaneLoader._filter_synoptic_main_track(frame)
        assert len(kept) == 7
        assert not kept["track_type"].str.lower().str.endswith("spur").any()

    def test_ri_measured_over_true_24h(self, tmp_path: Path) -> None:
        """After filtering, a +50 kt / 24 h climb is labelled rapid-intensifying."""
        from omni_mercury_engine.loaders.hurricane_loader import HurricaneLoader

        loader = HurricaneLoader(cache_dir=tmp_path / "hur2", max_retries=0)
        raw = self._mixed_track()
        with patch.object(loader, "_fetch_storm_track", return_value=raw):
            labels = loader.get_ground_truth("katrina_2005")
        # 7 synoptic rows; wind[4]-wind[0] = 100-50 = 50 kt over 24 h -> RI.
        assert len(labels) == 7
        assert labels.sum() > 0


# ======================================================================
# M2 -- flood mixed-timezone datetime parse
# ======================================================================
class TestM2FloodMixedTimezone:
    """A DST-spanning USGS series must parse (utc=True) instead of raising."""

    def test_dst_boundary_series_parses(self, tmp_path: Path) -> None:
        """Offsets flipping EDT->EST mid-series no longer raise 'Mixed timezones'."""
        from omni_mercury_engine.loaders.flood_loader import FloodLoader

        # USGS WaterServices IV JSON shape with a DST fallback in the window.
        usgs = {
            "value": {
                "timeSeries": [
                    {
                        "sourceInfo": {"siteName": "Test", "siteCode": [{"value": "03451500"}]},
                        "variable": {"variableCode": [{"value": "00065"}]},
                        "values": [
                            {
                                "value": [
                                    {"dateTime": "2018-11-04T00:45:00.000-04:00", "value": "5.0"},
                                    {"dateTime": "2018-11-04T01:00:00.000-04:00", "value": "5.1"},
                                    {"dateTime": "2018-11-04T01:00:00.000-05:00", "value": "5.2"},
                                    {"dateTime": "2018-11-04T02:00:00.000-05:00", "value": "5.3"},
                                ]
                            }
                        ],
                    }
                ]
            }
        }
        loader = FloodLoader(cache_dir=tmp_path / "flood", max_retries=0)
        with patch.object(loader, "_fetch_json", return_value=usgs):
            df = loader._fetch_sites(site_ids=["03451500"], period="P7D")
        assert not df.empty
        assert str(df["datetime"].dtype).startswith("datetime64")
        # tz-aware, monotonic in true UTC order.
        assert df["datetime"].is_monotonic_increasing


# ======================================================================
# M4 -- tornado multi-state segment dedup
# ======================================================================
class TestM4TornadoDedup:
    """Per-state segment rows (sg != 1) must not double-count a tornado."""

    def test_whole_track_only(self, tmp_path: Path) -> None:
        """A multi-state EF4 appears once, not once per state segment."""
        from omni_mercury_engine.loaders.tornado_loader import _EVENT_CATALOG, TornadoLoader

        evt = next(iter(_EVENT_CATALOG))
        day = pd.Timestamp(_EVENT_CATALOG[evt]["start"]).strftime("%Y-%m-%d")
        archive = pd.DataFrame(
            [
                {
                    "date": day,
                    "time": "12:00:00",
                    "mag": 4,
                    "sg": 1,
                    "slat": 35,
                    "slon": -90,
                    "elat": 36,
                    "elon": -89,
                    "len": 50,
                    "wid": 800,
                    "st": "AR",
                },
                {
                    "date": day,
                    "time": "12:00:00",
                    "mag": 4,
                    "sg": 2,
                    "slat": 35,
                    "slon": -90,
                    "elat": 35.5,
                    "elon": -89.5,
                    "len": 25,
                    "wid": 800,
                    "st": "AR",
                },
                {
                    "date": day,
                    "time": "12:30:00",
                    "mag": 4,
                    "sg": 3,
                    "slat": 35.5,
                    "slon": -89.5,
                    "elat": 36,
                    "elon": -89,
                    "len": 25,
                    "wid": 800,
                    "st": "TN",
                },
                {
                    "date": day,
                    "time": "14:00:00",
                    "mag": 1,
                    "sg": 1,
                    "slat": 40,
                    "slon": -100,
                    "elat": 40.1,
                    "elon": -99.9,
                    "len": 5,
                    "wid": 100,
                    "st": "KS",
                },
            ]
        )
        loader = TornadoLoader(cache_dir=tmp_path / "tor", max_retries=0)
        with patch.object(loader, "_load_archive", return_value=archive):
            df = loader.fetch_historical(evt)
            labels = loader.get_ground_truth(evt)
        assert len(df) == 2  # one whole-track EF4 + one EF1
        assert set(pd.to_numeric(df["sg"]).tolist()) == {1}
        assert int(labels.sum()) == 1  # EF3+ counted once, not three times


# ======================================================================
# M5 -- FEMA truncation warning and exact trailing-count scan
# ======================================================================
class TestM5Fema:
    """The record cap warns on truncation; the trailing count has no silent
    look-back cap."""

    def test_cap_truncation_warns(self, tmp_path: Path, caplog: Any) -> None:
        """Hitting the record cap on a full final page logs a warning."""
        import logging

        from omni_mercury_engine.loaders import fema_loader as fl
        from omni_mercury_engine.loaders.fema_loader import _PAGE_SIZE, FEMALoader

        # Always return a full page so pagination runs until the cap.
        full_page = {
            "DisasterDeclarationsSummaries": [
                {
                    "declarationDate": "2024-09-26T00:00:00.000Z",
                    "declarationType": "DR",
                    "state": "FL",
                    "incidentType": "Hurricane",
                    "paProgramDeclared": True,
                    "iaProgramDeclared": True,
                    "disasterNumber": i,
                }
                for i in range(_PAGE_SIZE)
            ]
        }
        loader = FEMALoader(cache_dir=tmp_path / "fema", max_retries=0)
        event_id = next(iter(loader.list_events()))["event_id"]
        with (
            patch.object(loader, "_fetch_fema_json", return_value=full_page),
            caplog.at_level(logging.WARNING, logger=fl.logger.name),
        ):
            loader.fetch_historical(event_id)
        assert any(
            "cap" in r.message.lower() and "truncat" in r.message.lower() for r in caplog.records
        ), "expected a truncation warning"

    def test_trailing_count_exact_beyond_old_cap(self) -> None:
        """Trailing-12-month count is exact even with > the old 5000-row window."""
        from omni_mercury_engine.loaders.fema_loader import FEMALoader

        n = 5200  # exceeds the removed i-5000 look-back cap
        base = pd.Timestamp("2020-01-01", tz="UTC")
        # All within one day -> every prior row is inside the 12-month window.
        df = pd.DataFrame(
            {
                "declarationDate": [base + pd.Timedelta(seconds=i) for i in range(n)],
                "state": ["FL"] * n,
            }
        )
        _, national = FEMALoader._compute_trailing_counts(df)
        # The last row must see all n-1 priors, not a capped 5000.
        assert national[-1] == float(n - 1)

    @pytest.mark.timeout(60)
    def test_trailing_counts_scale_with_the_number_of_declarations(self) -> None:
        """Sixty thousand declarations inside one window are counted at once.

        The count used to compare every row with every row inside its
        window: 5,200 declarations in one year took over five minutes of
        pandas lookups on a development machine and passed the suite's 300 s
        limit on the hosted runner (2026-11-21, ML Tests). At that rate this
        frame would take hours. Rows in date order are now counted by rank
        and binary search.
        """
        from omni_mercury_engine.loaders.fema_loader import FEMALoader

        n = 60_000
        base = pd.Timestamp("2020-01-01", tz="UTC")
        df = pd.DataFrame(
            {
                "declarationDate": [base + pd.Timedelta(seconds=i) for i in range(n)],
                "state": ["FL", "TX", "CA"] * (n // 3),
            }
        )
        same_state, national = FEMALoader._compute_trailing_counts(df)
        assert np.array_equal(national, np.arange(n, dtype=np.float64))
        assert np.array_equal(same_state, np.repeat(np.arange(n // 3, dtype=np.float64), 3))

    def test_trailing_counts_equal_the_backward_scan(self) -> None:
        """Rank-and-search counts are the backward scan's counts, row for row.

        ``_trailing_counts_by_scan`` is the definition: from each row walk to
        earlier rows, skip the undated, stop at the first one older than
        twelve months, count the rest. Frames here have ties, rows exactly on
        the window edge, undated rows, labels that are not strings, spans
        from minutes to years, and (one in ten) rows out of date order, which
        the loader's sort never produces and which take the scan itself.
        """
        from omni_mercury_engine.loaders.fema_loader import FEMALoader

        rng = np.random.default_rng(7)
        base = pd.Timestamp("2015-01-01", tz="UTC")
        out_of_order = 0
        for trial in range(150):
            n = int(rng.integers(0, 160))
            span_days = float(rng.choice([0.01, 30.0, 400.0, 2500.0]))
            offsets = np.sort(rng.uniform(0.0, span_days, n))
            if trial % 3 == 0:
                offsets = np.sort(np.round(offsets / 73.0) * 73.0)  # ties; 5 x 73 = 365
            df = pd.DataFrame(
                {
                    "declarationDate": [base + pd.Timedelta(days=float(o)) for o in offsets],
                    "state": list(rng.choice(np.array(["FL", "TX", "CA", None, 12], object), n)),
                }
            )
            if trial % 4 == 0 and n > 3:
                undated = rng.choice(n, size=max(1, n // 7), replace=False)
                df.loc[undated, "declarationDate"] = pd.NaT
            if trial % 10 == 9 and n > 3:
                df = df.sample(frac=1.0, random_state=trial).reset_index(drop=True)
                out_of_order += 1
            dates = pd.to_datetime(df["declarationDate"], errors="coerce", utc=True)
            want_state, want_national = FEMALoader._trailing_counts_by_scan(
                dates, df["state"].values
            )
            got_state, got_national = FEMALoader._compute_trailing_counts(df)
            assert np.array_equal(got_national, want_national), f"national differs, frame {trial}"
            assert np.array_equal(got_state, want_state), f"same-state differs, frame {trial}"
        assert out_of_order >= 10

    def test_trailing_counts_stop_at_twelve_months(self) -> None:
        """A declaration exactly 365 days earlier counts; one second older does not."""
        from omni_mercury_engine.loaders.fema_loader import FEMALoader

        now = pd.Timestamp("2024-06-01", tz="UTC")
        df = pd.DataFrame(
            {
                "declarationDate": [
                    now - pd.Timedelta(days=365, seconds=1),
                    now - pd.Timedelta(days=365),
                    now - pd.Timedelta(days=10),
                    now,
                ],
                "state": ["TX", "FL", "TX", "FL"],
            }
        )
        same_state, national = FEMALoader._compute_trailing_counts(df)
        assert national.tolist() == [0.0, 1.0, 2.0, 2.0]
        assert same_state.tolist() == [0.0, 0.0, 1.0, 1.0]


# ======================================================================
# M6 -- WHO emergency timestamp UTC normalisation
# ======================================================================
class TestM6PandemicTimezone:
    """tz-aware emergency dates convert to UTC before the tz is dropped."""

    def test_offset_converted_to_utc(self, tmp_path: Path) -> None:
        """An alert's UTC instant decides its month, not its local wall time.

        ``2020-06-30T22:00:00-09:00`` is ``2020-07-01T07:00Z`` -- the UTC
        instant is in July.  With the fix (convert to UTC, then drop tz) the
        alert lands in the July bin.  The old ``tz_localize(None)`` kept the
        local wall time (June 30), binning it into June (where it was then
        lost to the month-end boundary) -- so the July count would be 0.
        """
        from omni_mercury_engine.loaders.pandemic_loader import PandemicLoader

        loader = PandemicLoader(cache_dir=tmp_path / "pand", max_retries=0)
        data = {
            "value": [
                {"EmergencyStartDate": "2020-06-30T22:00:00-09:00", "Title": "X"},
            ]
        }
        with patch.object(loader, "_fetch_json", return_value=data):
            df = loader._fetch_who_emergencies("2020-01-01", "2020-12-31")
        assert not df.empty
        july = df[df["date"] == pd.Timestamp("2020-07-01")]
        assert float(july["alert_count"].iloc[0]) == 1.0


# ======================================================================
# M7 -- network-security BATADAL honesty
# ======================================================================
class TestM7NetworkSecurity:
    """A missing attack file raises; gaps fill with the column median."""

    def test_missing_attack_file_raises(self, tmp_path: Path) -> None:
        """Only the attack-free train file available -> refuse, don't serve."""
        from omni_mercury_engine.loaders.network_security_loader import NetworkSecurityLoader

        train_csv = "DATETIME,S1,S2,ATT_FLAG\n2017-01-01,1.0,2.0,0\n2017-01-02,1.1,2.1,0\n"

        def fake_fetch(url: str, *a: Any, **k: Any) -> bytes:
            if "dataset04" in url:  # the attack/test file
                raise ConnectionError("attack file unavailable")
            return train_csv.encode("utf-8")

        loader = NetworkSecurityLoader(cache_dir=tmp_path / "ns", max_retries=0)
        with (
            patch.object(loader, "_fetch_url", side_effect=fake_fetch),
            pytest.raises(ConnectionError, match="attack"),
        ):
            loader._download_batadal_direct()

    def test_non_numeric_filled_with_median_not_zero(self, tmp_path: Path) -> None:
        """A corrupt cell becomes the column median, never a fake 0.0 reading."""
        from omni_mercury_engine.loaders.network_security_loader import NetworkSecurityLoader

        train_csv = "DATETIME,S1,ATT_FLAG\n2017-01-01,10.0,0\n2017-01-02,12.0,0\n"
        # Column S1 values 10, 12, 14, then a corrupt cell -> median of {10,12,14,20}=13.
        test_csv = (
            "DATETIME,S1,ATT_FLAG\n"
            "2017-02-01,14.0,1\n"
            "2017-02-02,BAD,1\n"
            "2017-02-03,20.0,0\n"
        )

        def fake_fetch(url: str, *a: Any, **k: Any) -> bytes:
            return (test_csv if "dataset04" in url else train_csv).encode("utf-8")

        loader = NetworkSecurityLoader(cache_dir=tmp_path / "ns2", max_retries=0)
        with patch.object(loader, "_fetch_url", side_effect=fake_fetch):
            df = loader._download_batadal_direct()
        # The corrupt row's S1 must be the finite median, not 0.0.
        s1 = df["S1"].to_numpy(dtype=float)
        assert 0.0 not in s1
        assert np.isfinite(s1).all()
        assert int(df["label"].sum()) >= 1


# ======================================================================
# M1 -- financial loader: DFF funding-rate fallback, no weekend rows,
# forward-fill only (no future leakage)
# ======================================================================
class TestM1Financial:
    """Pre-2018 funding rate comes from DFF; the merge is business-day only
    and never fills a value from a later date."""

    def _frames(self, start: str, end: str, sofr_empty: bool) -> dict[str, pd.DataFrame]:
        bdays = pd.bdate_range(start, end)
        alldays = pd.date_range(start, end, freq="D")

        def make(col: str, dates: Any, values: Any) -> pd.DataFrame:
            return pd.DataFrame({"date": pd.to_datetime(dates), col: values})

        sofr = (
            pd.DataFrame(columns=["date", "funding_rate"])
            if sofr_empty
            else make("funding_rate", bdays, np.linspace(1.0, 2.0, len(bdays)))
        )
        return {
            "VIXCLS": make("VIX", bdays, np.linspace(20, 40, len(bdays))),
            "T10Y2Y": make("yield_curve_10y2y", bdays, np.linspace(0.5, -0.2, len(bdays))),
            "BAMLH0A0HYM2": make("high_yield_spread", bdays, np.linspace(3, 8, len(bdays))),
            "DFF": make("fed_funds_rate", alldays, np.linspace(4.0, 2.0, len(alldays))),
            "SOFR": sofr,
        }

    def _merged(self, tmp_path: Path, start: str, end: str, sofr_empty: bool) -> pd.DataFrame:
        from omni_mercury_engine.loaders.financial_loader import FinancialLoader

        frames = self._frames(start, end, sofr_empty)

        def fake_series(series_id: str, s: str, e: str) -> pd.DataFrame:
            return frames[series_id].copy()

        loader = FinancialLoader(cache_dir=tmp_path / "fin", api_key="x", max_retries=0)
        with patch.object(loader, "_fetch_fred_series", side_effect=fake_series):
            return loader._fetch_merged_series(start, end)

    def test_pre2018_funding_rate_from_dff(self, tmp_path: Path) -> None:
        """Pre-2018 (SOFR empty) funding_rate is a real DFF series, not constant 0."""
        from omni_mercury_engine.loaders.financial_loader import FinancialLoader

        merged = self._merged(tmp_path, "2008-01-01", "2008-03-31", sofr_empty=True)
        assert merged["funding_rate"].notna().all()
        assert merged["funding_rate"].nunique() > 1

        loader = FinancialLoader(cache_dir=tmp_path / "fin2", api_key="x", max_retries=0)
        features = loader.engineer_features(merged)
        funding_feat = features[:, 6]  # funding_rate column
        assert len(np.unique(funding_feat)) > 1, "funding_rate feature is constant"
        assert not np.allclose(funding_feat, 0.0)

    def test_no_weekend_rows_after_merge(self, tmp_path: Path) -> None:
        """DFF's 7-day publication must not add weekend rows to the merge."""
        merged = self._merged(tmp_path, "2008-01-01", "2008-03-31", sofr_empty=True)
        weekend = merged[merged["date"].dt.day_of_week >= 5]
        assert len(weekend) == 0
        assert len(merged) == len(pd.bdate_range("2008-01-01", "2008-03-31"))

    def test_no_future_value_backfilled(self, tmp_path: Path) -> None:
        """A leading gap is left NaN, never filled from a later observation."""
        from omni_mercury_engine.loaders.financial_loader import FinancialLoader

        start, end = "2008-01-01", "2008-03-31"
        frames = self._frames(start, end, sofr_empty=True)
        bdays = pd.bdate_range(start, end)
        # VIX missing for its first two trading days.
        vals = [np.nan, np.nan] + list(np.linspace(20, 40, len(bdays) - 2))
        frames["VIXCLS"] = pd.DataFrame({"date": pd.to_datetime(bdays), "VIX": vals})

        def fake_series(series_id: str, s: str, e: str) -> pd.DataFrame:
            return frames[series_id].copy()

        loader = FinancialLoader(cache_dir=tmp_path / "fin3", api_key="x", max_retries=0)
        with patch.object(loader, "_fetch_fred_series", side_effect=fake_series):
            merged = loader._fetch_merged_series(start, end)
        # Leading rows stay NaN (forward-fill only); the old bfill copied the
        # first real value (20.0) backward into them.
        assert pd.isna(merged["vix"].iloc[0])
        assert pd.isna(merged["vix"].iloc[1])


# ======================================================================
# Follow-up item 1 -- tsunami: no future leakage in fill / rolling, and
# OfflineModeError propagation + realtime caching
# ======================================================================
class TestTsunamiLeakage:
    """Leading gaps fill from the first (not last) value; windows are trailing."""

    def test_fill_non_finite_leading_uses_first_value(self) -> None:
        """Leading NaNs take the FIRST valid value, not the last (future)."""
        from omni_mercury_engine.loaders.tsunami_loader import _fill_non_finite

        arr = np.array([np.nan, np.nan, 10.0, 11.0, 12.0, 99.0])
        out = _fill_non_finite(arr.copy())
        # First-valid is 10.0; the old end-scan filled these with 99.0.
        assert out[0] == 10.0
        assert out[1] == 10.0

    def test_rolling_mean_is_trailing(self) -> None:
        """A rolling mean must not see a future impulse."""
        from omni_mercury_engine.loaders.tsunami_loader import _rolling_mean

        x = np.zeros(11)
        x[5] = 10.0
        rm = _rolling_mean(x.copy(), 5)
        # Index 3 is before the impulse -> trailing window sees only zeros.
        assert rm[3] == 0.0
        # Index 5 includes the impulse (5 trailing samples -> 10/5).
        assert rm[5] == pytest.approx(2.0)

    def test_rolling_std_is_trailing(self) -> None:
        """A rolling std must not see a future impulse."""
        from omni_mercury_engine.loaders.tsunami_loader import _rolling_std

        x = np.zeros(11)
        x[5] = 10.0
        rs = _rolling_std(x.copy(), 5)
        assert rs[3] == 0.0
        assert rs[5] > 0.0

    def test_fetch_realtime_propagates_offline_and_caches(self, tmp_path: Path) -> None:
        """OfflineModeError propagates unchanged; a successful fetch is cached."""
        from omni_mercury_engine.datasets.exceptions import OfflineModeError
        from omni_mercury_engine.loaders.tsunami_loader import TsunamiLoader

        loader = TsunamiLoader(cache_dir=tmp_path / "tsu", max_retries=0)
        with (
            patch.object(loader, "_fetch_station", side_effect=OfflineModeError("offline")),
            pytest.raises(OfflineModeError),
        ):
            loader.fetch_realtime()

        # A successful fetch should be cached (second call does not re-fetch).
        good = pd.DataFrame(
            {
                "station_id": ["21418"] * 3,
                "timestamp": pd.to_datetime(
                    ["2024-01-01T00:00:00Z", "2024-01-01T00:15:00Z", "2024-01-01T00:30:00Z"]
                ),
                "bpr": [1000.0, 1000.1, 1000.2],
            }
        )
        loader2 = TsunamiLoader(cache_dir=tmp_path / "tsu2", max_retries=0)
        calls = {"n": 0}

        def one_station(station_id: str) -> pd.DataFrame:
            calls["n"] += 1
            return good if station_id == "21418" else good.iloc[0:0]

        with patch.object(loader2, "_fetch_station", side_effect=one_station):
            first = loader2.fetch_realtime()
            after_first = calls["n"]
            second = loader2.fetch_realtime()  # should hit cache, no new fetches
        assert not first.empty
        assert len(second) == len(first)
        assert calls["n"] == after_first, "realtime was not cached (re-fetched)"


# ======================================================================
# Follow-up item 2 -- landslide: no worldwide stand-in, paginate, no empty cache
# ======================================================================
class TestLandslideNoStandin:
    """An empty area result raises; exceededTransferLimit is paged; empty is
    never cached."""

    def test_empty_area_raises_no_worldwide_fallback(self, tmp_path: Path) -> None:
        """An empty area query raises instead of substituting worldwide rows."""
        from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
        from omni_mercury_engine.loaders.landslide_loader import _EVENT_CATALOG, LandslideLoader

        evt = next(iter(_EVENT_CATALOG))
        loader = LandslideLoader(cache_dir=tmp_path / "ls", max_retries=0)
        with (
            patch.object(loader, "_fetch_json", return_value={"features": []}) as m,
            pytest.raises(DataSourceUnavailableError),
        ):
            loader.fetch_historical(evt)
        # Exactly one query: no second worldwide date-only fallback request.
        assert m.call_count == 1

    def test_empty_result_not_cached(self, tmp_path: Path) -> None:
        """An empty area result must not be frozen into the cache."""
        from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
        from omni_mercury_engine.loaders.landslide_loader import _EVENT_CATALOG, LandslideLoader

        evt = next(iter(_EVENT_CATALOG))
        loader = LandslideLoader(cache_dir=tmp_path / "ls2", max_retries=0)
        with (
            patch.object(loader, "_fetch_json", return_value={"features": []}),
            pytest.raises(DataSourceUnavailableError),
        ):
            loader.fetch_historical(evt)
        # A second attempt must query again (nothing cached), not serve empty.
        with (
            patch.object(loader, "_fetch_json", return_value={"features": []}) as m2,
            pytest.raises(DataSourceUnavailableError),
        ):
            loader.fetch_historical(evt)
        assert m2.call_count == 1

    def test_pagination_follows_transfer_limit(self, tmp_path: Path) -> None:
        """exceededTransferLimit pages are followed and concatenated."""
        from omni_mercury_engine.loaders.landslide_loader import _EVENT_CATALOG, LandslideLoader

        evt = next(iter(_EVENT_CATALOG))

        def feat(i: int) -> dict[str, Any]:
            return {
                "attributes": {
                    "event_id": i,
                    "event_date": 1500000000000 + i,
                    "fatality_count": 0,
                    "landslide_size": "small",
                    "latitude": 1.0,
                    "longitude": 2.0,
                }
            }

        pages: list[dict[str, Any]] = [
            {"features": [feat(i) for i in range(2000)], "exceededTransferLimit": True},
            {"features": [feat(i) for i in range(2000, 2500)]},
        ]
        seq = {"n": 0}

        def fake(url: str, params: dict[str, str] | None = None) -> dict[str, Any]:
            r = pages[seq["n"]]
            seq["n"] += 1
            return r

        loader = LandslideLoader(cache_dir=tmp_path / "ls3", max_retries=0)
        with patch.object(loader, "_fetch_json", side_effect=fake):
            df = loader.fetch_historical(evt)
        assert len(df) == 2500

    def test_a_service_that_ignores_the_offset_does_not_duplicate_rows(
        self, tmp_path: Path
    ) -> None:
        """Every page answered with the same rows yields each row exactly once.

        The recorded mirror fixture carries ``exceededTransferLimit: true``, and
        a replayed response answers the offset request with the same page; the
        first paging implementation appended it before its stop condition ran
        and returned every row twice.
        """
        from omni_mercury_engine.loaders.landslide_loader import _EVENT_CATALOG, LandslideLoader

        evt = next(iter(_EVENT_CATALOG))
        page = {
            "features": [
                {
                    "attributes": {
                        "FID": i,
                        "event_id": 100 + i,
                        "event_date": 1500000000000 + i,
                        "fatality_count": 0,
                        "landslide_size": "small",
                        "latitude": 1.0,
                        "longitude": 2.0,
                    }
                }
                for i in range(9)
            ],
            "exceededTransferLimit": True,
        }
        calls = {"n": 0}

        def same_page(url: str, params: dict[str, str] | None = None) -> dict[str, Any]:
            calls["n"] += 1
            return page

        loader = LandslideLoader(cache_dir=tmp_path / "ls4", max_retries=0)
        with patch.object(loader, "_fetch_json", side_effect=same_page):
            df = loader.fetch_historical(evt)
        assert len(df) == 9
        assert calls["n"] == 2, "one repeat is enough to see the service is not paging"


# ======================================================================
# Follow-up item 3 -- pandemic: outage distinguishable from no-data
# ======================================================================
class TestPandemicOutageDistinguishable:
    """fetch_historical raises on an outage but returns empty on true no-data."""

    def test_outage_raises(self, tmp_path: Path) -> None:
        """A WHO fetch failure surfaces as data-source-unavailable, not empty."""
        from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
        from omni_mercury_engine.loaders.pandemic_loader import PandemicLoader

        loader = PandemicLoader(cache_dir=tmp_path / "pm", max_retries=0)
        with (
            patch.object(loader, "_fetch_json", side_effect=ConnectionError("500")),
            pytest.raises(DataSourceUnavailableError),
        ):
            loader.fetch_historical("cholera_yemen_2017")

    def test_genuine_no_records_returns_empty(self, tmp_path: Path) -> None:
        """A genuine empty API response is still returned as an empty frame."""
        from omni_mercury_engine.loaders.pandemic_loader import PandemicLoader

        loader = PandemicLoader(cache_dir=tmp_path / "pm2", max_retries=0)
        with patch.object(loader, "_fetch_json", return_value={"value": []}):
            df = loader.fetch_historical("cholera_yemen_2017")
        assert df.empty

    def test_helper_keeps_graceful_empty_with_marker(self, tmp_path: Path) -> None:
        """The low-level helper still returns empty, now flagged as a failure."""
        from omni_mercury_engine.loaders.pandemic_loader import _FETCH_FAILED_ATTR, PandemicLoader

        loader = PandemicLoader(cache_dir=tmp_path / "pm3", max_retries=0)
        with patch.object(loader, "_fetch_json", side_effect=ConnectionError("500")):
            df = loader._fetch_who_gho("WHS3_41", "YEM", "2017-01-01", "2018-12-31")
        assert df.empty
        assert df.attrs.get(_FETCH_FAILED_ATTR) is True


# ======================================================================
# Follow-up item 4 -- stale cache: versioned keys retire pre-fix payloads
# ======================================================================
class TestCacheSchemaBump:
    """A frame cached under the pre-fix key is never read back after the fix."""

    def test_earthquake_old_cache_not_served(self, tmp_path: Path) -> None:
        """A stale worldwide-quake frame under the old key is not returned."""
        from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader

        loader = EarthquakeLoader(cache_dir=tmp_path / "eq", max_retries=0)
        # Plant a stale entry under the PRE-FIX key (no spatial filter era).
        stale = pd.DataFrame(
            {
                "time": [1_000_000],
                "latitude": [0.0],
                "longitude": [0.0],
                "depth": [10.0],
                "magnitude": [2.5],
                "place": ["STALE_WORLDWIDE"],
                "event_id": ["stale"],
            }
        )
        loader._write_cache("earthquake_historical_turkey_syria_2023", stale.to_dict(orient="list"))

        fresh = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"time": 1_700_000_000_000, "mag": 7.8, "place": "FRESH"},
                    "geometry": {"type": "Point", "coordinates": [37.0, 37.1, 10.0]},
                    "id": "fresh",
                }
            ],
        }
        with patch.object(loader, "_fetch_json", return_value=fresh) as m:
            df = loader.fetch_historical("turkey_syria_2023")
        assert m.called, "stale cache was served under the old key"
        assert "STALE_WORLDWIDE" not in df.get("place", pd.Series([], dtype=str)).tolist()

    def test_volcanic_old_cache_not_served(self, tmp_path: Path) -> None:
        """A stale current-alert frame cached as the 2018 event is not returned."""
        from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
        from omni_mercury_engine.loaders.volcanic_loader import VolcanicLoader

        loader = VolcanicLoader(cache_dir=tmp_path / "volc", max_retries=0)
        # Plant a stale entry under the PRE-FIX key (today's alert mislabeled).
        stale = pd.DataFrame(
            {
                "volcano_name": ["Kilauea"],
                "alert_level": ["WATCH"],
                "color_code": ["ORANGE"],
                "alert_level_numeric": [2],
                "color_code_numeric": [2],
                "alert_date": ["2099-01-01T00:00:00Z"],
                "volcano_number": ["332010"],
                "observatory": [""],
                "notice_id": [""],
            }
        )
        loader._write_cache("volcanic_historical_kilauea_2018", stale.to_dict(orient="list"))

        # With the old cache ignored, the current-alert guard now controls:
        # a present-day alert must raise rather than be served as 2018.
        today = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        alerts = {"alerts": [{"volcanoName": "Kilauea", "alertLevel": "WATCH", "alertDate": today}]}
        with (
            patch.object(loader, "_fetch_json", return_value=alerts),
            pytest.raises(DataSourceUnavailableError),
        ):
            loader.fetch_historical("kilauea_2018")


class TestFinancialGapImputation:
    """Gaps are filled from the past; only a leading gap takes the nearest later value."""

    def test_a_gap_takes_the_last_value_before_it_not_a_later_statistic(self) -> None:
        from omni_mercury_engine.loaders.financial_loader import FinancialLoader

        column = np.array([np.nan, np.nan, 1.0, np.nan, 100.0, np.inf, 100.0, 100.0])
        filled = FinancialLoader._impute_gaps(column.reshape(-1, 1))[:, 0]
        # The whole-window median (100.0) is what the leading rows used to get.
        np.testing.assert_array_equal(filled, [1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 100.0])

    def test_complete_and_empty_columns(self) -> None:
        from omni_mercury_engine.loaders.financial_loader import FinancialLoader

        features = np.column_stack([np.arange(4.0), np.full(4, np.nan)])
        filled = FinancialLoader._impute_gaps(features)
        np.testing.assert_array_equal(filled[:, 0], np.arange(4.0))
        np.testing.assert_array_equal(filled[:, 1], np.zeros(4))
        assert np.isnan(features[:, 1]).all(), "the caller's array must not be modified"


class TestRealtimeReadIsNotCachedLikeHistory:
    def test_tsunami_realtime_entry_expires_on_the_realtime_ttl(self, tmp_path: Path) -> None:
        """A live DART read older than a few minutes is fetched again.

        It was cached under the 24-hour historical TTL, so a buoy entering
        event mode after the first call of the day stayed invisible.
        """
        import json
        import time as _time

        from omni_mercury_engine.loaders.tsunami_loader import TsunamiLoader

        loader = TsunamiLoader(cache_dir=tmp_path / "tsu", max_retries=0)
        assert loader.REALTIME_CACHE_TTL < 3600 < loader.CACHE_TTL
        frame = pd.DataFrame({"station_id": ["21413"], "water_height": [5000.0]})
        loader._write_frame_cache("tsunami_realtime", frame)
        assert (
            loader._read_frame_cache("tsunami_realtime", max_age=loader.REALTIME_CACHE_TTL)
            is not None
        )

        path = loader._get_cache_path("tsunami_realtime")
        payload = json.loads(path.read_text())
        payload["timestamp"] = _time.time() - (loader.REALTIME_CACHE_TTL + 60)
        path.write_text(json.dumps(payload))

        # Stale for a live read, though still inside the historical TTL.
        assert (
            loader._read_frame_cache("tsunami_realtime", max_age=loader.REALTIME_CACHE_TTL) is None
        )
        assert loader._read_frame_cache("tsunami_realtime") is not None


# ======================================================================
# Catalogue defects: events that could not be read, and why
# ======================================================================
def _ibtracs_rows() -> list[dict[str, str]]:
    """IBTrACS's name, season and identifier for each catalogue storm (committed extract)."""
    import csv
    from pathlib import Path as _Path

    recorded = _Path(__file__).parent / "fixtures" / "ibtracs" / "catalogue_storm_ids.csv"
    with recorded.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _storm_name(meta: dict[str, Any]) -> str:
    """``"KATRINA"`` for the catalogue's ``"Hurricane Katrina (2005)"``."""
    return str(meta["name"]).split("(")[0].replace("Hurricane", "").strip().upper()


def _basin_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    """A basin frame in the loader's schema: five 6-hourly fixes per storm."""
    fixes: list[dict[str, Any]] = []
    for row in rows:
        first = pd.Timestamp(row["first_fix_utc"]).ceil("6h")
        for step in range(5):
            fixes.append(
                {
                    "sid": row["sid"],
                    "season": int(row["season"]),
                    "name": row["name"],
                    "iso_time": str(first + pd.Timedelta(hours=6 * step)),
                    "lat": 20.0,
                    "lon": -70.0,
                    "wind_kt": 40 + 10 * step,
                    "pressure_mb": 1000 - 5 * step,
                    "nature": "TS",
                    "track_type": "main",
                }
            )
    return pd.DataFrame(fixes)


class TestHurricaneCatalogueIdentifiers:
    """Every catalogue event names a storm IBTrACS holds, by its own identifier.

    Seven events (Harvey, Maria, Ian, Michael, Dorian, Ida, Matthew) carried
    identifiers that are not in IBTrACS, so they returned no track and were
    never measured. The fixture is the archive's own name, season and
    identifier for each catalogue storm (``tests/fixtures/ibtracs``).
    """

    def test_the_fixture_holds_one_identifier_per_catalogue_storm(self) -> None:
        from omni_mercury_engine.loaders.hurricane_loader import _EVENT_CATALOG

        recorded = {(row["name"], int(row["season"])) for row in _ibtracs_rows()}
        catalogued = {
            (_storm_name(meta), int(meta["date"][:4])) for meta in _EVENT_CATALOG.values()
        }
        assert recorded == catalogued
        assert len(recorded) == len(_EVENT_CATALOG) == 12

    def test_each_identifier_is_the_archives(self) -> None:
        from omni_mercury_engine.loaders.hurricane_loader import _EVENT_CATALOG

        archive = {(row["name"], int(row["season"])): row for row in _ibtracs_rows()}
        wrong = {}
        for event_id, meta in _EVENT_CATALOG.items():
            row = archive[(_storm_name(meta), int(meta["date"][:4]))]
            if meta["sid"] != row["sid"] or meta["basin"] != row["basin"]:
                wrong[event_id] = (meta["sid"], row["sid"])
        assert not wrong, f"catalogue identifiers that are not the archive's: {wrong}"

    def test_the_landfall_date_falls_inside_the_storms_track(self) -> None:
        from omni_mercury_engine.loaders.hurricane_loader import _EVENT_CATALOG

        archive = {(row["name"], int(row["season"])): row for row in _ibtracs_rows()}
        for event_id, meta in _EVENT_CATALOG.items():
            row = archive[(_storm_name(meta), int(meta["date"][:4]))]
            first = pd.Timestamp(row["first_fix_utc"]).normalize()
            last = pd.Timestamp(row["last_fix_utc"])
            assert first <= pd.Timestamp(meta["date"]) <= last, event_id

    @pytest.mark.parametrize(
        "event_id",
        [
            "katrina_2005",
            "harvey_2017",
            "maria_2017",
            "ian_2022",
            "helene_2024",
            "milton_2024",
            "irma_2017",
            "michael_2018",
            "dorian_2019",
            "ida_2021",
            "sandy_2012",
            "matthew_2016",
        ],
    )
    def test_every_event_reads_its_own_storm(self, event_id: str, tmp_path: Path) -> None:
        """From a basin file holding the archive's identifiers, no event is empty."""
        from omni_mercury_engine.loaders.hurricane_loader import _EVENT_CATALOG, HurricaneLoader

        loader = HurricaneLoader(cache_dir=tmp_path / "hur", max_retries=0)
        basin = _basin_frame(_ibtracs_rows())
        with patch.object(loader, "_download_ibtracs_csv", return_value=basin) as download:
            track = loader.fetch_historical(event_id)
        assert download.call_args.args[0].endswith("ibtracs.NA.list.v04r01.csv")
        assert len(track) == 5
        assert set(track["name"]) == {_storm_name(_EVENT_CATALOG[event_id])}

    def test_an_identifier_that_is_another_storm_is_refused(self, tmp_path: Path) -> None:
        """A catalogue identifier resolving to a different name is a defect, not data."""
        from omni_mercury_engine.loaders.hurricane_loader import HurricaneLoader

        loader = HurricaneLoader(cache_dir=tmp_path / "hur", max_retries=0)
        basin = _basin_frame(_ibtracs_rows())
        basin.loc[basin["name"] == "KATRINA", "name"] = "OPHELIA"
        with (
            patch.object(loader, "_download_ibtracs_csv", return_value=basin),
            pytest.raises(ValueError, match="OPHELIA"),
        ):
            loader.fetch_historical("katrina_2005")


class TestHurricaneCacheKeysRetireMainOnlyFrames:
    """A frame cached while only ``main`` rows were kept is not read back.

    The filter that kept only ``main`` dropped every provisional storm; when it
    was corrected the basin key was versioned but the event key and the
    realtime key were not, so such a cache went on serving the main-only frame.
    """

    @staticmethod
    def _stale() -> dict[str, list[Any]]:
        return {
            "sid": ["STALE"],
            "season": [2005],
            "name": ["STALE_MAIN_ONLY"],
            "iso_time": ["2005-08-25 00:00:00"],
            "lat": [0.0],
            "lon": [0.0],
            "wind_kt": [10],
            "pressure_mb": [1010],
            "nature": ["TS"],
            "track_type": ["main"],
        }

    def test_event_frame_under_the_v2_key_is_not_served(self, tmp_path: Path) -> None:
        from omni_mercury_engine.loaders.hurricane_loader import HurricaneLoader

        loader = HurricaneLoader(cache_dir=tmp_path / "hur", max_retries=0)
        loader._write_cache("hurricane_historical_v2_katrina_2005", self._stale())
        basin = _basin_frame(_ibtracs_rows())
        with patch.object(loader, "_download_ibtracs_csv", return_value=basin) as download:
            track = loader.fetch_historical("katrina_2005")
        assert download.called, "the main-only frame was served from the v2 key"
        assert set(track["name"]) == {"KATRINA"}

    def test_realtime_frame_under_the_v2_key_is_not_served(self, tmp_path: Path) -> None:
        from omni_mercury_engine.loaders.hurricane_loader import HurricaneLoader

        loader = HurricaneLoader(cache_dir=tmp_path / "hur", max_retries=0)
        loader._write_cache("hurricane_realtime_last3years_v2", self._stale())
        fresh = _basin_frame(_ibtracs_rows())
        fresh.loc[fresh["name"] == "MILTON", "track_type"] = "PROVISIONAL"
        with patch.object(loader, "_download_ibtracs_csv", return_value=fresh) as download:
            recent = loader.fetch_realtime()
        assert download.called, "the main-only frame was served from the v2 key"
        assert "STALE_MAIN_ONLY" not in set(recent["name"])
        assert "PROVISIONAL" in set(recent["track_type"])


class TestEbolaIsNotBenchmarkable:
    """WHO GHO serves no Ebola series, so the event has no data to score.

    The loader asked for the indicator ``EBOLA_CASESDEATHS``, which the API
    answers with HTTP 404; what the event served was a series the loader built.
    """

    def test_it_is_marked_with_its_reason(self) -> None:
        from omni_mercury_engine.loaders.pandemic_loader import _EVENT_CATALOG, PandemicLoader

        entry = _EVENT_CATALOG["ebola_2014"]
        assert entry["not_benchmarkable"] is True
        assert "404" in entry["_not_benchmarkable_reason"]
        listed = {event["event_id"]: event for event in PandemicLoader.list_events(MagicMock())}
        assert listed["ebola_2014"]["not_benchmarkable"] is True
        assert listed["ebola_2014"]["reason"] == entry["_not_benchmarkable_reason"]

    def test_no_request_is_made_and_nothing_is_returned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.loaders.pandemic_loader import PandemicLoader

        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
        loader = PandemicLoader(cache_dir=tmp_path / "pand", max_retries=0)
        with patch.object(loader, "_fetch_json", side_effect=AssertionError("no request")):
            frame = loader.fetch_historical("ebola_2014")
            labels = loader.get_ground_truth("ebola_2014")
        assert frame.empty
        assert labels.size == 0

    def test_a_frame_cached_while_it_was_wired_is_not_served(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.loaders.pandemic_loader import PandemicLoader

        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
        loader = PandemicLoader(cache_dir=tmp_path / "pand", max_retries=0)
        constructed = PandemicLoader._synthetic_ebola_2014()
        loader._write_frame_cache("pandemic_historical_v2_ebola_2014", constructed)
        assert loader.fetch_historical("ebola_2014").empty

    def test_the_constructed_series_needs_the_opt_in_and_carries_the_mark(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.loaders.base import SyntheticDataNotDeleakable
        from omni_mercury_engine.loaders.pandemic_loader import PandemicLoader

        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
        loader = PandemicLoader(cache_dir=tmp_path / "pand", max_retries=0)
        with patch.object(loader, "_fetch_json", side_effect=AssertionError("no request")):
            frame = loader.fetch_historical("ebola_2014")
        assert len(frame) == 306
        assert loader.is_synthetic(frame)
        with pytest.raises(SyntheticDataNotDeleakable):
            loader.engineer_scored_features(frame)


class TestPandemicStubsCannotBeTakenForData:
    """An event with no source says so when listed, and returns nothing."""

    STUBS = (
        "mpox_2022",
        "candida_auris_us_tracking",
        "malaria_subsaharan_2019_2022",
        "cjd_us_surveillance",
    )

    def test_each_is_listed_as_not_benchmarkable_with_a_reason(self) -> None:
        from omni_mercury_engine.loaders.pandemic_loader import _EVENT_CATALOG, PandemicLoader

        listed = {event["event_id"]: event for event in PandemicLoader.list_events(MagicMock())}
        for event_id in self.STUBS:
            assert _EVENT_CATALOG[event_id]["source"] == "stub"
            assert listed[event_id]["not_benchmarkable"] is True
            assert listed[event_id]["reason"] == _EVENT_CATALOG[event_id]["_stub_reason"]
        measurable = set(listed) - set(self.STUBS) - {"ebola_2014"}
        assert all("not_benchmarkable" not in listed[event_id] for event_id in measurable)

    @pytest.mark.parametrize("event_id", STUBS)
    def test_each_returns_nothing_and_asks_no_source(self, event_id: str, tmp_path: Path) -> None:
        from omni_mercury_engine.loaders.pandemic_loader import PandemicLoader

        loader = PandemicLoader(cache_dir=tmp_path / "pand", max_retries=0)
        # A frame left in the cache from before the event was a stub is not data either.
        loader._write_frame_cache(
            f"pandemic_historical_v2_{event_id}",
            pd.DataFrame({"date": ["2022-05-01"], "new_cases": [1.0]}),
        )
        with (
            patch.object(loader, "_fetch_json", side_effect=AssertionError("no request")),
            patch.object(loader, "_load_owid_data", side_effect=AssertionError("no request")),
        ):
            assert loader.fetch_historical(event_id).empty
            assert loader.get_ground_truth(event_id).size == 0


class TestTornadoArchiveCoversItsCatalogue:
    """The archive file the loader names holds every catalogue event's year.

    SPC names the file by its last year. The loader named
    ``1950-2023_actual_tornadoes.csv``, which has no row for ``midwest_2024``.
    """

    COLUMNS = ("om", "yr", "mo", "dy", "date", "time", "tz", "st", "mag", "inj", "fat")

    def _archive(self, last_year: int) -> pd.DataFrame:
        rows = [
            (1, 2011, 4, 27, "2011-04-27", "15:00:00", 3, "AL", 4, 10, 2),
            (2, 2013, 5, 20, "2013-05-20", "14:56:00", 3, "OK", 5, 200, 24),
            (3, 2024, 4, 26, "2024-04-26", "15:30:00", 3, "NE", 3, 4, 0),
            (4, 2024, 4, 26, "2024-04-26", "16:10:00", 3, "IA", 1, 0, 0),
            (5, 2024, 4, 27, "2024-04-27", "17:45:00", 3, "OK", -9, 0, 0),
        ]
        frame = pd.DataFrame(rows, columns=list(self.COLUMNS))
        for column, value in (("len", 1.0), ("wid", 50), ("slat", 36.0), ("slon", -97.0)):
            frame[column] = value
        frame["sg"] = 1
        return frame[frame["yr"] <= last_year].reset_index(drop=True)

    def _served(self, url: str, **_: Any) -> pd.DataFrame:
        """What SPC serves at ``url``: the archive up to the year in its name."""
        import re

        found = re.search(r"/1950-(\d{4})_actual_tornadoes\.csv$", url)
        assert found is not None, url
        return self._archive(int(found.group(1)))

    def test_the_2024_event_has_rows(self, tmp_path: Path) -> None:
        from omni_mercury_engine.loaders.tornado_loader import TornadoLoader

        loader = TornadoLoader(cache_dir=tmp_path / "tor", max_retries=0)
        with patch.object(loader, "_fetch_csv", side_effect=self._served):
            frame = loader.fetch_historical("midwest_2024")
            labels = loader.get_ground_truth("midwest_2024")
        assert len(frame) == 3
        assert labels.tolist() == [1, 0, 0]

    def test_an_unassigned_rating_is_missing_not_minus_nine(self, tmp_path: Path) -> None:
        from omni_mercury_engine.loaders.tornado_loader import TornadoLoader

        loader = TornadoLoader(cache_dir=tmp_path / "tor", max_retries=0)
        with patch.object(loader, "_fetch_csv", side_effect=self._served):
            frame = loader.fetch_historical("midwest_2024")
        assert pd.isna(frame["mag"].iloc[2])
        features = loader.engineer_features(frame)
        ef_scale = features[:, loader.FEATURE_COLUMNS.index("ef_scale")]
        assert ef_scale.min() >= 0.0

    def test_an_archive_cached_from_the_older_file_is_not_read_back(self, tmp_path: Path) -> None:
        from omni_mercury_engine.loaders.tornado_loader import TornadoLoader

        loader = TornadoLoader(cache_dir=tmp_path / "tor", max_retries=0)
        loader._write_cache("tornado_archive_full", self._archive(2023).to_dict(orient="list"))
        with patch.object(loader, "_fetch_csv", side_effect=self._served) as fetch:
            frame = loader.fetch_historical("midwest_2024")
        assert fetch.called, "the 1950-2023 archive was served from the cache"
        assert len(frame) == 3


class TestCicidsFrameIsBounded:
    def test_the_dataset_loader_is_asked_for_a_bounded_sample(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful CICIDS load cannot write a multi-gigabyte JSON cache.

        The event frame is cached as JSON; the full dataset is ~2.8M x 79, and
        the loader asked for all of it.
        """
        from omni_mercury_engine.datasets import security
        from omni_mercury_engine.loaders import network_security_loader as module

        seen: dict[str, Any] = {}

        class _StubCicids:
            def __init__(self, config: Any) -> None:
                seen["max_samples"] = config.max_samples

            def load_data(self) -> tuple[np.ndarray, np.ndarray]:
                rng = np.random.default_rng(0)
                return rng.normal(size=(50, 4)).astype(np.float32), rng.integers(0, 2, size=50)

        monkeypatch.setattr(security, "CICIDSLoader", _StubCicids)
        loader = module.NetworkSecurityLoader(cache_dir=tmp_path / "cic", max_retries=0)
        frame = loader._load_cicids_dataframe()

        assert seen["max_samples"] == module._CICIDS_MAX_ROWS
        assert 0 < module._CICIDS_MAX_ROWS <= 250_000
        assert len(frame) == 50 and "label" in frame.columns
