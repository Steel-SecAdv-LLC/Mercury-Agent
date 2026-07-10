# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the earthquake_precursor catalog-based training pipeline.

The binding spec is ``docs/research/EARTHQUAKE_PRECURSOR_LITERATURE_REVIEW.md``:
probabilistic P(M>=5.0, 30 d) seismicity-rate forecasting from real USGS
catalog features -- never EM/Schumann inputs, never deterministic magnitude
or time-to-event prediction. Fixtures are real ComCat excerpts (see
``tests/fixtures/hazard_training/earthquake_precursor/PROVENANCE.json``);
no network access is required here.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.ml.hazard_training import earthquake_precursor as ep
from omni_mercury_engine.models.checkpoint_paths import shipped_checkpoint_path

FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent / "fixtures" / "hazard_training" / "earthquake_precursor"
)
FIXTURE_CSV = FIXTURE_DIR / "usgs_ca_2019_jun_jul_m25.csv"

# 2019 Ridgecrest mainshocks (in the fixture): M6.4 on July 4, M7.1 on July 6.
RIDGECREST_LAT, RIDGECREST_LON = 35.77, -117.599


def _day(y: int, m: int, d: int) -> float:
    """Days since the pipeline's 1980-01-01 UTC origin."""
    return float((_dt.datetime(y, m, d, tzinfo=_dt.UTC) - ep.DAY0).days)


@pytest.fixture(scope="module")
def catalog() -> ep.Catalog:
    """Real Ridgecrest-window catalog excerpt (June-July 2019)."""
    return ep.parse_catalog([FIXTURE_CSV])


@pytest.fixture(scope="module")
def index(catalog: ep.Catalog) -> ep.CatalogIndex:
    return ep.CatalogIndex(catalog)


class TestFeatureSpec:
    def test_layout_matches_fixed_architecture(self) -> None:
        from omni_mercury_engine.space.disaster_precursor_detector import (
            EarthquakePrecursorAnalyzer,
        )

        assert len(ep.EQ_FEATURE_NAMES) == ep.EQ_FEATURE_DIM == 128
        assert ep._N_INFORMATIVE == 36  # v2: 32 catalog dims + 4 stacked-RJ dims
        assert all(n.startswith("reserved_") for n in ep.EQ_FEATURE_NAMES[ep._N_INFORMATIVE :])
        assert ep.N_LAT * ep.N_LON == 440
        model = EarthquakePrecursorAnalyzer(input_dim=ep.EQ_FEATURE_DIM)
        mag, t, conf = model(torch.zeros(2, ep.EQ_FEATURE_DIM))
        assert conf.shape == (2, 1)

    def test_no_em_or_gap_features(self) -> None:
        """The review bans EM/Schumann inputs and 'overdue'/gap framings."""
        banned = ("schumann", "em_", "geomag", "ionospher", "overdue", "gap", "deficit")
        for name in ep.EQ_FEATURE_NAMES:
            assert not any(b in name.lower() for b in banned), name

    def test_split_is_the_documented_one(self) -> None:
        assert ep.SPLIT.train_years == tuple(range(1985, 2010))
        assert ep.SPLIT.val_years == tuple(range(2010, 2017))
        # Ridgecrest (2019) must land in the held-out test years.
        assert ep.SPLIT.test_years == tuple(range(2017, 2025))
        assert 2019 in ep.SPLIT.test_years

    def test_epochs_are_causal_and_in_span(self) -> None:
        days, years = ep.epoch_days_and_years()
        assert days[0] == ep.EPOCH_START_DAY
        assert np.all(np.diff(days) == ep.EPOCH_STRIDE_DAYS)
        # Every label window must fit inside the fetched catalog span.
        assert days[-1] + ep.LABEL_WINDOW_DAYS <= ep.CATALOG_END_DAY
        assert set(np.unique(years)) <= set(ep.SPLIT.all_years)


class TestNoLookahead:
    def test_post_epoch_events_never_change_features(self, catalog: ep.Catalog) -> None:
        """Property: features depend only on catalog data strictly before t."""
        ix, iy = ep.cell_of(np.array([RIDGECREST_LAT]), np.array([RIDGECREST_LON]))
        ix, iy = int(ix[0]), int(iy[0])  # type: ignore[assignment]
        t_epoch = _day(2019, 7, 1)  # before the July 4/6 mainshocks

        full = ep.CatalogIndex(catalog)
        pre_mask = catalog.t_days < t_epoch
        truncated = ep.CatalogIndex(
            ep.Catalog(
                t_days=catalog.t_days[pre_mask],
                lat=catalog.lat[pre_mask],
                lon=catalog.lon[pre_mask],
                depth=catalog.depth[pre_mask],
                mag=catalog.mag[pre_mask],
            )
        )
        for jx in range(ep.N_LAT):
            for jy in range(ep.N_LON):
                np.testing.assert_array_equal(
                    ep.build_feature_vector(full, jx, jy, t_epoch),
                    ep.build_feature_vector(truncated, jx, jy, t_epoch),
                )

    def test_feature_windows_exclude_the_epoch_itself(self, catalog: ep.Catalog) -> None:
        """An event exactly at t belongs to the label window, not the features."""
        idx = ep.CatalogIndex(catalog)
        # Use a real event time as the epoch: rates must not count it.
        m71_i = int(np.argmax(catalog.mag))
        t_event = float(catalog.t_days[m71_i])
        ix, iy = ep.cell_of(catalog.lat[m71_i : m71_i + 1], catalog.lon[m71_i : m71_i + 1])
        vec_at = ep.build_feature_vector(idx, int(ix[0]), int(iy[0]), t_event)
        vec_before = ep.build_feature_vector(idx, int(ix[0]), int(iy[0]), t_event - 1e-6)
        assert vec_at[12] == vec_before[12] != pytest.approx(7.1)  # max_mag_nbhd_w30


class TestLabelBuilder:
    def test_ridgecrest_july_2019_cell_is_positive(self, index: ep.CatalogIndex) -> None:
        """The pipeline epoch immediately before July 4 must label the cell 1."""
        days, years = ep.epoch_days_and_years()
        t_m64 = _day(2019, 7, 4)
        t_epoch = float(days[days < t_m64][-1])  # real pipeline cadence
        assert t_m64 - t_epoch <= ep.EPOCH_STRIDE_DAYS
        ix, iy = ep.cell_of(np.array([RIDGECREST_LAT]), np.array([RIDGECREST_LON]))
        assert index.label(int(ix[0]), int(iy[0]), t_epoch) == 1
        assert int(years[list(days).index(t_epoch)]) in ep.SPLIT.test_years

    def test_cell_without_m5_events_is_negative(
        self, catalog: ep.Catalog, index: ep.CatalogIndex
    ) -> None:
        ix_all, iy_all = ep.cell_of(catalog.lat, catalog.lon)
        m5 = catalog.mag >= ep.LABEL_MIN_MAG
        hot = {(int(a), int(b)) for a, b in zip(ix_all[m5], iy_all[m5], strict=True)}
        quiet = next(
            (jx, jy) for jx in range(ep.N_LAT) for jy in range(ep.N_LON) if (jx, jy) not in hot
        )
        days, _ = ep.epoch_days_and_years()
        t_epoch = float(days[days < _day(2019, 7, 4)][-1])
        assert index.label(quiet[0], quiet[1], t_epoch) == 0

    def test_window_is_open_at_t(self, index: ep.CatalogIndex, catalog: ep.Catalog) -> None:
        """Label window is (t, t+30]: nothing after the fixture end counts."""
        ix, iy = ep.cell_of(np.array([RIDGECREST_LAT]), np.array([RIDGECREST_LON]))
        t_end = float(catalog.t_days[-1])
        assert index.label(int(ix[0]), int(iy[0]), t_end) == 0


class TestReasenbergJonesBaseline:
    """The documented deterministic clustering baseline of the merit gate."""

    @staticmethod
    def _cell(times: list[float], mags: list[float]) -> ep._CellArrays:
        t = np.asarray(times, dtype=np.float64)
        m = np.asarray(mags, dtype=np.float64)
        z = np.zeros_like(t)
        return ep._CellArrays(t, m, z, z, z)

    def test_deterministic(self, index: ep.CatalogIndex) -> None:
        ix, iy = ep.cell_of(np.array([RIDGECREST_LAT]), np.array([RIDGECREST_LON]))
        cell = index.cell(int(ix[0]), int(iy[0]))
        t = _day(2019, 7, 10)
        p1 = ep.rj_probability(cell, t, mu_30d=0.002)
        p2 = ep.rj_probability(cell, t, mu_30d=0.002)
        assert p1 == p2
        assert 0.0 < p1 < 1.0

    def test_more_recent_large_event_raises_probability(self) -> None:
        t = 1000.0
        recent = ep.rj_probability(self._cell([t - 2.0], [6.0]), t, 0.002)
        older = ep.rj_probability(self._cell([t - 60.0], [6.0]), t, 0.002)
        assert recent > older

    def test_larger_magnitude_raises_probability(self) -> None:
        t = 1000.0
        m7 = ep.rj_probability(self._cell([t - 5.0], [7.0]), t, 0.002)
        m5 = ep.rj_probability(self._cell([t - 5.0], [5.0]), t, 0.002)
        assert m7 > m5

    def test_more_events_raise_probability(self) -> None:
        t = 1000.0
        two = ep.rj_probability(self._cell([t - 5.0, t - 3.0], [6.0, 6.0]), t, 0.002)
        one = ep.rj_probability(self._cell([t - 5.0], [6.0]), t, 0.002)
        assert two > one

    def test_future_events_are_excluded(self) -> None:
        t = 1000.0
        with_future = ep.rj_probability(self._cell([t + 1.0], [7.9]), t, 0.002)
        assert with_future == pytest.approx(ep.poisson_probability(0.002))

    def test_ridgecrest_aftermath_dwarfs_quiet_baseline(self, index: ep.CatalogIndex) -> None:
        ix, iy = ep.cell_of(np.array([RIDGECREST_LAT]), np.array([RIDGECREST_LON]))
        cell = index.cell(int(ix[0]), int(iy[0]))
        after = ep.rj_probability(cell, _day(2019, 7, 10), 0.002)
        before = ep.rj_probability(cell, _day(2019, 6, 20), 0.002)
        assert after > 10 * before
        assert after > 0.5  # days after an M7.1, RJ says M5+ is likely


class TestStackedRJFeatures:
    """v2 stacked block (dims 32-35): the RJ baseline's forecast as an input."""

    def test_rj_features_present_in_spec(self) -> None:
        assert ep.FEATURE_SPEC_VERSION == "seismicity-catalog-v2"
        for offset, name in enumerate(
            (
                "rj_prob_30d_cell",
                "rj_log_lambda_30d_cell",
                "rj_rate_30d_cell",
                "rj_mu_bg_causal_log",
            )
        ):
            index = ep.EQ_FEATURE_NAMES.index(name)
            assert index == 32 + offset
            assert index < ep._N_INFORMATIVE

    def test_rj_prob_feature_matches_causal_baseline_formula(self, index: ep.CatalogIndex) -> None:
        """Dim 32 is exactly 1 - exp(-(causal mu + in-cell RJ 30-d count))."""
        ix_arr, iy_arr = ep.cell_of(np.array([RIDGECREST_LAT]), np.array([RIDGECREST_LON]))
        ix, iy = int(ix_arr[0]), int(iy_arr[0])
        t = _day(2019, 7, 10)  # inside the Ridgecrest aftershock sequence
        vec = ep.build_feature_vector(index, ix, iy, t)
        cell = index.cell(ix, iy)
        hi = int(np.searchsorted(cell.t, t, side="left"))
        rj30 = ep.rj_triggered_count_30d(cell.t[:hi], cell.mag[:hi], t)
        mu = ep.causal_background_mu(index, ix, iy, t)
        assert vec[32] == pytest.approx(1.0 - math.exp(-(mu + rj30)), rel=1e-5)
        assert vec[33] == pytest.approx(math.log(mu + rj30), rel=1e-5)
        assert vec[34] == pytest.approx(math.log1p(rj30), rel=1e-5)
        assert vec[35] == pytest.approx(math.log(mu), rel=1e-5)
        assert vec[32] > 0.5  # days after an M7.1, the RJ baseline says M5+ is likely

    def test_rj_features_have_no_lookahead(self, catalog: ep.Catalog) -> None:
        """Dims 32-35 must not see the label window or anything after t."""
        ix_arr, iy_arr = ep.cell_of(np.array([RIDGECREST_LAT]), np.array([RIDGECREST_LON]))
        ix, iy = int(ix_arr[0]), int(iy_arr[0])
        t_epoch = _day(2019, 7, 1)  # before the July 4 M6.4 / July 6 M7.1
        full = ep.CatalogIndex(catalog)
        pre_mask = catalog.t_days < t_epoch
        truncated = ep.CatalogIndex(
            ep.Catalog(
                t_days=catalog.t_days[pre_mask],
                lat=catalog.lat[pre_mask],
                lon=catalog.lon[pre_mask],
                depth=catalog.depth[pre_mask],
                mag=catalog.mag[pre_mask],
            )
        )
        vec_full = ep.build_feature_vector(full, ix, iy, t_epoch)
        vec_trunc = ep.build_feature_vector(truncated, ix, iy, t_epoch)
        np.testing.assert_array_equal(vec_full[32:36], vec_trunc[32:36])
        assert ep.causal_background_mu(full, ix, iy, t_epoch) == ep.causal_background_mu(
            truncated, ix, iy, t_epoch
        )
        # ... and the block is live, not constant: after the mainshocks the
        # stacked baseline probability must move sharply upward.
        vec_after = ep.build_feature_vector(full, ix, iy, _day(2019, 7, 10))
        assert vec_after[32] > vec_full[32]
        assert vec_after[33] > vec_full[33]

    def test_causal_mu_counts_only_pre_epoch_m5(self, index: ep.CatalogIndex) -> None:
        """The trailing mu uses M>=5 events strictly before t (Laplace +0.5)."""
        ix_arr, iy_arr = ep.cell_of(np.array([RIDGECREST_LAT]), np.array([RIDGECREST_LON]))
        ix, iy = int(ix_arr[0]), int(iy_arr[0])
        t_before = _day(2019, 7, 1)
        mu_before = ep.causal_background_mu(index, ix, iy, t_before)
        # Fixture holds no pre-July-2019 M>=5 in the cell: bare prior only.
        assert mu_before == pytest.approx(0.5 / t_before * ep.LABEL_WINDOW_DAYS)
        m5 = index.cell_m5_times(ix, iy)
        assert m5.size > 0  # July 2019 delivers many
        t_after = float(m5[-1]) + 1.0
        n_seen = int(np.searchsorted(m5, t_after, side="left"))
        assert ep.causal_background_mu(index, ix, iy, t_after) == pytest.approx(
            (n_seen + 0.5) / t_after * ep.LABEL_WINDOW_DAYS
        )
        assert ep.causal_background_mu(index, ix, iy, t_after) > mu_before


class TestBValueEstimators:
    """Aki-Utsu + Shi-Bolt + b-positive, hand-checked on real fixture data."""

    @staticmethod
    def _window_mags(catalog: ep.Catalog, index: ep.CatalogIndex) -> np.ndarray:
        ix, iy = ep.cell_of(np.array([RIDGECREST_LAT]), np.array([RIDGECREST_LON]))
        nbhd = index.nbhd(int(ix[0]), int(iy[0]))
        t = float(catalog.t_days[-1]) + 1.0
        w = ep._window_slice(nbhd, t, 365.0)
        return w.mag

    def test_mc_maxc_matches_hand_computation(
        self, catalog: ep.Catalog, index: ep.CatalogIndex
    ) -> None:
        mags = self._window_mags(catalog, index)
        assert mags.size >= ep.BVAL_MIN_N
        counts = Counter(round(float(m) * 10) / 10 for m in mags)
        mode = min(sorted(counts), key=lambda v: (-counts[v], v))
        assert ep.mc_maxc(mags) == pytest.approx(mode + 0.2)

    def test_aki_utsu_matches_hand_computation(
        self, catalog: ep.Catalog, index: ep.CatalogIndex
    ) -> None:
        mags = self._window_mags(catalog, index)
        mc = ep.mc_maxc(mags)
        b, stderr, n = ep.aki_utsu_b(mags, mc)
        sel = [float(m) for m in mags if m >= mc - 1e-9]
        b_hand = math.log10(math.e) / (sum(sel) / len(sel) - (mc - 0.05))
        assert n == len(sel) >= ep.BVAL_MIN_N
        assert b == pytest.approx(b_hand, rel=1e-9)
        mean = sum(sel) / len(sel)
        var = sum((m - mean) ** 2 for m in sel) / (len(sel) * (len(sel) - 1))
        assert stderr == pytest.approx(2.3026 * b_hand**2 * math.sqrt(var), rel=1e-9)
        assert 0.5 < b < 2.0  # plausible for real California data

    def test_b_positive_plausible_on_real_sequence(
        self, catalog: ep.Catalog, index: ep.CatalogIndex
    ) -> None:
        mags = self._window_mags(catalog, index)
        bpos, n = ep.b_positive(mags)
        assert n >= ep.BVAL_MIN_N
        assert 0.5 < bpos < 2.5

    def test_undersampled_windows_emit_flags_not_values(self, index: ep.CatalogIndex) -> None:
        """< 50 events -> presence flags 0 and zeroed values, never fabricated."""
        ix, iy = ep.cell_of(np.array([RIDGECREST_LAT]), np.array([RIDGECREST_LON]))
        vec = ep.build_feature_vector(index, int(ix[0]), int(iy[0]), _day(2019, 6, 10))
        assert vec[18] == vec[19] == vec[20] == 0.0  # flags off (quiet pre-sequence)
        assert vec[14] == vec[15] == vec[16] == vec[17] == 0.0  # values zeroed

    def test_ridgecrest_window_flags_on(self, catalog: ep.Catalog, index: ep.CatalogIndex) -> None:
        ix, iy = ep.cell_of(np.array([RIDGECREST_LAT]), np.array([RIDGECREST_LON]))
        vec = ep.build_feature_vector(
            index, int(ix[0]), int(iy[0]), float(catalog.t_days[-1]) + 1.0
        )
        assert vec[18] == vec[19] == vec[20] == 1.0
        assert vec[14] > 0.0 and vec[16] > 0.0
        assert vec[30] == 1.0 and 0.0 <= vec[28] <= 1.0  # Zaliapin stats present


class TestSplitEnforcement:
    def test_temporal_split_rejects_interleaving(self) -> None:
        from omni_mercury_engine.ml.hazard_training.common import TemporalSplit

        with pytest.raises(ValueError, match="temporal split violated"):
            TemporalSplit(train_years=(2010,), val_years=(2009,), test_years=(2020,))

    def test_background_rate_uses_train_years_only(self, index: ep.CatalogIndex) -> None:
        """Fixture holds only 2019 (test-year) data -> mu must be the bare prior.

        If background_mu leaked val/test years, the Ridgecrest cell (dozens of
        M>=5 events in July 2019) would show an elevated rate here.
        """
        mu = ep.background_mu(index)
        prior_only = 0.5 / 9131.0 * ep.LABEL_WINDOW_DAYS
        assert np.allclose(mu, prior_only)


class TestDetectorContract:
    def test_untrained_detector_abstains_on_features_only_payload(self) -> None:
        from omni_mercury_engine.space.disaster_precursor_detector import (
            DisasterPrecursorDetector,
        )

        det = DisasterPrecursorDetector()
        result = det.detect_disaster_precursor(
            {"seismicity_features": np.zeros(ep.EQ_FEATURE_DIM, dtype=np.float32)}
        )
        assert result.precursor_detected is False
        assert result.confidence == 0.0
        assert result.estimated_magnitude is None

    def test_untrained_direct_call_still_raises(self) -> None:
        from omni_mercury_engine.space.disaster_precursor_detector import (
            DisasterPrecursorDetector,
        )

        det = DisasterPrecursorDetector()
        with pytest.raises(RuntimeError, match="untrained"):
            det._predict_earthquake(np.zeros(ep.EQ_FEATURE_DIM, dtype=np.float32))


@pytest.mark.skipif(
    not shipped_checkpoint_path("earthquake_precursor_ca").exists(),
    reason="no shipped earthquake_precursor_ca checkpoint (merit gate may have refused)",
)
class TestShippedCheckpointDifferential:
    """Physics-vs-shipped differential on real fixture cases."""

    def _detector(self):
        from omni_mercury_engine.space.disaster_precursor_detector import (
            DisasterPrecursorDetector,
        )

        det = DisasterPrecursorDetector(enable_tsunami=False, enable_geomagnetic=False)
        det.load_neural_weights()  # None -> shipped default
        return det

    def test_default_load_and_probability_semantics(self, index: ep.CatalogIndex) -> None:
        det = self._detector()
        assert det._neural_trained is True
        assert det._feature_spec == ep.FEATURE_SPEC_VERSION
        ix_arr, iy_arr = ep.cell_of(np.array([RIDGECREST_LAT]), np.array([RIDGECREST_LON]))
        ix, iy = int(ix_arr[0]), int(iy_arr[0])
        hot = ep.build_feature_vector(index, ix, iy, _day(2019, 7, 10))
        quiet = ep.build_feature_vector(index, ix, iy, _day(2019, 6, 20))
        p_hot = det.detect_disaster_precursor({"seismicity_features": hot}).confidence
        p_quiet = det.detect_disaster_precursor({"seismicity_features": quiet}).confidence
        assert 0.0 <= p_quiet <= 1.0 and 0.0 <= p_hot <= 1.0
        # Differential vs the physics baseline: both must rank the post-M7.1
        # aftermath far above the quiet pre-sequence case.
        mu = 0.5 / 9131.0 * ep.LABEL_WINDOW_DAYS
        cell = index.cell(ix, iy)
        rj_hot = ep.rj_probability(cell, _day(2019, 7, 10), mu)
        rj_quiet = ep.rj_probability(cell, _day(2019, 6, 20), mu)
        assert rj_hot > rj_quiet
        assert p_hot > p_quiet

    def test_trained_path_never_emits_magnitude_or_time(self, index: ep.CatalogIndex) -> None:
        """Review-prohibited claims: no magnitude/time even when trained."""
        det = self._detector()
        ix, iy = ep.cell_of(np.array([RIDGECREST_LAT]), np.array([RIDGECREST_LON]))
        hot = ep.build_feature_vector(index, int(ix[0]), int(iy[0]), _day(2019, 7, 10))
        result = det.detect_disaster_precursor({"seismicity_features": hot})
        assert result.estimated_magnitude is None
        assert result.time_to_event_hours is None
        forecast = det._predict_earthquake(hot)
        assert set(forecast) == {
            "event_probability",
            "confidence",
            "diagnostic_max_magnitude",
            "diagnostic_days_to_m4",
        }

    def test_legacy_em_features_alias_still_routes(self, index: ep.CatalogIndex) -> None:
        det = self._detector()
        ix, iy = ep.cell_of(np.array([RIDGECREST_LAT]), np.array([RIDGECREST_LON]))
        vec = ep.build_feature_vector(index, int(ix[0]), int(iy[0]), _day(2019, 7, 10))
        p_new = det.detect_disaster_precursor({"seismicity_features": vec}).confidence
        p_old = det.detect_disaster_precursor({"em_features": vec}).confidence
        assert p_new == pytest.approx(p_old)

    def test_provenance_documents_the_merit_gate(self) -> None:
        from omni_mercury_engine.models.checkpoint_paths import load_shipped_checkpoint

        payload, provenance = load_shipped_checkpoint("earthquake_precursor_ca")
        assert payload["feature_spec"] == ep.FEATURE_SPEC_VERSION
        assert payload["label"] == "P(M>=5.0, 30d)"
        assert provenance is not None
        evaluation = provenance["evaluation"]
        assert evaluation["learned_beats_physics"] is True
        assert evaluation["primary_metric"] == "log_loss"
        assert evaluation["learned"]["log_loss"] < evaluation["physics"]["log_loss"]
        extras = evaluation["extras"]
        assert "aftershock_dominance_fraction_test_positives" in extras
