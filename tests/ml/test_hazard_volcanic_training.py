# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the volcanic-eruption hazard training pipeline (offline).

All fixtures are REAL data excerpts committed under
``tests/fixtures/hazard_training/volcanic/`` (full provenance in the
``provenance.json`` sidecar there):

* ``av_ssba_bhz_2023-07-01.mseed`` -- 65 minutes of real AV.SSBA.BHZ
  miniSEED (EarthScope FDSN dataselect; Shishaldin volcano, ten days before
  the GVP-cataloged 2023-07-11 eruption onset).
* ``gvp_shishaldin_eruptions.csv`` -- the real Shishaldin rows of the
  Smithsonian GVP Holocene eruption catalog WFS CSV export.

No network access: pipeline stages that need live archives are exercised in
the training environment, not here. The differential learned-vs-physics test
runs only when the merit-gated ``volcanic_avo_seismic`` checkpoint has been
shipped (skipped otherwise). Scope reminder: that checkpoint is trained for
the named AVO volcano set only, never as a universal eruption forecaster.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.ml.hazard_training import volcanic_eruption as ve
from omni_mercury_engine.ml.hazard_training.common import PipelineContext
from omni_mercury_engine.models.checkpoint_paths import shipped_checkpoint_path

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "hazard_training" / "volcanic"
MSEED_FIXTURE = FIXTURE_DIR / "av_ssba_bhz_2023-07-01.mseed"
GVP_FIXTURE = FIXTURE_DIR / "gvp_shishaldin_eruptions.csv"
PROVENANCE = FIXTURE_DIR / "provenance.json"


@pytest.fixture(scope="session")
def provenance() -> dict[str, Any]:
    """Fixture provenance sidecar (station metadata incl. real sensitivity)."""
    return json.loads(PROVENANCE.read_text())


@pytest.fixture(scope="session")
def day_record(provenance: dict[str, Any]) -> dict[str, Any]:
    """Day record computed once from the real committed miniSEED excerpt."""
    pytest.importorskip("obspy")
    return ve.compute_day_record(
        MSEED_FIXTURE.read_bytes(),
        scale=float(provenance["mseed"]["scale_counts_per_m_s"]),
        expected_sr=float(provenance["mseed"]["sample_rate_hz"]),
    )


class TestFeatureSpec:
    def test_feature_names_cover_128_dims_uniquely(self) -> None:
        assert len(ve.FEATURE_NAMES) == ve.FEATURE_DIM == 128
        assert len(set(ve.FEATURE_NAMES)) == 128
        assert len(ve.HOURLY_FEATURE_NAMES) == ve.HOURLY_DIM == 32

    def test_volcano_onehot_within_budget_and_named(self) -> None:
        assert 6 <= len(ve.VOLCANOES) <= 8
        onehot = [n for n in ve.FEATURE_NAMES if n.startswith("volcano_onehot_")]
        assert len(onehot) == len(ve.VOLCANOES)
        # The scoped-honesty contract: every trained volcano is named.
        assert {v.name for v in ve.VOLCANOES} >= {"Shishaldin", "Pavlof", "Great Sitkin"}

    def test_split_is_strictly_ordered_by_year(self) -> None:
        assert max(ve.SPLIT.train_years) < min(ve.SPLIT.val_years)
        assert max(ve.SPLIT.val_years) < min(ve.SPLIT.test_years)
        years = np.array([2016, 2017, 2020])
        train, val, test = ve.SPLIT.masks(years)
        assert train.tolist() == [True, False, False]
        assert val.tolist() == [False, True, False]
        assert test.tolist() == [False, False, True]


class TestDayRecordFromRealMiniseed:
    """Feature builder on a committed real AV.SSBA.BHZ excerpt."""

    def test_rsam_and_bands_are_physical(self, day_record: dict[str, Any]) -> None:
        rsam = float(day_record["rsam_24h"])
        assert np.isfinite(rsam) and rsam > 0
        # Ground velocity at a quiet-ish broadband station: sane magnitude.
        assert 1e-9 < rsam < 1e-3
        bands = np.asarray(day_record["day_bands"], dtype=float)
        assert bands.shape == (4,)
        assert np.isfinite(bands).all() and (bands > 0).all()

    def test_presence_metadata_matches_65_minute_excerpt(self, day_record: dict[str, Any]) -> None:
        minute = np.asarray(day_record["minute_rsam"], dtype=float)
        assert minute.shape == (1440,)
        observed = int(np.isfinite(minute).sum())
        assert 60 <= observed <= 70  # the excerpt holds ~65 minutes
        assert day_record["data_fraction"] == pytest.approx(observed / 1440.0)
        assert float(day_record["sampling_rate"]) == 50.0

    def test_deterministic(self, provenance: dict[str, Any], day_record: dict[str, Any]) -> None:
        again = ve.compute_day_record(
            MSEED_FIXTURE.read_bytes(),
            scale=float(provenance["mseed"]["scale_counts_per_m_s"]),
            expected_sr=50.0,
        )
        np.testing.assert_array_equal(
            np.asarray(day_record["minute_rsam"]), np.asarray(again["minute_rsam"])
        )
        assert day_record["trigger_count"] == again["trigger_count"]

    def test_sample_rate_mismatch_fails_loud(self, provenance: dict[str, Any]) -> None:
        pytest.importorskip("obspy")
        with pytest.raises(RuntimeError, match="sample rate"):
            ve.compute_day_record(
                MSEED_FIXTURE.read_bytes(),
                scale=float(provenance["mseed"]["scale_counts_per_m_s"]),
                expected_sr=100.0,
            )

    def test_feature_vector_flags_missing_baseline_never_imputes(
        self, day_record: dict[str, Any]
    ) -> None:
        vec = ve.assemble_feature_vector(
            day_record, volcano_index=0, is_broadband=True, baseline=None, prev_rsam=None
        )
        assert vec.shape == (128,)
        names = list(ve.FEATURE_NAMES)
        assert vec[names.index("baseline_available_flag")] == 0.0
        assert vec[names.index("rsam_over_quiet_baseline_log10")] == 0.0
        assert vec[names.index("delta_available_flag")] == 0.0
        assert vec[names.index("volcano_onehot_shishaldin")] == 1.0
        # Reserved dims stay zero.
        reserved = [i for i, n in enumerate(names) if n.startswith("reserved_zero_")]
        assert all(vec[i] == 0.0 for i in reserved)

    def test_feature_vector_baseline_ratio(self, day_record: dict[str, Any]) -> None:
        rsam = float(day_record["rsam_24h"])
        vec = ve.assemble_feature_vector(
            day_record,
            volcano_index=2,
            is_broadband=True,
            baseline=rsam / 10.0,
            prev_rsam=rsam,
        )
        names = list(ve.FEATURE_NAMES)
        assert vec[names.index("baseline_available_flag")] == 1.0
        assert vec[names.index("rsam_over_quiet_baseline_log10")] == pytest.approx(1.0, abs=1e-5)
        assert vec[names.index("rsam_delta_1d_log10")] == pytest.approx(0.0, abs=1e-5)

    def test_hourly_matrix_shape_and_presence(self, day_record: dict[str, Any]) -> None:
        seq = ve.assemble_hourly_matrix(day_record, baseline=None)
        assert seq.shape == (24, 32)
        names = list(ve.HOURLY_FEATURE_NAMES)
        frac = seq[:, names.index("hour_data_fraction")]
        assert frac[0] > 0.9  # the excerpt covers hour 0
        assert (frac[2:] == 0.0).all()  # and nothing after ~01:05


class TestLabelWindows:
    """Label geometry around the real Shishaldin 2023-07-11 onset."""

    @pytest.fixture(scope="class")
    def labels(self) -> dict[str, ve.VolcanoLabels]:
        return ve.parse_gvp_labels(GVP_FIXTURE, today=dt.date(2026, 7, 10))

    def test_onset_parsed_with_vei_and_uncertainty(
        self, labels: dict[str, ve.VolcanoLabels]
    ) -> None:
        onsets = {o.date.isoformat(): o for o in labels["Shishaldin"].onsets}
        assert "2023-07-11" in onsets
        assert onsets["2023-07-11"].vei == 3
        assert onsets["2023-07-11"].day_uncertainty == 1

    @pytest.mark.parametrize(
        ("day", "expected"),
        [
            ("2023-06-27", "positive"),  # exactly K=14 days before onset
            ("2023-07-01", "positive"),
            ("2023-07-10", "positive"),  # 1 day before onset
            ("2023-07-11", "eruptive"),  # onset day itself: excluded
            ("2023-09-01", "eruptive"),  # inside start..end (ends 2023-11-03)
            ("2023-12-15", "buffer"),  # < 60 d after the eruption end
            ("2023-06-20", "buffer"),  # < 60 d before onset, not in K-window
            ("2024-03-01", "negative"),  # > 60 d clear of everything
        ],
    )
    def test_window_classes(
        self, labels: dict[str, ve.VolcanoLabels], day: str, expected: str
    ) -> None:
        material = labels["Shishaldin"]
        cls, lead, vei = ve.label_day(
            dt.date.fromisoformat(day), material.onsets, material.eruptive
        )
        assert cls == expected
        if expected == "positive":
            assert lead == (dt.date(2023, 7, 11) - dt.date.fromisoformat(day)).days
            assert vei == 3

    def test_positive_lead_bounds(self, labels: dict[str, ve.VolcanoLabels]) -> None:
        material = labels["Shishaldin"]
        cls, _, _ = ve.label_day(dt.date(2023, 6, 26), material.onsets, material.eruptive)
        assert cls != "positive"  # 15 days ahead is outside the K-window


class TestBuildDatasetCausality:
    """Pipeline-level no-lookahead + split enforcement on real day records."""

    DAYS: tuple[tuple[str, str, int | None], ...] = (
        # (day, cls, days_to_onset) -- each split holds both classes.
        ("2016-01-05", "negative", None),
        ("2016-01-12", "negative", None),
        ("2016-01-19", "negative", None),
        ("2016-01-26", "negative", None),
        ("2016-02-02", "positive", 5),
        ("2018-03-01", "negative", None),
        ("2018-03-08", "positive", 3),
        ("2021-04-01", "negative", None),
        ("2021-04-08", "positive", 7),
    )

    @pytest.fixture()
    def ctx(self, tmp_path: Path, day_record: dict[str, Any]) -> PipelineContext:
        """A tmp pipeline dir whose day-record caches hold the real record."""
        ctx = PipelineContext(data_dir=tmp_path)
        plan = []
        for day, cls, lead in self.DAYS:
            cache = ve._day_record_cache_path(ctx, "SSBA", "BHZ", dt.date.fromisoformat(day))
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **{k: np.asarray(v) for k, v in day_record.items()})
            plan.append(
                {
                    "volcano": "Shishaldin",
                    "day": day,
                    "sta": "SSBA",
                    "cha": "BHZ",
                    "scale": 5e8,
                    "sample_rate": 50.0,
                    "cls": cls,
                    "days_to_onset": lead,
                    "vei": 3 if cls == "positive" else None,
                }
            )
        (tmp_path / "volcanic").mkdir(exist_ok=True)
        (tmp_path / "volcanic" / "sample_plan.json").write_text(json.dumps(plan))
        return ctx

    def test_split_masks_follow_years(self, ctx: PipelineContext) -> None:
        ds = ve.build_dataset(ctx)
        train, val, test = ve.SPLIT.masks(ds.years)
        assert int(train.sum()) == 5 and int(val.sum()) == 2 and int(test.sum()) == 2
        for mask in (train, val, test):
            assert ds.labels[mask].min() == 0.0 and ds.labels[mask].max() == 1.0

    def test_standardization_stats_come_from_train_only(self, ctx: PipelineContext) -> None:
        ds = ve.build_dataset(ctx)
        train, _, _ = ve.SPLIT.masks(ds.years)
        np.testing.assert_allclose(
            ds.feature_mean, ds.features[train].mean(axis=0), rtol=1e-5, atol=1e-6
        )

    def test_causal_baseline_needs_three_prior_quiet_days(self, ctx: PipelineContext) -> None:
        ds = ve.build_dataset(ctx)
        by_day = {m["day"]: i for i, m in enumerate(ds.meta)}
        flag = list(ve.FEATURE_NAMES).index("baseline_available_flag")
        assert ds.features[by_day["2016-01-05"], flag] == 0.0  # nothing before it
        assert ds.features[by_day["2016-02-02"], flag] == 1.0  # 4 prior quiet days

    def test_no_lookahead_future_day_cannot_change_past_features(
        self, ctx: PipelineContext, day_record: dict[str, Any]
    ) -> None:
        """Corrupting the FUTURE day's record must leave earlier days bit-identical."""
        before = ve.build_dataset(ctx)
        last_day = dt.date.fromisoformat(self.DAYS[-1][0])
        cache = ve._day_record_cache_path(ctx, "SSBA", "BHZ", last_day)
        perturbed = {k: np.asarray(v) for k, v in day_record.items()}
        perturbed["minute_rsam"] = perturbed["minute_rsam"] * 100.0
        perturbed["rsam_24h"] = perturbed["rsam_24h"] * 100.0
        cache.unlink()
        np.savez_compressed(cache, **perturbed)
        after = ve.build_dataset(ctx)
        earlier = [i for i, m in enumerate(before.meta) if m["day"] != last_day.isoformat()]
        np.testing.assert_array_equal(before.features[earlier], after.features[earlier])
        np.testing.assert_array_equal(before.hourly[earlier], after.hourly[earlier])
        # ... while the perturbed day itself did change (the test has teeth).
        (changed,) = [i for i, m in enumerate(before.meta) if m["day"] == last_day.isoformat()]
        assert not np.array_equal(before.features[changed], after.features[changed])


_SHIPPED = shipped_checkpoint_path("volcanic_avo_seismic")


@pytest.mark.skipif(not _SHIPPED.exists(), reason="volcanic_avo_seismic not shipped")
class TestShippedVolcanicCheckpoint:
    """Contract + differential tests against the merit-gated shipped artifact."""

    @pytest.fixture(scope="class")
    def payload(self) -> dict[str, Any]:
        from omni_mercury_engine.models.checkpoint_paths import load_shipped_checkpoint

        payload, provenance = load_shipped_checkpoint("volcanic_avo_seismic")
        assert provenance is not None, "shipped checkpoint must carry provenance"
        assert provenance["evaluation"]["learned_beats_physics"] is True
        return payload

    def test_payload_contract(self, payload: dict[str, Any]) -> None:
        assert payload["feature_spec"] == "volcano-seismic-v1"
        assert payload["label"] == "eruption onset within 14d"
        assert len(payload["feature_names"]) == 128
        assert len(payload["feature_mean"]) == 128 and len(payload["feature_std"]) == 128
        volcanoes = payload["volcanoes"]
        assert len(volcanoes) >= 6, "scoped deliverable: >= 6 NAMED volcanoes"
        assert set(volcanoes) == {v.name for v in ve.VOLCANOES}
        assert "eruption_model" in payload and "seismic_detector" in payload

    def test_differential_physics_vs_learned(
        self, payload: dict[str, Any], day_record: dict[str, Any]
    ) -> None:
        """Same real observation; physics abstains from forecasting, learned forecasts."""
        from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector

        minute = np.asarray(day_record["minute_rsam"], dtype=float)
        minute = minute[np.isfinite(minute)]

        physics = VolcanicEruptionDetector()
        physics_result = physics.predict_eruption({"seismic_sequence": minute})
        assert physics._neural_trained is False
        assert physics_result.time_to_eruption_hours is None  # forecast never ran

        mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        std = np.asarray(payload["feature_std"], dtype=np.float32)
        vec = ve.assemble_feature_vector(
            day_record, volcano_index=0, is_broadband=True, baseline=None, prev_rsam=None
        )
        hmean = np.asarray(payload["hourly_feature_mean"], dtype=np.float32)
        hstd = np.asarray(payload["hourly_feature_std"], dtype=np.float32)
        seq = (ve.assemble_hourly_matrix(day_record, baseline=None) - hmean) / hstd

        learned = VolcanicEruptionDetector()
        learned.load_neural_weights()  # shipped default
        assert learned._neural_trained is True
        result = learned.predict_eruption(
            {"fused_features": ((vec - mean) / std).tolist(), "seismic_sequence": seq}
        )
        assert np.isfinite(result.confidence) and 0.0 <= result.confidence <= 1.0
        assert result.time_to_eruption_hours is not None  # neural forecast ran
        assert result.vei_estimate is not None and 0 <= result.vei_estimate <= 7

    def test_corrupt_checkpoint_fails_loud(self, tmp_path: Path) -> None:
        import pickle

        from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector

        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"not a checkpoint")
        detector = VolcanicEruptionDetector()
        with pytest.raises((pickle.UnpicklingError, RuntimeError)):
            detector.load_neural_weights(str(bad))
        assert detector._neural_trained is False

    def test_tampered_shipped_checkpoint_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.models import checkpoint_paths as cp

        fake_dir = tmp_path / "checkpoints"
        fake_dir.mkdir()
        tampered = fake_dir / _SHIPPED.name
        shutil.copy(_SHIPPED, tampered)
        shutil.copy(
            _SHIPPED.with_name("volcanic_avo_seismic.provenance.json"),
            fake_dir / "volcanic_avo_seismic.provenance.json",
        )
        with tampered.open("r+b") as fh:
            fh.seek(256)
            byte = fh.read(1)
            fh.seek(256)
            fh.write(bytes([byte[0] ^ 0xFF]))
        monkeypatch.setattr(cp, "checkpoints_dir", lambda: fake_dir)
        with pytest.raises(RuntimeError, match="does not match its provenance"):
            cp.load_shipped_checkpoint("volcanic_avo_seismic")
