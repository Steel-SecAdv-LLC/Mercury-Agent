# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the hurricane_wind training pipeline (ERA5 + IBTrACS).

Offline by design: the committed fixture holds two REAL extracted samples
from the pipeline's own patch cache (ERA5 10 m u/v/speed patch sequences at
t-6h and t, 33x33 grid points, float32 m/s), one IBTrACS-labeled positive
and one far-from-storm negative, with their labels stored alongside. No
network, no synthetic wind fields; pipeline-stage network code runs in the
training lane, not here.

Fixture provenance (IBTrACS v04r01 rows + ARCO-ERA5 chunks, extracted by
``hurricane_wind.fetch`` with the default seed):

* sample 0 (positive): Super Typhoon Mawar, ``SID=2023138N05151``, BASIN=WP,
  ``ISO_TIME=2023-05-26 00:00:00`` (hours-since-1900 = 1081680), LAT=15.1,
  LON=139.2, ``USA_WIND=165.0`` kt, ``USA_SSHS=5`` -> category_idx 7
  (category_5). The ERA5 patch max is only ~63.1 kt -- the structural
  under-resolution of TC cores that the learned model must correct.
* sample 1 (negative): seeded far-from-storm point at (-24.16, 116.80),
  ``ISO_TIME=2023-01-20 00:00:00`` (hours-since-1900 = 1078656); its honest
  intensity label is its own observed patch max wind, ~17.51 kt,
  category_idx 0 (no_cyclone).

Covered: patch->tensor->detector plumbing (the physics path must report
exactly the patch's observed max wind), the USA_SSHS -> 8-class category
mapping against the detector's ``NEURAL_CATEGORY_ORDER``, temporal-split
enforcement, the learned path's floor-at-observed-wind guarantee with a
freshly saved checkpoint, and the differential physics-vs-shipped comparison
(skipped until a ``hurricane_era5`` checkpoint ships through the merit gate).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.detectors.geological.hurricane_detector import (
    NEURAL_CATEGORY_ORDER,
    HurricaneDetector,
    WindPatternAnalyzer,
)
from omni_mercury_engine.ml.hazard_training import hurricane_wind as hw
from omni_mercury_engine.ml.hazard_training.common import TemporalSplit
from omni_mercury_engine.models.checkpoint_paths import shipped_checkpoint_path

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "hazard_training"
    / "hurricane"
    / "era5_patch_sample.npz"
)


@pytest.fixture(name="samples")
def _samples() -> dict[str, np.ndarray[Any, Any]]:
    """Load the committed real-data fixture (see module docstring)."""
    assert FIXTURE.exists(), f"missing committed fixture {FIXTURE}"
    with np.load(FIXTURE, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _wind_field_case(x: np.ndarray[Any, Any]) -> dict[str, Any]:
    """Build the public-API input dict exactly as evaluate() does."""
    return {
        "wind_field": {
            "u": x[:, 0].astype(np.float64),
            "v": x[:, 1].astype(np.float64),
            "grid_spacing_m": hw.GRID_SPACING_M,
        }
    }


class TestPatchFixture:
    """The committed patches are internally consistent real data."""

    def test_tensor_shape_and_dtype(self, samples: dict[str, np.ndarray[Any, Any]]) -> None:
        x = samples["x"]
        assert x.dtype == np.float32
        assert x.shape[1:] == (len(hw.SEQ_OFFSET_HOURS), len(hw.CHANNELS), 33, 33)
        assert bool(samples["is_positive"][0]) and not bool(samples["is_positive"][1])

    def test_speed_channel_is_hypot_of_components(
        self, samples: dict[str, np.ndarray[Any, Any]]
    ) -> None:
        x = samples["x"]
        np.testing.assert_allclose(x[:, :, 2], np.hypot(x[:, :, 0], x[:, :, 1]), rtol=1e-5)

    def test_obs_max_kt_matches_speed_channel(
        self, samples: dict[str, np.ndarray[Any, Any]]
    ) -> None:
        x = samples["x"]
        expected = x[:, :, 2].reshape(x.shape[0], -1).max(axis=1) * hw.MS_TO_KT
        np.testing.assert_allclose(samples["obs_max_kt"], expected, rtol=1e-5)

    def test_negative_intensity_label_is_its_own_observed_wind(
        self, samples: dict[str, np.ndarray[Any, Any]]
    ) -> None:
        """The honest no-cyclone intensity truth = the patch's ERA5 max wind."""
        np.testing.assert_allclose(samples["intensity_kt"][1], samples["obs_max_kt"][1], rtol=1e-5)
        assert samples["category_idx"][1] == 0  # no_cyclone bucket

    def test_positive_label_is_hurricane_strength_best_track(
        self, samples: dict[str, np.ndarray[Any, Any]]
    ) -> None:
        """USA_WIND exceeds the patch max: ERA5 under-resolves TC cores."""
        assert samples["intensity_kt"][0] >= hw.HURRICANE_WIND_KT
        assert samples["intensity_kt"][0] > samples["obs_max_kt"][0]
        sshs = int(samples["usa_sshs"][0])
        assert hw.SSHS_TO_CATEGORY_INDEX[sshs] == int(samples["category_idx"][0])


class TestDetectorPlumbing:
    """The patch tensors drive the public detector API on both paths."""

    def test_physics_path_reports_observed_patch_max_wind(
        self, samples: dict[str, np.ndarray[Any, Any]]
    ) -> None:
        det = HurricaneDetector(load_shipped_weights=False)
        for i in range(samples["x"].shape[0]):
            result = det.predict_hurricane(_wind_field_case(samples["x"][i]))
            assert result.max_wind_speed_kt == pytest.approx(
                float(samples["obs_max_kt"][i]), rel=1e-4
            )

    def test_learned_path_is_floored_at_observed_wind(
        self, samples: dict[str, np.ndarray[Any, Any]], tmp_path: Path
    ) -> None:
        """Any loaded checkpoint (here: fresh weights) can only raise the
        estimate above the observed patch max, never mask real wind."""
        torch.manual_seed(0)
        ckpt = tmp_path / "fresh.pt"
        torch.save({"wind_analyzer": WindPatternAnalyzer().state_dict()}, ckpt)
        physics_det = HurricaneDetector(load_shipped_weights=False)
        det = HurricaneDetector()
        det.load_neural_weights(str(ckpt))
        assert det._neural_trained is True
        for i in range(samples["x"].shape[0]):
            case = _wind_field_case(samples["x"][i])
            physics_kt = physics_det.predict_hurricane(case).max_wind_speed_kt
            result = det.predict_hurricane(case)
            assert np.isfinite(result.max_wind_speed_kt)
            assert result.max_wind_speed_kt >= physics_kt

    def test_neural_path_falls_back_to_physics_on_unusable_input(
        self, samples: dict[str, np.ndarray[Any, Any]], tmp_path: Path
    ) -> None:
        """Bare speed arrays / mismatched components: physics, nothing imputed."""
        torch.manual_seed(0)
        ckpt = tmp_path / "fresh.pt"
        torch.save({"wind_analyzer": WindPatternAnalyzer().state_dict()}, ckpt)
        det = HurricaneDetector()
        det.load_neural_weights(str(ckpt))
        speed_only = samples["x"][0, -1, 2]
        out = det._analyze_wind_field_neural(speed_only)
        assert out == det._analyze_wind_field(speed_only)
        mismatched = {"u": np.zeros((3, 4)), "v": np.zeros((2, 2))}
        assert det._analyze_wind_field_neural(mismatched) == det._analyze_wind_field(mismatched)


class TestCategoryMapping:
    """USA_SSHS buckets mirror the detector's 8-class head EXACTLY."""

    def test_vocabulary_and_indices(self) -> None:
        assert len(NEURAL_CATEGORY_ORDER) == 8
        assert NEURAL_CATEGORY_ORDER[0] == "no_cyclone"
        expected = {
            -1: "tropical_depression",
            0: "tropical_storm",
            1: "category_1",
            2: "category_2",
            3: "category_3",
            4: "category_4",
            5: "category_5",
        }
        for sshs, name in expected.items():
            assert NEURAL_CATEGORY_ORDER[hw.SSHS_TO_CATEGORY_INDEX[sshs]] == name

    def test_non_tropical_sshs_codes_are_excluded_not_mislabeled(self) -> None:
        for sshs in (-2, -3, -4, -5):  # subtropical/disturbance/extratropical/unknown
            assert sshs not in hw.SSHS_TO_CATEGORY_INDEX

    def test_head_width_matches_vocabulary(self) -> None:
        model = WindPatternAnalyzer()
        final_linear = model.category_classifier[-1]
        assert final_linear.out_features == len(NEURAL_CATEGORY_ORDER)


class TestTemporalSplit:
    """Train < val < test by year; never random."""

    def test_pipeline_split_is_ordered_and_disjoint(self) -> None:
        assert max(hw.SPLIT.train_years) < min(hw.SPLIT.val_years)
        assert max(hw.SPLIT.val_years) < min(hw.SPLIT.test_years)
        years = np.array([1990, 2015, 2016, 2019, 2020, 2024])
        train, val, test = hw.SPLIT.masks(years)
        assert not np.any(train & val) and not np.any(val & test) and not np.any(train & test)
        assert int(train.sum()) == 2 and int(val.sum()) == 2 and int(test.sum()) == 2

    def test_interleaved_years_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="temporal split violated"):
            TemporalSplit(train_years=(2015, 2020), val_years=(2016,), test_years=(2021,))


@pytest.mark.skipif(
    not shipped_checkpoint_path("hurricane_era5").exists(),
    reason="no shipped hurricane_era5 checkpoint (merit gate not passed yet)",
)
class TestShippedDifferential:
    """Physics vs the shipped checkpoint on the held-out fixture patch."""

    def test_learned_beats_physics_on_fixture_hurricane(
        self, samples: dict[str, np.ndarray[Any, Any]]
    ) -> None:
        """On the committed hurricane-strength positive (a real test-year
        best-track point whose USA_WIND is stored in the fixture), the shipped
        network must land closer to the best-track intensity than the
        physics under-estimate -- the exact structural win the merit gate
        certified on the full held-out set."""
        case = _wind_field_case(samples["x"][0])
        truth_kt = float(samples["intensity_kt"][0])

        physics = HurricaneDetector(load_shipped_weights=False).predict_hurricane(case)
        learned_det = HurricaneDetector()
        learned_det.load_neural_weights()  # no path -> shipped default
        assert learned_det._feature_spec == hw.FEATURE_SPEC_VERSION
        learned = learned_det.predict_hurricane(case)

        assert learned.max_wind_speed_kt >= physics.max_wind_speed_kt  # observed floor
        assert learned.max_wind_speed_kt != pytest.approx(physics.max_wind_speed_kt)
        assert abs(learned.max_wind_speed_kt - truth_kt) < abs(physics.max_wind_speed_kt - truth_kt)

    def test_shipped_negative_stays_calm(self, samples: dict[str, np.ndarray[Any, Any]]) -> None:
        """The far-from-storm negative must not be inflated into a hurricane."""
        case = _wind_field_case(samples["x"][1])
        det = HurricaneDetector()
        det.load_neural_weights()
        result = det.predict_hurricane(case)
        assert result.max_wind_speed_kt < hw.HURRICANE_WIND_KT
