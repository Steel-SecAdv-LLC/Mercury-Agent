# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression suite for the "(1, W) already-batched single sample" crash class.

The solar geomagnetic lane crashed when a caller handed
``magnetosphere_data["features"]`` already batched as ``(1, 32)``: the value
passed a ``shape[-1] == contract`` guard, then ``unsqueeze(0)`` produced a 3-D
tensor and ``BatchNorm1d`` crashed (fixed in ``solar_storm_detector.py`` with
the squeeze-then-strict-``ndim == 1`` pattern).  A sweep of every other
net-backed trained detector lane (2026-07-17) reproduced the same class in
three more lanes and adjacent-rank holes in four; this suite pins each fix:

* landslide ``slope_features`` -- ``(1, 64)`` passed an ``ndim >= 1`` guard
  and crashed ``BatchNorm1d``;
* volcanic eruption ``fused_features`` -- NO input guard at all: ``(1, 128)``,
  ``(2, 128)``, 0-d and wrong-width all crashed the model stack;
* hurricane ``wind_field`` -- a ``(1, W)`` transect became an ``(H=1, W)``
  frame and ``MaxPool2d(2)`` pooled it to height 0;
* wildfire -- a ``(3, 1, W)`` raster passed the channel guard and died in the
  second ``MaxPool2d``;
* seismic ``predict_earthquake`` -- 0-d and rank>=3 inputs crashed in
  ``seismic_data[0]`` / ``scipy.signal.spectrogram`` with opaque errors;
* parapsychology -- a 0-d ``reg_output`` crashed ``len()``; an off-contract
  ``(N, W)`` stack was silently flattened into one N*W-step LSTM sequence;
* disaster-precursor ``_predict_earthquake`` -- latent unguarded
  ``BatchNorm1d`` feed (nothing ships for it, but operator-loaded weights
  would have crashed exactly like landslide).

Uniform contract: a single-row ``(1, W)`` input is the ``(W,)`` sample and is
*consumed*; anything else off-rank falls to the disclosed physics/neutral path
(or fails loud where no physics fallback exists).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("torch")


def _linear_width(sequential: Any) -> int:
    """Trained contract width from the first Linear of a module stack."""
    return int(cast("Any", sequential)[0].in_features)


from omni_mercury_engine.detectors.geological.disaster_detectors import EarthquakeDetector
from omni_mercury_engine.detectors.geological.hurricane_detector import HurricaneDetector
from omni_mercury_engine.detectors.geological.landslide import LandslideDetector
from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector
from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector
from omni_mercury_engine.models.parapsychology import ParapsychologyDetector
from omni_mercury_engine.space.disaster_precursor_detector import DisasterPrecursorDetector


class TestLandslideBatchedSlopeFeatures:
    """Landslide ``slope_features``: squeeze (1, W); off-rank -> physics."""

    @pytest.fixture(scope="class")
    def detector(self) -> LandslideDetector:
        det = LandslideDetector()
        if not det._neural_trained:
            pytest.skip("no shipped landslide checkpoint in this build")
        return det

    @staticmethod
    def _payload(slope_features: np.ndarray | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "rainfall_data": {"intensity_mm_hr": 20.0, "duration_hours": 8.0}
        }
        if slope_features is not None:
            payload["slope_features"] = slope_features
        return payload

    def test_batched_single_row_matches_flat_vector(self, detector: LandslideDetector) -> None:
        assert detector.stability_model is not None
        width = _linear_width(detector.stability_model.feature_encoder)
        rng = np.random.default_rng(7)
        features = rng.normal(size=width).astype(np.float32)
        flat = detector.predict_landslide(self._payload(features))
        batched = detector.predict_landslide(self._payload(features.reshape(1, width)))
        assert batched.slope_failure_probability == pytest.approx(
            flat.slope_failure_probability
        ), "(1, W) must run the trained encoder exactly like (W,)"

    @pytest.mark.parametrize("leading", [(2,), (1, 1)])
    def test_off_rank_falls_to_physics_without_crashing(
        self, detector: LandslideDetector, leading: tuple[int, ...]
    ) -> None:
        assert detector.stability_model is not None
        width = _linear_width(detector.stability_model.feature_encoder)
        physics = detector.predict_landslide(self._payload(None))
        off = detector.predict_landslide(self._payload(np.zeros((*leading, width))))
        assert off.slope_failure_probability == pytest.approx(
            physics.slope_failure_probability
        ), f"off-rank {(*leading, width)} must take the geotechnical physics path"


class TestVolcanicFusedFeaturesGuard:
    """Volcanic eruption ``fused_features``: previously wholly unguarded."""

    @pytest.fixture(scope="class")
    def detector(self) -> VolcanicEruptionDetector:
        det = VolcanicEruptionDetector()
        if not det._neural_trained:
            pytest.skip("no shipped volcanic checkpoint in this build")
        return det

    def test_batched_single_row_matches_flat_vector(
        self, detector: VolcanicEruptionDetector
    ) -> None:
        width = _linear_width(detector.eruption_model.feature_fusion)
        rng = np.random.default_rng(11)
        features = rng.normal(size=width).astype(np.float32)
        flat = detector._forecast_eruption({"fused_features": features}, 0.0, {})
        batched = detector._forecast_eruption(
            {"fused_features": features.reshape(1, width)}, 0.0, {}
        )
        assert flat["method"] == "neural"
        assert batched["method"] == "neural"
        assert batched["confidence"] == pytest.approx(flat["confidence"])

    def test_off_contract_falls_to_physics_without_crashing(
        self, detector: VolcanicEruptionDetector
    ) -> None:
        width = _linear_width(detector.eruption_model.feature_fusion)
        for bad in (
            np.zeros((2, width)),
            np.zeros((1, 1, width)),
            np.zeros(width // 2),
            np.array(1.0),
        ):
            forecast = detector._forecast_eruption({"fused_features": bad}, 0.0, {})
            assert forecast["method"] == "physics", f"shape {bad.shape} must fall to physics"


class TestHurricaneDegenerateWindField:
    """Hurricane wind field: single-row transects must not reach MaxPool2d."""

    @pytest.fixture(scope="class")
    def detector(self) -> HurricaneDetector:
        det = HurricaneDetector()
        if not det._neural_trained:
            pytest.skip("no shipped hurricane checkpoint in this build")
        return det

    def test_proper_field_runs_neural(self, detector: HurricaneDetector) -> None:
        rng = np.random.default_rng(3)
        u = rng.normal(10.0, 3.0, size=(16, 16)).astype(np.float32)
        result = detector._analyze_wind_field_neural({"u": u, "v": u})
        assert "neural_max_wind_kt" in result

    @pytest.mark.parametrize("shape", [(1, 32), (32, 1), (1, 1, 32)])
    def test_degenerate_spatial_dims_fall_to_physics(
        self, detector: HurricaneDetector, shape: tuple[int, ...]
    ) -> None:
        u = np.ones(shape, dtype=np.float32)
        result = detector._analyze_wind_field_neural({"u": u, "v": u})
        assert "neural_max_wind_kt" not in result, (
            f"wind field {shape} has a spatial dim < 2 and must take the "
            "physics analysis, not crash the conv encoder's pooling"
        )


class TestWildfireDegenerateRaster:
    """Wildfire: rasters below the 4x4 pooling minimum fall to physics."""

    @pytest.fixture(scope="class")
    def detector(self) -> WildfireDetector:
        det = WildfireDetector()
        if not det._neural_trained:
            pytest.skip("no shipped wildfire checkpoint in this build")
        return det

    def test_proper_raster_runs_neural(self, detector: WildfireDetector) -> None:
        raster = np.full((3, 32, 32), 300.0, dtype=np.float32)
        result = detector._detect_ignition(raster)
        assert result.get("method") != "physics_brightness_temperature"

    @pytest.mark.parametrize("shape", [(3, 1, 32), (3, 32, 1), (3, 3, 3)])
    def test_degenerate_raster_falls_to_physics(
        self, detector: WildfireDetector, shape: tuple[int, ...]
    ) -> None:
        raster = np.full(shape, 300.0, dtype=np.float32)
        result = detector._detect_ignition(raster)
        assert result.get("method") == "physics_brightness_temperature", (
            f"raster {shape} is below the CNN's two-MaxPool 4x4 spatial "
            "minimum and must take the physics detector"
        )


class TestSeismicWaveformRankContract:
    """predict_earthquake: 0-d / rank>=3 fail loud with the documented contract."""

    @pytest.fixture(scope="class")
    def detector(self) -> EarthquakeDetector:
        return EarthquakeDetector()

    def test_batched_single_trace_matches_flat_trace(self, detector: EarthquakeDetector) -> None:
        rng = np.random.default_rng(5)
        trace = rng.normal(size=2048)
        flat = detector.predict_earthquake(trace)
        batched = detector.predict_earthquake(trace.reshape(1, -1))
        assert batched.earthquake_detected == flat.earthquake_detected
        assert batched.confidence == pytest.approx(flat.confidence)

    @pytest.mark.parametrize("bad", [np.array(1.0), np.zeros((1, 1, 2048)), np.zeros((2, 2, 16))])
    def test_off_rank_raises_the_documented_contract(
        self, detector: EarthquakeDetector, bad: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match=r"\[seq_len\] or \[batch, seq_len\]"):
            detector.predict_earthquake(bad)


class TestParapsychologyRegOutputRanks:
    """reg_output: 0-d analyzed as one sample; (1, W) window analyzed; (N, W) abstains."""

    @pytest.fixture(scope="class")
    def detector(self) -> ParapsychologyDetector:
        return ParapsychologyDetector()

    def test_zero_d_reg_output_does_not_crash(self, detector: ParapsychologyDetector) -> None:
        result = detector.detect_psi_anomaly({"reg_output": np.array(100.0)})
        assert np.isfinite(result.z_score)
        assert np.isfinite(result.p_value)

    def test_multirow_counts_all_samples(self, detector: ParapsychologyDetector) -> None:
        rng = np.random.default_rng(9)
        samples = rng.normal(100.0, 15.0, size=1000)
        flat = detector._analyze_reg_output(samples)
        stacked = detector._analyze_reg_output(samples.reshape(10, 100))
        # Same bag of samples => identical statistics (len() used to see 10).
        assert stacked == pytest.approx(flat)

    def test_field_coherence_rank_contract(self, detector: ParapsychologyDetector) -> None:
        if not detector._neural_trained:
            pytest.skip("no shipped reg_deviation_gcp checkpoint in this build")
        rng = np.random.default_rng(13)
        window = rng.normal(100.0, 15.0, size=100)
        flat = detector._analyze_field_coherence(window)
        batched = detector._analyze_field_coherence(window.reshape(1, 100))
        # (1, W) is the single window, analyzed -- previously it abstained by
        # accident because len() saw the first axis.
        assert batched == pytest.approx(flat)
        # An off-contract (N>1, W) stack abstains instead of being silently
        # flattened into one N*W-step sequence.
        assert detector._analyze_field_coherence(window.reshape(10, 10)) == 0.5
        assert detector._analyze_field_coherence(np.array(100.0)) == 0.5


class TestPrecursorLatentFeedGuard:
    """_predict_earthquake: the latent BatchNorm feed is now guarded."""

    @pytest.fixture()
    def trained_detector(self) -> DisasterPrecursorDetector:
        det = DisasterPrecursorDetector()
        if det.earthquake_analyzer is None:
            pytest.skip("earthquake analyzer disabled in this configuration")
        # Nothing ships for this lane; mark the (random-weight) analyzer
        # trained ONLY to exercise the shape contract -- outputs are not
        # asserted on beyond structure.
        det._neural_trained = True
        return det

    def test_batched_single_row_is_accepted(
        self, trained_detector: DisasterPrecursorDetector
    ) -> None:
        assert trained_detector.earthquake_analyzer is not None
        width = _linear_width(trained_detector.earthquake_analyzer.em_feature_extractor)
        out = trained_detector._predict_earthquake(np.zeros((1, width), dtype=np.float32))
        assert set(out) >= {"event_probability", "confidence"}

    def test_off_contract_fails_loud(self, trained_detector: DisasterPrecursorDetector) -> None:
        assert trained_detector.earthquake_analyzer is not None
        width = _linear_width(trained_detector.earthquake_analyzer.em_feature_extractor)
        for bad in (np.zeros((2, width)), np.zeros(width // 2), np.zeros((1, 1, width))):
            with pytest.raises(ValueError, match="seismicity-catalog-v2"):
                trained_detector._predict_earthquake(bad)
