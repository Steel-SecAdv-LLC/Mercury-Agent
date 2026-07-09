# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tsunami/Earthquake detectors must derive from observed physics, never fabricate.

Regression locks for the anti-theater fixes in ``disaster_detectors``:

* ``TsunamiDetector`` previously took confidence and wave height from an
  UNTRAINED ``WaveformFFTAnalyzer``. Both now come from the observed record:
  wave height is the peak sea-level deviation from the median baseline (what a
  DART bottom-pressure recorder measures, noise floor from the quietest segment
  so a long excursion cannot mask itself) and confidence is a noisy-OR of that
  amplitude severity with the tsunami-band resonance score.
* ``EarthquakeDetector`` previously let the untrained ``SeismicWaveAnalyzer``'s
  random p_prob/s_prob heads GATE the real STA/LTA picker, and fabricated a
  Richter magnitude (net·4+2). The STA/LTA physics now runs unconditionally
  (S searched after the P trigger), confidence is the trigger severity blended
  with band resonance, and no magnitude is emitted without a trained model
  (``estimated_magnitude=None``, class ``"undetermined"``).
"""

from __future__ import annotations

import numpy as np
import torch

from omni_mercury_engine.detectors.geological.disaster_detectors import (
    EarthquakeDetector,
    TsunamiDetector,
)


def _tsunami_record(rng: np.random.Generator, amplitude: float = 1.5) -> np.ndarray:
    n = 4096
    t = np.arange(n)
    noise = rng.normal(0, 0.05, n)
    return np.asarray(noise + amplitude * np.sin(2 * np.pi * 0.005 * t) * (t > 2000))


def _quake_record(rng: np.random.Generator) -> np.ndarray:
    n = 6000
    trace = rng.normal(0, 0.1, n)
    trace[3000:3500] += rng.normal(0, 3.0, 500)  # P onset
    trace[3500:4500] += rng.normal(0, 6.0, 1000)  # S coda
    return trace


class TestTsunamiHonesty:
    def test_default_detector_is_untrained(self) -> None:
        assert TsunamiDetector()._neural_trained is False

    def test_untrained_path_ignores_neural_model_weights(self) -> None:
        det = TsunamiDetector(sampling_rate=1.0)
        record = _tsunami_record(np.random.default_rng(0))
        before = det.predict_tsunami(record).confidence
        with torch.no_grad():
            for p in det.waveform_analyzer.parameters():
                p.mul_(0).add_(4.0)
        after = det.predict_tsunami(record).confidence
        assert before == after

    def test_wave_height_is_the_observed_amplitude(self) -> None:
        """The reported height must track the physical excursion, not a net output."""
        det = TsunamiDetector(sampling_rate=1.0)
        rng = np.random.default_rng(1)
        small = det.predict_tsunami(_tsunami_record(rng, amplitude=0.5))
        rng = np.random.default_rng(1)
        large = det.predict_tsunami(_tsunami_record(rng, amplitude=3.0))
        assert 0.3 < small.estimated_wave_height_m < 1.0
        assert large.estimated_wave_height_m > 2.5
        assert large.severity in {"warning", "major"}

    def test_detects_wave_and_rejects_quiet_sea(self) -> None:
        det = TsunamiDetector(sampling_rate=1.0)
        rng = np.random.default_rng(0)
        wave = det.predict_tsunami(_tsunami_record(rng))
        quiet = det.predict_tsunami(np.random.default_rng(2).normal(0, 0.05, 4096))
        assert wave.tsunami_detected is True
        assert quiet.tsunami_detected is False
        assert wave.confidence > quiet.confidence

    def test_deterministic(self) -> None:
        det = TsunamiDetector(sampling_rate=1.0)
        record = _tsunami_record(np.random.default_rng(3))
        assert det.predict_tsunami(record).confidence == det.predict_tsunami(record).confidence


class TestEarthquakeHonesty:
    def test_default_detector_is_untrained(self) -> None:
        assert EarthquakeDetector()._neural_trained is False

    def test_no_magnitude_is_fabricated_when_untrained(self) -> None:
        det = EarthquakeDetector(sampling_rate=100.0)
        result = det.predict_earthquake(_quake_record(np.random.default_rng(0)))
        assert result.estimated_magnitude is None
        assert result.magnitude_class == "undetermined"
        assert result.aftershock_probability == 0.0

    def test_untrained_path_ignores_neural_model_weights(self) -> None:
        det = EarthquakeDetector(sampling_rate=100.0)
        record = _quake_record(np.random.default_rng(1))
        before = det.predict_earthquake(record).confidence
        with torch.no_grad():
            for p in det.seismic_analyzer.parameters():
                p.mul_(0).add_(2.0)
        after = det.predict_earthquake(record).confidence
        assert before == after

    def test_sta_lta_physics_detects_quake_and_rejects_calm(self) -> None:
        det = EarthquakeDetector(sampling_rate=100.0)
        quake = det.predict_earthquake(_quake_record(np.random.default_rng(0)))
        calm = det.predict_earthquake(np.random.default_rng(2).normal(0, 0.1, 6000))
        assert quake.earthquake_detected is True
        assert calm.earthquake_detected is False

    def test_p_and_s_arrivals_are_ordered_and_distance_computed(self) -> None:
        """P/S picks come from the STA/LTA physics (not random NN heads), the S
        pick searches after P, and the S−P time yields an epicenter distance."""
        det = EarthquakeDetector(sampling_rate=100.0)
        result = det.predict_earthquake(_quake_record(np.random.default_rng(0)))
        assert result.p_wave_detected and result.s_wave_detected
        assert result.p_wave_arrival_time is not None
        assert result.s_wave_arrival_time is not None
        assert result.s_wave_arrival_time > result.p_wave_arrival_time
        assert result.epicenter_distance_km is not None and result.epicenter_distance_km > 0

    def test_extract_features_survives_none_magnitude(self) -> None:
        det = EarthquakeDetector(sampling_rate=100.0)
        features = det.extract_features(_quake_record(np.random.default_rng(3)))
        assert np.isfinite(features).all()


class TestLegacyTrainerSyntheticFallbackOptIn:
    """Legacy trainers must not silently degrade to synthetic training data.

    Regression: when the real DART/USGS loaders failed, these helpers used
    to fall back to synthetic samples automatically — weights trained that
    way are indistinguishable from real-trained weights downstream.
    """

    def test_waveform_trainer_fails_loud_without_opt_in(self, monkeypatch: object) -> None:
        import pytest

        from omni_mercury_engine.detectors.geological import disaster_detectors as dd

        monkeypatch.setattr(dd, "load_dart_buoy_data", lambda *a, **k: None)  # type: ignore[attr-defined]
        model = dd.WaveformFFTAnalyzer()
        with pytest.raises(RuntimeError, match=r"synthetic .*not explicitly allowed"):
            dd.train_waveform_analyzer(model, n_epochs=1, n_samples=8)

    def test_waveform_trainer_synthetic_requires_explicit_flag(self, monkeypatch: object) -> None:
        from omni_mercury_engine.detectors.geological import disaster_detectors as dd

        monkeypatch.setattr(dd, "load_dart_buoy_data", lambda *a, **k: None)  # type: ignore[attr-defined]
        model = dd.WaveformFFTAnalyzer()
        history = dd.train_waveform_analyzer(
            model, n_epochs=1, n_samples=8, allow_synthetic_fallback=True
        )
        assert "loss" in history or history  # trained without raising

    def test_seismic_trainer_fails_loud_without_opt_in(self, monkeypatch: object) -> None:
        import pytest

        from omni_mercury_engine.detectors.geological import disaster_detectors as dd

        monkeypatch.setattr(dd, "load_usgs_earthquake_catalog", lambda *a, **k: None)  # type: ignore[attr-defined]
        model = dd.SeismicWaveAnalyzer()
        with pytest.raises(RuntimeError, match=r"synthetic .*not explicitly allowed"):
            dd.train_seismic_analyzer(model, n_epochs=1, n_samples=8)
