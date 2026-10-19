# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""``medical.pandemic.bio_threats.PathogenDetector``: a baseline-calibrated detector.

The released detector scored an unseeded random Ising matrix on readings
binarised at their own median, so the verdict depended on the seed (about 18%
of fresh detectors flagged the same input), shifts and rescalings were
invisible, and pathogen types and signatures were invented. The tests marked as
such fail on the released code.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="bio_threats package imports torch")

from omni_mercury_engine.medical.pandemic.bio_threats.pathogen_detector import (
    PathogenDetector,
)

_DIM = 12


_MIXING = np.eye(_DIM) + 0.4 * np.random.default_rng(99).normal(size=(_DIM, _DIM))


def _sensor_data(seed: int, n: int) -> np.ndarray:
    """Correlated readings from one fixed distribution; ``seed`` picks the draw."""
    return 50.0 + np.random.default_rng(seed).normal(size=(n, _DIM)) @ _MIXING.T


def _stats(detector: PathogenDetector) -> tuple[np.ndarray, np.ndarray]:
    """Fitted baseline mean and per-channel scale."""
    assert detector.mean_ is not None and detector.scale_ is not None
    return detector.mean_, detector.scale_


@pytest.fixture(scope="module")
def fitted() -> PathogenDetector:
    return PathogenDetector({"false_alarm_rate": 0.01}).fit(_sensor_data(0, 1000))


def test_verdict_does_not_depend_on_the_seed() -> None:
    """Fails on the released random coupling matrix."""
    baseline, snapshot = _sensor_data(0, 1000), _sensor_data(1, 1)[0] + 3.0
    verdicts = {
        PathogenDetector({"seed": seed}).fit(baseline).detect_pathogen(snapshot).energy_score
        for seed in range(5)
    }
    assert len(verdicts) == 1


def test_false_alarm_rate_is_bounded(fitted: PathogenDetector) -> None:
    normal = _sensor_data(2, 5000)
    rate = np.mean([fitted.detect_pathogen(row).threat_detected for row in normal])
    assert rate <= 0.02  # target 0.01; the conformal bound plus sampling error


def test_shift_and_rescaling_are_detected(fitted: PathogenDetector) -> None:
    """Fails on the released median binarisation (identical energy for all three)."""
    snapshot = _sensor_data(3, 1)[0]
    plain = fitted.detect_pathogen(snapshot)
    shifted = fitted.detect_pathogen(snapshot + 5.0)
    scaled = fitted.detect_pathogen(snapshot * 1.5)
    assert not plain.threat_detected
    assert shifted.threat_detected and scaled.threat_detected
    assert shifted.energy_score > plain.energy_score
    assert shifted.confidence > 0.99


def test_single_channel_spike_is_named(fitted: PathogenDetector) -> None:
    mean, scale = _stats(fitted)
    snapshot = mean.copy()
    snapshot[4] += 8.0 * scale[4]
    result = fitted.detect_pathogen(snapshot)
    assert result.threat_detected
    assert result.bio_signatures[0].startswith("channel 4: +8.0 sd above baseline")


def test_correlation_break_is_detected(fitted: PathogenDetector) -> None:
    corr = np.corrcoef(_sensor_data(0, 1000).T) - np.eye(_DIM)
    i, j = np.unravel_index(np.argmax(corr), corr.shape)
    mean, scale = _stats(fitted)
    snapshot = mean.copy()
    snapshot[i] += 2.5 * scale[i]  # strongly correlated pair pulled apart
    snapshot[j] -= 2.5 * scale[j]
    result = fitted.detect_pathogen(snapshot)
    assert result.threat_detected
    assert result.bio_signatures == [
        "Joint pattern atypical: no single channel beyond 3 sd (correlation change)"
    ]


def test_type_comes_from_a_masint_agent_report(fitted: PathogenDetector) -> None:
    """Fails on the released classifier, which invented types from spin ratios."""
    mean, scale = _stats(fitted)
    alarm = mean + 6.0 * scale
    assert fitted.detect_pathogen(alarm).pathogen_type == "unclassified_anomaly"
    reported = fitted.detect_pathogen(alarm, {"agent": "Anthrax", "threat_score": 0.9})
    assert reported.pathogen_type == "bacterial_reported"
    assert "MASINT reports agent: anthrax" in reported.masint_indicators
    assert reported.recommended_interdiction[0].startswith("Presumptive alarm")
    assert "Escalate to federal bio-threat response teams" in reported.recommended_interdiction
    quiet = fitted.detect_pathogen(mean, {"agent": "anthrax"})
    assert quiet.pathogen_type == "none_detected" and not quiet.recommended_interdiction


def test_couplings_are_the_negative_precision(fitted: PathogenDetector) -> None:
    precision = fitted.precision_
    assert precision is not None
    assert np.allclose(fitted.J_matrix, -(precision - np.diag(np.diag(precision))))
    assert np.allclose(fitted.J_matrix, fitted.J_matrix.T)


def test_features_are_baseline_deviations(fitted: PathogenDetector) -> None:
    """Fails on the released features, which overwrote the last spin with the energy."""
    mean, scale = _stats(fitted)
    snapshot = mean + 2.0 * scale
    features = fitted.extract_features(snapshot).numpy()
    assert features.shape == (_DIM,)
    assert np.allclose(features, 2.0, atol=1e-5)


def test_unfitted_detector_refuses() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        PathogenDetector().detect_pathogen(np.ones(_DIM))


@pytest.mark.parametrize(
    "bad", [np.ones(_DIM - 1), np.full(_DIM, np.nan)], ids=["wrong-length", "non-finite"]
)
def test_bad_snapshots_are_rejected(fitted: PathogenDetector, bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        fitted.detect_pathogen(bad)


def test_baseline_too_short_for_the_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="held-out snapshots"):
        PathogenDetector({"false_alarm_rate": 0.01}).fit(_sensor_data(0, 200))


def test_constant_baseline_is_rejected() -> None:
    with pytest.raises(ValueError, match="no variance"):
        PathogenDetector({"energy_threshold": 10.0}).fit(np.ones((50, _DIM)))


def test_fixed_threshold_skips_calibration() -> None:
    detector = PathogenDetector({"energy_threshold": 1e9}).fit(_sensor_data(0, 20))
    assert not detector.detect_pathogen(_sensor_data(1, 1)[0] + 5.0).threat_detected


def test_empty_input_is_no_threat(fitted: PathogenDetector) -> None:
    assert not fitted.detect_pathogen(np.array([])).threat_detected
    assert fitted.extract_features(np.array([])).shape == (_DIM,)
