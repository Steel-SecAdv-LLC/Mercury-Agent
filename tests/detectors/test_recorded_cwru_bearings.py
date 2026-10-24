# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vibration detectors on recorded bearing data (CWRU Bearing Data Center), measured and pinned.

``tests/fixtures/cwru`` holds 12 kHz drive-end accelerometer recordings: a
healthy bearing at 0 and 1 hp, and 0.007-inch inner-race, ball and outer-race
faults at 0 hp.  Signals are cut into 2048-sample windows.  A detector is fitted
on the first 40 healthy windows of each load (80 signals) and scores one window
at a time: the later 20 healthy windows of each load (40) and 20 windows of each
fault.  Pinned per detector: the AUROC of fault against healthy windows, the
share of fault windows it flags, and the share of healthy windows it flags.

The numbers are the measured behaviour on these recordings, not published
results.  Seeds are fixed, runs are repeatable.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.detectors.physics_integration import PhysicsIntegratedDetector
from omni_mercury_engine.detectors.spectral_vibration import SpectralVibrationDetector
from tests.recorded_data import auroc, cwru

WINDOW = 2048
FITTED = 40
FAULTS = ("inner_race", "ball", "outer_race")
AUROC_TOLERANCE = 0.05
RATE_TOLERANCE = 0.1


def _windows(name: str, first: int, last: int) -> np.ndarray:
    signal = cwru(name)
    return np.stack([signal[i * WINDOW : (i + 1) * WINDOW] for i in range(first, last)])


def _reference() -> np.ndarray:
    return np.concatenate([_windows("normal_0hp", 0, FITTED), _windows("normal_1hp", 0, FITTED)])


def _healthy() -> np.ndarray:
    return np.concatenate([_windows("normal_0hp", FITTED, 60), _windows("normal_1hp", FITTED, 60)])


def _per_window(detector: object, signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    results = [detector.detect(signal.reshape(1, -1)) for signal in signals]  # type: ignore[attr-defined]
    return (
        np.array([float(r["anomaly_score"]) for r in results]),
        np.array([bool(r["is_anomaly"]) for r in results]),
    )


def _measure(detector: object) -> dict[str, tuple[float, float]]:
    """Per fault: (AUROC against healthy windows, share of fault windows flagged);
    ``healthy``: (share of healthy windows flagged, 0)."""
    healthy_scores, healthy_flags = _per_window(detector, _healthy())
    out = {"healthy": (float(healthy_flags.mean()), 0.0)}
    for fault in FAULTS:
        scores, flags = _per_window(detector, _windows(fault, 0, 20))
        out[fault] = (auroc(scores, healthy_scores), float(flags.mean()))
    return out


def _spectral(reference: np.ndarray) -> SpectralVibrationDetector:
    torch.manual_seed(0)
    np.random.seed(0)
    return SpectralVibrationDetector({"sample_rate": 12000.0}).fit(reference)


def _physics(reference: np.ndarray) -> PhysicsIntegratedDetector:
    torch.manual_seed(0)
    np.random.seed(0)
    config = {"spectral_config": {"sample_rate": 12000.0}}
    return PhysicsIntegratedDetector(config).fit(reference)


def test_spectral_vibration_separates_every_fault_from_healthy_windows() -> None:
    measured = _measure(_spectral(_reference()))
    for fault in FAULTS:
        assert measured[fault][0] == pytest.approx(1.0, abs=AUROC_TOLERANCE), fault
        assert measured[fault][1] == pytest.approx(1.0, abs=RATE_TOLERANCE), fault
    assert measured["healthy"][0] == pytest.approx(0.025, abs=RATE_TOLERANCE)


def test_spectral_vibration_fitted_on_one_load_flags_every_window_of_another() -> None:
    """A reference without the operating condition is a different normal: pinned as measured."""
    detector = _spectral(_windows("normal_0hp", 0, 60))
    _, flags = _per_window(detector, _windows("normal_1hp", FITTED, 60))
    assert flags.mean() == pytest.approx(1.0, abs=RATE_TOLERANCE)


def test_physics_integrated_flags_faults_and_not_every_healthy_window() -> None:
    """Its fused score of a healthy window is 0.89-0.99, so the 0.5 default flagged them all."""
    detector = _physics(_reference())
    measured = _measure(detector)
    assert measured["healthy"][0] == pytest.approx(0.125, abs=0.1)
    for fault in FAULTS:
        assert measured[fault][0] == pytest.approx(0.975, abs=AUROC_TOLERANCE), fault
        assert measured[fault][1] == pytest.approx(1.0, abs=RATE_TOLERANCE), fault
    assert 0.5 < detector.threshold < 1.0


def test_physics_integrated_keeps_a_threshold_it_was_given() -> None:
    torch.manual_seed(0)
    detector = PhysicsIntegratedDetector({"threshold": 0.5}).fit(_reference())
    assert detector.threshold == 0.5


def test_physics_integrated_keeps_the_default_when_there_are_too_few_signals() -> None:
    torch.manual_seed(0)
    detector = PhysicsIntegratedDetector().fit(_reference()[:40])
    assert detector.threshold == 0.5
