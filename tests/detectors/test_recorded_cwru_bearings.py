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

from typing import Any, Protocol

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.detectors.physics_integration import PhysicsIntegratedDetector
from omni_mercury_engine.detectors.spectral_domain_frequency import SpectralDomainFrequency
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


class _WindowScorer(Protocol):
    """What these measurements need of a detector: one verdict per window."""

    def detect(self, data: np.ndarray) -> dict[str, Any]:
        """Score one ``(1, WINDOW)`` signal."""


def _per_window(detector: _WindowScorer, signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    results = [detector.detect(signal.reshape(1, -1)) for signal in signals]
    return (
        np.array([float(r["anomaly_score"]) for r in results]),
        np.array([bool(r["is_anomaly"]) for r in results]),
    )


def _measure(detector: _WindowScorer) -> dict[str, tuple[float, float]]:
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
    """Its fused score of a healthy window is 0.89-0.99, so the 0.5 default flagged them all.

    Re-measured after the Oracle's band z-score was put back into power units:
    healthy windows flagged 0.125 -> 0.075, fault AUROC 0.975 -> 1.000 on all
    three faults, calibrated threshold 0.983 -> 0.846.
    """
    detector = _physics(_reference())
    measured = _measure(detector)
    # Tolerances tight enough to tell the new measurement from the old one:
    # 40 healthy windows give a 0.025 granularity, so abs=0.03 spans 0.045 to
    # 0.105 and excludes the previous 0.125; abs=0.02 on the AUROC excludes
    # the previous 0.975.  The protocol is seeded and the fixtures are
    # recorded, so both values repeat exactly.
    assert measured["healthy"][0] == pytest.approx(0.075, abs=0.03)
    for fault in FAULTS:
        assert measured[fault][0] == pytest.approx(1.000, abs=0.02), fault
        assert measured[fault][1] == pytest.approx(1.0, abs=RATE_TOLERANCE), fault
    assert 0.5 < detector.threshold < 1.0


#: ``frequency_oracle``: (AUROC against healthy windows, mean score) per fault,
#: and ``healthy``: the mean score.  Before the band z-score was corrected the
#: Oracle returned exactly 0.21 for every one of these windows.
MEASURED_ORACLE_AUROC = 1.000
MEASURED_ORACLE_HEALTHY_MEAN = 0.056


def test_the_oracle_alone_separates_every_fault_from_healthy_windows() -> None:
    """The Oracle's own score was a constant, so it carried nothing into the fusion.

    ``_score_band`` z-scored a dimensionless power *ratio* against absolute
    band-power statistics.  On these recordings the band powers are 1e-6 to
    1e-4, so the z-score was 6.4e3 to 6.6e6 whatever the window held, ``|z|/3``
    clipped to 1.0, and every resolvable band scored exactly 0.6 for an
    aggregate of exactly 0.21 -- healthy and faulty alike.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    oracle = SpectralDomainFrequency({"sample_rate": 12000.0}).fit(_reference())

    healthy_scores, _ = _per_window(oracle, _healthy())
    assert healthy_scores.std() > 0.0, "a constant score ranks nothing"
    assert healthy_scores.mean() == pytest.approx(MEASURED_ORACLE_HEALTHY_MEAN, abs=0.02)

    for fault in FAULTS:
        scores, _ = _per_window(oracle, _windows(fault, 0, 20))
        assert auroc(scores, healthy_scores) == pytest.approx(
            MEASURED_ORACLE_AUROC, abs=AUROC_TOLERANCE
        ), fault


def test_the_oracle_alone_never_reaches_its_own_default_threshold() -> None:
    """Measured and pinned: it ranks these faults perfectly and decides nothing.

    The aggregate is a band-weighted mean of ``0.6 * z_anomaly + 0.4 * cp``,
    and most of the weight sits on bands this 12 kHz sample rate cannot
    resolve, so a single window cannot reach the 0.5 BaseDetector default.
    ``PhysicsIntegratedDetector`` consumes the Oracle's score and influence
    vector, not its verdict, so the fusion is unaffected; a caller using the
    Oracle standalone needs its own threshold.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    oracle = SpectralDomainFrequency({"sample_rate": 12000.0}).fit(_reference())

    for signals in (_healthy(), *(_windows(f, 0, 20) for f in FAULTS)):
        scores, flags = _per_window(oracle, signals)
        assert scores.max() < oracle.threshold
        assert not flags.any()


def test_physics_integrated_keeps_a_threshold_it_was_given() -> None:
    torch.manual_seed(0)
    detector = PhysicsIntegratedDetector({"threshold": 0.5}).fit(_reference())
    assert detector.threshold == 0.5


def test_physics_integrated_keeps_the_default_when_there_are_too_few_signals() -> None:
    torch.manual_seed(0)
    detector = PhysicsIntegratedDetector().fit(_reference()[:40])
    assert detector.threshold == 0.5


#: Mean squared bicoherence per class, and its AUROC against healthy windows.
#: Every fault class sits *above* healthy, which is the direction
#: ``_compute_influence_multiplier`` reads. The 1:1 phase-locking prototype
#: this replaced could not manage that: outer race 0.892, inner race 0.176 --
#: the opposite direction -- and ball 0.521, chance.
MEASURED_PHASE_COUPLING: dict[str, tuple[float, float]] = {
    "healthy": (0.0711, 0.0),
    "ball": (0.0788, 0.900),
    "inner_race": (0.1258, 1.000),
    "outer_race": (0.2010, 1.000),
}


def test_phase_coupling_separates_every_fault_class_from_healthy() -> None:
    """The direction evidence the withdrawn coherence term never had.

    The Oracle's inter-band term returned exactly 1.0 for every input and was
    withdrawn, its docstring recording that a replacement needed an n:m
    cross-frequency coupling measure *and* evidence of which direction is
    anomalous. This is that evidence, on the recorded corpus.
    """
    oracle = SpectralDomainFrequency()
    healthy = np.array([oracle._compute_phase_coherence(w) for w in _healthy()])
    expected_healthy, _ = MEASURED_PHASE_COUPLING["healthy"]

    assert healthy.mean() == pytest.approx(expected_healthy, abs=0.02)

    for fault in FAULTS:
        scores = np.array([oracle._compute_phase_coherence(w) for w in _windows(fault, 0, 20)])
        expected_mean, expected_auroc = MEASURED_PHASE_COUPLING[fault]
        assert scores.mean() == pytest.approx(expected_mean, abs=0.02), fault
        # Direction: the fault class scores higher, not merely differently.
        assert scores.mean() > healthy.mean(), fault
        assert auroc(scores, healthy) == pytest.approx(expected_auroc, abs=0.08), fault
