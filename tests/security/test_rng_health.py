# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the advisory hardware-RNG health monitor (security.rng_health).

The monitor applies the same REG deviation statistics (and fault channels)
as the ``reg_deviation_gcp`` training pipeline to a single raw RNG stream.
Power context for the assertions below (documented in the module): a 0->1
bit-flip fault with probability q drifts the Stouffer statistic as
``14.14 * q * sqrt(N)`` over N 200-bit trials, so q=0.02 over 4000 trials
sits ~18 sigma out -- detection is certain at the default z=4 threshold,
while a healthy stream trips it with probability ~2e-4.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from omni_mercury_engine.security.rng_health import (
    MIN_TRIALS,
    TRIAL_BYTES,
    RngHealthMonitor,
    RngHealthVerdict,
    bit_bias_channel,
    bytes_to_trial_sums,
)

N_TRIALS = 4000
STREAM_BYTES = N_TRIALS * TRIAL_BYTES  # 100 KB


def _seeded_stream(seed: int, n_bytes: int = STREAM_BYTES) -> bytes:
    """Deterministic stand-in stream for fault-channel tests."""
    return np.random.default_rng(seed).integers(0, 256, n_bytes, dtype=np.uint8).tobytes()


class TestTrialPacking:
    """Raw bytes pack into 200-bit Binomial(200, 0.5)-null trial sums."""

    def test_known_extremes(self) -> None:
        sums = bytes_to_trial_sums(b"\xff" * (TRIAL_BYTES * MIN_TRIALS))
        assert np.all(sums == 200.0)
        sums = bytes_to_trial_sums(b"\x00" * (TRIAL_BYTES * MIN_TRIALS))
        assert np.all(sums == 0.0)

    def test_trailing_partial_trial_discarded(self) -> None:
        raw = b"\x0f" * (TRIAL_BYTES * MIN_TRIALS + 7)
        assert len(bytes_to_trial_sums(raw)) == MIN_TRIALS

    def test_too_short_refuses(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            bytes_to_trial_sums(b"\x00" * (TRIAL_BYTES * (MIN_TRIALS - 1)))


class TestBitBiasChannel:
    """The reference fault channel is seeded, bounded, and monotone."""

    def test_deterministic_under_seed(self) -> None:
        raw = _seeded_stream(0)
        assert bit_bias_channel(raw, 0.02, seed=5) == bit_bias_channel(raw, 0.02, seed=5)
        assert bit_bias_channel(raw, 0.02, seed=5) != bit_bias_channel(raw, 0.02, seed=6)

    def test_q_extremes(self) -> None:
        raw = _seeded_stream(1)
        assert bit_bias_channel(raw, 0.0, seed=0) == raw
        assert bit_bias_channel(raw, 1.0, seed=0) == b"\xff" * len(raw)

    def test_only_flips_zeros_to_ones(self) -> None:
        raw = _seeded_stream(2)
        out = np.frombuffer(bit_bias_channel(raw, 0.05, seed=3), dtype=np.uint8)
        src = np.frombuffer(raw, dtype=np.uint8)
        assert np.all((out & src) == src)  # every 1-bit survives


class TestMonitorVerdicts:
    """Closed-form statistics render the documented typed verdicts."""

    def test_os_urandom_is_healthy(self) -> None:
        """A real OS entropy stream must pass (false-alarm rate ~2e-4)."""
        report = RngHealthMonitor().assess(os.urandom(STREAM_BYTES))
        assert isinstance(report.verdict, RngHealthVerdict)
        assert report.verdict is RngHealthVerdict.HEALTHY
        assert report.n_trials == N_TRIALS
        assert abs(report.stouffer_z) < 4.0

    def test_bias_fault_is_flagged_at_documented_power(self) -> None:
        """q=0.02 over 4000 trials sits ~18 sigma out: detection is certain."""
        raw = bit_bias_channel(_seeded_stream(10), 0.02, seed=11)
        report = RngHealthMonitor().assess(raw)
        assert report.verdict is RngHealthVerdict.BIAS_SUSPECT
        assert report.stouffer_z > 4.0

    def test_small_bias_fault_still_flagged(self) -> None:
        """Even q=0.005 drifts ~4.5 sigma over 4000 trials."""
        raw = bit_bias_channel(_seeded_stream(12), 0.005, seed=13)
        report = RngHealthMonitor().assess(raw)
        assert report.verdict is RngHealthVerdict.BIAS_SUSPECT

    def test_variance_collapse_is_correlation_suspect(self) -> None:
        """A stream stuck at exactly 100 ones/trial has zero variance."""
        raw = b"\x0f" * STREAM_BYTES  # 4 bits/byte -> every trial sums to 100
        report = RngHealthMonitor().assess(raw)
        assert report.verdict is RngHealthVerdict.CORRELATION_SUSPECT
        assert abs(report.stouffer_z) < 1e-9  # mean is exactly on the null
        assert report.variance_z < -4.0

    def test_serial_duplication_is_correlation_suspect(self) -> None:
        """Repeating each trial (buffer reuse) shows up as lag-1 correlation."""
        base = np.frombuffer(_seeded_stream(14, STREAM_BYTES // 2), dtype=np.uint8)
        trials = base.reshape(-1, TRIAL_BYTES)
        raw = np.repeat(trials, 2, axis=0).tobytes()
        report = RngHealthMonitor(z_threshold=6.0).assess(raw)
        assert report.verdict is RngHealthVerdict.CORRELATION_SUSPECT
        assert report.lag1_autocorr_z > 6.0

    def test_threshold_validation(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            RngHealthMonitor(z_threshold=0.0)


class TestDetectorInjection:
    """The optional learned path is advisory and honors quarantine."""

    def test_untrained_detector_reports_no_score(self) -> None:
        """Quarantined (untrained) weights must not surface as a score."""
        torch = pytest.importorskip("torch")
        del torch
        from omni_mercury_engine.models.parapsychology import ParapsychologyDetector

        monitor = RngHealthMonitor(detector=ParapsychologyDetector(load_shipped_weights=False))
        report = monitor.assess(os.urandom(STREAM_BYTES))
        assert report.detector_score is None
        assert report.verdict is RngHealthVerdict.HEALTHY

    def test_trained_detector_score_is_reported(self) -> None:
        """A detector with loaded weights contributes an advisory score."""
        torch = pytest.importorskip("torch")
        del torch
        from omni_mercury_engine.models.parapsychology import ParapsychologyDetector

        det = ParapsychologyDetector()
        assert det.field_analyzer is not None
        det.load_neural_weights(det.field_analyzer.state_dict())
        monitor = RngHealthMonitor(detector=det)
        report = monitor.assess(os.urandom(STREAM_BYTES))
        assert report.detector_score is not None
        assert 0.0 <= report.detector_score <= 1.0
