# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reference-sensor event detection and agreement statistics."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.biometric import vital_reference as vr

FS_ECG = 500.0
FS_RESP = 50.0


def _synthetic_ecg(
    beat_times_s: np.ndarray, *, fs: float = FS_ECG, seconds: float | None = None, seed: int = 0
) -> np.ndarray:
    """ECG-like trace: narrow biphasic QRS at each beat, T-wave bump, drift, noise."""
    seconds = float(beat_times_s[-1] + 1.0) if seconds is None else seconds
    t = np.arange(0.0, seconds, 1.0 / fs)
    x = np.zeros_like(t)
    for beat in beat_times_s:
        x += 1.0 * np.exp(-((t - beat) ** 2) / (2 * 0.008**2))
        x -= 0.3 * np.exp(-((t - beat - 0.03) ** 2) / (2 * 0.010**2))
        x += 0.25 * np.exp(-((t - beat - 0.30) ** 2) / (2 * 0.05**2))  # T wave
    rng = np.random.default_rng(seed)
    drift = 0.3 * np.sin(2 * np.pi * 0.3 * t)
    return x + drift + 0.02 * rng.normal(size=t.size)


def _beats(rate_bpm: float, seconds: float, *, jitter_s: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    times = np.arange(0.5, seconds, 60.0 / rate_bpm)
    return times + jitter_s * rng.normal(size=times.size)


class TestECGRPeaks:
    def test_recovers_beats_despite_t_waves_and_drift(self) -> None:
        beats = _beats(72.0, 60.0, jitter_s=0.01)
        detected = vr.detect_ecg_r_peaks(_synthetic_ecg(beats), FS_ECG) / FS_ECG
        matched = sum(np.abs(detected - b).min() < 0.05 for b in beats)
        assert matched == beats.size
        assert detected.size == beats.size  # no T-wave double counting

    def test_polarity_does_not_matter(self) -> None:
        beats = _beats(60.0, 30.0)
        ecg = _synthetic_ecg(beats)
        up = vr.detect_ecg_r_peaks(ecg, FS_ECG)
        down = vr.detect_ecg_r_peaks(-ecg, FS_ECG)
        assert np.abs(up.astype(float) - down.astype(float)).max() <= 0.1 * FS_ECG

    def test_under_sampled_lead_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="fs >= 100"):
            vr.detect_ecg_r_peaks(np.zeros(1000), 50.0)

    def test_sparse_dropouts_are_bridged(self) -> None:
        beats = _beats(70.0, 30.0)
        ecg = _synthetic_ecg(beats)
        ecg[1000:1010] = np.nan
        detected = vr.detect_ecg_r_peaks(ecg, FS_ECG)
        assert detected.size >= beats.size - 2

    def test_heavily_missing_lead_is_rejected(self) -> None:
        ecg = _synthetic_ecg(_beats(70.0, 30.0))
        ecg[: ecg.size // 2] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            vr.detect_ecg_r_peaks(ecg, FS_ECG)


class TestPulseAndBreaths:
    def test_ppg_pulse_rate(self) -> None:
        t = np.arange(0.0, 60.0, 1.0 / 100.0)
        ppg = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t - 1.0)
        peaks = vr.detect_pulse_peaks(ppg, 100.0)
        rate = 60.0 / np.median(np.diff(peaks) / 100.0)
        assert rate == pytest.approx(72.0, abs=1.0)

    def test_breath_rate_from_impedance(self) -> None:
        t = np.arange(0.0, 120.0, 1.0 / FS_RESP)
        rng = np.random.default_rng(1)
        z0 = np.sin(2 * np.pi * (15.0 / 60.0) * t) + 0.05 * rng.normal(size=t.size)
        peaks = vr.detect_breaths(z0, FS_RESP)
        rate = 60.0 / np.median(np.diff(peaks) / FS_RESP)
        assert rate == pytest.approx(15.0, abs=0.5)

    def test_under_sampled_inputs_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="fs >= 20"):
            vr.detect_pulse_peaks(np.zeros(200), 10.0)
        with pytest.raises(ValueError, match="fs >= 4"):
            vr.detect_breaths(np.zeros(20), 2.0)


class TestIntervals:
    def test_ectopic_and_missed_beats_are_dropped(self) -> None:
        times = list(np.arange(0.0, 60.0, 0.8))
        times.remove(times[20])  # a missed detection: one 1.6 s interval
        times.append(8.4)  # an extra detection: two 0.4 s intervals
        _, lengths = vr.valid_intervals(np.sort(times), vr.HEART_INTERVALS)
        assert np.allclose(lengths, 0.8, atol=0.01)
        assert lengths.size == len(times) - 4  # the 1.6 s and both 0.4 s intervals dropped

    def test_windowed_rates_need_coverage(self) -> None:
        beats = np.arange(0.0, 14.5, 0.75)  # beats stop 15 s into a 30 s window
        rates = vr.windowed_rates(beats, [0.0], [30.0], vr.HEART_INTERVALS, min_coverage=0.7)
        assert np.isnan(rates[0])
        rates = vr.windowed_rates(
            beats, [0.0], [15.0], vr.HEART_INTERVALS, min_coverage=0.7
        )
        assert rates[0] == pytest.approx(80.0, abs=0.1)

    def test_rate_is_the_mean_interval_rate(self) -> None:
        beats = np.arange(0.0, 30.0, 0.6)
        rates = vr.windowed_rates(beats, [0.0], [30.0], vr.HEART_INTERVALS, min_coverage=0.7)
        assert rates[0] == pytest.approx(100.0, abs=0.1)


class TestAgreement:
    def test_bland_altman_fields(self) -> None:
        reference = np.array([70.0, 72.0, 71.0, np.nan, 73.0])
        estimate = np.array([71.0, 71.0, np.nan, 70.0, 74.0])
        stats = vr.agreement(estimate, reference, tolerance=2.0)
        assert stats["reference_windows"] == 4
        assert stats["reported"] == 3
        assert stats["reported_without_reference"] == 1
        assert stats["coverage"] == pytest.approx(0.75)
        assert stats["mae"] == pytest.approx(1.0)
        assert stats["bias"] == pytest.approx(1.0 / 3.0)
        assert stats["within_tolerance"] == 1.0
        assert stats["loa_low"] < stats["bias"] < stats["loa_high"]

    def test_bootstrap_resamples_subjects(self) -> None:
        rows = [("a", 70.0 + e, 70.0) for e in (0.1, -0.2, 0.3)] + [
            ("b", 80.0 + e, 80.0) for e in (2.0, 2.5, 3.0)
        ]
        pooled = vr.bootstrap_by_subject(rows, tolerance=2.0, resamples=200, seed=1)
        assert pooled["subjects"] == 2
        low, high = pooled["mae_ci95"]
        # Resampling whole subjects spans all-a (tiny MAE) to all-b (large MAE).
        assert low < 0.5 and high > 1.5
        assert pooled["coverage"] == 1.0

    def test_empty_input_degrades_to_nan(self) -> None:
        pooled = vr.bootstrap_by_subject([], tolerance=1.0, resamples=10)
        assert pooled["subjects"] == 0
        assert pooled["mae_ci95"] is None
