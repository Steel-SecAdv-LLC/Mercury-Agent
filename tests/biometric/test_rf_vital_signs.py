# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioural tests for contactless respiration and heart-rate estimation.

The synthetic model below is a small, seeded version of the validation model
documented in ``omni_mercury_engine.biometric.rf_vital_signs``:

* Chest displacement is breathing (fundamental plus a second harmonic for
  inhale/exhale asymmetry; 4 to 12 mm peak-to-peak, De Groote et al. 1997)
  plus a heartbeat (three harmonics; 0.2 to 0.5 mm peak-to-peak, Ramachandran
  and Singh 1989), with optional sinusoidal rate variability.
* Wi-Fi CSI is static multipath plus one body-reflected path whose length
  changes by twice the displacement, so its phase moves by 4 pi d / lambda;
  per-subcarrier gain ripple, complex white noise at a receiver SNR, and
  per-packet timing/phase offsets shared by the antennas of one receiver.
* Radar phase is 4 pi d / lambda, wrapped.

Ground truth for a window is 60 times the mean instantaneous frequency.
Every expectation is either exact (computed independently here) or bounded
tightly around that ground truth.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import (
    given,
    settings,
    strategies as st,
)
from scipy import signal, stats

from omni_mercury_engine.biometric.rf_vital_signs import (
    MAD_TO_SIGMA,
    AbstainReason,
    ConsentRequiredError,
    VitalSignConfig,
    VitalSignEstimate,
    VitalSignTrack,
    antenna_phase_difference,
    band_power_ratio,
    cancel_respiration_harmonics,
    centre_wrapped_phase,
    count_cycle_rate,
    detect_motion,
    dominant_frequency,
    estimate_vital_signs,
    extract_waveform_excerpt,
    first_principal_component,
    hampel_filter,
    null_snr_threshold_db,
    rate_series,
    rates_within,
    resample_uniform,
    respiration_harmonics,
    respiration_series,
    sanitize_csi_phase,
    suppress_harmonics,
    track_vital_signs,
    windows_within,
)

C_LIGHT = 299_792_458.0
FS = 20.0
LAMBDA_24GHZ = C_LIGHT / 24e9
RESP_BAND = (0.1, 0.5)
HEART_BAND = (0.8, 2.0)


# ---------------------------------------------------------------------------
# Synthetic model
# ---------------------------------------------------------------------------


def chest(
    duration_s: float,
    resp_bpm: float,
    heart_bpm: float,
    *,
    fs: float = FS,
    resp_pp_m: float = 6e-3,
    heart_pp_m: float = 0.4e-3,
    rate_var: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return displacement (m) and instantaneous respiration/heart frequency (Hz)."""
    rng = np.random.default_rng(seed)
    n = round(duration_s * fs)
    t = np.arange(n) / fs
    f_r = resp_bpm / 60.0 * (1.0 + rate_var * np.sin(2 * np.pi * t / 40.0 + rng.uniform(0, 6.3)))
    f_h = heart_bpm / 60.0 * (1.0 + rate_var * np.sin(2 * np.pi * t / 50.0 + rng.uniform(0, 6.3)))
    phi_r = 2 * np.pi * np.cumsum(f_r) / fs + rng.uniform(0, 2 * np.pi)
    phi_h = 2 * np.pi * np.cumsum(f_h) / fs + rng.uniform(0, 2 * np.pi)
    breathing = 0.5 * resp_pp_m * (np.sin(phi_r) + 0.25 * np.sin(2 * phi_r + 0.6))
    heart = (
        0.5
        * heart_pp_m
        * (np.sin(phi_h) + 0.4 * np.sin(2 * phi_h + 1.0) + 0.15 * np.sin(3 * phi_h + 2.0))
    )
    return breathing + heart, f_r, f_h


def truth_bpm(freq_hz: np.ndarray, start_s: float, end_s: float, fs: float = FS) -> float:
    """Ground truth: 60 x mean instantaneous frequency over [start, end)."""
    return 60.0 * float(np.mean(freq_hz[round(start_s * fs) : round(end_s * fs)]))


def csi(
    displacement: np.ndarray,
    *,
    snr_db: float = 30.0,
    n_sub: int = 16,
    n_ant: int = 1,
    carrier_hz: float = 5.32e9,
    body_rel_db: float = -10.0,
    limb_m: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CSI amplitude and phase, shape (T, S, A), for a displacement series.

    ``limb_m`` (path-length change, m) adds a moving-limb path of twice the body
    path's amplitude wherever it is non-zero, as during gross body motion.
    """
    rng = np.random.default_rng(seed + 1000)
    n = displacement.shape[0]
    f_k = carrier_hz + np.linspace(-10e6, 10e6, n_sub)
    k_idx = np.arange(n_sub) - (n_sub - 1) / 2
    slope = np.asarray(rng.uniform(-0.3, 0.3, size=n))
    beta = np.asarray(rng.uniform(-np.pi, np.pi, size=n))
    amp = np.empty((n, n_sub, n_ant))
    pha = np.empty((n, n_sub, n_ant))
    for a in range(n_ant):
        tau = rng.uniform(5e-9, 60e-9, size=4)
        g = (rng.standard_normal(4) + 1j * rng.standard_normal(4)) * 0.35
        g[0] = np.exp(1j * rng.uniform(0, 2 * np.pi))
        h_static = (g[None, :] * np.exp(-2j * np.pi * f_k[:, None] * tau[None, :])).sum(axis=1)
        a_b = 10 ** (body_rel_db / 20) * np.sqrt(np.mean(np.abs(h_static) ** 2))
        path = rng.uniform(2.0, 4.0) + 2.0 * displacement
        h_body = a_b * np.exp(-2j * np.pi * f_k[None, :] * path[:, None] / C_LIGHT)
        if limb_m is not None:
            limb = 2 * a_b * np.exp(-2j * np.pi * f_k[None, :] * (3.0 + limb_m)[:, None] / C_LIGHT)
            h_body = h_body + np.where((limb_m != 0)[:, None], limb, 0.0)
        ripple = 10 ** (rng.normal(0, 1.0, size=n_sub) / 20)
        h = ripple[None, :] * (h_static[None, :] + h_body)
        sigma = np.sqrt(np.mean(np.abs(h) ** 2) / 10 ** (snr_db / 10) / 2)
        h = h + sigma * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
        amp[:, :, a] = np.abs(h)
        pha[:, :, a] = np.angle(h * np.exp(1j * (slope[:, None] * k_idx + beta[:, None])))
    return amp, pha


def radar_phase(displacement: np.ndarray, noise_m: float = 0.0, seed: int = 0) -> np.ndarray:
    """Wrapped 24 GHz CW radar phase for a displacement series."""
    rng = np.random.default_rng(seed + 2000)
    d = displacement + noise_m * rng.standard_normal(displacement.shape[0])
    return np.asarray(np.angle(np.exp(1j * 4 * np.pi * d / LAMBDA_24GHZ)), dtype=np.float64)


def burst(n: int, start_s: float, dur_s: float, pp_m: float = 0.08, fs: float = FS) -> np.ndarray:
    """Smooth large torso excursion (body motion) on [start, start + dur)."""
    out = np.zeros(n)
    i0, i1 = round(start_s * fs), round((start_s + dur_s) * fs)
    tt = np.arange(i1 - i0) / fs
    wave = np.sin(2 * np.pi * 0.7 * tt) + 0.6 * np.sin(2 * np.pi * 1.9 * tt + 1.0)
    out[i0:i1] = signal.windows.tukey(i1 - i0, 0.3) * wave / np.ptp(wave) * pp_m
    return out


def radar_estimate(d: np.ndarray, config: VitalSignConfig | None = None) -> VitalSignEstimate:
    return estimate_vital_signs(
        fs=FS, consent_confirmed=True, radar_displacement_m=d, config=config
    )


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------


class TestConsent:
    def test_estimate_refuses_without_consent(self) -> None:
        d, _, _ = chest(30, 15, 70)
        with pytest.raises(ConsentRequiredError, match="informed consent"):
            estimate_vital_signs(fs=FS, consent_confirmed=False, radar_displacement_m=d)

    @pytest.mark.parametrize("value", [1, "yes", None, np.bool_(True)])
    def test_only_literal_true_counts_as_consent(self, value: object) -> None:
        d, _, _ = chest(30, 15, 70)
        with pytest.raises(ConsentRequiredError):
            estimate_vital_signs(
                fs=FS, consent_confirmed=value, radar_displacement_m=d  # type: ignore[arg-type]
            )

    def test_track_refuses_without_consent(self) -> None:
        d, _, _ = chest(40, 15, 70)
        with pytest.raises(ConsentRequiredError):
            track_vital_signs(fs=FS, consent_confirmed=False, radar_displacement_m=d)

    def test_consent_is_checked_before_any_input(self) -> None:
        with pytest.raises(ConsentRequiredError):
            estimate_vital_signs(fs=-1.0, consent_confirmed=False)

    def test_consent_error_is_a_permission_error(self) -> None:
        assert issubclass(ConsentRequiredError, PermissionError)


# ---------------------------------------------------------------------------
# Hampel filter
# ---------------------------------------------------------------------------


class TestHampel:
    def test_spike_replaced_by_window_median(self) -> None:
        x = np.sin(2 * np.pi * 0.25 * np.arange(100) / FS)
        y = x.copy()
        y[50] += 100.0
        out, flags = hampel_filter(y, half_width=3, n_sigmas=3.0)
        assert np.flatnonzero(flags).tolist() == [50]
        assert out[50] == pytest.approx(np.median(y[47:54]), abs=0.0)
        np.testing.assert_array_equal(np.delete(out, 50), np.delete(y, 50))

    def test_end_spikes_detected_through_reflection(self) -> None:
        y = np.linspace(0.0, 1.0, 40)
        y[0] = 50.0
        y[-1] = -50.0
        out, flags = hampel_filter(y, half_width=3)
        assert flags[0] and flags[-1]
        assert np.count_nonzero(flags) == 2
        assert out[0] == pytest.approx(np.median([50.0, *y[1:4], *y[1:4]]))

    def test_ramp_is_untouched(self) -> None:
        y = np.linspace(-3.0, 7.0, 200)
        out, flags = hampel_filter(y, half_width=5)
        np.testing.assert_array_equal(out, y)
        assert not flags.any()

    def test_zero_threshold_is_running_median(self) -> None:
        rng = np.random.default_rng(3)
        y = rng.standard_normal(60)
        out, _ = hampel_filter(y, half_width=2, n_sigmas=0.0)
        expected = np.array([np.median(y[i - 2 : i + 3]) for i in range(2, 58)])
        np.testing.assert_allclose(out[2:58], expected)

    def test_threshold_uses_mad_consistency_constant(self) -> None:
        assert pytest.approx(1.0 / stats.norm.ppf(0.75), rel=1e-4) == MAD_TO_SIGMA
        # deviation just above / below n * 1.4826 * MAD of a window with MAD = 1
        base = np.array([0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0])
        for delta, expect in ((3 * MAD_TO_SIGMA + 0.01, True), (3 * MAD_TO_SIGMA - 0.01, False)):
            y = np.tile(base, 3)
            y[10] = delta
            _, flags = hampel_filter(y, half_width=3)
            assert bool(flags[10]) is expect

    def test_half_width_zero_is_identity_and_columns_are_independent(self) -> None:
        y = np.column_stack([np.linspace(0.0, 1.0, 30), np.linspace(5.0, -2.0, 30)])
        y[15, 1] += 40.0
        out, flags = hampel_filter(y, half_width=0)
        np.testing.assert_array_equal(out, y)
        assert not flags.any()
        out, flags = hampel_filter(y, half_width=3)
        assert np.argwhere(flags).tolist() == [[15, 1]]
        np.testing.assert_array_equal(out[:, 0], y[:, 0])
        # the window median counts the spike as its largest value: x[14]
        assert out[15, 1] == pytest.approx(5.0 - 7.0 * 14 / 29)

    @pytest.mark.parametrize(
        ("value", "kwargs"),
        [
            (np.zeros((3, 3, 3)), {}),
            (np.array([1.0, np.nan, 2.0]), {}),
            (np.zeros(5), {"half_width": -1}),
            (np.zeros(5), {"n_sigmas": -1.0}),
        ],
    )
    def test_rejects_bad_input(self, value: np.ndarray, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            hampel_filter(value, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Phase handling
# ---------------------------------------------------------------------------


class TestPhase:
    def test_per_packet_linear_offsets_are_removed_exactly(self) -> None:
        rng = np.random.default_rng(5)
        t_len, n_sub = 50, 12
        k = np.arange(n_sub)
        true = 0.3 * np.sin(np.linspace(0, 3, n_sub))[None, :] + 0.05 * rng.standard_normal(
            (t_len, n_sub)
        )
        offsets = rng.uniform(-0.4, 0.4, (t_len, 1)) * k[None, :] + rng.uniform(-3, 3, (t_len, 1))
        measured = np.angle(np.exp(1j * (true + offsets)))
        clean = sanitize_csi_phase(measured)
        kc = k - k.mean()
        fit = (true @ kc / (kc @ kc))[:, None] * kc[None, :] + true.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(clean, true - fit, atol=1e-9)

    def test_real_subcarrier_indices_define_the_line(self) -> None:
        idx = np.array([-28.0, -26, -24, -2, -1, 1, 2, 24, 26, 28])
        offsets = 0.05 * idx[None, :] * np.array([[1.0], [-2.0]]) + 1.0
        clean = sanitize_csi_phase(offsets, subcarrier_indices=idx)
        np.testing.assert_allclose(clean, 0.0, atol=1e-12)
        assert np.abs(sanitize_csi_phase(offsets)).max() > 0.1

    def test_nan_packet_stays_nan_and_does_not_leak(self) -> None:
        x = np.tile(np.linspace(0, 1, 6), (4, 1))
        x[2, 3] = np.nan
        out = sanitize_csi_phase(x)
        assert np.isnan(out[2]).all()
        assert np.isfinite(out[[0, 1, 3]]).all()

    @pytest.mark.parametrize(
        ("phase", "idx"),
        [
            (np.zeros((5, 2)), None),
            (np.zeros(5), None),
            (np.zeros((5, 4)), np.ones(4)),
            (np.zeros((5, 4)), np.arange(3)),
        ],
    )
    def test_sanitise_rejects_bad_input(self, phase: np.ndarray, idx: np.ndarray | None) -> None:
        with pytest.raises(ValueError):
            sanitize_csi_phase(phase, idx)

    def test_antenna_difference_cancels_shared_offsets(self) -> None:
        rng = np.random.default_rng(7)
        t_len, n_sub = 40, 5
        psi = 0.2 * rng.standard_normal((t_len, n_sub, 3))
        common = rng.uniform(-np.pi, np.pi, (t_len, n_sub, 1))
        out = antenna_phase_difference(np.angle(np.exp(1j * (psi + common))), reference_antenna=1)
        assert out.shape == (t_len, n_sub, 2)
        expected = centre_wrapped_phase(psi[:, :, [0, 2]] - psi[:, :, [1]])
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_centring_avoids_cycle_slips(self) -> None:
        t = np.arange(400) / FS
        series = np.pi - 0.05 + 0.3 * np.sin(2 * np.pi * 0.25 * t)
        wrapped = np.angle(np.exp(1j * series))
        assert np.abs(np.diff(wrapped)).max() > 6.0  # raw data jumps across +-pi
        out = centre_wrapped_phase(wrapped)
        assert np.abs(np.diff(out)).max() < 0.05
        circular_mean = np.angle(np.mean(np.exp(1j * series)))
        np.testing.assert_allclose(out, np.angle(np.exp(1j * (series - circular_mean))), atol=1e-12)

    @pytest.mark.parametrize(("shape", "ref"), [((5, 4, 1), 0), ((5, 4), 0), ((5, 4, 2), 2)])
    def test_antenna_difference_rejects_bad_input(self, shape: tuple[int, ...], ref: int) -> None:
        with pytest.raises(ValueError):
            antenna_phase_difference(np.zeros(shape), reference_antenna=ref)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


class TestResample:
    def test_linear_signal_is_interpolated_exactly(self) -> None:
        rng = np.random.default_rng(2)
        t = np.cumsum(rng.uniform(0.04, 0.06, 200))
        grid, v, missing = resample_uniform(t, 2.0 * t + 1.0, fs=FS, max_gap_s=0.25)
        assert grid[0] == t[0]
        np.testing.assert_allclose(np.diff(grid), 1.0 / FS)
        np.testing.assert_allclose(v, 2.0 * grid + 1.0, atol=1e-12)
        assert not missing.any()

    def test_long_gaps_and_lost_packets_are_marked_missing(self) -> None:
        t = np.concatenate([np.arange(0, 2, 0.05), np.arange(3, 5, 0.05)])
        v = np.ones((t.size, 2))
        v[60, 0] = np.nan  # lost packet at t = 4.0 s; neighbours 0.1 s apart (< max gap)
        grid, out, missing = resample_uniform(t, v, fs=FS, max_gap_s=0.25)
        gap = (grid > 1.95 + 1e-9) & (grid < 3.0 - 1e-9)
        np.testing.assert_array_equal(missing, gap)
        assert np.isnan(out[missing]).all()
        assert np.isfinite(out[~missing]).all()

    def test_gap_just_above_the_limit_is_missing(self) -> None:
        t = np.concatenate([np.arange(0, 2, 0.05), np.arange(2.35, 4, 0.05)])  # 0.4 s gap
        grid, _, missing = resample_uniform(t, np.ones(t.size), fs=FS, max_gap_s=0.25)
        np.testing.assert_allclose(grid[missing], [2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3])
        _, _, bridged = resample_uniform(t, np.ones(t.size), fs=FS, max_gap_s=0.45)
        assert not bridged.any()

    @pytest.mark.parametrize(
        ("t", "v"),
        [
            (np.array([0.0, 0.1, 0.1]), np.zeros(3)),
            (np.array([0.0, 0.1, 0.2]), np.zeros(4)),
            (np.array([0.0]), np.zeros(1)),
        ],
    )
    def test_rejects_bad_timestamps(self, t: np.ndarray, v: np.ndarray) -> None:
        with pytest.raises(ValueError):
            resample_uniform(t, v, fs=FS)


# ---------------------------------------------------------------------------
# Spectral primitives
# ---------------------------------------------------------------------------


def tone(freq: float, duration_s: float = 40.0, amp: float = 1.0, phase: float = 0.3) -> np.ndarray:
    t = np.arange(round(duration_s * FS)) / FS
    return np.asarray(amp * np.sin(2 * np.pi * freq * t + phase), dtype=np.float64)


class TestSpectral:
    def test_band_power_ratio_separates_in_and_out_of_band(self) -> None:
        x = np.column_stack([tone(0.25), tone(1.5), np.zeros(800)])
        ratio = band_power_ratio(x, FS, RESP_BAND)
        assert ratio[0] > 0.99
        assert ratio[1] < 0.01
        assert ratio[2] == 0.0

    @pytest.mark.parametrize("f0", [0.137, 0.2731, 0.4412])
    def test_peak_frequency_is_interpolated_between_bins(self, f0: float) -> None:
        peak = dominant_frequency(tone(f0), FS, RESP_BAND, signal_halfwidth_hz=0.05)
        assert peak.frequency_hz is not None
        assert abs(peak.frequency_hz - f0) < 1e-3
        assert peak.snr_db > 20.0
        coarse = dominant_frequency(
            tone(f0), FS, RESP_BAND, signal_halfwidth_hz=0.05, zero_pad_factor=1
        )
        assert coarse.frequency_hz is not None
        assert abs(coarse.frequency_hz - f0) < 0.25 / 40.0  # within a quarter of a DFT bin

    def test_halfwidth_never_below_hann_main_lobe(self) -> None:
        narrow = dominant_frequency(tone(0.3), FS, RESP_BAND, signal_halfwidth_hz=0.01)
        wide = dominant_frequency(tone(0.3), FS, RESP_BAND, signal_halfwidth_hz=0.08)
        assert narrow.halfwidth_hz == pytest.approx(2.0 / 40.0)
        assert wide.halfwidth_hz == pytest.approx(0.08)

    def test_edge_maximum_and_empty_band_return_no_peak(self) -> None:
        edge = dominant_frequency(tone(0.06), FS, RESP_BAND, signal_halfwidth_hz=0.05)
        assert edge.frequency_hz is None and edge.at_band_edge
        empty = dominant_frequency(np.zeros(800), FS, RESP_BAND, signal_halfwidth_hz=0.05)
        assert empty.frequency_hz is None and not empty.at_band_edge
        assert math.isnan(empty.snr_db)

    def test_second_source_reported_but_harmonic_is_not(self) -> None:
        two = dominant_frequency(
            tone(0.2) + tone(0.35, amp=0.9, phase=1.0), FS, RESP_BAND, signal_halfwidth_hz=0.05
        )
        assert two.secondary_hz == pytest.approx(0.35, abs=0.01)
        assert 0.7 < two.secondary_power_ratio < 0.95
        assert two.joint_snr_db > two.snr_db
        harmonic = dominant_frequency(
            tone(0.2) + tone(0.4, amp=0.5), FS, RESP_BAND, signal_halfwidth_hz=0.05
        )
        # the 0.4 Hz harmonic is never the secondary; only a weak sidelobe remains
        assert harmonic.secondary_hz is None or abs(harmonic.secondary_hz - 0.4) > 0.05
        assert harmonic.secondary_power_ratio < 0.1

    def test_null_threshold_matches_beta_model(self) -> None:
        n, hw = 600, 0.05
        duration = n / FS
        m, big = 2 * max(hw, 2 / duration) * duration, 0.4 * duration
        expected = stats.beta.isf(0.01 * m / big, 0.45 * m * 3, 0.45 * (big - m) * 3)
        expected_db = 10 * math.log10(expected / (1 - expected)) + 2.6
        got = null_snr_threshold_db(3, n, FS, RESP_BAND, hw)
        assert got == pytest.approx(expected_db, abs=1e-9)

    def test_null_threshold_falls_with_streams_and_duration(self) -> None:
        by_k = [null_snr_threshold_db(k, 600, FS, RESP_BAND, 0.05) for k in (1, 2, 5, 10)]
        by_t = [null_snr_threshold_db(1, n, FS, HEART_BAND, 0.1) for n in (300, 600, 1200)]
        assert by_k == sorted(by_k, reverse=True) and len(set(by_k)) == 4
        assert by_t == sorted(by_t, reverse=True) and len(set(by_t)) == 3
        assert null_snr_threshold_db(1, 200, FS, RESP_BAND, 0.05) == math.inf
        stricter = null_snr_threshold_db(1, 600, FS, RESP_BAND, 0.05, false_alarm=0.001)
        assert stricter > by_k[0]

    def test_null_threshold_false_alarm_on_white_noise(self) -> None:
        """Calibration record: noise alone passes in about 1 % of windows or less."""
        rng = np.random.default_rng(20251016)
        sos = signal.butter(4, HEART_BAND, btype="bandpass", fs=FS, output="sos")
        thr = null_snr_threshold_db(1, 600, FS, HEART_BAND, 0.1)
        passed = 0
        for _ in range(200):
            x = signal.sosfiltfilt(sos, rng.standard_normal(600))
            peak = dominant_frequency(x, FS, HEART_BAND, signal_halfwidth_hz=0.1)
            passed += int(peak.frequency_hz is not None and peak.snr_db >= thr)
        assert passed <= 4

    @pytest.mark.parametrize(
        "kwargs",
        [{"n_streams": 0}, {"n_samples": 0}, {"fs": 0.0}, {"false_alarm": 1.0}],
    )
    def test_null_threshold_rejects_bad_arguments(self, kwargs: dict[str, float]) -> None:
        args = {"n_streams": 1, "n_samples": 600, "fs": FS, "false_alarm": 0.01} | kwargs
        with pytest.raises(ValueError):
            null_snr_threshold_db(
                int(args["n_streams"]),
                int(args["n_samples"]),
                float(args["fs"]),
                RESP_BAND,
                0.05,
                false_alarm=float(args["false_alarm"]),
            )

    def test_first_principal_component_recovers_common_source(self) -> None:
        src = tone(0.25, 30.0)
        mix = np.column_stack([3.0 * src, -0.5 * src, 2.0 * src + 5.0, np.full(600, 7.0)])
        pc = first_principal_component(mix)
        # standardised: the three live columns load equally; the constant one is dropped
        np.testing.assert_allclose(pc, math.sqrt(3.0) * (src - src.mean()) / src.std(), atol=1e-9)
        flipped = first_principal_component(-mix)  # tie goes to column 0, now negated
        np.testing.assert_allclose(flipped, -pc, atol=1e-9)
        np.testing.assert_array_equal(first_principal_component(np.ones((10, 3))), 0.0)


class TestCycleCount:
    def test_counts_breaths_of_a_clean_waveform(self) -> None:
        assert count_cycle_rate(tone(0.25), FS, 0.5) == pytest.approx(15.0, abs=0.05)

    def test_noisy_waveform_within_half_a_breath(self) -> None:
        rng = np.random.default_rng(9)
        sos = signal.butter(4, RESP_BAND, btype="bandpass", fs=FS, output="sos")
        w = signal.sosfiltfilt(sos, tone(0.3) + 0.3 * rng.standard_normal(800))
        rate = count_cycle_rate(w, FS, 0.5)
        assert rate is not None and abs(rate - 18.0) < 0.5

    def test_double_humped_breath_counts_once(self) -> None:
        # each 5-s breath has two prominent humps 1.7 s apart; the 2-s minimum
        # spacing (30 breaths/min band edge) counts the breath once
        t = np.arange(800) / FS
        w = np.sin(2 * np.pi * 0.2 * t) + 0.6 * np.sin(2 * np.pi * 0.6 * t + 0.5)
        assert count_cycle_rate(w, FS, 0.5) == pytest.approx(12.0, abs=0.05)
        assert count_cycle_rate(w, FS, 1.0) == pytest.approx(36.1, abs=0.3)  # humps counted

    def test_minimum_spacing_suppresses_fast_wiggles(self) -> None:
        w = tone(0.2) + 0.3 * tone(1.7)
        assert count_cycle_rate(w, FS, 0.5) == pytest.approx(12.0, abs=0.3)

    @pytest.mark.parametrize(
        "w", [np.zeros(100), tone(0.25, 6.0), np.array([0.0, np.nan, 1.0] * 50), np.zeros((9, 2))]
    )
    def test_returns_none_without_three_peaks(self, w: np.ndarray) -> None:
        assert count_cycle_rate(w, FS, 0.5) is None


class TestHarmonics:
    def test_harmonics_inside_band_and_margin(self) -> None:
        assert respiration_harmonics(0.3, HEART_BAND, 5) == pytest.approx((0.9, 1.2, 1.5))
        assert respiration_harmonics(0.3, HEART_BAND, 5, margin_hz=0.25) == pytest.approx(
            (0.6, 0.9, 1.2, 1.5)
        )
        assert respiration_harmonics(0.3, HEART_BAND, 3) == pytest.approx((0.9,))
        assert respiration_harmonics(0.9, HEART_BAND, 5) == pytest.approx((1.8,))  # not k = 1
        assert respiration_harmonics(0.0, HEART_BAND, 5) == ()
        assert respiration_harmonics(math.nan, HEART_BAND, 5) == ()

    def test_notch_removes_target_and_keeps_neighbour(self) -> None:
        x = tone(1.2) + tone(1.5)
        out = suppress_harmonics(x, FS, (1.2, 25.0), 0.1)  # 25 Hz is above Nyquist: skipped
        mid = slice(100, 700)
        gain_target = np.abs(np.fft.rfft(out[mid] * np.hanning(600)))[36]
        gain_ref = np.abs(np.fft.rfft(x[mid] * np.hanning(600)))[36]
        assert 20 * math.log10(gain_target / gain_ref) < -30.0
        assert np.abs(out - tone(1.2) - x + tone(1.2)).max() > 0  # something changed
        residual = out - tone(1.5)
        assert np.sqrt(np.mean(residual[mid] ** 2)) < 0.05

    def test_phase_locked_regression_tracks_rate_variability(self) -> None:
        """Harmonics of a drifting breath survive a fixed notch but not the regression."""
        n = 600
        t = np.arange(n) / FS
        f_r = 0.3 * (1.0 + 0.06 * np.sin(2 * np.pi * t / 25.0))
        phi = 2 * np.pi * np.cumsum(f_r) / FS
        resp = np.sin(phi)
        harm = 1.0 * np.cos(3 * phi + 0.4) + 0.7 * np.sin(4 * phi)
        heart = 0.3 * np.sin(2 * np.pi * 1.33 * t)
        sos = signal.butter(4, HEART_BAND, btype="bandpass", fs=FS, output="sos")
        xh = signal.sosfiltfilt(sos, harm + heart)
        heart_f = signal.sosfiltfilt(sos, heart)
        harm_f = signal.sosfiltfilt(sos, harm)
        reg = cancel_respiration_harmonics(xh, resp, FS, 0.3, (3, 4), HEART_BAND)
        notch = suppress_harmonics(xh, FS, (0.9, 1.2), 0.1)
        mid = slice(60, 540)

        def residue(out: np.ndarray) -> float:
            return float(np.mean((out - heart_f)[mid] ** 2) / np.mean(harm_f[mid] ** 2))

        # measured: regression leaves 3.7 % of the harmonic power, the notch 15 %
        assert residue(reg) < 0.045
        assert residue(notch) > 3 * residue(reg)
        # a heart tone 0.06 Hz from the drifting 4th harmonic loses about 10 %
        kept = np.dot(reg[mid], heart_f[mid]) / np.dot(heart_f[mid], heart_f[mid])
        assert 0.85 < kept < 0.95

    def test_regression_input_checks(self) -> None:
        xh = np.zeros((50, 2))
        with pytest.raises(ValueError):
            cancel_respiration_harmonics(xh, np.zeros(49), FS, 0.3, (3,), HEART_BAND)
        with pytest.raises(ValueError):
            cancel_respiration_harmonics(xh, np.zeros(50), FS, 0.0, (3,), HEART_BAND)
        np.testing.assert_array_equal(
            cancel_respiration_harmonics(xh + 1.0, np.zeros(50), FS, 0.3, (), HEART_BAND), 1.0
        )


# ---------------------------------------------------------------------------
# Motion detection
# ---------------------------------------------------------------------------


class TestMotion:
    def test_steady_breathing_is_not_motion(self) -> None:
        d, _, _ = chest(60, 12, 70, resp_pp_m=12e-3)
        assert not detect_motion(d, FS).any()

    def test_burst_flagged_with_guard_and_nothing_else(self) -> None:
        d, _, _ = chest(60, 12, 70)
        d = d + burst(d.size, 30.0, 3.0)
        mask = detect_motion(d, FS, frame_s=1.0, guard_s=1.0)
        flagged = np.flatnonzero(mask) / FS
        assert flagged.min() >= 28.0 and flagged.max() < 35.0
        assert mask[round(30.5 * FS) : round(32.5 * FS)].all()
        assert not mask[: round(27.9 * FS)].any() and not mask[round(35.1 * FS) :].any()

    def test_coherent_motion_detected_in_noisy_csi(self) -> None:
        d, _, _ = chest(60, 12, 70)
        torso = burst(d.size, 20.0, 3.0, pp_m=0.08)
        amp, _ = csi(d + torso, snr_db=15.0, limb_m=3.0 * burst(d.size, 20.0, 3.0, pp_m=0.08))
        mask = detect_motion(amp[:, :, 0], FS)
        assert mask[round(20.5 * FS) : round(22.5 * FS)].all()
        assert not mask[: round(18.0 * FS)].any() and not mask[round(25.0 * FS) :].any()
        # torso motion alone barely changes CSI amplitude when breathing already
        # sweeps the body path's phase by about a radian (a documented limit)
        weak, _ = csi(d + torso, snr_db=15.0)
        assert not detect_motion(weak[:, :, 0], FS)[round(20.5 * FS) : round(22.5 * FS)].all()

    def test_absolute_excursion_limit(self) -> None:
        d, _, _ = chest(60, 12, 70, resp_pp_m=40e-3)  # every frame is large but uniform
        assert not detect_motion(d, FS).any()
        assert detect_motion(d, FS, max_excursion=0.025).mean() > 0.3

    def test_fast_small_vibration_caught_by_the_difference_statistic(self) -> None:
        d, _, _ = chest(60, 12, 70)
        vib = np.zeros(d.size)
        vib[600:620] = 1e-3 * np.sin(2 * np.pi * 4.0 * np.arange(20) / FS)
        frames = signal.detrend(d + vib)[:1200].reshape(60, 20)
        std_ratio = frames.std(axis=1)[30] / np.median(frames.std(axis=1))
        assert std_ratio < 2.0  # invisible to the variance statistic alone
        mask = detect_motion(d + vib, FS, guard_s=0.0)
        assert np.flatnonzero(mask).tolist() == list(range(600, 620))

    def test_flat_signal_with_a_step(self) -> None:
        x = np.zeros(400)
        x[200:210] = 1.0
        mask = detect_motion(x, FS, guard_s=0.0)
        assert mask[200:210].all() and mask.sum() == 20  # the one 1-s frame holding the step

    def test_short_input_and_nan(self) -> None:
        assert not detect_motion(np.ones(3), FS).any()
        with pytest.raises(ValueError):
            detect_motion(np.array([0.0, 1.0, np.nan, 2.0, 3.0]), FS)


# ---------------------------------------------------------------------------
# End-to-end estimation
# ---------------------------------------------------------------------------


class TestEstimateRadar:
    def test_clean_displacement_recovers_both_rates(self) -> None:
        d, f_r, f_h = chest(40, 15.0, 72.0)
        est = radar_estimate(d)
        assert est.respiration.rate_bpm == pytest.approx(truth_bpm(f_r, 0, 40), abs=0.1)
        assert est.heart.rate_bpm == pytest.approx(truth_bpm(f_h, 0, 40), abs=0.5)
        assert est.respiration.count_rate_bpm == pytest.approx(15.0, abs=0.6)
        assert est.respiration.snr_db > est.respiration.snr_threshold_db
        assert est.waveform_units == "m"
        assert est.selected_streams == (0,)
        assert est.suppressed_harmonics_hz == pytest.approx((1.0, 1.25), abs=1e-3)
        assert est.segment_start_s == 0.0 and est.segment_end_s == 40.0

    def test_wrapped_phase_input_matches_displacement_input(self) -> None:
        d, _, _ = chest(40, 15.0, 72.0)
        a = radar_estimate(d)
        b = estimate_vital_signs(
            fs=FS, consent_confirmed=True, radar_phase_rad=radar_phase(d), wavelength_m=LAMBDA_24GHZ
        )
        assert b.respiration.rate_bpm == pytest.approx(a.respiration.rate_bpm, abs=1e-6)
        assert b.heart.rate_bpm == pytest.approx(a.heart.rate_bpm, abs=1e-6)
        # d = lambda * phase / (4 pi): the waveform is in metres either way
        np.testing.assert_allclose(b.respiration_waveform, a.respiration_waveform, atol=1e-12)

    def test_noisy_variable_rates_within_tolerance(self) -> None:
        d, f_r, f_h = chest(40, 13.0, 72.0, rate_var=0.04, seed=3)
        rng = np.random.default_rng(11)
        est = radar_estimate(d + 50e-6 * rng.standard_normal(d.size))
        assert est.respiration.rate_bpm == pytest.approx(truth_bpm(f_r, 0, 40), abs=0.5)
        assert est.heart.rate_bpm == pytest.approx(truth_bpm(f_h, 0, 40), abs=1.5)

    def test_count_tolerance_is_the_decision_boundary(self) -> None:
        d, _, _ = chest(40, 15.0, 72.0, rate_var=0.04, seed=3)
        est = radar_estimate(d)
        assert est.respiration.rate_bpm is not None and est.respiration.count_rate_bpm is not None
        gap = abs(est.respiration.rate_bpm - est.respiration.count_rate_bpm)
        assert 0.0 < gap < 0.1
        tight = radar_estimate(d, config=VitalSignConfig(count_tolerance_bpm=gap / 2))
        assert tight.respiration.reason is AbstainReason.COUNT_DISAGREEMENT
        assert tight.respiration.count_rate_bpm == est.respiration.count_rate_bpm
        loose = radar_estimate(d, config=VitalSignConfig(count_tolerance_bpm=gap * 1.5))
        assert loose.respiration.rate_bpm == est.respiration.rate_bpm

    def test_decimation_from_100_hz_agrees(self) -> None:
        d20, _, _ = chest(40, 15.0, 72.0)
        d100, _, _ = chest(40, 15.0, 72.0, fs=100.0)
        a = radar_estimate(d20)
        b = estimate_vital_signs(fs=100.0, consent_confirmed=True, radar_displacement_m=d100)
        assert b.fs_analysis == pytest.approx(20.0)
        assert b.respiration.rate_bpm == pytest.approx(a.respiration.rate_bpm, abs=0.05)
        assert b.heart.rate_bpm == pytest.approx(a.heart.rate_bpm, abs=0.2)

    def test_irregular_timestamps_are_resampled(self) -> None:
        rng = np.random.default_rng(12)
        t = np.cumsum(rng.uniform(0.008, 0.012, 4000))
        f_r, f_h = 0.25, 1.2
        d = 3e-3 * np.sin(2 * np.pi * f_r * t) + 0.2e-3 * np.sin(2 * np.pi * f_h * t)
        est = estimate_vital_signs(
            fs=100.0, consent_confirmed=True, radar_displacement_m=d, timestamps_s=t
        )
        assert est.respiration.rate_bpm == pytest.approx(15.0, abs=0.1)
        assert est.heart.rate_bpm == pytest.approx(72.0, abs=0.3)

    def test_breath_hold_keeps_heart_rate(self) -> None:
        rng = np.random.default_rng(13)
        _, _, f_h = chest(40, 15.0, 80.0)
        d, _, _ = chest(40, 15.0, 80.0, resp_pp_m=0.0)
        est = radar_estimate(d + 20e-6 * rng.standard_normal(d.size))
        assert est.respiration.abstained
        assert est.respiration.reason in (AbstainReason.LOW_SNR, AbstainReason.PEAK_AT_BAND_EDGE)
        assert est.heart.rate_bpm == pytest.approx(truth_bpm(f_h, 0, 40), abs=0.5)

    def test_harmonic_suppression_modes(self) -> None:
        n = 800
        t = np.arange(n) / FS
        phi = 2 * np.pi * 0.3 * t
        breath = 3e-3 * (np.sin(phi) + 0.06 * np.sin(3 * phi + 0.5) + 0.04 * np.sin(4 * phi))
        d = breath + 0.1e-3 * np.sin(2 * np.pi * 1.37 * t)
        none = radar_estimate(d, config=VitalSignConfig(harmonic_suppression="none"))
        reg = radar_estimate(d)
        notch = radar_estimate(d, config=VitalSignConfig(harmonic_suppression="notch"))
        # unsuppressed, the third respiration harmonic wins and is recognised as such
        assert none.heart.peak_hz == pytest.approx(0.9, abs=0.005)
        assert none.heart.reason is AbstainReason.HARMONIC_AMBIGUITY
        assert "harmonic 3 x" in none.heart.detail
        assert none.suppressed_harmonics_hz == ()
        assert reg.heart.rate_bpm == pytest.approx(82.2, abs=0.3)
        assert notch.heart.rate_bpm == pytest.approx(82.2, abs=0.3)
        assert reg.suppressed_harmonics_hz == pytest.approx((0.9, 1.2, 1.5), abs=1e-3)

    @pytest.mark.parametrize(("heart_hz", "accepted"), [(1.13, True), (1.06, False)])
    def test_heart_on_an_unsuppressed_harmonic_is_withheld(
        self, heart_hz: float, accepted: bool
    ) -> None:
        """Slow breathing (9/min) with a strong 7th harmonic at 1.05 Hz (above order 5)."""
        t = np.arange(800) / FS
        phi = 2 * np.pi * 0.15 * t
        d = (
            3e-3 * np.sin(phi)
            + 0.06e-3 * np.sin(7 * phi)
            + 0.25e-3 * np.sin(2 * np.pi * heart_hz * t)
        )
        est = radar_estimate(d)
        if accepted:
            assert est.heart.rate_bpm == pytest.approx(60 * heart_hz, abs=0.3)
        else:
            assert est.heart.reason is AbstainReason.HARMONIC_AMBIGUITY
            assert "harmonic 7 x" in est.heart.detail
        lenient = radar_estimate(d, config=VitalSignConfig(harmonic_coincidence_fraction=0.0))
        assert lenient.heart.rate_bpm is not None

    def test_second_harmonic_coincidence(self) -> None:
        t = np.arange(800) / FS
        phi = 2 * np.pi * 0.45 * t
        d = 3e-3 * (np.sin(phi) + 0.2 * np.sin(2 * phi + 0.4)) + 0.1e-3 * np.sin(
            2 * np.pi * 1.4 * t
        )
        est = radar_estimate(d, config=VitalSignConfig(harmonic_suppression="none"))
        assert est.heart.reason is AbstainReason.HARMONIC_AMBIGUITY
        assert "harmonic 2 x" in est.heart.detail

    def test_amplitude_scale_and_offset_do_not_change_rates(self) -> None:
        d, _, _ = chest(40, 17.0, 63.0, seed=4)
        a = radar_estimate(d)
        b = radar_estimate(0.5 * d + 0.3)
        assert b.respiration.rate_bpm == pytest.approx(a.respiration.rate_bpm, abs=1e-9)
        assert b.heart.rate_bpm == pytest.approx(a.heart.rate_bpm, abs=1e-9)

    def test_deterministic(self) -> None:
        d, _, _ = chest(40, 17.0, 63.0, seed=5)
        a, b = radar_estimate(d), radar_estimate(d)
        assert a.respiration == b.respiration and a.heart == b.heart
        np.testing.assert_array_equal(a.respiration_waveform, b.respiration_waveform)


class TestEstimateCSI:
    def test_amplitude_recovers_respiration(self) -> None:
        d, f_r, _ = chest(40, 14.0, 75.0, seed=1)
        amp, _ = csi(d, snr_db=25.0, seed=1)
        est = estimate_vital_signs(fs=FS, consent_confirmed=True, csi_amplitude=amp[:, :, 0])
        assert est.respiration.rate_bpm == pytest.approx(truth_bpm(f_r, 0, 40), abs=0.3)
        assert est.waveform_units == "a.u."
        assert len(est.selected_streams) == 10
        assert est.respiration_waveform.shape == (800,)

    def test_two_antenna_phase_difference_recovers_respiration(self) -> None:
        d, f_r, _ = chest(40, 19.0, 75.0, seed=2)
        _, pha = csi(d, snr_db=20.0, n_ant=2, seed=2)
        est = estimate_vital_signs(fs=FS, consent_confirmed=True, csi_phase=pha)
        assert est.respiration.rate_bpm == pytest.approx(truth_bpm(f_r, 0, 40), abs=0.3)

    def test_single_antenna_phase_is_sanitised(self) -> None:
        d, f_r, _ = chest(40, 11.0, 75.0, seed=6)
        _, pha = csi(d, snr_db=30.0, seed=6)
        est = estimate_vital_signs(fs=FS, consent_confirmed=True, csi_phase=pha[:, :, 0])
        assert est.respiration.rate_bpm == pytest.approx(truth_bpm(f_r, 0, 40), abs=0.3)

    def test_high_snr_heart_rate(self) -> None:
        d, _, f_h = chest(40, 14.0, 75.0, heart_pp_m=0.5e-3, seed=8)
        amp, pha = csi(d, snr_db=40.0, n_ant=2, seed=8)
        est = estimate_vital_signs(fs=FS, consent_confirmed=True, csi_amplitude=amp, csi_phase=pha)
        assert est.heart.rate_bpm == pytest.approx(truth_bpm(f_h, 0, 40), abs=1.0)

    def test_amplitude_scaling_is_irrelevant(self) -> None:
        d, _, _ = chest(40, 14.0, 75.0, seed=1)
        amp, _ = csi(d, snr_db=25.0, seed=1)
        a = estimate_vital_signs(fs=FS, consent_confirmed=True, csi_amplitude=amp[:, :, 0])
        b = estimate_vital_signs(
            fs=FS, consent_confirmed=True, csi_amplitude=3.0 * amp[:, :, 0] + 5.0
        )
        assert b.respiration.rate_bpm == pytest.approx(a.respiration.rate_bpm, abs=1e-9)
        assert a.selected_streams == b.selected_streams

    def test_isolated_lost_packets_are_bridged_and_dead_subcarrier_dropped(self) -> None:
        d, f_r, _ = chest(40, 14.0, 75.0, seed=1)
        amp, _ = csi(d, snr_db=25.0, seed=1)
        a = amp[:, :, 0].copy()
        rng = np.random.default_rng(21)
        lost = np.sort(rng.choice(np.arange(0, 800, 7), 30, replace=False))  # isolated rows
        a[lost] = np.nan
        a[:, 4] = np.nan
        est = estimate_vital_signs(fs=FS, consent_confirmed=True, csi_amplitude=a)
        assert 4 not in est.selected_streams
        assert est.missing_fraction == pytest.approx(30 / 800)
        assert (est.segment_start_s, est.segment_end_s) == (0.0, 40.0)
        assert est.respiration.rate_bpm == pytest.approx(truth_bpm(f_r, 0, 40), abs=0.3)

    def test_long_gap_splits_the_clean_run(self) -> None:
        d, f_r, _ = chest(45, 14.0, 75.0, seed=1)
        d[round(15 * FS) : round(17 * FS)] = np.nan  # 2 s > max_gap_s
        d[round(30 * FS)] = np.nan  # one lost sample: bridged
        est = radar_estimate(d)
        assert (est.segment_start_s, est.segment_end_s) == (17.0, 45.0)
        assert est.respiration.rate_bpm == pytest.approx(truth_bpm(f_r, 17, 45), abs=0.2)
        short_gap = radar_estimate(d, config=VitalSignConfig(max_gap_s=2.5))
        assert (short_gap.segment_start_s, short_gap.segment_end_s) == (0.0, 45.0)


class TestAbstention:
    def test_two_breathing_sources_abstain_both(self) -> None:
        d1, _, _ = chest(40, 12.0, 70.0, seed=1)
        d2, _, _ = chest(40, 21.0, 90.0, seed=2)
        est = radar_estimate(d1 + 0.9 * d2)
        assert est.respiration.reason is AbstainReason.MULTIPLE_SOURCES
        assert est.heart.reason is AbstainReason.MULTIPLE_SOURCES
        assert est.respiration.rate_bpm is None and est.heart.rate_bpm is None

    def test_noise_alone_is_never_reported(self) -> None:
        rng = np.random.default_rng(31)
        accepted = 0
        for _ in range(15):
            r = radar_estimate(1e-4 * rng.standard_normal(600))
            c = estimate_vital_signs(
                fs=FS, consent_confirmed=True, csi_amplitude=10 + rng.standard_normal((600, 16))
            )
            accepted += sum(
                not e.abstained for e in (r.respiration, r.heart, c.respiration, c.heart)
            )
        assert accepted == 0

    def test_flat_signal(self) -> None:
        est = radar_estimate(np.full(800, 0.01))
        assert est.respiration.reason is AbstainReason.FLAT_SIGNAL
        assert est.heart.reason is AbstainReason.FLAT_SIGNAL

    def test_too_short(self) -> None:
        d, _, _ = chest(15, 15.0, 70.0)
        est = radar_estimate(d)
        assert est.respiration.reason is AbstainReason.INSUFFICIENT_DATA
        assert "15.0 s < 20.0 s" in est.respiration.detail

    def test_all_missing(self) -> None:
        est = radar_estimate(np.full(800, np.nan))
        assert est.respiration.reason is AbstainReason.INSUFFICIENT_DATA
        assert est.missing_fraction == 1.0

    def test_motion_in_the_middle_leaves_no_clean_run(self) -> None:
        d, _, _ = chest(40, 15.0, 70.0)
        est = radar_estimate(d + burst(d.size, 18.0, 3.0))
        assert est.respiration.reason is AbstainReason.MOTION
        assert 0.1 < est.motion_fraction < 0.3

    def test_motion_at_the_end_is_excluded(self) -> None:
        d, f_r, _ = chest(60, 15.0, 70.0)
        est = radar_estimate(d + burst(d.size, 50.0, 3.0))
        assert est.segment_start_s == 0.0
        assert 47.0 <= est.segment_end_s <= 50.0
        assert est.respiration.rate_bpm == pytest.approx(
            truth_bpm(f_r, 0, est.segment_end_s), abs=0.15
        )

    def test_mostly_moving_subject(self) -> None:
        d, _, _ = chest(40, 15.0, 70.0)
        moving = d + burst(d.size, 5.0, 6.0) + burst(d.size, 20.0, 6.0) + burst(d.size, 30.0, 6.0)
        est = radar_estimate(moving, config=VitalSignConfig(max_motion_fraction=0.3))
        assert est.respiration.reason is AbstainReason.MOTION
        assert "subject not stationary" in est.respiration.detail

    def test_slow_drift_does_not_create_a_rate(self) -> None:
        t = np.arange(800) / FS
        est = radar_estimate(1e-3 * t + 1e-7 * np.random.default_rng(3).standard_normal(800))
        assert est.respiration.abstained and est.heart.abstained


class TestInputValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"radar_displacement_m": np.zeros(10), "csi_amplitude": np.zeros((10, 3))},
            {"radar_displacement_m": np.zeros(10), "radar_phase_rad": np.zeros(10)},
            {"radar_displacement_m": np.zeros((10, 2))},
            {"radar_phase_rad": np.zeros(10)},
            {"radar_phase_rad": np.zeros(10), "wavelength_m": -1.0},
            {"radar_phase_rad": np.zeros((10, 2)), "wavelength_m": 0.01},
            {"csi_amplitude": np.zeros(10)},
            {"csi_phase": np.zeros((10, 3, 2, 2))},
            {"csi_amplitude": np.zeros((10, 3)), "csi_phase": np.zeros((11, 3))},
            {"radar_displacement_m": np.zeros(10), "timestamps_s": np.arange(9.0)},
        ],
    )
    def test_malformed_inputs_raise(self, kwargs: dict[str, np.ndarray]) -> None:
        with pytest.raises(ValueError):
            estimate_vital_signs(fs=FS, consent_confirmed=True, **kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("fs", [0.0, -5.0, math.nan, math.inf, True, "20"])
    def test_bad_sampling_rate(self, fs: object) -> None:
        with pytest.raises(ValueError):
            estimate_vital_signs(
                fs=fs, consent_confirmed=True, radar_displacement_m=np.zeros(400)  # type: ignore[arg-type]
            )

    def test_heart_band_above_nyquist(self) -> None:
        with pytest.raises(ValueError, match="fs/2"):
            estimate_vital_signs(fs=3.0, consent_confirmed=True, radar_displacement_m=np.zeros(90))

    def test_complex_input_rejected(self) -> None:
        with pytest.raises(TypeError, match=r"np\.abs"):
            estimate_vital_signs(
                fs=FS, consent_confirmed=True, csi_amplitude=np.ones((400, 3), dtype=complex)
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"respiration_band_hz": (0.5, 0.1)},
            {"heart_band_hz": (0.0, 2.0)},
            {"analysis_fs": 0.0},
            {"n_streams": 0},
            {"hampel_half_width": -1},
            {"max_harmonic_order": 0},
            {"count_prominence_fraction": -0.1},
            {"min_clean_s": 0.0},
            {"false_alarm": 1.0},
            {"null_colour_margin_db": math.inf},
            {"respiration_min_snr_db": math.nan},
            {"motion_guard_s": -1.0},
            {"max_motion_fraction": 1.5},
            {"max_radar_excursion_m": 0.0},
            {"selection": "pca"},
            {"harmonic_suppression": "adaptive"},
            {"harmonic_poly_degree": -1},
            {"harmonic_coincidence_fraction": -0.5},
        ],
    )
    def test_config_validation(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            VitalSignConfig(**kwargs)  # type: ignore[arg-type]

    def test_fixed_thresholds_override_the_null_model(self) -> None:
        d, _, _ = chest(40, 15.0, 72.0)
        strict = radar_estimate(
            d, config=VitalSignConfig(respiration_min_snr_db=80.0, heart_min_snr_db=80.0)
        )
        assert strict.respiration.reason is AbstainReason.LOW_SNR
        assert strict.respiration.snr_threshold_db == 80.0
        assert strict.heart.reason is AbstainReason.LOW_SNR

    def test_variance_selection(self) -> None:
        d, f_r, _ = chest(40, 14.0, 75.0, seed=1)
        amp, _ = csi(d, snr_db=25.0, seed=1)
        est = estimate_vital_signs(
            fs=FS,
            consent_confirmed=True,
            csi_amplitude=amp[:, :, 0],
            config=VitalSignConfig(selection="variance", n_streams=4),
        )
        variances = np.var(signal.detrend(amp[:, :, 0], axis=0), axis=0)
        assert set(est.selected_streams) == set(np.argsort(-variances)[:4].tolist())
        assert est.respiration.rate_bpm == pytest.approx(truth_bpm(f_r, 0, 40), abs=0.3)


# ---------------------------------------------------------------------------
# Tracking and stimulus-locked access
# ---------------------------------------------------------------------------


def stepped_chest(fs: float = FS) -> tuple[np.ndarray, float]:
    """Breathing at 12/min for 60 s, then 18/min for 60 s; heart 76/min."""
    t = np.arange(round(120 * fs)) / fs
    f_r = np.where(t < 60.0, 0.2, 0.3)
    phi = 2 * np.pi * np.cumsum(f_r) / fs
    d = 3e-3 * np.sin(phi) + 0.2e-3 * np.sin(2 * np.pi * (76 / 60) * t)
    return d, fs


class TestTrack:
    def test_window_clock_and_rate_step(self) -> None:
        d, fs = stepped_chest()
        tr = track_vital_signs(
            fs=fs, consent_confirmed=True, radar_displacement_m=d, window_s=30, step_s=5, t0_s=100.0
        )
        np.testing.assert_allclose(tr.window_start_s, 100.0 + 5.0 * np.arange(19))
        np.testing.assert_allclose(tr.window_end_s - tr.window_start_s, 30.0)
        np.testing.assert_allclose(tr.window_center_s, tr.window_start_s + 15.0)
        np.testing.assert_allclose(tr.sample_times_s, 100.0 + np.arange(2400) / fs)
        before = tr.window_end_s <= 160.0
        after = tr.window_start_s >= 160.0
        np.testing.assert_allclose(tr.respiration_bpm[before], 12.0, atol=0.1)
        np.testing.assert_allclose(tr.respiration_bpm[after], 18.0, atol=0.1)
        np.testing.assert_allclose(tr.heart_bpm[before | after], 76.0, atol=0.2)
        straddle = ~(before | after)
        ok = np.isfinite(tr.heart_bpm)
        np.testing.assert_allclose(tr.heart_bpm[ok], 76.0, atol=0.5)
        # windows straddling the breathing change may be withheld, never wrong
        assert {tr.heart_reasons[i] for i in np.flatnonzero(straddle & ~ok)} <= {
            AbstainReason.HARMONIC_AMBIGUITY
        }
        assert tr.fs_analysis == fs and tr.waveform_units == "m"

    def test_waveform_equals_independent_band_pass(self) -> None:
        d, fs = stepped_chest()
        tr = track_vital_signs(fs=fs, consent_confirmed=True, radar_displacement_m=d)
        sos = signal.butter(4, RESP_BAND, btype="bandpass", fs=fs, output="sos")
        expected = signal.sosfiltfilt(sos, signal.detrend(d))
        np.testing.assert_allclose(tr.respiration_waveform, expected, atol=1e-12)

    def test_motion_masks_waveform_and_windows(self) -> None:
        d, fs = stepped_chest()
        d = d + burst(d.size, 40.0, 3.0)
        tr = track_vital_signs(fs=fs, consent_confirmed=True, radar_displacement_m=d, step_s=5)
        assert np.isnan(tr.respiration_waveform[tr.motion_mask]).all()
        assert np.isfinite(tr.respiration_waveform[~tr.motion_mask]).all()
        assert tr.motion_mask[round(41 * fs)]
        overlapping = (tr.window_start_s < 44.0) & (tr.window_end_s > 39.0)
        assert np.all(tr.motion_fraction[overlapping] > 0)
        assert np.all(tr.motion_fraction[~overlapping] == 0)

    def test_timestamps_set_the_clock(self) -> None:
        d, fs = stepped_chest()
        t = 1000.0 + np.arange(d.size) / fs
        tr = track_vital_signs(
            fs=fs, consent_confirmed=True, radar_displacement_m=d, timestamps_s=t, step_s=10
        )
        assert tr.window_start_s[0] == pytest.approx(1000.0)
        assert tr.sample_times_s[0] == pytest.approx(1000.0)
        with pytest.raises(ValueError, match="t0_s"):
            track_vital_signs(
                fs=fs, consent_confirmed=True, radar_displacement_m=d, timestamps_s=t, t0_s=0.0
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"window_s": 10.0},
            {"window_s": 200.0},
            {"step_s": 0.0},
            {"window_s": math.nan},
        ],
    )
    def test_bad_window_settings(self, kwargs: dict[str, float]) -> None:
        d, fs = stepped_chest()
        with pytest.raises(ValueError):
            track_vital_signs(
                fs=fs,
                consent_confirmed=True,
                radar_displacement_m=d,
                window_s=kwargs.get("window_s", 30.0),
                step_s=kwargs.get("step_s", 1.0),
            )

    def test_short_windows_for_stimulus_locked_heart_rate(self) -> None:
        d, fs = stepped_chest()
        tr = track_vital_signs(
            fs=fs,
            consent_confirmed=True,
            radar_displacement_m=d,
            window_s=10.0,
            step_s=1.0,
            config=VitalSignConfig(min_clean_s=10.0),
        )
        assert tr.window_s == 10.0 and tr.window_start_s.size == 111
        ok = np.isfinite(tr.heart_bpm)
        # At 10 s the resolution (0.1 Hz) is coarser than the gap between the heart
        # and the nearest respiration harmonic, so most windows are withheld as
        # ambiguous; whatever is reported stays within 2.5/min (measured max 1.9).
        assert 0 < ok.sum() < 30
        np.testing.assert_allclose(tr.heart_bpm[ok], 76.0, atol=2.5)
        assert AbstainReason.HARMONIC_AMBIGUITY in set(tr.heart_reasons)
        # 10-s respiration windows are too short for the null test: always abstain
        assert set(tr.respiration_reasons) <= {
            AbstainReason.LOW_SNR,
            AbstainReason.PEAK_AT_BAND_EDGE,
        }


class TestStimulusLockedAccess:
    @pytest.fixture(scope="class")
    def track(self) -> VitalSignTrack:
        d, fs = stepped_chest()
        d = d + burst(d.size, 80.0, 3.0)
        return track_vital_signs(
            fs=fs, consent_confirmed=True, radar_displacement_m=d, step_s=5.0, t0_s=10.0
        )

    def test_excerpt_returns_exact_samples(self, track: VitalSignTrack) -> None:
        ex = extract_waveform_excerpt(track, 30.0, 45.0)
        assert ex.reason is None and ex.samples is not None and ex.times_s is not None
        i0 = round((30.0 - 10.0) * 20)
        np.testing.assert_array_equal(ex.samples, track.respiration_waveform[i0 : i0 + 300])
        np.testing.assert_allclose(ex.times_s, 30.0 + np.arange(300) / 20.0)
        assert ex.fs == 20.0 and ex.units == "m"

    def test_excerpt_abstains_on_motion_and_outside(self, track: VitalSignTrack) -> None:
        moving = extract_waveform_excerpt(track, 88.0, 95.0)
        assert moving.samples is None and moving.reason is AbstainReason.MOTION
        outside = extract_waveform_excerpt(track, 5.0, 15.0)
        assert outside.reason is AbstainReason.INSUFFICIENT_DATA
        late = extract_waveform_excerpt(track, 125.0, 131.0)
        assert late.reason is AbstainReason.INSUFFICIENT_DATA
        whole = extract_waveform_excerpt(track, 10.0, 130.0)
        assert whole.reason is AbstainReason.MOTION
        with pytest.raises(ValueError):
            extract_waveform_excerpt(track, 20.0, 20.0)

    def test_windows_and_rates_within(self, track: VitalSignTrack) -> None:
        idx = windows_within(track, 20.0, 60.0)
        assert idx.tolist() == [2, 3, 4]  # windows [20, 50), [25, 55) and [30, 60)
        centres, rates = rates_within(track, 20.0, 60.0, "respiration")
        np.testing.assert_allclose(centres, [35.0, 40.0, 45.0])
        np.testing.assert_allclose(rates, 12.0, atol=0.1)
        assert windows_within(track, 20.0, 40.0).size == 0
        with pytest.raises(ValueError):
            rates_within(track, 0.0, 1.0, "skin")  # type: ignore[arg-type]

    def test_series_helpers_are_finite_and_increasing(self, track: VitalSignTrack) -> None:
        t_h, hr = rate_series(track, "heart")
        assert np.all(np.isfinite(hr)) and np.all(np.diff(t_h) > 0)
        assert t_h.size == np.count_nonzero(np.isfinite(track.heart_bpm))
        t_r, w = respiration_series(track)
        assert np.all(np.isfinite(w)) and np.all(np.diff(t_r) > 0)
        assert t_r.size == np.count_nonzero(~track.motion_mask)
        with pytest.raises(ValueError):
            rate_series(track, "eeg")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@settings(derandomize=True, max_examples=15, deadline=None)
@given(resp=st.floats(8.0, 28.0), heart=st.floats(55.0, 110.0), seed=st.integers(0, 50))
def test_property_rates_recovered_across_bands(resp: float, heart: float, seed: int) -> None:
    d, f_r, f_h = chest(40, resp, heart, seed=seed)
    est = radar_estimate(d)
    assert est.respiration.rate_bpm == pytest.approx(truth_bpm(f_r, 0, 40), abs=0.15)
    if est.heart.rate_bpm is not None:
        assert est.heart.rate_bpm == pytest.approx(truth_bpm(f_h, 0, 40), abs=1.0)


@settings(derandomize=True, max_examples=10, deadline=None)
@given(resp=st.floats(9.0, 25.0), heart=st.floats(55.0, 110.0))
def test_property_time_reversal_invariance(resp: float, heart: float) -> None:
    d, _, _ = chest(40, resp, heart, seed=1)
    a, b = radar_estimate(d), radar_estimate(d[::-1].copy())
    assert b.respiration.rate_bpm == pytest.approx(a.respiration.rate_bpm, abs=0.05)
    if a.heart.rate_bpm is not None and b.heart.rate_bpm is not None:
        assert b.heart.rate_bpm == pytest.approx(a.heart.rate_bpm, abs=0.3)


@settings(derandomize=True, max_examples=10, deadline=None)
@given(shift=st.floats(-1.0, 1.0), scale=st.floats(0.2, 3.0))
def test_property_affine_invariance_of_csi(shift: float, scale: float) -> None:
    d, _, _ = chest(30, 16.0, 70.0, seed=2)
    amp, _ = csi(d, snr_db=25.0, seed=2)
    base = estimate_vital_signs(fs=FS, consent_confirmed=True, csi_amplitude=amp[:, :, 0])
    moved = estimate_vital_signs(
        fs=FS, consent_confirmed=True, csi_amplitude=scale * amp[:, :, 0] + shift
    )
    assert moved.respiration.rate_bpm == pytest.approx(base.respiration.rate_bpm, abs=1e-6)
    assert moved.selected_streams == base.selected_streams
