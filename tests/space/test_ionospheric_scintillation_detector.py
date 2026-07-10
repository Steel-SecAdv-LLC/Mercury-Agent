# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the ionospheric scintillation / GNSS-degradation detector.

S4 anchors are hand-computed directly from the Yeh & Liu (1982) definition;
sigma-phi tests verify the GISTM detrend convention removes slow trends but
preserves in-band fluctuation power. Test signals are synthetic *inputs*
constructed so their exact index values are known analytically — never
presented as recorded scintillation data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# The dev venv's editable install may point at a sibling worktree that
# predates ``ionospheric_scintillation_detector``; ``unused-ignore`` keeps
# a correctly installed tree (CI) clean.
from omni_mercury_engine.space.ionospheric_scintillation_detector import (  # type: ignore[import-not-found,unused-ignore]
    IonosphericScintillationDetector,
    classify_scintillation,
    compute_s4,
    compute_sigma_phi,
)

RNG = np.random.default_rng(20240510)


# ---------------------------------------------------------------------------
# S4 (Yeh & Liu 1982)
# ---------------------------------------------------------------------------


class TestS4:
    def test_constant_intensity_gives_zero(self) -> None:
        assert compute_s4(np.full(100, 5.0)) == pytest.approx(0.0, abs=1e-12)

    def test_hand_computed_four_sample_case(self) -> None:
        """I = [1, 1, 1, 3]: <I> = 1.5, <I^2> = 3, var = 0.75.

        S4^2 = 0.75 / 2.25 = 1/3 -> S4 = 0.57735.
        """
        s4 = compute_s4(np.array([1.0, 1.0, 1.0, 3.0]))
        assert s4 == pytest.approx(math.sqrt(1.0 / 3.0), abs=1e-12)

    def test_two_level_signal_hand_computed(self) -> None:
        """Equal-probability intensities a=0.5, b=1.5: <I>=1, var=0.25, S4=0.5."""
        intensity = np.array([0.5, 1.5] * 50)
        assert compute_s4(intensity) == pytest.approx(0.5, abs=1e-12)

    def test_scale_invariance(self) -> None:
        """S4 is normalized by <I>^2, so scaling intensity leaves it fixed."""
        intensity = np.abs(RNG.normal(10.0, 2.0, size=500))
        assert compute_s4(intensity * 7.3) == pytest.approx(compute_s4(intensity), rel=1e-12)

    @pytest.mark.parametrize(
        "bad",
        [
            np.array([]),
            np.array([1.0]),
            np.array([1.0, -0.5]),
            np.array([1.0, np.nan]),
            np.array([0.0, 0.0, 0.0]),
        ],
    )
    def test_invalid_inputs_fail_loud(self, bad: np.ndarray) -> None:
        with pytest.raises(ValueError):
            compute_s4(bad)


# ---------------------------------------------------------------------------
# sigma-phi (GISTM detrending, Van Dierendonck et al. 1993)
# ---------------------------------------------------------------------------


class TestSigmaPhi:
    def test_slow_trend_is_removed(self) -> None:
        """A pure sub-cutoff drift must contribute ~nothing after detrending."""
        fs = 50.0
        t = np.arange(0, 120.0, 1.0 / fs)
        slow_drift = 5.0 * np.sin(2 * np.pi * 0.01 * t)  # 0.01 Hz << 0.1 Hz cutoff
        sigma = compute_sigma_phi(slow_drift, sample_rate_hz=fs)
        assert sigma < 0.05

    def test_in_band_tone_amplitude_preserved(self) -> None:
        """A 1 Hz tone of amplitude A has std A/sqrt(2); the 0.1 Hz high-pass
        must pass it essentially unattenuated."""
        fs = 50.0
        t = np.arange(0, 120.0, 1.0 / fs)
        amplitude = 0.8
        tone = amplitude * np.sin(2 * np.pi * 1.0 * t)
        sigma = compute_sigma_phi(tone, sample_rate_hz=fs)
        assert sigma == pytest.approx(amplitude / math.sqrt(2.0), rel=0.02)

    def test_trend_plus_fluctuation_recovers_fluctuation(self) -> None:
        fs = 50.0
        t = np.arange(0, 120.0, 1.0 / fs)
        drift = 20.0 * t / t[-1]  # large linear ramp (clock/Doppler proxy)
        tone = 0.5 * np.sin(2 * np.pi * 0.7 * t)
        sigma = compute_sigma_phi(drift + tone, sample_rate_hz=fs)
        assert sigma == pytest.approx(0.5 / math.sqrt(2.0), rel=0.05)

    def test_missing_sample_rate_fails_loud(self) -> None:
        with pytest.raises(ValueError, match=r"sample_rate_hz"):
            compute_sigma_phi(np.zeros(1000), sample_rate_hz=0.0)

    def test_cutoff_above_nyquist_fails_loud(self) -> None:
        with pytest.raises(ValueError, match=r"Nyquist"):
            compute_sigma_phi(np.zeros(1000), sample_rate_hz=0.15)

    def test_too_short_series_fails_loud(self) -> None:
        with pytest.raises(ValueError, match=r"needs >="):
            compute_sigma_phi(np.zeros(10), sample_rate_hz=50.0)

    def test_nan_phase_fails_loud(self) -> None:
        arr = np.zeros(1000)
        arr[3] = np.nan
        with pytest.raises(ValueError, match=r"non-finite"):
            compute_sigma_phi(arr, sample_rate_hz=50.0)


# ---------------------------------------------------------------------------
# Classification (0.3 / 0.6 standard tiers)
# ---------------------------------------------------------------------------


class TestClassification:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, "weak"),
            (0.29, "weak"),
            (0.3, "moderate"),
            (0.45, "moderate"),
            (0.6, "moderate"),
            (0.61, "strong"),
            (0.7, "strong"),  # task anchor: S4 = 0.7 -> strong
            (1.2, "strong"),
        ],
    )
    def test_standard_tiers(self, value: float, expected: str) -> None:
        assert classify_scintillation(value) == expected

    @pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
    def test_invalid_index_fails_loud(self, bad: float) -> None:
        with pytest.raises(ValueError):
            classify_scintillation(bad)


# ---------------------------------------------------------------------------
# Measurement path (never fabricates)
# ---------------------------------------------------------------------------


class TestMeasure:
    def test_no_inputs_fails_loud(self) -> None:
        with pytest.raises(ValueError, match=r"no data"):
            IonosphericScintillationDetector().measure()

    def test_phase_without_sample_rate_fails_loud(self) -> None:
        with pytest.raises(ValueError, match=r"sample_rate_hz is required"):
            IonosphericScintillationDetector().measure(phase_rad=np.zeros(1000))

    def test_intensity_only(self) -> None:
        detector = IonosphericScintillationDetector()
        result = detector.measure(intensity=np.array([0.5, 1.5] * 50))
        assert result.is_measurement
        assert result.s4 == pytest.approx(0.5, abs=1e-12)
        assert result.amplitude_class == "moderate"
        assert result.sigma_phi_rad is None
        assert result.phase_class is None
        assert result.gnss_degradation == "moderate"

    def test_headline_takes_worst_class(self) -> None:
        detector = IonosphericScintillationDetector()
        fs = 50.0
        t = np.arange(0, 120.0, 1.0 / fs)
        strong_phase = 1.2 * np.sin(2 * np.pi * 1.0 * t)  # sigma ~ 0.85 rad
        weak_intensity = np.full(200, 3.0) + 0.01 * RNG.standard_normal(200)
        result = detector.measure(
            intensity=np.abs(weak_intensity), phase_rad=strong_phase, sample_rate_hz=fs
        )
        assert result.amplitude_class == "weak"
        assert result.phase_class == "strong"
        assert result.gnss_degradation == "strong"


# ---------------------------------------------------------------------------
# Climatological-risk path (labelled, never a measurement)
# ---------------------------------------------------------------------------


class TestClimatologicalRisk:
    def setup_method(self) -> None:
        self.detector = IonosphericScintillationDetector()

    def test_auroral_boundary_hand_computed(self) -> None:
        """Gussenhoven et al. (1983): 67.5 - 2.1*Kp. Kp=6 -> 54.9 deg."""
        assert self.detector.auroral_boundary_deg(0.0) == pytest.approx(67.5)
        assert self.detector.auroral_boundary_deg(6.0) == pytest.approx(54.9)

    def test_storm_time_auroral_high_risk(self) -> None:
        risk = self.detector.climatological_risk(
            kp=7.0, magnetic_latitude_deg=60.0, local_time_hours=23.0
        )
        assert risk.risk_level == "high"
        assert risk.risk_basis == "climatological"
        assert not risk.is_measurement
        assert any("auroral" in f for f in risk.factors)

    def test_quiet_time_same_latitude_not_flagged(self) -> None:
        """At Kp=1 the boundary sits at 65.4°, so 60° is equatorward of it."""
        risk = self.detector.climatological_risk(
            kp=1.0, magnetic_latitude_deg=60.0, local_time_hours=23.0
        )
        assert risk.risk_level == "low"

    def test_equatorial_post_sunset_window(self) -> None:
        risk = self.detector.climatological_risk(
            kp=2.0, magnetic_latitude_deg=5.0, local_time_hours=21.0
        )
        assert risk.risk_level == "moderate"
        assert any("post-sunset" in f for f in risk.factors)

    def test_equatorial_equinox_raises_to_high(self) -> None:
        risk = self.detector.climatological_risk(
            kp=2.0, magnetic_latitude_deg=5.0, local_time_hours=21.0, month=3
        )
        assert risk.risk_level == "high"
        assert any("equinox" in f for f in risk.factors)

    def test_equatorial_daytime_low(self) -> None:
        risk = self.detector.climatological_risk(
            kp=2.0, magnetic_latitude_deg=5.0, local_time_hours=12.0
        )
        assert risk.risk_level == "low"

    def test_midlatitude_severe_storm_moderate(self) -> None:
        risk = self.detector.climatological_risk(
            kp=8.0, magnetic_latitude_deg=40.0, local_time_hours=22.0
        )
        assert risk.risk_level == "moderate"
        assert any("storm-enhanced" in f for f in risk.factors)

    def test_risk_never_carries_scintillation_values(self) -> None:
        """The climatological path must not expose any S4/sigma-phi field."""
        risk = self.detector.climatological_risk(
            kp=5.0, magnetic_latitude_deg=65.0, local_time_hours=23.0
        )
        assert not hasattr(risk, "s4")
        assert not hasattr(risk, "sigma_phi_rad")
        assert risk.is_measurement is False

    @pytest.mark.parametrize(
        ("kp", "mlat", "lt", "month"),
        [
            (-1.0, 60.0, 12.0, None),
            (10.0, 60.0, 12.0, None),
            (3.0, 95.0, 12.0, None),
            (3.0, 60.0, 24.0, None),
            (3.0, 60.0, -0.1, None),
            (3.0, 60.0, 12.0, 13),
        ],
    )
    def test_out_of_range_inputs_fail_loud(
        self, kp: float, mlat: float, lt: float, month: int | None
    ) -> None:
        with pytest.raises(ValueError):
            self.detector.climatological_risk(
                kp=kp, magnetic_latitude_deg=mlat, local_time_hours=lt, month=month
            )
