# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the rockfall detector.

Literature worked example: Fukuzono's (1985) canonical accelerating creep,
``dv/dt = A v^2``, whose closed form is ``v(t) = 1 / (A (tf - t))`` — the
inverse velocity is exactly linear in time and extrapolates to zero at the
failure time tf. The detector must recover tf from the integrated
displacement series and must refuse to forecast when the fit gates fail
(R² gate per Carlà et al. 2017).

Also covered: moisture-gated freeze-thaw counting (Matsuoka & Murton 2008),
the D'Amato et al. (2016) 5 mm/h episode-intensity threshold with the
published frequency multipliers, the Poisson microseismic ramp test
(Amitrano et al. 2005), hazard mapping, and all fail-loud contracts.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.geological.rockfall_detector import (  # type: ignore[import-not-found,unused-ignore]
    FREEZE_THAW_FREQUENCY_MULTIPLIER,
    HIGH_RAIN_FREQUENCY_MULTIPLIER,
    RAIN_INTENSITY_THRESHOLD_MM_H,
    RockfallDetector,
    RockfallHazardLevel,
)

RNG = np.random.default_rng(19850401)  # Fukuzono 1985


def _fukuzono_displacement(
    tf: float = 10.0,
    a_coeff: float = 0.05,
    t_end: float = 9.0,
    n: int = 60,
    noise_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Displacement for Fukuzono alpha=2 creep: v = 1/(A (tf - t)).

    Integrating: d(t) = (1/A) * ln(tf / (tf - t)).
    """
    t = np.linspace(0.0, t_end, n)
    d = (1.0 / a_coeff) * np.log(tf / (tf - t))
    if noise_mm > 0:
        d = d + RNG.normal(0.0, noise_mm, size=t.shape)
    return d, t


class TestInverseVelocityFukuzono:
    """Fukuzono (1985) canonical worked example."""

    def test_recovers_failure_time_noise_free(self) -> None:
        d, t = _fukuzono_displacement(tf=10.0)
        iv = RockfallDetector().analyze_inverse_velocity(d, t)
        assert iv.accelerating
        assert iv.forecast_valid
        assert iv.r_squared > 0.99
        assert iv.slope < 0.0
        assert iv.failure_time is not None
        assert iv.failure_time == pytest.approx(10.0, abs=0.2)
        assert iv.failure_window is not None
        lo, hi = iv.failure_window
        assert lo <= iv.failure_time <= hi

    def test_recovers_failure_time_with_noise(self) -> None:
        d, t = _fukuzono_displacement(tf=10.0, noise_mm=0.05)
        iv = RockfallDetector().analyze_inverse_velocity(d, t)
        assert iv.forecast_valid
        assert iv.failure_time == pytest.approx(10.0, abs=1.0)

    def test_constant_velocity_is_not_a_forecast(self) -> None:
        """Steady creep: no acceleration, no failure time."""
        t = np.linspace(0.0, 9.0, 60)
        d = 2.0 * t
        iv = RockfallDetector().analyze_inverse_velocity(d, t)
        assert not iv.forecast_valid
        assert iv.failure_time is None

    def test_decelerating_creep_rejected(self) -> None:
        """Decaying creep (post-event relaxation) must not extrapolate."""
        t = np.linspace(0.0, 9.0, 60)
        d = 10.0 * (1.0 - np.exp(-t / 3.0))
        iv = RockfallDetector().analyze_inverse_velocity(d, t)
        assert not iv.forecast_valid
        assert iv.failure_time is None

    def test_noisy_acceleration_fails_r2_gate(self) -> None:
        """Acceleration buried in noise must be rejected by the R² gate."""
        d, t = _fukuzono_displacement(tf=10.0, t_end=6.0, noise_mm=3.0)
        iv = RockfallDetector().analyze_inverse_velocity(d, t)
        assert not iv.forecast_valid
        assert iv.rejection_reason != ""

    def test_too_short_series_raises(self) -> None:
        with pytest.raises(ValueError, match="need >="):
            RockfallDetector().analyze_inverse_velocity(np.arange(4.0), np.arange(4.0))

    def test_non_increasing_time_raises(self) -> None:
        d, t = _fukuzono_displacement()
        t2 = t.copy()
        t2[10] = t2[9]
        with pytest.raises(ValueError, match="strictly increasing"):
            RockfallDetector().analyze_inverse_velocity(d, t2)

    def test_nonfinite_input_raises(self) -> None:
        d, t = _fukuzono_displacement()
        d[3] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            RockfallDetector().analyze_inverse_velocity(d, t)


class TestFreezeThaw:
    """Moisture-gated freeze-thaw counting."""

    def _oscillating_temps(self, n_cycles: int = 5, samples_per_cycle: int = 24) -> np.ndarray:
        """Diurnal-style oscillation crossing 0 °C n_cycles times downward."""
        t = np.arange(n_cycles * samples_per_cycle, dtype=float)
        return 5.0 * np.sin(2.0 * np.pi * t / samples_per_cycle)

    def test_counts_wet_cycles(self) -> None:
        temps = self._oscillating_temps(5)
        precip = np.zeros_like(temps)
        precip[::12] = 1.0  # rain every 12 h keeps the face wet
        ft = RockfallDetector().analyze_freeze_thaw(temps, 1.0, precipitation_mm=precip)
        assert ft.effective_cycles == 5
        assert ft.total_crossings == 5
        assert ft.active

    def test_dry_crossings_do_not_count(self) -> None:
        """Crossings without moisture are counted as crossings, not cycles."""
        temps = self._oscillating_temps(5)
        ft = RockfallDetector().analyze_freeze_thaw(
            temps, 1.0, precipitation_mm=np.zeros_like(temps)
        )
        assert ft.total_crossings == 5
        assert ft.effective_cycles == 0
        assert not ft.active

    def test_surface_wet_flags_used(self) -> None:
        temps = self._oscillating_temps(3)
        wet = np.ones_like(temps, dtype=bool)
        ft = RockfallDetector().analyze_freeze_thaw(temps, 1.0, surface_wet=wet)
        assert ft.effective_cycles == 3

    def test_frost_cracking_fraction(self) -> None:
        """A series held at -5 °C sits fully inside the -8..-3 window."""
        temps = np.full(48, -5.0)
        ft = RockfallDetector().analyze_freeze_thaw(temps, 1.0, surface_wet=np.ones(48, dtype=bool))
        assert ft.frost_cracking_fraction == 1.0
        assert ft.effective_cycles == 0  # never crosses zero

    def test_missing_moisture_input_raises(self) -> None:
        """Refusing to assume wetness is the transparent failure mode."""
        with pytest.raises(ValueError, match="moisture"):
            RockfallDetector().analyze_freeze_thaw(self._oscillating_temps(2), 1.0)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            RockfallDetector().analyze_freeze_thaw(
                self._oscillating_temps(2), 1.0, precipitation_mm=np.zeros(3)
            )


class TestRainTrigger:
    """D'Amato et al. (2016) episode-intensity threshold."""

    def test_constants_match_publication(self) -> None:
        assert RAIN_INTENSITY_THRESHOLD_MM_H == 5.0
        assert HIGH_RAIN_FREQUENCY_MULTIPLIER == 26.0
        assert FREEZE_THAW_FREQUENCY_MULTIPLIER == 7.0

    def test_intense_episode_triggers(self) -> None:
        rain = np.concatenate([np.zeros(10), np.full(4, 8.0)])  # 8 mm/h for 4 h
        result = RockfallDetector().analyze_rain_trigger(rain, 1.0)
        assert result["episode_ongoing"]
        assert result["mean_intensity_mm_h"] == pytest.approx(8.0)
        assert result["trigger_active"]
        assert result["frequency_multiplier"] == 26.0

    def test_light_episode_does_not_trigger(self) -> None:
        rain = np.concatenate([np.zeros(10), np.full(6, 1.0)])
        result = RockfallDetector().analyze_rain_trigger(rain, 1.0)
        assert result["episode_ongoing"]
        assert not result["trigger_active"]
        assert result["frequency_multiplier"] == 1.0

    def test_dry_now_means_no_episode(self) -> None:
        rain = np.concatenate([np.full(4, 8.0), np.zeros(2)])
        result = RockfallDetector().analyze_rain_trigger(rain, 1.0)
        assert not result["episode_ongoing"]
        assert not result["trigger_active"]

    def test_negative_rain_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            RockfallDetector().analyze_rain_trigger(np.array([1.0, -0.5]), 1.0)


class TestMicroseismicRamp:
    """Poisson event-rate ramp test."""

    def test_strong_ramp_detected(self) -> None:
        base = np.linspace(0.0, 80.0, 20, endpoint=False)  # 0.25 events/unit
        recent = np.linspace(80.0, 100.0, 40, endpoint=False)  # 2 events/unit
        result = RockfallDetector().analyze_microseismic_rate(
            np.concatenate([base, recent]), (0.0, 100.0)
        )
        assert result["rate_ratio"] == pytest.approx(8.0, rel=0.05)
        assert result["p_value"] < 0.01
        assert result["ramp_detected"]

    def test_steady_rate_no_ramp(self) -> None:
        events = np.linspace(0.0, 100.0, 50, endpoint=False)
        result = RockfallDetector().analyze_microseismic_rate(events, (0.0, 100.0))
        assert not result["ramp_detected"]

    def test_small_baseline_raises(self) -> None:
        with pytest.raises(ValueError, match="baseline"):
            RockfallDetector().analyze_microseismic_rate(np.array([95.0, 96.0, 97.0]), (0.0, 100.0))

    def test_events_outside_window_raise(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            RockfallDetector().analyze_microseismic_rate(
                np.array([1.0, 2.0, 3.0, 120.0]), (0.0, 100.0)
            )


class TestPredictRockfall:
    """Full-channel composition and hazard mapping."""

    def test_valid_iv_forecast_is_critical_with_window(self) -> None:
        d, t = _fukuzono_displacement(tf=10.0)
        result = RockfallDetector().predict_rockfall({"displacement_mm": d, "displacement_time": t})
        assert result.hazard_level == RockfallHazardLevel.CRITICAL.value
        assert result.inverse_velocity is not None
        assert result.inverse_velocity.forecast_valid
        assert any("failure window" in w.lower() for w in result.warnings)
        assert result.channels_assessed == ["inverse_velocity"]

    def test_meteorological_triggers_compose(self) -> None:
        temps = 5.0 * np.sin(2.0 * np.pi * np.arange(120.0) / 24.0)
        rain = np.zeros(120)
        rain[100] = 2.0  # wets the face before the 0 °C crossing at t=108 h
        rain[116:120] = 8.0  # ongoing 8 mm/h episode at the end
        result = RockfallDetector().predict_rockfall(
            {
                "rock_temperature_c": temps,
                "precipitation_mm": rain,
                "rain_mm": rain,
                "sample_interval_hours": 1.0,
            }
        )
        assert result.hazard_level == RockfallHazardLevel.HIGH.value
        assert result.freeze_thaw is not None and result.freeze_thaw.active
        assert result.rain_trigger_active
        # Published multipliers are combined with max(), not multiplied.
        assert result.estimated_frequency_multiplier == 26.0

    def test_single_trigger_is_moderate(self) -> None:
        rain = np.concatenate([np.zeros(10), np.full(4, 8.0)])
        result = RockfallDetector().predict_rockfall({"rain_mm": rain})
        assert result.hazard_level == RockfallHazardLevel.MODERATE.value

    def test_quiet_channels_are_low(self) -> None:
        rain = np.zeros(24)
        result = RockfallDetector().predict_rockfall({"rain_mm": rain})
        assert result.hazard_level == RockfallHazardLevel.LOW.value
        assert result.confidence == 0.0

    def test_ramp_plus_trigger_is_critical(self) -> None:
        base = np.linspace(0.0, 80.0, 20, endpoint=False)
        recent = np.linspace(80.0, 100.0, 40, endpoint=False)
        rain = np.concatenate([np.zeros(10), np.full(4, 8.0)])
        result = RockfallDetector().predict_rockfall(
            {
                "microseismic_event_times": np.concatenate([base, recent]),
                "observation_window": (0.0, 100.0),
                "rain_mm": rain,
            }
        )
        assert result.hazard_level == RockfallHazardLevel.CRITICAL.value
        assert result.microseismic_ramp

    def test_no_channels_raises(self) -> None:
        with pytest.raises(ValueError, match="no channel input"):
            RockfallDetector().predict_rockfall({})

    def test_partial_iv_input_raises(self) -> None:
        with pytest.raises(ValueError, match="both displacement_mm"):
            RockfallDetector().predict_rockfall({"displacement_mm": np.arange(10.0)})

    def test_evidence_carries_citations(self) -> None:
        rain = np.concatenate([np.zeros(10), np.full(4, 8.0)])
        result = RockfallDetector().predict_rockfall({"rain_mm": rain})
        assert all("citation" in e for e in result.evidence)
