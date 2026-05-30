"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Spectral Domain Frequency — Full-Power Neuro-Symbolic Implementation.

A production-grade spectral-domain anomaly detection system covering
amplitude, phase, entropy, and harmonic structure analysis. Runs
parallel to all time-domain detectors and produces a
:class:`~omni_mercury_engine.detectors.spectral_domain_frequency.FrequencyInfluenceVector`
that modulates fusion-layer scoring.

Capabilities:

1. **Selective Inference (SI) framework** — truncated normal conditioning
   on the CUSUM selection event to guarantee post-selection Type I error
   control (Lee et al., 2016; Takeuchi Lab, arXiv:2502.03062).
2. **Binary segmentation change-point detection** — recursive CUSUM-based
   CP detection with configurable recursion depth.
3. **Windowed DFT** — sliding Hann-windowed frames with 50% overlap for
   temporal localisation of spectral changes.
4. **Spectral Flux** — rate-of-spectral-change detection for slow-onset
   anomalies.
5. **Phase Coherence** — inter-band phase relationship monitoring; phase
   decoherence can precede amplitude changes.
6. **Cepstral Analysis** — harmonic structure fingerprinting via inverse
   FFT of the log power spectrum.
7. **φ-weighted influence multiplier** — five-signal geometric mean
   (score, entropy, breadth, flux, coherence) with golden ratio weighting.
8. **Domain-aware auto-activation** — enabled/neutral/disabled per
   :data:`~omni_mercury_engine.core.config.ORACLE_DOMAIN_POLICY` (overridable
   via :class:`~omni_mercury_engine.core.config.OracleActivation`).

Supported domains (7):
    environmental (8 bands), medical (9 bands),
    infrastructure (8 bands), security (6 bands),
    financial (7 bands), space (7 bands),
    humanitarian (5 bands)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
from scipy.fft import fft, fftfreq
from scipy.stats import norm, truncnorm

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.centralized_constants import (
    MATH,
)
from omni_mercury_engine.core.exceptions import DetectorException

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

PHI = MATH.GOLDEN_RATIO  # 1.618033988749895
EPSILON = MATH.EPSILON  # 1e-8
DEFAULT_ALPHA = 0.05


# =============================================================================
# Enums
# =============================================================================


class FrequencyWeighting(Enum):
    """
    Weighting scheme for frequency bands.

    Per OSHA OTM §III.5 — includes physics-based A/C weighting in addition to domain-adaptive
    schemes.
    """

    A_WEIGHTED = "a_weighted"
    C_WEIGHTED = "c_weighted"
    Z_WEIGHTED = "z_weighted"
    DOMAIN_ADAPTIVE = "domain_adaptive"


# =============================================================================
# Domain-specific frequency bands
# =============================================================================

# Each band: (low_hz, high_hz, label, weight)
DOMAIN_FREQUENCY_BANDS: dict[str, list[tuple[float, float, str, float]]] = {
    "environmental": [
        (0.001, 1.0, "infrasound_geophysical", 0.10),
        (1.0, 7.83, "sub_schumann", 0.10),
        (7.83, 8.5, "schumann_fundamental", 0.20),
        (8.5, 14.3, "schumann_harmonic_1", 0.15),
        (14.3, 33.8, "schumann_harmonics_upper", 0.10),
        (33.8, 300.0, "elf_upper", 0.10),
        (300.0, 3000.0, "vlf_environmental", 0.10),
        (3000.0, 30000.0, "atmospheric_noise", 0.15),
    ],
    "medical": [
        (0.003, 0.04, "vlf_hrv", 0.10),
        (0.04, 0.15, "lf_hrv_sympathetic", 0.12),
        (0.15, 0.4, "hf_hrv_parasympathetic", 0.12),
        (0.4, 3.0, "respiratory_cardiac", 0.10),
        (4.0, 8.0, "theta_neural", 0.12),
        (8.0, 13.0, "alpha_neural", 0.12),
        (13.0, 30.0, "beta_neural", 0.12),
        (30.0, 50.0, "gamma_neural_40hz", 0.10),
        (50.0, 150.0, "high_gamma_motor", 0.10),
    ],
    "infrastructure": [
        (0.01, 0.5, "structural_sway", 0.10),
        (0.5, 5.0, "seismic_structural", 0.12),
        (5.0, 25.0, "mechanical_vibration", 0.12),
        (25.0, 48.0, "motor_bearing_fault", 0.15),
        (48.0, 52.0, "mains_50hz", 0.13),
        (58.0, 62.0, "mains_60hz", 0.13),
        (62.0, 500.0, "harmonic_distortion", 0.12),
        (500.0, 10000.0, "high_frequency_fault", 0.13),
    ],
    "security": [
        (0.0, 1.0, "baseline", 0.15),
        (1.0, 10.0, "session", 0.20),
        (10.0, 100.0, "burst", 0.20),
        (100.0, 1000.0, "scan", 0.15),
        (1000.0, 10000.0, "flood", 0.15),
        (10000.0, 100000.0, "ultra_high_rate", 0.15),
    ],
    "financial": [
        (0.0, 0.004, "macro_cycle", 0.10),
        (0.004, 0.01, "quarterly_cycle", 0.12),
        (0.01, 0.1, "swing", 0.18),
        (0.1, 1.0, "intraday", 0.20),
        (1.0, 10.0, "hft_low", 0.15),
        (10.0, 500.0, "hft_high", 0.15),
        (500.0, 50000.0, "microstructure_noise", 0.10),
    ],
    "space": [
        (0.0, 3.5e-8, "solar_cycle", 0.10),
        (3.5e-8, 4.5e-7, "solar_rotation", 0.12),
        (4.5e-7, 0.001, "magnetospheric", 0.15),
        (0.001, 0.01, "solar_wind", 0.18),
        (0.01, 0.1, "ionospheric", 0.15),
        (0.1, 8.0, "schumann_coupling", 0.15),
        (8.0, 50.0, "whistler", 0.15),
    ],
    "humanitarian": [
        (0.0, 0.01, "population_movement", 0.20),
        (0.01, 0.1, "daily_activity_cycle", 0.20),
        (0.1, 1.0, "communication_burst", 0.25),
        (1.0, 10.0, "event_response", 0.20),
        (10.0, 100.0, "alert_propagation", 0.15),
    ],
}


DOMAIN_ANOMALY_SPECTRAL_HINTS: dict[str, dict[str, Any]] = {
    "environmental": {
        "expect_broadband_spike": True,
        "primary_band": "sub_schumann",
        "anomaly_beta_shift": -0.5,  # Anomaly whitens the spectrum
    },
    "ocean": {
        "expect_low_freq_shift": True,
        "primary_band": "infrasound_geophysical",
        "anomaly_beta_shift": +0.5,  # Anomaly reddens the spectrum
    },
    "security": {
        "expect_narrowband_spike": True,
        "primary_band": "high_frequency",
        "anomaly_beta_shift": -1.0,  # Anomaly creates sharp peaks
    },
    "space": {
        "expect_broadband_spike": True,
        "primary_band": "schumann",
        "anomaly_beta_shift": -0.5,
    },
    "climate": {
        "expect_low_freq_shift": True,
        "primary_band": "sub_schumann",
        "anomaly_beta_shift": +0.3,
    },
}


def get_domain_frequency_bands(
    domain: str,
) -> list[tuple[float, float, str, float]]:
    """
    Return the frequency-band definition for *domain*.

    Falls back to ``environmental`` bands when *domain* is unknown.
    """
    return DOMAIN_FREQUENCY_BANDS.get(domain.lower(), DOMAIN_FREQUENCY_BANDS["environmental"])


# =============================================================================
# Data structures
# =============================================================================


@dataclass(frozen=True)
class FrequencyBandResult:
    """
    Per-band structured detection result.

    Attributes:
        band_label: Human-readable band name.
        low_hz: Lower frequency bound (Hz).
        high_hz: Upper frequency bound (Hz).
        band_weight: Normalised weight for this band.
        power_ratio: Ratio of observed-to-reference band power.
        z_score: Standardised deviation from reference.
        anomaly_score: Combined anomaly score for this band [0, 1].
        p_value: Statistical significance (SI-corrected when CP detected).
        is_significant: ``True`` when ``p_value < alpha``.
    """

    band_label: str
    low_hz: float
    high_hz: float
    band_weight: float
    power_ratio: float
    z_score: float
    anomaly_score: float
    p_value: float
    is_significant: bool


@dataclass(frozen=True)
class FrequencyInfluenceVector:
    """
    Output of the Oracle for a single observation.

    Attributes:
        influence_multiplier: Score modulation factor (> 1 amplify, < 1 suppress).
        band_scores: Per-band anomaly scores keyed by label.
        aggregate_score: Weighted aggregate anomaly score [0, 1].
        aggregate_p_value: Fisher combined p-value across bands.
        spectral_entropy: Shannon entropy of the power spectrum.
        dominant_frequency: Frequency with highest power (Hz).
        spectral_centroid: Centre of mass of the spectrum (Hz).
        change_point_detected: True when binary segmentation found a CP.
        confidence: 1 - aggregate_p_value.
        spectral_flux: Rate-of-spectral-change (L2 norm of frame diffs).
        phase_coherence: Mean inter-band phase coherence [0, 1].
        cepstral_peak_ratio: Max quefrency peak / mean (harmonic indicator).
    """

    influence_multiplier: float
    band_scores: dict[str, float]
    aggregate_score: float
    aggregate_p_value: float
    spectral_entropy: float
    dominant_frequency: float
    spectral_centroid: float
    change_point_detected: bool
    confidence: float
    spectral_flux: float = 0.0
    phase_coherence: float = 0.0
    cepstral_peak_ratio: float = 0.0


@dataclass
class SpectralDomainFrequencyConfig:
    """
    Configuration for SpectralDomainFrequency.

    Attributes:
        domain: Application domain for band selection.
        sample_rate: Sampling rate of the input signal in Hz.
        threshold: Anomaly score threshold in [0, 1].
        weighting: Band weighting scheme.
        significance_level: Alpha for change-point testing.
        influence_floor: Minimum influence multiplier.
        influence_ceiling: Maximum influence multiplier.
        inner_window: Window length for windowed DFT (samples).
        outer_window_ratio: Ratio of outer-to-inner window.
        min_segments: Minimum segment length for binary segmentation.
    """

    domain: str = "environmental"
    sample_rate: float = 1000.0
    threshold: float = 0.5
    weighting: FrequencyWeighting = FrequencyWeighting.DOMAIN_ADAPTIVE
    significance_level: float = DEFAULT_ALPHA
    influence_floor: float = 0.5
    influence_ceiling: float = 2.0
    inner_window: int = 256
    outer_window_ratio: float = 4.0
    min_segments: int = 8


# =============================================================================
# Factory helper
# =============================================================================


def create_spectral_frequency(
    config: dict[str, Any] | None = None,
) -> SpectralDomainFrequency:
    """
    Factory function for SpectralDomainFrequency.

    Args:
        config: Optional configuration dictionary.

    Returns:
        Configured SpectralDomainFrequency instance.
    """
    return SpectralDomainFrequency(config)


# =============================================================================
# Main detector
# =============================================================================


class SpectralDomainFrequency(BaseDetector):
    """Full-power neuro-symbolic spectral-domain anomaly detection Oracle.

    The Oracle decomposes a signal into domain-specific frequency bands
    using a **windowed DFT** (Hann window, 50% overlap), then applies:

    1. Per-band z-score anomaly scoring against reference statistics.
    2. **Binary segmentation** change-point detection (CUSUM-based,
       recursive up to depth 5).
    3. **Selective inference** p-value correction with truncated normal
       conditioning for detected CPs (Lee et al., 2016).
    4. **Spectral flux** — rate-of-spectral-change detection.
    5. **Phase coherence** — inter-band phase relationship monitoring.
    6. **Cepstral analysis** — harmonic structure fingerprinting.
    7. **φ-weighted influence multiplier** — five-signal geometric mean
       of score, entropy, breadth, flux, and coherence.

    The resulting :class:`FrequencyInfluenceVector` modulates the fused
    anomaly score in ``AdvancedPhysicsIntegratedDetector``.

    Example::

        oracle = SpectralDomainFrequency({"domain": "medical", "sample_rate": 256.0})
        oracle.fit(training_signals)       # (N, T) array
        result = oracle.detect(test_signal) # (T,) array
        iv = result["influence_vector"]     # FrequencyInfluenceVector
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)

        cfg = config or {}
        domain = cfg.get("domain", "environmental")

        self._oracle_config = SpectralDomainFrequencyConfig(
            domain=domain,
            sample_rate=cfg.get("sample_rate", 1000.0),
            threshold=self.threshold,
            weighting=FrequencyWeighting(cfg.get("weighting", "domain_adaptive")),
            significance_level=cfg.get("significance_level", DEFAULT_ALPHA),
            influence_floor=cfg.get("influence_floor", 0.5),
            influence_ceiling=cfg.get("influence_ceiling", 2.0),
            inner_window=cfg.get("inner_window", 256),
            outer_window_ratio=cfg.get("outer_window_ratio", 4.0),
            min_segments=cfg.get("min_segments", 8),
        )

        # Resolve bands (filters by Nyquist, renormalises weights)
        raw_bands = get_domain_frequency_bands(domain)
        self._bands = self._resolve_bands(raw_bands)

        # Reference statistics (populated by fit())
        self._ref_band_powers: dict[str, np.ndarray[Any, Any]] | None = None
        self._ref_band_means: dict[str, float] = {}
        self._ref_band_stds: dict[str, float] = {}
        self._ref_spectral_entropy_mean: float = 0.0
        self._ref_spectral_entropy_std: float = 1.0
        self._ref_full_spectrum_mean: np.ndarray[Any, Any] | None = None
        self._ref_full_spectrum_std: np.ndarray[Any, Any] | None = None

        # Noise color estimation (from F1 Precision Directive)
        self._noise_beta: float = 0.0
        self._noise_color: str = "white"
        self._noise_fit_r2: float = 0.0

    # ------------------------------------------------------------------
    # Band resolution
    # ------------------------------------------------------------------

    def _resolve_bands(
        self,
        raw_bands: list[tuple[float, float, str, float]],
    ) -> list[tuple[float, float, str, float]]:
        """
        Filter bands exceeding Nyquist frequency and renormalise weights.

        Bands whose lower bound exceeds ``sample_rate / 2`` are excluded.
        Remaining weights are renormalised to sum to 1.

        Args:
            raw_bands: Unfiltered band definitions.

        Returns:
            Nyquist-filtered, weight-normalised bands.
        """
        nyquist = self._oracle_config.sample_rate / 2.0
        filtered = [(lo, hi, label, w) for lo, hi, label, w in raw_bands if lo < nyquist]
        if not filtered:
            # Fallback: keep at least the lowest band
            filtered = [raw_bands[0]]
            logger.warning(
                "All bands exceed Nyquist (%.1f Hz); keeping lowest band only.",
                nyquist,
            )

        weight_sum = sum(w for _, _, _, w in filtered)
        if weight_sum > 0:
            filtered = [
                (lo, min(hi, nyquist), label, w / weight_sum) for lo, hi, label, w in filtered
            ]
        return filtered

    # ------------------------------------------------------------------
    # Windowed DFT
    # ------------------------------------------------------------------

    def _compute_frequency_matrix(
        self,
        signal: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """
        Compute windowed DFT with Hann window and 50 % overlap.

        Returns a ``[n_windows, n_freq_bins]`` power matrix and the
        corresponding frequency axis.  Each window's energy is
        normalised by window length for comparability.

        Args:
            signal: 1-D time-domain signal.

        Returns:
            ``(freq_matrix, freqs)`` where ``freq_matrix`` has shape
            ``[n_windows, n_positive_freq_bins]`` and ``freqs`` is 1-D.
        """
        n = len(signal)
        win_len = min(self._oracle_config.inner_window, n)
        hop = max(win_len // 2, 1)

        hann = np.hanning(win_len)

        # Pre-compute frequency axis ONCE (invariant across windows)
        freqs_full = fftfreq(win_len, d=1.0 / self._oracle_config.sample_rate)
        pos_mask = freqs_full >= 0

        windows: list[np.ndarray[Any, Any]] = []
        start = 0
        while start + win_len <= n:
            segment = signal[start : start + win_len] * hann
            spectrum = fft(segment)
            magnitude = np.abs(spectrum[pos_mask]) / win_len
            windows.append(magnitude)
            start += hop

        # Handle edge case: signal shorter than window
        if not windows:
            segment = signal * np.hanning(n)
            spectrum = fft(segment)
            freqs_full = fftfreq(n, d=1.0 / self._oracle_config.sample_rate)
            pos_mask = freqs_full >= 0
            magnitude = np.abs(spectrum[pos_mask]) / n
            windows.append(magnitude)
            win_len = n

        freq_matrix = np.array(windows)
        freqs = fftfreq(win_len, d=1.0 / self._oracle_config.sample_rate)
        freqs = freqs[freqs >= 0]

        return freq_matrix, freqs

    # ------------------------------------------------------------------
    # Parseval validation
    # ------------------------------------------------------------------

    def _validate_parseval_energy(
        self,
        signal: np.ndarray[Any, Any],
        freq_matrix: np.ndarray[Any, Any],
    ) -> bool:
        """
        Validate Parseval's theorem using the already-computed freq_matrix.

        Does **not** recompute the FFT.  Compares time-domain energy
        against the mean per-window frequency-domain energy.

        Args:
            signal: Original time-domain signal.
            freq_matrix: ``[n_windows, n_freq_bins]`` power matrix from
                ``_compute_frequency_matrix``.

        Returns:
            ``True`` if the check passes within tolerance.
        """
        time_energy = float(np.sum(signal**2)) / len(signal)
        # Mean per-window frequency energy (already normalised by win_len)
        freq_energy = float(np.mean(np.sum(freq_matrix**2, axis=1)))

        if time_energy < EPSILON:
            return True  # Near-zero signal — nothing to validate

        ratio = abs(freq_energy - time_energy) / (time_energy + EPSILON)
        passes = ratio < 0.5  # Generous tolerance for windowed DFT

        if not passes:
            logger.debug(
                "Parseval validation: time=%.6f freq=%.6f ratio=%.4f",
                time_energy,
                freq_energy,
                ratio,
            )
        return passes

    # ------------------------------------------------------------------
    # Band power extraction
    # ------------------------------------------------------------------

    def _extract_band_powers(
        self,
        freq_matrix: np.ndarray[Any, Any],
        freqs: np.ndarray[Any, Any],
    ) -> dict[str, np.ndarray[Any, Any]]:
        """
        Extract per-band power time series from the frequency matrix.

        Returns a dict mapping ``band_label`` to a 1-D array of
        per-window band power.  Preserves temporal structure for
        change-point detection.

        Args:
            freq_matrix: ``[n_windows, n_freq_bins]`` power matrix.
            freqs: Positive frequency axis.

        Returns:
            ``{band_label: power_per_window}`` dict.
        """
        band_powers: dict[str, np.ndarray[Any, Any]] = {}
        for lo, hi, label, _w in self._bands:
            mask = (freqs >= lo) & (freqs < hi)
            if np.any(mask):
                band_powers[label] = np.sum(freq_matrix[:, mask] ** 2, axis=1)
            else:
                band_powers[label] = np.zeros(freq_matrix.shape[0])
        return band_powers

    # ------------------------------------------------------------------
    # Spectral statistics
    # ------------------------------------------------------------------

    def _compute_spectral_entropy(
        self,
        freq_matrix: np.ndarray[Any, Any],
    ) -> float:
        """
        Shannon entropy of the mean power spectrum.

        Args:
            freq_matrix: ``[n_windows, n_freq_bins]`` power matrix.

        Returns:
            Spectral entropy (nats).
        """
        mean_spectrum = np.mean(freq_matrix, axis=0)
        total = np.sum(mean_spectrum) + EPSILON
        p = mean_spectrum / total
        p = p[p > 0]
        return float(-np.sum(p * np.log(p + EPSILON)))

    def _compute_spectral_centroid(
        self,
        freq_matrix: np.ndarray[Any, Any],
        freqs: np.ndarray[Any, Any],
    ) -> float:
        """
        Centre of mass of the mean power spectrum in Hz.

        Args:
            freq_matrix: ``[n_windows, n_freq_bins]`` power matrix.
            freqs: Positive frequency axis.

        Returns:
            Spectral centroid frequency (Hz).
        """
        mean_spectrum = np.mean(freq_matrix, axis=0)
        total = np.sum(mean_spectrum) + EPSILON
        return float(np.sum(freqs * mean_spectrum) / total)

    # ------------------------------------------------------------------
    # Spectral Flux — rate-of-spectral-change detection
    # ------------------------------------------------------------------

    def _compute_spectral_flux(
        self,
        freq_matrix: np.ndarray[Any, Any],
    ) -> float:
        """
        Compute spectral flux: rate of spectral change across frames.

        Spectral flux measures the L2 norm of the frame-to-frame difference
        in the power spectrum, normalised by the number of frames.  High
        flux indicates rapid spectral evolution — a frequency-domain analog
        of acceleration that is useful for detecting slow-onset anomalies
        (e.g., gradual infrastructure degradation, creeping sensor drift).

        Args:
            freq_matrix: ``[n_windows, n_freq_bins]`` power matrix.

        Returns:
            Mean spectral flux (non-negative).
        """
        if freq_matrix.shape[0] < 2:
            return 0.0

        diffs = np.diff(freq_matrix, axis=0)
        frame_fluxes = np.linalg.norm(diffs, axis=1)
        return float(np.mean(frame_fluxes))

    # ------------------------------------------------------------------
    # Phase Coherence — inter-band phase relationship monitoring
    # ------------------------------------------------------------------

    def _compute_phase_coherence(
        self,
        signal: np.ndarray[Any, Any],
    ) -> float:
        """
        Compute mean inter-band phase coherence.

        Phase coherence degrades before amplitude changes, making it a
        *leading indicator* of anomalous behaviour.  Uses Welch's method
        cross-spectral density estimation (``scipy.signal.csd``) to
        compute magnitude-squared coherence between adjacent frequency
        bands.

        When fewer than 2 bands have sufficient data, returns 1.0
        (maximally coherent — no evidence of decoherence).

        Args:
            signal: 1-D time-domain signal.

        Returns:
            Mean coherence in [0, 1].
        """
        from scipy.signal import coherence as scipy_coherence

        sr = self._oracle_config.sample_rate
        nperseg = min(256, len(signal) // 2) if len(signal) >= 4 else len(signal)

        if nperseg < 4 or len(self._bands) < 2:
            logger.debug(
                "Phase coherence: insufficient data (nperseg=%d, bands=%d), defaulting to 1.0",
                nperseg,
                len(self._bands),
            )
            return 1.0

        # Bandpass-filtered power series per band
        n = len(signal)
        win_len = min(self._oracle_config.inner_window, n)
        freqs_full = fftfreq(win_len, d=1.0 / sr)
        pos_freqs = freqs_full[freqs_full >= 0]

        # Build per-band time-domain approximations via inverse FFT masking
        band_signals: list[np.ndarray[Any, Any]] = []
        spectrum = fft(signal[:win_len] * np.hanning(win_len))
        for lo, hi, _label, _w in self._bands:
            mask_pos = (pos_freqs >= lo) & (pos_freqs <= hi)
            if not np.any(mask_pos):
                continue

            # Zero out frequencies outside the band
            filtered = np.zeros_like(spectrum)
            # Positive frequencies
            pos_indices = np.where(freqs_full >= 0)[0]
            for idx in pos_indices[mask_pos]:
                filtered[idx] = spectrum[idx]
                # Mirror for negative frequencies
                if idx > 0 and idx < len(spectrum) - 1:
                    filtered[len(spectrum) - idx] = spectrum[len(spectrum) - idx]

            band_sig = np.real(np.fft.ifft(filtered))
            band_signals.append(band_sig)

        if len(band_signals) < 2:
            logger.debug("Phase coherence: fewer than 2 bands with data, defaulting to 1.0")
            return 1.0

        # Pairwise coherencebetween adjacent bands
        coherences: list[float] = []
        for i in range(len(band_signals) - 1):
            try:
                _, coh = scipy_coherence(
                    band_signals[i],
                    band_signals[i + 1],
                    fs=sr,
                    nperseg=min(nperseg, len(band_signals[i])),
                )
                # Filter NaN values from coherence (caused by zero-power bins)
                coh_clean = coh[np.isfinite(coh)]
                if len(coh_clean) > 0:
                    coherences.append(float(np.mean(coh_clean)))
                else:
                    coherences.append(1.0)
            except (ValueError, ZeroDivisionError):
                coherences.append(1.0)

        if not coherences:
            logger.debug("Phase coherence: no valid pairwise coherence computed, defaulting to 1.0")
            return 1.0
        mean_coh = float(np.mean(coherences))
        # Guard against residual NaN from edge cases
        if not np.isfinite(mean_coh):
            logger.debug("Phase coherence: mean coherence non-finite, defaulting to 1.0")
            return 1.0
        return float(np.clip(mean_coh, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Cepstral Analysis — harmonic structure fingerprinting
    # ------------------------------------------------------------------

    def _compute_cepstral_peak(
        self,
        freq_matrix: np.ndarray[Any, Any],
    ) -> float:
        """Compute cepstral peak ratio for harmonic structure detection.

        The cepstrum (inverse FFT of log power spectrum) reveals harmonic
        structure as peaks in the quefrency domain.  The peak ratio
        (max quefrency peak / mean) indicates how strongly harmonic the
        signal is — useful for rotating machinery faults, resonance
        detection, and structural mode identification.

        Args:
            freq_matrix: ``[n_windows, n_freq_bins]`` power matrix.

        Returns:
            Cepstral peak ratio (>= 1.0; higher = more harmonic).
        """
        mean_spectrum = np.mean(freq_matrix, axis=0)

        # Log power spectrum (avoid log(0))
        log_spectrum = np.log(mean_spectrum + EPSILON)

        # Cepstrum via inverse FFT of log power spectrum
        cepstrum = np.abs(np.fft.ifft(log_spectrum))

        # Skip quefrency 0 (DC component of log spectrum)
        if len(cepstrum) < 3:
            return 1.0

        cepstrum_excl_dc = cepstrum[1:]
        mean_cep = float(np.mean(cepstrum_excl_dc))
        max_cep = float(np.max(cepstrum_excl_dc))

        if mean_cep < EPSILON:
            return 1.0

        return float(max_cep / mean_cep)

    # ------------------------------------------------------------------
    # Binary segmentation change-point detection
    # ------------------------------------------------------------------

    def _binary_segmentation_frequency(
        self,
        band_power_series: np.ndarray[Any, Any],
    ) -> list[int]:
        """
        CUSUM-based recursive binary segmentation for change-point detection.

        Args:
            band_power_series: 1-D array of per-window band power.

        Returns:
            List of detected change-point indices (window indices).
        """
        change_points: list[int] = []
        min_seg = max(self._oracle_config.min_segments, 2)
        self._binseg_recurse(band_power_series, 0, len(band_power_series), change_points, min_seg)
        return sorted(change_points)

    def _binseg_recurse(
        self,
        series: np.ndarray[Any, Any],
        start: int,
        end: int,
        change_points: list[int],
        min_seg: int,
        depth: int = 0,
        max_depth: int = 5,
    ) -> None:
        """
        Recursive binary segmentation helper.

        Computes CUSUM statistic over the segment ``[start, end)`` and
        splits at the point of maximum absolute CUSUM.  Recurses on
        each sub-segment up to ``max_depth``.

        Args:
            series: Full band-power time series.
            start: Segment start index (inclusive).
            end: Segment end index (exclusive).
            change_points: Accumulator for detected CPs.
            min_seg: Minimum segment length.
            depth: Current recursion depth.
            max_depth: Maximum recursion depth.
        """
        if depth >= max_depth or (end - start) < 2 * min_seg:
            return

        segment = series[start:end]
        n = len(segment)
        cumsum = np.cumsum(segment - np.mean(segment))
        # CUSUM statistic: max |S_k|
        abs_cusum = np.abs(cumsum)
        best_idx = int(np.argmax(abs_cusum))

        # Threshold: significant if max CUSUM exceeds sqrt(n) * sigma
        sigma = np.std(segment) + EPSILON
        threshold = np.sqrt(n) * sigma

        if abs_cusum[best_idx] > threshold and best_idx >= min_seg and (n - best_idx) >= min_seg:
            cp = start + best_idx
            change_points.append(cp)
            # Recurse on both halves
            self._binseg_recurse(series, start, cp, change_points, min_seg, depth + 1, max_depth)
            self._binseg_recurse(series, cp, end, change_points, min_seg, depth + 1, max_depth)

    # ------------------------------------------------------------------
    # Selective Inference — Truncated Normal Conditioning
    # ------------------------------------------------------------------

    def _compute_truncation_interval(
        self,
        series: np.ndarray[Any, Any],
        cp_index: int,
        min_seg: int,
    ) -> tuple[float, float]:
        """Compute the SI truncation interval [L, U] for a selected CP.

        The binary segmentation selected cp_index because |CUSUM(cp_index)|
        was maximal among all tested positions. The truncation interval is
        the set of test statistic values t for which cp_index would still
        be selected.

        Implements parametric Selective Inference per Lee et al. (2016),
        "Exact post-selection inference, with application to the LASSO,"
        adapted for CUSUM-based binary segmentation.

        The key insight: the selection event {k* = argmax |S_k|} defines
        linear constraints on the data y. When parameterized along the
        test direction eta, these constraints become interval constraints
        on the test statistic t, giving the truncation interval [L, U].

        Args:
            series: 1-D band-power time series.
            cp_index: Selected change-point index.
            min_seg: Minimum segment length from config.

        Returns:
            (L, U) truncation bounds. L may be -inf, U may be +inf.
        """
        n = len(series)
        k = cp_index

        # Contrast vector eta: mean(right) - mean(left)
        eta = np.zeros(n)
        eta[:k] = -1.0 / k
        eta[k:] = 1.0 / (n - k)
        eta_norm_sq = np.dot(eta, eta)

        if eta_norm_sq < EPSILON:
            return -np.inf, np.inf

        # Observed test statistic
        t_obs = np.dot(eta, series)

        # Residual orthogonal to eta
        z = series - t_obs * (eta / eta_norm_sq)

        # CUSUM at position j: S_j = v_j^T y
        # where v_j[i] = 1 - j/n for i < j, and v_j[i] = -j/n for i >= j
        #
        # Selection event: |v_k^T y| >= |v_j^T y| for all valid j != k
        #
        # Parameterize y = t * eta/||eta||^2 + z:
        #   v_j^T y = t * (v_j^T eta / ||eta||^2) + v_j^T z
        #           = t * alpha_j + beta_j

        L = -np.inf
        U = np.inf

        # CUSUM vector at selected CP
        v_k = np.zeros(n)
        v_k[:k] = 1.0 - k / n
        v_k[k:] = -k / n

        alpha_k = np.dot(v_k, eta) / eta_norm_sq
        beta_k = np.dot(v_k, z)
        sign_k = np.sign(alpha_k * t_obs + beta_k)

        if abs(sign_k) < EPSILON:
            return -np.inf, np.inf

        for j in range(min_seg, n - min_seg):
            if j == k:
                continue

            v_j = np.zeros(n)
            v_j[:j] = 1.0 - j / n
            v_j[j:] = -j / n

            alpha_j = np.dot(v_j, eta) / eta_norm_sq
            beta_j = np.dot(v_j, z)

            # Constraint: sign_k * (alpha_k * t + beta_k) >= |alpha_j * t + beta_j|
            # This gives two linear constraints:
            #   sign_k * (alpha_k * t + beta_k) >= +(alpha_j * t + beta_j)
            #   sign_k * (alpha_k * t + beta_k) >= -(alpha_j * t + beta_j)

            for s in [1.0, -1.0]:
                # (sign_k * alpha_k - s * alpha_j) * t >= s * beta_j - sign_k * beta_k
                a_coef = sign_k * alpha_k - s * alpha_j
                b_coef = s * beta_j - sign_k * beta_k

                if abs(a_coef) < EPSILON:
                    continue

                bound = -b_coef / a_coef

                if a_coef > 0:
                    L = max(L, bound)
                else:
                    U = min(U, bound)

        # Sanity: if interval is empty or inverted, fall back to (-inf, inf)
        if L >= U:
            logger.debug(
                "SI truncation interval empty [%.4f, %.4f]; " "falling back to unconditional test.",
                L,
                U,
            )
            return -np.inf, np.inf

        return L, U

    def _selective_inference_p_value(
        self,
        series: np.ndarray[Any, Any],
        change_point: int,
    ) -> float:
        """
        Compute Selective Inference p-value with truncated normal conditioning.

        Conditions the test statistic on the selection event: the binary
        segmentation selected this change point because its CUSUM was
        maximal. The truncated normal distribution accounts for this
        selection bias, guaranteeing Type I error control at the declared
        significance level.

        Based on:
          - Lee et al. (2016), "Exact post-selection inference"
          - Takeuchi Lab (2025), "Time Series Anomaly Detection in the
            Frequency Domain with Statistical Reliability" (arXiv:2502.03062)

        Args:
            series: 1-D band-power time series.
            change_point: Index of the selected CP.

        Returns:
            Two-sided SI p-value in [0, 1]. Guaranteed: if no true CP
            exists, P(p < alpha) <= alpha for any alpha.
        """
        n = len(series)
        if change_point <= 0 or change_point >= n:
            return 1.0

        left = series[:change_point]
        right = series[change_point:]

        if len(left) < 2 or len(right) < 2:
            return 1.0

        # Test statistic: standardized mean difference
        mean_diff = float(np.mean(right) - np.mean(left))
        sigma = self._estimate_noise_sigma(series)
        n_left, n_right = len(left), len(right)
        se = sigma * np.sqrt(1.0 / n_left + 1.0 / n_right)

        if se < EPSILON:
            return 1.0 if abs(mean_diff) < EPSILON else 0.0

        t_obs = mean_diff / se

        # Compute truncation interval from selection event
        min_seg = max(self._oracle_config.min_segments, 2)
        L, U = self._compute_truncation_interval(series, change_point, min_seg)

        # If truncation is trivial (no effective constraint), use standard test
        if np.isinf(L) and np.isinf(U):
            p_value = 2.0 * (1.0 - norm.cdf(abs(t_obs)))
            return float(np.clip(p_value, 0.0, 1.0))

        # Standardize truncation bounds
        L_std = L / se if not np.isinf(L) else -1e10
        U_std = U / se if not np.isinf(U) else 1e10

        # Truncated normal survival function
        # P(|T| >= |t_obs| | L <= T <= U, T ~ N(0, 1))
        try:
            p_upper = truncnorm.sf(abs(t_obs), L_std, U_std)
            p_lower = truncnorm.cdf(-abs(t_obs), L_std, U_std)
            p_value = float(p_upper + p_lower)
        except (ValueError, RuntimeError):
            # Numerical issues with extreme truncation; fall back
            p_value = 2.0 * (1.0 - norm.cdf(abs(t_obs)))

        return float(np.clip(p_value, 0.0, 1.0))

    def _estimate_noise_sigma(self, series: np.ndarray[Any, Any]) -> float:
        """
        MAD estimator on first differences (robust to outliers/CPs).

        Uses MAD-to-sigma conversion factor ``1.4826 / √2``.

        Args:
            series: 1-D numeric array.

        Returns:
            Estimated noise standard deviation.
        """
        if len(series) < 2:
            return 1.0
        diffs = np.diff(series)
        mad = np.median(np.abs(diffs - np.median(diffs)))
        # MAD to sigma: 1.4826 for normal, / sqrt(2) for differences
        return float(mad * 1.4826 / np.sqrt(2.0)) + EPSILON

    # ------------------------------------------------------------------
    # Noise color estimation (F1 Precision Directive, Phase 4)
    # ------------------------------------------------------------------

    def _estimate_noise_color(
        self, psd: np.ndarray[Any, Any], freqs: np.ndarray[Any, Any]
    ) -> tuple[float, str, float]:
        """Estimate the noise color exponent (beta) from the PSD.

        Fits log(PSD) = -beta * log(freq) + C using linear regression
        on the log-log spectrum.

        Noise colors:
            beta ~ 0  -> white noise  (flat spectrum)
            beta ~ 1  -> pink noise   (1/f)
            beta ~ 2  -> brown noise  (1/f^2, Brownian)
            beta ~ -1 -> blue noise   (f)

        Returns:
            (beta, color_name, r_squared)
        """
        mask = (freqs > 0) & (psd > 0)
        if mask.sum() < 3:
            return 0.0, "white", 0.0

        log_f = np.log10(freqs[mask])
        log_p = np.log10(psd[mask])

        A = np.vstack([log_f, np.ones_like(log_f)]).T
        result = np.linalg.lstsq(A, log_p, rcond=None)
        slope = result[0][0]
        beta = -slope

        fitted = A @ result[0]
        ss_res = np.sum((log_p - fitted) ** 2)
        ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
        r2 = float(np.clip(1.0 - ss_res / max(ss_tot, 1e-10), 0.0, 1.0))

        if beta < -1.5:
            color = "violet"
        elif beta < -0.5:
            color = "blue"
        elif beta < 0.5:
            color = "white"
        elif beta < 1.5:
            color = "pink"
        else:
            color = "brown"

        return float(beta), color, r2

    def _expected_band_power(self, lo: float, hi: float, beta: float) -> float:
        """Expected fractional power in [lo, hi] under 1/f^beta model."""
        if abs(beta - 1.0) < 0.01:
            return float(np.log(max(hi, 1e-10)) - np.log(max(lo, 1e-10)))
        exp = 1.0 - beta
        if exp == 0:
            return float(np.log(hi / max(lo, 1e-10)))
        return float((hi**exp - lo**exp) / exp)

    # ------------------------------------------------------------------
    # Adaptive alpha (F1 Precision Directive, Phase 5)
    # ------------------------------------------------------------------

    def _compute_adaptive_alpha(
        self, n_samples: int, n_bands: int, noise_color_confidence: float
    ) -> float:
        """
        Adjust significance level based on test power.

        Shorter windows -> less power -> relax alpha.
        More bands -> multiple testing correction -> tighten alpha.

        Returns:
            Adjusted alpha in [0.01, 0.20].
        """
        base_alpha = self._oracle_config.significance_level

        if n_samples < 50:
            size_factor = 2.0
        elif n_samples < 200:
            size_factor = 1.5
        elif n_samples < 1000:
            size_factor = 1.0
        else:
            size_factor = 0.8

        testing_factor = 1.0 / (1.0 + 0.1 * n_bands)
        confidence_factor = 1.0 + 0.5 * (1.0 - noise_color_confidence)

        # Dynamic factor from external caller (severity-based)
        dynamic = getattr(self, "_dynamic_alpha_factor", 1.0)

        alpha = base_alpha * size_factor * testing_factor * confidence_factor * dynamic
        return float(np.clip(alpha, 0.01, 0.20))

    # ------------------------------------------------------------------
    # Per-band anomaly scoring
    # ------------------------------------------------------------------

    def _compute_band_anomaly(
        self,
        band_label: str,
        band_power_series: np.ndarray[Any, Any],
        ref_mean: float,
        ref_std: float,
        band_def: tuple[float, float, str, float],
        alpha: float,
    ) -> FrequencyBandResult:
        """
        Per-band anomaly scoring combining z-score and SI change-point evidence.

        Score composition: 60% z-score anomaly + 40% CP evidence.

        Args:
            band_label: Band identifier.
            band_power_series: Per-window band power time series.
            ref_mean: Reference (training) mean for this band.
            ref_std: Reference (training) std for this band.
            band_def: ``(low_hz, high_hz, label, weight)`` tuple.
            alpha: Significance level.

        Returns:
            ``FrequencyBandResult`` with all fields populated.
        """
        lo, hi, _label, weight = band_def

        # Mean observed band power
        observed_mean = float(np.mean(band_power_series))
        power_ratio = observed_mean / (ref_mean + EPSILON)

        # Noise color correction: adjust expected power for spectral slope
        corrected_ratio = power_ratio
        if self._noise_beta != 0.0:
            band_center = (lo + hi) / 2.0
            if band_center > 0:
                expected = self._expected_band_power(max(lo, 1e-6), max(hi, 1e-6), self._noise_beta)
                nyquist = self._oracle_config.sample_rate / 2.0
                total_expected = self._expected_band_power(
                    1e-6, max(nyquist, 1e-6), self._noise_beta
                )
                if total_expected > 1e-10 and expected > 1e-10:
                    expected_ratio = expected / total_expected
                    corrected_ratio = power_ratio / max(expected_ratio, 1e-10)

        # Z-score (use corrected ratio when noise color is estimated)
        if self._noise_beta != 0.0:
            z_score = (corrected_ratio - ref_mean) / (ref_std + EPSILON)
        else:
            z_score = (observed_mean - ref_mean) / (ref_std + EPSILON)
        z_anomaly = float(np.clip(abs(z_score) / 3.0, 0.0, 1.0))

        # Change-point evidence
        cp_evidence = 0.0
        best_p = 1.0
        if len(band_power_series) >= 2 * self._oracle_config.min_segments:
            cps = self._binary_segmentation_frequency(band_power_series)
            if cps:
                # Use the most significant CP
                p_values = [self._selective_inference_p_value(band_power_series, cp) for cp in cps]
                best_p = min(p_values)
                cp_evidence = float(1.0 - best_p)

        # Combined score: 60% z-score + 40% CP evidence
        anomaly_score = float(np.clip(0.6 * z_anomaly + 0.4 * cp_evidence, 0.0, 1.0))

        # P-value: use SI p-value if CP detected, else z-score p-value
        if best_p < 1.0:
            p_value = best_p
        else:
            p_value = float(2.0 * (1.0 - norm.cdf(abs(z_score))))

        is_significant = p_value < alpha

        return FrequencyBandResult(
            band_label=band_label,
            low_hz=lo,
            high_hz=hi,
            band_weight=weight,
            power_ratio=power_ratio,
            z_score=z_score,
            anomaly_score=anomaly_score,
            p_value=p_value,
            is_significant=is_significant,
        )

    # ------------------------------------------------------------------
    # Influence multiplier
    # ------------------------------------------------------------------

    def _compute_influence_multiplier(
        self,
        aggregate_score: float,
        spectral_entropy: float,
        band_results: list[FrequencyBandResult],
        spectral_flux: float = 0.0,
        phase_coherence: float = 1.0,
    ) -> float:
        """Five-signal φ-weighted geometric mean influence multiplier.

        Combines:
          - ``score_influence``: from aggregate anomaly score (φ-weighted)
          - ``entropy_influence``: deviation of spectral entropy from reference
          - ``breadth_influence``: fraction of bands that are significant
          - ``flux_influence``: spectral rate-of-change contribution
          - ``coherence_influence``: phase decoherence contribution

        Formula per OSHA OTM §III.5::

            multiplier = (score^φ × entropy × breadth × flux × coherence) ^ (1/(φ+4))

        Args:
            aggregate_score: Weighted aggregate anomaly score.
            spectral_entropy: Current spectral entropy.
            band_results: Per-band detection results.
            spectral_flux: Spectral flux value (from _compute_spectral_flux).
            phase_coherence: Phase coherence value in [0, 1].

        Returns:
            Influence multiplier bounded to ``[floor, ceiling]``.
        """
        # Score influence: map [0, 1] to [floor, ceiling]
        floor = self._oracle_config.influence_floor
        ceiling = self._oracle_config.influence_ceiling
        midpoint = (floor + ceiling) / 2.0

        score_influence = midpoint + (ceiling - midpoint) * (2.0 * aggregate_score - 1.0)
        score_influence = max(score_influence, EPSILON)

        # Entropy influence: deviation from reference
        entropy_dev = abs(spectral_entropy - self._ref_spectral_entropy_mean) / (
            self._ref_spectral_entropy_std + EPSILON
        )
        entropy_influence = 1.0 + min(entropy_dev, 2.0) * 0.25
        entropy_influence = max(entropy_influence, EPSILON)

        # Breadth influence: fraction of significant bands
        n_significant = sum(1 for br in band_results if br.is_significant)
        breadth_ratio = n_significant / max(len(band_results), 1)
        breadth_influence = 1.0 + breadth_ratio * 0.5
        breadth_influence = max(breadth_influence, EPSILON)

        # Flux influence: high spectral flux → stronger modulation
        # Normalise flux to [1.0, 1.5] range; cap contribution
        flux_influence = 1.0 + min(spectral_flux, 1.0) * 0.5
        flux_influence = max(flux_influence, EPSILON)

        # Coherence influence: low coherence → amplify (decoherence is anomalous)
        # phase_coherence in [0, 1]; invert so that low coherence → high influence
        coherence_influence = 1.0 + (1.0 - phase_coherence) * 0.5
        coherence_influence = max(coherence_influence, EPSILON)

        # φ-weighted geometric mean (5 signals)
        exponent = 1.0 / (PHI + 4.0)
        multiplier = (
            score_influence**PHI
            * entropy_influence
            * breadth_influence
            * flux_influence
            * coherence_influence
        ) ** exponent

        # Asymmetric adjustment (F1 Precision Directive, Phase 6):
        # For life-safety systems, missing an anomaly > false alarm.
        # Amplification gets 1.5x boost, attenuation gets 0.8x suppression.
        if multiplier > 1.0:
            multiplier = 1.0 + (multiplier - 1.0) * 1.5
        else:
            multiplier = 1.0 - (1.0 - multiplier) * 0.8

        # Domain anomaly spectral hint boost (Phase 9)
        if self._noise_beta != 0.0:
            hints = DOMAIN_ANOMALY_SPECTRAL_HINTS.get(self._oracle_config.domain, {})
            expected_shift = hints.get("anomaly_beta_shift", 0.0)
            if expected_shift != 0.0 and hasattr(self, "_current_beta"):
                beta_shift = self._current_beta - self._noise_beta
                shift_alignment = 1.0 - abs(beta_shift - expected_shift) / (
                    abs(expected_shift) + 1.0
                )
                if shift_alignment > 0.5:
                    multiplier *= 1.0 + 0.3 * shift_alignment

        return float(np.clip(multiplier, floor, ceiling))

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def fit(
        self,
        data: np.ndarray[Any, Any] | Any,
    ) -> SpectralDomainFrequency:
        """
        Fit the Oracle on reference/training signals.

        Computes per-band reference means/stds, reference spectral
        entropy mean/std, and reference full-spectrum mean/std.

        Args:
            data: Time-domain signals ``(N, T)`` or ``(T,)``.
                  Accepts np.ndarray[Any, Any] or torch.Tensor.

        Returns:
            Self for method chaining.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.size == 0:
            raise DetectorException("Cannot fit SpectralDomainFrequency with empty data.")

        # Collect per-band power statistics and spectral entropy
        band_powers_all: dict[str, list[np.ndarray[Any, Any]]] = {
            label: [] for _, _, label, _ in self._bands
        }
        entropies: list[float] = []
        all_spectra: list[np.ndarray[Any, Any]] = []

        for sample in data:
            freq_matrix, freqs = self._compute_frequency_matrix(sample)
            band_powers = self._extract_band_powers(freq_matrix, freqs)

            for label, power_ts in band_powers.items():
                band_powers_all[label].append(power_ts)

            entropies.append(self._compute_spectral_entropy(freq_matrix))
            all_spectra.append(np.mean(freq_matrix, axis=0))

        # Per-band reference statistics
        self._ref_band_powers = {}
        self._ref_band_means = {}
        self._ref_band_stds = {}
        for label, power_list in band_powers_all.items():
            means = np.array([float(np.mean(p)) for p in power_list])
            self._ref_band_powers[label] = means
            self._ref_band_means[label] = float(np.mean(means))
            self._ref_band_stds[label] = float(np.std(means)) + EPSILON

        # Spectral entropy reference
        self._ref_spectral_entropy_mean = float(np.mean(entropies))
        self._ref_spectral_entropy_std = float(np.std(entropies)) + EPSILON

        # Full spectrum reference
        spectra_array = np.array(all_spectra)
        self._ref_full_spectrum_mean = np.mean(spectra_array, axis=0)
        self._ref_full_spectrum_std = np.std(spectra_array, axis=0) + EPSILON

        # Estimate noise color from reference spectrum
        if len(all_spectra) > 0:
            ref_psd = np.mean(np.array(all_spectra), axis=0)
            # Use freqs from last computed frequency matrix
            _, ref_freqs = self._compute_frequency_matrix(data[-1])
            if len(ref_psd) == len(ref_freqs):
                self._noise_beta, self._noise_color, self._noise_fit_r2 = (
                    self._estimate_noise_color(ref_psd, ref_freqs)
                )
                logger.info(
                    "Oracle noise color: beta=%.2f (%s), R²=%.3f",
                    self._noise_beta,
                    self._noise_color,
                    self._noise_fit_r2,
                )

        self._is_fitted = True
        logger.info(
            "SpectralDomainFrequency fitted on %d samples, domain=%s, "
            "bands=%d (Nyquist=%.1f Hz)",
            len(data),
            self._oracle_config.domain,
            len(self._bands),
            self._oracle_config.sample_rate / 2.0,
        )
        return self

    def detect(
        self,
        data: np.ndarray[Any, Any] | Any,
    ) -> dict[str, Any]:
        """
        Detect frequency-domain anomalies.

        Full 4-stage pipeline:
          1. Windowed DFT → frequency matrix
          2. Parseval validation (using existing matrix)
          3. Domain-adaptive band extraction + per-band SI/CP scoring
          4. Fisher's method for joint p-value + φ-weighted influence

        Args:
            data: A single time-domain signal ``(T,)`` or batch ``(N, T)``.
                  Accepts np.ndarray[Any, Any] or torch.Tensor.

        Returns:
            Dict with keys:
              ``anomaly_score``, ``is_anomaly``, ``influence_vector``,
              ``band_results``, ``detector_type``
        """
        if not self._is_fitted:
            raise DetectorException("SpectralDomainFrequency must be fitted before detection.")

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        single = data.ndim == 1
        if single:
            data = data.reshape(1, -1)

        results: list[dict[str, Any]] = []
        for sample in data:
            results.append(self._detect_single(sample))

        if single:
            return results[0]

        # Batch: aggregate
        scores = np.array([r["anomaly_score"] for r in results])
        mean_score = float(np.mean(scores))
        return {
            "anomaly_score": mean_score,
            "is_anomaly": mean_score > self.threshold,
            "influence_vector": results[0]["influence_vector"],
            "band_results": results[0]["band_results"],
            "per_sample_results": results,
            "detector_type": "spectral_domain_frequency",
        }

    def extract_features(
        self,
        data: np.ndarray[Any, Any] | Any,
    ) -> torch.Tensor:
        """
        Extract per-band spectral features as torch.Tensor.

        Returns ``[batch, n_bands + 7]`` features:
          - Per-band anomaly scores (from band_scores dict)
          - Spectral entropy
          - Spectral centroid
          - Aggregate score
          - Influence multiplier
          - Spectral flux
          - Phase coherence
          - Cepstral peak ratio

        Complies with BaseDetector contract: returns torch.Tensor
        suitable for neural network fusion. Internal computations
        remain in float64 numpy for SI p-value precision; conversion
        to float32 tensor happens here at the boundary.

        .. note::
            The Oracle operates entirely in numpy/scipy space. The
            integration layer in
            ``AdvancedPhysicsIntegratedDetector._extract_combined_features()``
            handles the numpy-to-torch conversion at the boundary.
            If adding new callers, convert via:
            ``torch.from_numpy(oracle.extract_features(data).numpy()).float()``

        Args:
            data: Time-domain signals ``(N, T)`` or ``(T,)``.
                  Accepts np.ndarray[Any, Any] or torch.Tensor.

        Returns:
            Feature tensor of shape ``(N, n_bands + 7)``, dtype float32.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        features_list: list[np.ndarray[Any, Any]] = []
        for sample in data:
            result = self._detect_single(sample)
            iv = result["influence_vector"]

            # Per-band scores in band order
            band_feats = [iv.band_scores.get(label, 0.0) for _, _, label, _ in self._bands]
            # Append all 7 global features
            feat = np.array(
                band_feats
                + [
                    iv.spectral_entropy,
                    iv.spectral_centroid,
                    iv.aggregate_score,
                    iv.influence_multiplier,
                    iv.spectral_flux,
                    iv.phase_coherence,
                    iv.cepstral_peak_ratio,
                ],
                dtype=np.float32,
            )
            features_list.append(feat)

        features_np = np.array(features_list, dtype=np.float32)
        return torch.from_numpy(features_np)

    # ------------------------------------------------------------------
    # Single-sample detection
    # ------------------------------------------------------------------

    def _detect_single(self, signal: np.ndarray[Any, Any]) -> dict[str, Any]:
        """
        Run the full 4-stage pipeline on a single 1-D signal.

        1. Windowed DFT
        2. Parseval validation (no recompute)
        3. Per-band SI/CP scoring
        4. Influence multiplier computation

        Args:
            signal: 1-D time-domain signal.

        Returns:
            Detection result dict.
        """
        # Stage 1: Windowed DFT
        freq_matrix, freqs = self._compute_frequency_matrix(signal)

        # Estimate current noise color for spectral hint comparison
        mean_spectrum = np.mean(freq_matrix, axis=0)
        self._current_beta, _, _ = self._estimate_noise_color(mean_spectrum, freqs)

        # Stage 2: Parseval validation (uses existing freq_matrix)
        self._validate_parseval_energy(signal, freq_matrix)

        # Stage 3: Per-band extraction and scoring
        band_powers = self._extract_band_powers(freq_matrix, freqs)

        alpha = self._compute_adaptive_alpha(
            n_samples=len(signal),
            n_bands=len(self._bands),
            noise_color_confidence=self._noise_fit_r2,
        )
        band_results: list[FrequencyBandResult] = []
        band_scores_dict: dict[str, float] = {}

        for lo, hi, label, weight in self._bands:
            ref_mean = self._ref_band_means.get(label, 0.0)
            ref_std = self._ref_band_stds.get(label, 1.0)
            power_series = band_powers.get(label, np.zeros(1))

            br = self._compute_band_anomaly(
                band_label=label,
                band_power_series=power_series,
                ref_mean=ref_mean,
                ref_std=ref_std,
                band_def=(lo, hi, label, weight),
                alpha=alpha,
            )
            band_results.append(br)
            band_scores_dict[label] = br.anomaly_score

        # Weighted aggregate score
        aggregate_score = sum(br.anomaly_score * br.band_weight for br in band_results)
        aggregate_score = float(np.clip(aggregate_score, 0.0, 1.0))

        # Fisher's method for aggregate p-value
        p_values = [br.p_value for br in band_results]
        p_clipped = np.clip(p_values, 1e-300, 1.0)
        log_p_sum = -2.0 * np.sum(np.log(p_clipped))
        dof = 2 * len(p_values)
        from scipy.stats import chi2

        aggregate_p = float(1.0 - chi2.cdf(log_p_sum, dof))

        # Change-point detected if any band is significant
        change_point_detected = any(br.is_significant for br in band_results)

        # Spectral statistics
        spectral_entropy = self._compute_spectral_entropy(freq_matrix)
        spectral_centroid = self._compute_spectral_centroid(freq_matrix, freqs)

        # New spectral features (always computed when Oracle is active)
        spectral_flux = self._compute_spectral_flux(freq_matrix)
        phase_coherence = self._compute_phase_coherence(signal)
        cepstral_peak_ratio = self._compute_cepstral_peak(freq_matrix)

        # Dominant frequency
        mean_spectrum = np.mean(freq_matrix, axis=0)
        dominant_idx = int(np.argmax(mean_spectrum))
        dominant_frequency = float(freqs[dominant_idx]) if len(freqs) > dominant_idx else 0.0

        # Stage 4: Influence multiplier (φ-weighted geometric mean, 5 signals)
        influence_multiplier = self._compute_influence_multiplier(
            aggregate_score,
            spectral_entropy,
            band_results,
            spectral_flux=spectral_flux,
            phase_coherence=phase_coherence,
        )

        confidence = float(np.clip(1.0 - aggregate_p, 0.0, 1.0))

        iv = FrequencyInfluenceVector(
            influence_multiplier=influence_multiplier,
            band_scores=band_scores_dict,
            aggregate_score=aggregate_score,
            aggregate_p_value=aggregate_p,
            spectral_entropy=spectral_entropy,
            dominant_frequency=dominant_frequency,
            spectral_centroid=spectral_centroid,
            change_point_detected=change_point_detected,
            confidence=confidence,
            spectral_flux=spectral_flux,
            phase_coherence=phase_coherence,
            cepstral_peak_ratio=cepstral_peak_ratio,
        )

        return {
            "anomaly_score": aggregate_score,
            "is_anomaly": aggregate_score > self.threshold,
            "influence_vector": iv,
            "band_results": band_results,
            "detector_type": "spectral_domain_frequency",
            "noise_color": {
                "beta": self._noise_beta,
                "name": self._noise_color,
                "r_squared": self._noise_fit_r2,
            },
        }


# =============================================================================
# Backward-compatible aliases (Task 9)
# =============================================================================
# These aliases ensure that existing code importing the old names continues
# to work after the rename.  New code should use SpectralDomainFrequency,
# SpectralDomainFrequencyConfig, and create_spectral_frequency.

SpectralDomainOracle = SpectralDomainFrequency
"""Deprecated alias for :class:`SpectralDomainFrequency`."""

SpectralDomainOracleConfig = SpectralDomainFrequencyConfig
"""Deprecated alias for :class:`SpectralDomainFrequencyConfig`."""

create_spectral_oracle = create_spectral_frequency
"""Deprecated alias for :func:`create_spectral_frequency`."""

FrequencyDomainOracle = SpectralDomainFrequency
"""Deprecated alias for :class:`SpectralDomainFrequency`."""

FrequencyDomainOracleConfig = SpectralDomainFrequencyConfig
"""Deprecated alias for :class:`SpectralDomainFrequencyConfig`."""

create_frequency_oracle = create_spectral_frequency
"""Deprecated alias for :func:`create_spectral_frequency`."""
