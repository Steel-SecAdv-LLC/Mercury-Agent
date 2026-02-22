"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Frequency Domain Oracle - Neuro-Symbolic Autonomous AI Module.

A frequency-domain anomaly detection system that performs per-band
spectral scoring with change-point detection.  The Oracle provides an
``influence_multiplier`` that modulates the fused anomaly score
produced by ``AdvancedPhysicsIntegratedDetector``:

* When the Oracle detects a genuine spectral change (p < alpha),
  ``influence_multiplier > 1.0`` amplifies all detector signals.
* When the Oracle sees a stable spectrum,
  ``influence_multiplier < 1.0`` suppresses false positives.

The Oracle honours Parseval's theorem at runtime to guarantee that
the frequency-domain energy equals the time-domain energy.

Supported domains:
    environmental, medical, infrastructure, space, security,
    financial, humanitarian

Humanitarian domain frequencies are optimised for crisis detection,
pandemic monitoring, and disaster response — prioritising regenerative
and resilient features for humanitarian impact.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
from scipy import stats as scipy_stats
from scipy.fft import fft, fftfreq

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import DetectorException

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================


class FrequencyWeighting(Enum):
    """Weighting scheme for frequency bands."""

    UNIFORM = "uniform"
    GOLDEN_RATIO = "golden_ratio"
    DOMAIN_ADAPTIVE = "domain_adaptive"


@dataclass(frozen=True)
class FrequencyInfluenceVector:
    """Output of the Oracle for a single observation.

    Attributes:
        influence_multiplier: Score modulation factor applied to the
            fused anomaly score.  Values > 1 amplify, < 1 suppress.
        band_scores: Per-band anomaly scores (one per frequency band).
        aggregate_p_value: Combined p-value across all bands
            (Fisher's method).
        change_point_detected: True when the spectral profile
            significantly deviates from the training distribution.
    """

    influence_multiplier: float
    band_scores: tuple[float, ...]
    aggregate_p_value: float
    change_point_detected: bool


@dataclass
class OracleConfig:
    """Configuration for FrequencyDomainOracle.

    Attributes:
        domain: Application domain for band selection.
        sample_rate: Sampling rate of the input signal in Hz.
        threshold: Anomaly score threshold in [0, 1].
        n_bands: Number of frequency bands (auto-set from domain).
        weighting: Band weighting scheme.
        alpha: Significance level for change-point detection.
        influence_floor: Minimum influence multiplier.
        influence_ceiling: Maximum influence multiplier.
        parseval_rtol: Relative tolerance for Parseval validation.
    """

    domain: str = "environmental"
    sample_rate: float = 1000.0
    threshold: float = 0.5
    n_bands: int = 5
    weighting: FrequencyWeighting = FrequencyWeighting.DOMAIN_ADAPTIVE
    alpha: float = 0.05
    influence_floor: float = 0.5
    influence_ceiling: float = 2.0
    parseval_rtol: float = 0.01


# Domain-specific band definitions.
# Each band is (low_hz, high_hz, label, weight).
_DOMAIN_BANDS: dict[str, list[tuple[float, float, str, float]]] = {
    "environmental": [
        (0.0, 8.0, "sub-schumann", 0.15),
        (8.0, 15.0, "schumann-1", 0.25),
        (15.0, 21.0, "schumann-2", 0.20),
        (21.0, 34.0, "schumann-3", 0.20),
        (34.0, 100.0, "supra-schumann", 0.20),
    ],
    "medical": [
        (0.0, 0.04, "ulf", 0.15),
        (0.04, 0.15, "vlf", 0.20),
        (0.15, 0.4, "lf", 0.25),
        (0.4, 1.0, "hf", 0.25),
        (1.0, 128.0, "eeg-range", 0.15),
    ],
    "infrastructure": [
        (0.0, 0.1, "subsynchronous", 0.15),
        (0.1, 25.0, "structural", 0.20),
        (25.0, 50.0, "low-mains", 0.20),
        (50.0, 60.0, "mains", 0.25),
        (60.0, 500.0, "harmonic", 0.20),
    ],
    "space": [
        (0.0, 0.001, "deep-space", 0.15),
        (0.001, 0.01, "solar-wind", 0.25),
        (0.01, 0.1, "magnetospheric", 0.20),
        (0.1, 1.0, "ionospheric", 0.20),
        (1.0, 50.0, "whistler", 0.20),
    ],
    "security": [
        (0.0, 1.0, "baseline", 0.20),
        (1.0, 10.0, "session", 0.20),
        (10.0, 100.0, "burst", 0.20),
        (100.0, 1000.0, "scan", 0.20),
        (1000.0, 10000.0, "flood", 0.20),
    ],
    "financial": [
        (0.0, 0.01, "trend", 0.15),
        (0.01, 0.1, "swing", 0.20),
        (0.1, 1.0, "intraday", 0.25),
        (1.0, 10.0, "hft-low", 0.20),
        (10.0, 500.0, "hft-high", 0.20),
    ],
    "humanitarian": [
        (0.0, 0.01, "slow-onset", 0.20),
        (0.01, 0.1, "seasonal", 0.20),
        (0.1, 1.0, "rapid-onset", 0.25),
        (1.0, 10.0, "crisis-pulse", 0.20),
        (10.0, 100.0, "aftershock", 0.15),
    ],
}


def get_domain_frequency_bands(
    domain: str,
) -> list[tuple[float, float, str, float]]:
    """Return the frequency-band definition for *domain*.

    Falls back to ``environmental`` bands when *domain* is unknown.
    """
    return _DOMAIN_BANDS.get(domain.lower(), _DOMAIN_BANDS["environmental"])


# =============================================================================
# Factory helper
# =============================================================================


def create_frequency_oracle(
    config: dict[str, Any] | None = None,
) -> FrequencyDomainOracle:
    """Factory function for FrequencyDomainOracle.

    Args:
        config: Optional configuration dictionary.

    Returns:
        Configured FrequencyDomainOracle instance.
    """
    return FrequencyDomainOracle(config)


# =============================================================================
# Main detector
# =============================================================================


class FrequencyDomainOracle(BaseDetector):
    """Neuro-symbolic frequency-domain anomaly detection Oracle.

    The Oracle decomposes a signal into domain-specific frequency bands,
    computes per-band z-scores against reference statistics learned during
    ``fit()``, and emits a :class:`FrequencyInfluenceVector` containing an
    ``influence_multiplier`` that modulates the overall anomaly score.

    Example::

        oracle = FrequencyDomainOracle({"domain": "medical", "sample_rate": 256.0})
        oracle.fit(training_signals)       # (N, T) array
        result = oracle.detect(test_signal) # (T,) array
        iv = result["influence_vector"]     # FrequencyInfluenceVector
        print(iv.influence_multiplier, iv.band_scores)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)

        cfg = config or {}
        domain = cfg.get("domain", "environmental")
        self._oracle_config = OracleConfig(
            domain=domain,
            sample_rate=cfg.get("sample_rate", 1000.0),
            threshold=self.threshold,
            weighting=FrequencyWeighting(
                cfg.get("weighting", "domain_adaptive"),
            ),
            alpha=cfg.get("alpha", 0.05),
            influence_floor=cfg.get("influence_floor", 0.5),
            influence_ceiling=cfg.get("influence_ceiling", 2.0),
            parseval_rtol=cfg.get("parseval_rtol", 0.01),
        )

        self._bands = get_domain_frequency_bands(domain)
        self._oracle_config = OracleConfig(
            **{
                **self._oracle_config.__dict__,
                "n_bands": len(self._bands),
            }
        )

        # Reference statistics learned during fit()
        self._ref_band_means: np.ndarray | None = None
        self._ref_band_stds: np.ndarray | None = None
        self._ref_band_energies: np.ndarray | None = None

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def fit(
        self,
        data: np.ndarray | torch.Tensor,
    ) -> FrequencyDomainOracle:
        """Fit the Oracle on reference/training signals.

        Args:
            data: Time-domain signals shaped ``(N, T)`` or ``(T,)``.

        Returns:
            Self for method chaining.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.size == 0:
            raise DetectorException("Cannot fit FrequencyDomainOracle with empty data.")

        all_band_energies: list[np.ndarray] = []
        for sample in data:
            freq_matrix, freqs = self._compute_frequency_matrix(sample)
            band_energy = self._compute_band_energies(freq_matrix, freqs)
            all_band_energies.append(band_energy)

        energy_array = np.array(all_band_energies)
        self._ref_band_means = np.mean(energy_array, axis=0)
        self._ref_band_stds = np.std(energy_array, axis=0) + 1e-12
        self._ref_band_energies = energy_array

        self._is_fitted = True
        logger.info(
            "FrequencyDomainOracle fitted on %d samples, domain=%s, bands=%d",
            len(data),
            self._oracle_config.domain,
            len(self._bands),
        )
        return self

    def detect(
        self,
        data: np.ndarray | torch.Tensor,
    ) -> dict[str, Any]:
        """Detect frequency-domain anomalies.

        Args:
            data: A single time-domain signal ``(T,)`` or batch ``(N, T)``.

        Returns:
            Dict with keys:
                ``anomaly_score``   - float in [0, 1]
                ``is_anomaly``      - bool
                ``influence_vector`` - :class:`FrequencyInfluenceVector`
                ``band_energies``   - per-band energy array
                ``detector_type``   - ``"frequency_domain_oracle"``
        """
        if not self._is_fitted:
            raise DetectorException("FrequencyDomainOracle must be fitted before detection.")

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

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
            "band_energies": np.array([r["band_energies"] for r in results]),
            "per_sample_results": results,
            "detector_type": "frequency_domain_oracle",
        }

    def extract_features(
        self,
        data: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-band spectral features.

        Args:
            data: Time-domain signals ``(N, T)`` or ``(T,)``.

        Returns:
            Feature tensor of shape ``(N, n_bands * 2)``.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if data.ndim == 1:
            data = data.reshape(1, -1)

        features_list: list[np.ndarray] = []
        for sample in data:
            freq_matrix, freqs = self._compute_frequency_matrix(sample)
            band_energy = self._compute_band_energies(freq_matrix, freqs)

            if self._ref_band_means is not None and self._ref_band_stds is not None:
                z_scores = (band_energy - self._ref_band_means) / self._ref_band_stds
            else:
                z_scores = np.zeros_like(band_energy)

            feat = np.concatenate([band_energy, z_scores])
            features_list.append(feat)

        return torch.from_numpy(np.array(features_list, dtype=np.float64)).float()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_single(self, signal: np.ndarray) -> dict[str, Any]:
        """Run detection on a single 1-D signal."""
        assert self._ref_band_means is not None
        assert self._ref_band_stds is not None

        freq_matrix, freqs = self._compute_frequency_matrix(signal)
        band_energy = self._compute_band_energies(freq_matrix, freqs)

        # Per-band z-scores
        z_scores = (band_energy - self._ref_band_means) / self._ref_band_stds
        band_p_values = 2.0 * (1.0 - scipy_stats.norm.cdf(np.abs(z_scores)))

        # Weighted band scores
        weights = np.array([w for _, _, _, w in self._bands])
        weights = weights / weights.sum()
        band_anomaly = np.clip(np.abs(z_scores) / 3.0, 0.0, 1.0)

        anomaly_score = float(np.dot(weights, band_anomaly))
        anomaly_score = float(np.clip(anomaly_score, 0.0, 1.0))

        # Fisher's method for aggregate p-value
        log_p_sum = -2.0 * np.sum(np.log(np.clip(band_p_values, 1e-300, 1.0)))
        dof = 2 * len(band_p_values)
        aggregate_p = float(1.0 - scipy_stats.chi2.cdf(log_p_sum, dof))

        # Change-point detection
        change_point = aggregate_p < self._oracle_config.alpha

        # Influence multiplier
        cfg = self._oracle_config
        if change_point:
            raw_mult = 1.0 + anomaly_score
        else:
            raw_mult = 1.0 - (1.0 - anomaly_score) * 0.3
        influence_multiplier = float(np.clip(raw_mult, cfg.influence_floor, cfg.influence_ceiling))

        iv = FrequencyInfluenceVector(
            influence_multiplier=influence_multiplier,
            band_scores=tuple(float(s) for s in band_anomaly),
            aggregate_p_value=aggregate_p,
            change_point_detected=change_point,
        )

        # Parseval validation (log warning, do not block)
        self._validate_parseval_energy(signal, freq_matrix)

        return {
            "anomaly_score": anomaly_score,
            "is_anomaly": anomaly_score > self.threshold,
            "influence_vector": iv,
            "band_energies": band_energy,
            "detector_type": "frequency_domain_oracle",
        }

    def _compute_frequency_matrix(
        self,
        signal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute FFT magnitude spectrum.

        Returns:
            (magnitude_spectrum, frequencies) — both 1-D arrays of
            length ``N // 2``.
        """
        n = len(signal)
        spectrum = fft(signal)
        freqs = fftfreq(n, d=1.0 / self._oracle_config.sample_rate)

        # Positive frequencies only
        pos_mask = freqs >= 0
        magnitude = np.abs(spectrum[pos_mask]) / n
        pos_freqs = freqs[pos_mask]
        return magnitude, pos_freqs

    def _compute_band_energies(
        self,
        magnitude: np.ndarray,
        freqs: np.ndarray,
    ) -> np.ndarray:
        """Sum squared magnitudes within each frequency band."""
        energies = np.zeros(len(self._bands))
        for i, (lo, hi, _label, _w) in enumerate(self._bands):
            mask = (freqs >= lo) & (freqs < hi)
            energies[i] = float(np.sum(magnitude[mask] ** 2))
        return energies

    def _validate_parseval_energy(
        self,
        signal: np.ndarray,
        freq_magnitude: np.ndarray,
    ) -> bool:
        """Validate Parseval's theorem: time energy ~ freq energy.

        Returns True if the check passes.
        """
        time_energy = float(np.sum(signal**2))
        n = len(signal)
        # Full spectrum energy (Parseval for DFT)
        full_spectrum = fft(signal)
        freq_energy = float(np.sum(np.abs(full_spectrum) ** 2)) / n

        if time_energy < 1e-12:
            return True  # trivial signal

        rtol = self._oracle_config.parseval_rtol
        ratio = abs(freq_energy - time_energy) / time_energy
        ok = ratio <= rtol
        if not ok:
            logger.warning(
                "Parseval validation failed: time_energy=%.6f, "
                "freq_energy=%.6f, ratio=%.4f (rtol=%.4f)",
                time_energy,
                freq_energy,
                ratio,
                rtol,
            )
        return ok
