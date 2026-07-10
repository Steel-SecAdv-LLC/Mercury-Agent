# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hardware-RNG health monitoring via REG deviation statistics (advisory).

This is the fault-detection application of the REG statistical-deviation
machinery in :mod:`omni_mercury_engine.models.gcp_ingest` /
:mod:`omni_mercury_engine.models.parapsychology`: raw bytes from ANY RNG
(e.g. the crypto/PQC layer's entropy source) are packed into 200-bit trial
sums -- Binomial(200, 0.5) under the fair-coin null, the exact null of the
Global Consciousness Project egg trials -- and the same pre-registered
closed-form statistics are run over them:

* **Stouffer Z** over trial z-scores -> mean bias (0/1 imbalance);
* **chi-square variance tail** (Wilson-Hilferty standardized) -> variance
  inflation/deficit (correlated or stuck bits);
* **lag-1 serial-correlation z** -> time correlation between trials
  (oscillation, common-mode coupling, buffer reuse).

Spirit of NIST SP 800-90B section 4 ("Health Tests"): entropy sources must
run continuous tests (Repetition Count, Adaptive Proportion) that catch
catastrophic noise-source failures during operation. This monitor is a
statistical supplement in that spirit -- an ADVISORY tool, deliberately NOT
wired into any runtime crypto path: it renders a typed verdict for
operators and test harnesses, and consuming code decides what to do.

Detection power (documented, matches the training pipeline's bias channel):
a 0->1 bit-flip fault with probability ``q`` shifts each trial mean by
``(200 - E[v]) * q`` (= ``100q`` under the null), so the Stouffer statistic
drifts as ``100q * sqrt(N) / sqrt(50) ~= 14.14 * q * sqrt(N)`` over ``N``
trials -- e.g. q=0.01 exceeds the default z=4 threshold with near-certainty
by N ~= 4000 trials (100 KB of stream).

Pure stdlib + numpy. Optionally, a trained
:class:`~omni_mercury_engine.models.parapsychology.ParapsychologyDetector`
can be injected to report the learned ``reg_deviation_gcp`` score alongside
the closed-form verdict (advisory only; it never changes the verdict).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

TRIAL_BITS = 200
TRIAL_BYTES = TRIAL_BITS // 8  # 25 bytes per trial
NULL_MEAN = TRIAL_BITS / 2.0  # 100
NULL_STD = math.sqrt(TRIAL_BITS / 4.0)  # sqrt(50)

#: Fewer trials than this cannot support the chi-square normal approximation
#: or a meaningful autocorrelation estimate; assess() refuses loudly.
MIN_TRIALS = 40


class RngHealthVerdict(StrEnum):
    """Typed verdict for an assessed RNG stream."""

    HEALTHY = "healthy"
    BIAS_SUSPECT = "bias-suspect"
    CORRELATION_SUSPECT = "correlation-suspect"


@dataclass(frozen=True)
class RngHealthReport:
    """Outcome of one RNG stream assessment (all z-scores ~N(0,1) under null).

    Attributes:
        verdict: Typed verdict (bias takes precedence over correlation).
        n_trials: Number of 200-bit trials assessed.
        stouffer_z: Mean-bias statistic, ``sum(z_i)/sqrt(N)``.
        variance_z: Wilson-Hilferty-standardized chi-square variance tail.
        lag1_autocorr_z: Lag-1 serial correlation times ``sqrt(N)``.
        z_threshold: |z| flag threshold the verdict used.
        detector_score: Mean learned deviation score in [0, 1] from an
            injected trained detector, or None (advisory; never affects the
            verdict).
        note: Human-readable summary of what fired.
    """

    verdict: RngHealthVerdict
    n_trials: int
    stouffer_z: float
    variance_z: float
    lag1_autocorr_z: float
    z_threshold: float
    detector_score: float | None
    note: str


def bytes_to_trial_sums(raw: bytes) -> np.ndarray:
    """Pack raw RNG bytes into 200-bit trial sums (Binomial(200, 0.5) null).

    Trailing bytes that do not fill a whole 25-byte trial are discarded.

    Raises:
        ValueError: If the stream holds fewer than :data:`MIN_TRIALS` trials.
    """
    n_trials = len(raw) // TRIAL_BYTES
    if n_trials < MIN_TRIALS:
        raise ValueError(
            f"need at least {MIN_TRIALS * TRIAL_BYTES} bytes "
            f"({MIN_TRIALS} trials of {TRIAL_BYTES} bytes); got {len(raw)} bytes"
        )
    bits = np.unpackbits(np.frombuffer(raw[: n_trials * TRIAL_BYTES], dtype=np.uint8))
    return bits.reshape(n_trials, TRIAL_BITS).sum(axis=1).astype(np.float64)


def bit_bias_channel(raw: bytes, q: float, seed: int) -> bytes:
    """Reference 0->1 bit-flip fault channel (for validation and tests).

    Each 0-bit of ``raw`` flips to 1 with probability ``q`` -- the same
    channel the ``reg_deviation_gcp`` training pipeline injects at the
    trial-sum level (``v' = v + Binomial(200 - v, q)``), here applied to the
    physical bits via a seeded OR mask. NOT random hardware: a documented,
    deterministic-given-seed fault model.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    rng = np.random.default_rng(seed)
    mask_bits = (rng.random(len(raw) * 8) < q).astype(np.uint8)
    mask = np.packbits(mask_bits)
    data = np.frombuffer(raw, dtype=np.uint8)
    return (data | mask).tobytes()


def _wilson_hilferty_z(chi2: float, df: int) -> float:
    """Standardize a chi-square value to ~N(0,1) (Wilson-Hilferty cube root)."""
    if df <= 0:
        return 0.0
    c = 2.0 / (9.0 * df)
    return float(((chi2 / df) ** (1.0 / 3.0) - (1.0 - c)) / math.sqrt(c))


class RngHealthMonitor:
    """Advisory health monitor for raw RNG output streams.

    Runs the closed-form REG deviation statistics (module docstring) over a
    byte stream and renders a typed verdict. Deliberately NOT wired into
    runtime crypto paths -- callers (operators, CI, the crypto layer's own
    diagnostics) consume the report and decide.

    Args:
        z_threshold: |z| at which a statistic is flagged. The default 4.0
            corresponds to a per-statistic two-sided false-alarm rate of
            ~6.3e-5 (~1.9e-4 across the three statistics, Bonferroni),
            trading alarm rarity against the documented detection power.
        detector: Optional trained ParapsychologyDetector (duck-typed;
            injected, never constructed here). When provided, consecutive
            100-trial windows are scored through its public API and the mean
            learned score is reported -- advisory only.
    """

    def __init__(self, *, z_threshold: float = 4.0, detector: Any | None = None) -> None:
        """Initialize the monitor (see class docstring for arguments)."""
        if z_threshold <= 0:
            raise ValueError(f"z_threshold must be positive, got {z_threshold}")
        self.z_threshold = float(z_threshold)
        self.detector = detector

    def assess(self, raw: bytes) -> RngHealthReport:
        """Assess an RNG byte stream and return a typed verdict.

        Verdict rule (documented, deterministic): ``bias-suspect`` when the
        mean-bias |Stouffer Z| crosses the threshold; else
        ``correlation-suspect`` when the variance or lag-1 serial statistic
        crosses it; else ``healthy``.

        Raises:
            ValueError: If ``raw`` holds fewer than :data:`MIN_TRIALS`
                trials (an unmeasurable stream must refuse, not pass).
        """
        sums = bytes_to_trial_sums(raw)
        z = (sums - NULL_MEAN) / NULL_STD
        n = len(z)

        stouffer = float(z.sum() / math.sqrt(n))
        variance_z = _wilson_hilferty_z(float((z**2).sum()), n)
        centered = z - z.mean()
        denom = float((centered**2).sum())
        r1 = float((centered[:-1] * centered[1:]).sum() / denom) if denom > 0 else 0.0
        lag1_z = r1 * math.sqrt(n)

        detector_score = self._detector_score(z)

        if abs(stouffer) >= self.z_threshold:
            verdict = RngHealthVerdict.BIAS_SUSPECT
            note = f"mean bias: |Stouffer Z|={abs(stouffer):.2f} >= {self.z_threshold}"
        elif max(abs(variance_z), abs(lag1_z)) >= self.z_threshold:
            verdict = RngHealthVerdict.CORRELATION_SUSPECT
            note = (
                f"variance/serial structure: |variance z|={abs(variance_z):.2f}, "
                f"|lag-1 z|={abs(lag1_z):.2f} vs threshold {self.z_threshold}"
            )
        else:
            verdict = RngHealthVerdict.HEALTHY
            note = f"all statistics within |z| < {self.z_threshold}"

        return RngHealthReport(
            verdict=verdict,
            n_trials=n,
            stouffer_z=stouffer,
            variance_z=variance_z,
            lag1_autocorr_z=lag1_z,
            z_threshold=self.z_threshold,
            detector_score=detector_score,
            note=note,
        )

    def _detector_score(self, z: np.ndarray) -> float | None:
        """Mean learned deviation score over 100-trial windows (advisory).

        Returns None when no detector is injected, when the stream is too
        short for a full window, or when the detector abstains (untrained
        quarantine returns no usable signal and is not reported as one).
        """
        if self.detector is None or len(z) < 100:
            return None
        if not bool(getattr(self.detector, "_neural_trained", False)):
            return None  # quarantined/untrained: 0.5 abstention is not a score
        scores = []
        for start in range(0, len(z) - 99, 100):
            result = self.detector.detect_psi_anomaly({"reg_output": z[start : start + 100]})
            coherence = getattr(result, "coherence_score", None)
            if coherence is not None:
                scores.append(float(coherence))
        return float(np.mean(scores)) if scores else None


__all__ = [
    "MIN_TRIALS",
    "NULL_MEAN",
    "NULL_STD",
    "TRIAL_BITS",
    "TRIAL_BYTES",
    "RngHealthMonitor",
    "RngHealthReport",
    "RngHealthVerdict",
    "bit_bias_channel",
    "bytes_to_trial_sums",
]
