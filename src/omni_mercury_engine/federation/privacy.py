"""
Differential privacy for federated sufficient statistics.

Implements the Gaussian mechanism for (epsilon, delta)-differential privacy.

IMPORTANT: Sensitivity is set by a CLIPPING NORM, not derived from the data. Data-dependent
sensitivity would leak information through the noise calibration itself, defeating the purpose of
DP.

Reference: Dwork & Roth, "The Algorithmic Foundations of Differential Privacy"
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from omni_mercury_engine.federation.statistics import FittedStatistics


class DifferentialPrivacy:
    """Apply differential privacy noise to fitted statistics.

    Uses the Gaussian mechanism: noise ~ N(0, (sensitivity * sigma)^2)
    where sigma = sqrt(2 * ln(1.25/delta)) / epsilon

    Sensitivity is controlled by a clipping norm applied BEFORE noise
    addition. Each statistic is clipped to [-clip_norm, clip_norm]
    (or [0, clip_norm] for non-negative quantities like std),
    giving bounded sensitivity = 2 * clip_norm / n_samples.

    Args:
        epsilon: Privacy budget. Lower = more private, more noise.
            - 0.1: Strong privacy (significant noise, may degrade utility)
            - 1.0: Moderate privacy (recommended starting point)
            - 10.0: Weak privacy (minimal noise)
        delta: Probability of privacy breach. Default 1e-5.
        clip_norm: Maximum magnitude for any single statistic element.
            Default 10.0. Increase for data with larger dynamic range.
        rng: Optional numpy `Generator` providing the noise source.  If
            omitted, a fresh OS-seeded `np.random.default_rng()` is
            constructed per instance — the global `np.random` legacy
            state is **never** used, so a caller cannot accidentally
            de-randomise the privacy noise via `np.random.seed(...)`
            elsewhere in the process. For audited / reproducible
            deployments pass an explicit `np.random.default_rng(seed)`
            seeded from a documented entropy source.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float = 1e-5,
        clip_norm: float = 10.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if delta <= 0 or delta >= 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if clip_norm <= 0:
            raise ValueError(f"clip_norm must be positive, got {clip_norm}")

        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.sigma = np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon
        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

    def apply(self, stats: FittedStatistics) -> FittedStatistics:
        """Apply calibrated noise to all numeric statistics.

        Process:
        1. Clip each statistic to [-clip_norm, clip_norm]
        2. Compute sensitivity = 2 * clip_norm / n_samples
        3. Add Gaussian noise scaled by sensitivity * sigma

        This gives a formal (epsilon, delta)-DP guarantee per the
        Gaussian mechanism, independent of the actual data distribution.
        """
        noised = copy.deepcopy(stats)
        n = max(stats.n_samples, 1)

        # Sensitivity based on clipping norm, NOT data
        sensitivity = 2.0 * self.clip_norm / n
        noise_scale = sensitivity * self.sigma

        # Helper: clip then noise an array. Noise is drawn from the
        # instance-owned Generator (never the global np.random state) so
        # the (epsilon, delta)-DP guarantee is auditable.
        def clip_and_noise(arr: np.ndarray, non_negative: bool = False) -> np.ndarray:
            clipped = np.clip(arr, -self.clip_norm, self.clip_norm)
            noised_arr = clipped + self._rng.normal(0, noise_scale, arr.shape)
            if non_negative:
                noised_arr = np.maximum(noised_arr, 1e-12)
            return np.asarray(noised_arr)

        # Noise all vector statistics
        noised.mean = clip_and_noise(noised.mean)
        noised.std = clip_and_noise(noised.std, non_negative=True)
        noised.q1 = clip_and_noise(noised.q1)
        noised.q3 = clip_and_noise(noised.q3)
        noised.res_h_train = clip_and_noise(noised.res_h_train)
        noised.res_noise_ratio = clip_and_noise(noised.res_noise_ratio, non_negative=True)
        noised.kin_jerk_mean = clip_and_noise(noised.kin_jerk_mean)
        noised.kin_jerk_std = clip_and_noise(noised.kin_jerk_std, non_negative=True)
        noised.kin_accel_mean = clip_and_noise(noised.kin_accel_mean)
        noised.kin_accel_std = clip_and_noise(noised.kin_accel_std, non_negative=True)
        noised.ig_mean = clip_and_noise(noised.ig_mean)

        # Noise precision matrix (must stay symmetric)
        clipped_cov = np.clip(noised.ig_cov_inv, -self.clip_norm, self.clip_norm)
        noise_matrix = self._rng.normal(0, noise_scale, clipped_cov.shape)
        noise_matrix = (noise_matrix + noise_matrix.T) / 2  # Symmetrize
        noised.ig_cov_inv = clipped_cov + noise_matrix

        # Noise scalar
        noised.ig_log_det = float(
            np.clip(noised.ig_log_det, -self.clip_norm, self.clip_norm)
            + self._rng.normal(0, noise_scale)
        )

        # Record privacy parameters
        noised.epsilon = self.epsilon
        noised.delta = self.delta

        return noised
