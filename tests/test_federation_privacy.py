# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The federated DP release must actually deliver the (epsilon, delta) it prints.

Four independent defects made the previous mechanism unsound, and each has a
test here that fails against the old implementation:

* **Sensitivity divided by ``n_samples``.** ``clip_and_noise`` clips the
  *released aggregate*, so the record count cannot appear in the bound. The
  ``/ n`` overstated privacy by a factor of ``n`` and made the noise scale a
  function of the data.
* **Thirteen mechanisms, one epsilon.** Every statistic was noised under its
  own full-epsilon Gaussian and the bundle was labelled ``epsilon``; basic
  composition puts the true cost at up to 13x that. The release is now one
  joint-sensitivity mechanism, so the label is exact.
* **A calibration valid only for ``epsilon <= 1``.** The classical
  ``sqrt(2 ln(1.25/delta)) / epsilon`` form under-noises above 1, and the
  class docstring advertised 10.0.
* **The raw-data fingerprint rode along.** ``data_hash`` is a SHA-256 of the
  training bytes; transmitting it defeats every other protection here.

Plus the symmetrisation bug the joint mechanism would otherwise inherit:
averaging a noise matrix with its own transpose halves the variance of every
off-diagonal, so the matrix was noised below its own calibration.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from omni_mercury_engine.federation.privacy import (
    REDACTED_DATA_HASH,
    DifferentialPrivacy,
    analytic_gaussian_sigma,
    gaussian_mechanism_delta,
)
from omni_mercury_engine.federation.statistics import FittedStatistics

_EPSILONS = (0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 3.0, 10.0, 40.0)
_DELTAS = (1e-3, 1e-5, 1e-7, 1e-9)


def _zero_stats(n_features: int, n_samples: int = 1000) -> FittedStatistics:
    """All-zero statistics, so a release *is* its own noise."""
    zeros = np.zeros(n_features, dtype=np.float64)
    return FittedStatistics(
        node_id="node",
        timestamp=0.0,
        n_samples=n_samples,
        n_features=n_features,
        mean=zeros.copy(),
        std=zeros.copy(),
        q1=zeros.copy(),
        q3=zeros.copy(),
        res_h_train=zeros.copy(),
        res_noise_ratio=zeros.copy(),
        kin_jerk_mean=zeros.copy(),
        kin_jerk_std=zeros.copy(),
        kin_accel_mean=zeros.copy(),
        kin_accel_std=zeros.copy(),
        ig_mean=zeros.copy(),
        ig_cov_inv=np.zeros((n_features, n_features), dtype=np.float64),
        ig_log_det=0.0,
        data_hash="deadbeefcafebabe",
    )


class TestAnalyticCalibration:
    """The noise scale is the exact solution, on all of ``epsilon > 0``."""

    @pytest.mark.parametrize("epsilon", _EPSILONS)
    @pytest.mark.parametrize("delta", _DELTAS)
    def test_calibration_is_sound_and_tight(self, epsilon: float, delta: float) -> None:
        """The achieved delta equals the requested delta, never exceeds it."""
        sigma = analytic_gaussian_sigma(epsilon, delta, sensitivity=3.0)
        achieved = gaussian_mechanism_delta(epsilon, sigma, sensitivity=3.0)
        assert achieved <= delta * (1 + 1e-6)
        assert achieved >= delta * (1 - 1e-6)

    def test_sigma_is_linear_in_sensitivity(self) -> None:
        base = analytic_gaussian_sigma(1.0, 1e-5, 1.0)
        assert analytic_gaussian_sigma(1.0, 1e-5, 7.0) == pytest.approx(7.0 * base)

    def test_sigma_decreases_with_epsilon(self) -> None:
        sigmas = [analytic_gaussian_sigma(eps, 1e-5, 1.0) for eps in _EPSILONS]
        assert sigmas == sorted(sigmas, reverse=True)

    @pytest.mark.parametrize("epsilon", (10.0, 20.0, 40.0))
    def test_classical_bound_is_unsound_at_the_advertised_weak_setting(
        self, epsilon: float
    ) -> None:
        """``sqrt(2 ln(1.25/delta)) / epsilon`` misses delta above ~8.5.

        That form is derived under ``epsilon <= 1``. Past a crossover near
        8.5 (for ``delta = 1e-5``) it returns a *smaller* sigma than the
        mechanism needs, so the release misses its own delta -- and 10.0 is
        exactly the value the class docstring advertised as "weak privacy".
        The miss grows fast: 2.3x at epsilon 10, 150x at 20, 2e4 at 40.
        """
        delta = 1e-5
        classical = math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
        assert gaussian_mechanism_delta(epsilon, classical, 1.0) > delta
        assert analytic_gaussian_sigma(epsilon, delta, 1.0) > classical

    @pytest.mark.parametrize("epsilon", (0.1, 1.0, 2.0, 5.0))
    def test_classical_bound_is_merely_wasteful_below_the_crossover(self, epsilon: float) -> None:
        """Below the crossover the old form is sound but over-noises.

        So neither regime was correct: above ~8.5 it broke the guarantee,
        below it burned utility that the exact calibration keeps. At epsilon
        1.0 the classical sigma is 30% larger than necessary.
        """
        delta = 1e-5
        classical = math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
        exact = analytic_gaussian_sigma(epsilon, delta, 1.0)
        assert exact < classical
        assert gaussian_mechanism_delta(epsilon, classical, 1.0) < delta

    @pytest.mark.parametrize(
        ("epsilon", "delta", "sensitivity"),
        [(0.0, 1e-5, 1.0), (-1.0, 1e-5, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1e-5, 0.0)],
    )
    def test_domain_is_enforced(self, epsilon: float, delta: float, sensitivity: float) -> None:
        with pytest.raises(ValueError):
            analytic_gaussian_sigma(epsilon, delta, sensitivity)


class TestSensitivityModel:
    """Sensitivity comes from the clip, and from nothing else."""

    def test_is_independent_of_sample_count(self) -> None:
        """No ``1 / n``: the clip bounds the released aggregate, not a record."""
        dp = DifferentialPrivacy(epsilon=1.0, seed=0)
        dp.apply(_zero_stats(5, n_samples=10))
        small = dp.last_noise_scale
        dp.apply(_zero_stats(5, n_samples=10_000_000))
        assert dp.last_noise_scale == small

    def test_component_count_matches_what_is_released(self) -> None:
        """11 vectors + the precision matrix's upper triangle + log-det."""
        dp = DifferentialPrivacy(epsilon=1.0, seed=0)
        for d in (1, 3, 8):
            assert dp.component_count(d) == 11 * d + d * (d + 1) // 2 + 1

    def test_lower_triangle_is_not_double_counted(self) -> None:
        """A symmetric matrix releases d(d+1)/2 numbers, not d^2."""
        dp = DifferentialPrivacy(epsilon=1.0, seed=0)
        d = 6
        assert dp.component_count(d) < 11 * d + d * d + 1

    def test_sensitivity_is_two_clips_over_the_component_norm(self) -> None:
        dp = DifferentialPrivacy(epsilon=1.0, clip_norm=2.5, seed=0)
        expected = 2.0 * 2.5 * math.sqrt(dp.component_count(4))
        assert dp.sensitivity(4) == pytest.approx(expected)

    def test_sensitivity_scales_linearly_with_clip_norm(self) -> None:
        a = DifferentialPrivacy(epsilon=1.0, clip_norm=1.0, seed=0).sensitivity(4)
        b = DifferentialPrivacy(epsilon=1.0, clip_norm=4.0, seed=0).sensitivity(4)
        assert b == pytest.approx(4.0 * a)


class TestJointRelease:
    """One mechanism over the whole bundle -- no hidden composition."""

    def test_every_component_carries_the_calibrated_noise(self) -> None:
        """A single release is enough to measure the noise on ~1100 scalars.

        If any statistic were noised under a *separate*, differently scaled
        mechanism -- the composition defect -- its empirical spread would not
        match ``last_noise_scale``.
        """
        d = 40
        dp = DifferentialPrivacy(epsilon=1.0, clip_norm=1.0, seed=1234)
        noised = dp.apply(_zero_stats(d))
        assert dp.last_noise_scale is not None
        scale = dp.last_noise_scale

        for field in (
            "mean",
            "q1",
            "q3",
            "res_h_train",
            "kin_jerk_mean",
            "kin_accel_mean",
            "ig_mean",
        ):
            observed = float(np.std(getattr(noised, field)))
            assert observed == pytest.approx(scale, rel=0.35), field

    def test_matrix_off_diagonals_are_not_under_noised(self) -> None:
        """Off-diagonal variance must equal sigma^2, not sigma^2 / 2.

        ``(N + N.T) / 2`` symmetrises by averaging two independent draws,
        which halves off-diagonal variance -- the matrix would then be
        protected at ``sigma / sqrt(2)`` while the accounting charged
        ``sigma``. Mirroring the upper triangle symmetrises without touching
        the distribution of what is actually released.
        """
        d = 60
        dp = DifferentialPrivacy(epsilon=1.0, clip_norm=1.0, seed=99)
        noise = dp.apply(_zero_stats(d)).ig_cov_inv
        assert dp.last_noise_scale is not None
        scale = dp.last_noise_scale

        upper = noise[np.triu_indices(d, k=1)]
        diagonal = np.diag(noise)
        assert float(np.std(upper)) == pytest.approx(scale, rel=0.1)
        assert float(np.std(diagonal)) == pytest.approx(scale, rel=0.3)

    def test_matrix_stays_symmetric(self) -> None:
        noise = DifferentialPrivacy(epsilon=1.0, seed=5).apply(_zero_stats(9)).ig_cov_inv
        np.testing.assert_allclose(noise, noise.T, rtol=0, atol=0)

    def test_recorded_budget_is_the_requested_budget(self) -> None:
        noised = DifferentialPrivacy(epsilon=0.7, delta=1e-6, seed=3).apply(_zero_stats(4))
        assert noised.epsilon == 0.7
        assert noised.delta == 1e-6


class TestFingerprintRedaction:
    """The digest of the raw training bytes must not leave the node."""

    def test_data_hash_is_redacted(self) -> None:
        raw = _zero_stats(4)
        noised = DifferentialPrivacy(epsilon=1.0, seed=0).apply(raw)
        assert noised.data_hash == REDACTED_DATA_HASH
        assert noised.data_hash != raw.data_hash

    def test_source_statistics_are_not_mutated(self) -> None:
        raw = _zero_stats(4)
        DifferentialPrivacy(epsilon=1.0, seed=0).apply(raw)
        assert raw.data_hash == "deadbeefcafebabe"
        assert float(np.max(np.abs(raw.mean))) == 0.0

    def test_redacted_marker_is_not_a_truncated_digest(self) -> None:
        """A shorter hash would still be a membership oracle; a constant is not."""
        assert not all(c in "0123456789abcdef" for c in REDACTED_DATA_HASH)


class TestPostProcessing:
    """Clamping and clipping behave as documented."""

    def test_non_negative_fields_stay_positive(self) -> None:
        noised = DifferentialPrivacy(epsilon=1.0, seed=11).apply(_zero_stats(6))
        for field in ("std", "res_noise_ratio", "kin_jerk_std", "kin_accel_std"):
            assert np.all(getattr(noised, field) > 0.0), field

    def test_values_are_clipped_before_noise(self) -> None:
        """A wild input cannot widen the sensitivity the noise was sized for."""
        stats = _zero_stats(3)
        stats.mean = np.array([1e9, -1e9, 0.0])
        dp = DifferentialPrivacy(epsilon=50.0, clip_norm=1.0, delta=1e-5, seed=2)
        noised = dp.apply(stats)
        assert dp.last_noise_scale is not None
        # Clipped to +/-1 then perturbed: the released value cannot retain the
        # 1e9 magnitude it arrived with.
        assert float(np.max(np.abs(noised.mean))) < 1.0 + 20.0 * dp.last_noise_scale

    def test_seeded_releases_are_reproducible(self) -> None:
        a = DifferentialPrivacy(epsilon=1.0, seed=42).apply(_zero_stats(5))
        b = DifferentialPrivacy(epsilon=1.0, seed=42).apply(_zero_stats(5))
        np.testing.assert_array_equal(a.mean, b.mean)
        np.testing.assert_array_equal(a.ig_cov_inv, b.ig_cov_inv)

    def test_stronger_privacy_means_more_noise(self) -> None:
        strong = DifferentialPrivacy(epsilon=0.1, seed=8).apply(_zero_stats(5))
        weak = DifferentialPrivacy(epsilon=10.0, seed=8).apply(_zero_stats(5))
        assert float(np.linalg.norm(strong.mean)) > float(np.linalg.norm(weak.mean))
