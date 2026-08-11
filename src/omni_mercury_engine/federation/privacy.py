# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Differential privacy for federated sufficient statistics.

Releases a node's whole :class:`~omni_mercury_engine.federation.statistics.
FittedStatistics` bundle under a single (epsilon, delta)-differentially
private Gaussian mechanism.

The guarantee this module signs
-------------------------------
**Neighbouring relation: replace-one (bounded DP).** Two datasets are
neighbours when they have the same number of records and differ in exactly
one of them. This relation is declared explicitly because every number below
depends on it -- most importantly, ``n_samples`` is *invariant* across
neighbours under replace-one and therefore carries zero privacy loss when
released exactly, which is what makes sample-size weighting in the aggregator
sound rather than a leak.

**Sensitivity comes from clipping, never from the data.** Every released
scalar is clipped into ``[-clip_norm, clip_norm]`` before noise, so replacing
one record moves any single released component by at most ``2 * clip_norm``.
There is no ``1 / n_samples`` divisor: the clip bounds the *released
aggregate*, not a per-record contribution, so the record count cannot enter
the bound. (The previous implementation divided by ``n_samples``, which both
overstated privacy by a factor of ``n`` and made the noise calibration itself
a function of the data.)

**One mechanism, one release, no composition debt.** The bundle is treated as
a single vector-valued query. Its L2 sensitivity is the norm over all
released components -- ``sqrt(sum of per-component sensitivities squared)``
-- and one spherical Gaussian is added to the whole vector. Noising each of
the thirteen statistics under a *separately* budgeted mechanism (the previous
behaviour) costs up to 13x the recorded epsilon under basic composition; a
joint-sensitivity release costs exactly the recorded epsilon and is tighter
than any composition theorem could make the split version.

**Calibration is exact for every epsilon.** Noise is calibrated with the
analytic Gaussian mechanism of Balle & Wang (ICML 2018), which inverts the
mechanism's exact delta expression. The textbook closed form
``sigma = Delta * sqrt(2 ln(1.25/delta)) / epsilon`` is only valid for
``epsilon <= 1`` -- it silently under-noises above that, so the "weak
privacy, epsilon = 10" setting this module's own docstring advertised was not
(10, delta)-DP at all. :func:`analytic_gaussian_sigma` is correct on
``(0, inf)`` and is verified against the exact delta formula in
``tests/test_federation_privacy.py``.

**Nothing that fingerprints the raw data survives.** ``data_hash`` is a
SHA-256 digest of the exact bytes of the training matrix. No amount of noise
on the statistics matters if that digest is transmitted alongside them: it
confirms or refutes any guessed dataset outright. :meth:`DifferentialPrivacy.
apply` replaces it with :data:`REDACTED_DATA_HASH`.

Cost of the guarantee
---------------------
Honest calibration is expensive here, and the expense is visible: releasing
``11 * d`` vector entries plus ``d * (d + 1) / 2`` precision-matrix entries
plus a log-determinant means the joint sensitivity grows like
``2 * clip_norm * sqrt(d^2 / 2)``. Operators buy utility back through the two
knobs that legitimately affect it -- a ``clip_norm`` matched to the data's
actual scale (standardise features and a clip of 1.0 is ample; the 10.0
default is a loose fallback) and epsilon. :attr:`DifferentialPrivacy.
sigma_multiplier`, and the ``last_sensitivity`` / ``last_noise_scale``
attributes recorded by each :meth:`~DifferentialPrivacy.apply` call, expose
the arithmetic so the trade can be made with numbers rather than hope.

Each :meth:`~DifferentialPrivacy.apply` call is one (epsilon, delta) release.
Exporting the same node's statistics *k* times spends *k* releases; use
:class:`~omni_mercury_engine.federated_learning.privacy.PrivacyAccountant` to
track a cumulative budget across rounds.

References:
    Dwork & Roth (2014), *The Algorithmic Foundations of Differential
    Privacy*, Section 3.5 (Gaussian mechanism).
    Balle & Wang (2018), *Improving the Gaussian Mechanism for Differential
    Privacy: Analytical Calibration and Optimal Denoising*, ICML.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from omni_mercury_engine.federation.statistics import FittedStatistics

__all__ = [
    "REDACTED_DATA_HASH",
    "DifferentialPrivacy",
    "analytic_gaussian_sigma",
    "gaussian_mechanism_delta",
]

#: Value written over ``FittedStatistics.data_hash`` on a DP release. A raw
#: digest of the training bytes is a membership oracle for the whole dataset,
#: so it is replaced with a constant rather than merely truncated.
REDACTED_DATA_HASH = "dp-redacted"

#: Names of the length-``n_features`` vectors released by :meth:`apply`.
_VECTOR_FIELDS: tuple[tuple[str, bool], ...] = (
    ("mean", False),
    ("std", True),
    ("q1", False),
    ("q3", False),
    ("res_h_train", False),
    ("res_noise_ratio", True),
    ("kin_jerk_mean", False),
    ("kin_jerk_std", True),
    ("kin_accel_mean", False),
    ("kin_accel_std", True),
    ("ig_mean", False),
)

_SQRT2 = math.sqrt(2.0)


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * math.erfc(-x / _SQRT2)


def _log_phi(x: float) -> float:
    """``log Phi(x)``, returning ``-inf`` where ``Phi(x)`` underflows.

    Used so ``exp(epsilon) * Phi(-y)`` can be evaluated as
    ``exp(epsilon + log Phi(-y))``: for the large ``y`` that a large epsilon
    produces, the factors are respectively enormous and denormal, and the
    naive product loses the result to overflow or to a spurious ``0 * inf``.
    """
    value = 0.5 * math.erfc(-x / _SQRT2)
    if value <= 0.0:
        return -math.inf
    return math.log(value)


def gaussian_mechanism_delta(epsilon: float, sigma: float, sensitivity: float) -> float:
    """Exact delta of the Gaussian mechanism at a given sigma.

    This is the closed form the analytic calibration inverts (Balle & Wang,
    Theorem 8)::

        delta = Phi(Delta/(2 sigma) - epsilon sigma/Delta)
                - e^epsilon * Phi(-Delta/(2 sigma) - epsilon sigma/Delta)

    Exposed so callers -- and the test suite -- can *verify* a calibration
    rather than trust it.

    Args:
        epsilon: Privacy loss parameter, > 0.
        sigma: Standard deviation of the noise added to each component.
        sensitivity: L2 sensitivity of the query the noise is protecting.

    Returns:
        The smallest delta for which the mechanism is (epsilon, delta)-DP.
    """
    if sigma <= 0.0:
        return 1.0
    ratio = sensitivity / (2.0 * sigma)
    shift = epsilon * sigma / sensitivity
    return max(0.0, _phi(ratio - shift) - math.exp(epsilon + _log_phi(-ratio - shift)))


def _bisect(
    func: Any,
    lo: float,
    hi: float,
    target: float,
    *,
    decreasing: bool,
    iterations: int = 200,
) -> float:
    """Bisect a monotone ``func`` for ``func(x) == target`` on ``[lo, hi]``."""
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        value = func(mid)
        if (value > target) != decreasing:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def analytic_gaussian_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
    """Smallest sigma for which the Gaussian mechanism is (epsilon, delta)-DP.

    Implements Algorithm 1 of Balle & Wang (2018). Unlike the classical
    ``sqrt(2 ln(1.25/delta)) / epsilon`` bound it is exact and valid for every
    ``epsilon > 0``, not only ``epsilon <= 1``.

    Args:
        epsilon: Privacy loss parameter, > 0.
        delta: Failure probability, in (0, 1).
        sensitivity: L2 sensitivity of the query, > 0.

    Returns:
        The noise standard deviation to add to each output component.

    Raises:
        ValueError: Any argument is outside its domain.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if sensitivity <= 0:
        raise ValueError(f"sensitivity must be positive, got {sensitivity}")

    # delta at alpha = 1 (i.e. sigma = Delta / sqrt(2 epsilon)) splits the two
    # branches: below it the mechanism needs *more* noise than the reference
    # point, above it less.
    delta_0 = _phi(0.0) - math.exp(epsilon + _log_phi(-math.sqrt(2.0 * epsilon)))

    if delta >= delta_0:
        # B+ rises from delta_0 to 1; find where it reaches delta.
        def b_plus(v: float) -> float:
            return _phi(math.sqrt(epsilon * v)) - math.exp(
                epsilon + _log_phi(-math.sqrt(epsilon * (v + 2.0)))
            )

        hi = 1.0
        while b_plus(hi) < delta:
            hi *= 2.0
            if hi > 1e12:  # pragma: no cover - unreachable: B+ -> 1 > delta
                break
        v_star = _bisect(b_plus, 0.0, hi, delta, decreasing=False)
        alpha = math.sqrt(1.0 + v_star / 2.0) - math.sqrt(v_star / 2.0)
    else:
        # B- falls from delta_0 to 0; find where it drops to delta.
        def b_minus(u: float) -> float:
            return _phi(-math.sqrt(epsilon * u)) - math.exp(
                epsilon + _log_phi(-math.sqrt(epsilon * (u + 2.0)))
            )

        hi = 1.0
        while b_minus(hi) > delta:
            hi *= 2.0
            if hi > 1e12:  # pragma: no cover - unreachable: B- -> 0 < delta
                break
        u_star = _bisect(b_minus, 0.0, hi, delta, decreasing=True)
        alpha = math.sqrt(1.0 + u_star / 2.0) + math.sqrt(u_star / 2.0)

    return float(alpha * sensitivity / math.sqrt(2.0 * epsilon))


class DifferentialPrivacy:
    """Apply a joint (epsilon, delta)-DP Gaussian release to fitted statistics.

    See the module docstring for the neighbouring relation, the sensitivity
    derivation, and why the release is a single mechanism rather than thirteen.

    Args:
        epsilon: Privacy budget for **one** :meth:`apply` call. Lower = more
            private, more noise. 0.1 is strong, 1.0 moderate, 10.0 weak. Every
            value > 0 is calibrated exactly (see :func:`analytic_gaussian_sigma`).
        delta: Probability the (pure) epsilon bound is exceeded. Default 1e-5.
            Should be well below ``1 / n_samples``.
        clip_norm: Magnitude bound applied to every released scalar before
            noise. This is the sensitivity knob: replacing one record can move
            a clipped component by at most ``2 * clip_norm``. Default 10.0 is a
            loose fallback -- match it to the data's scale (standardised
            features want ~1.0) to buy back utility honestly.
        rng: Optional numpy ``Generator`` providing the noise source. If
            omitted, a fresh OS-seeded ``np.random.default_rng()`` is
            constructed per instance -- the global ``np.random`` legacy state
            is **never** used, so a caller cannot accidentally de-randomise the
            privacy noise via ``np.random.seed(...)`` elsewhere in the process.
            For audited / reproducible deployments pass an explicit
            ``np.random.default_rng(seed)`` seeded from a documented entropy
            source.
        seed: Convenience seed used when ``rng`` is not supplied.

    Attributes:
        sigma_multiplier: ``sigma / sensitivity`` for this (epsilon, delta)
            pair. Known at construction; multiply by the sensitivity of any
            query to get its noise scale.
        last_sensitivity: L2 sensitivity of the most recent :meth:`apply`.
        last_noise_scale: Per-component noise standard deviation of the most
            recent :meth:`apply`.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float = 1e-5,
        clip_norm: float = 10.0,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the instance."""
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if delta <= 0 or delta >= 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if clip_norm <= 0:
            raise ValueError(f"clip_norm must be positive, got {clip_norm}")

        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        # Calibrated against unit sensitivity: sigma scales linearly in
        # sensitivity, so one calibration serves every output shape.
        self.sigma_multiplier = analytic_gaussian_sigma(epsilon, delta, 1.0)
        self.last_sensitivity: float | None = None
        self.last_noise_scale: float | None = None
        if rng is not None:
            self._rng: np.random.Generator = rng
        else:
            self._rng = np.random.default_rng(seed)

    def component_count(self, n_features: int) -> int:
        """Number of independent scalars released for ``n_features`` features.

        Eleven length-``d`` vectors, the *upper triangle* of the symmetric
        precision matrix (the lower triangle is a deterministic mirror, not an
        independent release), and the scalar log-determinant.

        Args:
            n_features: Feature dimension of the statistics being released.

        Returns:
            The count of independently noised scalars.
        """
        d = max(int(n_features), 0)
        return len(_VECTOR_FIELDS) * d + d * (d + 1) // 2 + 1

    def sensitivity(self, n_features: int) -> float:
        """L2 sensitivity of the full release for ``n_features`` features.

        Each of :meth:`component_count` components is clipped into
        ``[-clip_norm, clip_norm]``, so replacing one record moves it by at
        most ``2 * clip_norm``; the joint L2 sensitivity is the norm of that
        per-component vector.

        Args:
            n_features: Feature dimension of the statistics being released.

        Returns:
            The L2 sensitivity used to calibrate the Gaussian noise.
        """
        return 2.0 * self.clip_norm * math.sqrt(self.component_count(n_features))

    def apply(self, stats: FittedStatistics) -> FittedStatistics:
        """Release ``stats`` under one (epsilon, delta)-DP Gaussian mechanism.

        Process:

        1. Clip every released scalar into ``[-clip_norm, clip_norm]``,
           bounding the replace-one sensitivity at ``2 * clip_norm`` each.
        2. Take the joint L2 sensitivity over all released components
           (:meth:`sensitivity`) and calibrate one spherical Gaussian to it.
        3. Add that noise to every component, mirroring the precision matrix's
           upper triangle so the released matrix stays symmetric without
           halving the variance of its off-diagonals.
        4. Redact the raw-data fingerprint.

        Steps that only rewrite already-noised values -- clamping
        non-negative quantities away from zero, symmetrising -- are
        post-processing and consume no additional budget.

        Args:
            stats: The node's fitted statistics. Not mutated; a deep copy is
                returned.

        Returns:
            A noised copy carrying the achieved ``epsilon`` / ``delta``.
        """
        noised = copy.deepcopy(stats)

        sensitivity = self.sensitivity(stats.n_features)
        noise_scale = self.sigma_multiplier * sensitivity
        self.last_sensitivity = sensitivity
        self.last_noise_scale = noise_scale

        def clip_and_noise(
            arr: np.ndarray[Any, Any], non_negative: bool = False
        ) -> np.ndarray[Any, Any]:
            clipped = np.clip(arr, -self.clip_norm, self.clip_norm)
            noised_arr = clipped + self._rng.normal(0, noise_scale, np.shape(arr))
            if non_negative:
                # Post-processing: a variance estimate must stay positive for
                # the reconstructed detector to be usable at all.
                noised_arr = np.maximum(noised_arr, 1e-12)
            return np.asarray(noised_arr)

        for field, non_negative in _VECTOR_FIELDS:
            setattr(noised, field, clip_and_noise(getattr(noised, field), non_negative))

        # Precision matrix. Only the upper triangle is an independent release:
        # symmetrise first, clip, then draw noise for the upper triangle and
        # mirror it. Drawing a full matrix and averaging it with its transpose
        # -- the previous approach -- gives off-diagonals variance
        # ``noise_scale^2 / 2``, i.e. less noise than the calibration paid for,
        # which breaks the guarantee it was supposed to deliver.
        symmetric = 0.5 * (
            np.asarray(noised.ig_cov_inv, dtype=np.float64)
            + np.asarray(noised.ig_cov_inv, dtype=np.float64).T
        )
        clipped_cov = np.clip(symmetric, -self.clip_norm, self.clip_norm)
        upper = self._rng.normal(0, noise_scale, clipped_cov.shape)
        upper = np.triu(upper)
        noise_matrix = upper + np.triu(upper, k=1).T
        noised.ig_cov_inv = clipped_cov + noise_matrix

        noised.ig_log_det = float(
            np.clip(noised.ig_log_det, -self.clip_norm, self.clip_norm)
            + self._rng.normal(0, noise_scale)
        )

        # The raw-data fingerprint is a direct function of the training bytes;
        # no noise on the statistics protects anything while it travels with
        # them. ``n_samples`` is invariant under the replace-one neighbouring
        # relation this mechanism is calibrated for, so it is released as-is
        # (see the module docstring) and, unlike before, never enters the
        # noise calibration.
        noised.data_hash = REDACTED_DATA_HASH

        noised.epsilon = self.epsilon
        noised.delta = self.delta

        return noised
