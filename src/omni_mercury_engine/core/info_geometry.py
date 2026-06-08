# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Information Geometry for Out-of-Distribution Detection -- Phase 4C.

Based on: IGEOOD - An Information Geometry Approach to Out-of-Distribution Detection
(ICLR 2022: https://openreview.net/pdf?id=mfwdY3U_9ea)

Phase 4C adds Fisher Information Metric Adaptive Thresholds:
  - FisherInformationMatrix: closed-form Gaussian and empirical score estimation
  - NaturalGradient: F-inverse times Euclidean gradient via Cholesky decomposition
  - FisherRaoAdaptiveThreshold: adaptive anomaly thresholds from the FIM
  - StatisticalManifold: manifold bookkeeping, geodesic distance, exponential map
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg as sp_linalg

__all__ = [
    "FisherInformationMatrix",
    "FisherRaoAdaptiveThreshold",
    "InformationGeometryDetector",
    "NaturalGradient",
    "StatisticalManifold",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MIN_VARIANCE: float = 1e-12
_DEFAULT_TIKHONOV: float = 1e-6


def _ensure_2d_square(matrix: np.ndarray[Any, Any], label: str) -> np.ndarray[Any, Any]:
    """Validate that *matrix* is 2-D and square, returning it unchanged."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{label} must be a square 2-D array, got shape {matrix.shape}")
    return matrix


def _regularize(
    matrix: np.ndarray[Any, Any],
    tikhonov_lambda: float = _DEFAULT_TIKHONOV,
) -> np.ndarray[Any, Any]:
    """Apply Tikhonov regularization: M + lambda * I."""
    n = matrix.shape[0]
    return matrix + tikhonov_lambda * np.eye(n, dtype=matrix.dtype)


def _safe_cholesky(
    matrix: np.ndarray[Any, Any],
    tikhonov_lambda: float = _DEFAULT_TIKHONOV,
) -> np.ndarray[Any, Any]:
    """Cholesky decomposition with automatic Tikhonov fallback.

    Returns the lower-triangular factor *L* such that ``L @ L.T == matrix`` (up to regularization).
    """
    try:
        result: np.ndarray[Any, Any] = sp_linalg.cholesky(matrix, lower=True)
        return result
    except sp_linalg.LinAlgError:
        regularized = _regularize(matrix, tikhonov_lambda)
        result = sp_linalg.cholesky(regularized, lower=True)
        return result


# =========================================================================
# FisherInformationMatrix
# =========================================================================


class FisherInformationMatrix:
    """Compute and store the Fisher Information Matrix (FIM).

    Supports two modes:
      * **Gaussian closed-form** -- the FIM for a multivariate Gaussian
        with parameters (mean, covariance) is assembled analytically.
      * **Empirical score function** -- the FIM is estimated from samples
        via ``F = E[score * score^T]`` where the score is the gradient
        of the log-likelihood.

    All matrices are stored after Tikhonov regularization for numerical
    stability: ``F_reg = F + lambda * I``.
    """

    def __init__(self, tikhonov_lambda: float = _DEFAULT_TIKHONOV) -> None:
        """Initialize the instance."""
        self.tikhonov_lambda: float = tikhonov_lambda
        self.fim: np.ndarray[Any, Any] | None = None
        self.dim: int | None = None

    # -- Gaussian closed-form -----------------------------------------------

    def compute_gaussian(
        self,
        covariance: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Closed-form FIM for a multivariate Gaussian (mean parameters).

        For a Gaussian with known covariance sigma, the Fisher information
        w.r.t. the mean vector is ``F = sigma^{-1}``.

        For the scalar (1-D) case: ``F = 1 / sigma^2``.

        Args:
            covariance: Covariance matrix (d x d) of the Gaussian. A 1-D
                array of length *d* is interpreted as a diagonal covariance.

        Returns:
            Regularized Fisher Information Matrix (d x d).
        """
        if covariance.ndim == 1:
            # Diagonal covariance supplied as a vector.
            safe_var = np.maximum(covariance, _MIN_VARIANCE)
            fim_raw = np.diag(1.0 / safe_var)
        else:
            covariance = _ensure_2d_square(covariance, "covariance")
            # Regularize covariance before inversion to avoid singularity.
            cov_reg = _regularize(covariance, self.tikhonov_lambda)
            try:
                fim_raw = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                fim_raw = np.eye(covariance.shape[0], dtype=covariance.dtype)

        # Symmetrise to eliminate floating-point asymmetry.
        fim_raw = 0.5 * (fim_raw + fim_raw.T)

        self.fim = _regularize(fim_raw, self.tikhonov_lambda)
        self.dim = self.fim.shape[0]
        return self.fim

    # -- Empirical estimation -----------------------------------------------

    def compute_empirical(
        self,
        samples: np.ndarray[Any, Any],
        log_likelihood_grad_fn: Any | None = None,
    ) -> np.ndarray[Any, Any]:
        """Estimate the FIM from samples via the empirical score function.

        When *log_likelihood_grad_fn* is ``None`` a Gaussian model is
        assumed and the score is computed as ``sigma^{-1} (x - mu)``.

        Args:
            samples: Array of shape ``(n_samples, d)``.
            log_likelihood_grad_fn: Optional callable ``(x, params) -> grad``
                that returns the score vector for a single sample *x*.

        Returns:
            Regularized empirical FIM (d x d).
        """
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)

        n_samples, d = samples.shape
        if n_samples < 2:
            self.fim = _regularize(np.eye(d), self.tikhonov_lambda)
            self.dim = d
            return self.fim

        if log_likelihood_grad_fn is not None:
            scores = np.array([log_likelihood_grad_fn(x, None) for x in samples])
        else:
            # Default: Gaussian score = cov_inv @ (x - mean).
            mean = np.mean(samples, axis=0)
            cov = np.cov(samples.T)
            if cov.ndim == 0:
                cov = np.atleast_2d(cov)
            cov_reg = _regularize(cov, self.tikhonov_lambda)
            try:
                cov_inv = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                cov_inv = np.eye(d)
            centered = samples - mean
            scores = centered @ cov_inv.T

        # FIM = (1/n) * sum(score_i @ score_i^T)
        fim_raw = (scores.T @ scores) / n_samples
        fim_raw = 0.5 * (fim_raw + fim_raw.T)

        self.fim = _regularize(fim_raw, self.tikhonov_lambda)
        self.dim = d
        return self.fim

    # -- Utilities -----------------------------------------------------------

    def get_fim(self) -> np.ndarray[Any, Any]:
        """Return the computed FIM, raising if not yet computed."""
        if self.fim is None:
            raise RuntimeError("FIM has not been computed yet.")
        return self.fim

    def trace_inverse(self) -> float:
        """Return ``trace(F^{-1})``, useful for threshold calibration."""
        fim = self.get_fim()
        try:
            fim_inv = np.linalg.inv(fim)
        except np.linalg.LinAlgError:
            fim_inv = np.linalg.pinv(fim)
        return float(np.trace(fim_inv))


# =========================================================================
# NaturalGradient
# =========================================================================


class NaturalGradient:
    """Compute the natural gradient: ``g_nat = F^{-1} g_euclid``.

    Uses Cholesky decomposition for numerically stable inversion.
    Supports an optional damping factor (Tikhonov) so that the effective
    inverse is ``(F + lambda * I)^{-1}``.
    """

    def __init__(self, damping: float = _DEFAULT_TIKHONOV) -> None:
        """Initialize the instance."""
        self.damping: float = damping

    def compute(
        self,
        fim: np.ndarray[Any, Any],
        euclidean_gradient: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Compute the natural gradient.

        Args:
            fim: Fisher Information Matrix (d x d).
            euclidean_gradient: Standard Euclidean gradient vector (d,).

        Returns:
            Natural gradient vector (d,).
        """
        fim = _ensure_2d_square(fim, "FIM")
        if euclidean_gradient.ndim != 1:
            raise ValueError("euclidean_gradient must be a 1-D vector.")
        if fim.shape[0] != euclidean_gradient.shape[0]:
            raise ValueError(
                f"Dimension mismatch: FIM is {fim.shape[0]}x{fim.shape[0]} "
                f"but gradient has length {euclidean_gradient.shape[0]}."
            )

        damped = _regularize(fim, self.damping)
        result: np.ndarray[Any, Any]
        try:
            cho_lower = sp_linalg.cholesky(damped, lower=True)
            result = sp_linalg.cho_solve((cho_lower, True), euclidean_gradient)
        except sp_linalg.LinAlgError:
            # Fall back to pseudo-inverse if Cholesky still fails.
            result = np.linalg.pinv(damped) @ euclidean_gradient

        return result

    def compute_from_samples(
        self,
        samples: np.ndarray[Any, Any],
        euclidean_gradient: np.ndarray[Any, Any],
        tikhonov_lambda: float = _DEFAULT_TIKHONOV,
    ) -> np.ndarray[Any, Any]:
        """Convenience: estimate FIM from *samples*, then compute natural gradient."""
        fim_computer = FisherInformationMatrix(tikhonov_lambda=tikhonov_lambda)
        fim = fim_computer.compute_empirical(samples)
        return self.compute(fim, euclidean_gradient)


# =========================================================================
# StatisticalManifold
# =========================================================================


class StatisticalManifold:
    """A point on the manifold of probability distributions.

    Stores the reference parameters (mean, covariance) together with the Fisher metric at that
    point, and provides geodesic-distance and exponential-map utilities.
    """

    def __init__(
        self,
        mean: np.ndarray[Any, Any],
        covariance: np.ndarray[Any, Any],
        tikhonov_lambda: float = _DEFAULT_TIKHONOV,
    ) -> None:
        """Initialize the instance."""
        self.mean: np.ndarray[Any, Any] = np.asarray(mean, dtype=np.float64)
        if self.mean.ndim == 0:
            self.mean = self.mean.reshape(1)
        self.dim: int = self.mean.shape[0]

        covariance = np.asarray(covariance, dtype=np.float64)
        if covariance.ndim < 2:
            covariance = np.diag(np.maximum(np.atleast_1d(covariance), _MIN_VARIANCE))
        self.covariance: np.ndarray[Any, Any] = covariance

        self.tikhonov_lambda: float = tikhonov_lambda

        # Pre-compute and cache the FIM at this point.
        fim_obj = FisherInformationMatrix(tikhonov_lambda=tikhonov_lambda)
        self.fisher_metric: np.ndarray[Any, Any] = fim_obj.compute_gaussian(self.covariance)

    # -- Geodesic distance ---------------------------------------------------

    def geodesic_distance(
        self,
        other_mean: np.ndarray[Any, Any],
        other_covariance: np.ndarray[Any, Any] | None = None,
    ) -> float:
        """Fisher-Rao geodesic distance between *self* and another point.

        For two Gaussians the geodesic distance in the mean-parameter
        sub-manifold is:

            d(p, q) = sqrt( (mu_p - mu_q)^T  F  (mu_p - mu_q) )

        When *other_covariance* is supplied an additional term capturing
        the covariance mismatch is added (Skovgaard divergence
        approximation).

        Args:
            other_mean: Mean of the second distribution.
            other_covariance: Optional covariance of the second distribution.

        Returns:
            Non-negative geodesic distance.
        """
        other_mean = np.asarray(other_mean, dtype=np.float64).ravel()
        delta_mu = self.mean - other_mean

        # Mean-parameter contribution.
        mean_term: float = float(delta_mu @ self.fisher_metric @ delta_mu)

        cov_term: float = 0.0
        if other_covariance is not None:
            other_covariance = np.asarray(other_covariance, dtype=np.float64)
            if other_covariance.ndim < 2:
                other_covariance = np.diag(
                    np.maximum(np.atleast_1d(other_covariance), _MIN_VARIANCE)
                )
            # Skovgaard-type approximation for the covariance contribution:
            # 0.5 * || log(sigma_1^{-1} sigma_2) ||_F^2
            cov_reg = _regularize(self.covariance, self.tikhonov_lambda)
            try:
                cov_inv = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov_reg)

            product = cov_inv @ other_covariance
            # Eigenvalues of cov_inv @ cov_other; take real part for safety.
            eigvals = np.real(np.linalg.eigvals(product))
            eigvals = np.maximum(eigvals, _MIN_VARIANCE)
            cov_term = 0.5 * float(np.sum(np.log(eigvals) ** 2))

        distance_sq = max(mean_term + cov_term, 0.0)
        return float(np.sqrt(distance_sq))

    # -- Exponential map -----------------------------------------------------

    def exponential_map(
        self,
        tangent_vector: np.ndarray[Any, Any],
        step_size: float = 1.0,
    ) -> np.ndarray[Any, Any]:
        """First-order exponential map approximation for parameter update.

        Moves along the geodesic from the current mean in the direction of
        *tangent_vector* (lifted to the tangent space of the manifold):

            mu_new = mu + step_size * F^{-1} v

        Args:
            tangent_vector: Direction in the tangent space (d,).
            step_size: How far to move along the geodesic.

        Returns:
            Updated mean parameter array (d,).
        """
        tangent_vector = np.asarray(tangent_vector, dtype=np.float64).ravel()
        if tangent_vector.shape[0] != self.dim:
            raise ValueError(
                f"tangent_vector length {tangent_vector.shape[0]} != manifold dim {self.dim}"
            )
        nat_grad_solver = NaturalGradient(damping=self.tikhonov_lambda)
        displacement = nat_grad_solver.compute(self.fisher_metric, tangent_vector)
        return self.mean + step_size * displacement


# =========================================================================
# FisherRaoAdaptiveThreshold
# =========================================================================


class FisherRaoAdaptiveThreshold:
    """Derive and adapt anomaly-detection thresholds from the Fisher metric.

    The threshold is set as:

        tau = mu_distance + k * sqrt(trace(F^{-1}))

    where *mu_distance* is the mean geodesic distance of calibration samples
    to the reference, *k* is a confidence multiplier (analogous to the number
    of standard deviations), and ``trace(F^{-1})`` captures the local
    curvature of the statistical manifold.

    The threshold is automatically recalibrated when the FIM changes
    beyond a configurable drift tolerance.
    """

    def __init__(
        self,
        confidence_k: float = 3.0,
        drift_tolerance: float = 0.1,
        tikhonov_lambda: float = _DEFAULT_TIKHONOV,
    ) -> None:
        """Initialize the instance."""
        self.confidence_k: float = confidence_k
        self.drift_tolerance: float = drift_tolerance
        self.tikhonov_lambda: float = tikhonov_lambda

        # State populated by ``calibrate``.
        self._manifold: StatisticalManifold | None = None
        self._fim_obj: FisherInformationMatrix | None = None
        self._mu_distance: float = 0.0
        self._threshold: float = float("inf")
        self._fim_norm: float = 0.0

        # Monitoring history: list of (threshold, timestamp_index) tuples.
        self._history: list[tuple[float, int]] = []
        self._calibration_count: int = 0

    # -- Public interface ----------------------------------------------------

    def calibrate(
        self,
        reference_data: np.ndarray[Any, Any],
        calibration_data: np.ndarray[Any, Any] | None = None,
    ) -> float:
        """Calibrate the adaptive threshold from data.

        Args:
            reference_data: In-distribution samples used to define the
                reference point on the manifold (n_ref, d).
            calibration_data: Optional held-out in-distribution samples
                whose geodesic distances determine *mu_distance*.  When
                ``None``, *reference_data* is reused.

        Returns:
            The calibrated threshold value.
        """
        if reference_data.ndim == 1:
            reference_data = reference_data.reshape(-1, 1)

        ref_mean = np.mean(reference_data, axis=0)
        ref_cov = (
            np.cov(reference_data.T)
            if reference_data.shape[0] > 1
            else np.eye(reference_data.shape[1])
        )
        if ref_cov.ndim == 0:
            ref_cov = np.atleast_2d(ref_cov)

        self._manifold = StatisticalManifold(
            ref_mean, ref_cov, tikhonov_lambda=self.tikhonov_lambda
        )

        self._fim_obj = FisherInformationMatrix(tikhonov_lambda=self.tikhonov_lambda)
        fim = self._fim_obj.compute_gaussian(ref_cov)
        self._fim_norm = float(np.linalg.norm(fim, ord="fro"))

        # Compute mean geodesic distance on calibration set.
        cal = calibration_data if calibration_data is not None else reference_data
        if cal.ndim == 1:
            cal = cal.reshape(-1, 1)

        distances = np.array([self._manifold.geodesic_distance(sample) for sample in cal])
        self._mu_distance = float(np.mean(distances))

        # tau = mu_distance + k * sqrt(trace(F^{-1}))
        trace_inv = self._fim_obj.trace_inverse()
        self._threshold = self._mu_distance + self.confidence_k * np.sqrt(max(trace_inv, 0.0))

        self._calibration_count += 1
        self._history.append((self._threshold, self._calibration_count))

        return self._threshold

    @property
    def threshold(self) -> float:
        """Current anomaly threshold."""
        return self._threshold

    @property
    def history(self) -> list[tuple[float, int]]:
        """List of ``(threshold, calibration_index)`` entries."""
        return list(self._history)

    def score(self, sample: np.ndarray[Any, Any]) -> float:
        """Geodesic distance of a single sample to the reference.

        Args:
            sample: Observation vector (d,).

        Returns:
            Non-negative geodesic distance.
        """
        if self._manifold is None:
            raise RuntimeError("Must call calibrate() before scoring.")
        return self._manifold.geodesic_distance(np.asarray(sample).ravel())

    def is_anomalous(self, sample: np.ndarray[Any, Any]) -> bool:
        """Return ``True`` when the sample exceeds the adaptive threshold."""
        return self.score(sample) > self._threshold

    # -- Drift detection and recalibration ----------------------------------

    def check_drift(
        self,
        new_data: np.ndarray[Any, Any],
    ) -> bool:
        """Detect distribution drift by comparing FIM norms.

        Computes the FIM on *new_data* and checks whether its Frobenius
        norm deviates from the reference FIM norm by more than
        *drift_tolerance* (relative).

        Args:
            new_data: Recent observation window (n, d).

        Returns:
            ``True`` if drift is detected.
        """
        if self._fim_obj is None or self._fim_norm == 0.0:
            return False

        if new_data.ndim == 1:
            new_data = new_data.reshape(-1, 1)

        new_fim_obj = FisherInformationMatrix(tikhonov_lambda=self.tikhonov_lambda)
        new_cov = np.cov(new_data.T) if new_data.shape[0] > 1 else np.eye(new_data.shape[1])
        if new_cov.ndim == 0:
            new_cov = np.atleast_2d(new_cov)

        new_fim = new_fim_obj.compute_gaussian(new_cov)
        new_norm = float(np.linalg.norm(new_fim, ord="fro"))

        relative_change = abs(new_norm - self._fim_norm) / max(self._fim_norm, 1e-30)
        return relative_change > self.drift_tolerance

    def recalibrate_if_drifted(
        self,
        new_data: np.ndarray[Any, Any],
    ) -> bool:
        """Check for drift and recalibrate when detected.

        Args:
            new_data: Recent observation window.

        Returns:
            ``True`` if recalibration was performed.
        """
        if self.check_drift(new_data):
            self.calibrate(new_data)
            return True
        return False


# =========================================================================
# InformationGeometryDetector (enhanced)
# =========================================================================


class InformationGeometryDetector:
    """Information geometry-based OOD detector.

    Phase 4C enhancement: when ``adaptive_threshold`` is enabled (the
    default), the detector delegates threshold management to
    :class:`FisherRaoAdaptiveThreshold`, which derives the cutoff from
    the Fisher information metric and recalibrates on drift.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize information geometry detector.

        Args:
            config: Configuration including:
                - distance_metric: 'fisher_rao' or 'kl_divergence'
                  (default: 'fisher_rao')
                - manifold_dim: Dimension of statistical manifold
                  (default: 10)
                - approximation_method: 'closed_form' or 'sampling'
                  (default: 'closed_form')
                - adaptive_threshold: Enable FIM-based adaptive
                  threshold (default: True)
                - confidence_k: Multiplier for adaptive threshold
                  (default: 3.0)
                - drift_tolerance: Relative FIM change that triggers
                  recalibration (default: 0.1)
                - tikhonov_lambda: Regularization strength
                  (default: 1e-6)
        """
        self.config = config or {}
        self.distance_metric: str = self.config.get("distance_metric", "fisher_rao")
        self.manifold_dim: int = self.config.get("manifold_dim", 10)
        self.approximation_method: str = self.config.get("approximation_method", "closed_form")
        self.reference_distribution: dict[str, Any] | None = None
        self.fisher_matrix: np.ndarray[Any, Any] | None = None

        # Phase 4C: adaptive threshold support.
        self._adaptive_enabled: bool = self.config.get("adaptive_threshold", True)
        self._tikhonov_lambda: float = self.config.get("tikhonov_lambda", _DEFAULT_TIKHONOV)
        self._adaptive: FisherRaoAdaptiveThreshold | None = None
        if self._adaptive_enabled:
            self._adaptive = FisherRaoAdaptiveThreshold(
                confidence_k=self.config.get("confidence_k", 3.0),
                drift_tolerance=self.config.get("drift_tolerance", 0.1),
                tikhonov_lambda=self._tikhonov_lambda,
            )

    # -- Fitting -------------------------------------------------------------

    def fit_reference_distribution(self, in_distribution_data: np.ndarray[Any, Any]) -> None:
        """Fit reference distribution from in-distribution training data.

        Args:
            in_distribution_data: Training data from in-distribution (ID).
        """
        self.reference_distribution = {
            "mean": np.mean(in_distribution_data, axis=0),
            "cov": np.cov(in_distribution_data.T),
        }

        self.fisher_matrix = self._compute_fisher_matrix(
            self.reference_distribution["mean"],
            self.reference_distribution["cov"],
        )

        # Calibrate the adaptive threshold if enabled.
        if self._adaptive is not None:
            self._adaptive.calibrate(in_distribution_data)

    def _compute_fisher_matrix(
        self,
        mean: np.ndarray[Any, Any],
        cov: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Compute Fisher Information Matrix.

        For Gaussian distributions, the Fisher matrix has a closed form.

        Args:
            mean: Distribution mean vector.
            cov: Distribution covariance matrix.

        Returns:
            Fisher Information Matrix.
        """
        cov = np.atleast_2d(cov)
        fim_obj = FisherInformationMatrix(tikhonov_lambda=self._tikhonov_lambda)
        return fim_obj.compute_gaussian(cov)

    # -- Distance computation ------------------------------------------------

    def fisher_rao_distance(
        self,
        distribution_1: dict[str, np.ndarray[Any, Any]],
        distribution_2: dict[str, np.ndarray[Any, Any]],
    ) -> float:
        """Compute Fisher-Rao geodesic distance between two distributions.

        The Fisher-Rao distance is the natural distance on statistical
        manifolds.

        Args:
            distribution_1: First distribution ``{'mean': ..., 'cov': ...}``.
            distribution_2: Second distribution ``{'mean': ..., 'cov': ...}``.

        Returns:
            Fisher-Rao distance (geodesic distance on manifold).
        """
        mean_diff = distribution_1["mean"] - distribution_2["mean"]

        if self.fisher_matrix is not None:
            distance = np.sqrt(float(mean_diff.T @ self.fisher_matrix @ mean_diff))
        else:
            distance = float(np.linalg.norm(mean_diff))

        return float(distance)

    # -- Detection -----------------------------------------------------------

    def detect_ood(
        self,
        test_data: np.ndarray[Any, Any],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Detect out-of-distribution samples using information geometry.

        Args:
            test_data: Test samples to evaluate.
            threshold: OOD detection threshold.  When ``None`` and adaptive
                thresholds are enabled the FIM-derived threshold is used;
                otherwise falls back to 3.0.

        Returns:
            Detection results with OOD scores and labels.
        """
        if self.reference_distribution is None:
            raise ValueError(
                "Must fit reference distribution first using " "fit_reference_distribution()"
            )

        test_distribution = {
            "mean": np.mean(test_data, axis=0),
            "cov": (np.cov(test_data.T) if test_data.shape[0] > 1 else np.eye(test_data.shape[1])),
        }

        ood_score = self.fisher_rao_distance(self.reference_distribution, test_distribution)

        # Determine threshold.
        user_provided_threshold = threshold is not None
        if threshold is None:
            if self._adaptive is not None:
                threshold = self._adaptive.threshold
            else:
                threshold = 3.0

        # Check for drift and recalibrate when appropriate.
        recalibrated = False
        if self._adaptive is not None:
            recalibrated = self._adaptive.recalibrate_if_drifted(test_data)
            if recalibrated and not user_provided_threshold:
                threshold = self._adaptive.threshold

        results: dict[str, Any] = {
            "ood_score": ood_score,
            "is_ood": ood_score > threshold,
            "threshold": threshold,
            "method": "fisher_rao_geometry",
            "adaptive": self._adaptive_enabled,
            "recalibrated": recalibrated,
        }

        return results
