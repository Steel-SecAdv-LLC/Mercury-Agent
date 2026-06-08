# Copyright (C) 2025 Steel Security Advisors LLC
"""Phase 4D: Riemannian Optimization."""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.linalg

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = [
    "ConstrainedParameterOptimizer",
    "Manifold",
    "OptimizationResult",
    "RiemannianAdam",
    "RiemannianGradientDescent",
    "SPDManifold",
    "SimplexManifold",
]

# ---------------------------------------------------------------------------
# Numerical stability constants
# ---------------------------------------------------------------------------
_EPS: float = 1e-12
_SQRT_EPS: float = 1e-6
_SPD_MIN_EIGENVALUE: float = 1e-8
_DEFAULT_MAX_ITER: int = 500
_DEFAULT_TOL: float = 1e-7


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class OptimizationResult:
    """Result of a Riemannian optimization run."""

    x: np.ndarray[Any, Any]
    objective_value: float
    converged: bool
    num_iterations: int
    gradient_norm: float
    history: list[float] = field(default_factory=list)
    message: str = ""


# ---------------------------------------------------------------------------
# Abstract manifold
# ---------------------------------------------------------------------------
class Manifold(abc.ABC):
    """Abstract base class for a Riemannian manifold."""

    @abc.abstractmethod
    def project(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Project a point onto the manifold."""

    @abc.abstractmethod
    def retraction(self, x: np.ndarray[Any, Any], v: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Retract a tangent vector *v* at *x* back onto the manifold."""

    @abc.abstractmethod
    def exp_map(self, x: np.ndarray[Any, Any], v: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Exponential map: move along the geodesic from *x* in direction *v*."""

    @abc.abstractmethod
    def log_map(self, x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Logarithmic map: tangent vector at *x* pointing toward *y*."""

    @abc.abstractmethod
    def inner_product(
        self, x: np.ndarray[Any, Any], u: np.ndarray[Any, Any], v: np.ndarray[Any, Any]
    ) -> float:
        """Riemannian inner product of tangent vectors *u*, *v* at *x*."""

    @abc.abstractmethod
    def geodesic_distance(self, x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> float:
        """Geodesic distance between two points on the manifold."""

    def norm(self, x: np.ndarray[Any, Any], v: np.ndarray[Any, Any]) -> float:
        """Riemannian norm of tangent vector *v* at *x*."""
        return float(np.sqrt(max(self.inner_product(x, v, v), 0.0)))

    def parallel_transport(
        self,
        x: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
        v: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Transport tangent vector *v* from T_x M to T_y M.

        Default implementation uses the vector-transport-by-retraction (identity), which is a first-
        order approximation.  Subclasses may override with exact parallel transport.
        """
        return v.copy()


# ---------------------------------------------------------------------------
# Probability simplex manifold
# ---------------------------------------------------------------------------
class SimplexManifold(Manifold):
    """The probability simplex Delta_n = {x in R^n : x_i >= 0, sum x_i = 1}.

    The simplex is treated as a Riemannian submanifold of R^n with the
    Fisher information metric (element-wise 1/x_i scaling).  This is the
    natural metric for probability distributions.
    """

    def __init__(self, dimension: int | None = None) -> None:
        """Initialise the simplex manifold.

        Args:
            dimension: Expected dimensionality (used for validation only).
                       Pass ``None`` to skip dimension checks.
        """
        self.dimension = dimension

    # -- helpers ----------------------------------------------------------

    def _validate(self, x: np.ndarray[Any, Any]) -> None:
        if self.dimension is not None and x.shape[-1] != self.dimension:
            msg = f"Expected dimension {self.dimension}, got {x.shape[-1]}"
            raise ValueError(msg)

    @staticmethod
    def _clamp_to_interior(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Clamp point to the strict interior of the simplex."""
        x = np.maximum(x, _EPS)
        result: np.ndarray[Any, Any] = x / x.sum()
        return result

    # -- Manifold interface -----------------------------------------------

    def project(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Project an arbitrary point onto the probability simplex.

        Uses the algorithm of Duchi et al. (2008) "Efficient Projections onto the l1-Ball for
        Learning in High Dimensions".
        """
        self._validate(x)
        n = x.shape[-1]
        # Sort in descending order
        u = np.sort(x)[..., ::-1]
        cumsum = np.cumsum(u, axis=-1)
        rho_candidates = u - (cumsum - 1.0) / np.arange(1, n + 1)
        # rho is the largest index where the candidate is still positive
        rho = n - 1 - np.argmax(rho_candidates[..., ::-1] > 0, axis=-1)
        if np.ndim(rho) == 0:
            theta = (cumsum[..., int(rho)] - 1.0) / (float(rho) + 1.0)
        else:
            # batched case
            theta = np.array([(cumsum[i, r] - 1.0) / (r + 1.0) for i, r in enumerate(rho)])
        projected: np.ndarray[Any, Any] = np.maximum(x - theta[..., np.newaxis], 0.0)
        # Normalise to handle residual floating-point error
        projected = projected / (projected.sum(axis=-1, keepdims=True) + _EPS)
        return projected

    def retraction(self, x: np.ndarray[Any, Any], v: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Retraction: project x + v back onto the simplex."""
        return self.project(x + v)

    def exp_map(self, x: np.ndarray[Any, Any], v: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Exponential map on the simplex with Fisher metric.

        Uses the softmax retraction which is the natural exponential map
        for the simplex under the Fisher information metric:

            exp_x(v)_i = x_i * exp(v_i / x_i) / Z

        where Z is the normalisation constant.
        """
        x_safe = self._clamp_to_interior(x)
        # v_i / x_i gives the natural parameterisation
        scaled = v / x_safe
        # Subtract max for numerical stability (softmax trick)
        log_result = np.log(x_safe + _EPS) + scaled
        log_result -= log_result.max()
        result = np.exp(log_result)
        result = result / (result.sum() + _EPS)
        clamped: np.ndarray[Any, Any] = np.maximum(result, 0.0)
        return clamped

    def log_map(self, x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Logarithmic map: inverse of exp_map.

        Returns tangent vector v at x such that exp_x(v) = y:

            v_i = x_i * (log(y_i) - log(x_i))

        projected onto the tangent space (sum = 0).
        """
        x_safe = self._clamp_to_interior(x)
        y_safe = self._clamp_to_interior(y)
        v = x_safe * (np.log(y_safe + _EPS) - np.log(x_safe + _EPS))
        # Project onto the tangent space of the simplex: sum(v) = 0
        v_proj: np.ndarray[Any, Any] = v - x_safe * v.sum()
        return v_proj

    def inner_product(
        self, x: np.ndarray[Any, Any], u: np.ndarray[Any, Any], v: np.ndarray[Any, Any]
    ) -> float:
        """Fisher information inner product on the simplex.

        <u, v>_x = sum_i (u_i * v_i / x_i)
        """
        x_safe = self._clamp_to_interior(x)
        return float(np.sum(u * v / x_safe))

    def geodesic_distance(self, x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> float:
        """Geodesic distance under the Fisher metric.

        On the simplex with the Fisher metric the geodesic distance is
        related to the Hellinger distance:

            d(x, y) = 2 * arccos(sum(sqrt(x_i * y_i)))

        This is the arc-length of the great circle on the positive
        orthant of the unit sphere (via the map z_i = sqrt(x_i)).
        """
        x_safe = self._clamp_to_interior(x)
        y_safe = self._clamp_to_interior(y)
        cos_angle = np.sum(np.sqrt(x_safe * y_safe))
        # Clamp to valid range for arccos
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        return 2.0 * float(np.arccos(cos_angle))

    def parallel_transport(
        self,
        x: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
        v: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Parallel transport of *v* from T_x to T_y on the simplex.

        Re-scales the tangent vector so it lies in T_y (sum = 0
        constraint) while preserving the Fisher inner product as
        closely as possible.
        """
        y_safe = self._clamp_to_interior(y)
        # Scale by sqrt(y/x) then project onto tangent space at y
        x_safe = self._clamp_to_interior(x)
        scale = np.sqrt(y_safe / (x_safe + _EPS))
        v_transported = v * scale
        # Project onto tangent space: sum(v_transported) must be 0
        v_transported -= y_safe * v_transported.sum()
        return np.asarray(v_transported)


# ---------------------------------------------------------------------------
# Symmetric Positive Definite manifold
# ---------------------------------------------------------------------------
class SPDManifold(Manifold):
    """Manifold of Symmetric Positive Definite (SPD) matrices.

    Uses the affine-invariant Riemannian metric:

        <U, V>_X = trace(X^{-1} U X^{-1} V)

    This metric is invariant under the congruence action
    X -> A X A^T for invertible A, making it natural for covariance
    matrices in statistical models.
    """

    def __init__(
        self,
        size: int | None = None,
        *,
        min_eigenvalue: float = _SPD_MIN_EIGENVALUE,
    ) -> None:
        """Initialise the SPD manifold.

        Args:
            size: Expected matrix size n (for n x n matrices).
                  ``None`` to skip validation.
            min_eigenvalue: Floor for eigenvalues when projecting to
                ensure positive definiteness.
        """
        self.size = size
        self.min_eigenvalue = min_eigenvalue

    # -- helpers ----------------------------------------------------------

    def _validate_shape(self, x: np.ndarray[Any, Any]) -> None:
        if x.ndim < 2 or x.shape[-1] != x.shape[-2]:
            msg = f"Expected square matrix, got shape {x.shape}"
            raise ValueError(msg)
        if self.size is not None and x.shape[-1] != self.size:
            msg = f"Expected {self.size}x{self.size}, got {x.shape}"
            raise ValueError(msg)

    @staticmethod
    def _sym(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Symmetrise a matrix: (X + X^T) / 2."""
        result: np.ndarray[Any, Any] = 0.5 * (x + x.T)
        return result

    def _safe_inv(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute matrix inverse with fallback for near-singular matrices."""
        try:
            result: np.ndarray[Any, Any] = scipy.linalg.inv(x)
            return result
        except scipy.linalg.LinAlgError:
            logger.warning(
                "Singular matrix encountered in SPD inverse; falling back to pseudo-inverse."
            )
            result = scipy.linalg.pinvh(x)
            return result

    def _safe_sqrtm(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute matrix square root, ensuring real and symmetric output."""
        result = scipy.linalg.sqrtm(x)
        result = np.real(result)
        return self._sym(result)

    def _safe_logm(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute matrix logarithm, ensuring real and symmetric output."""
        result = scipy.linalg.logm(x)
        result = np.real(result)
        return self._sym(result)

    def _safe_expm(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute matrix exponential, ensuring real and symmetric output."""
        result = scipy.linalg.expm(x)
        result = np.real(result)
        return self._sym(result)

    # -- Manifold interface -----------------------------------------------

    def project(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Project a matrix onto the SPD cone.

        1. Symmetrise the matrix.
        2. Eigendecompose and clamp eigenvalues to ``min_eigenvalue``.
        """
        self._validate_shape(x)
        x_sym = self._sym(x)
        eigenvalues, eigenvectors = scipy.linalg.eigh(x_sym)
        eigenvalues = np.maximum(eigenvalues, self.min_eigenvalue)
        result: np.ndarray[Any, Any] = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return result

    def retraction(self, x: np.ndarray[Any, Any], v: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Retraction on the SPD manifold.

        Uses the first-order approximation: project(X + V).
        This is cheaper than the full exponential map while staying on
        the manifold.
        """
        return self.project(x + v)

    def exp_map(self, x: np.ndarray[Any, Any], v: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Riemannian exponential map on the SPD manifold.

        exp_X(V) = X^{1/2} expm(X^{-1/2} V X^{-1/2}) X^{1/2}

        where expm is the matrix exponential.
        """
        self._validate_shape(x)
        x_sqrt = self._safe_sqrtm(x)
        x_inv_sqrt = self._safe_inv(x_sqrt)
        inner = x_inv_sqrt @ v @ x_inv_sqrt
        inner = self._sym(inner)
        exp_inner = self._safe_expm(inner)
        result = x_sqrt @ exp_inner @ x_sqrt
        return self._sym(result)

    def log_map(self, x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Riemannian logarithmic map on the SPD manifold.

        log_X(Y) = X^{1/2} logm(X^{-1/2} Y X^{-1/2}) X^{1/2}

        where logm is the matrix logarithm.
        """
        self._validate_shape(x)
        self._validate_shape(y)
        x_sqrt = self._safe_sqrtm(x)
        x_inv_sqrt = self._safe_inv(x_sqrt)
        inner = x_inv_sqrt @ y @ x_inv_sqrt
        inner = self._sym(inner)
        log_inner = self._safe_logm(inner)
        result = x_sqrt @ log_inner @ x_sqrt
        return self._sym(result)

    def inner_product(
        self, x: np.ndarray[Any, Any], u: np.ndarray[Any, Any], v: np.ndarray[Any, Any]
    ) -> float:
        """Affine-invariant Riemannian metric.

        <U, V>_X = trace(X^{-1} U X^{-1} V)
        """
        x_inv = self._safe_inv(x)
        return float(np.trace(x_inv @ u @ x_inv @ v))

    def geodesic_distance(self, x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> float:
        """Affine-invariant geodesic distance.

        d(X, Y) = || logm(X^{-1/2} Y X^{-1/2}) ||_F

        where ||.||_F is the Frobenius norm.
        """
        self._validate_shape(x)
        self._validate_shape(y)
        x_sqrt = self._safe_sqrtm(x)
        x_inv_sqrt = self._safe_inv(x_sqrt)
        inner = x_inv_sqrt @ y @ x_inv_sqrt
        inner = self._sym(inner)
        log_inner = self._safe_logm(inner)
        return float(scipy.linalg.norm(log_inner, "fro"))

    def parallel_transport(
        self,
        x: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
        v: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Parallel transport from T_X to T_Y on the SPD manifold.

        Uses the Schild's ladder approximation:

            E = (Y X^{-1})^{1/2}
            transport(V) = E V E^T
        """
        x_inv = self._safe_inv(x)
        e_squared = y @ x_inv
        e = self._safe_sqrtm(e_squared)
        return self._sym(e @ v @ e.T)


# ---------------------------------------------------------------------------
# Riemannian Gradient Descent
# ---------------------------------------------------------------------------
class RiemannianGradientDescent:
    """Riemannian gradient descent with Armijo backtracking line search.

    Performs updates of the form:

        x_{k+1} = R_x( -alpha_k * grad f(x_k) )

    where R_x is the retraction on the manifold and alpha_k is chosen via
    an Armijo sufficient-decrease condition.
    """

    def __init__(
        self,
        manifold: Manifold,
        *,
        learning_rate: float = 0.01,
        armijo_beta: float = 0.5,
        armijo_sigma: float = 1e-4,
        max_armijo_iters: int = 25,
    ) -> None:
        """Initialise the Riemannian gradient descent optimiser.

        Args:
            manifold: The Riemannian manifold to optimise on.
            learning_rate: Initial step size (before line search).
            armijo_beta: Contraction factor for backtracking (in (0, 1)).
            armijo_sigma: Sufficient decrease parameter (in (0, 1)).
            max_armijo_iters: Maximum number of line-search halvings.
        """
        if not 0 < armijo_beta < 1:
            msg = f"armijo_beta must be in (0, 1), got {armijo_beta}"
            raise ValueError(msg)
        if not 0 < armijo_sigma < 1:
            msg = f"armijo_sigma must be in (0, 1), got {armijo_sigma}"
            raise ValueError(msg)

        self.manifold = manifold
        self.learning_rate = learning_rate
        self.armijo_beta = armijo_beta
        self.armijo_sigma = armijo_sigma
        self.max_armijo_iters = max_armijo_iters

    def _armijo_line_search(
        self,
        x: np.ndarray[Any, Any],
        grad: np.ndarray[Any, Any],
        objective_fn: Callable[[np.ndarray[Any, Any]], float],
        f_x: float,
    ) -> float:
        """Armijo backtracking line search.

        Finds the largest step size alpha = learning_rate * beta^k such that

            f(R_x(-alpha * grad)) <= f(x) - sigma * alpha * ||grad||^2

        Args:
            x: Current point on the manifold.
            grad: Riemannian gradient at x.
            objective_fn: The objective function to minimise.
            f_x: Current objective value f(x).

        Returns:
            Step size satisfying the Armijo condition.
        """
        grad_norm_sq = self.manifold.inner_product(x, grad, grad)
        if grad_norm_sq < _EPS:
            return self.learning_rate

        alpha = self.learning_rate
        for _ in range(self.max_armijo_iters):
            candidate = self.manifold.retraction(x, -alpha * grad)
            f_candidate = objective_fn(candidate)
            decrease = self.armijo_sigma * alpha * grad_norm_sq
            if f_candidate <= f_x - decrease:
                return alpha
            alpha *= self.armijo_beta

        # If line search fails, return smallest step tried
        return alpha

    def step(
        self,
        x: np.ndarray[Any, Any],
        grad: np.ndarray[Any, Any],
        objective_fn: Callable[[np.ndarray[Any, Any]], float] | None = None,
        f_x: float | None = None,
    ) -> np.ndarray[Any, Any]:
        """Perform a single Riemannian gradient descent step.

        Args:
            x: Current point on the manifold.
            grad: Riemannian gradient at x.
            objective_fn: Objective function (needed for Armijo search).
                If ``None``, uses a fixed step size.
            f_x: Objective value at x (to avoid recomputation).

        Returns:
            Updated point on the manifold.
        """
        if objective_fn is not None:
            if f_x is None:
                f_x = objective_fn(x)
            alpha = self._armijo_line_search(x, grad, objective_fn, f_x)
        else:
            alpha = self.learning_rate

        return self.manifold.retraction(x, -alpha * grad)

    def optimize(
        self,
        x0: np.ndarray[Any, Any],
        objective_fn: Callable[[np.ndarray[Any, Any]], float],
        grad_fn: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
        *,
        max_iter: int = _DEFAULT_MAX_ITER,
        tol: float = _DEFAULT_TOL,
    ) -> OptimizationResult:
        """Run Riemannian gradient descent to convergence.

        Args:
            x0: Initial point (should be on the manifold).
            objective_fn: Scalar objective to minimise.
            grad_fn: Returns the Riemannian gradient at a point.
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance on gradient norm.

        Returns:
            An ``OptimizationResult`` with the final iterate and
            diagnostics.
        """
        x = self.manifold.project(x0.copy())
        history: list[float] = []
        converged = False
        grad_norm = float("inf")

        for iteration in range(max_iter):
            f_x = objective_fn(x)
            history.append(f_x)
            grad = grad_fn(x)
            grad_norm = self.manifold.norm(x, grad)

            if grad_norm < tol:
                converged = True
                logger.debug(
                    "RGD converged at iteration %d (grad norm %.2e)",
                    iteration,
                    grad_norm,
                )
                break

            x = self.step(x, grad, objective_fn=objective_fn, f_x=f_x)

        if not converged:
            logger.info(
                "RGD did not converge in %d iterations (final grad norm %.2e).",
                max_iter,
                grad_norm,
            )

        return OptimizationResult(
            x=x,
            objective_value=float(objective_fn(x)),
            converged=converged,
            num_iterations=min(iteration + 1, max_iter) if max_iter > 0 else 0,
            gradient_norm=grad_norm,
            history=history,
            message="converged" if converged else "max_iter_reached",
        )


# ---------------------------------------------------------------------------
# Riemannian Adam
# ---------------------------------------------------------------------------
class RiemannianAdam:
    """Adam optimiser adapted for Riemannian manifolds.

    Maintains exponential moving averages of the gradient (first moment)
    and its squared norm (second moment) in the tangent space, using
    parallel transport to move momentum between iterates.

    Reference:
        Becigneul & Ganea (2019) "Riemannian Adaptive Optimization Methods"
        (ICLR 2019).
    """

    def __init__(
        self,
        manifold: Manifold,
        *,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialise Riemannian Adam.

        Args:
            manifold: The Riemannian manifold to optimise on.
            learning_rate: Base learning rate (alpha).
            beta1: Exponential decay rate for first moment.
            beta2: Exponential decay rate for second moment.
            epsilon: Small constant for numerical stability in the
                denominator.
        """
        if not 0.0 <= beta1 < 1.0:
            msg = f"beta1 must be in [0, 1), got {beta1}"
            raise ValueError(msg)
        if not 0.0 <= beta2 < 1.0:
            msg = f"beta2 must be in [0, 1), got {beta2}"
            raise ValueError(msg)

        self.manifold = manifold
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # State
        self._m: np.ndarray[Any, Any] | None = None  # First moment (tangent vector)
        self._v_scalar: float = 0.0  # Second moment (scalar)
        self._t: int = 0  # Time step

    def reset(self) -> None:
        """Reset optimiser state."""
        self._m = None
        self._v_scalar = 0.0
        self._t = 0

    def step(
        self,
        x: np.ndarray[Any, Any],
        grad: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Perform a single Riemannian Adam step.

        Args:
            x: Current point on the manifold.
            grad: Riemannian gradient at x.

        Returns:
            Updated point on the manifold.
        """
        self._t += 1

        # Initialise moments on first call
        if self._m is None:
            self._m = np.zeros_like(grad)

        # Parallel-transport previous first moment from T_{x_{t-1}} to T_{x_t}
        # On first step, this is just zero so transport is a no-op.
        # For subsequent steps the caller is responsible for providing the
        # gradient at the *current* x; we transport _m from the previous
        # tangent space.

        # Update biased first moment estimate
        self._m = self.beta1 * self._m + (1.0 - self.beta1) * grad

        # Update biased second moment estimate (scalar, per-iteration)
        grad_norm_sq = self.manifold.inner_product(x, grad, grad)
        self._v_scalar = self.beta2 * self._v_scalar + (1.0 - self.beta2) * grad_norm_sq

        # Bias correction
        m_hat = self._m / (1.0 - self.beta1**self._t)
        v_hat = self._v_scalar / (1.0 - self.beta2**self._t)

        # Adaptive step size
        adaptive_lr = self.learning_rate / (np.sqrt(v_hat) + self.epsilon)

        # Retract
        x_new = self.manifold.retraction(x, -adaptive_lr * m_hat)

        # Parallel-transport momentum to new tangent space
        self._m = self.manifold.parallel_transport(x, x_new, self._m)

        return x_new

    def optimize(
        self,
        x0: np.ndarray[Any, Any],
        objective_fn: Callable[[np.ndarray[Any, Any]], float],
        grad_fn: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
        *,
        max_iter: int = _DEFAULT_MAX_ITER,
        tol: float = _DEFAULT_TOL,
    ) -> OptimizationResult:
        """Run Riemannian Adam to convergence.

        Args:
            x0: Initial point (should be on the manifold).
            objective_fn: Scalar objective to minimise.
            grad_fn: Returns the Riemannian gradient at a point.
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance on gradient norm.

        Returns:
            An ``OptimizationResult`` with the final iterate and
            diagnostics.
        """
        self.reset()
        x = self.manifold.project(x0.copy())
        history: list[float] = []
        converged = False
        grad_norm = float("inf")

        for iteration in range(max_iter):
            f_x = objective_fn(x)
            history.append(f_x)
            grad = grad_fn(x)
            grad_norm = self.manifold.norm(x, grad)

            if grad_norm < tol:
                converged = True
                logger.debug(
                    "Riemannian Adam converged at iteration %d (grad norm %.2e)",
                    iteration,
                    grad_norm,
                )
                break

            x = self.step(x, grad)

        if not converged:
            logger.info(
                "Riemannian Adam did not converge in %d iterations (final grad norm %.2e).",
                max_iter,
                grad_norm,
            )

        return OptimizationResult(
            x=x,
            objective_value=float(objective_fn(x)),
            converged=converged,
            num_iterations=min(iteration + 1, max_iter) if max_iter > 0 else 0,
            gradient_norm=grad_norm,
            history=history,
            message="converged" if converged else "max_iter_reached",
        )


# ---------------------------------------------------------------------------
# High-level constrained parameter optimiser
# ---------------------------------------------------------------------------
class ConstrainedParameterOptimizer:
    """High-level API for Riemannian-constrained parameter optimisation.

    Designed for Mercury Agent integration:

    * **OAE weights** live on the probability simplex (they must be
      non-negative and sum to one).  Using Riemannian optimisation
      respects this constraint *exactly* at every iterate, avoiding
      the projection-after-the-fact approach that can cause zig-zagging.

    * **Covariance parameters** live on the SPD manifold.  The
      affine-invariant metric naturally respects positive definiteness.

    Example usage::

        opt = ConstrainedParameterOptimizer()

        # Optimise OAE weights
        result = opt.optimize_simplex_weights(
            initial_weights=np.array([0.4, 0.3, 0.3]),
            objective_fn=my_loss,
            grad_fn=my_grad,
        )
        print(result.x)  # optimised weights on the simplex

        # Optimise a covariance matrix
        result = opt.optimize_spd_parameter(
            initial_matrix=np.eye(3),
            objective_fn=cov_loss,
            grad_fn=cov_grad,
        )
    """

    def __init__(
        self,
        *,
        optimizer_type: str = "adam",
        learning_rate: float = 0.005,
        max_iter: int = _DEFAULT_MAX_ITER,
        tol: float = _DEFAULT_TOL,
        beta1: float = 0.9,
        beta2: float = 0.999,
        armijo_beta: float = 0.5,
        armijo_sigma: float = 1e-4,
    ) -> None:
        """Initialise the constrained parameter optimiser.

        Args:
            optimizer_type: ``"adam"`` or ``"rgd"`` (Riemannian gradient
                descent).
            learning_rate: Base learning rate.
            max_iter: Default maximum iterations.
            tol: Default convergence tolerance.
            beta1: Adam first-moment decay (ignored for ``"rgd"``).
            beta2: Adam second-moment decay (ignored for ``"rgd"``).
            armijo_beta: RGD line-search contraction (ignored for
                ``"adam"``).
            armijo_sigma: RGD line-search sufficient decrease (ignored
                for ``"adam"``).
        """
        valid_types = {"adam", "rgd"}
        if optimizer_type not in valid_types:
            msg = f"optimizer_type must be one of {valid_types}, got {optimizer_type!r}"
            raise ValueError(msg)

        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.beta1 = beta1
        self.beta2 = beta2
        self.armijo_beta = armijo_beta
        self.armijo_sigma = armijo_sigma

    # -- factory helpers --------------------------------------------------

    def _make_optimizer(
        self,
        manifold: Manifold,
    ) -> RiemannianGradientDescent | RiemannianAdam:
        """Build the appropriate optimiser for the given manifold."""
        if self.optimizer_type == "adam":
            return RiemannianAdam(
                manifold,
                learning_rate=self.learning_rate,
                beta1=self.beta1,
                beta2=self.beta2,
            )
        return RiemannianGradientDescent(
            manifold,
            learning_rate=self.learning_rate,
            armijo_beta=self.armijo_beta,
            armijo_sigma=self.armijo_sigma,
        )

    # -- simplex optimisation ---------------------------------------------

    def optimize_simplex_weights(
        self,
        initial_weights: np.ndarray[Any, Any],
        objective_fn: Callable[[np.ndarray[Any, Any]], float],
        grad_fn: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
        *,
        max_iter: int | None = None,
        tol: float | None = None,
    ) -> OptimizationResult:
        """Optimise weights constrained to the probability simplex.

        This is the recommended method for learning OAE weights
        (w_R, w_H, w_O) because it respects the simplex constraint
        intrinsically at every step.

        Args:
            initial_weights: Starting point (will be projected onto the
                simplex if not already there).
            objective_fn: Loss function to minimise.
            grad_fn: Euclidean gradient function.  The method will
                project it onto the tangent space of the simplex
                internally.
            max_iter: Override for ``self.max_iter``.
            tol: Override for ``self.tol``.

        Returns:
            ``OptimizationResult`` with optimised weights.
        """
        _max_iter = max_iter if max_iter is not None else self.max_iter
        _tol = tol if tol is not None else self.tol

        dim = initial_weights.shape[-1]
        manifold = SimplexManifold(dimension=dim)
        optimizer = self._make_optimizer(manifold)

        # Wrap grad_fn to project Euclidean gradient onto simplex
        # tangent space.  The tangent space at x is
        # {v : sum(v) = 0}.  The Riemannian gradient under the Fisher
        # metric is: rgrad_i = x_i * (egrad_i - <egrad, x>)
        def riemannian_grad(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            eg = grad_fn(x)
            # Fisher-metric Riemannian gradient on the simplex
            x_safe = np.maximum(x, _EPS)
            rg: np.ndarray[Any, Any] = x_safe * (eg - np.dot(eg, x_safe))
            return rg

        return optimizer.optimize(
            initial_weights,
            objective_fn,
            riemannian_grad,
            max_iter=_max_iter,
            tol=_tol,
        )

    # -- SPD optimisation -------------------------------------------------

    def optimize_spd_parameter(
        self,
        initial_matrix: np.ndarray[Any, Any],
        objective_fn: Callable[[np.ndarray[Any, Any]], float],
        grad_fn: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
        *,
        max_iter: int | None = None,
        tol: float | None = None,
    ) -> OptimizationResult:
        """Optimise a matrix constrained to be symmetric positive definite.

        This is the recommended method for learning covariance
        parameters in statistical models, because every iterate is
        guaranteed to be SPD.

        Args:
            initial_matrix: Starting SPD matrix (will be projected if
                not already SPD).
            objective_fn: Loss function to minimise.
            grad_fn: Euclidean gradient function (matrix-valued).  The
                method will convert it to a Riemannian gradient.
            max_iter: Override for ``self.max_iter``.
            tol: Override for ``self.tol``.

        Returns:
            ``OptimizationResult`` with the optimised SPD matrix.
        """
        _max_iter = max_iter if max_iter is not None else self.max_iter
        _tol = tol if tol is not None else self.tol

        n = initial_matrix.shape[-1]
        manifold = SPDManifold(size=n)
        optimizer = self._make_optimizer(manifold)

        # Riemannian gradient from Euclidean gradient:
        # rgrad = X @ sym(egrad) @ X
        # (This is the standard formula for the affine-invariant metric.)
        def riemannian_grad(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            eg = grad_fn(x)
            eg_sym = 0.5 * (eg + eg.T)
            rg: np.ndarray[Any, Any] = x @ eg_sym @ x
            return rg

        return optimizer.optimize(
            initial_matrix,
            objective_fn,
            riemannian_grad,
            max_iter=_max_iter,
            tol=_tol,
        )

    # -- OAE integration helper ------------------------------------------

    def optimize_oae_weights(
        self,
        initial_weights: dict[str, float] | np.ndarray[Any, Any] | None = None,
        objective_fn: Callable[[np.ndarray[Any, Any]], float] | None = None,
        grad_fn: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] | None = None,
        *,
        max_iter: int | None = None,
        tol: float | None = None,
    ) -> OptimizationResult:
        """Convenience wrapper specifically for OAE weight optimisation.

        OAE uses three weights (w_R, w_H, w_O) that must lie on the
        probability simplex.  If no objective/gradient are provided a
        default uniform initialisation is returned without optimisation.

        Args:
            initial_weights: Starting weights as a dict with keys
                ``"w_R"``, ``"w_H"``, ``"w_O"`` or a 3-element array.
                Defaults to uniform (1/3, 1/3, 1/3).
            objective_fn: Loss to minimise.
            grad_fn: Euclidean gradient of the loss.
            max_iter: Override for ``self.max_iter``.
            tol: Override for ``self.tol``.

        Returns:
            ``OptimizationResult``.  Access ``result.x`` for the
            optimised weight vector [w_R, w_H, w_O].
        """
        # Parse initial weights
        if initial_weights is None:
            w0 = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        elif isinstance(initial_weights, dict):
            w0 = np.array(
                [
                    initial_weights.get("w_R", 1.0 / 3.0),
                    initial_weights.get("w_H", 1.0 / 3.0),
                    initial_weights.get("w_O", 1.0 / 3.0),
                ]
            )
        else:
            w0 = np.asarray(initial_weights, dtype=np.float64)

        if w0.shape != (3,):
            msg = f"OAE weights must have shape (3,), got {w0.shape}"
            raise ValueError(msg)

        # If no objective is provided, just project and return
        if objective_fn is None or grad_fn is None:
            manifold = SimplexManifold(dimension=3)
            projected = manifold.project(w0)
            return OptimizationResult(
                x=projected,
                objective_value=0.0,
                converged=True,
                num_iterations=0,
                gradient_norm=0.0,
                history=[],
                message="no_objective_provided",
            )

        return self.optimize_simplex_weights(
            w0,
            objective_fn,
            grad_fn,
            max_iter=max_iter,
            tol=tol,
        )

    # Backward compatibility alias
    optimize_aafe_weights = optimize_oae_weights
