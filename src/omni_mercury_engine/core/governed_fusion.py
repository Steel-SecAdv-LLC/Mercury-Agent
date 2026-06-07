"""Governed fusion primitives lifted from FINDOYOU mercury_equation.

Attribution: formulas and API shape are adapted from FINDOYOU
``mercury_equation/{certify,dependence}.py``. This module is NumPy-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_EPS = 1e-12


def precision_sqrt(precision: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return symmetric PSD ``A`` such that ``A.T @ A ~= precision``."""
    theta = np.asarray(precision, dtype=np.float64)
    theta = 0.5 * (theta + theta.T)
    vals, vecs = np.linalg.eigh(theta)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


@dataclass(frozen=True)
class JointCertificate:
    """Mahalanobis certified-radius and witness calculator."""

    loc: np.ndarray[Any, Any]
    precision: np.ndarray[Any, Any]
    p_tau: float
    _smax: float = 0.0

    def __post_init__(self) -> None:
        loc = np.asarray(self.loc, dtype=np.float64).reshape(-1)
        precision = np.asarray(self.precision, dtype=np.float64)
        object.__setattr__(self, "loc", loc)
        object.__setattr__(self, "precision", 0.5 * (precision + precision.T))
        object.__setattr__(
            self,
            "_smax",
            float(np.sqrt(max(np.linalg.eigvalsh(self.precision).max(initial=0.0), 0.0))),
        )

    def price(self, logits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return Mahalanobis distance for each row in *logits*."""
        z = np.atleast_2d(np.asarray(logits, dtype=np.float64)) - self.loc
        mahal = np.einsum("ij,jk,ik->i", z, self.precision, z)
        return np.sqrt(np.maximum(mahal, 0.0))

    def certified_radius(self, logits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return the L2 certified radius for each row in *logits*."""
        return np.abs(self.price(logits) - self.p_tau) / (self._smax + _EPS)

    def witness(self, logits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return the witness gradient for each row in *logits*."""
        x = np.atleast_2d(np.asarray(logits, dtype=np.float64))
        z = x - self.loc
        p = self.price(x) + _EPS
        return (z @ self.precision) / p[:, None]

    def witness_channel(self, logits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return the index of the most influential channel per row."""
        return np.argmax(np.abs(self.witness(logits)), axis=1)

    def certify(self, logits: np.ndarray[Any, Any]) -> dict[str, np.ndarray[Any, Any]]:
        """Compute full certificate payload (price, radius, witness, channel)."""
        return {
            "price": self.price(logits),
            "certified_l2_radius": self.certified_radius(logits),
            "witness": self.witness(logits),
            "witness_channel": self.witness_channel(logits),
        }


def mahalanobis_score_to_price_threshold(threshold: float, n_features: int) -> float:
    """Invert ``score=1-exp(-p^2/(2d))`` to the Mahalanobis price threshold."""
    t = float(np.clip(threshold, 0.0, 1.0 - 1e-12))
    return float(np.sqrt(max(-2.0 * max(n_features, 1) * np.log1p(-t), 0.0)))


def pgd_flip_distance(
    certificate: JointCertificate,
    point: np.ndarray[Any, Any],
    *,
    steps: int = 120,
    step_size: float = 0.05,
) -> float:
    """Small deterministic PGD probe distance to the Mahalanobis boundary."""
    x0 = np.asarray(point, dtype=np.float64).reshape(1, -1)
    x = x0.copy()
    start_side = certificate.price(x0)[0] >= certificate.p_tau
    best = float("inf")
    for _ in range(steps):
        grad = certificate.witness(x)[0]
        direction = -grad if start_side else grad
        norm = float(np.linalg.norm(direction))
        if norm <= _EPS:
            break
        x = x + step_size * direction.reshape(1, -1) / norm
        if (certificate.price(x)[0] >= certificate.p_tau) != start_side:
            best = min(best, float(np.linalg.norm(x - x0)))
            break
    return best
