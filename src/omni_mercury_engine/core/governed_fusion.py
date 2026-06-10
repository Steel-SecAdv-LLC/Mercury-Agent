# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Info-geometry component certificate primitives (NumPy-only).

Scope (read this before using the certificate): these primitives certify the
**information-geometry component's** Mahalanobis price level-set — the radius in
input space within which that one component's price ``p(x)`` cannot cross its
own operating threshold ``p_tau``.  They do **not** certify the fused/gated
verdict.  A certificate on the actual fused decision would have to bound the
Lipschitz constant through the neural fusion, the calibration map, and both
hard ethics gates; that is a separate, larger task and is deliberately out of
scope here.

Soundness of the radius: with ``p(x) = sqrt((x-mu)^T Theta (x-mu))`` the
gradient is ``grad p = Theta (x-mu) / p`` and ``||grad p|| <= sigma_max(A)``
where ``A = Theta^{1/2}`` and ``sigma_max(A) = sqrt(lambda_max(Theta))``.  Hence
``p`` is ``sigma_max(A)``-Lipschitz and cannot move from ``p`` to ``p_tau``
within an L2 ball of radius ``|p - p_tau| / sigma_max(A)``.

Attribution: the Mahalanobis-radius construction is adapted as a *blueprint*
from FINDOYOU certified-radius work; the operating point, threshold inversion,
and the numbers Mercury reports are Mercury's own.
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
class InfoGeometryCertificate:
    """Mahalanobis certified-radius/witness for the info-geometry component.

    Certifies the info-geometry component's price level-set only: ``p_tau`` is
    that component's own operating threshold mapped into price space, and
    ``certified_radius`` is the sound L2 radius within which the component's
    price cannot cross ``p_tau`` (see module docstring).  It says nothing about
    the fused or gated verdict.
    """

    loc: np.ndarray[Any, Any]
    precision: np.ndarray[Any, Any]
    p_tau: float
    _smax: float = 0.0

    def __post_init__(self) -> None:
        """Normalise ``loc``/``precision`` and cache the precision spectral radius."""
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
    """Inverse of the info-geometry component's score map ``g``.

    The component maps Mahalanobis price ``p`` to a score via
    ``g(p) = 1 - exp(-p^2 / (2 * n_features))`` (see
    ``MercuryAnomalyDetector._compute_info_geometry_score``).  This returns
    ``g^{-1}(threshold)`` — the price at which the component's score equals
    ``threshold``.  Passing the **component's own** operating threshold yields a
    ``p_tau`` on the component's real decision boundary; passing an unrelated
    (e.g. ensemble) threshold does not, which was the original defect.
    """
    t = float(np.clip(threshold, 0.0, 1.0 - 1e-12))
    return float(np.sqrt(max(-2.0 * max(n_features, 1) * np.log1p(-t), 0.0)))


def pgd_flip_distance(
    certificate: InfoGeometryCertificate,
    point: np.ndarray[Any, Any],
    *,
    steps: int = 120,
    step_size: float = 0.05,
) -> float:
    """Deterministic PGD probe distance to the component's price boundary."""
    x0 = np.asarray(point, dtype=np.float64).reshape(1, -1)
    x = x0.copy()
    start_side = certificate.price(x0)[0] >= certificate.p_tau
    for _ in range(steps):
        grad = certificate.witness(x)[0]
        direction = -grad if start_side else grad
        norm = float(np.linalg.norm(direction))
        if norm <= _EPS:
            break
        x = x + step_size * direction.reshape(1, -1) / norm
        if (certificate.price(x)[0] >= certificate.p_tau) != start_side:
            # Boundary crossed: this displacement is the probe's flip distance.
            return float(np.linalg.norm(x - x0))
    # No flip within the step budget. Return the distance actually walked --
    # a finite *lower bound* on the true flip distance -- rather than ``inf``,
    # which would silently satisfy ``>=`` comparisons and mask the fact that
    # the probe never crossed the price boundary.
    return float(np.linalg.norm(x - x0))
