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
Training-time fairness regularizers for anomaly detectors.

Two training-time penalties and one offline auditing helper:

* :class:`HSICRegularizer` — Hilbert-Schmidt Independence Criterion
  (Gretton et al. 2005). Penalises statistical dependence between a model's
  internal features and a sensitive attribute, encouraging the detector to
  reach the same anomaly score distribution irrespective of the protected
  group. The empirical estimator used here is
  ``HSIC(X, Y) = tr(K H L H) / m^2`` with RBF (Gaussian) kernels ``K`` and
  ``L`` and the centering matrix ``H = I - (1/m) 1 1^T``; it is
  non-negative and is zero iff ``X`` and ``Y`` are independent.
* :class:`DemographicParityLoss` — direct soft penalty on the absolute
  difference between mean predicted scores across demographic groups.
* :func:`compute_fairness_metrics` — offline diagnostic returning
  demographic parity, equalized odds, and per-group precision / recall.

These regularizers are training-time aids only and do not on their own
constitute a fairness guarantee for downstream decisions; use them
together with the offline metrics and an independent audit.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn


class HSICRegularizer(nn.Module):
    """
    HSIC penalty for independence between features and sensitive attributes.

    The HSIC empirical estimator with RBF kernels is

    .. math::

        \\widehat{\\mathrm{HSIC}}(X, Y) = \\frac{1}{m^2}\\, \\operatorname{tr}(K H L H)

    where ``K_{ij} = exp(-||X_i - X_j||^2 / (2 \\sigma^2))``, similarly for
    ``L`` on ``Y``, and ``H = I - (1/m) 1 1^T`` is the centering matrix.
    The estimator is non-negative, and convexity of HSIC with RBF kernels
    ensures the perturbation to the loss surface keeps the smallest
    eigenvalue ``σ_min > 0`` (Weyl's inequality), so the regularised
    objective stays positive-definite in a neighbourhood of the optimum.

    Reference:
        Gretton, A. et al. *Measuring Statistical Dependence with
        Hilbert-Schmidt Norms.* ALT 2005.
    """

    def __init__(self, sigma: float = 1.0, beta: float = 0.001, kernel: str = "rbf") -> None:
        """
        Args:
            sigma: RBF kernel bandwidth.
            beta: Scalar weight applied to the HSIC value before it is added
                to the host loss.
            kernel: Kernel type. ``"rbf"`` (default) or ``"linear"``.
        """
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if kernel not in {"rbf", "linear"}:
            raise ValueError(f"Unknown kernel: {kernel!r}")
        self.sigma = sigma
        self.beta = beta
        self.kernel = kernel

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the (scaled) empirical HSIC between ``X`` and ``Y``.

        Args:
            X: Features tensor of shape ``(m, d_x)``.
            Y: Sensitive-attribute tensor of shape ``(m, d_y)`` or a 1-D
                label tensor of length ``m``.

        Returns:
            Scalar tensor ``beta * HSIC(X, Y)``.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y must have matching batch dimension; got {X.shape[0]} vs {Y.shape[0]}"
            )

        m = X.shape[0]
        if m < 2:
            return torch.tensor(0.0, device=X.device)

        if Y.dim() == 1:
            Y = Y.unsqueeze(1)

        K = self._compute_kernel(X, X)
        L = self._compute_kernel(Y.to(dtype=X.dtype), Y.to(dtype=X.dtype))

        H = torch.eye(m, device=X.device, dtype=X.dtype) - (1.0 / m) * torch.ones(
            m, m, device=X.device, dtype=X.dtype
        )

        KH = torch.mm(K, H)
        LH = torch.mm(L, H)
        hsic_value = torch.trace(torch.mm(KH, LH)) / (m**2)
        return self.beta * hsic_value

    def _compute_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if self.kernel == "rbf":
            return self._rbf_kernel(X, Y)
        return torch.mm(X, Y.t())

    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X_norm = (X**2).sum(1).view(-1, 1)
        Y_norm = (Y**2).sum(1).view(1, -1)
        dist_sq = X_norm + Y_norm - 2.0 * torch.mm(X, Y.t())
        dist_sq = torch.clamp(dist_sq, min=0.0)
        return torch.exp(-dist_sq / (2 * self.sigma**2))


class DemographicParityLoss(nn.Module):
    """
    Soft demographic-parity penalty.

    Computes ``max(group_mean) - min(group_mean)`` over the predicted
    scores partitioned by the integer ``groups`` tensor and returns
    ``beta * disparity``. The loss is zero when every group's mean
    predicted score is equal — which is exactly the demographic-parity
    criterion ``P(Ŷ=1 | A=a)`` constant across ``a``.
    """

    def __init__(self, beta: float = 0.01) -> None:
        """
        Args:
            beta: Scalar weight applied before the loss is added to the
                host objective.
        """
        super().__init__()
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        self.beta = beta

    def forward(self, predictions: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Anomaly scores of shape ``(m,)``.
            groups: Integer group labels of shape ``(m,)``.

        Returns:
            Scalar tensor — zero when there is only one represented group.
        """
        if predictions.shape[0] != groups.shape[0]:
            raise ValueError(
                "predictions and groups must have matching length; "
                f"got {predictions.shape[0]} vs {groups.shape[0]}"
            )

        unique_groups = torch.unique(groups)
        if unique_groups.numel() < 2:
            return torch.tensor(0.0, device=predictions.device)

        group_rates: list[torch.Tensor] = []
        for group in unique_groups:
            mask = groups == group
            if mask.sum() > 0:
                group_rates.append(predictions[mask].mean())

        if len(group_rates) < 2:
            return torch.tensor(0.0, device=predictions.device)

        rates = torch.stack(group_rates)
        return self.beta * (rates.max() - rates.min()).abs()


def compute_fairness_metrics(
    predictions: npt.NDArray[np.float64],
    labels: npt.NDArray[np.float64],
    groups: npt.NDArray[np.float64],
) -> dict[str, Any]:
    """
    Offline diagnostic for fairness of binary anomaly predictions.

    Args:
        predictions: Binary predictions, shape ``(n,)``, values in ``{0, 1}``.
        labels: Ground-truth binary labels, shape ``(n,)``.
        groups: Integer group labels, shape ``(n,)``.

    Returns:
        Dict with::

            {
                "demographic_parity": max_group_positive_rate - min_group_positive_rate,
                "equalized_odds": max over (TPR-gap, FPR-gap),
                "per_group_metrics": {
                    int(group): {
                        "accuracy", "precision", "recall",
                        "tpr", "fpr", "positive_rate",
                    },
                },
            }

        Per-group denominators are protected by a 1e-8 floor so that a
        group with no positives still yields a finite (but uninformative)
        precision / recall value rather than a divide-by-zero.
    """
    if not (predictions.shape == labels.shape == groups.shape):
        raise ValueError(
            "predictions, labels, groups must have matching shape; got "
            f"{predictions.shape}, {labels.shape}, {groups.shape}"
        )

    unique_groups = np.unique(groups)
    per_group_metrics: dict[int, dict[str, float]] = {}
    positive_rates: list[float] = []
    tprs: list[float] = []
    fprs: list[float] = []

    for group in unique_groups:
        mask = groups == group
        if not mask.any():
            continue

        gp = predictions[mask]
        gl = labels[mask]

        positive_rate = float(gp.mean())
        positive_rates.append(positive_rate)

        tp = float(((gp == 1) & (gl == 1)).sum())
        fp = float(((gp == 1) & (gl == 0)).sum())
        tn = float(((gp == 0) & (gl == 0)).sum())
        fn = float(((gp == 0) & (gl == 1)).sum())

        accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        tpr = recall
        fpr = fp / (fp + tn + 1e-8)

        tprs.append(tpr)
        fprs.append(fpr)

        per_group_metrics[int(group)] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "tpr": tpr,
            "fpr": fpr,
            "positive_rate": positive_rate,
        }

    demographic_parity = (
        max(positive_rates) - min(positive_rates) if len(positive_rates) > 1 else 0.0
    )
    equalized_odds = max(
        max(tprs) - min(tprs) if len(tprs) > 1 else 0.0,
        max(fprs) - min(fprs) if len(fprs) > 1 else 0.0,
    )

    return {
        "demographic_parity": demographic_parity,
        "equalized_odds": equalized_odds,
        "per_group_metrics": per_group_metrics,
    }
