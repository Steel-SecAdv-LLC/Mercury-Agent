"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from omni_mercury_engine.ml.fairness_regularizers import (
    DemographicParityLoss,
    HSICRegularizer,
    compute_fairness_metrics,
)

# ---------------------------------------------------------------------------
# HSICRegularizer
# ---------------------------------------------------------------------------


def test_hsic_constructor_validates_sigma() -> None:
    with pytest.raises(ValueError):
        HSICRegularizer(sigma=0.0)
    with pytest.raises(ValueError):
        HSICRegularizer(sigma=-0.5)


def test_hsic_constructor_validates_beta() -> None:
    with pytest.raises(ValueError):
        HSICRegularizer(beta=-1.0)


def test_hsic_constructor_validates_kernel() -> None:
    with pytest.raises(ValueError):
        HSICRegularizer(kernel="cubic")  # type: ignore[arg-type]


def test_hsic_returns_zero_for_small_batch() -> None:
    """The empirical HSIC estimator needs m >= 2; otherwise short-circuit to 0."""
    reg = HSICRegularizer()
    X = torch.tensor([[1.0, 2.0]])
    Y = torch.tensor([[0.0]])
    out = reg(X, Y)
    assert out.item() == 0.0


def test_hsic_rejects_mismatched_batch_dim() -> None:
    reg = HSICRegularizer()
    X = torch.randn(8, 4)
    Y = torch.randn(7)
    with pytest.raises(ValueError):
        reg(X, Y)


def test_hsic_is_nonnegative_on_random_input() -> None:
    """HSIC with RBF kernels is non-negative by construction (tr(KHLH)/m^2 >= 0)."""
    torch.manual_seed(0)
    reg = HSICRegularizer(beta=1.0)
    for _ in range(10):
        X = torch.randn(32, 4)
        Y = torch.randn(32, 2)
        assert reg(X, Y).item() >= -1e-9


def test_hsic_detects_dependence() -> None:
    """When Y is a deterministic function of X, HSIC must exceed the independent baseline."""
    torch.manual_seed(42)
    m = 64
    X = torch.randn(m, 2)

    Y_dependent = X.sum(dim=1, keepdim=True)
    Y_independent = torch.randn(m, 1)

    reg = HSICRegularizer(sigma=1.0, beta=1.0)
    dep = reg(X, Y_dependent).item()
    ind = reg(X, Y_independent).item()
    assert dep > ind + 1e-4, f"dependent HSIC {dep} should exceed independent {ind}"


def test_hsic_linear_kernel_branch() -> None:
    """The linear-kernel branch executes and returns a finite scalar."""
    torch.manual_seed(0)
    reg = HSICRegularizer(kernel="linear", beta=1.0)
    X = torch.randn(16, 3)
    Y = torch.randn(16, 1)
    out = reg(X, Y)
    assert out.dim() == 0
    assert torch.isfinite(out).item()


def test_hsic_beta_scaling() -> None:
    """``beta`` linearly scales the returned HSIC value."""
    torch.manual_seed(0)
    X = torch.randn(32, 4)
    Y = X[:, :1] + 0.1 * torch.randn(32, 1)

    reg_low = HSICRegularizer(beta=0.01, sigma=1.0)
    reg_high = HSICRegularizer(beta=1.0, sigma=1.0)
    ratio = reg_high(X, Y).item() / max(reg_low(X, Y).item(), 1e-12)
    assert 99.0 < ratio < 101.0, f"beta scaling broken: ratio={ratio}"


def test_hsic_1d_y_is_promoted_to_2d() -> None:
    """A 1-D ``Y`` tensor is auto-unsqueezed to ``(m, 1)`` and still computes."""
    torch.manual_seed(0)
    reg = HSICRegularizer()
    X = torch.randn(16, 2)
    Y_1d = torch.randint(0, 2, (16,)).float()
    out = reg(X, Y_1d)
    assert torch.isfinite(out).item()


# ---------------------------------------------------------------------------
# DemographicParityLoss
# ---------------------------------------------------------------------------


def test_dpl_zero_for_single_group() -> None:
    loss = DemographicParityLoss(beta=1.0)
    preds = torch.tensor([0.1, 0.2, 0.3, 0.4])
    groups = torch.tensor([0, 0, 0, 0])
    assert loss(preds, groups).item() == 0.0


def test_dpl_zero_when_groups_have_equal_means() -> None:
    loss = DemographicParityLoss(beta=1.0)
    preds = torch.tensor([0.2, 0.4, 0.2, 0.4])  # group means: 0.3 and 0.3
    groups = torch.tensor([0, 0, 1, 1])
    assert loss(preds, groups).item() == pytest.approx(0.0, abs=1e-7)


def test_dpl_positive_when_groups_differ() -> None:
    loss = DemographicParityLoss(beta=1.0)
    preds = torch.tensor([0.1, 0.1, 0.9, 0.9])  # group means: 0.1 and 0.9
    groups = torch.tensor([0, 0, 1, 1])
    assert loss(preds, groups).item() == pytest.approx(0.8, abs=1e-6)


def test_dpl_three_groups_disparity() -> None:
    """Three groups: loss is max - min of group means."""
    loss = DemographicParityLoss(beta=1.0)
    preds = torch.tensor([0.1, 0.3, 0.5, 0.7])
    groups = torch.tensor([0, 1, 2, 2])  # means: 0.1, 0.3, 0.6
    assert loss(preds, groups).item() == pytest.approx(0.5, abs=1e-6)


def test_dpl_beta_scales_output() -> None:
    preds = torch.tensor([0.0, 1.0])
    groups = torch.tensor([0, 1])
    assert DemographicParityLoss(beta=0.5)(preds, groups).item() == pytest.approx(0.5)
    assert DemographicParityLoss(beta=2.0)(preds, groups).item() == pytest.approx(2.0)


def test_dpl_rejects_mismatched_lengths() -> None:
    loss = DemographicParityLoss()
    with pytest.raises(ValueError):
        loss(torch.tensor([0.1, 0.2]), torch.tensor([0, 0, 1]))


def test_dpl_constructor_validates_beta() -> None:
    with pytest.raises(ValueError):
        DemographicParityLoss(beta=-1.0)


# ---------------------------------------------------------------------------
# compute_fairness_metrics
# ---------------------------------------------------------------------------


def _hand_metrics_simple() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4 samples per group, hand-checkable confusion matrices.

    Group 0: preds=[1,1,0,0]  labels=[1,0,1,0]  -> TP=1 FP=1 TN=1 FN=1
        accuracy=0.5  precision=0.5  recall=0.5  TPR=0.5  FPR=0.5  pos_rate=0.5
    Group 1: preds=[1,1,1,0]  labels=[1,1,0,0]  -> TP=2 FP=1 TN=1 FN=0
        accuracy=0.75 precision=2/3 recall=1.0   TPR=1.0  FPR=0.5  pos_rate=0.75
    """
    preds = np.array([1, 1, 0, 0, 1, 1, 1, 0], dtype=np.float64)
    labels = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.float64)
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64)
    return preds, labels, groups


def test_compute_fairness_metrics_demographic_parity_is_max_minus_min() -> None:
    preds, labels, groups = _hand_metrics_simple()
    out = compute_fairness_metrics(preds, labels, groups)
    # 0.75 (group 1) - 0.5 (group 0)
    assert out["demographic_parity"] == pytest.approx(0.25, abs=1e-6)


def test_compute_fairness_metrics_equalized_odds_picks_worst_gap() -> None:
    preds, labels, groups = _hand_metrics_simple()
    out = compute_fairness_metrics(preds, labels, groups)
    # TPR gap: 1.0 - 0.5 = 0.5; FPR gap: 0.5 - 0.5 = 0.0; max = 0.5
    assert out["equalized_odds"] == pytest.approx(0.5, abs=1e-6)


def test_compute_fairness_metrics_per_group_dict() -> None:
    preds, labels, groups = _hand_metrics_simple()
    out = compute_fairness_metrics(preds, labels, groups)
    per_group = out["per_group_metrics"]
    assert set(per_group.keys()) == {0, 1}

    g0 = per_group[0]
    assert g0["positive_rate"] == pytest.approx(0.5, abs=1e-6)
    assert g0["accuracy"] == pytest.approx(0.5, abs=1e-6)
    assert g0["tpr"] == pytest.approx(0.5, abs=1e-6)
    assert g0["fpr"] == pytest.approx(0.5, abs=1e-6)

    g1 = per_group[1]
    assert g1["positive_rate"] == pytest.approx(0.75, abs=1e-6)
    assert g1["accuracy"] == pytest.approx(0.75, abs=1e-6)
    assert g1["tpr"] == pytest.approx(1.0, abs=1e-6)
    assert g1["fpr"] == pytest.approx(0.5, abs=1e-6)


def test_compute_fairness_metrics_single_group_zero_disparity() -> None:
    preds = np.array([1, 1, 0, 0], dtype=np.float64)
    labels = np.array([1, 0, 1, 0], dtype=np.float64)
    groups = np.array([0, 0, 0, 0], dtype=np.float64)
    out = compute_fairness_metrics(preds, labels, groups)
    assert out["demographic_parity"] == 0.0
    assert out["equalized_odds"] == 0.0
    assert set(out["per_group_metrics"].keys()) == {0}


def test_compute_fairness_metrics_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        compute_fairness_metrics(
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0, 1.0]),
            np.array([0.0, 0.0]),
        )
