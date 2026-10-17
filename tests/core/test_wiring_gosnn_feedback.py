# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the wired ``actual_score`` feedback path of GOSNN.

Each call computes the loss *before* stepping the optimizer, so comparing
fresh identically-seeded instances isolates the error-proportional scaling
factor ``min(1 + |target - actual|, 10)`` exactly.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.core.learnable_gosnn import LearnableGOSNN

_NAMES = ["omni_patient_safety"]


def _fresh_loss(target: float, actual: float | None) -> float:
    torch.manual_seed(0)
    gosnn = LearnableGOSNN(n_scalars=180, embedding_dim=8, n_heads=2)
    return gosnn.learn_from_feedback(_NAMES, target, actual_score=actual)


def test_actual_score_scales_loss_proportionally() -> None:
    base = _fresh_loss(1.0, 1.0)  # |target - actual| = 0 -> factor 1.0
    assert base > 0.0

    # actual_score=None must match the zero-discrepancy (factor 1.0) baseline.
    assert abs(_fresh_loss(1.0, None) - base) < 1e-6 * base

    # |1 - (-2)| = 3 -> factor 4.0
    assert abs(_fresh_loss(1.0, -2.0) - 4.0 * base) < 1e-4 * base

    # |1 - 2.5| = 1.5 -> factor 2.5
    assert abs(_fresh_loss(1.0, 2.5) - 2.5 * base) < 1e-4 * base


def test_discrepancy_scaling_is_clipped_at_ten() -> None:
    base = _fresh_loss(1.0, 1.0)
    # |1 - (-19)| = 20 -> factor clipped to 10.0 (not 21.0)
    assert abs(_fresh_loss(1.0, -19.0) - 10.0 * base) < 1e-4 * base
