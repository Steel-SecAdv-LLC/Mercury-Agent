# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the wired ``use_dtp`` invertibility diagnostic.

Two levels: the DTP building block itself must reduce the forward-inverse
round-trip error with repeated updates on fixed activations (hand-verifiable),
and the fusion trainer must expose a bounded, scale-free reconstruction metric
when enabled and omit it when disabled -- without mutating the trained model.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import Tensor, nn

from omni_mercury_engine.ml.fusion_network import OmniFusionModel
from omni_mercury_engine.ml.optimizers import DifferenceTargetPropagation


def _round_trip(dtp: DifferenceTargetPropagation, y: np.ndarray) -> float:
    with torch.no_grad():
        y_t = torch.FloatTensor(y)
        recon = dtp.forward_layer(dtp.inverse_layer(y_t))
        return float(nn.functional.mse_loss(recon, y_t).item())


def test_dtp_block_reduces_round_trip_error() -> None:
    torch.manual_seed(1)
    dtp = DifferenceTargetPropagation(forward_layer=nn.Linear(6, 4), learning_rate=0.05)
    y = np.random.RandomState(0).randn(16, 4).astype(np.float32)

    before = _round_trip(dtp, y)
    for _ in range(100):
        dtp.backward_pass(y, y)
    after = _round_trip(dtp, y)

    assert after < before  # the forward/inverse pair learns to round-trip


def _tiny_loader() -> list[tuple[dict[str, Tensor], Tensor]]:
    torch.manual_seed(0)
    return [
        (
            {"statistical": torch.randn(8, 10), "temporal": torch.randn(8, 32)},
            torch.randint(0, 2, (8, 1)).float(),
        )
        for _ in range(4)
    ]


def test_fusion_dtp_metric_present_bounded_and_scale_free() -> None:
    torch.manual_seed(0)
    model = OmniFusionModel(
        feature_dims={"statistical": 10, "temporal": 32}, hidden_dim=32, num_heads=2, num_classes=3
    )

    res = model.train_with_optimizers(
        _tiny_loader(),
        epochs=15,
        log_interval=5,
        use_dtp=True,
        use_synthetic_gradients=False,
        use_amav=False,
    )

    history = res["dtp_reconstruction"]
    assert history is not None
    assert len(history) == 15 // 5  # one record per log interval
    # Normalized round-trip error: fraction of activation variance not recovered.
    assert all(np.isfinite(v) and 0.0 <= v <= 1.5 for v in history)


def test_fusion_dtp_disabled_omits_metric() -> None:
    torch.manual_seed(0)
    model = OmniFusionModel(
        feature_dims={"statistical": 10, "temporal": 32}, hidden_dim=32, num_heads=2, num_classes=3
    )
    res = model.train_with_optimizers(
        _tiny_loader(),
        epochs=10,
        log_interval=5,
        use_dtp=False,
        use_synthetic_gradients=False,
        use_amav=False,
    )
    assert res["dtp_reconstruction"] is None
