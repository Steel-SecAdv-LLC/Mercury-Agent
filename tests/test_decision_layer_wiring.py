# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Integration: the decision layer wires into detect_with_fusion as an opt-in.

Mirrors the conformal wiring tests -- a tiny separable fixture, a short fit, and
a contract check that ``enable_decision_layer()`` makes every detection carry a
``"decision"`` section while leaving the default serve path an exact no-op.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

pytestmark = pytest.mark.xdist_group("decision_layer_wiring")


def _separable_fixture(seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    n_normal, n_anom, dim = 500, 80, 10
    normal = rng.normal(0.0, 1.0, (n_normal, dim))
    anomaly = rng.normal(2.8, 1.0, (n_anom, dim))
    X = np.vstack([normal, anomaly]).astype(np.float32)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anom)]).astype(np.int64)
    order = rng.permutation(len(X))
    return X[order], y[order]


def _engine() -> Any:
    from omni_mercury_engine.engine import OmniMercuryEngine

    return OmniMercuryEngine(mode="fusion", device="cpu")


def _three_way(X: np.ndarray, y: np.ndarray) -> tuple[Any, ...]:
    n = len(X)
    a, b = int(n * 0.5), int(n * 0.75)
    return X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:]


class TestDecisionLayerWiring:
    def test_no_op_until_enabled(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _separable_fixture()
        X_tr, y_tr, _, _, X_te, _ = _three_way(X, y)
        engine = _engine()
        engine.fit_fusion(X_tr, y_tr, epochs=8, batch_size=32)
        # Default serve path: no decision section.
        assert "decision" not in engine.detect_with_fusion(X_te[:1])

    def test_enable_attaches_decision_section(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _separable_fixture()
        X_tr, y_tr, _, _, X_te, _ = _three_way(X, y)
        engine = _engine()
        engine.fit_fusion(X_tr, y_tr, epochs=8, batch_size=32)

        engine.enable_decision_layer()
        result = engine.detect_with_fusion(X_te[:1])

        assert "decision" in result
        decision = result["decision"]
        assert decision["state"] in {"grounded", "unavailable", "undecidable"}
        assert decision["disposition"] in {"act", "clear", "defer", "hold"}
        assert "response" in decision
        assert decision["response"]["action"] in {
            "monitor",
            "alert",
            "recommend_mitigation",
            "escalate_to_human",
            "request_input",
            "hold",
        }
        # The decision rests on the same calibrated probability the result reports.
        assert decision["anomaly_prob"] == pytest.approx(result["anomaly_prob"])

    def test_decision_is_calibrated_when_conformal_is_fit(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _separable_fixture()
        X_tr, y_tr, X_cal, y_cal, X_te, _ = _three_way(X, y)
        engine = _engine()
        engine.fit_fusion(X_tr, y_tr, epochs=15, batch_size=32, early_stopping_patience=10)
        engine.calibrate_fusion_conformal(X_cal, y_cal, coverage=0.9)
        engine.enable_decision_layer()

        result = engine.detect_with_fusion(X_te[:1])
        decision = result["decision"]
        # With a conformal certificate present, the decision is coverage-backed.
        assert decision["calibrated"] is True
        assert result["conformal"]["set_size"] == decision["signals"]["conformal_set_size"]
        if decision["state"] == "grounded":
            assert decision["coverage"] == pytest.approx(0.9)
