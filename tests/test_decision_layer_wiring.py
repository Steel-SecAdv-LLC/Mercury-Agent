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
    def test_attached_by_default(self) -> None:
        """The decision layer is on by default; opting out is explicit.

        This replaces ``test_no_op_until_enabled``, which asserted the opposite
        and pinned the opt-in default in place. Opt-in meant the three
        first-party entry points (CLI, MCP, HTTP) closed the decision loop while
        every library embedder silently did not — the gap
        ``CAPABILITY_MATRIX.md`` now records as closed. A test asserting
        ``"decision" not in result`` would make reopening that gap the passing
        state, so it is inverted rather than deleted: the property still has a
        guard, pointing the other way.
        """
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _separable_fixture()
        X_tr, y_tr, _, _, X_te, _ = _three_way(X, y)
        engine = _engine()
        engine.fit_fusion(X_tr, y_tr, epochs=8, batch_size=32)

        assert "decision" in engine.detect_with_fusion(X_te[:1])

    def test_opt_out_is_explicit_and_honoured(self) -> None:
        """``decision_layer=False`` is the only way off the default path."""
        from omni_mercury_engine.engine import OmniMercuryEngine

        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _separable_fixture()
        X_tr, y_tr, _, _, X_te, _ = _three_way(X, y)
        engine = OmniMercuryEngine(mode="fusion", device="cpu", decision_layer=False)
        engine.fit_fusion(X_tr, y_tr, epochs=8, batch_size=32)

        assert engine.decision_layer is None
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

    def test_optional_ledger_records_decisions(self) -> None:
        from omni_mercury_engine.decision import DecisionLedger

        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _separable_fixture()
        X_tr, y_tr, _, _, X_te, _ = _three_way(X, y)
        engine = _engine()
        engine.fit_fusion(X_tr, y_tr, epochs=8, batch_size=32)

        ledger = DecisionLedger()
        engine.enable_decision_layer(ledger=ledger)
        for i in range(3):
            engine.detect_with_fusion(X_te[i : i + 1])

        # The 'verify' step recorded each detection's decision.
        assert len(ledger) == 3
        summary = ledger.summary()
        assert summary["total"] == 3
        assert sum(summary["by_state"].values()) == 3

    def test_no_ledger_keeps_serve_path_stateless(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _separable_fixture()
        X_tr, y_tr, _, _, X_te, _ = _three_way(X, y)
        engine = _engine()
        engine.fit_fusion(X_tr, y_tr, epochs=8, batch_size=32)

        engine.enable_decision_layer()  # no ledger
        assert engine.decision_ledger is None
        result = engine.detect_with_fusion(X_te[:1])
        assert "decision" in result  # decision still attached, just not recorded
