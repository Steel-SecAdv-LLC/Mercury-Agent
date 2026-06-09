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

"""``OmniMercuryEngine.decide_and_respond`` wires the layer onto real detection.

Network-free torch tier: trains a tiny fusion model on a deterministic separable
fixture and drives the full identify -> interpret -> decide -> deter -> verify
loop, asserting the engine surface and the safety invariants (the response
boundary re-asserts the engine's own dual hard ethical gate before any
actuation, so a passed gate is the contract).
"""

import json
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

pytestmark = pytest.mark.xdist_group("decide_and_respond")

_VALID_VERDICTS = {"positive", "negative", "abstain"}
_VALID_STATES = {"grounded", "unavailable"}
_VALID_STATUSES = {"noop", "applied", "deferred", "blocked"}


def _separable_fixture(seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    n_normal, n_anom, dim = 600, 90, 12
    normal = rng.normal(0.0, 1.0, (n_normal, dim))
    anomaly = rng.normal(2.6, 1.0, (n_anom, dim))
    features = np.vstack([normal, anomaly]).astype(np.float32)
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anom)]).astype(np.int64)
    order = rng.permutation(len(features))
    return features[order], labels[order]


def _trained_engine() -> Any:
    from omni_mercury_engine.engine import OmniMercuryEngine

    torch.manual_seed(0)
    np.random.seed(0)
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    features, labels = _separable_fixture()
    cut = int(len(features) * 0.6)
    engine.fit_fusion(
        features[:cut], labels[:cut], epochs=15, batch_size=32, early_stopping_patience=10
    )
    return engine, features[cut:], labels[cut:]


class TestDecideAndRespondSurface:
    """The engine method returns the base detection plus the decision certificate."""

    def test_augments_detection_with_decision_response_loop(self) -> None:
        engine, x_test, _ = _trained_engine()
        result = engine.decide_and_respond(x_test[:1], domain="network_security")

        # Base detection fields survive.
        assert "anomaly_prob" in result
        assert "is_anomaly" in result
        # New certificate fields.
        assert {"decision", "response", "loop"} <= set(result)

        decision = result["decision"]
        assert decision["verdict"] in _VALID_VERDICTS
        assert decision["state"] in _VALID_STATES
        # The decision's confidence is exactly the calibrated probability decided on.
        assert decision["confidence"] == pytest.approx(result["anomaly_prob"], abs=1e-6)

        response = result["response"]
        assert response["status"] in _VALID_STATUSES
        # Any non-NOOP response ran the engine's dual hard ethical gate and passed
        # (the same boundary detect_with_fusion already cleared for this input).
        if response["status"] != "noop":
            assert response["ethical_gate_passed"] is True

    def test_loop_certificate_is_json_serializable(self) -> None:
        engine, x_test, _ = _trained_engine()
        result = engine.decide_and_respond(x_test[:1], domain="energy")
        json.dumps(result["loop"])
        assert result["loop"]["three_state"] in _VALID_STATES

    def test_conformal_calibration_drives_the_decision(self) -> None:
        engine, x_test, y_test = _trained_engine()
        # Before calibration the band fallback decides (no prediction set).
        before = engine.decide_and_respond(x_test[:1])
        assert before["decision"]["prediction_set"] is None

        engine.calibrate_fusion_conformal(x_test[:60], y_test[:60], coverage=0.9)
        after = engine.decide_and_respond(x_test[60:61])
        assert after["decision"]["prediction_set"] is not None
        assert set(after["decision"]["prediction_set"]).issubset({0, 1})
        assert after["decision"]["coverage"] == pytest.approx(0.9)

    def test_shared_loop_accumulates_an_audit_ledger(self) -> None:
        from omni_mercury_engine.decision import DecisionResponseLoop, permit_all_gate

        engine, x_test, _ = _trained_engine()
        loop = DecisionResponseLoop(ethical_gate=permit_all_gate)
        for sample in x_test[:5]:
            engine.decide_and_respond(sample.reshape(1, -1), loop=loop)
        assert len(loop.ledger) == 5
        assert loop.ledger.summary()["total"] == 5
