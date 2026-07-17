# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The GOSNN consequential decision channel: isolation and routing contracts.

The fused state's only consequential outlet is the decision layer's
disagreement overlay (``gosnn_metadata["detection"]`` ->
``Evidence.gosnn_anomaly_prob`` -> demotion to ``DEFER``).  Two contracts are
pinned here at the engine level:

1. **Isolation (the ``default_fusion.pt`` guard).**  Attaching or removing the
   GOSNN detection head must not move a single bit of the ``OmniFusionModel``
   input features, the calibrated ``anomaly_prob``, or the σ_Immutable
   verdict.  The shipped ``default_fusion.pt`` was trained against a fixed
   feature-group distribution (see ``models/affective.py``) -- this test is
   the proof the channel does not shift it.

2. **Routing.**  With a head attached and the decision layer enabled, strong
   disagreement demotes the grounded verdict to a deferral; the signals block
   carries the evidence.  Without a head the channel is structurally absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "adbench" / "pima_real.npz"

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="pima ADBench fixture not present")


@pytest.fixture(scope="module")
def fitted_engine() -> Any:
    """A real engine fitted on a small slice of the pima fixture."""
    from omni_mercury_engine.engine import OmniMercuryEngine

    data = np.load(FIXTURE)
    x = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.int64)
    half = len(x) // 2
    engine = OmniMercuryEngine()
    rng = np.random.default_rng(0)
    fit_idx = rng.choice(half, size=50, replace=False)
    engine.fit_fusion(x[:half][fit_idx], y[:half][fit_idx])
    return engine


@pytest.fixture()
def stub_head(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Attach a deterministic detection head + thresholds to the GOSNN singleton."""
    from omni_mercury_engine.core.attention_fusion_stack import FusionDetectionHead
    from omni_mercury_engine.core.global_omni_scalar_network import get_global_scalar_network

    fusion = get_global_scalar_network().attention_fusion
    torch.manual_seed(20260717)
    head = FusionDetectionHead()
    head.eval()
    monkeypatch.setattr(fusion, "detection_head", head)
    monkeypatch.setattr(
        fusion,
        "decision_thresholds",
        {"demote_act_below": 0.35, "demote_clear_above": 0.65},
    )
    return fusion


def _row(index: int = 3) -> np.ndarray:
    data = np.load(FIXTURE)
    x = np.asarray(data["X"], dtype=np.float64)
    return x[len(x) // 2 + index : len(x) // 2 + index + 1]


class TestDetectionBlockSurface:
    def test_block_present_iff_head_loaded(self, fitted_engine: Any) -> None:
        """The detection block's presence mirrors the shipped-head state."""
        from omni_mercury_engine.core.global_omni_scalar_network import get_global_scalar_network

        fusion = get_global_scalar_network().attention_fusion
        result = fitted_engine.detect_with_fusion(_row())
        meta = result["gosnn_metadata"]
        assert ("detection" in meta) == (fusion.detection_head is not None)

    def test_block_carries_probability_and_thresholds(
        self, fitted_engine: Any, stub_head: Any
    ) -> None:
        result = fitted_engine.detect_with_fusion(_row())
        detection = result["gosnn_metadata"]["detection"]
        assert 0.0 <= detection["anomaly_prob"] <= 1.0
        assert detection["demote_act_below"] == pytest.approx(0.35)
        assert detection["demote_clear_above"] == pytest.approx(0.65)
        assert detection["backend"] == "gosnn_detection_head"


class TestChannelIsolation:
    """The default_fusion.pt guard: the channel must not touch the fusion path."""

    def test_fusion_inputs_prob_and_sigma_identical_with_and_without_head(
        self, fitted_engine: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.core.global_omni_scalar_network import get_global_scalar_network

        fusion = get_global_scalar_network().attention_fusion
        recorded: list[Any] = []
        original_predict = fitted_engine.fusion_inference.predict

        def recording_predict(features: Any, *args: Any, **kwargs: Any) -> Any:
            recorded.append({k: np.array(v, copy=True) for k, v in features.items()})
            return original_predict(features, *args, **kwargs)

        monkeypatch.setattr(fitted_engine.fusion_inference, "predict", recording_predict)

        row = _row()

        def run() -> Any:
            return fitted_engine.detect_with_fusion(row)

        # Without a head (observability-only posture).
        monkeypatch.setattr(fusion, "detection_head", None)
        monkeypatch.setattr(fusion, "decision_thresholds", None)
        without_head = run()

        # With a deterministic stub head + thresholds.
        from omni_mercury_engine.core.attention_fusion_stack import FusionDetectionHead

        torch.manual_seed(20260717)
        head = FusionDetectionHead()
        head.eval()
        monkeypatch.setattr(fusion, "detection_head", head)
        monkeypatch.setattr(
            fusion,
            "decision_thresholds",
            {"demote_act_below": 0.35, "demote_clear_above": 0.65},
        )
        with_head = run()

        assert "detection" not in without_head["gosnn_metadata"]
        assert "detection" in with_head["gosnn_metadata"]

        # Bit-identical OmniFusionModel inputs: the head cannot shift the
        # trained feature distribution of default_fusion.pt.
        assert len(recorded) == 2
        assert recorded[0].keys() == recorded[1].keys()
        for key in recorded[0]:
            np.testing.assert_array_equal(
                recorded[0][key],
                recorded[1][key],
                err_msg=f"fusion input feature group {key!r} moved when the "
                "GOSNN detection head was attached",
            )

        # Identical calibrated verdict and σ_Immutable outcome.
        assert with_head["anomaly_prob"] == without_head["anomaly_prob"]
        assert with_head["is_anomaly"] == without_head["is_anomaly"]
        assert with_head["threshold_used"] == without_head["threshold_used"]
        assert (
            with_head["gosnn_metadata"]["sigma_immutable_score"]
            == without_head["gosnn_metadata"]["sigma_immutable_score"]
        )
        assert (
            with_head["gosnn_metadata"]["ethical_gate_passed"]
            == without_head["gosnn_metadata"]["ethical_gate_passed"]
        )


class TestDecisionLayerRouting:
    def test_strong_disagreement_demotes_to_deferral(
        self, fitted_engine: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: forced disagreement flips the decision to DEFER."""
        from omni_mercury_engine.core.global_omni_scalar_network import get_global_scalar_network

        fitted_engine.enable_decision_layer()
        try:
            fusion = get_global_scalar_network().attention_fusion
            row = _row()

            # Baseline verdict without the channel.
            monkeypatch.setattr(fusion, "detection_head", None)
            monkeypatch.setattr(fusion, "decision_thresholds", None)
            baseline = fitted_engine.detect_with_fusion(row)
            baseline_decision = baseline["decision"]
            assert baseline_decision["signals"]["gosnn_anomaly_prob"] is None

            # Force maximal disagreement with whatever the verdict was.
            disagreeing = 0.0 if baseline["is_anomaly"] else 1.0
            monkeypatch.setattr(fusion, "detection_probability", lambda fused: disagreeing)
            monkeypatch.setattr(fusion, "detection_head", object())
            monkeypatch.setattr(
                fusion,
                "decision_thresholds",
                {"demote_act_below": 0.35, "demote_clear_above": 0.65},
            )
            contested = fitted_engine.detect_with_fusion(row)
            decision = contested["decision"]

            assert decision["signals"]["gosnn_anomaly_prob"] == disagreeing
            if baseline_decision["state"] == "grounded":
                assert decision["state"] == "unavailable"
                assert decision["disposition"] == "defer"
                assert any("GOSNN" in reason for reason in decision["reasons"])
            else:
                # A verdict that already abstained must stay abstained -- the
                # overlay can only ever weaken.
                assert decision["state"] in ("unavailable", "undecidable")
        finally:
            fitted_engine.decision_layer = None

    def test_agreement_leaves_decision_unchanged(
        self, fitted_engine: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.core.global_omni_scalar_network import get_global_scalar_network

        fitted_engine.enable_decision_layer()
        try:
            fusion = get_global_scalar_network().attention_fusion
            row = _row()
            monkeypatch.setattr(fusion, "detection_head", None)
            monkeypatch.setattr(fusion, "decision_thresholds", None)
            baseline = fitted_engine.detect_with_fusion(row)

            agreeing = 1.0 if baseline["is_anomaly"] else 0.0
            monkeypatch.setattr(fusion, "detection_probability", lambda fused: agreeing)
            monkeypatch.setattr(fusion, "detection_head", object())
            monkeypatch.setattr(
                fusion,
                "decision_thresholds",
                {"demote_act_below": 0.35, "demote_clear_above": 0.65},
            )
            corroborated = fitted_engine.detect_with_fusion(row)

            assert (
                corroborated["decision"]["state"] == baseline["decision"]["state"]
            ), "an agreeing GOSNN probability must never change the decision"
            assert corroborated["decision"]["disposition"] == baseline["decision"]["disposition"]
        finally:
            fitted_engine.decision_layer = None
