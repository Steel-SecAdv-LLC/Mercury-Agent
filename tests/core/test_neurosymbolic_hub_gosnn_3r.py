# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The hub's GOSNN-3R step reports what it does, and the verdict matches the score.

``NeuroSymbolicHub.predict`` asked the GOSNN-3R integration for adjusted
fusion weights on every sample, reported a ``lyapunov_stability`` beside them
and could replace the fused score with a blend under those weights -- after
``is_anomaly`` had already been taken from the score being replaced.

None of it ran. The call the hub made never returns (the integration takes
its own non-reentrant lock twice), so ``predict`` hung with the default
``enable_gosnn_3r=True``; the refinement it was waiting for cannot be
requested, and the stability it reported was a constant 0.0. These tests pin
the outcome: ``predict`` returns, the report says "not evaluated" rather than
0.0, and no score is rewritten after the verdict.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from omni_mercury_engine.core.neurosymbolic_hub import FusionMode, NeuroSymbolicHub

if TYPE_CHECKING:
    import pytest


def _bypass_sigma_gate(hub: NeuroSymbolicHub, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the sigma-immutable gate (its enforce is heavy), as the veto tests do."""
    from omni_mercury_engine.security.sigma_immutable_gate import SigmaImmutableEvaluation

    hub._benevolence_threshold = 0.0
    monkeypatch.setattr(
        hub._sigma_immutable_gate,
        "enforce",
        lambda action, scalar_vector, details=None: SigmaImmutableEvaluation(
            score=0.99, threshold=0.93, passes=True, backend="torch"
        ),
    )


def _hub(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> NeuroSymbolicHub:
    hub = NeuroSymbolicHub(
        input_dim=8,
        seed=0,
        enable_domain_features=False,
        enable_adaptive_thresholding=False,
        **kwargs,
    )
    _bypass_sigma_gate(hub, monkeypatch)
    return hub


def _predict_with_deadline(hub: NeuroSymbolicHub, X: np.ndarray, seconds: float = 60.0) -> Any:
    """Run ``predict`` on a daemon thread so a hang fails the test instead of the run."""
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["result"] = hub.predict(X)
        except Exception as exc:  # surfaced below
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), f"NeuroSymbolicHub.predict did not return within {seconds:.0f}s"
    if "error" in box:
        raise box["error"]
    return box["result"]


def _gosnn_steps(outputs: list[Any]) -> list[dict[str, Any]]:
    return [
        step
        for output in outputs
        for step in output.reasoning_chain
        if step["step"] == "gosnn_3r_integration"
    ]


class TestPredictWithTheIntegrationEnabled:
    def test_predict_returns_with_the_default_integration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default hub used to hang on its first sample."""
        hub = _hub(monkeypatch)
        assert hub._gosnn_3r is not None
        # The sliding-window normaliser needs history before it will normalise;
        # it is not what is under test.
        hub._sliding_normalizer = None
        X = np.random.default_rng(0).normal(size=(12, 8))

        outputs = _predict_with_deadline(hub, X)

        assert len(outputs) == 12

    def test_stability_is_reported_as_not_evaluated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Never a 0.0 that reads as a measured Lyapunov stability."""
        hub = _hub(monkeypatch)
        hub._sliding_normalizer = None
        X = np.random.default_rng(1).normal(size=(12, 8))

        steps = _gosnn_steps(_predict_with_deadline(hub, X))

        assert len(steps) == 12
        for step in steps:
            assert step["lyapunov_stability"] is None
            assert step["lyapunov_status"] == "not_evaluated"
            assert step["refinement_applied"] is False
            assert step["gosnn_3r_enabled"] is True
            assert step["gosnn_3r_active"] is False

    def test_no_trajectory_is_fed_to_the_integration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nothing is measured, so nothing is recorded as if it had been."""
        hub = _hub(monkeypatch)
        hub._sliding_normalizer = None
        assert hub._gosnn_3r is not None
        X = np.random.default_rng(2).normal(size=(12, 8))

        _predict_with_deadline(hub, X)

        assert list(hub._gosnn_3r.fusion_equation.convergence_history) == []

    def test_a_disabled_integration_adds_no_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hub = _hub(monkeypatch, enable_gosnn_3r=False)
        X = np.random.default_rng(3).normal(size=(4, 8))

        assert _gosnn_steps(hub.predict(X)) == []


class _RefiningIntegration:
    """An integration that returns at once and always asks for a zero blend."""

    def __init__(self) -> None:
        self.calls = 0

    def adjust_weights(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return {
            "neural_weight": 0.0,
            "symbolic_weight": 0.0,
            "lyapunov_stability": 1.0,
            "refine_score": True,
        }


class TestTheVerdictIsTakenFromTheReportedScore:
    def test_a_refinement_request_cannot_split_score_and_verdict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The score used to be rewritten after ``is_anomaly`` was derived."""
        hub = _hub(
            monkeypatch,
            fusion_mode=FusionMode.NEURAL_DOMINANT,
            enable_gosnn_3r=False,
            enable_symbolic_veto=False,
            use_calibration=False,
        )
        monkeypatch.setattr(hub.neural_encoder, "encode", lambda X: np.full(len(X), 0.9))
        hub._gosnn_3r = cast("Any", _RefiningIntegration())
        threshold = hub._thresholds.anomaly_default

        (output,) = hub.predict(np.zeros((1, 8)))

        # Neural-dominant fusion of a 0.9 neural score is above the threshold
        # whatever the symbolic score is.
        assert output.anomaly_score >= 0.7 * 0.9
        assert output.is_anomaly is True
        assert output.is_anomaly == (output.anomaly_score > threshold)
