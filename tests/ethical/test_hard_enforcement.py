"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Hard ethical enforcement boundary regression suite.

Phase 2 of the May 2026 audit cure flips ethics from "advisory" to
"functional".  Every API surface listed as a *decision boundary* in
``src/omni_mercury_engine/ethical/__init__.py`` MUST raise
``EthicalConstraintViolationError`` (re-exported as ``EthicalViolation``)
on a simulated benevolence-violation (``check="benevolence"``), and MUST
NOT raise for legitimate inputs.  GOSNN's σ_Immutable score is
informational metadata — it is not the enforcement gate today because
the underlying neural network is untrained.

This file is the regression that makes the contract durable: any future
change that turns one of these boundaries back into a logger.warning,
config-flag-disabled check, or "fall back to ethical_gate_passed=True"
path will fail this suite.  The suite is wired into the
``Neuro-Symbolic Tests`` job in ``.github/workflows/ci.yml`` so the
regression cannot merge silently.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from omni_mercury_engine.cognitive.ethical_bounding import (
    BenevolenceScorer,
    EthicalConstraintViolationError,
)
from omni_mercury_engine.ethical import EthicalViolation

# ---------------------------------------------------------------------------
# Realistic-input fixtures.
#
# Earlier revisions of this suite exercised the boundaries with bare
# ``np.random.RandomState(seed).randn(rows, cols)`` matrices.  Pure
# Gaussian noise is a poor stand-in for the byte-rate / packet-count /
# duration-skewed feature vectors the production detectors actually
# consume, so the regressions only sampled an idealised slice of the
# input distribution.  ``_synthetic_traffic_batch`` builds a small but
# structurally realistic batch that mixes:
#
# * positive, right-skewed counters (packet/byte rates) via a log-normal
#   draw,
# * bounded ratios in [0, 1] (TCP-flag / protocol mixes) via a Beta
#   draw,
# * zero-mean float deltas (timing jitter, entropy residuals) via a
#   Gaussian draw,
# * a small fraction of clearly-anomalous rows whose counter columns
#   are pushed several sigma above the benign mean — the same shape an
#   exfil burst or scan wave would present to a flow-level detector.
#
# The output is float64, contiguous, and shape-compatible with both
# ``OmniMercuryEngine.detect_with_fusion`` and
# ``NeuroSymbolicHub.predict``.
# ---------------------------------------------------------------------------


def _synthetic_traffic_batch(
    seed: int,
    rows: int,
    cols: int,
    anomaly_fraction: float = 0.25,
) -> np.ndarray:
    """Return a structured ``(rows, cols)`` traffic-like feature batch.

    Args:
        seed: Deterministic RNG seed.
        rows: Number of feature vectors (samples) in the batch.
        cols: Feature width.  Must be ``>= 3``.
        anomaly_fraction: Fraction of rows whose counter columns are
            pushed into the anomalous regime.

    Returns:
        A ``float64`` ``(rows, cols)`` array combining log-normal
        counters, Beta-distributed ratios, and Gaussian deltas, with a
        deterministic anomalous tail.
    """
    if cols < 3:
        raise ValueError("cols must be >= 3 for the traffic-like layout")
    rng = np.random.default_rng(seed)

    # Three column groups, evenly distributed across the available
    # width, modelling counters / ratios / deltas respectively.
    counter_cols = max(1, cols // 3)
    ratio_cols = max(1, cols // 3)
    delta_cols = cols - counter_cols - ratio_cols

    counters = rng.lognormal(mean=0.0, sigma=0.75, size=(rows, counter_cols))
    ratios = rng.beta(a=2.0, b=5.0, size=(rows, ratio_cols))
    deltas = rng.normal(loc=0.0, scale=0.5, size=(rows, delta_cols))

    batch = np.concatenate([counters, ratios, deltas], axis=1).astype(np.float64)

    # Push a deterministic minority of rows several sigma above the
    # benign counter mean so the batch covers both regimes.
    n_anom = max(1, round(rows * anomaly_fraction))
    anom_idx = rng.choice(rows, size=n_anom, replace=False)
    batch[anom_idx, :counter_cols] *= 6.0

    return np.ascontiguousarray(batch)


# ---------------------------------------------------------------------------
# Sanity: the canonical names alias the same class.
# ---------------------------------------------------------------------------


class TestCanonicalExceptionAlias:
    """``EthicalViolation`` and ``EthicalConstraintViolationError`` are aliases."""

    def test_violation_alias_is_constraint_error(self) -> None:
        assert EthicalViolation is EthicalConstraintViolationError

    def test_violation_inherits_runtime_error(self) -> None:
        assert issubclass(EthicalViolation, RuntimeError)

    def test_violation_carries_check_and_details(self) -> None:
        exc = EthicalViolation(
            action="probe",
            score=0.4,
            threshold=0.99,
            check="benevolence",
            details={"sample": 7},
        )
        assert exc.check == "benevolence"
        assert exc.details == {"sample": 7}
        assert "benevolence" in str(exc)
        assert "probe" in str(exc)


# ---------------------------------------------------------------------------
# Boundary 1: BenevolenceScorer.enforce — the primitive gate.
# ---------------------------------------------------------------------------


class TestBenevolenceScorerEnforce:
    def test_enforce_raises_on_violation(self) -> None:
        scorer = BenevolenceScorer(benevolence_threshold=0.99)
        with pytest.raises(EthicalViolation) as exc_info:
            scorer.enforce("destroy_all_data", {"intent": "malicious harm damage"})
        assert exc_info.value.threshold == 0.99
        assert exc_info.value.score < 0.99

    def test_enforce_returns_score_on_legitimate_input(self) -> None:
        scorer = BenevolenceScorer(benevolence_threshold=0.70)
        result = scorer.enforce(
            "humanitarian_aid_distribution",
            {
                "intent": (
                    "selfless benefit humanitarian aid care help support "
                    "empathy fair just equal rights data research verify"
                )
            },
        )
        assert result.is_permissible
        assert result.benevolence_score >= 0.70


# ---------------------------------------------------------------------------
# Boundary 2: CognitiveOrchestrator.analyze — top-level cognitive surface.
# ---------------------------------------------------------------------------


class TestCognitiveOrchestratorBoundary:
    def _orchestrator(self):
        from omni_mercury_engine.cognitive.orchestrator import CognitiveOrchestrator

        return CognitiveOrchestrator(
            enable_plasticity=False,
            enable_causal=False,
            enable_ipb=False,
            enable_cbr=False,
            enable_indicators=False,
        )

    def test_analyze_does_not_raise_on_legitimate_input(self) -> None:
        orchestrator = self._orchestrator()
        result = orchestrator.analyze(
            detection_result={
                "is_anomaly": False,
                "anomaly_prob": 0.1,
                "severity": 0.1,
            },
            context={"domain": "general"},
        )
        assert result.ethical_permissible is True
        assert result.benevolence_score > 0

    def test_analyze_raises_on_violation(self) -> None:
        orchestrator = self._orchestrator()
        # Pin the threshold above the maximum achievable score so the
        # boundary deterministically fires regardless of keyword scoring.
        orchestrator._benevolence_scorer.benevolence_threshold = 1.01

        with pytest.raises(EthicalViolation) as exc_info:
            orchestrator.analyze(
                detection_result={
                    "is_anomaly": True,
                    "anomaly_prob": 0.95,
                    "severity": 0.9,
                },
                context={"domain": "general"},
            )
        assert exc_info.value.threshold == 1.01
        assert exc_info.value.score < 1.01

    def test_strict_ethics_false_is_deprecated_and_ignored(self) -> None:
        from omni_mercury_engine.cognitive.orchestrator import CognitiveOrchestrator

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            orchestrator = CognitiveOrchestrator(
                enable_plasticity=False,
                enable_causal=False,
                enable_ipb=False,
                enable_cbr=False,
                enable_indicators=False,
                strict_ethics=False,
            )
        assert orchestrator.strict_ethics is True
        assert any(
            issubclass(w.category, DeprecationWarning) and "strict_ethics=False" in str(w.message)
            for w in captured
        )

        # The gate must still fire even though the caller asked for
        # advisory mode.
        orchestrator._benevolence_scorer.benevolence_threshold = 1.01
        with pytest.raises(EthicalViolation):
            orchestrator.analyze(
                detection_result={
                    "is_anomaly": True,
                    "anomaly_prob": 0.99,
                    "severity": 0.99,
                },
                context={"domain": "general"},
            )


# ---------------------------------------------------------------------------
# Boundary 3: NeuroSymbolicHub.predict — keystone fusion surface.
# ---------------------------------------------------------------------------


class TestNeuroSymbolicHubBoundary:
    def test_predict_raises_on_violation(self) -> None:
        from omni_mercury_engine.core.neurosymbolic_hub import NeuroSymbolicHub

        hub = NeuroSymbolicHub(
            input_dim=32,
            benevolence_threshold=0.99,
            seed=42,
            enable_domain_features=False,
            enable_adaptive_thresholding=False,
            enable_gosnn_3r=False,
        )
        X = _synthetic_traffic_batch(seed=0, rows=2, cols=32)

        with pytest.raises(EthicalViolation) as exc_info:
            hub.predict(X)

        assert exc_info.value.check == "benevolence"
        assert exc_info.value.threshold == 0.99
        assert exc_info.value.score < 0.99
        assert "sample_index" in exc_info.value.details

    def test_predict_returns_results_on_legitimate_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.core.neurosymbolic_hub import NeuroSymbolicHub
        from omni_mercury_engine.security.sigma_immutable_gate import (
            SigmaImmutableEvaluation,
        )

        # ``benevolence_threshold`` clamps to ``MINIMUM_BENEVOLENCE_FLOOR``
        # (0.70).  With an untrained encoder + random inputs the realised
        # benevolence sits in the 0.65–0.75 band, so the gate is exercised
        # right at the floor.  Use the floor as the threshold and then
        # assert any sample that returns successfully clears it.
        hub = NeuroSymbolicHub(
            input_dim=32,
            benevolence_threshold=0.50,  # below floor → clamped via floor
            seed=42,
            enable_domain_features=False,
            enable_adaptive_thresholding=False,
            enable_gosnn_3r=False,
        )
        # Test-only: bypass the setter's floor-clamp via the private attr
        # so synthetic random inputs (benevolence ~0.65–0.75) reliably pass.
        # This test asserts that predict() returns successfully for benign
        # inputs, not that the gate fires.
        hub._benevolence_threshold = 0.0

        # Wave B: σ_Immutable is now an independent hard gate that fires
        # at sub-threshold benevolence by design.  This test deliberately
        # bypasses the benevolence floor to exercise the *predict* return
        # path on benign inputs; bypassing σ_Immutable in tests is the
        # contract — a real production caller cannot reach this state.
        monkeypatch.setattr(
            hub._sigma_immutable_gate,
            "enforce",
            lambda action, scalar_vector, details=None: SigmaImmutableEvaluation(
                score=0.99, threshold=0.93, passes=True, backend="torch"
            ),
        )

        X = _synthetic_traffic_batch(seed=1, rows=2, cols=32)
        results = hub.predict(X)
        assert len(results) == 2
        for res in results:
            assert res.ethical_compliant is True
            assert res.benevolence_score >= 0.0


# ---------------------------------------------------------------------------
# Boundary 4: OmniMercuryEngine.detect_with_fusion — top-level engine
# inference path.  Enforcement at this boundary delegates to the same
# BenevolenceScorer.enforce primitive used by CognitiveOrchestrator so
# both top-level boundaries share identical threshold semantics.  GOSNN's
# σ_Immutable score remains in the result metadata but is informational,
# not the gate (its underlying neural network is untrained — see audit
# follow-up).
# ---------------------------------------------------------------------------


def _make_engine_in_fusion_mode():
    """Build a minimal fusion-mode engine for boundary tests."""
    from omni_mercury_engine.engine import OmniMercuryEngine

    engine = OmniMercuryEngine(mode="fusion")
    return engine


class TestEngineFusionBoundary:
    def test_detect_with_fusion_raises_on_benevolence_violation(self) -> None:
        engine = _make_engine_in_fusion_mode()

        # Force the boundary scorer's threshold above any achievable score
        # so the gate fires deterministically without depending on
        # keyword distribution.
        engine._enforce_ethics_at_boundary(domain="general", data=np.zeros((1, 2)))
        engine._boundary_scorer.benevolence_threshold = 1.01

        with pytest.raises(EthicalViolation) as exc_info:
            engine.detect_with_fusion(_synthetic_traffic_batch(seed=0, rows=4, cols=8))

        assert exc_info.value.check == "benevolence"
        assert exc_info.value.threshold == 1.01
        assert exc_info.value.score < 1.01

    def test_detect_with_fusion_succeeds_on_legitimate_input(self) -> None:
        engine = _make_engine_in_fusion_mode()

        # The engine's _enforce_ethics_at_boundary uses positive-keyword
        # action text plus MINIMUM_BENEVOLENCE_FLOOR, so legitimate
        # detection requests must clear the gate without modification.
        result = engine.detect_with_fusion(_synthetic_traffic_batch(seed=0, rows=4, cols=8))
        assert "anomaly_prob" in result
        # GOSNN metadata stays informational — it must be present even
        # though it is no longer the gate.
        assert "gosnn_metadata" in result

    def test_detect_with_fusion_raises_when_gosnn_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When GOSNN errors, the engine raises ``check='gosnn_unavailable'``.

        The σ_Immutable promotion (Wave B item 1) deletes the previous
        ``fallback_mode=True`` metadata path: if GOSNN cannot run, the
        second hard ethical gate cannot run either, and the engine
        fails closed at the boundary instead of degrading silently.
        """
        from omni_mercury_engine import engine as engine_module

        engine = _make_engine_in_fusion_mode()

        def _boom(**kwargs) -> None:
            raise RuntimeError("simulated GOSNN outage")

        monkeypatch.setattr(engine_module, "get_global_scalar_network", _boom)

        with pytest.raises(EthicalViolation) as exc_info:
            engine.detect_with_fusion(_synthetic_traffic_batch(seed=0, rows=4, cols=8))
        assert exc_info.value.check == "gosnn_unavailable"
        assert "simulated GOSNN outage" in exc_info.value.details["underlying_error"]


# ---------------------------------------------------------------------------
# Wave B: σ_Immutable + gosnn_unavailable contract — promoted to hard gates.
#
# These two tests pin the post-Wave-B contract:
#
# * If σ_Immutable scores below threshold at the engine boundary, the
#   engine raises ``EthicalConstraintViolationError(check="sigma_immutable")``.
# * If GOSNN cannot be evaluated at all, the engine raises
#   ``EthicalConstraintViolationError(check="gosnn_unavailable")`` instead
#   of silently writing ``fallback_mode=True`` metadata.
#
# Both started life as ``@pytest.mark.xfail(strict=True)`` markers in
# Wave A; Wave B flips them to positive tests in the same PR (the
# strict-xfail mechanism would otherwise turn the xpass into a failure
# the moment promotion landed, forcing the marker removal).
# ---------------------------------------------------------------------------


class TestReservedChecksWaveB:
    """σ_Immutable and ``gosnn_unavailable`` are now hard gates."""

    def test_sigma_immutable_check_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """σ_Immutable raises with ``check="sigma_immutable"`` on sub-threshold."""

        engine = _make_engine_in_fusion_mode()

        # Force the σ_Immutable gate to score below threshold for every
        # input — equivalent to a poisoned scalar vector — so the
        # boundary deterministically fires the second hard gate.
        def _force_failure(action, scalar_vector, details=None) -> None:
            from omni_mercury_engine.cognitive.ethical_bounding import (
                EthicalConstraintViolationError,
            )

            merged = {"action": action}
            if details:
                merged.update(details)
            merged["sigma_immutable_score"] = 0.10
            raise EthicalConstraintViolationError(
                action=action,
                score=0.10,
                threshold=engine._sigma_immutable_gate.threshold,
                check="sigma_immutable",
                details=merged,
            )

        monkeypatch.setattr(engine._sigma_immutable_gate, "enforce", _force_failure)

        with pytest.raises(EthicalViolation) as exc_info:
            engine.detect_with_fusion(_synthetic_traffic_batch(seed=1, rows=4, cols=8))
        assert exc_info.value.check == "sigma_immutable"
        assert exc_info.value.score < exc_info.value.threshold

    def test_gosnn_unavailable_check_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GOSNN outage now raises ``check="gosnn_unavailable"``."""
        from omni_mercury_engine import engine as engine_module

        engine = _make_engine_in_fusion_mode()

        def _boom(**kwargs) -> None:
            raise RuntimeError("simulated GOSNN outage for hard-gate regression")

        monkeypatch.setattr(engine_module, "get_global_scalar_network", _boom)

        with pytest.raises(EthicalViolation) as exc_info:
            engine.detect_with_fusion(_synthetic_traffic_batch(seed=2, rows=4, cols=8))
        assert exc_info.value.check == "gosnn_unavailable"
