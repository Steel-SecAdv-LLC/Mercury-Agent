# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the symbolic veto / conjunctive fusion / disagreement learning and the removal of the fake Lyapunov damping (issue #5)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from omni_mercury_engine.core.neurosymbolic_hub import (
    FusionMode,
    NeuroSymbolicHub,
)

if TYPE_CHECKING:
    import pytest


class TestSymbolicEvidenceNoisyOr:
    def test_single_rule_is_not_diluted(self) -> None:
        hub = NeuroSymbolicHub(input_dim=8, seed=0)
        # high_deviation (conf 0.85) fires on deviation_score >= 2.0.
        ev = hub.knowledge_graph.get_symbolic_evidence({"deviation_score": 3.0})
        # Undiluted: ~its own confidence, NOT confidence / 8 (= 0.106).
        assert ev > 0.5
        assert abs(ev - 0.85) < 0.2

    def test_no_rule_fires_is_zero(self) -> None:
        hub = NeuroSymbolicHub(input_dim=8, seed=0)
        assert hub.knowledge_graph.get_symbolic_evidence({"deviation_score": 0.0}) == 0.0


class TestHardConstraints:
    def test_hard_rule_surfaced(self) -> None:
        hub = NeuroSymbolicHub(input_dim=8, seed=0)
        hard = hub.knowledge_graph.get_hard_constraints(
            {"threat_score": 0.9}  # threat_detected: threat_score>=0.7 AND not authorized
        )
        ids = [r for r, _ in hard]
        assert "threat_detected" in ids
        assert all(0.0 <= c <= 1.0 for _, c in hard)

    def test_non_hard_rule_not_surfaced(self) -> None:
        hub = NeuroSymbolicHub(input_dim=8, seed=0)
        # high_deviation fires but is NOT hard.
        hard = hub.knowledge_graph.get_hard_constraints({"deviation_score": 3.0})
        assert "high_deviation" not in [r for r, _ in hard]

    def test_malformed_confidence_is_clipped(self) -> None:
        # A hard rule whose confidence is malformed (> 1.0) must not leak an
        # out-of-range veto level: get_hard_constraints clips to [0, 1], the
        # same as get_symbolic_evidence.
        from omni_mercury_engine.core.neurosymbolic_hub import SymbolicRule

        hub = NeuroSymbolicHub(input_dim=8, seed=0)
        hub.knowledge_graph.add_rule(
            SymbolicRule(
                rule_id="bad_conf",
                premise="threat_score >= 0.7",
                conclusion="threat_detected",
                confidence=5.0,  # malformed
                hard=True,
            )
        )
        hard = dict(hub.knowledge_graph.get_hard_constraints({"threat_score": 0.9}))
        assert hard["bad_conf"] == 1.0
        assert all(0.0 <= c <= 1.0 for c in hard.values())


class TestLyapunovRemoval:
    def test_benevolence_has_no_static_squash(self) -> None:
        hub = NeuroSymbolicHub(input_dim=8, seed=0)
        # base = 0.5 + 0.5*0.98 = 0.99; the old code multiplied by
        # exp(-0.25*(1-0.99)) = 0.9975 -> 0.9875. It must now be 0.99.
        b = hub._compute_benevolence({"neural_score": 0.98}, 0.3)
        assert abs(b - 0.99) < 1e-6
        assert b > 0.9875 + 1e-4  # strictly above the old damped value


class TestConjunctiveDefault:
    def test_default_mode_is_conjunctive(self) -> None:
        assert NeuroSymbolicHub(input_dim=8).fusion_mode is FusionMode.CONJUNCTIVE


class TestMetaFeaturesAndOutcomeLearning:
    def test_meta_features_include_disagreement(self) -> None:
        feats = NeuroSymbolicHub._meta_features(np.array([0.8, 0.2]), np.array([0.3, 0.9]))
        assert feats.shape == (2, 3)
        # third column is |neural - symbolic|
        assert np.allclose(feats[:, 2], np.abs(feats[:, 0] - feats[:, 1]))

    def test_update_from_outcome_refits_stacking(self) -> None:
        hub = NeuroSymbolicHub(input_dim=8, seed=0, fusion_mode=FusionMode.STACKING)
        rng = np.random.default_rng(0)
        refit = False
        for _ in range(40):
            n = float(rng.uniform(0, 1))
            s = float(rng.uniform(0, 1))
            label = int(n > 0.5)  # learnable signal
            refit = hub.update_from_outcome(n, s, label, refit_every=32) or refit
        assert refit is True
        assert hub._meta_learner is not None

    def test_update_from_outcome_noop_for_conjunctive(self) -> None:
        hub = NeuroSymbolicHub(input_dim=8, seed=0)  # CONJUNCTIVE default
        for _ in range(40):
            assert hub.update_from_outcome(0.5, 0.5, 1, refit_every=32) is False


def _bypass_gates(hub: NeuroSymbolicHub, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the σ_Immutable gate (its PQC enforce is heavy) and drop the
    benevolence floor so predict() exercises the *detection* paths -- matching
    the established pattern in tests/ethical/test_hard_enforcement.py."""
    from omni_mercury_engine.security.sigma_immutable_gate import SigmaImmutableEvaluation

    hub._benevolence_threshold = 0.0
    monkeypatch.setattr(
        hub._sigma_immutable_gate,
        "enforce",
        lambda action, scalar_vector, details=None: SigmaImmutableEvaluation(
            score=0.99, threshold=0.93, passes=True, backend="torch"
        ),
    )


class TestVetoOverridesNeural:
    def test_hard_rule_forces_anomaly_on_low_neural(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hub = NeuroSymbolicHub(
            input_dim=8,
            seed=0,
            enable_domain_features=False,
            enable_adaptive_thresholding=False,
            enable_gosnn_3r=False,
        )
        _bypass_gates(hub, monkeypatch)
        X = np.zeros((1, 8))  # low / neutral neural signal
        out = hub.predict(X, context={"threat_score": 0.95})  # fires hard threat_detected
        o = out[0]
        assert o.is_anomaly is True
        assert o.anomaly_score >= 0.9
        assert "threat_detected" in o.rules_fired
        assert any("symbolic veto" in e for e in o.explanations)

    def test_veto_disabled_does_not_force(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hub = NeuroSymbolicHub(
            input_dim=8,
            seed=0,
            enable_symbolic_veto=False,
            enable_domain_features=False,
            enable_adaptive_thresholding=False,
            enable_gosnn_3r=False,
        )
        _bypass_gates(hub, monkeypatch)
        X = np.zeros((1, 8))
        out = hub.predict(X, context={"threat_score": 0.95})
        # Without the veto, a single (diluted) symbolic rule cannot push the
        # conjunctive/neural fusion to a forced high-confidence anomaly.
        assert out[0].anomaly_score < 0.9

    def test_malformed_hard_rule_keeps_score_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from omni_mercury_engine.core.neurosymbolic_hub import SymbolicRule

        hub = NeuroSymbolicHub(
            input_dim=8,
            seed=0,
            enable_domain_features=False,
            enable_adaptive_thresholding=False,
            enable_gosnn_3r=False,
        )
        _bypass_gates(hub, monkeypatch)
        hub.knowledge_graph.add_rule(
            SymbolicRule(
                rule_id="bad_conf",
                premise="threat_score >= 0.7",
                conclusion="threat_detected",
                confidence=5.0,  # malformed; veto must not exceed 1.0
                hard=True,
            )
        )
        X = np.zeros((1, 8))
        out = hub.predict(X, context={"threat_score": 0.95})
        o = out[0]
        assert o.is_anomaly is True
        assert 0.0 <= o.anomaly_score <= 1.0  # invariant holds despite bad rule
