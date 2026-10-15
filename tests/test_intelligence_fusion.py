# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for omni_mercury_engine.security.intelligence_fusion module.

Tests all-source intelligence fusion, the multi-INT report modality contract,
and the fused-encoder network dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("torch")

import numpy as np

from omni_mercury_engine.security.intelligence_fusion import (
    MIN_HELDOUT_CASES,
    AllSourceFusionNetwork,
    IntelligenceFusionEngine,
    IntelligenceFusionResult,
    ThreatLevel,
    _mcnemar_one_sided,
)

if TYPE_CHECKING:
    from pathlib import Path


def _sample_reports() -> dict[str, dict[str, object]]:
    """Build a representative multi-INT report mapping."""
    return {
        "open_source": {
            "confidence": 0.72,
            "timeliness": 0.6,
            "relevance": 0.8,
            "completeness": 0.5,
            "indicators": ["recruitment", "finance"],
            "threat_score": 0.65,
        },
        "communications": {
            "confidence": 0.81,
            "timeliness": 0.9,
            "relevance": 0.85,
            "completeness": 0.7,
            "indicators": ["covert_comms"],
            "threat_score": 0.78,
            "encryption_detected": True,
        },
        "human": {
            "confidence": 0.88,
            "timeliness": 0.5,
            "relevance": 0.9,
            "completeness": 0.6,
            "indicators": ["insider_threat", "training"],
            "threat_score": 0.82,
        },
        "cyber": {
            "confidence": 0.79,
            "timeliness": 0.85,
            "relevance": 0.88,
            "completeness": 0.72,
            "indicators": ["reconnaissance", "c2"],
            "threat_score": 0.9,
            "encryption_algorithm": "AES-256",
        },
    }


class TestModalityContract:
    """Tests for the multi-INT report modality contract."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.engine = IntelligenceFusionEngine()

    def test_off_modality_array_extract_features_rejected(self) -> None:
        """A generic float array must yield a clean, actionable ValueError."""
        off_modality = np.random.rand(200, 8).astype(np.float32)

        with pytest.raises(ValueError, match="multi-INT report mapping"):
            self.engine.extract_features(off_modality)  # type: ignore[arg-type]

    def test_off_modality_array_predict_rejected(self) -> None:
        """The predict path must reject a generic float array cleanly."""
        off_modality = np.random.rand(200, 8).astype(np.float32)

        with pytest.raises(ValueError, match=r"shape \(200, 8\)"):
            self.engine.predict(off_modality)  # type: ignore[arg-type]


class TestOnModalityFusion:
    """Tests driving the fusion network on correct-modality intel reports."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.engine = IntelligenceFusionEngine()

    def test_extract_features_shape(self) -> None:
        """Feature extraction returns a concatenated per-discipline tensor."""
        features = self.engine.extract_features(_sample_reports())
        assert features.shape[0] == 1
        # Declared fusion width: 13 disciplines x 9 dims (117) zero-padded
        # to the constant 128 so every path agrees.
        assert features.shape[1] == 128

    def test_fuse_intelligence_end_to_end(self) -> None:
        """Multi-INT reports flow through the full fusion network."""
        result = self.engine.fuse_intelligence(_sample_reports())

        assert isinstance(result, IntelligenceFusionResult)
        assert result.threat_level in {
            "LOW",
            "MODERATE",
            "SUBSTANTIAL",
            "SEVERE",
            "CRITICAL",
        }
        assert 0.0 <= result.confidence <= 1.0
        assert result.primary_intel_sources
        assert result.threat_indicators

    def test_predict_returns_scores(self) -> None:
        """The predict adapter surfaces anomaly scores and threat level."""
        prediction = self.engine.predict(_sample_reports())

        assert prediction["anomaly_scores"].shape == (1,)
        assert prediction["threat_level"]
        assert 0.0 <= prediction["confidence"] <= 1.0

    def test_fuse_intelligence_with_temporal_context(self) -> None:
        """The optional temporal LSTM path fuses without dimension errors."""
        temporal = [
            {"threat_level": 2, "confidence": 0.5, "num_sources": 2},
            {"threat_level": 3, "confidence": 0.6, "num_sources": 3},
            {"threat_level": 4, "confidence": 0.75, "num_sources": 5},
        ]

        result = self.engine.fuse_intelligence(_sample_reports(), temporal_context=temporal)

        assert isinstance(result, IntelligenceFusionResult)
        assert result.temporal_patterns["trend"] == "escalating"


class TestFeatureWidthContract:
    """extract_features must emit the declared (1, 128) width on every path."""

    def test_populated_and_empty_paths_agree_on_width(self) -> None:
        engine = IntelligenceFusionEngine()
        populated = engine.extract_features(
            {"open_source": {"confidence": 0.9, "threat_score": 0.4}}
        )
        empty = engine.extract_features({"not_a_discipline": {}})
        assert tuple(populated.shape) == (1, 128)
        assert tuple(empty.shape) == (1, 128)

    def test_pad_is_allocated_on_the_feature_device(self, monkeypatch: Any) -> None:
        """The zero-pad must inherit the source tensor's device (and dtype).

        Regression: the pad was allocated on the default (CPU) device, so a
        fusion engine moved off CPU would hit a device mismatch in ``torch.cat``.
        """
        import torch

        engine = IntelligenceFusionEngine()
        pad_kwargs: list[dict[str, Any]] = []
        real_zeros = torch.zeros

        def _spy(*args: Any, **kwargs: Any) -> Any:
            if "device" in kwargs:
                pad_kwargs.append(kwargs)
            return real_zeros(*args, **kwargs)

        monkeypatch.setattr(torch, "zeros", _spy)
        features = engine.extract_features(
            {"open_source": {"confidence": 0.9, "threat_score": 0.4}}
        )

        assert features.shape[-1] == 128
        assert pad_kwargs, "padding tensor was not allocated with an explicit device"
        assert all(str(k["device"]) == str(features.device) for k in pad_kwargs)


class TestUntrainedDecisionContract:
    """Decisions must come from evidence, never from an untrained network.

    Regression: ``fuse_intelligence`` previously thresholded the random
    -initialisation logits of ``AllSourceFusionNetwork`` (scaled by golden
    -ratio constants), so ``threat_detected`` was noise. Untrained engines now
    score from the reported evidence and are therefore deterministic.
    """

    def test_untrained_decision_is_deterministic_across_engines(self) -> None:
        """Two engines (different random nets) agree on every decision field."""
        a = IntelligenceFusionEngine().fuse_intelligence(_sample_reports())
        b = IntelligenceFusionEngine().fuse_intelligence(_sample_reports())

        assert a.threat_detected == b.threat_detected
        assert a.threat_level == b.threat_level
        assert a.risk_score == pytest.approx(b.risk_score)
        assert a.confidence == pytest.approx(b.confidence)

    def test_high_evidence_detects_and_low_evidence_does_not(self) -> None:
        """threat_detected tracks the reported threat scores, not noise."""
        engine = IntelligenceFusionEngine()

        high = engine.fuse_intelligence(_sample_reports())
        assert high.threat_detected
        assert high.threat_level in {"SEVERE", "CRITICAL"}

        low = engine.fuse_intelligence({"open_source": {"confidence": 0.9, "threat_score": 0.05}})
        assert not low.threat_detected
        assert low.threat_level == "LOW"

    def test_no_evidence_yields_low_and_zero_scores(self) -> None:
        """An empty / unscored report mapping is LOW with zeroed scores."""
        result = IntelligenceFusionEngine().fuse_intelligence({})

        assert not result.threat_detected
        assert result.threat_level == "LOW"
        assert result.risk_score == 0.0
        assert result.confidence == 0.0

    def test_single_source_confidence_is_damped(self) -> None:
        """Corroboration scaling caps single-source confidence at one third."""
        engine = IntelligenceFusionEngine()
        result = engine.fuse_intelligence({"open_source": {"confidence": 0.9, "threat_score": 0.8}})
        assert result.confidence == pytest.approx(0.9 / 3.0)


@dataclass(frozen=True)
class _Pool:
    """Candidate weights plus cases split by whether candidate and evidence agree."""

    checkpoint: Path
    agree: list[tuple[dict[str, Any], int]]
    disagree: list[tuple[dict[str, Any], int, int]]


@pytest.fixture(scope="module")
def pool(tmp_path_factory: pytest.TempPathFactory) -> _Pool:
    """A seeded random network and 300 varied cases scored by both methods.

    Held-out labels are then chosen per test, which fixes exactly how many
    cases each method gets right and so pins the gate's decision boundary.
    """
    import torch

    torch.manual_seed(0)
    network = AllSourceFusionNetwork(input_dim=128)
    checkpoint = tmp_path_factory.mktemp("fusion") / "candidate.pt"
    torch.save({"state_dict": network.state_dict()}, checkpoint)

    engine = IntelligenceFusionEngine()
    rng = np.random.default_rng(0)
    agree: list[tuple[dict[str, Any], int]] = []
    disagree: list[tuple[dict[str, Any], int, int]] = []
    for _ in range(300):
        reports: dict[str, Any] = {
            name: {
                "confidence": float(rng.uniform(0.2, 1.0)),
                "relevance": float(rng.uniform(0.0, 1.0)),
                "indicators": ["x"] * int(rng.integers(0, 12)),
                "threat_score": float(rng.uniform(0.02, 0.98)),
            }
            for name in rng.choice(["open_source", "cyber", "financial", "signals"], 2, False)
        }
        by_network = engine._network_assessment(network, reports)[0]
        by_evidence = engine._evidence_assessment(reports)[0]
        if by_network == by_evidence:
            agree.append((reports, by_network))
        else:
            disagree.append((reports, by_network, by_evidence))
    assert len(agree) >= MIN_HELDOUT_CASES and len(disagree) >= 6, "pool too narrow to test"
    return _Pool(checkpoint, agree, disagree)


def _heldout(pool: _Pool, network_wins: int, evidence_wins: int = 0) -> list[Any]:
    """Agreeing cases plus discordant ones labelled for the chosen side."""
    cases: list[Any] = list(pool.agree[:MIN_HELDOUT_CASES])
    wins = pool.disagree[:network_wins]
    losses = pool.disagree[network_wins : network_wins + evidence_wins]
    cases += [(reports, by_network) for reports, by_network, _ in wins]
    cases += [(reports, by_evidence) for reports, _, by_evidence in losses]
    return cases


class TestFusionWeightMeritGate:
    """Fusion weights replace evidence scoring only on measured held-out merit.

    Regression: ``load_trained_weights`` installed any shape-matching state
    dict, so random-initialisation weights were enough to switch threat
    decisions onto noise.
    """

    def test_random_weights_are_refused_and_decisions_are_untouched(self, pool: _Pool) -> None:
        engine = IntelligenceFusionEngine()
        network_before = engine.fusion_network
        evidence_labelled = [(reports, by_evidence) for reports, _, by_evidence in pool.disagree]
        with pytest.raises(RuntimeError, match="refused"):
            engine.load_trained_weights(pool.checkpoint, pool.agree + evidence_labelled)
        assert not engine._trained and engine.merit is None
        assert engine.fusion_network is network_before
        reports, _, by_evidence = pool.disagree[0]
        assert engine.fuse_intelligence(reports).threat_level == ThreatLevel(by_evidence).name

    @pytest.mark.parametrize(
        ("network_wins", "evidence_wins", "admitted"),
        [
            (4, 0, False),  # p = 1/16: more accurate, not significantly so
            (5, 0, True),  # p = 1/32
            (6, 1, False),  # p = 8/128
            (6, 0, True),  # p = 1/64
        ],
    )
    def test_admission_needs_significance_not_just_higher_accuracy(
        self, pool: _Pool, network_wins: int, evidence_wins: int, admitted: bool
    ) -> None:
        engine = IntelligenceFusionEngine()
        heldout = _heldout(pool, network_wins, evidence_wins)
        if not admitted:
            with pytest.raises(RuntimeError, match="refused"):
                engine.load_trained_weights(pool.checkpoint, heldout)
            assert not engine._trained
            return
        merit = engine.load_trained_weights(pool.checkpoint, heldout)
        assert engine._trained and engine.merit == merit
        assert (merit.network_only, merit.evidence_only) == (network_wins, evidence_wins)
        assert merit.n_cases == len(heldout)
        assert merit.network_accuracy == 1.0
        assert merit.evidence_accuracy == pytest.approx(MIN_HELDOUT_CASES / len(heldout))

    def test_admitted_weights_take_over_the_decision(self, pool: _Pool) -> None:
        engine = IntelligenceFusionEngine()
        engine.load_trained_weights(pool.checkpoint, _heldout(pool, 6))
        for reports, by_network, by_evidence in pool.disagree[6:12]:
            level = engine.fuse_intelligence(reports).threat_level
            assert level == ThreatLevel(by_network).name != ThreatLevel(by_evidence).name

    @pytest.mark.parametrize("bad_label", [0, 6])
    def test_labels_outside_threat_levels_are_rejected(self, pool: _Pool, bad_label: int) -> None:
        engine = IntelligenceFusionEngine()
        heldout = _heldout(pool, 6)
        heldout[0] = (heldout[0][0], bad_label)
        with pytest.raises(ValueError):
            engine.load_trained_weights(pool.checkpoint, heldout)
        assert not engine._trained

    def test_too_few_heldout_cases_are_rejected(self, pool: _Pool) -> None:
        engine = IntelligenceFusionEngine()
        heldout = [(reports, by_network) for reports, by_network, _ in pool.disagree]
        with pytest.raises(ValueError, match="held-out cases"):
            engine.load_trained_weights(pool.checkpoint, heldout[: MIN_HELDOUT_CASES - 1])
        assert not engine._trained

    def test_mismatched_shapes_are_rejected(self, tmp_path: Path, pool: _Pool) -> None:
        import torch

        checkpoint = tmp_path / "wrong.pt"
        torch.save(torch.nn.Linear(3, 2).state_dict(), checkpoint)
        engine = IntelligenceFusionEngine()
        with pytest.raises(RuntimeError, match="state_dict"):
            engine.load_trained_weights(checkpoint, _heldout(pool, 6))
        assert not engine._trained

    @pytest.mark.parametrize(("wins", "losses"), [(0, 0), (0, 5), (3, 0), (9, 4), (812, 700)])
    def test_mcnemar_matches_the_exact_binomial_tail(self, wins: int, losses: int) -> None:
        stats = pytest.importorskip("scipy.stats")
        expected = (
            1.0
            if wins + losses == 0
            else stats.binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue
        )
        assert _mcnemar_one_sided(wins, losses) == pytest.approx(expected, rel=1e-9)


class TestNetworkDisciplineParameterization:
    """AllSourceFusionNetwork sizing follows the IntelligenceDiscipline enum.

    Regression: the cross-INT attention head count and the ``hidden_3``
    rounding hard-coded 13 while the per-discipline encoders were built from
    the enum, so a divergent ``num_int_types`` (or a changed enum) could
    silently mis-size the stack or violate MultiheadAttention's
    ``embed_dim % num_heads == 0`` constraint.
    """

    def test_attention_heads_and_rounding_follow_the_enum(self) -> None:
        from omni_mercury_engine.security.intelligence_fusion import (
            AllSourceFusionNetwork,
            IntelligenceDiscipline,
        )

        net = AllSourceFusionNetwork(input_dim=128)
        n = len(IntelligenceDiscipline)
        assert net.cross_int_attention.num_heads == n
        assert net.cross_int_attention.embed_dim % n == 0
        assert len(net.int_encoders) == n

    def test_divergent_num_int_types_fails_loud(self) -> None:
        import pytest

        from omni_mercury_engine.security.intelligence_fusion import AllSourceFusionNetwork

        with pytest.raises(ValueError, match="IntelligenceDiscipline"):
            AllSourceFusionNetwork(input_dim=128, num_int_types=5)


class TestNonFiniteEvidence:
    """Non-finite report fields are dropped, never propagated or crashing.

    Regression: a caller-supplied NaN ``threat_score`` reached
    ``math.ceil(nan * 5)`` in the evidence path and raised ``ValueError``;
    a NaN ``confidence`` made the mean NaN and poisoned ``confidence``.
    """

    def test_nan_threat_score_is_dropped_not_crashing(self) -> None:
        engine = IntelligenceFusionEngine()
        result = engine.fuse_intelligence(
            {
                "open_source": {"confidence": 0.9, "threat_score": float("nan")},
                "cyber": {"confidence": 0.8, "threat_score": 0.9},
            }
        )
        # Only the finite score counts; both finite confidences count.
        assert result.risk_score == pytest.approx(0.9)
        assert 0.0 <= result.confidence <= 1.0
        assert result.threat_detected

    def test_all_non_finite_reports_degrade_to_low(self) -> None:
        engine = IntelligenceFusionEngine()
        result = engine.fuse_intelligence(
            {
                "open_source": {
                    "confidence": float("inf"),
                    "threat_score": float("-inf"),
                }
            }
        )
        assert not result.threat_detected
        assert result.threat_level == "LOW"
        assert result.risk_score == 0.0
        assert result.confidence == 0.0

    def test_non_numeric_strings_are_dropped_not_crashing(self) -> None:
        """float('high') must drop the field, not raise ValueError."""
        engine = IntelligenceFusionEngine()
        result = engine.fuse_intelligence(
            {
                "signals": {"confidence": "unknown", "threat_score": "high"},
                "cyber": {"confidence": 0.8, "threat_score": 0.9},
            }
        )
        assert result.risk_score == pytest.approx(0.9)
        assert result.threat_detected
