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

"""The abstention policy: the first-class don't-know gate over calibrated confidence.

Pure-Python tier (no torch): the policy is a deterministic function of a
:class:`ConfidenceSignal`, so the full truth table runs in every environment.
"""

import pytest

from omni_mercury_engine.decision.abstention import (
    POLICY_VERSION,
    AbstentionPolicy,
    AbstentionThresholds,
)
from omni_mercury_engine.decision.confidence import ConfidenceSignal
from omni_mercury_engine.decision.types import Verdict
from omni_mercury_engine.verifiers.three_state import ThreeState


def _conformal(prediction_set: tuple[int, ...], prob: float = 0.5) -> ConfidenceSignal:
    return ConfidenceSignal(anomaly_probability=prob, prediction_set=prediction_set, coverage=0.9)


class TestConformalPath:
    """When a conformal set is present, the policy honours its set-size semantics."""

    @pytest.mark.parametrize(
        ("prediction_set", "prob", "verdict", "state", "novelty"),
        [
            ((1,), 0.93, Verdict.POSITIVE, ThreeState.GROUNDED, False),
            ((0,), 0.04, Verdict.NEGATIVE, ThreeState.GROUNDED, False),
            ((0, 1), 0.55, Verdict.ABSTAIN, ThreeState.UNAVAILABLE, False),
            ((), 0.50, Verdict.ABSTAIN, ThreeState.UNAVAILABLE, True),
        ],
    )
    def test_truth_table(
        self,
        prediction_set: tuple[int, ...],
        prob: float,
        verdict: Verdict,
        state: ThreeState,
        novelty: bool,
    ) -> None:
        decision = AbstentionPolicy().decide(_conformal(prediction_set, prob))
        assert decision.verdict is verdict
        assert decision.state is state
        assert decision.novelty is novelty
        # Coverage and the originating set are carried for the audit certificate.
        assert decision.coverage == pytest.approx(0.9)
        assert decision.prediction_set == tuple(sorted(prediction_set))

    def test_singletons_are_grounded_decisions(self) -> None:
        assert AbstentionPolicy().decide(_conformal((1,), 0.9)).is_grounded
        assert AbstentionPolicy().decide(_conformal((0,), 0.1)).is_grounded

    def test_empty_set_flags_novelty(self) -> None:
        decision = AbstentionPolicy().decide(_conformal((), 0.5))
        assert decision.abstained
        assert decision.novelty
        assert (
            "novel" in decision.reason.lower() or "out-of-distribution" in decision.reason.lower()
        )

    def test_conformal_overrides_extreme_point_when_uncertain(self) -> None:
        # Even with a high point probability, a two-label set must abstain --
        # the coverage-guaranteed set wins over the bare point.
        decision = AbstentionPolicy().decide(_conformal((0, 1), 0.88))
        assert decision.verdict is Verdict.ABSTAIN


class TestBandFallback:
    """With only a calibrated point, the explicit indecision band decides."""

    @pytest.mark.parametrize(
        ("prob", "verdict"),
        [
            (0.95, Verdict.POSITIVE),
            (0.70, Verdict.POSITIVE),  # boundary is inclusive (>= positive)
            (0.50, Verdict.ABSTAIN),
            (0.31, Verdict.ABSTAIN),
            (0.30, Verdict.NEGATIVE),  # boundary is inclusive (<= negative)
            (0.02, Verdict.NEGATIVE),
        ],
    )
    def test_band_decisions(self, prob: float, verdict: Verdict) -> None:
        decision = AbstentionPolicy().decide(ConfidenceSignal(anomaly_probability=prob))
        assert decision.verdict is verdict

    def test_custom_thresholds_widen_band(self) -> None:
        policy = AbstentionPolicy(AbstentionThresholds(positive=0.9, negative=0.1))
        assert policy.decide(ConfidenceSignal(anomaly_probability=0.8)).verdict is Verdict.ABSTAIN
        assert policy.decide(ConfidenceSignal(anomaly_probability=0.95)).verdict is Verdict.POSITIVE

    def test_require_conformal_abstains_without_set(self) -> None:
        policy = AbstentionPolicy(require_conformal=True)
        decision = policy.decide(ConfidenceSignal(anomaly_probability=0.99))
        assert decision.verdict is Verdict.ABSTAIN
        assert "require_conformal" in decision.reason


class TestHonestyInvariants:
    """The contract a detection abstention must obey."""

    @pytest.mark.parametrize("prob", [0.40, 0.50, 0.60])
    def test_abstain_is_unavailable_never_undecidable(self, prob: float) -> None:
        decision = AbstentionPolicy().decide(ConfidenceSignal(anomaly_probability=prob))
        assert decision.verdict is Verdict.ABSTAIN
        # A detection is decidable in principle -> UNAVAILABLE, never UNDECIDABLE.
        assert decision.state is ThreeState.UNAVAILABLE
        assert decision.state is not ThreeState.UNDECIDABLE

    def test_abstention_records_what_would_decide_it(self) -> None:
        decision = AbstentionPolicy().decide(_conformal((0, 1), 0.5))
        assert "would_decide" in decision.provenance

    def test_margin_zero_on_abstain_positive_on_commit(self) -> None:
        assert AbstentionPolicy().decide(_conformal((0, 1), 0.5)).margin == 0.0
        committed = AbstentionPolicy().decide(ConfidenceSignal(anomaly_probability=0.95))
        assert 0.0 < committed.margin <= 1.0

    def test_decision_is_deterministic(self) -> None:
        signal = _conformal((1,), 0.9)
        first = AbstentionPolicy().decide(signal)
        second = AbstentionPolicy().decide(signal)
        assert (first.verdict, first.state, first.confidence) == (
            second.verdict,
            second.state,
            second.confidence,
        )

    def test_policy_version_is_stamped(self) -> None:
        decision = AbstentionPolicy().decide(ConfidenceSignal(anomaly_probability=0.95))
        assert decision.policy == POLICY_VERSION

    def test_metadata_is_json_friendly(self) -> None:
        import json

        decision = AbstentionPolicy().decide(_conformal((0, 1), 0.5))
        json.dumps(decision.as_metadata())


class TestThresholdsValidation:
    """Operating points must form a valid band."""

    @pytest.mark.parametrize(
        ("positive", "negative"),
        [(0.3, 0.7), (1.2, 0.5), (0.5, -0.1)],
    )
    def test_invalid_thresholds_raise(self, positive: float, negative: float) -> None:
        with pytest.raises(ValueError):
            AbstentionThresholds(positive=positive, negative=negative)

    def test_batch_matches_individual(self) -> None:
        signals = [_conformal((1,), 0.9), _conformal((0, 1), 0.5), ConfidenceSignal(0.1)]
        policy = AbstentionPolicy()
        batch = policy.decide_batch(signals)
        assert [d.verdict for d in batch] == [policy.decide(s).verdict for s in signals]
