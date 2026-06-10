# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Property-based invariants for the abstention gate (Hypothesis).

The example-based suites pin specific behaviours; these pin the *laws* the gate
must obey for every input the certificate can take:

* **Fail-closed dominance** -- an explicit ethical block forces a fail-closed
  hold over any score, certificate, severity, agreement or drift.
* **Monotonicity** -- more uncertainty (less neuro-symbolic agreement, a wider
  indecision band) never yields *less* abstention.
* **Structural soundness** -- the three-state honesty contract, the
  disposition projection, and the bounded/non-destructive response hold for
  arbitrary evidence; every record is deterministic and round-trips through
  ``to_dict`` / ``from_dict`` and JSON.

Pure-Python tier.  ``hypothesis`` is a declared test dependency; the import is
guarded (like ``torch`` elsewhere) so the tier still passes where it is absent.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("hypothesis")

from hypothesis import (
    given,
    strategies as st,
)

from omni_mercury_engine.decision import (
    DecisionAbstentionResponder,
    DecisionPolicy,
    DecisionRecord,
    Disposition,
    ResponseAction,
    ThreeState,
)

_R = DecisionAbstentionResponder()

_unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_thresholds = st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False)
_label_sets = st.sampled_from([[], [0], [1], [0, 1]])
_drift_sev = st.sampled_from(["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL", None])


@st.composite
def _conformal(draw: st.DrawFn) -> dict[str, Any]:
    labels = draw(_label_sets)
    return {
        "prediction_set": labels,
        "set_size": len(labels),
        "abstain": len(labels) == 2,
        "coverage": draw(st.floats(0.5, 0.999, allow_nan=False)),
    }


@st.composite
def _results(draw: st.DrawFn) -> dict[str, Any]:
    """An arbitrary, possibly-sparse detect_with_fusion-style result mapping."""
    res: dict[str, Any] = {
        "anomaly_prob": draw(_unit),
        "threshold_used": draw(_thresholds),
        "severity": draw(_unit),
        "is_anomaly": draw(st.booleans()),
    }
    if draw(st.booleans()):
        res["conformal"] = draw(_conformal())
    ethical = draw(st.sampled_from([True, False, None, "absent"]))
    if ethical != "absent":
        res["gosnn_metadata"] = {"ethical_gate_passed": ethical}
    if draw(st.booleans()):
        res["symbolic_consistency"] = {"satisfaction": draw(_unit)}
    if draw(st.booleans()):
        res["drift_detection"] = {"is_drift": draw(st.booleans()), "severity": draw(_drift_sev)}
    return res


class TestFailClosedDominance:
    """A refused ethical boundary beats everything else, always."""

    @given(res=_results())
    def test_ethical_block_forces_hold_over_any_evidence(self, res: dict[str, Any]) -> None:
        res = {**res, "gosnn_metadata": {"ethical_gate_passed": False}}
        rec = _R.decide(res)
        assert rec.state is ThreeState.UNDECIDABLE
        assert rec.disposition is Disposition.HOLD
        assert rec.decision_label is None
        assert rec.response.fail_closed is True
        assert rec.response.requires_human is True


class TestMonotonicity:
    """More uncertainty never produces less abstention."""

    @given(base=_results(), sats=st.tuples(_unit, _unit))
    def test_more_symbolic_disagreement_never_less_abstention(
        self, base: dict[str, Any], sats: tuple[float, float]
    ) -> None:
        # Isolate the symbolic axis: drop the ethical / drift overlays so only
        # the agreement level moves between the two runs.
        base = {k: v for k, v in base.items() if k not in ("gosnn_metadata", "drift_detection")}
        s_low, s_high = sorted(sats)
        a_low = _R.decide({**base, "symbolic_consistency": {"satisfaction": s_low}}).abstained
        a_high = _R.decide({**base, "symbolic_consistency": {"satisfaction": s_high}}).abstained
        # Lower agreement (more disagreement) => at least as much abstention.
        assert int(a_low) >= int(a_high)

    @given(p=_unit, thr=_thresholds, margins=st.tuples(st.floats(0.0, 0.49), st.floats(0.0, 0.49)))
    def test_wider_indecision_band_never_less_abstention(
        self, p: float, thr: float, margins: tuple[float, float]
    ) -> None:
        m_small, m_large = sorted(margins)
        res = {"anomaly_prob": p, "threshold_used": thr, "is_anomaly": p > thr, "severity": 0.0}
        small = DecisionAbstentionResponder(DecisionPolicy(indecision_margin=m_small))
        large = DecisionAbstentionResponder(DecisionPolicy(indecision_margin=m_large))
        assert int(large.decide(res).abstained) >= int(small.decide(res).abstained)


class TestStructuralSoundness:
    """The honesty contract and the bounded response hold for arbitrary evidence."""

    @given(res=_results())
    def test_three_state_and_response_invariants(self, res: dict[str, Any]) -> None:
        rec = _R.decide(res)
        grounded = rec.state is ThreeState.GROUNDED

        # Abstention is exactly "not grounded"; a label exists iff grounded.
        assert rec.abstained is (not grounded)
        assert (rec.decision_label is None) is rec.abstained
        if grounded:
            assert rec.decision_label in (0, 1)
        else:
            # An honest abstention never claims a confidence in a label it didn't make.
            assert rec.decision_confidence is None

        # The disposition is the action projection of the honesty state.
        expected = {
            ThreeState.GROUNDED: {Disposition.ACT, Disposition.CLEAR},
            ThreeState.UNAVAILABLE: {Disposition.DEFER},
            ThreeState.UNDECIDABLE: {Disposition.HOLD},
        }[rec.state]
        assert rec.disposition in expected

        # The response is always within the bounded, non-destructive catalogue,
        # and a fail-closed refusal always demands a human.
        assert rec.response.action in set(ResponseAction)
        if rec.response.fail_closed:
            assert rec.response.requires_human is True

    @given(res=_results())
    def test_pure_and_serialisable(self, res: dict[str, Any]) -> None:
        rec = _R.decide(res)
        # Pure: same certificate in, same record out.
        assert _R.decide(res).to_dict() == rec.to_dict()
        # Round-trips through the record's own (de)serialisers and through JSON.
        assert DecisionRecord.from_dict(rec.to_dict()).to_dict() == rec.to_dict()
        assert json.loads(json.dumps(rec.to_dict())) == rec.to_dict()
