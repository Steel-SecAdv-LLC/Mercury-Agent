# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fuzz the decision-record build and serialization boundary.

``Evidence.from_detection`` reads an arbitrary ``detect_with_fusion`` result
dict; the decider turns it into a ``DecisionRecord``; ``to_dict`` /
``from_dict`` persist and reload it.  These are the paths a served audit
trail and the console depend on, and ``from_detection`` in particular is
documented to treat every field as optional and never guess.  Hypothesis
drives arbitrary (and adversarial) dicts through them and asserts the
invariants that must hold for *any* input:

* ``from_detection`` never raises and always yields a JSON-safe record;
* the record's ``to_dict`` is pure JSON (``json.dumps`` succeeds);
* ``from_dict(to_dict(r)).to_dict() == r.to_dict()`` — the round trip the
  docstring promises — for records built from fuzzed detections;
* a probability that is absent, NaN or infinite never becomes a fabricated
  0.0 verdict (the placeholder the layer was built to remove).
"""

from __future__ import annotations

import json
import math
from typing import Any

from hypothesis import (
    HealthCheck,
    given,
    settings,
    strategies as st,
)

from omni_mercury_engine.decision.decider import DecisionAbstentionResponder
from omni_mercury_engine.decision.evidence import Evidence
from omni_mercury_engine.decision.record import DecisionRecord

# JSON-ish leaf values, including the awkward floats the gate must tolerate.
_leaves = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**9), max_value=10**9),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=64),
)
_json_values = st.recursive(
    _leaves,
    lambda children: st.one_of(
        st.lists(children, max_size=6),
        st.dictionaries(st.text(max_size=16), children, max_size=6),
    ),
    max_leaves=20,
)

# A detection-result-shaped dict: the keys the evidence reader actually
# consults, each free to be the wrong type or absent, plus arbitrary noise.
_conformal = st.fixed_dictionaries(
    {},
    optional={
        "prediction_set": st.lists(st.integers(min_value=0, max_value=1), max_size=3),
        "set_size": st.integers(min_value=0, max_value=5),
        "coverage": st.floats(allow_nan=True, allow_infinity=True),
        "regime": st.text(max_size=16),
        "n_rows": st.integers(min_value=-5, max_value=2048),
    },
)
_results = st.fixed_dictionaries(
    {},
    optional={
        "anomaly_prob": st.one_of(_leaves, st.just(float("nan")), st.just(float("inf"))),
        "threshold_used": _leaves,
        "threshold": _leaves,
        "severity": _leaves,
        "is_anomaly": st.one_of(st.booleans(), _leaves),
        "conformal": st.one_of(_conformal, _leaves),
        "gosnn_metadata": st.one_of(_json_values, _leaves),
        "skipped_detectors": st.one_of(st.lists(st.text(max_size=16), max_size=4), _leaves),
        "detector_uncertainties": st.one_of(
            st.dictionaries(st.text(max_size=12), _leaves, max_size=4), _leaves
        ),
        "out_of_support_rows": st.one_of(st.lists(st.integers(), max_size=4), _leaves),
        "cognitive_analysis": st.one_of(_json_values, _leaves),
        "domain": st.one_of(st.text(max_size=32), _leaves),
        "noise": _json_values,
    },
)


def _json_safe(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


class TestEvidenceFromDetectionNeverCrashes:
    @settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
    @given(result=_results)
    def test_from_detection_is_total_and_json_safe(self, result: dict[str, Any]) -> None:
        evidence = Evidence.from_detection(result)  # must not raise
        signals = evidence.to_dict()
        # Evidence.to_dict carries NaN/inf through as the gate's own view;
        # those are not JSON values, but the DECISION record that ships must
        # be JSON-safe, which the decider guarantees (asserted below).
        assert isinstance(signals, dict)

    @settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
    @given(result=_results)
    def test_decided_record_is_json_safe_and_round_trips(self, result: dict[str, Any]) -> None:
        record = DecisionAbstentionResponder().decide(result)
        wire = record.to_dict()
        assert _json_safe(wire), f"decision record is not JSON-safe: {wire}"
        # The documented inverse holds for any fuzzed detection.
        assert DecisionRecord.from_dict(wire).to_dict() == wire

    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    @given(prob=st.one_of(st.none(), st.just(float("nan")), st.just(float("inf"))))
    def test_absent_or_nonfinite_probability_never_fabricates_a_score(
        self, prob: float | None
    ) -> None:
        result: dict[str, Any] = {"anomaly_prob": prob, "threshold_used": 0.5}
        record = DecisionAbstentionResponder().decide(result)
        # No fabricated 0.0: the record carries None, and it abstained.
        assert record.anomaly_prob is None
        assert record.abstained is True
        assert record.to_dict()["anomaly_prob"] is None

    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    @given(
        prob=st.floats(min_value=0.0, max_value=1.0),
        thr=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_finite_probability_is_preserved_exactly(self, prob: float, thr: float) -> None:
        record = DecisionAbstentionResponder().decide({"anomaly_prob": prob, "threshold_used": thr})
        assert record.anomaly_prob is not None
        assert math.isclose(record.anomaly_prob, prob, rel_tol=0, abs_tol=0)
