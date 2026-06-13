# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Evidence extraction reads the real pipeline keys, defensively."""

from __future__ import annotations

import math

from omni_mercury_engine.decision import Evidence


class TestExtraction:
    def test_reads_core_fields(self) -> None:
        ev = Evidence.from_detection(
            {"anomaly_prob": 0.8, "is_anomaly": True, "threshold_used": 0.4, "severity": 0.6}
        )
        assert ev.anomaly_prob == 0.8
        assert ev.is_anomaly is True
        assert ev.threshold == 0.4
        assert ev.severity == 0.6
        assert ev.calibrated is False

    def test_reads_conformal_certificate(self) -> None:
        ev = Evidence.from_detection(
            {
                "anomaly_prob": 0.7,
                "conformal": {
                    "prediction_set": [0, 1],
                    "set_size": 2,
                    "abstain": True,
                    "coverage": 0.9,
                },
            }
        )
        assert ev.calibrated is True
        assert ev.conformal_set_size == 2
        assert ev.conformal_labels == (0, 1)
        assert ev.coverage == 0.9

    def test_reads_gosnn_and_symbolic_and_drift(self) -> None:
        ev = Evidence.from_detection(
            {
                "anomaly_prob": 0.7,
                "gosnn_metadata": {
                    "ethical_gate_passed": True,
                    "sigma_immutable_score": 0.97,
                    "sigma_immutable_threshold": 0.93,
                },
                "symbolic_consistency": {"satisfaction": 0.42},
                "drift_detection": {"is_drift": True, "severity": "HIGH"},
            }
        )
        assert ev.ethical_gate_passed is True
        assert ev.ethical_score == 0.97
        assert ev.symbolic_satisfaction == 0.42
        assert ev.drift_detected is True
        assert ev.drift_severity == "HIGH"

    def test_threshold_falls_back(self) -> None:
        # No threshold_used, but a bare "threshold" -> used; else 0.5 default.
        assert Evidence.from_detection({"anomaly_prob": 0.6, "threshold": 0.3}).threshold == 0.3
        assert Evidence.from_detection({"anomaly_prob": 0.6}).threshold == 0.5


class TestDefensiveParsing:
    def test_empty_result_does_not_raise(self) -> None:
        ev = Evidence.from_detection({})
        assert ev.anomaly_prob == 0.0
        assert ev.calibrated is False
        assert ev.ethical_gate_passed is None

    def test_non_finite_probability_is_dropped_to_zero(self) -> None:
        ev = Evidence.from_detection({"anomaly_prob": math.nan})
        assert ev.anomaly_prob == 0.0

    def test_malformed_sections_are_ignored(self) -> None:
        ev = Evidence.from_detection(
            {"anomaly_prob": 0.5, "conformal": "not-a-dict", "gosnn_metadata": 7}
        )
        assert ev.conformal_set_size is None
        assert ev.ethical_gate_passed is None

    def test_none_ethical_verdict_preserved_as_none(self) -> None:
        ev = Evidence.from_detection(
            {"anomaly_prob": 0.5, "gosnn_metadata": {"ethical_gate_passed": None}}
        )
        assert ev.ethical_gate_passed is None

    def test_domain_override_beats_result(self) -> None:
        ev = Evidence.from_detection(
            {"anomaly_prob": 0.5, "domain": "in_result"}, domain="explicit"
        )
        assert ev.domain == "explicit"
        ev2 = Evidence.from_detection({"anomaly_prob": 0.5, "domain": "in_result"})
        assert ev2.domain == "in_result"

    def test_to_dict_round_trips_keys(self) -> None:
        ev = Evidence.from_detection({"anomaly_prob": 0.5})
        d = ev.to_dict()
        assert d["anomaly_prob"] == 0.5
        assert d["calibrated"] is False


class _BadItem:
    """A scalar-like whose ``.item()`` raises -- exercises the coercion guard."""

    def item(self) -> float:
        raise ValueError("cannot materialise")


class TestScalarCoercionEdges:
    def test_item_that_raises_is_treated_as_absent(self) -> None:
        # ``.item()`` raising must collapse to the 0.0 fallback, never propagate.
        ev = Evidence.from_detection({"anomaly_prob": _BadItem(), "severity": _BadItem()})
        assert ev.anomaly_prob == 0.0
        assert ev.severity == 0.0

    def test_unfloatable_value_is_treated_as_absent(self) -> None:
        ev = Evidence.from_detection({"anomaly_prob": "not-a-number"})
        assert ev.anomaly_prob == 0.0

    def test_conformal_set_size_without_prediction_set_is_no_certificate(self) -> None:
        # A bare set_size with no usable label set must NOT masquerade as a
        # calibrated certificate -- otherwise the gate would treat it as a
        # singleton and guess a label the certificate never made.
        ev = Evidence.from_detection(
            {"anomaly_prob": 0.7, "conformal": {"set_size": 1, "coverage": 0.9}}
        )
        assert ev.conformal_set_size is None
        assert ev.conformal_labels is None
        assert ev.calibrated is False
        # No certificate -> no coverage provenance (no contradictory pairing).
        assert ev.coverage is None

    def test_conformal_prediction_set_without_set_size(self) -> None:
        # set_size is derived from the label set (the source of truth).
        ev = Evidence.from_detection(
            {"anomaly_prob": 0.7, "conformal": {"prediction_set": [0, 1], "coverage": 0.9}}
        )
        assert ev.conformal_set_size == 2
        assert ev.conformal_labels == (0, 1)

    def test_non_integer_label_set_is_treated_as_no_certificate(self) -> None:
        # A non-numeric or non-integral label invalidates the whole set, so a
        # malformed certificate is reported as absent rather than half-coerced.
        for bad_labels in (["x"], [1.5], [0, "x"], "not-a-list"):
            ev = Evidence.from_detection(
                {"anomaly_prob": 0.7, "conformal": {"prediction_set": bad_labels, "coverage": 0.9}}
            )
            assert ev.conformal_set_size is None, bad_labels
            assert ev.conformal_labels is None
            assert ev.calibrated is False
            assert ev.coverage is None  # coverage is not recorded without a certificate

    def test_string_int_labels_coerce(self) -> None:
        # Clean string-ints are still coerced (defensive, not brittle).
        ev = Evidence.from_detection(
            {"anomaly_prob": 0.7, "conformal": {"prediction_set": ["1"], "coverage": 0.9}}
        )
        assert ev.conformal_labels == (1,)
        assert ev.conformal_set_size == 1

    def test_non_bool_ethical_verdict_is_treated_as_absent(self) -> None:
        # A hard safety gate must not fail open: a stray "False" string (or any
        # non-bool) is treated as absent, never coerced to True via bool().
        for bad in ("False", "true", 1, 0):
            ev = Evidence.from_detection(
                {"anomaly_prob": 0.5, "gosnn_metadata": {"ethical_gate_passed": bad}}
            )
            assert ev.ethical_gate_passed is None, bad
