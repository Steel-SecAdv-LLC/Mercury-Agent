# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A quantity is measured only when it carries the artifact that proves it.

``MeasurementEvidence`` is that artifact: how the quantity was measured, on
how many observations, and the SHA-256 of the numbers it was measured on.
These tests pin the three properties the contract rests on:

* evidence that proves nothing cannot be built (it raises);
* evidence is bound to content: the digest is of the numbers themselves, the
  same on every platform, and different numbers give a different digest;
* reading evidence back from a record is total: anything that is not valid
  evidence is the absence of evidence, never an exception and never a pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import (
    given,
    settings,
    strategies as st,
)

from omni_mercury_engine.measurement import METHODS, MeasurementEvidence, digest_of

_DIGEST = "ab" * 32


class TestEvidenceMustProveSomething:
    def test_a_valid_evidence_round_trips(self) -> None:
        evidence = MeasurementEvidence("ensemble_variance", 5, _DIGEST)
        assert MeasurementEvidence.from_dict(evidence.to_dict()) == evidence
        assert evidence.to_dict() == {"method": "ensemble_variance", "n": 5, "digest": _DIGEST}

    def test_a_verdict_carries_its_statistic_and_cut(self) -> None:
        evidence = MeasurementEvidence(
            "conformal_rank", 40, _DIGEST, statistic=0.012, threshold=0.05
        )
        assert evidence.to_dict()["statistic"] == 0.012
        assert evidence.to_dict()["threshold"] == 0.05
        assert MeasurementEvidence.from_dict(evidence.to_dict()) == evidence

    @pytest.mark.parametrize("method", sorted(METHODS))
    def test_fewer_observations_than_the_method_needs_is_not_a_measurement(
        self, method: str
    ) -> None:
        floor = METHODS[method]
        MeasurementEvidence(method, floor, _DIGEST)  # exactly enough
        with pytest.raises(ValueError, match="at least"):
            MeasurementEvidence(method, floor - 1, _DIGEST)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"method": "trust_me", "n": 9, "digest": _DIGEST}, "unknown measurement method"),
            ({"method": "mc_dropout", "n": True, "digest": _DIGEST}, "integer count"),
            ({"method": "mc_dropout", "n": 2.0, "digest": _DIGEST}, "integer count"),
            ({"method": "mc_dropout", "n": 9, "digest": ""}, "SHA-256"),
            ({"method": "mc_dropout", "n": 9, "digest": "AB" * 32}, "SHA-256"),
            ({"method": "mc_dropout", "n": 9, "digest": "ab" * 31}, "SHA-256"),
            (
                {"method": "conformal_rank", "n": 30, "digest": _DIGEST, "statistic": float("nan")},
                "finite",
            ),
            (
                {"method": "conformal_rank", "n": 30, "digest": _DIGEST, "threshold": float("inf")},
                "finite",
            ),
        ],
    )
    def test_malformed_evidence_cannot_be_built(self, kwargs: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            MeasurementEvidence(**kwargs)

    def test_evidence_is_immutable(self) -> None:
        evidence = MeasurementEvidence("mc_sampling", 20, _DIGEST)
        with pytest.raises(AttributeError):
            evidence.n = 1  # type: ignore[misc]


class TestEvidenceIsBoundToContent:
    def test_from_samples_takes_the_count_and_digest_from_the_numbers(self) -> None:
        stack = np.linspace(0.0, 1.0, 12).reshape(6, 2)
        evidence = MeasurementEvidence.from_samples("ensemble_variance", stack)
        assert evidence.n == 6
        assert evidence.digest == digest_of(stack)

    def test_the_digest_is_the_same_on_every_platform(self) -> None:
        """A fixed vector has a fixed digest: little-endian doubles and the shape."""
        assert (
            digest_of([0.0, 0.5, 1.0])
            == "57b1168e9e5e02ba4866cb6f974d357bbdf0f764a9839252b5f80039e1974a00"
        )
        # Byte order and dtype of the input do not matter; the values do.
        native = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        swapped = native.astype(">f8")
        single = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        assert digest_of(native) == digest_of(swapped) == digest_of(single)
        assert digest_of(native) == digest_of([0.0, 0.5, 1.0])

    def test_different_numbers_or_a_different_arrangement_give_a_different_digest(self) -> None:
        base = np.arange(6.0)
        assert digest_of(base) != digest_of(base + 1e-12)
        assert digest_of(base.reshape(2, 3)) != digest_of(base.reshape(3, 2))
        assert digest_of(base[:3], base[3:]) != digest_of(base)

    def test_evidence_needs_numbers(self) -> None:
        with pytest.raises(ValueError, match="needs the numbers"):
            MeasurementEvidence.from_samples("mc_dropout")


_junk = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-(10**6), max_value=10**6),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(max_size=70),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=4), st.dictionaries(st.text(max_size=10), children, max_size=5)
    ),
    max_leaves=12,
)
_almost = st.fixed_dictionaries(
    {},
    optional={
        "method": st.one_of(st.sampled_from(sorted(METHODS)), _junk),
        "n": st.one_of(st.integers(min_value=-5, max_value=60), _junk),
        "digest": st.one_of(st.just(_DIGEST), _junk),
        "statistic": _junk,
        "threshold": _junk,
    },
)


class TestReadingEvidenceBackIsTotal:
    @settings(max_examples=500)
    @given(payload=st.one_of(_junk, _almost))
    def test_anything_is_either_valid_evidence_or_none(self, payload: Any) -> None:
        evidence = MeasurementEvidence.from_dict(payload)  # must not raise
        if evidence is None:
            return
        # What it accepted is evidence in every respect, and survives the wire.
        assert evidence.method in METHODS
        assert evidence.n >= METHODS[evidence.method]
        assert len(evidence.digest) == 64
        assert MeasurementEvidence.from_dict(evidence.to_dict()) == evidence

    @pytest.mark.parametrize(
        "payload",
        [None, True, 1, "measured", [], {}, {"method": "mc_dropout"}, {"n": 5, "digest": _DIGEST}],
    )
    def test_what_is_not_evidence_reads_as_none(self, payload: Any) -> None:
        assert MeasurementEvidence.from_dict(payload) is None
