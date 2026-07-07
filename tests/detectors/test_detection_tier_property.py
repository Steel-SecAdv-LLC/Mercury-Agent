# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Property-based contract tests for the hardened detector tier.

Hypothesis generates arbitrary finite and non-finite arrays and asserts the tier
contract holds for every one of them:

* the per-point ``scores`` vector is finite and within ``[0, 1]``;
* the scalar ``anomaly_score`` is finite and within ``[0, 1]``;
* scalar ``metadata`` fields (e.g. SPOT's ``z_q`` / ``gamma``) are finite.

These are the invariants the explicit NaN policy + metadata guards + unified
magnitude regime are supposed to guarantee regardless of input.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

hyp = pytest.importorskip("hypothesis")
from hypothesis import (
    HealthCheck,
    given,
    settings,
    strategies as st,
)
from hypothesis.extra import numpy as hnp

from omni_mercury_engine.detectors.bocpd import BOCPDDetector
from omni_mercury_engine.detectors.digital_twin import (
    DigitalTwinResidualDetector,
)
from omni_mercury_engine.detectors.rca import RootCauseGraphDetector
from omni_mercury_engine.detectors.spot_evt import SPOTDetector

# Arbitrary values including NaN / +-inf.
_any_float = st.floats(allow_nan=True, allow_infinity=True, width=64)
_series = hnp.arrays(dtype=np.float64, shape=st.integers(12, 150), elements=_any_float)
_train = hnp.arrays(
    dtype=np.float64,
    shape=st.integers(40, 200),
    elements=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
)


def _assert_scores_contract(result: dict[str, Any]) -> None:
    scores = np.asarray(result["scores"], dtype=np.float64)
    assert np.all(np.isfinite(scores)), "score vector must be finite"
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0), "scores must be in [0, 1]"
    anomaly = float(result["anomaly_score"])
    assert np.isfinite(anomaly) and 0.0 <= anomaly <= 1.0


def _assert_metadata_finite(result: dict[str, Any]) -> None:
    for key, value in result.get("metadata", {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            assert np.isfinite(value), f"metadata field {key!r} must be finite"


_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


class TestScoreVectorContract:
    @_SETTINGS
    @given(_train, _series)
    def test_spot_scores_finite_bounded(self, train: np.ndarray, probe: np.ndarray) -> None:
        det = SPOTDetector().fit(train)
        result = det.detect(probe)
        _assert_scores_contract(result)
        _assert_metadata_finite(result)

    @_SETTINGS
    @given(_train, _series)
    def test_digital_twin_scores_finite_bounded(self, train: np.ndarray, probe: np.ndarray) -> None:
        det = DigitalTwinResidualDetector().fit(train)
        _assert_scores_contract(det.detect(probe))

    @_SETTINGS
    @given(_train, _series)
    def test_bocpd_scores_finite_bounded(self, train: np.ndarray, probe: np.ndarray) -> None:
        det = BOCPDDetector(max_run_length=40).fit(train)
        _assert_scores_contract(det.detect(probe))


class TestMetadataInvariants:
    @_SETTINGS
    @given(_train, _series)
    def test_spot_metadata_always_finite(self, train: np.ndarray, probe: np.ndarray) -> None:
        det = SPOTDetector().fit(train)
        meta = det.detect(probe)["metadata"]
        assert np.isfinite(meta["z_q"]) and np.isfinite(meta["gamma"])

    @settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        hnp.arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(10, 60), st.integers(2, 5)),
            elements=_any_float,
        )
    )
    def test_rca_scores_and_ranked_causes_finite(self, rows: np.ndarray) -> None:
        det = RootCauseGraphDetector().fit(np.nan_to_num(rows))
        result = det.detect(rows)
        _assert_scores_contract(result)
        # ranked-cause attributions are finite probabilities that sum to ~1
        ranked = result["metadata"]["ranked_causes"]
        weights = [w for _, w in ranked]
        assert all(np.isfinite(w) for w in weights)
        if weights:
            assert abs(sum(weights) - 1.0) < 1e-6

    @_SETTINGS
    @given(_train, _series, st.sampled_from(["neutral", "impute", "flag"]))
    def test_scores_finite_under_all_policies(
        self, train: np.ndarray, probe: np.ndarray, policy: str
    ) -> None:
        det = SPOTDetector(config={"nan_policy": policy}).fit(train)
        _assert_scores_contract(det.detect(probe))
