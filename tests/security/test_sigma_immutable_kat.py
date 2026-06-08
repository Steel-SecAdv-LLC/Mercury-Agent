# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""KAT-style regression for the trained σ_Immutable EthicalGate.

These tests assert that the trained gate produces deterministic,
correct outputs on hand-curated known-good and known-bad scalar
vectors.  Any re-training that shifts the decision boundary must
update the KAT vectors here — the test is the contract.

The labelling source is documented in
``scripts/train_sigma_immutable.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.core.global_omni_scalar_network import EthicalGate
from omni_mercury_engine.security.sigma_immutable_gate import (
    SIGMA_ETHICAL_BAND_END,
    SIGMA_IMMUTABLE_DEFAULT_THRESHOLD,
    SIGMA_IMMUTABLE_DIM,
    SIGMA_USED_BAND_END,
)

# ---------------------------------------------------------------------------
# KAT vectors — hand-curated, deterministic
#
# Training data distribution (scripts/train_sigma_immutable.py):
#   - Ethical scalars (indices 0..SIGMA_ETHICAL_BAND_END):
#     U[threshold, 2.0] for positive,
#     1-5 drawn from U[0, threshold-0.01] for negative
#   - Non-ethical scalars (indices SIGMA_ETHICAL_BAND_END..SIGMA_USED_BAND_END):
#     U[0.0, 2.0]
#   - Remaining indices (SIGMA_USED_BAND_END..SIGMA_IMMUTABLE_DIM):
#     zero-padded
# ---------------------------------------------------------------------------

_NONETHICAL_DIMS = SIGMA_USED_BAND_END - SIGMA_ETHICAL_BAND_END
_THRESHOLD = SIGMA_IMMUTABLE_DEFAULT_THRESHOLD

# Deterministic RNG for non-ethical fill (seed=999, distinct from training)
_rng = np.random.default_rng(999)

# ── Positive vector: all ethical scalars well above threshold ──
KAT_POSITIVE = np.zeros(SIGMA_IMMUTABLE_DIM, dtype=np.float32)
KAT_POSITIVE[:SIGMA_ETHICAL_BAND_END] = _rng.uniform(
    _THRESHOLD, 2.0, SIGMA_ETHICAL_BAND_END
).astype(np.float32)
KAT_POSITIVE[SIGMA_ETHICAL_BAND_END:SIGMA_USED_BAND_END] = _rng.uniform(
    0.0, 2.0, _NONETHICAL_DIMS
).astype(np.float32)

# ── Negative vector: 5 critical ethical scalars below threshold ──
KAT_NEGATIVE = np.zeros(SIGMA_IMMUTABLE_DIM, dtype=np.float32)
KAT_NEGATIVE[:SIGMA_ETHICAL_BAND_END] = np.array(
    [
        0.1,
        0.2,
        0.15,
        0.3,
        0.05,  # 5 violations
        1.1,
        1.7,
        1.8,
        1.5,
        1.3,  # rest above threshold
        1.4,
        1.2,
        1.6,
        1.1,
        1.5,
        1.3,
        1.4,
        1.7,
        1.2,
        1.6,
        1.1,
        1.5,
        1.8,
        1.3,
        1.4,
        1.2,
        1.6,
    ],
    dtype=np.float32,
)
KAT_NEGATIVE[SIGMA_ETHICAL_BAND_END:SIGMA_USED_BAND_END] = _rng.uniform(
    0.0, 2.0, _NONETHICAL_DIMS
).astype(np.float32)

# ── All-low ethical vector: all ethical scalars below threshold ──
KAT_ALL_LOW = np.zeros(SIGMA_IMMUTABLE_DIM, dtype=np.float32)
KAT_ALL_LOW[:SIGMA_ETHICAL_BAND_END] = np.array(
    [
        0.1,
        0.2,
        0.15,
        0.3,
        0.05,
        0.25,
        0.1,
        0.2,
        0.15,
        0.3,
        0.05,
        0.25,
        0.1,
        0.2,
        0.15,
        0.3,
        0.05,
        0.25,
        0.1,
        0.2,
        0.15,
        0.3,
        0.05,
        0.25,
        0.1,
        0.2,
        0.15,
    ],
    dtype=np.float32,
)
KAT_ALL_LOW[SIGMA_ETHICAL_BAND_END:SIGMA_USED_BAND_END] = _rng.uniform(
    0.0, 2.0, _NONETHICAL_DIMS
).astype(np.float32)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSigmaImmutableKAT:
    """KAT regression for the trained σ_Immutable gate."""

    @pytest.fixture(autouse=True)
    def _gate(self) -> None:
        self.gate = EthicalGate(threshold=_THRESHOLD)

    def test_gate_loads_trained_weights(self) -> None:
        """The gate must load trained weights from the in-repo artifact."""
        assert self.gate._trained, (
            "EthicalGate did not load trained weights — "
            "ensure security/sigma_immutable_weights.pt exists"
        )

    def test_positive_vector_passes(self) -> None:
        """A known-good ethical scalar vector must pass the gate."""
        passes, score = self.gate.evaluate(KAT_POSITIVE)
        assert passes, f"KAT positive vector failed with score={score:.4f}"
        assert score >= _THRESHOLD, f"Score {score:.4f} below threshold {_THRESHOLD}"

    def test_negative_vector_fails(self) -> None:
        """A known-bad vector (5 ethical violations) must fail the gate."""
        passes, score = self.gate.evaluate(KAT_NEGATIVE)
        assert not passes, f"KAT negative vector incorrectly passed with score={score:.4f}"
        assert score < _THRESHOLD, f"Score {score:.4f} unexpectedly above threshold {_THRESHOLD}"

    def test_all_low_fails(self) -> None:
        """A vector with all ethical scalars below threshold must fail."""
        passes, score = self.gate.evaluate(KAT_ALL_LOW)
        assert not passes, f"All-low vector passed with score={score:.4f}"
        assert score < 0.1, f"All-low score {score:.4f} unexpectedly high"

    def test_all_zeros_fails(self) -> None:
        """An all-zeros vector (no ethical signal) must fail the gate."""
        zeros = np.zeros(SIGMA_IMMUTABLE_DIM, dtype=np.float32)
        passes, score = self.gate.evaluate(zeros)
        assert not passes, f"All-zeros vector passed with score={score:.4f}"

    def test_deterministic(self) -> None:
        """Gate output must be deterministic across repeated evaluations."""
        _, score1 = self.gate.evaluate(KAT_POSITIVE)
        _, score2 = self.gate.evaluate(KAT_POSITIVE)
        assert score1 == score2, "Gate produced non-deterministic scores"

    def test_nan_handling(self) -> None:
        """A non-finite scalar fails the gate closed (no NaN->0 coercion).

        A NaN/±inf scalar means an upstream computation broke; the gate
        refuses (score 0.0) rather than coercing NaN->0 and scoring the
        result, which used to let a NaN-collapsed dim pass as healthy.
        """
        vec = KAT_POSITIVE.copy()
        vec[0] = float("nan")
        vec[5] = float("nan")
        passes, score = self.gate.evaluate(vec)
        assert passes is False
        assert score == 0.0

    def test_boundary_ethics_at_threshold(self) -> None:
        """Scalars exactly at threshold — the gate's learned boundary."""
        boundary = np.zeros(SIGMA_IMMUTABLE_DIM, dtype=np.float32)
        boundary[:SIGMA_ETHICAL_BAND_END] = _THRESHOLD
        boundary[SIGMA_ETHICAL_BAND_END:SIGMA_USED_BAND_END] = 1.0
        _, score = self.gate.evaluate(boundary)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
