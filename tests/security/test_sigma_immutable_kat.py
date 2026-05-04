"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

KAT-style regression for the trained σ_Immutable EthicalGate.

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

# ---------------------------------------------------------------------------
# KAT vectors — hand-curated, deterministic
#
# Training data distribution (scripts/train_sigma_immutable.py):
#   - Ethical scalars (indices 0-26): U[threshold, 2.0] for positive,
#     1-5 drawn from U[0, threshold-0.01] for negative
#   - Non-ethical scalars (indices 27-179): U[0.0, 2.0]
#   - Remaining indices (180-255): zero-padded
# ---------------------------------------------------------------------------

# Deterministic RNG for non-ethical fill (seed=999, distinct from training)
_rng = np.random.default_rng(999)

# ── Positive vector: all 27 ethical scalars well above threshold ──
KAT_POSITIVE = np.zeros(256, dtype=np.float32)
KAT_POSITIVE[:27] = _rng.uniform(0.93, 2.0, 27).astype(np.float32)
KAT_POSITIVE[27:180] = _rng.uniform(0.0, 2.0, 153).astype(np.float32)

# ── Negative vector: 5 critical ethical scalars below threshold ──
KAT_NEGATIVE = np.zeros(256, dtype=np.float32)
KAT_NEGATIVE[:27] = np.array(
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
KAT_NEGATIVE[27:180] = _rng.uniform(0.0, 2.0, 153).astype(np.float32)

# ── All-low ethical vector: all 27 ethical scalars below threshold ──
KAT_ALL_LOW = np.zeros(256, dtype=np.float32)
KAT_ALL_LOW[:27] = np.array(
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
KAT_ALL_LOW[27:180] = _rng.uniform(0.0, 2.0, 153).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSigmaImmutableKAT:
    """KAT regression for the trained σ_Immutable gate."""

    @pytest.fixture(autouse=True)
    def _gate(self) -> None:
        self.gate = EthicalGate(threshold=0.93)

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
        assert score >= 0.93, f"Score {score:.4f} below threshold 0.93"

    def test_negative_vector_fails(self) -> None:
        """A known-bad vector (5 ethical violations) must fail the gate."""
        passes, score = self.gate.evaluate(KAT_NEGATIVE)
        assert not passes, f"KAT negative vector incorrectly passed with score={score:.4f}"
        assert score < 0.93, f"Score {score:.4f} unexpectedly above threshold 0.93"

    def test_all_low_fails(self) -> None:
        """A vector with all ethical scalars below threshold must fail."""
        passes, score = self.gate.evaluate(KAT_ALL_LOW)
        assert not passes, f"All-low vector passed with score={score:.4f}"
        assert score < 0.1, f"All-low score {score:.4f} unexpectedly high"

    def test_all_zeros_fails(self) -> None:
        """An all-zeros vector (no ethical signal) must fail the gate."""
        zeros = np.zeros(256, dtype=np.float32)
        passes, score = self.gate.evaluate(zeros)
        assert not passes, f"All-zeros vector passed with score={score:.4f}"

    def test_deterministic(self) -> None:
        """Gate output must be deterministic across repeated evaluations."""
        _, score1 = self.gate.evaluate(KAT_POSITIVE)
        _, score2 = self.gate.evaluate(KAT_POSITIVE)
        assert score1 == score2, "Gate produced non-deterministic scores"

    def test_nan_handling(self) -> None:
        """NaN values in the scalar vector are replaced with zeros."""
        vec = KAT_POSITIVE.copy()
        vec[0] = float("nan")
        vec[5] = float("nan")
        passes, score = self.gate.evaluate(vec)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_boundary_ethics_at_threshold(self) -> None:
        """Scalars exactly at threshold — the gate's learned boundary."""
        boundary = np.zeros(256, dtype=np.float32)
        boundary[:27] = 0.93
        boundary[27:180] = 1.0
        _, score = self.gate.evaluate(boundary)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
