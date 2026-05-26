# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Federated silent-failure regression suite (punch-list item 9).

Covers two gaps the 2026-03 in-tree audit flagged on the federated path:

(A) Conformal prediction: previously caught
    ``(ValueError, RuntimeError, AttributeError)`` and silently set
    ``confidence_intervals=None``. Must now raise
    :class:`ConformalMisconfigurationError` instead, so a misconfig
    cannot silently degrade the engine.

(B) GOSNN integration: previously one-way (server → client). Bidirectional
    coupling now round-trips a deterministic, hash-checked weight vector
    through ``GOSNNCouplingClient → GOSNNCouplingServer → GOSNNCouplingClient``
    with proof of arrival on each leg.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.core.gosnn_integration import (
    ConformalMisconfigurationError,
    GOSNNIntegration,
)
from omni_mercury_engine.federated_learning.gosnn_coupling import (
    GOSNNCouplingClient,
    GOSNNCouplingError,
    GOSNNCouplingServer,
)

# ---------------------------------------------------------------------------
# (A) Conformal misconfiguration must raise, not return None
# ---------------------------------------------------------------------------


class _BrokenConformal:
    """Test double whose predict() always raises."""

    def __init__(self, exc_factory: Any) -> None:
        self._exc_factory = exc_factory

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise self._exc_factory()


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: ValueError("calibration size mismatch"),
        lambda: RuntimeError("conformal predictor not fitted"),
        lambda: AttributeError("missing quantile_threshold"),
    ],
)
def test_conformal_failure_raises_instead_of_returning_none(exc_factory: Any) -> None:
    """A broken conformal predictor must raise, not silently return None."""

    integration = GOSNNIntegration(use_conformal=True)
    integration._conformal = _BrokenConformal(exc_factory)  # type: ignore[assignment]
    integration._fitted = True
    # Minimal viable state for detect() — we only need to reach the conformal
    # branch; everything before it must succeed without external dependencies.
    integration.domains = {}
    integration._domain_weights = {}

    X = np.zeros((4, 2), dtype=np.float64)
    with pytest.raises(ConformalMisconfigurationError) as excinfo:
        integration.detect(X)
    # The cause is preserved for diagnostics.
    assert excinfo.value.original is not None
    assert isinstance(excinfo.value.original, (ValueError, RuntimeError, AttributeError))


def test_conformal_disabled_does_not_raise_and_returns_none_intervals() -> None:
    """``use_conformal=False`` is an explicit, audit-trail-clean opt-out.

    The engine never silently degrades into the no-interval mode; it only
    enters that mode when the caller asked for it. This test pins the
    distinction.
    """
    integration = GOSNNIntegration(use_conformal=False)
    integration._fitted = True
    integration.domains = {}
    integration._domain_weights = {}
    assert integration._conformal is None  # explicit opt-out

    X = np.zeros((4, 2), dtype=np.float64)
    result = integration.detect(X)
    assert result.confidence_intervals is None


def test_gosnn_integration_fit_wires_live_conformal_intervals() -> None:
    """The documented fit() path must produce real conformal bounds."""

    class _Detector:
        def __init__(self, offset: float) -> None:
            self.offset = offset

        def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> _Detector:
            return self

        def detect(self, X: np.ndarray) -> dict[str, np.ndarray]:
            scores = np.clip(np.mean(X, axis=1) + self.offset, 0.0, 1.0)
            return {"scores": scores}

    X = np.linspace(0.0, 1.0, 80, dtype=np.float64).reshape(40, 2)
    y = (np.mean(X, axis=1) > 0.5).astype(int)
    integration = GOSNNIntegration(
        use_calibration=True,
        use_conformal=True,
        conformal_alpha=0.9,
        benevolence_threshold=0.98,
    )
    integration.add_domain("statistical", detector=_Detector(0.0), weight=2.0)
    integration.add_domain("temporal", detector=_Detector(0.1), weight=1.0)

    integration.fit(X, y)
    result = integration.detect(X[:6], return_details=True, use_cache=False)

    assert result.confidence_intervals is not None
    assert sorted(result.domain_scores) == ["statistical", "temporal"]
    assert result.calibration_method == "auto"
    assert result.ethical_compliance is True
    intervals = result.confidence_intervals
    assert intervals["coverage_level"] == 0.9
    lower_bound = intervals["lower_bound"]
    upper_bound = intervals["upper_bound"]
    assert isinstance(lower_bound, np.ndarray)
    assert isinstance(upper_bound, np.ndarray)
    assert lower_bound.shape == (6,)
    assert upper_bound.shape == (6,)
    assert np.all(lower_bound <= upper_bound)


# ---------------------------------------------------------------------------
# (B) Bidirectional GOSNN round-trip
# ---------------------------------------------------------------------------


def _initial_weights(seed: int = 0, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.1, size=dim).astype(np.float64)


def test_gosnn_bidirectional_round_trip_two_clients() -> None:
    """A two-client round trip must produce the FedAvg weighted mean on
    the server and propagate it back to both clients with a digest match."""
    initial = _initial_weights(seed=0, dim=6)
    server = GOSNNCouplingServer(initial_weights=initial)

    client_a = GOSNNCouplingClient(client_id="A", local_weights=initial)
    client_b = GOSNNCouplingClient(client_id="B", local_weights=initial)

    # Each client performs a local "step" (we synthesise the post-step weights
    # directly so the test is independent of any trainer implementation).
    a_step = initial + np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    b_step = initial + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.2])

    a_update = client_a.publish(round_num=0, local_update=a_step, n_samples=100)
    b_update = client_b.publish(round_num=0, local_update=b_step, n_samples=300)

    server.ingest(a_update)
    server.ingest(b_update)
    new_state = server.aggregate()

    expected_global = (100 / 400) * a_step + (300 / 400) * b_step
    np.testing.assert_allclose(new_state.weights, expected_global, atol=1e-12)
    assert new_state.round_num == 1
    assert new_state.contributing_client_ids == ["A", "B"]

    # Server → client propagation: both clients must end up with the same
    # global state, and the digest computed on the receive side must equal
    # the digest published by the server.
    client_a.receive(new_state)
    client_b.receive(new_state)
    np.testing.assert_allclose(client_a.local_weights, expected_global, atol=1e-12)
    np.testing.assert_allclose(client_b.local_weights, expected_global, atol=1e-12)
    assert client_a.last_received_state is not None
    assert client_a.last_received_state.digest == new_state.digest
    assert client_b.last_received_state is not None
    assert client_b.last_received_state.digest == new_state.digest


def test_gosnn_round_trip_idempotent_under_repeat_aggregation() -> None:
    """Aggregating with identical updates twice yields identical global state."""
    initial = _initial_weights(seed=1, dim=4)
    server = GOSNNCouplingServer(initial_weights=initial)
    client = GOSNNCouplingClient(client_id="solo", local_weights=initial)

    update_payload = initial + np.array([0.05, -0.05, 0.05, -0.05])

    server.ingest(client.publish(round_num=0, local_update=update_payload, n_samples=50))
    state_a = server.aggregate()

    server.ingest(client.publish(round_num=1, local_update=update_payload, n_samples=50))
    state_b = server.aggregate()

    np.testing.assert_allclose(state_a.weights, state_b.weights, atol=1e-12)


def test_gosnn_ingest_rejects_shape_mismatch() -> None:
    server = GOSNNCouplingServer(initial_weights=np.zeros(4, dtype=np.float64))
    bad_client = GOSNNCouplingClient(client_id="X", local_weights=np.zeros(7))
    with pytest.raises(GOSNNCouplingError, match="shape"):
        server.ingest(bad_client.publish(round_num=0, n_samples=1))


def test_gosnn_ingest_rejects_round_mismatch() -> None:
    server = GOSNNCouplingServer(initial_weights=np.zeros(4))
    client = GOSNNCouplingClient(client_id="X", local_weights=np.zeros(4))
    with pytest.raises(GOSNNCouplingError, match="round"):
        server.ingest(client.publish(round_num=42, n_samples=1))


def test_gosnn_ingest_rejects_corrupted_digest() -> None:
    server = GOSNNCouplingServer(initial_weights=np.zeros(4))
    client = GOSNNCouplingClient(client_id="X", local_weights=np.zeros(4))
    update = client.publish(round_num=0, n_samples=1)
    # Tamper with the weight payload after the digest was computed.
    update.weights[0] = 999.0
    with pytest.raises(GOSNNCouplingError, match="digest"):
        server.ingest(update)


def test_gosnn_aggregate_rejects_empty_round() -> None:
    server = GOSNNCouplingServer(initial_weights=np.zeros(4))
    with pytest.raises(GOSNNCouplingError, match="no client updates"):
        server.aggregate()


def test_gosnn_client_receive_rejects_corrupted_global_state() -> None:
    server = GOSNNCouplingServer(initial_weights=np.zeros(4))
    client = GOSNNCouplingClient(client_id="X", local_weights=np.zeros(4))
    server.ingest(
        client.publish(round_num=0, local_update=np.array([1.0, 2.0, 3.0, 4.0]), n_samples=1)
    )
    state = server.aggregate()
    state.weights[2] = -999.0  # corruption after publication
    with pytest.raises(GOSNNCouplingError, match="digest"):
        client.receive(state)


def test_gosnn_ingest_is_immune_to_post_ingest_weight_mutation() -> None:
    """Mutating the post-ingest update payload must not affect aggregation.

    Pinning the defensive-snapshot contract.  Without this, a holder of
    the original ``GOSNNUpdate`` object (the publishing client, a
    transport layer buffering it for retransmission, etc.) could mutate
    ``update.weights`` after ``ingest()`` returned and silently change
    the next ``aggregate()`` result — the digest is verified at
    ingest time and not again at aggregation, so the protection has to
    be a deep copy at storage time.
    """
    initial = np.array([0.0, 0.0, 0.0, 0.0])
    server = GOSNNCouplingServer(initial_weights=initial)
    client = GOSNNCouplingClient(client_id="A", local_weights=initial)

    expected = np.array([1.0, 2.0, 3.0, 4.0])
    update = client.publish(round_num=0, local_update=expected.copy(), n_samples=10)
    server.ingest(update)

    # Try to mutate the published payload buffer post-ingest.  Whether
    # the assignment raises (because numpy flagged the buffer as
    # read-only) or succeeds (because the assignment hit the
    # client-side copy), the server's internal snapshot must remain
    # untouched.
    try:
        update.weights[:] = -999.0
    except (ValueError, RuntimeError):
        pass

    state = server.aggregate()
    np.testing.assert_allclose(state.weights, expected)


def test_gosnn_round_trip_preserves_information_across_three_rounds() -> None:
    """Full client → server → client round-trip survives multiple rounds."""
    rng = np.random.default_rng(7)
    initial = rng.normal(0.0, 0.05, size=10)
    server = GOSNNCouplingServer(initial_weights=initial)
    clients = [GOSNNCouplingClient(client_id=f"C{i}", local_weights=initial) for i in range(3)]

    for round_num in range(3):
        for i, c in enumerate(clients):
            local_step = c.local_weights + rng.normal(0.0, 0.01, size=initial.shape)
            server.ingest(c.publish(round_num=round_num, local_update=local_step, n_samples=10 + i))
        state = server.aggregate()
        for c in clients:
            c.receive(state)
        # All clients have identical state after broadcast.
        for c in clients[1:]:
            np.testing.assert_allclose(c.local_weights, clients[0].local_weights, atol=1e-12)
        assert state.round_num == round_num + 1
