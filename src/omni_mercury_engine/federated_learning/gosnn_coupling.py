# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Bidirectional GOSNN weight coupling for federated learning.

Replaces the prior one-way (server → client) integration flagged by the
2026-03 in-tree audit (``docs/COMPREHENSIVE_REPO_AUDIT.md`` §1) and the
ROADMAP's federated-learning row. Provides a deterministic, hash-checked
round-trip protocol so a client can publish its local GOSNN scalar update
to the server, the server can fold it into the global state via FedAvg,
and the client can read the updated global state back.

Integrity layer
---------------
Every payload is digested with **SHA3-256** (FIPS 202), matching the
``hash_algorithm = "sha3-256"`` standard pinned by Mercury's AMA
Cryptography surface (``security/crypto_api.py::CryptoPackageConfig``).
This anchors federated-learning integrity to the same hash function the
rest of the AMA-backed crypto stack uses for content hashes, so a Mercury
deployment with the AMA native C library built will see identical hash
outputs from this module and from
``MercuryCrypto.create_crypto_package(...).data_hash``.

Authenticated signing of updates with Ed25519 / ML-DSA-65 sits above this
layer and is deferred to ``federated_learning/server.py`` /
``client.py`` — this module only owns the protocol shape and the hash
contract, not key distribution.

Out of scope: this module does not perform federated *training* of GOSNN
weights — that is the federated_learning/server.py + client.py responsibility.
This module is the protocol layer that makes the round-trip *bidirectional*
rather than one-way, and is independently testable.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _digest(weights: NDArray[np.float64]) -> str:
    """SHA3-256 digest of contiguous float64 weight bytes.

    SHA3-256 (FIPS 202) is the content-hash standard pinned by Mercury's
    AMA Cryptography surface — see
    ``security/crypto_api.py::CryptoPackageConfig.hash_algorithm`` —
    so federated-coupling integrity hashes line up with the
    ``CryptoPackageResult.data_hash`` field produced by the AMA
    ``create_crypto_package`` pipeline for the same byte stream.
    """
    arr = np.ascontiguousarray(weights, dtype=np.float64)
    return hashlib.sha3_256(arr.tobytes()).hexdigest()


@dataclass
class GOSNNUpdate:
    """A single client's GOSNN scalar update for one federated round."""

    client_id: str
    round_num: int
    weights: NDArray[np.float64]
    n_samples: int
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        self.digest = _digest(self.weights)


@dataclass
class GOSNNGlobalState:
    """Server-side aggregated GOSNN state distributed back to clients."""

    round_num: int
    weights: NDArray[np.float64]
    contributing_client_ids: list[str]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        self.digest = _digest(self.weights)


class GOSNNCouplingError(RuntimeError):
    """Raised on bidirectional GOSNN protocol violations."""


class GOSNNCouplingServer:
    """Server-side aggregator for bidirectional GOSNN scalar coupling.

    Maintains a single global weight vector. Each ``ingest`` call records
    one client's update for the current round; ``aggregate`` produces the
    new global state by FedAvg-style sample-weighted mean and broadcasts
    it back to clients via ``current_global_state``.
    """

    def __init__(self, initial_weights: NDArray[np.float64]) -> None:
        self._global_weights = np.ascontiguousarray(
            np.asarray(initial_weights, dtype=np.float64).copy()
        )
        self._lock = threading.Lock()
        self._round_num = 0
        self._pending_updates: dict[str, GOSNNUpdate] = {}
        self._last_aggregation_client_ids: list[str] = []

    @property
    def global_weights(self) -> NDArray[np.float64]:
        """Return a defensive copy of the current global weight vector."""
        with self._lock:
            return self._global_weights.copy()

    @property
    def round_num(self) -> int:
        """Return the round number the server is currently accepting updates for."""
        with self._lock:
            return self._round_num

    @property
    def n_pending_updates(self) -> int:
        """Return the count of client updates ingested into the current round."""
        with self._lock:
            return len(self._pending_updates)

    def current_global_state(self) -> GOSNNGlobalState:
        """Snapshot of the current global state for clients to consume."""
        with self._lock:
            return GOSNNGlobalState(
                round_num=self._round_num,
                weights=self._global_weights.copy(),
                contributing_client_ids=list(self._last_aggregation_client_ids),
            )

    def ingest(self, update: GOSNNUpdate) -> None:
        """Record one client's update for the current round.

        The server stores a frozen snapshot of the update so that a
        client (or any other holder of a reference to the original
        ``update`` object) cannot mutate ``update.weights`` after
        ingest and silently change the next ``aggregate()`` result —
        the digest is verified at ingest time, not again at aggregation.
        We achieve this by copying ``weights`` into a contiguous,
        write-protected buffer and reconstructing the dataclass; the
        recomputed digest on the snapshot is what the rest of the
        pipeline trusts.

        Raises:
            GOSNNCouplingError: If the update's shape doesn't match the
                global weight vector, the update's recomputed digest
                disagrees with the published one (transmission corruption),
                or the round number is not the server's current round.
        """
        if update.weights.shape != self._global_weights.shape:
            raise GOSNNCouplingError(
                f"GOSNN update from {update.client_id!r} has shape "
                f"{update.weights.shape}, expected {self._global_weights.shape}"
            )
        recomputed = _digest(update.weights)
        if recomputed != update.digest:
            raise GOSNNCouplingError(
                f"GOSNN update from {update.client_id!r} failed digest check "
                f"(expected {update.digest}, recomputed {recomputed})"
            )
        # Defensive deep snapshot.  ``np.array(..., copy=True)`` always
        # allocates a new buffer; ``writeable=False`` makes any
        # downstream attempt to mutate the stored array fail loudly
        # rather than silently corrupting an in-flight aggregation.
        frozen_weights = np.array(update.weights, dtype=np.float64, copy=True)
        frozen_weights.setflags(write=False)
        snapshot = GOSNNUpdate(
            client_id=update.client_id,
            round_num=update.round_num,
            weights=frozen_weights,
            n_samples=update.n_samples,
        )
        with self._lock:
            if snapshot.round_num != self._round_num:
                raise GOSNNCouplingError(
                    f"GOSNN update from {snapshot.client_id!r} targets round "
                    f"{snapshot.round_num}, server is at round {self._round_num}"
                )
            self._pending_updates[snapshot.client_id] = snapshot

    def aggregate(self) -> GOSNNGlobalState:
        """Aggregate pending updates into the global state and advance the round.

        FedAvg-style weighted mean by ``n_samples``. Empty rounds raise.
        """
        with self._lock:
            if not self._pending_updates:
                raise GOSNNCouplingError(
                    f"Cannot aggregate round {self._round_num}: no client " "updates were ingested"
                )
            total_samples = sum(u.n_samples for u in self._pending_updates.values())
            if total_samples <= 0:
                raise GOSNNCouplingError(
                    "Total sample count across pending updates is non-positive"
                )

            new_weights = np.zeros_like(self._global_weights)
            client_ids: list[str] = []
            for client_id, update in self._pending_updates.items():
                weight = update.n_samples / total_samples
                new_weights += weight * update.weights
                client_ids.append(client_id)

            self._global_weights = np.ascontiguousarray(new_weights)
            self._last_aggregation_client_ids = sorted(client_ids)
            self._round_num += 1
            self._pending_updates.clear()

            return GOSNNGlobalState(
                round_num=self._round_num,
                weights=self._global_weights.copy(),
                contributing_client_ids=list(self._last_aggregation_client_ids),
            )


class GOSNNCouplingClient:
    """Client-side coupling: receives global state, publishes local updates."""

    def __init__(
        self,
        client_id: str,
        local_weights: NDArray[np.float64],
    ) -> None:
        self._client_id = client_id
        self._local_weights = np.ascontiguousarray(
            np.asarray(local_weights, dtype=np.float64).copy()
        )
        self._last_received_state: GOSNNGlobalState | None = None
        self._lock = threading.Lock()

    @property
    def client_id(self) -> str:
        """Return this client's stable identifier used in publish/aggregate."""
        return self._client_id

    @property
    def local_weights(self) -> NDArray[np.float64]:
        """Return a defensive copy of the client's current local weight vector."""
        with self._lock:
            return self._local_weights.copy()

    @property
    def last_received_state(self) -> GOSNNGlobalState | None:
        """Return the most recent global state installed via ``receive`` (or None)."""
        with self._lock:
            return self._last_received_state

    def receive(self, global_state: GOSNNGlobalState) -> None:
        """Server → client: install the latest global state locally.

        Raises:
            GOSNNCouplingError: On shape mismatch or digest corruption.
        """
        if global_state.weights.shape != self._local_weights.shape:
            raise GOSNNCouplingError(
                f"Global state shape {global_state.weights.shape} does not "
                f"match local shape {self._local_weights.shape} for client "
                f"{self._client_id!r}"
            )
        if _digest(global_state.weights) != global_state.digest:
            raise GOSNNCouplingError(
                f"Global state digest mismatch on receive at client " f"{self._client_id!r}"
            )
        with self._lock:
            self._local_weights = np.ascontiguousarray(global_state.weights.copy())
            self._last_received_state = global_state

    def publish(
        self,
        round_num: int,
        local_update: NDArray[np.float64] | None = None,
        n_samples: int = 1,
    ) -> GOSNNUpdate:
        """Client → server: produce a publishable update.

        Args:
            round_num: The federated round this update targets.
            local_update: Optional explicit weight tensor; when ``None``,
                the client publishes its current ``local_weights`` (e.g.
                after a local gradient step).
            n_samples: Number of training samples backing this update,
                used for the FedAvg weighted mean on the server.
        """
        if n_samples <= 0:
            raise GOSNNCouplingError(
                f"n_samples must be positive (got {n_samples}) — federated "
                "averaging requires non-zero per-client weight"
            )
        with self._lock:
            payload = (
                self._local_weights.copy()
                if local_update is None
                else np.ascontiguousarray(np.asarray(local_update, dtype=np.float64).copy())
            )
        if payload.shape != self._local_weights.shape:
            raise GOSNNCouplingError(
                f"local_update shape {payload.shape} does not match local "
                f"shape {self._local_weights.shape}"
            )
        return GOSNNUpdate(
            client_id=self._client_id,
            round_num=round_num,
            weights=payload,
            n_samples=n_samples,
        )


__all__ = [
    "GOSNNCouplingClient",
    "GOSNNCouplingError",
    "GOSNNCouplingServer",
    "GOSNNGlobalState",
    "GOSNNUpdate",
]
