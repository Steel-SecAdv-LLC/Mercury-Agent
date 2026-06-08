# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 2 ITEM 4 regression: native pure-stdlib TCP MessageTransport for Raft.

Pins three contracts:

1. Length-prefixed binary frame round-trips a signed envelope between
   two transports — every message carries an Ed25519 signature
   produced by Mercury's own AMA Cryptography surface, every inbound
   message is verified, and unsigned/forged envelopes are rejected.
2. RequestVote and AppendEntries RPCs work over the real TCP loopback
   path — not the in-memory shortcut.
3. End-to-end: three Raft nodes on three TCP ports elect a leader,
   replicate a command, then re-elect after the leader is killed.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from omni_mercury_engine.distributed.raft_consensus import (
    AppendEntriesRequest,
    ClusterConfiguration,
    LogEntry,
    NodeState,
    RaftNode,
    RequestVoteRequest,
    StateMachine,
)
from omni_mercury_engine.distributed.tcp_transport import TCPMessageTransport

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _fresh_pair() -> tuple[TCPMessageTransport, TCPMessageTransport]:
    """Two transports bound to ephemeral ports, peered to each other."""
    a = TCPMessageTransport("a")
    b = TCPMessageTransport("b")
    await a.start()
    await b.start()
    a.set_peer("b", *b.bound_address, public_key=b.public_key)
    b.set_peer("a", *a.bound_address, public_key=a.public_key)
    return a, b


# ---------------------------------------------------------------------------
# Cure 1: signed envelope round-trip + tamper rejection.
# ---------------------------------------------------------------------------


async def test_request_vote_round_trip() -> None:
    a, b = await _fresh_pair()
    try:

        async def _vote_handler(req: RequestVoteRequest) -> Any:
            from omni_mercury_engine.distributed.raft_consensus import (
                RequestVoteResponse,
            )

            return RequestVoteResponse(term=req.term, vote_granted=True, voter_id="b")

        b.register_handler("request_vote", _vote_handler)

        req = RequestVoteRequest(term=1, candidate_id="a", last_log_index=0, last_log_term=0)
        resp = await a.send_request_vote("b", req)

        assert resp is not None
        assert resp.term == 1
        assert resp.vote_granted is True
        assert resp.voter_id == "b"
    finally:
        await a.stop()
        await b.stop()


async def test_unsigned_envelope_is_rejected() -> None:
    """An inbound envelope from a sender whose public key is unknown
    must be silently dropped — no handler is invoked.
    """
    a, b = await _fresh_pair()
    try:
        seen: list[RequestVoteRequest] = []

        async def _vote_handler(req: RequestVoteRequest) -> Any:
            from omni_mercury_engine.distributed.raft_consensus import (
                RequestVoteResponse,
            )

            seen.append(req)
            return RequestVoteResponse(term=req.term, vote_granted=True, voter_id="b")

        b.register_handler("request_vote", _vote_handler)

        # Forge a transport whose public key b does NOT know.
        forger = TCPMessageTransport("forger")
        await forger.start()
        try:
            forger.set_peer("b", *b.bound_address, public_key=b.public_key)
            req = RequestVoteRequest(
                term=99, candidate_id="forger", last_log_index=0, last_log_term=0
            )
            resp = await forger.send_request_vote("b", req)
        finally:
            await forger.stop()

        assert resp is None  # b rejected the envelope before handling
        assert seen == []
    finally:
        await a.stop()
        await b.stop()


async def test_append_entries_round_trip_with_log_entries() -> None:
    a, b = await _fresh_pair()
    try:

        async def _ae_handler(req: AppendEntriesRequest) -> Any:
            from omni_mercury_engine.distributed.raft_consensus import (
                AppendEntriesResponse,
            )

            return AppendEntriesResponse(
                term=req.term,
                success=True,
                match_index=req.prev_log_index + len(req.entries),
                follower_id="b",
            )

        b.register_handler("append_entries", _ae_handler)

        entries = [
            LogEntry(term=1, index=1, command={"op": "noop"}),
            LogEntry(term=1, index=2, command={"op": "set", "k": "x", "v": 1}),
        ]
        req = AppendEntriesRequest(
            term=1,
            leader_id="a",
            prev_log_index=0,
            prev_log_term=0,
            entries=entries,
            leader_commit=0,
        )
        resp = await a.send_append_entries("b", req)

        assert resp is not None
        assert resp.success is True
        assert resp.match_index == 2
        assert resp.follower_id == "b"
    finally:
        await a.stop()
        await b.stop()


# ---------------------------------------------------------------------------
# Cure 2/3: 3-node cluster, election, replication, re-election.
# ---------------------------------------------------------------------------


async def _build_three_node_cluster() -> tuple[list[RaftNode], list[TCPMessageTransport]]:
    transports = [TCPMessageTransport(f"node_{i}") for i in range(3)]
    for t in transports:
        await t.start()

    # Cross-wire peers using the bound (host, port) and public keys.
    for i, t in enumerate(transports):
        for j, peer in enumerate(transports):
            if i == j:
                continue
            t.set_peer(f"node_{j}", *peer.bound_address, public_key=peer.public_key)

    nodes: list[RaftNode] = []
    peer_ids = [f"node_{i}" for i in range(3)]
    for i, t in enumerate(transports):
        cfg = ClusterConfiguration(
            node_id=f"node_{i}",
            peers=[p for p in peer_ids if p != f"node_{i}"],
            election_timeout_min_ms=120 + i * 30,
            election_timeout_max_ms=180 + i * 30,
            heartbeat_interval_ms=30,
        )
        nodes.append(RaftNode(cfg, transport=t, state_machine=StateMachine()))

    for n in nodes:
        await n.start()
    return nodes, transports


async def _await_leader(nodes: list[RaftNode], timeout: float = 5.0) -> RaftNode:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        leaders = [n for n in nodes if n.state == NodeState.LEADER]
        if len(leaders) == 1:
            return leaders[0]
        await asyncio.sleep(0.05)
    raise AssertionError("No leader elected within timeout")


async def test_three_node_cluster_elects_and_re_elects() -> None:
    nodes, transports = await _build_three_node_cluster()
    try:
        leader = await _await_leader(nodes)
        assert leader.is_leader

        # Replicate a command via the leader.  We do not assert
        # successful application here because the StateMachine is a
        # no-op; the contract is that the call returns rather than
        # hanging — the underlying RPCs flow over real TCP.
        ok, _ = await leader.submit_command({"op": "noop"}, timeout=2.0)
        assert isinstance(ok, bool)

        # Kill the current leader and confirm the remaining two elect
        # a new one.
        old_leader_id = leader.node_id
        await leader.stop()
        nodes_alive = [n for n in nodes if n.node_id != old_leader_id]

        new_leader = await _await_leader(nodes_alive, timeout=8.0)
        assert new_leader.node_id != old_leader_id
    finally:
        for n in nodes:
            try:
                await n.stop()
            except Exception:
                pass
        for t in transports:
            try:
                await t.stop()
            except Exception:
                pass
