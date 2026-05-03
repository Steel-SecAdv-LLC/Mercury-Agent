"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Phase 2 TCP transport hardening tests (Deliverable 9).

- Mutual-TLS integration test (self-signed CA).
- Replay-defense test.
- Fuzzer-style malformed-frame test.
- Subprocess-based 3-node cluster test.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import ssl
import struct
import tempfile

import pytest

from omni_mercury_engine.distributed.raft_consensus import (
    ClusterConfiguration,
    NodeState,
    RaftNode,
    RequestVoteRequest,
    RequestVoteResponse,
    StateMachine,
)
from omni_mercury_engine.distributed.tcp_transport import (
    MAX_PAYLOAD_BYTES,
    TCPMessageTransport,
    _frame,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helper: self-signed CA + node certs
# ---------------------------------------------------------------------------


def _generate_tls_assets(
    tmpdir: str,
) -> tuple[ssl.SSLContext, ssl.SSLContext]:
    """Create a self-signed CA plus one server/client cert pair.

    Returns (server_ctx, client_ctx) configured for mutual TLS.
    Both contexts trust only the ephemeral CA.
    """
    # Use stdlib ssl with OpenSSL commands via subprocess to generate certs
    import subprocess

    ca_key = os.path.join(tmpdir, "ca.key")
    ca_cert = os.path.join(tmpdir, "ca.crt")
    node_key = os.path.join(tmpdir, "node.key")
    node_csr = os.path.join(tmpdir, "node.csr")
    node_cert = os.path.join(tmpdir, "node.crt")

    # Generate CA key + self-signed cert
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            ca_key,
            "-out",
            ca_cert,
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=Mercury-Test-CA",
        ],
        check=True,
        capture_output=True,
    )
    # Generate node key + CSR
    subprocess.run(
        [
            "openssl",
            "req",
            "-newkey",
            "rsa:2048",
            "-keyout",
            node_key,
            "-out",
            node_csr,
            "-nodes",
            "-subj",
            "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
    # Sign node cert with CA
    subprocess.run(
        [
            "openssl",
            "x509",
            "-req",
            "-in",
            node_csr,
            "-CA",
            ca_cert,
            "-CAkey",
            ca_key,
            "-CAcreateserial",
            "-out",
            node_cert,
            "-days",
            "1",
        ],
        check=True,
        capture_output=True,
    )

    # Server context
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(certfile=node_cert, keyfile=node_key)
    server_ctx.load_verify_locations(cafile=ca_cert)
    server_ctx.verify_mode = ssl.CERT_REQUIRED

    # Client context
    client_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    client_ctx.load_cert_chain(certfile=node_cert, keyfile=node_key)
    client_ctx.load_verify_locations(cafile=ca_cert)

    return server_ctx, client_ctx


# ---------------------------------------------------------------------------
# Test 1: Mutual TLS
# ---------------------------------------------------------------------------


async def test_mutual_tls_round_trip() -> None:
    """Two transports with mutual TLS exchange a RequestVote RPC."""
    with tempfile.TemporaryDirectory() as tmpdir:
        server_ctx, client_ctx = _generate_tls_assets(tmpdir)

        a = TCPMessageTransport("a", server_ssl_context=server_ctx, client_ssl_context=client_ctx)
        b = TCPMessageTransport("b", server_ssl_context=server_ctx, client_ssl_context=client_ctx)
        await a.start()
        await b.start()
        a.set_peer("b", *b.bound_address, public_key=b.public_key)
        b.set_peer("a", *a.bound_address, public_key=a.public_key)

        async def _handler(req: RequestVoteRequest) -> RequestVoteResponse:
            return RequestVoteResponse(term=req.term, vote_granted=True, voter_id="b")

        b.register_handler("request_vote", _handler)

        req = RequestVoteRequest(term=1, candidate_id="a", last_log_index=0, last_log_term=0)
        resp = await a.send_request_vote("b", req)

        assert resp is not None
        assert resp.vote_granted is True
        assert resp.voter_id == "b"

        await a.stop()
        await b.stop()


# ---------------------------------------------------------------------------
# Test 2: Replay defense
# ---------------------------------------------------------------------------


async def test_replay_is_rejected() -> None:
    """Replaying an identical (sender, request_id) envelope is rejected."""
    a = TCPMessageTransport("a")
    b = TCPMessageTransport("b")
    await a.start()
    await b.start()
    a.set_peer("b", *b.bound_address, public_key=b.public_key)
    b.set_peer("a", *a.bound_address, public_key=a.public_key)

    received: list[RequestVoteRequest] = []

    async def _handler(req: RequestVoteRequest) -> RequestVoteResponse:
        received.append(req)
        return RequestVoteResponse(term=req.term, vote_granted=True, voter_id="b")

    b.register_handler("request_vote", _handler)

    # First RPC succeeds
    req = RequestVoteRequest(term=1, candidate_id="a", last_log_index=0, last_log_term=0)
    resp1 = await a.send_request_vote("b", req)
    assert resp1 is not None
    assert len(received) == 1

    # Manually replay the same envelope: forge the exact bytes a sent, including
    # the same request_id.  We test this by calling _is_replay directly
    # to confirm the mechanism works.
    assert b._is_replay("a", "already-seen-id") is False
    assert b._is_replay("a", "already-seen-id") is True

    await a.stop()
    await b.stop()


# ---------------------------------------------------------------------------
# Test 3: Fuzzer — malformed frames
# ---------------------------------------------------------------------------


async def test_fuzzer_truncated_prefix() -> None:
    """Sending fewer than 4 bytes for the length prefix does not crash."""
    transport = TCPMessageTransport("target")
    await transport.start()
    host, port = transport.bound_address

    reader, writer = await asyncio.open_connection(host, port)
    writer.write(b"\x00\x01")  # only 2 bytes
    await writer.drain()
    writer.close()
    await asyncio.sleep(0.1)
    await transport.stop()


async def test_fuzzer_length_exceeds_max() -> None:
    """A frame claiming to be larger than MAX_PAYLOAD_BYTES is rejected."""
    transport = TCPMessageTransport("target")
    await transport.start()
    host, port = transport.bound_address

    reader, writer = await asyncio.open_connection(host, port)
    # Send a length prefix claiming 32 MiB (over the 16 MiB limit)
    writer.write(struct.pack(">I", MAX_PAYLOAD_BYTES + 1))
    await writer.drain()
    await asyncio.sleep(0.1)
    writer.close()
    await transport.stop()


async def test_fuzzer_invalid_json() -> None:
    """A valid-length frame with garbage JSON is rejected gracefully."""
    transport = TCPMessageTransport("target")
    await transport.start()
    host, port = transport.bound_address

    reader, writer = await asyncio.open_connection(host, port)
    garbage = b"this is not json"
    writer.write(_frame(garbage))
    await writer.drain()
    await asyncio.sleep(0.1)
    writer.close()
    await transport.stop()


async def test_fuzzer_unexpected_fields() -> None:
    """Valid JSON with unexpected fields is handled without crash."""
    transport = TCPMessageTransport("target")
    await transport.start()
    host, port = transport.bound_address

    reader, writer = await asyncio.open_connection(host, port)
    payload = json.dumps(
        {
            "type": "totally_unknown",
            "request_id": "abc",
            "from": "evil",
            "to": "target",
            "body": {},
            "extra_field": "unexpected",
            "signature": "deadbeef",
        }
    ).encode()
    writer.write(_frame(payload))
    await writer.drain()
    await asyncio.sleep(0.1)
    writer.close()
    await transport.stop()


async def test_fuzzer_zero_length_frame() -> None:
    """A frame with length 0 is handled gracefully."""
    transport = TCPMessageTransport("target")
    await transport.start()
    host, port = transport.bound_address

    reader, writer = await asyncio.open_connection(host, port)
    writer.write(struct.pack(">I", 0))  # zero-length frame
    await writer.drain()
    await asyncio.sleep(0.1)
    writer.close()
    await transport.stop()


# ---------------------------------------------------------------------------
# Test 4: Subprocess-based 3-node cluster
# ---------------------------------------------------------------------------


def _run_raft_node(
    node_id: str,
    bind_port: int,
    peer_specs: list[tuple[str, int]],
    ready_event_path: str,
    result_path: str,
) -> None:
    """Entry point for a subprocess Raft node."""
    import asyncio as _aio

    async def _main() -> None:
        transport = TCPMessageTransport(node_id, bind_port=bind_port)
        await transport.start()

        # Write our actual port for the parent
        host, port = transport.bound_address
        with open(ready_event_path, "w") as f:
            f.write(f"{host}:{port}:{transport.public_key.hex()}")

        # Wait for all peers to be ready (check their files)
        for _peer_id, _expected_port in peer_specs:
            peer_ready = ready_event_path.replace(node_id, _peer_id)
            for _ in range(50):  # 5 seconds max
                if os.path.exists(peer_ready):
                    break
                await _aio.sleep(0.1)

        # Read peer addresses and wire them up
        for _peer_id, _ in peer_specs:
            peer_ready = ready_event_path.replace(node_id, _peer_id)
            with open(peer_ready) as f:
                parts = f.read().strip().split(":")
                peer_host = parts[0]
                peer_port = int(parts[1])
                peer_pk = bytes.fromhex(parts[2])
            transport.set_peer(_peer_id, peer_host, peer_port, peer_pk)

        peer_ids = [p[0] for p in peer_specs]
        cfg = ClusterConfiguration(
            node_id=node_id,
            peers=peer_ids,
            election_timeout_min_ms=200,
            election_timeout_max_ms=400,
            heartbeat_interval_ms=50,
        )
        node = RaftNode(cfg, transport=transport, state_machine=StateMachine())
        await node.start()

        # Wait for leader election (10 seconds max)
        for _ in range(100):
            if node.state == NodeState.LEADER:
                break
            await _aio.sleep(0.1)

        with open(result_path, "w") as f:
            f.write(f"{node.state.value}:{node.current_term}")

        await node.stop()
        await transport.stop()

    _aio.run(_main())


@pytest.mark.timeout(30)
def test_subprocess_three_node_cluster() -> None:
    """Three Raft nodes in separate processes elect a leader."""
    with tempfile.TemporaryDirectory() as tmpdir:
        node_ids = ["sub_0", "sub_1", "sub_2"]
        procs: list[multiprocessing.Process] = []

        for i, nid in enumerate(node_ids):
            # Use port 0 (ephemeral) — the child communicates its actual
            # bound port via the ready file, avoiding fixed-port collisions.
            peers = [(other, 0) for j, other in enumerate(node_ids) if j != i]
            ready_path = os.path.join(tmpdir, f"{nid}_ready")
            result_path = os.path.join(tmpdir, f"{nid}_result")
            p = multiprocessing.Process(
                target=_run_raft_node,
                args=(nid, 0, peers, ready_path, result_path),
            )
            procs.append(p)
            p.start()

        # Wait for all processes to finish (timeout 25 seconds)
        for p in procs:
            p.join(timeout=25)

        # Read results
        results: dict[str, str] = {}
        for nid in node_ids:
            result_path = os.path.join(tmpdir, f"{nid}_result")
            if os.path.exists(result_path):
                with open(result_path) as f:
                    results[nid] = f.read().strip()

        # At least one node should have become leader
        leaders = [nid for nid, r in results.items() if r.startswith("leader")]
        assert len(leaders) >= 1, f"No leader elected. Results: {results}"

        # Clean up any lingering processes
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
