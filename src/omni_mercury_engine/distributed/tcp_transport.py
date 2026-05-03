"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Native TCP MessageTransport for Raft.

Pure-stdlib implementation:
- ``asyncio.start_server`` listens on a per-node TCP port.
- Length-prefixed binary frames: 4 bytes big-endian length, then
  ``payload``.  Each ``payload`` is JSON-encoded UTF-8 bytes.
- Per-message authentication via Mercury's own AMA Cryptography
  signing surface (``Ed25519Provider`` from
  :mod:`omni_mercury_engine.security.crypto_api`) — every wire message
  carries a signature over its envelope.  No third-party RPC framework,
  no protobuf, no msgpack, no zeromq.

The wire format is Mercury's own, owned end-to-end:

    4 bytes BE length  |  payload (JSON bytes)
    ---------------------------------------------
    payload = {
        "type":        "request_vote" | "append_entries" |
                       "request_vote_response" | "append_entries_response",
        "request_id":  uuid4 hex,
        "from":        sender_node_id,
        "to":          recipient_node_id,
        "body":        <message-specific JSON object>,
        "signature":   hex-encoded Ed25519 signature over
                       JSON-serialised {type, request_id, from, to, body}
    }

Rationale: the spec required *no* gRPC, *no* protobuf, *no* third-party
RPC framework.  JSON over length-prefixed TCP keeps the wire format
inspectable and the codepath auditable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import struct
from dataclasses import asdict
from typing import Any

from omni_mercury_engine.distributed.raft_consensus import (
    AppendEntriesRequest,
    AppendEntriesResponse,
    LogEntry,
    MessageTransport,
    RequestVoteRequest,
    RequestVoteResponse,
)
from omni_mercury_engine.security.crypto_api import Ed25519Provider, KeyPair

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wire-format constants
# ---------------------------------------------------------------------------

#: Length-prefix is a 4-byte unsigned big-endian integer, so the largest
#: payload representable on the wire is ~4 GiB.  We hard-cap inbound
#: payloads at ``MAX_PAYLOAD_BYTES`` to bound memory; values above this
#: are dropped without parsing.
_LEN_PREFIX_FMT = ">I"
_LEN_PREFIX_SIZE = struct.calcsize(_LEN_PREFIX_FMT)
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024  # 16 MiB

#: Connection-level timeouts (seconds).
CONNECT_TIMEOUT = 2.0
RPC_TIMEOUT = 2.0


def _frame(payload: bytes) -> bytes:
    """Wrap a JSON payload in the length-prefix wire frame."""
    return struct.pack(_LEN_PREFIX_FMT, len(payload)) + payload


async def _read_frame(reader: asyncio.StreamReader) -> bytes | None:
    """Read one length-prefixed frame.  Returns ``None`` on clean EOF."""
    if reader.at_eof():
        return None
    try:
        header = await reader.readexactly(_LEN_PREFIX_SIZE)
    except asyncio.IncompleteReadError:
        return None
    (length,) = struct.unpack(_LEN_PREFIX_FMT, header)
    if length == 0:
        return b""
    if length > MAX_PAYLOAD_BYTES:
        raise ValueError(f"Inbound frame too large: {length} > {MAX_PAYLOAD_BYTES}")
    try:
        return await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        return None


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _serialize_request_vote(req: RequestVoteRequest) -> dict[str, Any]:
    return asdict(req)


def _deserialize_request_vote(body: dict[str, Any]) -> RequestVoteRequest:
    return RequestVoteRequest(
        term=int(body["term"]),
        candidate_id=str(body["candidate_id"]),
        last_log_index=int(body["last_log_index"]),
        last_log_term=int(body["last_log_term"]),
    )


def _serialize_append_entries(req: AppendEntriesRequest) -> dict[str, Any]:
    return {
        "term": req.term,
        "leader_id": req.leader_id,
        "prev_log_index": req.prev_log_index,
        "prev_log_term": req.prev_log_term,
        "entries": [e.to_dict() for e in req.entries],
        "leader_commit": req.leader_commit,
    }


def _deserialize_append_entries(body: dict[str, Any]) -> AppendEntriesRequest:
    return AppendEntriesRequest(
        term=int(body["term"]),
        leader_id=str(body["leader_id"]),
        prev_log_index=int(body["prev_log_index"]),
        prev_log_term=int(body["prev_log_term"]),
        entries=[LogEntry.from_dict(e) for e in body["entries"]],
        leader_commit=int(body["leader_commit"]),
    )


def _serialize_request_vote_response(resp: RequestVoteResponse) -> dict[str, Any]:
    return asdict(resp)


def _deserialize_request_vote_response(body: dict[str, Any]) -> RequestVoteResponse:
    return RequestVoteResponse(
        term=int(body["term"]),
        vote_granted=bool(body["vote_granted"]),
        voter_id=str(body["voter_id"]),
    )


def _serialize_append_entries_response(resp: AppendEntriesResponse) -> dict[str, Any]:
    return asdict(resp)


def _deserialize_append_entries_response(body: dict[str, Any]) -> AppendEntriesResponse:
    return AppendEntriesResponse(
        term=int(body["term"]),
        success=bool(body["success"]),
        match_index=int(body["match_index"]),
        follower_id=str(body["follower_id"]),
    )


# ---------------------------------------------------------------------------
# Envelope signing / verification
# ---------------------------------------------------------------------------


def _envelope_bytes(envelope: dict[str, Any]) -> bytes:
    """Canonical bytes for signing/verifying.

    Excludes ``signature`` so the field can be set after canonicalisation
    without affecting the digest.
    """
    canonical = {k: envelope[k] for k in ("type", "request_id", "from", "to", "body")}
    return json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# TCPMessageTransport
# ---------------------------------------------------------------------------


class TCPMessageTransport(MessageTransport):
    """Native pure-stdlib TCP transport for Raft consensus.

    The transport is symmetric — every node both serves a TCP listener
    and connects out to its peers on demand.

    Args:
        node_id: Identifier for this node (matches Raft ``node_id``).
        bind_host: Listener bind host (default ``"127.0.0.1"``).
        bind_port: Listener bind port; ``0`` requests an ephemeral port.
        peers: Mapping of ``peer_id -> (host, port)``.  Resolved at
            ``send_*`` time, so a peer that joins the cluster after
            transport ``start()`` can be added via :meth:`set_peer`.
        keypair: Optional pre-generated Ed25519 ``KeyPair`` for
            signing.  When omitted, a fresh keypair is generated at
            ``__init__`` — production deployments should provide a
            persistent keypair so peer public keys can be pre-shared.
        peer_public_keys: Mapping of ``peer_id -> public_key_bytes``
            used to verify inbound signatures.  An unknown sender is
            rejected; an invalid signature is rejected.
    """

    def __init__(
        self,
        node_id: str,
        *,
        bind_host: str = "127.0.0.1",
        bind_port: int = 0,
        peers: dict[str, tuple[str, int]] | None = None,
        keypair: KeyPair | None = None,
        peer_public_keys: dict[str, bytes] | None = None,
    ) -> None:
        super().__init__()
        self._node_id = node_id
        self._bind_host = bind_host
        self._bind_port = bind_port
        self._peers: dict[str, tuple[str, int]] = dict(peers or {})
        self._peer_public_keys: dict[str, bytes] = dict(peer_public_keys or {})

        self._signer = Ed25519Provider()
        self._keypair = keypair or self._signer.generate_keypair()

        self._server: asyncio.base_events.Server | None = None
        # Outstanding RPCs awaiting their response — keyed by request_id.
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._running = False

    # -- introspection --------------------------------------------------

    @property
    def public_key(self) -> bytes:
        """The node's Ed25519 public key (share with peers)."""
        return self._keypair.public_key

    @property
    def bound_address(self) -> tuple[str, int]:
        """Concrete (host, port) the listener bound to (after start())."""
        if self._server is None:
            raise RuntimeError("Transport not started")
        sockets = self._server.sockets or []
        if not sockets:
            raise RuntimeError("Transport has no bound socket")
        host, port = sockets[0].getsockname()[:2]
        return host, port

    def set_peer(self, peer_id: str, host: str, port: int, public_key: bytes) -> None:
        """Add or update a peer's network address and public key."""
        self._peers[peer_id] = (host, port)
        self._peer_public_keys[peer_id] = public_key

    # -- lifecycle ------------------------------------------------------

    async def start(self) -> None:
        """Start the TCP listener."""
        if self._running:
            return
        self._server = await asyncio.start_server(
            self._handle_connection, host=self._bind_host, port=self._bind_port
        )
        self._running = True
        logger.info(
            "TCPMessageTransport[%s] listening on %s",
            self._node_id,
            self.bound_address,
        )

    async def stop(self) -> None:
        """Stop the TCP listener and cancel any pending RPCs."""
        if not self._running:
            return
        self._running = False
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

    # -- outbound RPCs --------------------------------------------------

    async def send_request_vote(
        self,
        peer_id: str,
        request: RequestVoteRequest,
    ) -> RequestVoteResponse | None:
        body = _serialize_request_vote(request)
        resp = await self._send_and_wait(peer_id, "request_vote", body)
        if resp is None:
            return None
        return _deserialize_request_vote_response(resp["body"])

    async def send_append_entries(
        self,
        peer_id: str,
        request: AppendEntriesRequest,
    ) -> AppendEntriesResponse | None:
        body = _serialize_append_entries(request)
        resp = await self._send_and_wait(peer_id, "append_entries", body)
        if resp is None:
            return None
        return _deserialize_append_entries_response(resp["body"])

    async def _send_and_wait(
        self,
        peer_id: str,
        msg_type: str,
        body: dict[str, Any],
    ) -> dict[str, Any] | None:
        addr = self._peers.get(peer_id)
        if addr is None:
            logger.debug("No address for peer %s", peer_id)
            return None
        request_id = secrets.token_hex(8)
        envelope = {
            "type": msg_type,
            "request_id": request_id,
            "from": self._node_id,
            "to": peer_id,
            "body": body,
        }
        signature = self._signer.sign(_envelope_bytes(envelope), self._keypair.secret_key)
        envelope["signature"] = signature.hex()

        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future

        try:
            await asyncio.wait_for(self._send(addr, envelope), timeout=CONNECT_TIMEOUT)
            return await asyncio.wait_for(future, timeout=RPC_TIMEOUT)
        except (TimeoutError, OSError, ConnectionError) as exc:
            logger.debug("RPC %s to %s failed: %s", msg_type, peer_id, exc)
            return None
        finally:
            self._pending.pop(request_id, None)

    async def _send(self, addr: tuple[str, int], envelope: dict[str, Any]) -> None:
        host, port = addr
        reader, writer = await asyncio.open_connection(host=host, port=port)
        try:
            payload = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
            writer.write(_frame(payload))
            await writer.drain()

            # Wait for the response frame on the same connection.
            resp_frame = await _read_frame(reader)
            if resp_frame is None:
                return
            resp_envelope = json.loads(resp_frame.decode("utf-8"))
            self._dispatch_response(resp_envelope)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    def _dispatch_response(self, envelope: dict[str, Any]) -> None:
        request_id = envelope.get("request_id")
        future = self._pending.get(request_id) if request_id else None
        if future is None or future.done():
            return
        if not self._verify_envelope(envelope):
            future.set_exception(ConnectionError("response signature failed verification"))
            return
        future.set_result(envelope)

    # -- inbound handling ----------------------------------------------

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            frame = await _read_frame(reader)
            if frame is None:
                return
            envelope = json.loads(frame.decode("utf-8"))
            if not self._verify_envelope(envelope):
                logger.warning(
                    "Rejected inbound frame on %s: invalid signature from %s",
                    self._node_id,
                    envelope.get("from"),
                )
                return

            response = await self._handle_envelope(envelope)
            if response is None:
                return
            payload = json.dumps(response, separators=(",", ":")).encode("utf-8")
            writer.write(_frame(payload))
            await writer.drain()
        except (asyncio.IncompleteReadError, json.JSONDecodeError, ValueError) as exc:
            logger.debug("Bad inbound frame on %s: %s", self._node_id, exc)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _handle_envelope(self, envelope: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch an inbound RPC to the registered handler and sign the
        response envelope before returning it on the same connection."""
        msg_type = envelope.get("type")
        if msg_type == "request_vote":
            handler = self._message_handlers.get("request_vote")
            if handler is None:
                return None
            req = _deserialize_request_vote(envelope["body"])
            resp: RequestVoteResponse = await handler(req)
            return self._make_response_envelope(envelope, "request_vote_response", asdict(resp))
        if msg_type == "append_entries":
            handler = self._message_handlers.get("append_entries")
            if handler is None:
                return None
            req = _deserialize_append_entries(envelope["body"])
            resp_ae: AppendEntriesResponse = await handler(req)
            return self._make_response_envelope(
                envelope, "append_entries_response", asdict(resp_ae)
            )
        logger.debug("Unknown inbound message type %s", msg_type)
        return None

    def _make_response_envelope(
        self,
        request_envelope: dict[str, Any],
        response_type: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        envelope = {
            "type": response_type,
            "request_id": request_envelope["request_id"],
            "from": self._node_id,
            "to": request_envelope["from"],
            "body": body,
        }
        signature = self._signer.sign(_envelope_bytes(envelope), self._keypair.secret_key)
        envelope["signature"] = signature.hex()
        return envelope

    # -- crypto ---------------------------------------------------------

    def _verify_envelope(self, envelope: dict[str, Any]) -> bool:
        sender = envelope.get("from")
        sig_hex = envelope.get("signature")
        if not isinstance(sender, str) or not isinstance(sig_hex, str):
            return False
        public_key = self._peer_public_keys.get(sender)
        if public_key is None:
            logger.debug("No public key for sender %s — rejecting envelope", sender)
            return False
        try:
            signature = bytes.fromhex(sig_hex)
        except ValueError:
            return False
        return self._signer.verify(_envelope_bytes(envelope), signature, public_key)
