# Copyright (C) 2025 Steel Security Advisors LLC
"""Raft Consensus Protocol Implementation for Mercury Agent Distributed Processing.

This module implements the Raft consensus algorithm for leader election and
log replication across a distributed Mercury Agent cluster.

References:
- Ongaro & Ousterhout (2014): In Search of an Understandable Consensus Algorithm
- https://raft.github.io/raft.pdf
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

T = TypeVar("T")


class NodeState(Enum):
    """Raft node state."""

    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


@dataclass
class LogEntry:
    """A single entry in the Raft log."""

    term: int
    index: int
    command: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "term": self.term,
            "index": self.index,
            "command": self.command,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogEntry:
        """Create from dictionary."""
        return cls(
            term=data["term"],
            index=data["index"],
            command=data["command"],
            timestamp=data.get("timestamp", time.time()),
        )

    def checksum(self) -> str:
        """Compute checksum for integrity verification."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha3_256(content.encode()).hexdigest()[:16]


@dataclass
class RequestVoteRequest:
    """Request for vote in leader election."""

    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int


@dataclass
class RequestVoteResponse:
    """Response to vote request."""

    term: int
    vote_granted: bool
    voter_id: str


@dataclass
class AppendEntriesRequest:
    """Request to append entries (also serves as heartbeat)."""

    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: list[LogEntry]
    leader_commit: int


@dataclass
class AppendEntriesResponse:
    """Response to append entries request."""

    term: int
    success: bool
    match_index: int
    follower_id: str


@dataclass
class ClusterConfiguration:
    """Cluster configuration for Raft."""

    node_id: str
    peers: list[str]
    election_timeout_min_ms: int = 150
    election_timeout_max_ms: int = 300
    heartbeat_interval_ms: int = 50
    max_entries_per_append: int = 100
    snapshot_threshold: int = 1000


class RaftLog:
    """Persistent log storage for Raft consensus.

    Manages log entries with support for compaction and snapshots.
    """

    def __init__(self, snapshot_threshold: int = 1000) -> None:
        """Initialize the Raft log."""
        self._entries: list[LogEntry] = []
        self._snapshot_index: int = 0
        self._snapshot_term: int = 0
        self._snapshot_data: dict[str, Any] | None = None
        self._snapshot_threshold = snapshot_threshold
        self._lock = asyncio.Lock()

    @property
    def last_index(self) -> int:
        """Get the index of the last log entry."""
        if self._entries:
            return self._entries[-1].index
        return self._snapshot_index

    @property
    def last_term(self) -> int:
        """Get the term of the last log entry."""
        if self._entries:
            return self._entries[-1].term
        return self._snapshot_term

    async def append(self, entry: LogEntry) -> None:
        """Append an entry to the log."""

        async with self._lock:
            self._entries.append(entry)

            if len(self._entries) >= self._snapshot_threshold:
                await self._compact()

    async def append_entries(
        self,
        prev_index: int,
        prev_term: int,
        entries: list[LogEntry],
    ) -> bool:
        """Append entries from leader, checking consistency.

        Returns True if entries were successfully appended.
        """

        async with self._lock:
            if prev_index > 0:
                if prev_index > self.last_index:
                    return False

                prev_entry = self._get_entry_at_index(prev_index)
                if prev_entry is None or prev_entry.term != prev_term:
                    return False

            for entry in entries:
                existing = self._get_entry_at_index(entry.index)
                if existing is not None:
                    if existing.term != entry.term:
                        self._truncate_from(entry.index)
                        self._entries.append(entry)
                else:
                    self._entries.append(entry)

            return True

    def _get_entry_at_index(self, index: int) -> LogEntry | None:
        """Get entry at specific index."""
        if index <= self._snapshot_index:
            return None

        for entry in self._entries:
            if entry.index == index:
                return entry
        return None

    def _truncate_from(self, index: int) -> None:
        """Remove all entries from index onwards."""
        self._entries = [e for e in self._entries if e.index < index]

    async def _compact(self) -> None:
        """Compact log by creating snapshot."""
        if len(self._entries) < self._snapshot_threshold:
            return

        compact_index = len(self._entries) // 2
        compact_entry = self._entries[compact_index - 1]

        self._snapshot_index = compact_entry.index
        self._snapshot_term = compact_entry.term
        self._entries = self._entries[compact_index:]

        logger.info(
            "Log compacted: snapshot_index=%d, remaining_entries=%d",
            self._snapshot_index,
            len(self._entries),
        )

    def get_entries_from(self, start_index: int) -> list[LogEntry]:
        """Get all entries starting from index."""
        return [e for e in self._entries if e.index >= start_index]

    def get_term_at_index(self, index: int) -> int:
        """Get term at specific index."""
        if index == 0:
            return 0
        if index == self._snapshot_index:
            return self._snapshot_term

        entry = self._get_entry_at_index(index)
        return entry.term if entry else 0


class StateMachine(Generic[T]):
    """State machine that applies committed log entries.

    This is the application-specific state that Raft replicates.
    """

    def __init__(self) -> None:
        """Initialize the state machine."""
        self._state: dict[str, Any] = {}
        self._last_applied: int = 0
        self._handlers: dict[str, Callable[[dict[str, Any]], T]] = {}
        self._lock = asyncio.Lock()

    def register_handler(
        self,
        command_type: str,
        handler: Callable[[dict[str, Any]], T],
    ) -> None:
        """Register a command handler."""
        self._handlers[command_type] = handler

    async def apply(self, entry: LogEntry) -> T | None:
        """Apply a log entry to the state machine."""

        async with self._lock:
            if entry.index <= self._last_applied:
                return None

            command = entry.command
            command_type = command.get("type", "")
            handler = self._handlers.get(command_type)

            result = None
            if handler:
                try:
                    result = handler(command)
                except Exception as e:
                    logger.error("Error applying command %s: %s", command_type, e)

            self._last_applied = entry.index
            return result

    @property
    def last_applied(self) -> int:
        """Get the index of the last applied entry."""
        return self._last_applied

    def get_state(self) -> dict[str, Any]:
        """Get current state snapshot."""
        return self._state.copy()


class MessageTransport:
    """Abstract message transport for Raft communication.

    Implementations should handle actual network communication.
    """

    def __init__(self) -> None:
        """Initialize the transport."""
        self._message_handlers: dict[str, Callable[..., Any]] = {}
        self._connected_peers: set[str] = set()

    async def send_request_vote(
        self,
        peer_id: str,
        request: RequestVoteRequest,
    ) -> RequestVoteResponse | None:
        """Send vote request to peer."""
        raise NotImplementedError

    async def send_append_entries(
        self,
        peer_id: str,
        request: AppendEntriesRequest,
    ) -> AppendEntriesResponse | None:
        """Send append entries to peer."""
        raise NotImplementedError

    def register_handler(self, message_type: str, handler: Callable[..., Any]) -> None:
        """Register message handler."""
        self._message_handlers[message_type] = handler

    async def start(self) -> None:
        """Start the transport."""
        raise NotImplementedError

    async def stop(self) -> None:
        """Stop the transport."""
        raise NotImplementedError


class InMemoryTransport(MessageTransport):
    """In-memory transport for testing and single-process clusters.

    Routes messages directly between RaftNode instances.
    """

    _instances: dict[str, RaftNode] = {}

    def __init__(self, node_id: str) -> None:
        """Initialize in-memory transport."""
        super().__init__()
        self._node_id = node_id
        self._running = False

    @classmethod
    def register_node(cls, node_id: str, node: RaftNode) -> None:
        """Register a node for message routing."""
        cls._instances[node_id] = node

    @classmethod
    def unregister_node(cls, node_id: str) -> None:
        """Unregister a node."""
        cls._instances.pop(node_id, None)

    async def send_request_vote(
        self,
        peer_id: str,
        request: RequestVoteRequest,
    ) -> RequestVoteResponse | None:
        """Send vote request to peer."""
        peer = self._instances.get(peer_id)
        if peer is None:
            return None

        try:
            return await peer.handle_request_vote(request)
        except Exception as e:
            logger.debug("Error sending vote request to %s: %s", peer_id, e)
            return None

    async def send_append_entries(
        self,
        peer_id: str,
        request: AppendEntriesRequest,
    ) -> AppendEntriesResponse | None:
        """Send append entries to peer."""
        peer = self._instances.get(peer_id)
        if peer is None:
            return None

        try:
            return await peer.handle_append_entries(request)
        except Exception as e:
            logger.debug("Error sending append entries to %s: %s", peer_id, e)
            return None

    async def start(self) -> None:
        """Start the transport."""
        self._running = True

    async def stop(self) -> None:
        """Stop the transport."""
        self._running = False


class RaftNode:
    """A single node in a Raft cluster.

    Implements the complete Raft consensus algorithm including:
    - Leader election
    - Log replication
    - Safety guarantees
    """

    def __init__(
        self,
        config: ClusterConfiguration,
        transport: MessageTransport,
        state_machine: StateMachine[Any] | None = None,
    ) -> None:
        """Initialize a Raft node."""
        self._config = config
        self._transport = transport
        self._state_machine: StateMachine[Any] = state_machine or StateMachine()
        self._log = RaftLog(config.snapshot_threshold)

        self._current_term: int = 0
        self._voted_for: str | None = None
        self._state: NodeState = NodeState.FOLLOWER
        self._leader_id: str | None = None

        self._commit_index: int = 0
        self._next_index: dict[str, int] = {}
        self._match_index: dict[str, int] = {}

        self._election_timer: asyncio.Task[None] | None = None
        self._heartbeat_timer: asyncio.Task[None] | None = None
        self._replication_task: asyncio.Task[None] | None = None
        self._running = False

        self._pending_commands: dict[int, asyncio.Future[Any]] = {}
        self._lock = asyncio.Lock()

        self._votes_received: set[str] = set()
        self._last_heartbeat = time.time()

    @property
    def node_id(self) -> str:
        """Get node ID."""
        return self._config.node_id

    @property
    def state(self) -> NodeState:
        """Get current node state."""
        return self._state

    @property
    def current_term(self) -> int:
        """Get current term."""
        return self._current_term

    @property
    def is_leader(self) -> bool:
        """Check if this node is the leader."""
        return self._state == NodeState.LEADER

    @property
    def leader_id(self) -> str | None:
        """Get current leader ID."""
        return self._leader_id

    async def start(self) -> None:
        """Start the Raft node."""
        self._running = True
        await self._transport.start()

        if isinstance(self._transport, InMemoryTransport):
            InMemoryTransport.register_node(self.node_id, self)
        else:
            # Network transports route inbound RPCs through registered
            # handlers — the in-memory transport bypasses the registry
            # because it owns direct ``RaftNode`` references, but every
            # other transport (TCP and beyond) needs the wiring here.
            self._transport.register_handler("request_vote", self.handle_request_vote)
            self._transport.register_handler("append_entries", self.handle_append_entries)

        self._reset_election_timer()
        logger.info("Raft node %s started", self.node_id)

    async def stop(self) -> None:
        """Stop the Raft node."""
        self._running = False

        if self._election_timer:
            self._election_timer.cancel()
        if self._heartbeat_timer:
            self._heartbeat_timer.cancel()

        await self._transport.stop()

        if isinstance(self._transport, InMemoryTransport):
            InMemoryTransport.unregister_node(self.node_id)

        logger.info("Raft node %s stopped", self.node_id)

    async def submit_command(
        self,
        command: dict[str, Any],
        timeout: float = 5.0,
    ) -> tuple[bool, Any]:
        """Submit a command to be replicated.

        Returns (success, result) tuple.
        """
        if not self.is_leader:
            return False, {"error": "not_leader", "leader_id": self._leader_id}

        async with self._lock:
            index = self._log.last_index + 1
            entry = LogEntry(
                term=self._current_term,
                index=index,
                command=command,
            )
            await self._log.append(entry)

            future: asyncio.Future[Any] = asyncio.Future()
            self._pending_commands[index] = future

        self._replication_task = asyncio.create_task(self._replicate_entries())

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return True, result
        except TimeoutError:
            self._pending_commands.pop(index, None)
            return False, {"error": "timeout"}

    async def handle_request_vote(
        self,
        request: RequestVoteRequest,
    ) -> RequestVoteResponse:
        """Handle incoming vote request."""

        async with self._lock:
            if request.term > self._current_term:
                self._current_term = request.term
                self._state = NodeState.FOLLOWER
                self._voted_for = None
                self._leader_id = None

            vote_granted = False

            if request.term >= self._current_term:
                log_ok = request.last_log_term > self._log.last_term or (
                    request.last_log_term == self._log.last_term
                    and request.last_log_index >= self._log.last_index
                )

                if log_ok and (self._voted_for is None or self._voted_for == request.candidate_id):
                    self._voted_for = request.candidate_id
                    vote_granted = True
                    self._reset_election_timer()

            return RequestVoteResponse(
                term=self._current_term,
                vote_granted=vote_granted,
                voter_id=self.node_id,
            )

    async def handle_append_entries(
        self,
        request: AppendEntriesRequest,
    ) -> AppendEntriesResponse:
        """Handle incoming append entries request."""

        async with self._lock:
            if request.term > self._current_term:
                self._current_term = request.term
                self._state = NodeState.FOLLOWER
                self._voted_for = None

            if request.term < self._current_term:
                return AppendEntriesResponse(
                    term=self._current_term,
                    success=False,
                    match_index=0,
                    follower_id=self.node_id,
                )

            self._leader_id = request.leader_id
            self._state = NodeState.FOLLOWER
            self._last_heartbeat = time.time()
            self._reset_election_timer()

            success = await self._log.append_entries(
                request.prev_log_index,
                request.prev_log_term,
                request.entries,
            )

            if success:
                if request.leader_commit > self._commit_index:
                    self._commit_index = min(
                        request.leader_commit,
                        self._log.last_index,
                    )
                    await self._apply_committed_entries()

            return AppendEntriesResponse(
                term=self._current_term,
                success=success,
                match_index=self._log.last_index if success else 0,
                follower_id=self.node_id,
            )

    def _reset_election_timer(self) -> None:
        """Reset the election timeout timer."""
        if self._election_timer:
            self._election_timer.cancel()

        if self._running and self._state != NodeState.LEADER:
            timeout = (
                random.randint(
                    self._config.election_timeout_min_ms,
                    self._config.election_timeout_max_ms,
                )
                / 1000.0
            )

            self._election_timer = asyncio.create_task(self._election_timeout(timeout))

    async def _election_timeout(self, timeout: float) -> None:
        """Handle election timeout."""
        await asyncio.sleep(timeout)

        if self._running and self._state != NodeState.LEADER:
            await self._start_election()

    async def _start_election(self) -> None:
        """Start a new election."""

        async with self._lock:
            self._current_term += 1
            self._state = NodeState.CANDIDATE
            self._voted_for = self.node_id
            self._votes_received = {self.node_id}
            self._leader_id = None

        logger.info(
            "Node %s starting election for term %d",
            self.node_id,
            self._current_term,
        )

        request = RequestVoteRequest(
            term=self._current_term,
            candidate_id=self.node_id,
            last_log_index=self._log.last_index,
            last_log_term=self._log.last_term,
        )

        tasks = [
            self._transport.send_request_vote(peer_id, request) for peer_id in self._config.peers
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for response in responses:
            if isinstance(response, RequestVoteResponse):
                await self._process_vote_response(response)

        self._reset_election_timer()

    async def _process_vote_response(self, response: RequestVoteResponse) -> None:
        """Process a vote response."""

        async with self._lock:
            if response.term > self._current_term:
                self._current_term = response.term
                self._state = NodeState.FOLLOWER
                self._voted_for = None
                return

            if (
                self._state == NodeState.CANDIDATE
                and response.term == self._current_term
                and response.vote_granted
            ):
                self._votes_received.add(response.voter_id)

                quorum = (len(self._config.peers) + 1) // 2 + 1
                if len(self._votes_received) >= quorum:
                    await self._become_leader()

    async def _become_leader(self) -> None:
        """Transition to leader state."""
        self._state = NodeState.LEADER
        self._leader_id = self.node_id

        for peer_id in self._config.peers:
            self._next_index[peer_id] = self._log.last_index + 1
            self._match_index[peer_id] = 0

        logger.info("Node %s became leader for term %d", self.node_id, self._current_term)

        if self._election_timer:
            self._election_timer.cancel()

        self._start_heartbeat_timer()

        await self._send_heartbeats()

    def _start_heartbeat_timer(self) -> None:
        """Start the heartbeat timer."""
        if self._heartbeat_timer:
            self._heartbeat_timer.cancel()

        if self._running and self._state == NodeState.LEADER:
            interval = self._config.heartbeat_interval_ms / 1000.0
            self._heartbeat_timer = asyncio.create_task(self._heartbeat_loop(interval))

    async def _heartbeat_loop(self, interval: float) -> None:
        """Send periodic heartbeats."""
        while self._running and self._state == NodeState.LEADER:
            await self._send_heartbeats()
            await asyncio.sleep(interval)

    async def _send_heartbeats(self) -> None:
        """Send heartbeats to all followers."""
        await self._replicate_entries()

    async def _replicate_entries(self) -> None:
        """Replicate log entries to all followers."""
        if self._state != NodeState.LEADER:
            return

        tasks = []
        for peer_id in self._config.peers:
            tasks.append(self._replicate_to_peer(peer_id))

        await asyncio.gather(*tasks, return_exceptions=True)

        await self._advance_commit_index()

    async def _replicate_to_peer(self, peer_id: str) -> None:
        """Replicate entries to a specific peer."""
        next_idx = self._next_index.get(peer_id, 1)
        prev_idx = next_idx - 1
        prev_term = self._log.get_term_at_index(prev_idx)

        entries = self._log.get_entries_from(next_idx)
        entries = entries[: self._config.max_entries_per_append]

        request = AppendEntriesRequest(
            term=self._current_term,
            leader_id=self.node_id,
            prev_log_index=prev_idx,
            prev_log_term=prev_term,
            entries=entries,
            leader_commit=self._commit_index,
        )

        response = await self._transport.send_append_entries(peer_id, request)

        if response is None:
            return

        async with self._lock:
            if response.term > self._current_term:
                self._current_term = response.term
                self._state = NodeState.FOLLOWER
                self._voted_for = None
                self._reset_election_timer()
                return

            if response.success:
                self._next_index[peer_id] = response.match_index + 1
                self._match_index[peer_id] = response.match_index
            else:
                self._next_index[peer_id] = max(1, self._next_index[peer_id] - 1)

    async def _advance_commit_index(self) -> None:
        """Advance commit index based on replication status."""
        if self._state != NodeState.LEADER:
            return

        async with self._lock:
            for n in range(self._commit_index + 1, self._log.last_index + 1):
                if self._log.get_term_at_index(n) != self._current_term:
                    continue

                replicated = 1
                for peer_id in self._config.peers:
                    if self._match_index.get(peer_id, 0) >= n:
                        replicated += 1

                quorum = (len(self._config.peers) + 1) // 2 + 1
                if replicated >= quorum:
                    self._commit_index = n

            await self._apply_committed_entries()

    async def _apply_committed_entries(self) -> None:
        """Apply committed but not yet applied entries."""
        while self._state_machine.last_applied < self._commit_index:
            next_index = self._state_machine.last_applied + 1
            entries = self._log.get_entries_from(next_index)

            for entry in entries:
                if entry.index > self._commit_index:
                    break

                result = await self._state_machine.apply(entry)

                future = self._pending_commands.pop(entry.index, None)
                if future and not future.done():
                    future.set_result(result)


class RaftCluster:
    """Manages a cluster of Raft nodes.

    Provides high-level interface for distributed operations.
    """

    def __init__(
        self,
        node_configs: list[ClusterConfiguration],
        use_in_memory_transport: bool = True,
    ) -> None:
        """Initialize the Raft cluster."""
        self._nodes: dict[str, RaftNode] = {}
        self._use_in_memory = use_in_memory_transport

        for config in node_configs:
            transport: MessageTransport
            if use_in_memory_transport:
                transport = InMemoryTransport(config.node_id)
            else:
                # Native pure-stdlib TCP transport — see
                # ``omni_mercury_engine.distributed.tcp_transport``.
                # Imported lazily to avoid pulling crypto deps when only
                # the in-memory path is used.
                from omni_mercury_engine.distributed.tcp_transport import (
                    TCPMessageTransport,
                )

                transport = TCPMessageTransport(config.node_id)

            state_machine: StateMachine[Any] = StateMachine()
            node = RaftNode(config, transport, state_machine)
            self._nodes[config.node_id] = node

    async def start(self) -> None:
        """Start all nodes in the cluster."""
        for node in self._nodes.values():
            await node.start()

    async def stop(self) -> None:
        """Stop all nodes in the cluster."""
        for node in self._nodes.values():
            await node.stop()

    def get_leader(self) -> RaftNode | None:
        """Get the current leader node."""
        for node in self._nodes.values():
            if node.is_leader:
                return node
        return None

    def get_node(self, node_id: str) -> RaftNode | None:
        """Get a specific node."""
        return self._nodes.get(node_id)

    async def submit_command(
        self,
        command: dict[str, Any],
        timeout: float = 5.0,
    ) -> tuple[bool, Any]:
        """Submit a command to the cluster leader."""
        leader = self.get_leader()
        if leader is None:
            return False, {"error": "no_leader"}

        return await leader.submit_command(command, timeout)

    async def wait_for_leader(self, timeout: float = 10.0) -> RaftNode | None:
        """Wait for a leader to be elected."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            leader = self.get_leader()
            if leader is not None:
                return leader
            await asyncio.sleep(0.1)
        return None


def create_cluster_configs(
    node_ids: list[str],
    **kwargs: Any,
) -> list[ClusterConfiguration]:
    """Create cluster configurations for a set of node IDs."""
    configs = []
    for node_id in node_ids:
        peers = [nid for nid in node_ids if nid != node_id]
        config = ClusterConfiguration(
            node_id=node_id,
            peers=peers,
            **kwargs,
        )
        configs.append(config)
    return configs
