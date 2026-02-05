"""
Distributed Processing Module for Mercury Agent.

Provides multi-node deployment with Raft consensus for fault-tolerant,
horizontally scalable anomaly detection.

Key Components:
- RaftCluster: Raft consensus implementation for leader election and log replication
- DistributedMercuryCluster: High-level interface for distributed operations
- WorkStealingScheduler: Load balancing via work stealing
- DataPartitioner: Data distribution across nodes
- ResultAggregator: Fusion of distributed results
"""

from omni_mercury_engine.distributed.cluster import (
    ClusterHealth,
    DataPartitioner,
    DistributedAnomalyDetector,
    DistributedMercuryCluster,
    DistributedTask,
    NodeHealth,
    PartitionStrategy,
    ResultAggregator,
    TaskResult,
    TaskStatus,
    WorkStealingScheduler,
)
from omni_mercury_engine.distributed.raft_consensus import (
    AppendEntriesRequest,
    AppendEntriesResponse,
    ClusterConfiguration,
    InMemoryTransport,
    LogEntry,
    MessageTransport,
    NodeState,
    RaftCluster,
    RaftLog,
    RaftNode,
    RequestVoteRequest,
    RequestVoteResponse,
    StateMachine,
    create_cluster_configs,
)


__all__ = [
    "AppendEntriesRequest",
    "AppendEntriesResponse",
    "ClusterConfiguration",
    "ClusterHealth",
    # Data handling
    "DataPartitioner",
    "DistributedAnomalyDetector",
    # Cluster management
    "DistributedMercuryCluster",
    # Task[Any] management
    "DistributedTask",
    "InMemoryTransport",
    "LogEntry",
    # Transport
    "MessageTransport",
    "NodeHealth",
    "NodeState",
    "PartitionStrategy",
    # Raft consensus
    "RaftCluster",
    "RaftLog",
    "RaftNode",
    # Messages
    "RequestVoteRequest",
    "RequestVoteResponse",
    "ResultAggregator",
    # State machine
    "StateMachine",
    "TaskResult",
    "TaskStatus",
    "WorkStealingScheduler",
    "create_cluster_configs",
]
