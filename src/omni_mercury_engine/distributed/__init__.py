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
    # Cluster management
    "DistributedMercuryCluster",
    "DistributedAnomalyDetector",
    "ClusterHealth",
    "NodeHealth",
    # Task management
    "DistributedTask",
    "TaskResult",
    "TaskStatus",
    "WorkStealingScheduler",
    # Data handling
    "DataPartitioner",
    "PartitionStrategy",
    "ResultAggregator",
    # Raft consensus
    "RaftCluster",
    "RaftNode",
    "RaftLog",
    "NodeState",
    "LogEntry",
    "ClusterConfiguration",
    "create_cluster_configs",
    # Transport
    "MessageTransport",
    "InMemoryTransport",
    # Messages
    "RequestVoteRequest",
    "RequestVoteResponse",
    "AppendEntriesRequest",
    "AppendEntriesResponse",
    # State machine
    "StateMachine",
]
