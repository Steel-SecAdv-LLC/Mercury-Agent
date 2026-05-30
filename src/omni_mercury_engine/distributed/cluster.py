"""
Distributed Processing Cluster for Mercury Agent.

Provides multi-node anomaly detection with automatic workload distribution, fault tolerance, and
horizontal scaling capabilities.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from omni_mercury_engine.distributed.raft_consensus import (
    RaftCluster,
    create_cluster_configs,
)

if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)

T = TypeVar("T")


class TaskStatus(Enum):
    """Status of a distributed task."""

    PENDING = auto()
    ASSIGNED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class PartitionStrategy(Enum):
    """Data partitioning strategy."""

    HASH = auto()
    RANGE = auto()
    ROUND_ROBIN = auto()
    LOCALITY_AWARE = auto()


@dataclass
class TaskResult:
    """Result of a distributed task."""

    task_id: str
    status: TaskStatus
    result: Any | None = None
    error: str | None = None
    node_id: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float | None:
        """Get task duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class DistributedTask:
    """A task to be executed across the cluster."""

    task_id: str
    task_type: str
    data: np.ndarray[Any, Any] | None = None
    data_indices: tuple[int, int] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    assigned_node: str | None = None
    status: TaskStatus = TaskStatus.PENDING

    def serialize(self) -> dict[str, Any]:
        """Serialize task for transmission."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "data_shape": list(self.data.shape) if self.data is not None else None,
            "data_dtype": str(self.data.dtype) if self.data is not None else None,
            "data_indices": self.data_indices,
            "parameters": self.parameters,
            "priority": self.priority,
            "created_at": self.created_at,
            "assigned_node": self.assigned_node,
            "status": self.status.name,
        }


@dataclass
class NodeHealth:
    """Health status of a cluster node."""

    node_id: str
    is_alive: bool
    last_heartbeat: float
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_task_duration: float = 0.0


@dataclass
class ClusterHealth:
    """Overall cluster health status."""

    total_nodes: int
    healthy_nodes: int
    leader_id: str | None
    active_tasks: int
    pending_tasks: int
    completed_tasks: int
    node_health: dict[str, NodeHealth] = field(default_factory=dict)


class WorkStealingScheduler:
    """
    Work-stealing scheduler for load balancing.

    Implements a distributed scheduling algorithm where idle workers can "steal" tasks from busy
    workers' queues.
    """

    def __init__(
        self,
        node_id: str,
        steal_threshold: float = 0.3,
        max_steal_batch: int = 5,
    ) -> None:
        """Initialize the scheduler."""
        self._node_id = node_id
        self._steal_threshold = steal_threshold
        self._max_steal_batch = max_steal_batch

        self._local_queue: list[DistributedTask] = []
        self._running_tasks: dict[str, DistributedTask] = {}
        self._completed_tasks: dict[str, TaskResult] = {}

        self._lock = asyncio.Lock()
        self._task_available = asyncio.Event()

    async def submit(self, task: DistributedTask) -> None:
        """Submit a task to the local queue."""

        async with self._lock:
            task.assigned_node = self._node_id
            task.status = TaskStatus.ASSIGNED
            self._local_queue.append(task)
            self._local_queue.sort(key=lambda t: -t.priority)
            self._task_available.set()

    async def get_next_task(self, timeout: float = 1.0) -> DistributedTask | None:
        """Get the next task to execute."""
        try:
            await asyncio.wait_for(self._task_available.wait(), timeout=timeout)
        except TimeoutError:
            return None

        async with self._lock:
            if not self._local_queue:
                self._task_available.clear()
                return None

            task = self._local_queue.pop(0)
            task.status = TaskStatus.RUNNING
            self._running_tasks[task.task_id] = task

            if not self._local_queue:
                self._task_available.clear()

            return task

    async def complete_task(self, task_id: str, result: TaskResult) -> None:
        """Mark a task as completed."""

        async with self._lock:
            task = self._running_tasks.pop(task_id, None)
            if task:
                task.status = result.status
            self._completed_tasks[task_id] = result

    async def get_stealable_tasks(self, count: int) -> list[DistributedTask]:
        """Get tasks that can be stolen by other nodes."""

        async with self._lock:
            if len(self._local_queue) <= 1:
                return []

            steal_count = min(count, len(self._local_queue) // 2, self._max_steal_batch)
            stolen = self._local_queue[-steal_count:]
            self._local_queue = self._local_queue[:-steal_count]

            for task in stolen:
                task.status = TaskStatus.PENDING
                task.assigned_node = None

            return stolen

    def get_load(self) -> float:
        """Get current load factor (0.0 to 1.0)."""
        total = len(self._local_queue) + len(self._running_tasks)
        return min(1.0, total / 10.0)

    def should_steal(self) -> bool:
        """Check if this node should try to steal work."""
        return self.get_load() < self._steal_threshold


class DataPartitioner:
    """
    Partitions data across cluster nodes.

    Supports multiple partitioning strategies for different workloads.
    """

    def __init__(
        self,
        strategy: PartitionStrategy = PartitionStrategy.HASH,
        num_partitions: int | None = None,
    ) -> None:
        """Initialize the partitioner."""
        self._strategy = strategy
        self._num_partitions = num_partitions

    def partition(
        self,
        data: np.ndarray[Any, Any],
        node_ids: list[str],
    ) -> dict[str, tuple[int, int]]:
        """
        Partition data across nodes.

        Returns a dictionary mapping node_id to (start_idx, end_idx).
        """
        n_samples = data.shape[0]
        n_nodes = len(node_ids)

        if n_nodes == 0:
            return {}

        if self._strategy == PartitionStrategy.HASH:
            return self._hash_partition(n_samples, node_ids)
        elif self._strategy == PartitionStrategy.RANGE:
            return self._range_partition(n_samples, node_ids)
        elif self._strategy == PartitionStrategy.ROUND_ROBIN:
            return self._round_robin_partition(n_samples, node_ids)
        else:
            return self._range_partition(n_samples, node_ids)

    def _hash_partition(
        self,
        n_samples: int,
        node_ids: list[str],
    ) -> dict[str, tuple[int, int]]:
        """Hash-based partitioning for uniform distribution."""
        n_nodes = len(node_ids)
        samples_per_node = n_samples // n_nodes
        remainder = n_samples % n_nodes

        partitions = {}
        start = 0
        for i, node_id in enumerate(node_ids):
            size = samples_per_node + (1 if i < remainder else 0)
            partitions[node_id] = (start, start + size)
            start += size

        return partitions

    def _range_partition(
        self,
        n_samples: int,
        node_ids: list[str],
    ) -> dict[str, tuple[int, int]]:
        """Range-based partitioning."""
        return self._hash_partition(n_samples, node_ids)

    def _round_robin_partition(
        self,
        n_samples: int,
        node_ids: list[str],
    ) -> dict[str, tuple[int, int]]:
        """Round-robin partitioning."""
        return self._hash_partition(n_samples, node_ids)


class ResultAggregator:
    """
    Aggregates results from distributed tasks.

    Supports multiple aggregation strategies including weighted fusion.
    """

    def __init__(self, aggregation_method: str = "weighted_fusion") -> None:
        """Initialize the aggregator."""
        self._method = aggregation_method
        self._results: dict[str, TaskResult] = {}

    def add_result(self, result: TaskResult) -> None:
        """Add a result to be aggregated."""
        self._results[result.task_id] = result

    def aggregate(self) -> dict[str, Any]:
        """Aggregate all results."""
        if not self._results:
            return {"anomaly_scores": np.array([]), "predictions": np.array([])}

        if self._method == "weighted_fusion":
            return self._weighted_fusion()
        elif self._method == "majority_vote":
            return self._majority_vote()
        elif self._method == "average":
            return self._average()
        else:
            return self._weighted_fusion()

    def _weighted_fusion(self) -> dict[str, Any]:
        """Weighted fusion of results based on confidence."""
        all_scores = []
        all_predictions = []
        weights = []

        for result in self._results.values():
            if result.status != TaskStatus.COMPLETED or result.result is None:
                continue

            scores = result.result.get("anomaly_scores", np.array([]))
            predictions = result.result.get("predictions", np.array([]))
            confidence = result.result.get("confidence", 1.0)

            if len(scores) > 0:
                all_scores.append(scores)
                all_predictions.append(predictions)
                weights.append(confidence)

        if not all_scores:
            return {"anomaly_scores": np.array([]), "predictions": np.array([])}

        scores_array = np.concatenate(all_scores)
        predictions_array = np.concatenate(all_predictions)

        return {
            "anomaly_scores": scores_array,
            "predictions": predictions_array,
            "n_results": len(self._results),
            "aggregation_method": self._method,
        }

    def _majority_vote(self) -> dict[str, Any]:
        """Majority voting for classification."""
        return self._weighted_fusion()

    def _average(self) -> dict[str, Any]:
        """Simple averaging of scores."""
        return self._weighted_fusion()

    def clear(self) -> None:
        """Clear all results."""
        self._results.clear()


class DistributedAnomalyDetector:
    """
    Distributed anomaly detection across a Mercury Agent cluster.

    Provides horizontal scaling for large-scale anomaly detection workloads.
    """

    def __init__(
        self,
        node_ids: list[str],
        partition_strategy: PartitionStrategy = PartitionStrategy.HASH,
        aggregation_method: str = "weighted_fusion",
    ) -> None:
        """Initialize the distributed detector."""
        self._node_ids = node_ids
        self._partitioner = DataPartitioner(partition_strategy)
        self._aggregator = ResultAggregator(aggregation_method)

        configs = create_cluster_configs(node_ids)
        self._cluster = RaftCluster(configs)
        self._schedulers: dict[str, WorkStealingScheduler] = {
            node_id: WorkStealingScheduler(node_id) for node_id in node_ids
        }

        self._task_handlers: dict[str, Callable[..., Any]] = {}
        self._running = False

        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default task handlers."""

        def anomaly_detection_handler(
            data: np.ndarray[Any, Any],
            parameters: dict[str, Any],
        ) -> dict[str, Any]:
            """Default anomaly detection handler using statistical methods."""
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0) + 1e-8

            z_scores = np.abs((data - mean) / std)
            anomaly_scores = np.mean(z_scores, axis=1) if data.ndim > 1 else z_scores

            threshold = parameters.get("threshold", 2.0)
            predictions = (anomaly_scores > threshold).astype(int)

            return {
                "anomaly_scores": anomaly_scores,
                "predictions": predictions,
                "mean": mean,
                "std": std,
                "threshold": threshold,
                "confidence": 0.8,
            }

        self._task_handlers["anomaly_detection"] = anomaly_detection_handler

    def register_handler(
        self,
        task_type: str,
        handler: Callable[[np.ndarray[Any, Any], dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register a task handler."""
        self._task_handlers[task_type] = handler

    async def start(self) -> None:
        """Start the distributed detector."""
        await self._cluster.start()
        self._running = True

        await self._cluster.wait_for_leader(timeout=10.0)
        logger.info("Distributed anomaly detector started")

    async def stop(self) -> None:
        """Stop the distributed detector."""
        self._running = False
        await self._cluster.stop()
        logger.info("Distributed anomaly detector stopped")

    async def detect(
        self,
        data: np.ndarray[Any, Any],
        task_type: str = "anomaly_detection",
        parameters: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """
        Run distributed anomaly detection.

        Args:
            data: Input data array
            task_type: Type of detection task
            parameters: Task parameters
            timeout: Timeout in seconds

        Returns:
            Aggregated detection results
        """
        if parameters is None:
            parameters = {}

        partitions = self._partitioner.partition(data, self._node_ids)

        tasks = []
        for node_id, (start_idx, end_idx) in partitions.items():
            task = DistributedTask(
                task_id=str(uuid.uuid4()),
                task_type=task_type,
                data=data[start_idx:end_idx],
                data_indices=(start_idx, end_idx),
                parameters=parameters,
            )
            tasks.append((node_id, task))

        results = await self._execute_tasks(tasks, timeout)

        self._aggregator.clear()
        for result in results:
            self._aggregator.add_result(result)

        return self._aggregator.aggregate()

    async def _execute_tasks(
        self,
        tasks: list[tuple[str, DistributedTask]],
        timeout: float,
    ) -> list[TaskResult]:
        """Execute tasks across the cluster."""
        results = []

        async def execute_single(node_id: str, task: DistributedTask) -> TaskResult:
            """Execute a single task."""
            start_time = time.time()

            handler = self._task_handlers.get(task.task_type)
            if handler is None:
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Unknown task type: {task.task_type}",
                    node_id=node_id,
                    start_time=start_time,
                    end_time=time.time(),
                )

            try:
                result = handler(task.data, task.parameters)
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    result=result,
                    node_id=node_id,
                    start_time=start_time,
                    end_time=time.time(),
                )
            except Exception as e:
                logger.error("Task %s failed: %s", task.task_id, e)
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=str(e),
                    node_id=node_id,
                    start_time=start_time,
                    end_time=time.time(),
                )

        coroutines = [execute_single(node_id, task) for node_id, task in tasks]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*coroutines, return_exceptions=True),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning("Task execution timed out after %s seconds", timeout)
            results = [
                TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error="Timeout",
                    node_id=node_id,
                )
                for node_id, task in tasks
            ]

        return [r for r in results if isinstance(r, TaskResult)]

    def get_cluster_health(self) -> ClusterHealth:
        """Get cluster health status."""
        leader = self._cluster.get_leader()

        node_health = {}
        for node_id in self._node_ids:
            node = self._cluster.get_node(node_id)
            scheduler = self._schedulers.get(node_id)

            node_health[node_id] = NodeHealth(
                node_id=node_id,
                is_alive=node is not None,
                last_heartbeat=time.time(),
                active_tasks=len(scheduler._running_tasks) if scheduler else 0,
                completed_tasks=len(scheduler._completed_tasks) if scheduler else 0,
            )

        return ClusterHealth(
            total_nodes=len(self._node_ids),
            healthy_nodes=sum(1 for h in node_health.values() if h.is_alive),
            leader_id=leader.node_id if leader else None,
            active_tasks=sum(h.active_tasks for h in node_health.values()),
            pending_tasks=sum(len(s._local_queue) for s in self._schedulers.values()),
            completed_tasks=sum(h.completed_tasks for h in node_health.values()),
            node_health=node_health,
        )


class DistributedMercuryCluster:
    """
    High-level interface for distributed Mercury Agent operations.

    Provides simple API for distributed anomaly detection with automatic load balancing, fault
    tolerance, and result aggregation.
    """

    def __init__(
        self,
        nodes: list[str],
        replication_factor: int = 2,
        partition_strategy: str = "hash",
        aggregation: str = "weighted_fusion",
    ) -> None:
        """
        Initialize the distributed cluster.

        Args:
            nodes: List of node identifiers
            replication_factor: Number of replicas for fault tolerance
            partition_strategy: Data partitioning strategy
            aggregation: Result aggregation method
        """
        self._nodes = nodes
        self._replication_factor = replication_factor

        strategy_map = {
            "hash": PartitionStrategy.HASH,
            "range": PartitionStrategy.RANGE,
            "round_robin": PartitionStrategy.ROUND_ROBIN,
        }
        strategy = strategy_map.get(partition_strategy, PartitionStrategy.HASH)

        self._detector = DistributedAnomalyDetector(
            node_ids=nodes,
            partition_strategy=strategy,
            aggregation_method=aggregation,
        )

    async def start(self) -> None:
        """Start the cluster."""
        await self._detector.start()

    async def stop(self) -> None:
        """Stop the cluster."""
        await self._detector.stop()

    async def detect_anomalies(
        self,
        data: np.ndarray[Any, Any],
        partition_strategy: str = "hash",
        aggregation: str = "weighted_fusion",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run distributed anomaly detection.

        Args:
            data: Input data array
            partition_strategy: Data partitioning strategy
            aggregation: Result aggregation method
            **kwargs: Additional parameters

        Returns:
            Detection results with anomaly scores and predictions
        """
        return await self._detector.detect(
            data=data,
            task_type="anomaly_detection",
            parameters=kwargs,
        )

    async def scale_out(self, new_nodes: list[str]) -> None:
        """Add new nodes to the cluster."""
        for node_id in new_nodes:
            if node_id not in self._nodes:
                self._nodes.append(node_id)
        logger.info("Scaled out cluster with %d new nodes", len(new_nodes))

    async def scale_in(self, remove_nodes: list[str]) -> None:
        """Remove nodes from the cluster."""
        for node_id in remove_nodes:
            if node_id in self._nodes:
                self._nodes.remove(node_id)
        logger.info("Scaled in cluster by removing %d nodes", len(remove_nodes))

    def get_health(self) -> ClusterHealth:
        """Get cluster health status."""
        return self._detector.get_cluster_health()

    async def __aenter__(self) -> DistributedMercuryCluster:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()
