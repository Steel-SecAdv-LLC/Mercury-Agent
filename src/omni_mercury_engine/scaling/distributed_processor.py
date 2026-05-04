"""
Mercury Agent - Distributed Processing Module

Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Scalability enhancements for large-scale anomaly detection including:
- Distributed processing with worker pools
- Chunked data processing for memory efficiency
- Parallel detector execution
- Async processing pipelines
- Load balancing strategies
- Fault tolerance and recovery
- Progress tracking and monitoring
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class ProcessingStrategy(StrEnum):
    """Available processing strategies."""

    SEQUENTIAL = "sequential"
    THREADED = "threaded"
    MULTIPROCESS = "multiprocess"
    ASYNC = "async"
    HYBRID = "hybrid"  # Threads for I/O, processes for compute


class LoadBalancer(StrEnum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED = "weighted"
    ADAPTIVE = "adaptive"


@dataclass
class ProcessingConfig:
    """Configuration for distributed processing."""

    strategy: ProcessingStrategy = ProcessingStrategy.THREADED
    num_workers: int = 4
    chunk_size: int = 1000
    batch_size: int = 32
    timeout_seconds: float = 300.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    memory_limit_mb: int = 4096
    enable_progress: bool = True
    fault_tolerance: bool = True


@dataclass
class ChunkResult:
    """Result from processing a data chunk."""

    chunk_id: int
    start_idx: int
    end_idx: int
    scores: NDArray[np.float64]
    is_anomaly: NDArray[np.bool_]
    processing_time_ms: float
    worker_id: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingStats:
    """Statistics for processing operation."""

    total_samples: int = 0
    processed_samples: int = 0
    failed_samples: int = 0
    total_chunks: int = 0
    completed_chunks: int = 0
    failed_chunks: int = 0
    total_time_seconds: float = 0.0
    throughput_samples_per_sec: float = 0.0
    avg_chunk_time_ms: float = 0.0
    memory_peak_mb: float = 0.0
    retries: int = 0


class ChunkGenerator:
    """
    Generator for creating data chunks for distributed processing.

    Supports memory-efficient iteration over large datasets.
    """

    def __init__(
        self,
        data: NDArray[np.float64],
        chunk_size: int = 1000,
        overlap: int = 0,
    ):
        """
        Initialize chunk generator.

        Args:
            data: Input data array
            chunk_size: Size of each chunk
            overlap: Overlap between chunks (for time-series continuity)
        """
        self.data = data
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._current_idx = 0

    def __iter__(self) -> Iterator[tuple[int, int, NDArray[np.float64]]]:
        """Iterate over chunks."""
        n_samples = len(self.data)
        chunk_id = 0

        while self._current_idx < n_samples:
            start_idx = max(0, self._current_idx - self.overlap)
            end_idx = min(self._current_idx + self.chunk_size, n_samples)

            chunk_data = self.data[start_idx:end_idx]

            yield chunk_id, start_idx, chunk_data

            self._current_idx = end_idx
            chunk_id += 1

    def __len__(self) -> int:
        """Return number of chunks."""
        n_samples = len(self.data)
        return (n_samples + self.chunk_size - 1) // self.chunk_size

    def reset(self) -> None:
        """Reset iterator."""
        self._current_idx = 0


class WorkerPool(ABC):
    """Abstract base class for worker pools."""

    @abstractmethod
    def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Submit task to pool."""
        pass

    @abstractmethod
    def map(self, func: Callable[..., Any], items: list[Any]) -> list[Any]:
        """Map function over items."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown pool."""
        pass


class ThreadWorkerPool(WorkerPool):
    """Thread-based worker pool for I/O-bound tasks."""

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self._executor = ThreadPoolExecutor(max_workers=num_workers)

    def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Submit."""
        return self._executor.submit(func, *args, **kwargs)

    def map(self, func: Callable[..., Any], items: list[Any]) -> list[Any]:
        """Map."""
        futures = [self._executor.submit(func, item) for item in items]
        return [f.result() for f in as_completed(futures)]

    def shutdown(self) -> None:
        """Shutdown."""
        self._executor.shutdown(wait=True)


class ProcessWorkerPool(WorkerPool):
    """Process-based worker pool for CPU-bound tasks."""

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self._executor = ProcessPoolExecutor(max_workers=num_workers)

    def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Submit."""
        return self._executor.submit(func, *args, **kwargs)

    def map(self, func: Callable[..., Any], items: list[Any]) -> list[Any]:
        """Map."""
        return list(self._executor.map(func, items))

    def shutdown(self) -> None:
        """Shutdown."""
        self._executor.shutdown(wait=True)


class DistributedProcessor:
    """
    Distributed processor for large-scale anomaly detection.

    Handles chunking, parallel processing, and result aggregation.
    """

    def __init__(
        self,
        detector: Any,
        config: ProcessingConfig | None = None,
    ):
        """
        Initialize distributed processor.

        Args:
            detector: Anomaly detector with detect() method
            config: Processing configuration
        """
        self.detector = detector
        self.config = config or ProcessingConfig()
        self._pool: WorkerPool | None = None
        self._stats = ProcessingStats()
        self._progress_callback: Callable[[int, int], None] | None = None
        self._lock = threading.Lock()

    def set_progress_callback(
        self,
        callback: Callable[[int, int], None],
    ) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def _init_pool(self) -> WorkerPool:
        """Initialize worker pool based on strategy."""
        if self.config.strategy == ProcessingStrategy.THREADED:
            return ThreadWorkerPool(self.config.num_workers)
        elif self.config.strategy == ProcessingStrategy.MULTIPROCESS:
            return ProcessWorkerPool(self.config.num_workers)
        else:
            return ThreadWorkerPool(self.config.num_workers)

    def _process_chunk(
        self,
        chunk_id: int,
        start_idx: int,
        chunk_data: NDArray[np.float64],
        worker_id: int = 0,
    ) -> ChunkResult:
        """Process a single data chunk."""
        start_time = time.time()

        try:
            # Run detection
            result = self.detector.detect(chunk_data)

            scores = np.asarray(result.get("scores", np.zeros(len(chunk_data))))
            is_anomaly = np.asarray(result.get("is_anomaly", scores > 0.5))

            processing_time = (time.time() - start_time) * 1000

            return ChunkResult(
                chunk_id=chunk_id,
                start_idx=start_idx,
                end_idx=start_idx + len(chunk_data),
                scores=scores,
                is_anomaly=is_anomaly,
                processing_time_ms=processing_time,
                worker_id=worker_id,
                metadata=result.get("metadata", {}),
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            return ChunkResult(
                chunk_id=chunk_id,
                start_idx=start_idx,
                end_idx=start_idx + len(chunk_data),
                scores=np.zeros(len(chunk_data)),
                is_anomaly=np.zeros(len(chunk_data), dtype=bool),
                processing_time_ms=processing_time,
                worker_id=worker_id,
                error=str(e),
            )

    def process(
        self,
        data: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], ProcessingStats]:
        """
        Process data using distributed strategy.

        Args:
            data: Input data array

        Returns:
            Tuple of (scores, is_anomaly, stats)
        """
        n_samples = len(data)
        self._stats = ProcessingStats(total_samples=n_samples)

        start_time = time.time()

        # Initialize pool
        self._pool = self._init_pool()

        # Create chunk generator
        chunk_gen = ChunkGenerator(data, self.config.chunk_size)
        self._stats.total_chunks = len(chunk_gen)

        # Collect results
        results: list[ChunkResult] = []

        try:
            if self.config.strategy == ProcessingStrategy.SEQUENTIAL:
                results = self._process_sequential(chunk_gen)
            else:
                results = self._process_parallel(chunk_gen)
        finally:
            if self._pool:
                self._pool.shutdown()

        # Aggregate results
        scores = np.zeros(n_samples)
        is_anomaly = np.zeros(n_samples, dtype=bool)

        for result in results:
            if result.error is None:
                scores[result.start_idx : result.end_idx] = result.scores
                is_anomaly[result.start_idx : result.end_idx] = result.is_anomaly
                self._stats.processed_samples += len(result.scores)
                self._stats.completed_chunks += 1
            else:
                self._stats.failed_chunks += 1
                self._stats.failed_samples += result.end_idx - result.start_idx
                logger.error(f"Chunk {result.chunk_id} failed: {result.error}")

        # Compute final stats
        self._stats.total_time_seconds = time.time() - start_time
        if self._stats.total_time_seconds > 0:
            self._stats.throughput_samples_per_sec = (
                self._stats.processed_samples / self._stats.total_time_seconds
            )

        chunk_times = [r.processing_time_ms for r in results if r.error is None]
        if chunk_times:
            self._stats.avg_chunk_time_ms = float(np.mean(chunk_times))  # type: ignore[assignment, unused-ignore]

        return scores, is_anomaly, self._stats

    def _process_sequential(
        self,
        chunk_gen: ChunkGenerator,
    ) -> list[ChunkResult]:
        """Process chunks sequentially."""
        results = []

        for chunk_id, start_idx, chunk_data in chunk_gen:
            result = self._process_chunk(chunk_id, start_idx, chunk_data)
            results.append(result)

            if self._progress_callback:
                self._progress_callback(chunk_id + 1, len(chunk_gen))

        return results

    def _process_parallel(
        self,
        chunk_gen: ChunkGenerator,
    ) -> list[ChunkResult]:
        """Process chunks in parallel."""
        results: list[ChunkResult] = []
        futures = []

        # Submit all chunks
        assert self._pool is not None
        for chunk_id, start_idx, chunk_data in chunk_gen:
            worker_id = chunk_id % self.config.num_workers
            future = self._pool.submit(
                self._process_chunk_wrapper,
                chunk_id,
                start_idx,
                chunk_data.tolist(),
                worker_id,
            )
            futures.append((chunk_id, future))

        # Collect results
        for chunk_id, future in futures:
            try:
                result = future.result(timeout=self.config.timeout_seconds)
                results.append(result)

                if self._progress_callback:
                    self._progress_callback(len(results), len(futures))

            except Exception as e:
                logger.error(f"Chunk {chunk_id} failed: {e}")
                self._stats.failed_chunks += 1

        return results

    def _process_chunk_wrapper(
        self,
        chunk_id: int,
        start_idx: int,
        chunk_data_list: list[Any],
        worker_id: int,
    ) -> ChunkResult:
        """Wrapper for multiprocessing (data must be serializable)."""
        chunk_data = np.array(chunk_data_list)
        return self._process_chunk(chunk_id, start_idx, chunk_data, worker_id)

    def get_stats(self) -> ProcessingStats:
        """Get processing statistics."""
        return self._stats


class AsyncProcessor:
    """
    Async processor for non-blocking anomaly detection.

    Suitable for integration with async web frameworks.
    """

    def __init__(
        self,
        detector: Any,
        config: ProcessingConfig | None = None,
    ):
        """
        Initialize async processor.

        Args:
            detector: Anomaly detector
            config: Processing configuration
        """
        self.detector = detector
        self.config = config or ProcessingConfig()
        self._stats = ProcessingStats()

    async def process(
        self,
        data: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], ProcessingStats]:
        """
        Process data asynchronously.

        Args:
            data: Input data array

        Returns:
            Tuple of (scores, is_anomaly, stats)
        """
        n_samples = len(data)
        self._stats = ProcessingStats(total_samples=n_samples)

        start_time = time.time()

        # Create chunks
        chunk_gen = ChunkGenerator(data, self.config.chunk_size)
        self._stats.total_chunks = len(chunk_gen)

        # Process chunks concurrently
        tasks = []
        for chunk_id, start_idx, chunk_data in chunk_gen:
            task = asyncio.create_task(self._process_chunk_async(chunk_id, start_idx, chunk_data))
            tasks.append(task)

        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        scores = np.zeros(n_samples)
        is_anomaly = np.zeros(n_samples, dtype=bool)

        for result in results:
            if isinstance(result, ChunkResult) and result.error is None:
                scores[result.start_idx : result.end_idx] = result.scores
                is_anomaly[result.start_idx : result.end_idx] = result.is_anomaly
                self._stats.processed_samples += len(result.scores)
                self._stats.completed_chunks += 1
            else:
                self._stats.failed_chunks += 1

        # Compute stats
        self._stats.total_time_seconds = time.time() - start_time
        if self._stats.total_time_seconds > 0:
            self._stats.throughput_samples_per_sec = (
                self._stats.processed_samples / self._stats.total_time_seconds
            )

        return scores, is_anomaly, self._stats

    async def _process_chunk_async(
        self,
        chunk_id: int,
        start_idx: int,
        chunk_data: NDArray[np.float64],
    ) -> ChunkResult:
        """Process chunk asynchronously."""
        start_time = time.time()

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.detector.detect, chunk_data)

            scores = np.asarray(result.get("scores", np.zeros(len(chunk_data))))
            is_anomaly = np.asarray(result.get("is_anomaly", scores > 0.5))

            processing_time = (time.time() - start_time) * 1000

            return ChunkResult(
                chunk_id=chunk_id,
                start_idx=start_idx,
                end_idx=start_idx + len(chunk_data),
                scores=scores,
                is_anomaly=is_anomaly,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            return ChunkResult(
                chunk_id=chunk_id,
                start_idx=start_idx,
                end_idx=start_idx + len(chunk_data),
                scores=np.zeros(len(chunk_data)),
                is_anomaly=np.zeros(len(chunk_data), dtype=bool),
                processing_time_ms=processing_time,
                error=str(e),
            )


class StreamProcessor:
    """
    Stream processor for real-time anomaly detection.

    Handles continuous data streams with backpressure management.
    """

    def __init__(
        self,
        detector: Any,
        config: ProcessingConfig | None = None,
        queue_size: int = 1000,
    ):
        """
        Initialize stream processor.

        Args:
            detector: Anomaly detector
            config: Processing configuration
            queue_size: Maximum queue size for backpressure
        """
        self.detector = detector
        self.config = config or ProcessingConfig()
        self.queue_size = queue_size

        self._input_queue: queue.Queue[tuple[NDArray[np.float64], dict[str, Any]]] = queue.Queue(
            maxsize=queue_size
        )
        self._output_queue: queue.Queue[
            tuple[NDArray[np.float64], NDArray[np.bool_], dict[str, Any]]
        ] = queue.Queue(maxsize=queue_size)
        self._running = False
        self._workers: list[threading.Thread] = []
        self._stats = ProcessingStats()
        self._stats_lock = threading.Lock()

    def start(self) -> None:
        """Start stream processing workers."""
        self._running = True

        for i in range(self.config.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

        logger.info(f"Started {self.config.num_workers} stream processing workers")

    def stop(self) -> None:
        """Stop stream processing."""
        self._running = False

        for worker in self._workers:
            worker.join(timeout=5.0)

        self._workers.clear()
        logger.info("Stream processing stopped")

    def submit(
        self,
        data: NDArray[np.float64],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Submit data to processing queue.

        Args:
            data: Input data
            metadata: Optional metadata

        Returns:
            True if submitted, False if queue full
        """
        try:
            self._input_queue.put_nowait((data, metadata or {}))
            return True
        except queue.Full:
            logger.warning("Input queue full, data dropped")
            return False

    def get_result(
        self,
        timeout: float | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], dict[str, Any]] | None:
        """
        Get processing result from output queue.

        Args:
            timeout: Timeout in seconds

        Returns:
            Tuple of (scores, is_anomaly, metadata) or None if timeout
        """
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _worker_loop(self, worker_id: int) -> None:
        """Worker processing loop."""
        while self._running:
            try:
                data, metadata = self._input_queue.get(timeout=1.0)

                start_time = time.time()
                result = self.detector.detect(data)
                processing_time = time.time() - start_time

                scores = np.asarray(result.get("scores", np.zeros(len(data))))
                is_anomaly = np.asarray(result.get("is_anomaly", scores > 0.5))

                result_metadata = {
                    **metadata,
                    "worker_id": worker_id,
                    "processing_time_ms": processing_time * 1000,
                }

                self._output_queue.put((scores, is_anomaly, result_metadata))

                with self._stats_lock:
                    self._stats.processed_samples += len(data)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    def get_queue_status(self) -> dict[str, int]:
        """Get queue status."""
        return {
            "input_queue_size": self._input_queue.qsize(),
            "output_queue_size": self._output_queue.qsize(),
            "max_queue_size": self.queue_size,
        }


def create_processor(
    detector: Any,
    strategy: str = "threaded",
    **kwargs: Any,
) -> DistributedProcessor | AsyncProcessor | StreamProcessor:
    """
    Factory function to create appropriate processor.

    Args:
        detector: Anomaly detector
        strategy: Processing strategy
        **kwargs: Additional configuration

    Returns:
        Configured processor instance
    """
    config = ProcessingConfig(
        strategy=ProcessingStrategy(strategy),
        num_workers=kwargs.get("num_workers", 4),
        chunk_size=kwargs.get("chunk_size", 1000),
    )

    if strategy == "async":
        return AsyncProcessor(detector, config)
    elif strategy == "stream":
        return StreamProcessor(detector, config)
    else:
        return DistributedProcessor(detector, config)


# Exports
__all__ = [
    "AsyncProcessor",
    "ChunkGenerator",
    "ChunkResult",
    "DistributedProcessor",
    "LoadBalancer",
    "ProcessWorkerPool",
    "ProcessingConfig",
    "ProcessingStats",
    "ProcessingStrategy",
    "StreamProcessor",
    "ThreadWorkerPool",
    "WorkerPool",
    "create_processor",
]
