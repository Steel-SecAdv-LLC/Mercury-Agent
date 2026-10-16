# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Concurrent execution of independent detector computations.

This restores the capability of the ``ParallelDetectorExecutor`` removed from
``core/base_domains.py`` in 7ab59916 — running a set of detectors concurrently
to cut the latency of one fusion pass — without its three defects:

* **A slow detector discarded every finished result.** The old loop used
  ``as_completed(..., timeout=...)``; on expiry it raised ``TimeoutError`` out of
  the loop, losing results that had already completed, and the ``with`` block
  then waited for the slow detector anyway (measured 3.0 s against a 0.5 s
  timeout). Here the deadline is a wall-clock budget for the whole batch:
  finished results are kept, unfinished tasks are reported as
  :attr:`OutcomeStatus.TIMEOUT`, tasks that never started are cancelled, and the
  call returns at the deadline.
* **A hung detector starved the calls after it.** A worker stuck past the
  deadline kept its thread, so the next call queued behind it and timed out
  too (measured on ``DetectorRegistry`` at 25d624e9). Here a deadline overrun
  retires the pool: the stuck threads finish on their own, and the next call
  gets fresh workers. The number of abandoned threads is capped; past the cap
  the executor refuses parallel work with :class:`ExecutorSaturatedError`
  instead of leaking threads. The pool is dedicated, and a call made from one
  of its own worker threads runs sequentially in that thread instead of
  submitting again.
* **Results came back in completion order.** Outcomes are returned in the
  order the tasks were given, so downstream assembly is deterministic.

Concurrency is opt-in per task. A computation that touches process-global
mutable state — for example, a model that draws from the shared random
generator during inference — would produce scheduling-dependent results if run
in parallel, so it is submitted with ``parallel_safe=False`` and runs
sequentially in the calling thread, in its original order. Threads, not
processes, are used: the heavy work in Mercury's detectors is numpy and torch
code that releases the GIL, and the detector objects (fitted state, caches)
must be shared without pickling.

Python threads cannot be killed. A task that overruns the deadline keeps
running in its worker until it finishes; its result is discarded. The outcome
says so, and the pool keeps serving the remaining workers.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.core.exceptions import InputModalityError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "DetectorTask",
    "ExecutorSaturatedError",
    "OutcomeStatus",
    "ParallelDetectorExecutor",
    "TaskOutcome",
    "first_failure",
    "reraise",
]

#: Default worker cap. Four workers already cover the five default detectors
#: with one queued; beyond the core count, threads only add contention.
DEFAULT_MAX_WORKERS = 4

#: Abandoned (overrun, still running) threads allowed per worker before the
#: executor refuses parallel work. Python threads cannot be killed, so this is
#: what bounds the threads a persistently hanging detector can strand.
OVERRUN_LIMIT_PER_WORKER = 4


class OutcomeStatus(StrEnum):
    """How one task ended."""

    OK = "ok"
    DECLINED = "declined"  # raised InputModalityError: not this component's input
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class DetectorTask:
    """One unit of work.

    Attributes:
        name: Identifier reported back in the outcome (usually the detector name).
        fn: Zero-argument callable doing the work.
        parallel_safe: ``True`` when ``fn`` touches no process-global mutable
            state, so it may run concurrently with other tasks.
    """

    name: str
    fn: Callable[[], Any]
    parallel_safe: bool = True


@dataclass(frozen=True)
class TaskOutcome:
    """The result of one task.

    Attributes:
        name: The task name.
        status: How the task ended.
        value: The callable's return value when ``status`` is ``OK``.
        error: The exception raised, for ``DECLINED`` and ``ERROR``.
        duration_s: Wall-clock time from the task starting to finishing, or to
            the deadline for ``TIMEOUT``. Queue time is not included.
    """

    name: str
    status: OutcomeStatus
    value: Any = None
    error: Exception | None = None
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        """Whether the task returned normally."""
        return self.status is OutcomeStatus.OK


class ExecutorSaturatedError(RuntimeError):
    """Too many overrun tasks are still running; parallel work is refused."""


def _classify(exc: Exception) -> OutcomeStatus:
    return OutcomeStatus.DECLINED if isinstance(exc, InputModalityError) else OutcomeStatus.ERROR


class ParallelDetectorExecutor:
    """Run detector computations concurrently on a dedicated thread pool.

    Example:
        >>> executor = ParallelDetectorExecutor(max_workers=4)
        >>> tasks = [DetectorTask(name, (lambda d=d: d.detect(x))) for name, d in detectors.items()]
        >>> outcomes = executor.run(tasks)          # same order as ``tasks``
        >>> results = {o.name: o.value for o in outcomes if o.ok}
    """

    def __init__(
        self,
        max_workers: int | None = None,
        timeout_s: float | None = None,
        thread_name_prefix: str = "mercury-detector",
    ) -> None:
        """Configure the executor; the pool is created on first parallel use.

        Args:
            max_workers: Worker threads. ``None`` uses
                ``min(DEFAULT_MAX_WORKERS, os.cpu_count())``. ``1`` runs every
                task sequentially in the calling thread.
            timeout_s: Default wall-clock budget for one :meth:`run` call;
                ``None`` waits for every task.
            thread_name_prefix: Prefix for the worker thread names.

        Raises:
            ValueError: ``max_workers`` is below 1 or ``timeout_s`` is not positive.
        """
        if max_workers is None:
            max_workers = min(DEFAULT_MAX_WORKERS, os.cpu_count() or 1)
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")
        if timeout_s is not None and not timeout_s > 0:
            raise ValueError(f"timeout_s must be positive or None, got {timeout_s}")
        self.max_workers = max_workers
        self.timeout_s = timeout_s
        self._thread_name_prefix = thread_name_prefix
        self._pool: ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()
        self._worker = threading.local()
        self._overrun_limit = OVERRUN_LIMIT_PER_WORKER * max_workers
        self._overrun_running = 0

    # ------------------------------------------------------------------
    # Pool lifecycle
    # ------------------------------------------------------------------
    def _get_pool(self) -> ThreadPoolExecutor:
        with self._pool_lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix=self._thread_name_prefix,
                )
            return self._pool

    @property
    def overrun_running(self) -> int:
        """Tasks abandoned at a deadline that are still running in a thread."""
        with self._pool_lock:
            return self._overrun_running

    def _retire_pool(self, pool: ThreadPoolExecutor) -> None:
        """Stop handing work to ``pool``; its threads exit once their tasks end.

        Queued work from a concurrent :meth:`run` call on the same pool still
        runs: nothing is cancelled here.
        """
        with self._pool_lock:
            if self._pool is pool:
                self._pool = None
        pool.shutdown(wait=False)

    def _overrun_finished(self, _future: Future[TaskOutcome]) -> None:
        with self._pool_lock:
            self._overrun_running -= 1

    def shutdown(self, wait: bool = True) -> None:
        """Stop the pool. Not-yet-started tasks are cancelled.

        Args:
            wait: Block until running tasks finish.
        """
        with self._pool_lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self) -> ParallelDetectorExecutor:
        """Return the executor for use as a context manager."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Shut the pool down on leaving the ``with`` block."""
        self.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _in_worker_thread(self) -> bool:
        return bool(getattr(self._worker, "active", False))

    def _call(self, task: DetectorTask) -> TaskOutcome:
        """Run one task in the current thread and capture how it ended."""
        start = time.perf_counter()
        try:
            value = task.fn()
        except Exception as exc:  # classified and reported, never swallowed silently
            return TaskOutcome(task.name, _classify(exc), None, exc, time.perf_counter() - start)
        return TaskOutcome(task.name, OutcomeStatus.OK, value, None, time.perf_counter() - start)

    def _call_in_worker(self, task: DetectorTask, started: dict[str, float]) -> TaskOutcome:
        self._worker.active = True
        started[task.name] = time.perf_counter()
        try:
            return self._call(task)
        finally:
            self._worker.active = False

    def run(
        self, tasks: Sequence[DetectorTask], timeout_s: float | None = None
    ) -> list[TaskOutcome]:
        """Execute ``tasks`` and return one outcome per task, in input order.

        Parallel-safe tasks run on the pool; the others run sequentially in the
        calling thread while the pool works. A call from one of this executor's
        own worker threads, a single-worker executor, or a batch with fewer than
        two parallel-safe tasks runs everything sequentially.

        Args:
            tasks: The work to do. Task names must be unique.
            timeout_s: Wall-clock budget for this call, overriding the default.
                Only parallel tasks can time out; sequential ones run to
                completion in the calling thread.

        Returns:
            Outcomes aligned with ``tasks``.

        Raises:
            ValueError: duplicate task names.

        When overrun tasks from earlier calls still occupy the thread cap, the
        parallel tasks are not run: their outcomes are ``ERROR`` with an
        :class:`ExecutorSaturatedError`. The sequential tasks still run.
        """
        names = [t.name for t in tasks]
        if len(set(names)) != len(names):
            raise ValueError("task names must be unique")
        budget = self.timeout_s if timeout_s is None else timeout_s

        parallel = [t for t in tasks if t.parallel_safe]
        if self.max_workers == 1 or len(parallel) < 2 or self._in_worker_thread():
            return [self._call(t) for t in tasks]

        stranded = self.overrun_running
        if stranded >= self._overrun_limit:
            refused = ExecutorSaturatedError(
                f"{stranded} overrun detector tasks are still running "
                f"(limit {self._overrun_limit}); parallel work refused"
            )
            return [
                (
                    TaskOutcome(t.name, OutcomeStatus.ERROR, None, refused)
                    if t.parallel_safe
                    else self._call(t)
                )
                for t in tasks
            ]

        deadline = None if budget is None else time.monotonic() + budget
        started: dict[str, float] = {}
        pool = self._get_pool()
        futures: dict[Future[TaskOutcome], DetectorTask] = {
            pool.submit(self._call_in_worker, t, started): t for t in parallel
        }

        # The non-parallel tasks run here while the pool works.
        outcomes: dict[str, TaskOutcome] = {
            t.name: self._call(t) for t in tasks if not t.parallel_safe
        }

        pending = set(futures)
        while pending:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                break
            done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
            for future in done:
                outcomes[futures[future].name] = future.result()

        now = time.perf_counter()
        overran = False
        for future in pending:
            task = futures[future]
            if not future.cancel() and future.done():
                outcomes[task.name] = future.result()  # finished at the deadline
                continue
            if not future.cancelled():  # started: it keeps its thread until it ends
                overran = True
                with self._pool_lock:
                    self._overrun_running += 1
                future.add_done_callback(self._overrun_finished)
            began = started.get(task.name)
            outcomes[task.name] = TaskOutcome(
                task.name,
                OutcomeStatus.TIMEOUT,
                None,
                None,
                0.0 if began is None else now - began,
            )
        if overran:
            self._retire_pool(pool)
        return [outcomes[name] for name in names]


def first_failure(outcomes: Sequence[TaskOutcome]) -> TaskOutcome | None:
    """Return the first non-OK outcome in task order, or ``None`` if all succeeded."""
    return next((o for o in outcomes if not o.ok), None)


def reraise(outcome: TaskOutcome) -> None:
    """Re-raise the exception an outcome captured (no-op for OK outcomes).

    Raises:
        Exception: the captured exception, unchanged.
        TimeoutError: the outcome timed out.
    """
    if outcome.error is not None:
        raise outcome.error
    if outcome.status is OutcomeStatus.TIMEOUT:
        raise TimeoutError(f"{outcome.name} exceeded the detector time budget")
