"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Resilience and error recovery utilities for Mercury Agent ♱.

This module provides patterns for building resilient applications:
- Circuit breaker pattern for failing fast
- Retry with exponential backoff
- Graceful shutdown handling
- Health check utilities
- Bulkhead isolation

Example:
    Using circuit breaker::

        from omni_mercury_engine.utils.resilience import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60)

        @breaker
        def call_external_service() -> None:
            return service.call()

    Using retry decorator::

        from omni_mercury_engine.utils.resilience import retry

        @retry(max_attempts=3, backoff_factor=2.0)
        def unreliable_operation() -> None:
            return do_something()

    Graceful shutdown::

        from omni_mercury_engine.utils.resilience import GracefulShutdown

        shutdown = GracefulShutdown()
        shutdown.register_handler(cleanup_resources)

        # In your main loop
        while not shutdown.should_stop:
            process_work()
"""

from __future__ import annotations

import functools
import logging
import random
import signal
import threading
import time
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, TypeVar


# Import CircuitState from canonical location with fallback for backwards compatibility
try:
    from omni_mercury_engine.core.types import CircuitState
except ImportError:
    # Fallback for backwards compatibility if core.types is not available
    class CircuitState(Enum):  # type: ignore[no-redef]
        """States for circuit breaker pattern."""

        CLOSED = auto()  # Normal operation
        OPEN = auto()  # Failing, reject calls
        HALF_OPEN = auto()  # Testing if service recovered


__all__ = [
    "Bulkhead",
    "BulkheadFullError",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitState",
    "GracefulShutdown",
    "HealthChecker",
    "HealthStatus",
    "ShutdownInProgressError",
    "retry",
    "timeout",
]


logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker.

    Attributes:
        failure_threshold: Number of failures before opening circuit.
        success_threshold: Number of successes to close circuit from half-open.
        reset_timeout: Seconds to wait before attempting half-open.
        excluded_exceptions: Exceptions that don't count as failures.
        enable_exponential_backoff: Enable exponential backoff on repeated failures.
        backoff_base: Base multiplier for exponential backoff.
        max_backoff_timeout: Maximum backoff timeout in seconds.
        enable_jitter: Add random jitter to prevent thundering herd.
        jitter_factor: Jitter factor (0-1) for randomizing timeout.
    """

    failure_threshold: int = 5
    success_threshold: int = 1
    reset_timeout: float = 60.0
    excluded_exceptions: tuple[type[BaseException], ...] = ()
    enable_exponential_backoff: bool = False
    backoff_base: float = 2.0
    max_backoff_timeout: float = 3600.0
    enable_jitter: bool = False
    jitter_factor: float = 0.1


class CircuitBreaker:
    """Circuit breaker pattern implementation.

    Prevents cascading failures by failing fast when a service
    is experiencing issues.

    Features:
    - Thread-safe with RLock
    - 3-state circuit breaker (CLOSED/OPEN/HALF_OPEN)
    - Decorator pattern support via __call__
    - Direct call support via call() method
    - Exponential backoff with configurable base and max timeout
    - Jitter option to prevent thundering herd problem
    - Statistics tracking for monitoring

    Example:
        Using as decorator::

            >>> breaker = CircuitBreaker(failure_threshold=3)
            >>> @breaker
            ... def call_service() -> None:
            ...     return external_service.call()

        Using call() method::

            >>> breaker = CircuitBreaker(failure_threshold=3)
            >>> result = breaker.call(external_service.call, arg1, arg2)

        With exponential backoff::

            >>> breaker = CircuitBreaker(
            ...     failure_threshold=5,
            ...     enable_exponential_backoff=True,
            ...     backoff_base=2.0,
            ...     max_backoff_timeout=600.0
            ... )
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 1,
        reset_timeout: float = 60.0,
        excluded_exceptions: tuple[type[BaseException], ...] = (),
        name: str = "default",
        enable_exponential_backoff: bool = False,
        backoff_base: float = 2.0,
        max_backoff_timeout: float = 3600.0,
        enable_jitter: bool = False,
        jitter_factor: float = 0.1,
        # Backwards compatibility alias
        recovery_timeout: float | None = None,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening.
            success_threshold: Successes to close from half-open.
            reset_timeout: Seconds before trying half-open.
            excluded_exceptions: Exceptions that don't trigger failure.
            name: Name for this circuit breaker.
            enable_exponential_backoff: Enable exponential backoff on repeated failures.
            backoff_base: Base multiplier for exponential backoff (default: 2.0).
            max_backoff_timeout: Maximum backoff timeout in seconds (default: 1 hour).
            enable_jitter: Add random jitter to prevent thundering herd.
            jitter_factor: Jitter factor (0-1) for randomizing timeout.
            recovery_timeout: Alias for reset_timeout (backwards compatibility).
        """
        # Handle backwards compatibility alias
        effective_timeout = recovery_timeout if recovery_timeout is not None else reset_timeout

        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            reset_timeout=effective_timeout,
            excluded_exceptions=excluded_exceptions,
            enable_exponential_backoff=enable_exponential_backoff,
            backoff_base=backoff_base,
            max_backoff_timeout=max_backoff_timeout,
            enable_jitter=enable_jitter,
            jitter_factor=jitter_factor,
        )
        self.name = name
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._lock = threading.RLock()

        # Statistics tracking
        self._total_failures = 0
        self._total_successes = 0
        self._open_count = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            if self._state == CircuitState.OPEN and self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info(f"Circuit '{self.name}' transitioning to HALF_OPEN for testing")
            return self._state

    @property
    def open_count(self) -> int:
        """Get the number of times the circuit has opened."""
        with self._lock:
            return self._open_count

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try half-open."""
        if self._last_failure_time is None:
            return True
        current_timeout = self._get_current_timeout()
        return time.time() - self._last_failure_time >= current_timeout

    def _get_current_timeout(self) -> float:
        """Calculate current timeout with optional exponential backoff and jitter.

        Returns:
            The calculated timeout in seconds, considering exponential backoff
            and optional jitter.
        """
        if not self.config.enable_exponential_backoff or self._open_count <= 1:
            base_timeout = self.config.reset_timeout
        else:
            # Limit exponent to prevent overflow (max 10 doublings)
            exponent = min(self._open_count - 1, 10)
            base_timeout = self.config.reset_timeout * (self.config.backoff_base**exponent)
            base_timeout = min(base_timeout, self.config.max_backoff_timeout)

        if self.config.enable_jitter and base_timeout > 0:
            # Add jitter: random value between -jitter_factor and +jitter_factor
            jitter = base_timeout * self.config.jitter_factor * (2 * random.random() - 1)
            base_timeout = max(0, base_timeout + jitter)

        return base_timeout

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._total_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._open_count = 0  # Reset open count on successful recovery
                    logger.info(f"Circuit '{self.name}' closed after recovery")
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                self._failure_count = 0

    def _record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._total_failures += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._open_count += 1
                logger.warning(f"Circuit '{self.name}' re-opened after test failure")
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.config.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._open_count += 1
                logger.warning(
                    f"Circuit '{self.name}' opened after {self._failure_count} failures "
                    f"(open count: {self._open_count})"
                )

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call function with circuit breaker protection.

        This is an alternative to the decorator pattern for cases where
        you want to protect a call without decorating the function.

        Args:
            func: Function to call.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the function call.

        Raises:
            CircuitBreakerOpenError: If the circuit is open.
            Exception: Any exception raised by the function.

        Example:
            >>> breaker = CircuitBreaker()
            >>> result = breaker.call(requests.get, 'https://api.example.com')
        """
        if self.state == CircuitState.OPEN:
            current_timeout = self._get_current_timeout()
            raise CircuitBreakerOpenError(
                f"Circuit '{self.name}' is open, call rejected "
                f"(timeout: {current_timeout:.1f}s)"
            )

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.config.excluded_exceptions:
            raise
        except Exception:
            self._record_failure()
            raise

    def __call__(self, func: F) -> F:
        """Decorator to wrap function with circuit breaker.

        Args:
            func: Function to wrap.

        Returns:
            Wrapped function.

        Example:
            >>> breaker = CircuitBreaker()
            >>> @breaker
            ... def my_function():
            ...     return external_service.call()
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.call(func, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    def reset(self) -> None:
        """Manually reset the circuit breaker.

        This resets the circuit to CLOSED state and clears all failure/success
        counts, but preserves total statistics.
        """
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._open_count = 0
            logger.info(f"Circuit '{self.name}' manually reset to CLOSED")

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with circuit breaker statistics including:
            - state: Current circuit state
            - failure_count: Current consecutive failure count
            - success_count: Current consecutive success count (in half-open)
            - open_count: Number of times circuit has opened
            - total_failures: Total failures since creation
            - total_successes: Total successes since creation
            - current_timeout: Current calculated timeout
            - exponential_backoff_enabled: Whether backoff is enabled
            - jitter_enabled: Whether jitter is enabled
            - last_failure_time: Timestamp of last failure (or None)

        Example:
            >>> breaker = CircuitBreaker()
            >>> stats = breaker.get_stats()
            >>> print(f"State: {stats['state']}, Failures: {stats['total_failures']}")
        """
        with self._lock:
            return {
                "state": self._state.name,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "open_count": self._open_count,
                "total_failures": self._total_failures,
                "total_successes": self._total_successes,
                "current_timeout": self._get_current_timeout(),
                "exponential_backoff_enabled": self.config.enable_exponential_backoff,
                "jitter_enabled": self.config.enable_jitter,
                "last_failure_time": self._last_failure_time,
                "name": self.name,
            }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


def retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[F], F]:
    """Decorator for retry with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts.
        backoff_factor: Multiplier for delay between retries.
        initial_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        exceptions: Exception types to retry on.
        on_retry: Callback function called on each retry.

    Returns:
        Decorator function.

    Example:
        >>> @retry(max_attempts=3, backoff_factor=2.0)
        ... def unstable_function() -> None:
        ...     return call_unreliable_service()
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception: Exception | None = None

            if max_attempts < 1:
                raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(f"Final retry attempt failed for {func.__name__}: {e}")
                        raise

                    if on_retry:
                        on_retry(e, attempt)

                    logger.warning(
                        f"Retry {attempt}/{max_attempts} for {func.__name__} "
                        f"after {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)

            raise RuntimeError(f"Retry logic error in {func.__name__}: {last_exception}")

        return wrapper  # type: ignore[return-value]

    return decorator


class GracefulShutdown:
    """Handler for graceful application shutdown.

    This class manages the graceful shutdown process, allowing
    in-flight requests to complete and resources to be cleaned up.

    Example:
        >>> shutdown = GracefulShutdown(timeout=30)
        >>> shutdown.register_handler(cleanup_database)
        >>> shutdown.register_handler(close_connections)
        >>>
        >>> while not shutdown.should_stop:
        ...     process_next_request()
    """

    def __init__(
        self,
        timeout: float = 30.0,
        signals: list[int] | None = None,
    ) -> None:
        """Initialize graceful shutdown handler.

        Args:
            timeout: Maximum time to wait for shutdown completion.
            signals: Signals to handle. Defaults to SIGTERM, SIGINT.
        """
        self.timeout = timeout
        self._should_stop = threading.Event()
        self._handlers: list[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._in_flight_count = 0
        self._shutdown_started = False

        # Register signal handlers
        signals = signals or [signal.SIGTERM, signal.SIGINT]
        for sig in signals:
            try:
                signal.signal(sig, self._signal_handler)
            except (ValueError, OSError):
                # Can't set signal handler (not main thread, etc.)
                pass

    @property
    def should_stop(self) -> bool:
        """Check if shutdown has been requested."""
        return self._should_stop.is_set()

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.initiate_shutdown()

    def register_handler(self, handler: Callable[[], None]) -> None:
        """Register a cleanup handler.

        Args:
            handler: Function to call during shutdown.
        """
        with self._lock:
            self._handlers.append(handler)

    @contextmanager
    def track_request(self) -> Generator[Any, None, None]:
        """Context manager to track in-flight requests.

        Yields:
            None

        Example:
            >>> with shutdown.track_request():
            ...     handle_request()
        """
        with self._lock:
            if self._shutdown_started:
                raise ShutdownInProgressError("Cannot accept new requests during shutdown")
            self._in_flight_count += 1

        try:
            yield
        finally:
            with self._lock:
                self._in_flight_count -= 1

    def initiate_shutdown(self) -> None:
        """Start the shutdown process."""
        with self._lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True

        self._should_stop.set()
        logger.info("Shutdown initiated, waiting for in-flight requests")

        # Wait for in-flight requests
        start_time = time.time()
        while self._in_flight_count > 0:
            if time.time() - start_time > self.timeout:
                logger.warning(
                    f"Shutdown timeout, {self._in_flight_count} requests still in flight"
                )
                break
            time.sleep(0.1)

        # Run cleanup handlers
        logger.info(f"Running {len(self._handlers)} shutdown handlers")
        for handler in reversed(self._handlers):
            try:
                handler()
            except Exception as e:
                logger.error(f"Shutdown handler error: {e}")

        logger.info("Graceful shutdown complete")

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        """Wait for shutdown signal.

        Args:
            timeout: Maximum time to wait (None = wait forever).

        Returns:
            True if shutdown was signaled, False if timeout.
        """
        return self._should_stop.wait(timeout)


class ShutdownInProgressError(Exception):
    """Raised when operation rejected due to shutdown."""

    pass


class Bulkhead:
    """Bulkhead pattern for resource isolation.

    Limits concurrent access to a resource to prevent
    one component from exhausting shared resources.

    Example:
        >>> bulkhead = Bulkhead(max_concurrent=10)
        >>> with bulkhead.acquire():
        ...     use_shared_resource()
    """

    def __init__(
        self,
        max_concurrent: int,
        max_waiting: int = 100,
        timeout: float = 30.0,
        name: str = "default",
    ) -> None:
        """Initialize bulkhead.

        Args:
            max_concurrent: Maximum concurrent executions.
            max_waiting: Maximum waiting requests.
            timeout: Timeout for acquiring slot.
            name: Name for this bulkhead.
        """
        self.max_concurrent = max_concurrent
        self.max_waiting = max_waiting
        self.timeout = timeout
        self.name = name
        self._semaphore = threading.Semaphore(max_concurrent)
        self._waiting_count = 0
        self._lock = threading.Lock()

    @contextmanager
    def acquire(self, timeout: float | None = None) -> Generator[Any, None, None]:
        """Acquire a slot in the bulkhead.

        Args:
            timeout: Override default timeout.

        Yields:
            None

        Raises:
            BulkheadFullError: If bulkhead is at capacity.
        """
        timeout = timeout if timeout is not None else self.timeout

        with self._lock:
            if self._waiting_count >= self.max_waiting:
                raise BulkheadFullError(f"Bulkhead '{self.name}' queue full ({self.max_waiting})")
            self._waiting_count += 1

        try:
            acquired = self._semaphore.acquire(timeout=timeout)
            if not acquired:
                raise BulkheadFullError(f"Bulkhead '{self.name}' timeout waiting for slot")

            with self._lock:
                self._waiting_count -= 1

            try:
                yield
            finally:
                self._semaphore.release()
        except BulkheadFullError:
            with self._lock:
                self._waiting_count -= 1
            raise


class BulkheadFullError(Exception):
    """Raised when bulkhead is at capacity."""

    pass


@dataclass
class HealthStatus:
    """Health check status."""

    healthy: bool
    name: str
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class HealthChecker:
    """Health check manager for service health monitoring.

    Example:
        >>> checker = HealthChecker()
        >>> checker.add_check("database", check_database)
        >>> checker.add_check("cache", check_cache)
        >>> status = checker.check_all()
    """

    def __init__(self) -> None:
        """Initialize health checker."""
        self._checks: dict[str, Callable[[], HealthStatus]] = {}

    def add_check(
        self,
        name: str,
        check_fn: Callable[[], HealthStatus],
    ) -> None:
        """Add a health check.

        Args:
            name: Name of the check.
            check_fn: Function that returns HealthStatus.
        """
        self._checks[name] = check_fn

    def check(self, name: str) -> HealthStatus:
        """Run a specific health check.

        Args:
            name: Name of the check.

        Returns:
            Health status.
        """
        if name not in self._checks:
            return HealthStatus(
                healthy=False,
                name=name,
                message=f"Unknown health check: {name}",
            )

        try:
            return self._checks[name]()
        except Exception as e:
            return HealthStatus(
                healthy=False,
                name=name,
                message=f"Check failed: {e}",
            )

    def check_all(self) -> dict[str, HealthStatus]:
        """Run all health checks.

        Returns:
            Dictionary of check names to statuses.
        """
        results = {}
        for name in self._checks:
            results[name] = self.check(name)
        return results

    def is_healthy(self) -> bool:
        """Check if all checks pass.

        Returns:
            True if all checks are healthy.
        """
        return all(status.healthy for status in self.check_all().values())


def timeout(seconds: float) -> Callable[[F], F]:
    """Decorator to add timeout to a function.

    Args:
        seconds: Timeout in seconds.

    Returns:
        Decorator function.

    Example:
        >>> @timeout(30.0)
        ... def long_operation() -> None:
        ...     return process_data()
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future: Future[Any] = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except TimeoutError:
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")

        return wrapper  # type: ignore[return-value]

    return decorator
