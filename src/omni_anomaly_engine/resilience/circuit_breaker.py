"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Circuit breaker pattern implementation
"""

import time
from collections.abc import Callable
from enum import Enum
from typing import Any


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance with exponential backoff.

    Prevents cascading failures by opening circuit after threshold failures.
    Enhanced with exponential backoff for adaptive recovery timing, which
    increases wait time after repeated failures to prevent system overload.

    Features:
    - 3-state circuit breaker (CLOSED/OPEN/HALF_OPEN)
    - Exponential backoff with configurable base and max timeout
    - Success threshold for transitioning from HALF_OPEN to CLOSED
    - Jitter option to prevent thundering herd problem
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        enable_exponential_backoff: bool = False,
        backoff_base: float = 2.0,
        max_backoff_timeout: float = 3600.0,
        success_threshold: int = 1,
        enable_jitter: bool = False,
        jitter_factor: float = 0.1,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Base timeout in seconds before attempting reset
            expected_exception: Exception type to catch
            enable_exponential_backoff: Enable exponential backoff on repeated failures
            backoff_base: Base multiplier for exponential backoff (default: 2.0)
            max_backoff_timeout: Maximum backoff timeout in seconds (default: 1 hour)
            success_threshold: Successes needed in HALF_OPEN to close circuit
            enable_jitter: Add random jitter to prevent thundering herd
            jitter_factor: Jitter factor (0-1) for randomizing timeout
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.enable_exponential_backoff = enable_exponential_backoff
        self.backoff_base = backoff_base
        self.max_backoff_timeout = max_backoff_timeout
        self.success_threshold = success_threshold
        self.enable_jitter = enable_jitter
        self.jitter_factor = jitter_factor

        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = CircuitState.CLOSED

        self.open_count = 0
        self.half_open_success_count = 0
        self._total_failures = 0
        self._total_successes = 0

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.half_open_success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            if isinstance(e, self.expected_exception):
                self._on_failure()
            raise e

    def _on_success(self) -> None:
        """Handle successful call."""
        self._total_successes += 1

        if self.state == CircuitState.HALF_OPEN:
            self.half_open_success_count += 1
            if self.half_open_success_count >= self.success_threshold:
                self.failure_count = 0
                self.state = CircuitState.CLOSED
                self.open_count = 0
        else:
            self.failure_count = 0
            self.state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self._total_failures += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.open_count += 1

    def _should_attempt_reset(self) -> bool:
        """Check if should attempt to reset circuit."""
        if self.last_failure_time is None:
            return False

        current_timeout = self._get_current_timeout()
        return (time.time() - self.last_failure_time) >= current_timeout

    def _get_current_timeout(self) -> float:
        """Calculate current timeout with optional exponential backoff."""
        if not self.enable_exponential_backoff or self.open_count <= 1:
            base_timeout = float(self.recovery_timeout)
        else:
            exponent = min(self.open_count - 1, 10)
            base_timeout = float(self.recovery_timeout) * (self.backoff_base**exponent)
            base_timeout = min(base_timeout, self.max_backoff_timeout)

        if self.enable_jitter and base_timeout > 0:
            import random

            jitter = base_timeout * self.jitter_factor * (2 * random.random() - 1)
            base_timeout = max(0, base_timeout + jitter)

        return base_timeout

    def reset(self) -> None:
        """Manually reset circuit breaker."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        self.open_count = 0
        self.half_open_success_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with circuit breaker statistics
        """
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "open_count": self.open_count,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "current_timeout": self._get_current_timeout(),
            "exponential_backoff_enabled": self.enable_exponential_backoff,
        }
