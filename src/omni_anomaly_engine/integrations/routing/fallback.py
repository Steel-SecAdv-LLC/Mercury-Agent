"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Fallback handler chain for graceful degradation.

Example:
    Basic fallback chain::

        from omni_anomaly_engine.integrations.routing import FallbackChain, FallbackHandler

        chain = FallbackChain()

        @chain.handler(priority=0)
        async def primary_handler(request):
            return await external_service.call()

        @chain.handler(priority=1)
        async def cached_fallback(request):
            return cache.get(request.key)

        @chain.handler(priority=2)
        async def default_fallback(request):
            return {"status": "degraded", "data": default_response}

        # Execute with automatic fallback
        result = await chain.execute(request)
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class FallbackReason(Enum):
    """Reasons for falling back to next handler."""

    SUCCESS = "success"
    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"
    VALIDATION_FAILED = "validation_failed"
    SKIP = "skip"


@dataclass
class FallbackResult[T]:
    """Result from fallback chain execution.

    Attributes:
        value: The result value (if successful).
        handler_name: Name of handler that produced result.
        fallback_count: Number of handlers tried before success.
        reasons: List of reasons for each fallback.
        elapsed: Total time taken.
        degraded: Whether result is from a fallback handler.
    """

    value: T | None
    handler_name: str
    fallback_count: int
    reasons: list[tuple[str, FallbackReason, str]]
    elapsed: float
    degraded: bool = False

    @property
    def successful(self) -> bool:
        """Check if execution produced a result."""
        return self.value is not None


@dataclass
class FallbackHandler:
    """Individual fallback handler.

    Attributes:
        name: Handler name for logging/metrics.
        handler: Async function to execute.
        priority: Lower priority executes first.
        timeout: Timeout in seconds (None = no timeout).
        condition: Optional condition function to check before execution.
        on_error: Optional callback on error.
    """

    name: str
    handler: Callable[..., Awaitable[Any]]
    priority: int = 0
    timeout: float | None = None
    condition: Callable[..., bool] | None = None
    on_error: Callable[[Exception], None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Metrics
    call_count: int = field(default=0, repr=False)
    success_count: int = field(default=0, repr=False)
    error_count: int = field(default=0, repr=False)

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the handler.

        Args:
            *args: Positional arguments for handler.
            **kwargs: Keyword arguments for handler.

        Returns:
            Handler result.

        Raises:
            asyncio.TimeoutError: If handler times out.
            Exception: Any exception from handler.
        """
        self.call_count += 1

        try:
            if self.timeout:
                result = await asyncio.wait_for(
                    self.handler(*args, **kwargs),
                    timeout=self.timeout,
                )
            else:
                result = await self.handler(*args, **kwargs)

            self.success_count += 1
            return result

        except Exception as e:
            self.error_count += 1
            if self.on_error:
                self.on_error(e)
            raise

    def should_execute(self, *args: Any, **kwargs: Any) -> bool:
        """Check if handler should execute.

        Args:
            *args: Positional arguments to pass to condition.
            **kwargs: Keyword arguments to pass to condition.

        Returns:
            True if handler should execute.
        """
        if self.condition is None:
            return True
        return self.condition(*args, **kwargs)

    def get_metrics(self) -> dict[str, Any]:
        """Get handler metrics."""
        return {
            "name": self.name,
            "priority": self.priority,
            "call_count": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": (self.success_count / self.call_count if self.call_count > 0 else 0.0),
        }


class FallbackError(Exception):
    """Raised when all fallback handlers fail."""

    def __init__(
        self,
        message: str,
        errors: list[tuple[str, Exception]],
    ):
        super().__init__(message)
        self.errors = errors


class FallbackChain:
    """Chain of fallback handlers for graceful degradation.

    Features:
    - Priority-based handler execution
    - Automatic fallback on failure
    - Timeout support per handler
    - Conditional execution
    - Comprehensive metrics

    Example:
        >>> chain = FallbackChain()
        >>>
        >>> @chain.handler(priority=0, timeout=5.0)
        ... async def primary(data):
        ...     return await api.fetch(data)
        >>>
        >>> @chain.handler(priority=1)
        ... async def cache_fallback(data):
        ...     return cache.get(data.key)
        >>>
        >>> result = await chain.execute(data)
    """

    def __init__(
        self,
        name: str = "default",
        fail_fast: bool = False,
    ):
        """Initialize fallback chain.

        Args:
            name: Chain name for logging.
            fail_fast: If True, raise immediately on first failure.
        """
        self.name = name
        self.fail_fast = fail_fast
        self._handlers: list[FallbackHandler] = []
        self._execution_count = 0
        self._fallback_count = 0

    def add_handler(
        self,
        handler: Callable[..., Awaitable[Any]],
        name: str | None = None,
        priority: int = 0,
        timeout: float | None = None,
        condition: Callable[..., bool] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        **metadata: Any,
    ) -> FallbackHandler:
        """Add a handler to the chain.

        Args:
            handler: Async handler function.
            name: Handler name (defaults to function name).
            priority: Execution priority (lower = first).
            timeout: Handler timeout in seconds.
            condition: Condition function for conditional execution.
            on_error: Error callback.
            **metadata: Additional metadata.

        Returns:
            Created FallbackHandler.
        """
        fb_handler = FallbackHandler(
            name=name or handler.__name__,
            handler=handler,
            priority=priority,
            timeout=timeout,
            condition=condition,
            on_error=on_error,
            metadata=metadata,
        )
        self._handlers.append(fb_handler)
        # Sort by priority
        self._handlers.sort(key=lambda h: h.priority)
        return fb_handler

    def handler(
        self,
        priority: int = 0,
        name: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Callable:
        """Decorator to add a handler.

        Args:
            priority: Execution priority.
            name: Handler name.
            timeout: Handler timeout.
            **kwargs: Additional options.

        Returns:
            Decorator function.
        """

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable:
            self.add_handler(
                func,
                name=name or func.__name__,
                priority=priority,
                timeout=timeout,
                **kwargs,
            )
            return func

        return decorator

    async def execute(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> FallbackResult[Any]:
        """Execute the fallback chain.

        Tries handlers in priority order until one succeeds.

        Args:
            *args: Arguments to pass to handlers.
            **kwargs: Keyword arguments to pass to handlers.

        Returns:
            FallbackResult with value and metadata.

        Raises:
            FallbackError: If all handlers fail (and fail_fast=False).
        """
        self._execution_count += 1
        start_time = time.time()
        reasons: list[tuple[str, FallbackReason, str]] = []
        errors: list[tuple[str, Exception]] = []
        fallback_count = 0

        for handler in self._handlers:
            # Check condition
            if not handler.should_execute(*args, **kwargs):
                reasons.append((handler.name, FallbackReason.SKIP, "condition failed"))
                continue

            try:
                result = await handler.execute(*args, **kwargs)

                elapsed = time.time() - start_time
                degraded = fallback_count > 0

                if degraded:
                    self._fallback_count += 1
                    logger.warning(
                        f"Chain '{self.name}' degraded to '{handler.name}' "
                        f"after {fallback_count} fallbacks"
                    )

                return FallbackResult(
                    value=result,
                    handler_name=handler.name,
                    fallback_count=fallback_count,
                    reasons=reasons,
                    elapsed=elapsed,
                    degraded=degraded,
                )

            except TimeoutError as e:
                fallback_count += 1
                reason_msg = f"timeout after {handler.timeout}s"
                reasons.append((handler.name, FallbackReason.TIMEOUT, reason_msg))
                errors.append((handler.name, e))
                logger.warning(f"Handler '{handler.name}' timed out: {reason_msg}")

                if self.fail_fast:
                    raise FallbackError(
                        f"Handler '{handler.name}' timed out",
                        errors=errors,
                    ) from e

            except Exception as e:
                fallback_count += 1
                reason_msg = str(e)
                reasons.append((handler.name, FallbackReason.EXCEPTION, reason_msg))
                errors.append((handler.name, e))
                logger.warning(f"Handler '{handler.name}' failed: {reason_msg}")

                if self.fail_fast:
                    raise FallbackError(
                        f"Handler '{handler.name}' failed: {e}",
                        errors=errors,
                    ) from e

        # All handlers failed
        elapsed = time.time() - start_time
        error_msg = f"All {len(self._handlers)} handlers in chain '{self.name}' failed"
        logger.error(error_msg)

        raise FallbackError(error_msg, errors=errors)

    def get_handlers(self) -> list[FallbackHandler]:
        """Get all handlers in priority order."""
        return list(self._handlers)

    def get_metrics(self) -> dict[str, Any]:
        """Get chain metrics.

        Returns:
            Dictionary with execution counts and handler metrics.
        """
        return {
            "name": self.name,
            "execution_count": self._execution_count,
            "fallback_count": self._fallback_count,
            "fallback_rate": (
                self._fallback_count / self._execution_count if self._execution_count > 0 else 0.0
            ),
            "handlers": [h.get_metrics() for h in self._handlers],
        }

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._execution_count = 0
        self._fallback_count = 0
        for handler in self._handlers:
            handler.call_count = 0
            handler.success_count = 0
            handler.error_count = 0


class FallbackRegistry:
    """Registry of fallback chains.

    Allows centralized management of fallback chains across the application.

    Example:
        >>> registry = FallbackRegistry()
        >>> registry.register("user_service", user_chain)
        >>> registry.register("data_service", data_chain)
        >>>
        >>> result = await registry.execute("user_service", user_id)
    """

    def __init__(self) -> None:
        """Initialize registry."""
        self._chains: dict[str, FallbackChain] = {}

    def register(
        self,
        name: str,
        chain: FallbackChain | None = None,
    ) -> FallbackChain:
        """Register a fallback chain.

        Args:
            name: Chain name.
            chain: Existing chain or None to create new.

        Returns:
            Registered chain.
        """
        if chain is None:
            chain = FallbackChain(name=name)
        self._chains[name] = chain
        return chain

    def get(self, name: str) -> FallbackChain | None:
        """Get chain by name."""
        return self._chains.get(name)

    async def execute(
        self,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> FallbackResult[Any]:
        """Execute a named chain.

        Args:
            name: Chain name.
            *args: Arguments for handlers.
            **kwargs: Keyword arguments for handlers.

        Returns:
            FallbackResult.

        Raises:
            KeyError: If chain not found.
        """
        chain = self._chains.get(name)
        if not chain:
            raise KeyError(f"Fallback chain not found: {name}")
        return await chain.execute(*args, **kwargs)

    def get_all_metrics(self) -> dict[str, Any]:
        """Get metrics for all chains."""
        return {name: chain.get_metrics() for name, chain in self._chains.items()}
