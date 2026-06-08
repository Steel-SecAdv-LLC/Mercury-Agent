# Copyright (C) 2025 Steel Security Advisors LLC
"""Retry policy implementation."""

from __future__ import annotations

import time
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class RetryPolicy:
    """Retry policy with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        exceptions: tuple[type[BaseException], ...] = (Exception,),
    ):
        """Initialize the instance."""
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.exceptions = exceptions

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to apply retry policy."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except self.exceptions as e:
                    last_exception = e

                    if attempt < self.max_retries:
                        delay = min(
                            self.base_delay * (self.exponential_base**attempt),
                            self.max_delay,
                        )
                        time.sleep(delay)

            if last_exception is not None:
                raise last_exception

        return wrapper
