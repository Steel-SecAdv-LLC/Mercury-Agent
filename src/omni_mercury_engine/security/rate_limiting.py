"""
Mercury Agent ♱
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
Rate limiting utilities
"""

import time
from collections import defaultdict


class RateLimiter:
    """Simple rate limiter for API endpoints"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        window_start = now - self.window_seconds

        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier] if req_time > window_start
        ]

        if len(self.requests[identifier]) >= self.max_requests:
            return False

        self.requests[identifier].append(now)
        return True

    def reset(self, identifier: str) -> None:
        """Reset rate limit for identifier"""
        if identifier in self.requests:
            del self.requests[identifier]
