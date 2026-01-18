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

"""REST API for Mercury Agent ♱ anomaly detection."""

from .auth import APIKeyAuth, JWTAuth, Permission, User, require_permission, require_role
from .health import HealthChecker, get_health_checker, health_router
from .server import app
from .voice import (
    add_voice_routes,
    router as voice_router,
)

__all__ = [
    # Auth
    "APIKeyAuth",
    "HealthChecker",
    "JWTAuth",
    "Permission",
    "User",
    "add_voice_routes",
    "app",
    "get_health_checker",
    # Health
    "health_router",
    "require_permission",
    "require_role",
    # Voice
    "voice_router",
]
