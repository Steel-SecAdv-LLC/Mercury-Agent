# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""REST API for Mercury Agent anomaly detection."""

from __future__ import annotations

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
