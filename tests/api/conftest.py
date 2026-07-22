# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared fixtures for the API test suite.

The account routes keep two process-wide singletons (the auth service and the
per-action rate limiter). Tests override the auth service through FastAPI's
dependency system, but the action limiter is called directly by the handlers,
so a counter filled by one test would bleed 429s into the next. The autouse
fixture below resets it around every test, giving each a fresh in-memory
counter store (``MERCURY_KEYSTORE_PATH`` is unset under tests).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.api.routes import accounts

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _fresh_action_limiter() -> Iterator[None]:
    """Give every test its own per-action rate-limit counters."""
    accounts._action_limiter = None
    yield
    accounts._action_limiter = None
