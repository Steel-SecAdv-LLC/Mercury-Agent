# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared fixtures for the API test suite.

The account routes keep process-wide singletons (the auth service, the
per-action rate limiter, and the shared quota enforcer). Tests override the
auth service through FastAPI's dependency system, but the action limiter is
called directly by the handlers, so a counter filled by one test would bleed
429s into the next; likewise the shared quota enforcer's in-memory ledger
would carry usage across tests. The autouse fixtures below reset both around
every test, giving each a fresh in-memory backend (``MERCURY_KEYSTORE_PATH``
is unset under tests).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.api import quota
from omni_mercury_engine.api.routes import accounts

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _fresh_action_limiter() -> Iterator[None]:
    """Give every test its own per-action rate-limit counters."""
    accounts._action_limiter = None
    yield
    accounts._action_limiter = None


@pytest.fixture(autouse=True)
def _fresh_quota_enforcer() -> Iterator[None]:
    """Reset the shared quota enforcer so no in-memory ledger leaks across tests."""
    quota._shared_enforcer = None
    yield
    quota._shared_enforcer = None
