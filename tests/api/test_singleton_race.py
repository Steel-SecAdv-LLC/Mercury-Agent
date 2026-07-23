# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Concurrency tests for the lazily-built process-wide singletons.

A double-checked-locking bug on first build lets two threads each construct a
singleton; with the in-memory backends the loser's instance (and any state
written to it) is silently discarded when the winner is later returned. These
tests hammer the first call from many threads and assert exactly one instance
is ever handed out.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

# Classic TypeVar (not PEP 695 ``[T]`` syntax): this codebase runs on
# Python 3.11, where the parameter-list generic form is a SyntaxError.
T = TypeVar("T")


def _hammer_first_build(  # noqa: UP047 - PEP 695 unavailable on the 3.11 runtime
    reset: Callable[[], None], getter: Callable[[], T]
) -> set[int]:
    """Reset the singleton, then call ``getter`` from many threads at once.

    Returns the set of distinct object ids observed — a correct singleton
    yields exactly one.
    """
    reset()
    barrier = threading.Barrier(24)
    seen: list[T] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()  # maximise the race on the first call
        instance = getter()
        with lock:
            seen.append(instance)

    threads = [threading.Thread(target=worker) for _ in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return {id(x) for x in seen}


def test_rate_limiter_singleton_is_unique() -> None:
    """get_rate_limiter hands out one instance under a first-call stampede."""
    from omni_mercury_engine.security import rate_limiting

    ids = _hammer_first_build(rate_limiting.reset_default_limiter, rate_limiting.get_rate_limiter)
    assert len(ids) == 1


def test_api_key_store_singleton_is_unique(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_api_key_store hands out one instance under a first-call stampede."""
    monkeypatch.delenv("MERCURY_KEYSTORE_PATH", raising=False)
    from omni_mercury_engine.api import auth

    def reset() -> None:
        auth._api_key_store = None

    ids = _hammer_first_build(reset, auth.get_api_key_store)
    assert len(ids) == 1


def test_auth_service_singleton_is_unique(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_auth_service hands out one instance under a first-call stampede."""
    monkeypatch.delenv("MERCURY_KEYSTORE_PATH", raising=False)
    from omni_mercury_engine.api.routes import accounts

    def reset() -> None:
        accounts._service = None

    ids = _hammer_first_build(reset, accounts.get_auth_service)
    assert len(ids) == 1


def test_action_limiter_singleton_is_unique(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_action_limiter hands out one instance under a first-call stampede."""
    monkeypatch.delenv("MERCURY_KEYSTORE_PATH", raising=False)
    from omni_mercury_engine.api.routes import accounts

    def reset() -> None:
        accounts._action_limiter = None

    ids = _hammer_first_build(reset, accounts.get_action_limiter)
    assert len(ids) == 1


def test_audit_logger_singleton_is_unique() -> None:
    """get_audit_logger hands out one instance under a first-call stampede."""
    from omni_mercury_engine.security import secure_audit_logging as audit

    def reset() -> None:
        audit._audit_logger = None

    ids = _hammer_first_build(reset, audit.get_audit_logger)
    assert len(ids) == 1
