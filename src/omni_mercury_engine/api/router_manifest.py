# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Record which API routers the server mounted, and why any did not.

``api/server.py`` attaches each router inside a ``try``/``except ImportError``
so a missing optional dependency removes one surface instead of stopping the
process. Before this module the only trace of a removed surface was one log
line at startup: ``/health`` reported ``healthy``, the OpenAPI document was
simply shorter, and nothing a monitor scrapes said that the server was running
with part of itself missing.

The manifest gives every router one of three states:

``mounted``
    The router imported and its routes are served.
``disabled``
    Configuration left the surface off (the browser frontend and the decoy
    surface are opt-in). This is the operator's choice, not a fault.
``unavailable``
    The import failed. The server is running without routes it was built to
    serve, and the exception type and message are kept as the reason.

The states are exported on ``/metrics`` as ``omni_mercury_api_router_state``
(one series per router and state, valued ``1`` for the current state), so an
alert on ``state="unavailable"`` fires on exactly the condition that used to
be silent. ``/health`` stays the minimal, unauthenticated probe it is pinned to
be by ``tests/api/test_health_minimal.py``; the manifest never appears there.

The module holds no FastAPI import and no engine import, so recording a state
can never itself be the thing that fails.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging
    from collections.abc import Iterator

__all__ = [
    "METRIC_NAME",
    "RouterMount",
    "RouterRecord",
    "RouterState",
    "mounting",
    "render_prometheus",
    "reset_manifest",
    "router_manifest",
    "unavailable_routers",
]

#: Name of the exported state-set metric.
METRIC_NAME = "omni_mercury_api_router_state"

#: A router name is a metric label value and a log token: lower-case words
#: joined by underscores, nothing that needs escaping.
_NAME_RE = re.compile(r"[a-z][a-z0-9_]*\Z")

#: Longest failure reason kept. An exception message can carry a path or a
#: long resolver explanation; the manifest keeps enough to name the cause.
_MAX_DETAIL = 300


class RouterState(StrEnum):
    """The three outcomes of attaching one router."""

    MOUNTED = "mounted"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RouterRecord:
    """The outcome of attaching one router.

    Attributes:
        name: Stable identifier, used as the ``router`` metric label.
        label: Human-readable name used in the startup log line.
        state: Whether the router is mounted, disabled or unavailable.
        detail: Why the router is not mounted; ``None`` when it is.
    """

    name: str
    label: str
    state: RouterState
    detail: str | None = None


class RouterMount:
    """Handle yielded by :func:`mounting` for the surface being attached."""

    def __init__(self) -> None:
        """Start with no opt-out recorded."""
        self.disabled_reason: str | None = None

    def disable(self, reason: str) -> None:
        """Mark the surface as left off by configuration.

        Args:
            reason: The setting that keeps the surface off.
        """
        self.disabled_reason = reason


_RECORDS: dict[str, RouterRecord] = {}


def _record(name: str, label: str, state: RouterState, detail: str | None = None) -> None:
    """Store one router's outcome, replacing any earlier one of the same name."""
    _RECORDS[name] = RouterRecord(name=name, label=label, state=state, detail=detail)


@contextmanager
def mounting(name: str, label: str, logger: logging.Logger) -> Iterator[RouterMount]:
    """Attach one router and record whether it mounted.

    The body imports the router and includes it in the application. An
    ``ImportError`` raised there is the one failure the server tolerates: it is
    logged, recorded as ``unavailable`` and suppressed. Any other exception is
    a defect in the router and propagates, so the process does not start.

    Args:
        name: Stable identifier (``[a-z][a-z0-9_]*``).
        label: Human-readable name for the log line, such as ``Account routes``.
        logger: The server's logger, so the lines keep their existing origin.

    Yields:
        A :class:`RouterMount`; call ``disable(reason)`` on it when
        configuration leaves the surface off.

    Raises:
        ValueError: If ``name`` is not a valid identifier.
    """
    if not _NAME_RE.match(name):
        raise ValueError(f"router name {name!r} must match {_NAME_RE.pattern}")
    mount = RouterMount()
    try:
        yield mount
    except ImportError as exc:
        detail = f"{type(exc).__name__}: {exc}"[:_MAX_DETAIL]
        logger.warning("%s not available: %s", label, exc)
        _record(name, label, RouterState.UNAVAILABLE, detail)
        return
    if mount.disabled_reason is not None:
        _record(name, label, RouterState.DISABLED, mount.disabled_reason[:_MAX_DETAIL])
        return
    logger.info("%s registered", label)
    _record(name, label, RouterState.MOUNTED)


def router_manifest() -> tuple[RouterRecord, ...]:
    """Return every recorded router outcome, in the order they were attached."""
    return tuple(_RECORDS.values())


def unavailable_routers() -> tuple[RouterRecord, ...]:
    """Return the routers whose import failed."""
    return tuple(r for r in _RECORDS.values() if r.state is RouterState.UNAVAILABLE)


def reset_manifest() -> None:
    """Forget every recorded outcome (for tests that build a private app)."""
    _RECORDS.clear()


def render_prometheus() -> list[str]:
    """Render the manifest as Prometheus exposition lines.

    The metric is a state set: each router has one series per state, and the
    series for its current state is ``1`` while the others are ``0``. A router
    that changes state therefore never leaves a stale series behind, and
    ``omni_mercury_api_router_state{state="unavailable"} == 1`` selects exactly
    the routers the server is running without.

    Returns:
        The ``HELP``/``TYPE`` header and one line per router and state; empty
        when nothing has been recorded.
    """
    records = router_manifest()
    if not records:
        return []
    lines = [
        f"# HELP {METRIC_NAME} API router state "
        "(1 on the series naming the router's current state)",
        f"# TYPE {METRIC_NAME} gauge",
    ]
    for record in records:
        for state in RouterState:
            value = 1 if record.state is state else 0
            lines.append(f'{METRIC_NAME}{{router="{record.name}",state="{state.value}"}} {value}')
    return lines
