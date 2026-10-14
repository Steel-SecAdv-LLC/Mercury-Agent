# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""W3C Trace Context (``traceparent`` / ``tracestate``) for the API.

Lets Mercury take part in a distributed trace without a tracing SDK: an
incoming ``traceparent`` (from an instrumented gateway or client) is
validated per the W3C Trace Context recommendation and continued -- same
trace-id, a fresh span-id for this request, the caller's flags -- and the
response carries the resulting ``traceparent`` (``tracestate`` is passed
through unchanged). Without a valid header a new trace is started. The ids
are logged with every request and stamped on SIEM alerts, so a downstream
alert, a log line and the caller's trace join on one trace-id.

Spans are not recorded or exported here (that needs an OpenTelemetry SDK,
which is not a project dependency); the propagation contract alone is what
correlates Mercury with the rest of a traced system.
"""

from __future__ import annotations

import contextvars
import re
import secrets
from dataclasses import dataclass

__all__ = [
    "TraceContext",
    "continue_or_start",
    "current_trace",
    "parse_traceparent",
    "trace_context_ctx",
]

_TRACEPARENT = re.compile(
    r"^(?P<version>[0-9a-f]{2})-(?P<trace_id>[0-9a-f]{32})-"
    r"(?P<parent_id>[0-9a-f]{16})-(?P<flags>[0-9a-f]{2})(?P<rest>(-.*)?)$"
)
_ZERO_TRACE = "0" * 32
_ZERO_SPAN = "0" * 16


@dataclass(frozen=True)
class TraceContext:
    """The trace this request belongs to.

    Attributes:
        trace_id: 32 lowercase hex digits, shared by the whole trace.
        span_id: 16 hex digits identifying this request's span.
        parent_id: The caller's span-id, ``None`` when this request started
            the trace.
        flags: Trace flags (``"01"`` = the caller sampled the trace).
        tracestate: Vendor state passed through unchanged, if any.
    """

    trace_id: str
    span_id: str
    parent_id: str | None
    flags: str
    tracestate: str | None = None

    @property
    def traceparent(self) -> str:
        """``traceparent`` header value naming this request's span."""
        return f"00-{self.trace_id}-{self.span_id}-{self.flags}"


trace_context_ctx: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "mercury_trace_context", default=None
)


def parse_traceparent(value: str | None) -> tuple[str, str, str] | None:
    """``(trace_id, parent_id, flags)`` from a valid header, else ``None``.

    Rejects the invalid version ``ff``, all-zero ids, upper-case hex and,
    for version ``00``, any trailing fields. A higher version is parsed
    by its first four fields, as the recommendation requires.
    """
    if not value:
        return None
    match = _TRACEPARENT.match(value.strip())
    if match is None:
        return None
    version = match.group("version")
    if version == "ff" or (version == "00" and match.group("rest")):
        return None
    trace_id = match.group("trace_id")
    parent_id = match.group("parent_id")
    if trace_id == _ZERO_TRACE or parent_id == _ZERO_SPAN:
        return None
    return trace_id, parent_id, match.group("flags")


def _new_id(n_bytes: int) -> str:
    while True:
        value = secrets.token_hex(n_bytes)
        if value.strip("0"):
            return value


def continue_or_start(traceparent: str | None, tracestate: str | None = None) -> TraceContext:
    """Continue the caller's trace when *traceparent* is valid, else start one.

    A new span-id is minted either way; ``tracestate`` is kept only when the
    trace is continued (it belongs to the caller's trace).
    """
    parsed = parse_traceparent(traceparent)
    if parsed is None:
        return TraceContext(_new_id(16), _new_id(8), None, "00")
    trace_id, parent_id, flags = parsed
    return TraceContext(trace_id, _new_id(8), parent_id, flags, tracestate or None)


def current_trace() -> TraceContext | None:
    """The trace context of the request being served, if any."""
    return trace_context_ctx.get()
