# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""W3C Trace Context (``traceparent`` / ``tracestate``) for the API.

Lets Mercury take part in a distributed trace without a tracing SDK: an
incoming ``traceparent`` (from an instrumented gateway or client) is
validated per the W3C Trace Context recommendation and continued -- same
trace-id, a fresh span-id for this request, the caller's flags -- and the
response carries the resulting ``traceparent``. ``tracestate`` is passed
through only when it satisfies the recommendation's grammar and limits (at
most 32 list members, 512 characters, unique keys); otherwise it is dropped,
so arbitrary client bytes are never echoed into a response header or carried
on the request's trace. Without a valid ``traceparent`` a new trace is
started. The ids
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
    "parse_tracestate",
    "trace_context_ctx",
]

_TRACEPARENT = re.compile(
    r"^(?P<version>[0-9a-f]{2})-(?P<trace_id>[0-9a-f]{32})-"
    r"(?P<parent_id>[0-9a-f]{16})-(?P<flags>[0-9a-f]{2})(?P<rest>(-.*)?)$"
)
#: ``key=value`` list member (W3C Trace Context section 3.3.1.3): a
#: ``simple-key`` or a ``tenant@system`` multi-tenant key, and a value of up
#: to 256 printable ASCII characters (no ``,`` or ``=``) that does not end in
#: a space.
_TRACESTATE_MEMBER = re.compile(
    r"(?P<key>[a-z][a-z0-9_\-*/]{0,255}"
    r"|[a-z0-9][a-z0-9_\-*/]{0,240}@[a-z][a-z0-9_\-*/]{0,13})"
    r"=(?P<value>[\x20-\x2b\x2d-\x3c\x3e-\x7e]{0,255}[\x21-\x2b\x2d-\x3c\x3e-\x7e])"
)
#: Recommendation limits: at most 32 list members; vendors must propagate at
#: least 512 characters, and anything longer is dropped here.
TRACESTATE_MAX_MEMBERS = 32
TRACESTATE_MAX_CHARS = 512
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


def parse_tracestate(value: str | None) -> str | None:
    """Return a normalised ``tracestate`` if *value* is valid, else ``None``.

    Applies the W3C Trace Context ``tracestate`` grammar: comma-separated
    ``key=value`` members with optional surrounding spaces/tabs, empty
    members ignored, at most :data:`TRACESTATE_MAX_MEMBERS` members, unique
    keys, and -- the propagation floor this service honours as a ceiling --
    at most :data:`TRACESTATE_MAX_CHARS` characters. A header that breaks
    any rule is dropped entirely rather than repaired, as the recommendation
    requires for an unparseable list.

    Args:
        value: The (possibly combined) incoming header value.

    Returns:
        The members joined by ``","`` without optional whitespace, or
        ``None`` when the header is absent, empty or invalid.
    """
    if not value or len(value) > TRACESTATE_MAX_CHARS:
        return None
    members: list[str] = []
    keys: set[str] = set()
    for raw in value.split(","):
        member = raw.strip(" \t")
        if not member:
            continue
        match = _TRACESTATE_MEMBER.fullmatch(member)
        if match is None or match.group("key") in keys:
            return None
        keys.add(match.group("key"))
        members.append(member)
        if len(members) > TRACESTATE_MAX_MEMBERS:
            return None
    return ",".join(members) or None


def _new_id(n_bytes: int) -> str:
    while True:
        value = secrets.token_hex(n_bytes)
        if value.strip("0"):
            return value


def continue_or_start(traceparent: str | None, tracestate: str | None = None) -> TraceContext:
    """Continue the caller's trace when *traceparent* is valid, else start one.

    A new span-id is minted either way; ``tracestate`` is kept only when the
    trace is continued (it belongs to the caller's trace) and it passes
    :func:`parse_tracestate`.
    """
    parsed = parse_traceparent(traceparent)
    if parsed is None:
        return TraceContext(_new_id(16), _new_id(8), None, "00")
    trace_id, parent_id, flags = parsed
    return TraceContext(trace_id, _new_id(8), parent_id, flags, parse_tracestate(tracestate))


def current_trace() -> TraceContext | None:
    """The trace context of the request being served, if any."""
    return trace_context_ctx.get()
