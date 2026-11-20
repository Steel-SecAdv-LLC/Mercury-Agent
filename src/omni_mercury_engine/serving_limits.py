# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The served API's resource bounds: one source of truth, operator-visible.

Every bound the serving path enforces — request body size, flagship queue
depth and field caps, live alert-stream subscribers, the IPv6 rate-limit
prefix, the token-bucket rate — used to be a literal or an independent
``os.environ`` read in whichever module enforced it.  Ten values across five
files, each with its own default, and no way for an operator to see the
active numbers without reading source.  Worse, the OpenAPI document
*hardcoded* ``100/min, burst 20`` while the real limiter read
``OMNI_RATE_LIMIT_*`` from the environment, so an operator who tuned the
limit was shown the wrong one.

This module is the single place those bounds are defined, read and
validated.  Every enforcing site reads its value from :func:`serving_limits`;
the OpenAPI ``info.x-serving-limits`` extension and
``GET /api/v1/detect/flagship/limits`` render the same object, so what is
advertised is, by construction, what is enforced.

Each value is read once from its environment variable (falling back to the
documented default) and validated eagerly: a non-integer or out-of-range
value raises :class:`ServingLimitConfigError` naming the variable, the loud
failure ``_env.py`` uses for an unknown ``MERCURY_ENV`` — a typo'd bound
must not silently fall back to a default.

The module is import-safe: stdlib only, no third-party imports.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Limit",
    "ServingLimitConfigError",
    "ServingLimits",
    "load_serving_limits",
    "serving_limits",
]


class ServingLimitConfigError(ValueError):
    """A serving-limit environment variable holds an unusable value."""


@dataclass(frozen=True)
class Limit:
    """One enforced bound, as an operator sees it.

    Attributes:
        name: Stable key (also the ``ServingLimits`` attribute name).
        env: The environment variable that overrides it.
        default: The value used when ``env`` is unset.
        value: The active value this process enforces.
        unit: What the number counts (``bytes``, ``rows``, ...).
        description: What the bound protects against.
    """

    name: str
    env: str
    default: int
    value: int
    unit: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view (the shape the limits endpoint and OpenAPI serve)."""
        return {
            "value": self.value,
            "default": self.default,
            "env": self.env,
            "unit": self.unit,
            "description": self.description,
            "overridden": self.value != self.default,
        }


# (name, env var, default, minimum, maximum, unit, description). The minimum
# for the two released OMNI_RATE_LIMIT_* variables is 0, preserving their
# released semantics; every bound introduced alongside this module requires
# at least 1, since a zero cap would refuse all traffic rather than bound it.
_SPECS: tuple[tuple[str, str, int, int, int | None, str, str], ...] = (
    (
        "max_body_bytes",
        "MERCURY_MAX_BODY_BYTES",
        10 * 1024 * 1024,
        1,
        None,
        "bytes",
        "request body cap on every route, enforced before the body is parsed "
        "(declared Content-Length refused up front; chunked bodies counted)",
    ),
    (
        "max_upload_body_bytes",
        "MERCURY_MAX_UPLOAD_BODY_BYTES",
        600 * 1024 * 1024,
        1,
        None,
        "bytes",
        "request body cap on the authenticated /api/v1/models surface, whose "
        "upload handler enforces its own 500 MB streaming bound",
    ),
    (
        "flagship_max_queue",
        "MERCURY_FLAGSHIP_MAX_QUEUE",
        8,
        1,
        None,
        "requests",
        "flagship detections admitted concurrently (running or waiting on the "
        "single engine lock); beyond this a caller is refused with 429 rather "
        "than parking a threadpool thread",
    ),
    (
        "flagship_max_rows",
        "MERCURY_FLAGSHIP_MAX_ROWS",
        65536,
        1,
        None,
        "rows",
        "rows per /detect/flagship request",
    ),
    (
        "flagship_max_domain_chars",
        "MERCURY_FLAGSHIP_MAX_DOMAIN_CHARS",
        128,
        1,
        None,
        "characters",
        "length of the /detect/flagship 'domain' field, which rides into "
        "DecisionRecord.domain and the decision ledger",
    ),
    (
        "flagship_max_subject_id_chars",
        "MERCURY_FLAGSHIP_MAX_SUBJECT_ID_CHARS",
        256,
        1,
        None,
        "characters",
        "length of the /detect/flagship 'subject_id' field",
    ),
    (
        "alert_bus_max_subscribers",
        "MERCURY_ALERT_BUS_MAX_SUBSCRIBERS",
        64,
        1,
        None,
        "streams",
        "concurrent /api/v1/voice/alerts SSE subscribers; at capacity a new "
        "stream is refused with 503",
    ),
    (
        "ipv6_rate_limit_prefix",
        "MERCURY_IPV6_RATE_LIMIT_PREFIX",
        64,
        1,
        128,
        "prefix length",
        "IPv6 addresses sharing this many leading bits share one rate-limit "
        "identity (/64 is the unit providers delegate to one subscriber)",
    ),
    (
        "rate_limit_requests_per_minute",
        "OMNI_RATE_LIMIT_REQUESTS_PER_MINUTE",
        100,
        0,
        None,
        "requests/minute",
        "token-bucket refill rate per client identity",
    ),
    (
        "rate_limit_burst",
        "OMNI_RATE_LIMIT_BURST",
        20,
        0,
        None,
        "requests",
        "token-bucket burst size per client identity",
    ),
)


def _read(env: str, default: int, minimum: int, maximum: int | None) -> int:
    raw = os.environ.get(env)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise ServingLimitConfigError(
            f"{env}={raw!r} is not an integer; unset it to use the default ({default})"
        ) from None
    if value < minimum or (maximum is not None and value > maximum):
        bound = f">= {minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise ServingLimitConfigError(
            f"{env}={value} is out of range (must be {bound}); unset it to use "
            f"the default ({default})"
        )
    return value


@dataclass(frozen=True)
class ServingLimits:
    """Every serving bound, typed, plus the operator-facing :class:`Limit` view."""

    max_body_bytes: int
    max_upload_body_bytes: int
    flagship_max_queue: int
    flagship_max_rows: int
    flagship_max_domain_chars: int
    flagship_max_subject_id_chars: int
    alert_bus_max_subscribers: int
    ipv6_rate_limit_prefix: int
    rate_limit_requests_per_minute: int
    rate_limit_burst: int

    def limits(self) -> list[Limit]:
        """The operator-facing view of every bound, in spec order."""
        return [
            Limit(
                name=name,
                env=env,
                default=default,
                value=getattr(self, name),
                unit=unit,
                description=description,
            )
            for name, env, default, _minimum, _maximum, unit, description in _SPECS
        ]

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Map every limit's name to its operator view (what the endpoint and OpenAPI render)."""
        return {limit.name: limit.to_dict() for limit in self.limits()}


def load_serving_limits() -> ServingLimits:
    """Read and validate every bound from the environment, uncached.

    Raises:
        ServingLimitConfigError: A variable is set to a non-integer or an
            out-of-range value.
    """
    values = {
        name: _read(env, default, minimum, maximum)
        for name, env, default, minimum, maximum, _unit, _description in _SPECS
    }
    return ServingLimits(**values)


@functools.lru_cache(maxsize=1)
def serving_limits() -> ServingLimits:
    """The process's active serving limits, read once.

    Every enforcing site and every surface that advertises a bound reads this
    one object.  Cached because several bounds are bound at import (pydantic
    field caps, the admission semaphore); a value changed in the environment
    after startup takes effect on restart, exactly as before this module.
    """
    return load_serving_limits()
