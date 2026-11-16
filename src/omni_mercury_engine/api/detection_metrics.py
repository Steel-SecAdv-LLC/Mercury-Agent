# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Record the detection metrics for API routes that detect without the engine.

``omni_detection_requests_total``, ``omni_detection_success_total`` and
``omni_detection_duration_seconds`` are written by
:func:`omni_mercury_engine.core.metrics.time_detection`, which the engine
calls around each of its detectors. Of the eight detection routes only
``/api/v1/detect/flagship`` runs the engine. The other seven -- univariate,
multivariate, fusion, tier, neurosymbolic, three-r and rca -- and the batch
worker compute in the route itself, so a request to any of them answered 200
and moved none of the three series. Measured against a running server with
Prometheus scraping it: after a successful call to each route the "Detections
by Type" and "Detection Processing Time" panels had no data, and
``OmniMercuryDetectionSuccessRateDrop`` and ``OmniMercurySlowDetection`` had
nothing to evaluate. Those alerts could not fire for the routes most clients
call.

:func:`records_detection` decorates a route handler and writes the same three
series for it, under a ``detector_type`` prefixed ``api_`` so a route never
shares a label with an engine detector of the same name (``neurosymbolic`` is
both).

What counts follows :func:`time_detection`:

* a handler that returns counts a request and a success;
* a handler that raises counts a request without a success, so
  ``success / requests`` is the route's detection success rate;
* a request the handler refuses -- an ``HTTPException`` below 500: invalid
  input, a missing credential, a rate limit -- is not a detection attempt and
  touches no series, as an off-modality decline touches none in the engine.
"""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

# Starlette's class, which FastAPI's ``HTTPException`` subclasses: a handler
# (or a dependency it calls) raising the base class with a 4xx is refusing the
# request just the same, and was being counted as a failed detection.
from starlette.exceptions import HTTPException

from omni_mercury_engine.core.metrics import (
    DETECTION_DURATION,
    DETECTION_REQUESTS,
    DETECTION_SUCCESS,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

__all__ = ["ROUTE_PREFIX", "records_detection"]

#: Prefix of every ``detector_type`` this module writes.
ROUTE_PREFIX = "api_"

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _attempted(detector_type: str, started: float, *, succeeded: bool) -> None:
    """Write one attempt: a request, its duration, and a success if it had one."""
    DETECTION_REQUESTS.labels(detector_type=detector_type).inc()
    # ``labels()`` alone creates the series at 0, so a route that has only
    # ever failed still exports a success series for the ratio to read.
    success = DETECTION_SUCCESS.labels(detector_type=detector_type)
    if succeeded:
        success.inc()
    DETECTION_DURATION.labels(detector_type=detector_type).observe(time.perf_counter() - started)


def records_detection(
    name: str,
    *,
    skip: Callable[[], bool] | None = None,
    failed: Callable[[Any], bool] | None = None,
) -> Callable[[Callable[_P, Coroutine[Any, Any, _R]]], Callable[_P, Coroutine[Any, Any, _R]]]:
    """Decorate an async detection handler so it writes the detection series.

    Place it under the route decorator, so the router registers the wrapped
    handler; FastAPI reads the signature of the function underneath.

    Args:
        name: The route's name, such as ``"univariate"``. The series are
            written under ``detector_type="api_<name>"``.
        skip: Called per request; when it returns true the call is not
            recorded. The server passes its warm-up flag, so the two calls it
            makes to itself at startup are not counted as traffic.
        failed: Called with the handler's return value; when it returns true
            the attempt is recorded without a success. For a handler that
            reports failure in its result instead of raising (the batch chunk
            processor catches each sample's exception), which would otherwise
            record a success for every call whatever happened inside it.

    Returns:
        The decorator.

    Raises:
        ValueError: If ``name`` is not a lower-case identifier.
    """
    if not name.isidentifier() or name != name.lower():
        raise ValueError(f"detection route name {name!r} must be a lower-case identifier")
    detector_type = ROUTE_PREFIX + name
    # Export both counters at 0 from the moment the route exists. ``rate`` and
    # ``increase`` do not count the sample that creates a series, so a route
    # whose first request failed was invisible to the per-component alert
    # until its second.
    DETECTION_REQUESTS.labels(detector_type=detector_type)
    DETECTION_SUCCESS.labels(detector_type=detector_type)

    def decorate(
        handler: Callable[_P, Coroutine[Any, Any, _R]],
    ) -> Callable[_P, Coroutine[Any, Any, _R]]:
        @functools.wraps(handler)
        async def recorded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            if skip is not None and skip():
                return await handler(*args, **kwargs)
            started = time.perf_counter()
            try:
                result = await handler(*args, **kwargs)
            except HTTPException as refusal:
                if refusal.status_code >= 500:
                    _attempted(detector_type, started, succeeded=False)
                raise
            except Exception:
                _attempted(detector_type, started, succeeded=False)
                raise
            _attempted(detector_type, started, succeeded=failed is None or not failed(result))
            return result

        return recorded

    return decorate
