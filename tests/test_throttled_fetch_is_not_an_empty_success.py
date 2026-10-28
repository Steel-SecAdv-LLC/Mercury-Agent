# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A throttled fetch is reported as a failure, never as "no data found".

``NASANeoWsSource._fetch_impl`` paginates over date windows and logged each
window's failure before moving on, so when every window failed the method
returned an empty list and the base funnel rendered that as
``FetchResult(success=True, data_points=[], rate_limited=False,
unreachable=False)``.  A caller reading that sees "no near-Earth close
approaches this week"; the truth was an upstream ``HTTP 429
OVER_RATE_LIMIT``.  The v3 audit's network lane caught it as a hard failure in
``tests/test_live_wiring_network.py::TestRealSourcesLive::test_nasa_neows`` --
whose helper skips only on ``result.unreachable``, and so never got the
chance, because the source claimed success.

The source now re-raises when nothing at all was collected, which puts the
classification back where the base class does it properly:
``SourceUnreachableError`` for a 429 -> ``success=False, unreachable=True``.
A genuine *partial* (some windows served, some did not) still returns what it
got, with the per-window failures logged.

Offline: the transport is replaced, so no network is touched.
"""

from __future__ import annotations

from typing import Any

import pytest

from omni_mercury_engine.data_sources.base import SourceUnreachableError
from omni_mercury_engine.data_sources.space_weather import NASANeoWsSource


class _Throttled:
    """Stands in for ``_http_get``, refusing every window the way a 429 does."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        raise SourceUnreachableError(
            'HTTP 429: {"error": {"code": "OVER_RATE_LIMIT", "message": "..."}}',
            retryable=True,
        )


def test_total_throttle_is_a_failed_fetch_not_an_empty_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every window refused: failure, flagged unreachable, with no data claimed."""
    source = NASANeoWsSource()
    transport = _Throttled()
    monkeypatch.setattr(source, "_http_get", transport)

    result = source.fetch_sync()

    assert transport.calls > 0, "the stand-in transport was never reached"
    assert not result.success, "a wholly throttled fetch reported success"
    assert result.unreachable, "a 429 must be classified unreachable"
    assert not result.data_points
    assert result.error


def test_the_live_lane_would_skip_rather_than_fail_on_that_result() -> None:
    """The network lane's own helper skips on exactly this shape.

    Pinned here so the contract the source now honours is the one the live
    test reads: skip when unreachable, fail when a reachable service drifts.
    """
    from tests.test_live_wiring_network import _fetch_or_loud

    class _Source:
        source_id = "nasa_neows"

        @staticmethod
        def fetch_sync() -> Any:
            from omni_mercury_engine.data_sources.base import FetchResult

            return FetchResult(success=False, error="HTTP 429", unreachable=True)

    with pytest.raises(BaseException, match="upstream unreachable") as caught:
        _fetch_or_loud(_Source())

    assert isinstance(caught.value, pytest.skip.Exception), "should skip, not fail"
