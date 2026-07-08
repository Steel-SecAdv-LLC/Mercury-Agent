# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Delivery-integrity regressions for cross_platform_hub adapters.

These pin that a missing transport dependency fails loud (reports
not-connected / not-delivered) rather than being masked as success — the
class of data-loss bug where an integration silently drops every anomaly
event while its status reads healthy.
"""

from __future__ import annotations

import builtins
from datetime import datetime

import pytest

from omni_mercury_engine.integrations.cross_platform_hub import (
    AnomalyEvent,
    HTTPPlatformAdapter,
    OpenTelemetryAdapter,
    PlatformConfig,
    PlatformType,
)


def _no_aiohttp_import():
    """Return an ``__import__`` shim that hides aiohttp."""
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "aiohttp" or name.startswith("aiohttp."):
            raise ImportError("No module named 'aiohttp'")
        return real_import(name, *args, **kwargs)

    return mock_import


def _event() -> AnomalyEvent:
    return AnomalyEvent(
        event_id="e1",
        timestamp=datetime.now(),
        source="unit-test",
        severity="high",
        score=0.9,
        is_anomaly=True,
    )


@pytest.mark.asyncio
async def test_http_adapter_not_connected_without_aiohttp() -> None:
    """HTTP adapter must report NOT connected when its only transport is absent."""
    config = PlatformConfig(
        platform_type=PlatformType.CUSTOM,
        name="http-test",
        endpoint="https://example.invalid",
    )
    adapter = HTTPPlatformAdapter(config)

    from unittest.mock import patch

    with patch.object(builtins, "__import__", side_effect=_no_aiohttp_import()):
        connected = await adapter.connect()

    assert connected is False
    assert adapter._connected is False


@pytest.mark.asyncio
async def test_otlp_send_event_reports_failure_without_aiohttp() -> None:
    """OTLP send_event must return False (not True) when nothing could be sent."""
    config = PlatformConfig(
        platform_type=PlatformType.CUSTOM,
        name="otlp-test",
        endpoint="https://example.invalid",
    )
    adapter = OpenTelemetryAdapter(config)

    from unittest.mock import patch

    with patch.object(builtins, "__import__", side_effect=_no_aiohttp_import()):
        delivered = await adapter.send_event(_event())

    assert delivered is False


@pytest.mark.asyncio
async def test_otlp_batch_counts_only_delivered_without_aiohttp() -> None:
    """A failed batch reports zero delivered — no success inflation."""
    config = PlatformConfig(
        platform_type=PlatformType.CUSTOM,
        name="otlp-test",
        endpoint="https://example.invalid",
    )
    adapter = OpenTelemetryAdapter(config)

    from unittest.mock import patch

    with patch.object(builtins, "__import__", side_effect=_no_aiohttp_import()):
        delivered = await adapter.send_batch([_event(), _event(), _event()])

    assert delivered == 0
