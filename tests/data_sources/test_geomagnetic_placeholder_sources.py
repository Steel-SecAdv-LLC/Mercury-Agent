# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""INTERMAGNET and SuperMAG placeholders can no longer pass as live feeds.

Neither source implements a network client: ``_fetch_impl`` makes no request
and returns one hard-coded catalog entry per observatory (INTERMAGNET) or per
index name (SuperMAG). Those entries used to ship unlabelled with
``confidence=0.85``, so the live-ingestion seam stamped them
``data_provenance="live"`` -- constants presented as a magnetometer feed.
They now follow the module's honesty contract (the one
``HeartMathGCMSSource`` and ``BGSELFStationSource`` already keep): every point
is labelled ``metadata["simulated"] = True`` with ``confidence=0.0`` and the
seam refuses it without an explicit ``allow_simulated=True``.

No test here touches the network: the sources make no request at all, which
the tests pin by failing on any use of the HTTP transport.
"""

from __future__ import annotations

from typing import Any

import pytest

from omni_mercury_engine.data_sources.base import AlertLevel, DataSourceBase
from omni_mercury_engine.data_sources.geomagnetic import INTERMAGNETSource, SuperMAGSource
from omni_mercury_engine.data_sources.live_ingestion import (
    PROVENANCE_SIMULATED,
    SimulatedDataError,
    fetch_live_datapoints,
)

_SOURCES = [INTERMAGNETSource, SuperMAGSource]


@pytest.fixture(autouse=True)
def _no_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if a placeholder source ever reaches for the transport."""

    async def _refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("placeholder source attempted an HTTP request")

    monkeypatch.setattr(DataSourceBase, "_http_get", _refuse)


@pytest.mark.parametrize("source_cls", _SOURCES)
async def test_placeholder_points_are_labelled_simulated(
    source_cls: type[DataSourceBase],
) -> None:
    """Was: unlabelled constants with ``confidence=0.85``. Now: labelled, zero confidence."""
    points = await source_cls()._fetch_impl()

    assert len(points) == 3
    for point in points:
        assert point.metadata["simulated"] is True
        assert point.metadata["data_provenance"] == PROVENANCE_SIMULATED
        assert point.confidence == 0.0
        assert point.alert_level == AlertLevel.NONE
        assert "PLACEHOLDER" in point.data["note"]


@pytest.mark.parametrize("source_cls", _SOURCES)
def test_live_ingestion_refuses_the_placeholder_without_opt_in(
    source_cls: type[DataSourceBase],
) -> None:
    """Was: ``fetch_live_datapoints`` returned the constants as ``data_provenance="live"``."""
    source = source_cls()

    with pytest.raises(SimulatedDataError, match=source.source_id):
        fetch_live_datapoints(source)

    fetched = fetch_live_datapoints(source, allow_simulated=True)
    assert fetched.data_provenance == PROVENANCE_SIMULATED
    assert len(fetched.data_points) == 3
