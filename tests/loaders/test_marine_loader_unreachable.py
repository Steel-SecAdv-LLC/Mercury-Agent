# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``MarineLoader.fetch_historical`` refuses rather than fabricates when OBIS is down.

With an empty cache and OBIS unreachable (the transport raising
``ConnectionError``, or ``MERCURY_OFFLINE=1``), ``fetch_historical`` returned
4000 rows from ``_synthesize_event``. A broad ``except Exception`` around every
species query swallowed the failure -- including the typed air-gap refusal --
and the frame carried no synthetic mark (``is_synthetic()`` was ``False``), so
``engineer_scored_features`` scored fabricated occurrences as a real event.

All tests are offline: the shared ``SafeHTTPClient.get_bytes`` transport is
patched, and the air-gap tests additionally fail on any DNS lookup.
"""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING, Any

import pytest

from omni_mercury_engine.datasets.exceptions import (
    DataSourceUnavailableError,
    OfflineModeError,
)
from omni_mercury_engine.loaders.base import SyntheticDataNotDeleakable
from omni_mercury_engine.loaders.marine_loader import MarineLoader

if TYPE_CHECKING:
    from pathlib import Path

_GET_BYTES = "omni_mercury_engine.loaders.base.SafeHTTPClient.get_bytes"
_EVENT_ID = "gbr_bleaching_2016"


def _unreachable(*_args: Any, **_kwargs: Any) -> bytes:
    raise ConnectionError("simulated: network unreachable")


def _empty_obis_page(*_args: Any, **_kwargs: Any) -> bytes:
    return b'{"total": 0, "results": []}'


def _no_dns(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("DNS resolution attempted under MERCURY_OFFLINE")


def _loader(tmp_path: Path) -> MarineLoader:
    return MarineLoader(cache_dir=tmp_path, max_retries=0)


def test_unreachable_obis_raises_instead_of_fabricating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: 4000 synthetic rows with ``is_synthetic()`` False. Now: a typed refusal naming OBIS."""
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
    monkeypatch.setattr(_GET_BYTES, _unreachable)
    loader = _loader(tmp_path)

    with pytest.raises(DataSourceUnavailableError) as excinfo:
        loader.fetch_historical(_EVENT_ID)

    message = str(excinfo.value)
    assert excinfo.value.loader_name == "MarineLoader"
    assert "OBIS" in message
    assert "api.obis.org" in message
    assert _EVENT_ID in message
    # Nothing may be parked in the cache slot for a later run to serve as real.
    assert loader._read_cache(f"marine_historical_{_EVENT_ID}") is None


def test_empty_obis_response_raises_instead_of_fabricating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reachable OBIS that holds no records for the event is not a licence to synthesise."""
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
    monkeypatch.setattr(_GET_BYTES, _empty_obis_page)

    with pytest.raises(DataSourceUnavailableError, match="returned no occurrence records"):
        _loader(tmp_path).fetch_historical(_EVENT_ID)


def test_offline_mode_surfaces_the_typed_air_gap_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: ``OfflineModeError`` swallowed and replaced by synthetic rows.

    The opt-in is left on deliberately: offline mode never degrades to
    synthetic data, so the refusal must win even with
    ``MERCURY_ALLOW_SYNTHETIC=1``.
    """
    monkeypatch.setenv("MERCURY_OFFLINE", "1")
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
    monkeypatch.setattr(socket, "getaddrinfo", _no_dns)

    with pytest.raises(OfflineModeError, match=r"api\.obis\.org"):
        _loader(tmp_path).fetch_historical(_EVENT_ID)


def test_offline_mode_refusal_also_survives_fetch_realtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: a generic ``ConnectionError`` that no longer named ``MERCURY_OFFLINE``."""
    monkeypatch.setenv("MERCURY_OFFLINE", "1")
    monkeypatch.setattr(socket, "getaddrinfo", _no_dns)

    with pytest.raises(OfflineModeError):
        _loader(tmp_path).fetch_realtime()


def test_opted_in_synthetic_stand_in_is_marked_and_never_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the explicit opt-in the stand-in is returned, but honestly marked."""
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
    monkeypatch.setattr(_GET_BYTES, _unreachable)
    loader = _loader(tmp_path)

    frame = loader.fetch_historical(_EVENT_ID)

    assert len(frame) > 0
    assert loader.is_synthetic(frame)
    assert not frame["is_real_data"].any()
    assert (frame["dataset_id"] == "synthetic").all()
    assert loader._read_cache(f"marine_historical_{_EVENT_ID}") is None
    with pytest.raises(SyntheticDataNotDeleakable):
        loader.engineer_scored_features(frame)
