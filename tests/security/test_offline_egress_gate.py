# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Air-gap egress contract for every outbound path outside the dataset layer.

``tests/datasets/test_offline_mode.py`` pins the dataset-loader chokepoint.
This module pins the *other* egress points that ``MERCURY_OFFLINE`` must also
close, so a single environment variable makes Mercury operable end-to-end with
every external source cut off while an on-box (loopback) model keeps working:

* ``SafeHTTPClient.validate_url`` -- the shared egress gate used by narrative
  retrieval, integrations, medical/geological loaders, and the Ollama adapter.
  Under ``MERCURY_OFFLINE`` it refuses every non-loopback destination **before
  any DNS resolution or socket**, and permits loopback targets (``127.0.0.1``,
  ``::1``, ``localhost``) so a local model still runs.
* ``DataSourceBase._http_get`` / ``_http_get_sync`` -- the live data-source
  httpx transport, refused before the socket opens.
* ``WebSearchRetriever`` -- honors the master switch even when constructed with
  ``offline_mode=False``.

The point is fail-closed sovereignty: offline mode never silently degrades to a
network attempt, and it never blocks the loopback path a local model needs.
"""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import pytest

from omni_mercury_engine.data_sources.base import (
    DataSourceBase,
    DataSourceConfig,
    DataSourceType,
)
from omni_mercury_engine.datasets.exceptions import OfflineModeError
from omni_mercury_engine.narrative.external_retrieval import (
    ExternalSearchConfig,
    WebSearchRetriever,
)
from omni_mercury_engine.security.safe_http import SafeHTTPClient


class TestSafeHTTPClientOfflineGate:
    """The shared egress gate closes external traffic and keeps loopback open."""

    def test_external_refused_before_any_dns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An external host raises OfflineModeError with zero DNS resolution.

        DNS is itself egress in a true air-gap, so the refusal must land
        before ``getaddrinfo`` is ever called.
        """
        monkeypatch.setenv("MERCURY_OFFLINE", "1")

        def _boom(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("DNS resolution attempted under MERCURY_OFFLINE")

        monkeypatch.setattr(socket, "getaddrinfo", _boom)
        with pytest.raises(OfflineModeError):
            SafeHTTPClient.validate_url("https://example.com/data", user_configured=True)

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:11434/api/generate",
            "http://[::1]:11434/api/generate",
        ],
    )
    def test_loopback_ip_permitted_without_dns(
        self, url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A loopback IP literal is permitted with no resolution at all."""
        monkeypatch.setenv("MERCURY_OFFLINE", "1")

        def _boom(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("DNS resolution should not run for an IP literal")

        monkeypatch.setattr(socket, "getaddrinfo", _boom)
        # No raise == permitted (on-box model stays reachable air-gapped).
        SafeHTTPClient.validate_url(url, allow_http=True, user_configured=True, loopback_only=True)

    def test_localhost_permitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``localhost`` is permitted (resolves locally via the hosts file)."""
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        SafeHTTPClient.validate_url(
            "http://localhost:11434/api/generate",
            allow_http=True,
            user_configured=True,
            loopback_only=True,
        )

    def test_offline_off_leaves_external_to_normal_gates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the switch unset, a trusted external host validates as before."""
        monkeypatch.setenv("MERCURY_OFFLINE", "0")
        # A class-constant trusted URL passes the allowlist with no offline block.
        SafeHTTPClient.validate_url("https://earthquake.usgs.gov/fdsnws/event/1/query")

    def test_offline_blocks_even_trusted_external(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Offline refuses even an allowlisted external host -- air-gap wins."""
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        with pytest.raises(OfflineModeError):
            SafeHTTPClient.validate_url("https://earthquake.usgs.gov/fdsnws/event/1/query")


class _ProbeSource(DataSourceBase):
    """Minimal concrete data source for exercising the transport gate."""

    @property
    def source_id(self) -> str:
        return "offline_probe"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        return [next(iter(DataSourceType))]

    async def _fetch_impl(self, *_a: Any, **_k: Any) -> list[Any]:
        return []


class TestDataSourceTransportOfflineGate:
    """The live data-source httpx transport refuses before opening a socket."""

    def _source(self) -> _ProbeSource:
        return _ProbeSource(DataSourceConfig(base_url="https://api.example.com"))

    def test_sync_transport_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MERCURY_OFFLINE", "1")

        def _no_socket(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("socket opened under MERCURY_OFFLINE")

        monkeypatch.setattr(socket, "socket", _no_socket)
        with pytest.raises(OfflineModeError):
            self._source()._http_get_sync("/endpoint")

    def test_async_transport_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        # Build the loop before any socket patching (asyncio needs its own
        # self-pipe socket); the gate fires as the first statement of the
        # coroutine, before the transport is touched.
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(OfflineModeError):
                loop.run_until_complete(self._source()._http_get("/endpoint"))
        finally:
            loop.close()


class TestNarrativeRetrieverOfflineGate:
    """Narrative web search honors the master switch, not just per-instance config."""

    def test_master_switch_forces_offline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        retriever = WebSearchRetriever(ExternalSearchConfig(offline_mode=False))
        assert retriever._offline_active() is True
        assert retriever.search("anything") == []
        assert retriever.is_available() is False

    def test_online_when_switch_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MERCURY_OFFLINE", "0")
        retriever = WebSearchRetriever(ExternalSearchConfig(offline_mode=False))
        assert retriever._offline_active() is False


class _RecordingModel:
    """Fake transformers class recording the kwargs ``from_pretrained`` receives."""

    last_kwargs: dict[str, Any] = {}

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> str:
        cls.last_kwargs = dict(kwargs)
        return f"loaded:{model_id}"


class TestSafeHFLoaderOfflineGate:
    """The HF weight loader forces cache-only resolution for Hub ids offline."""

    # A validly-pinned Hub id: 40-char lowercase hex revision.
    HUB_ID = "acme/model"
    REVISION = "a" * 40

    def test_offline_forces_local_files_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from omni_mercury_engine.security.model_policy import SafeHFLoader

        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _RecordingModel.last_kwargs = {}
        SafeHFLoader.load_model(_RecordingModel, self.HUB_ID, revision=self.REVISION)
        assert _RecordingModel.last_kwargs.get("local_files_only") is True

    def test_online_does_not_force_local_files_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from omni_mercury_engine.security.model_policy import SafeHFLoader

        monkeypatch.setenv("MERCURY_OFFLINE", "0")
        _RecordingModel.last_kwargs = {}
        SafeHFLoader.load_model(_RecordingModel, self.HUB_ID, revision=self.REVISION)
        assert "local_files_only" not in _RecordingModel.last_kwargs

    def test_offline_honors_explicit_caller_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from omni_mercury_engine.security.model_policy import SafeHFLoader

        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _RecordingModel.last_kwargs = {}
        SafeHFLoader.load_model(
            _RecordingModel, self.HUB_ID, revision=self.REVISION, local_files_only=False
        )
        # setdefault must not clobber an explicit caller choice.
        assert _RecordingModel.last_kwargs.get("local_files_only") is False
