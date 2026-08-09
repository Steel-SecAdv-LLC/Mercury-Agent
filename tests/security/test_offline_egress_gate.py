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
* ``safe_http.enforce_offline_egress`` and every own-transport callsite that
  uses it -- the MIT-BIH wfdb loader (uncached refused, primed cache served),
  the cognitive httpx sources, the NIST CSF reference fetcher (uncached
  refused, fresh cache served), the batch webhook callback (suppressed as a
  logged skip), the integrations ``HTTPClient``, and the cross-platform hub
  adapter transports.

The point is fail-closed sovereignty: offline mode never silently degrades to a
network attempt, and it never blocks the loopback path a local model needs.
"""

from __future__ import annotations

import asyncio
import socket
from datetime import UTC, datetime
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
from omni_mercury_engine.security.safe_http import SafeHTTPClient, enforce_offline_egress


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

    def test_offline_refuses_localhost_subdomain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``*.localhost`` subdomains are refused offline, pre-DNS.

        Their resolution is resolver-dependent (RFC 6761 SHOULD): a
        hosts-file entry or hostile resolver can map ``foo.localhost`` to a
        public address, so a name-based permit would be an egress bypass.
        Only the literal ``localhost`` and loopback IPs are carved out.
        """
        monkeypatch.setenv("MERCURY_OFFLINE", "1")

        def _boom(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("DNS resolution attempted under MERCURY_OFFLINE")

        monkeypatch.setattr(socket, "getaddrinfo", _boom)
        with pytest.raises(OfflineModeError):
            SafeHTTPClient.validate_url(
                "http://exfil.localhost/x", allow_http=True, user_configured=True
            )


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


def _forbid_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any DNS resolution attempt fail the test."""

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("DNS resolution attempted under MERCURY_OFFLINE")

    monkeypatch.setattr(socket, "getaddrinfo", _boom)


def _forbid_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any socket construction fail the test (sync callsites only).

    Async tests must not use this: the running event loop owns live
    sockets, and pytest-asyncio's machinery may construct more.  For the
    async gates below, the pre-socket guarantee is carried by
    ``_forbid_dns`` plus the gate firing before the transport import.
    """

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("socket opened under MERCURY_OFFLINE")

    monkeypatch.setattr(socket, "socket", _boom)


class TestEnforceOfflineEgressHelper:
    """The shared gate for own-transport callsites mirrors the SafeHTTPClient policy."""

    def test_external_refused_before_any_dns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        _forbid_sockets(monkeypatch)
        with pytest.raises(OfflineModeError):
            enforce_offline_egress("https://example.com/api")

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:11434/api/generate",
            "http://[::1]:9090/metrics",
            "http://localhost:8000/api",
        ],
    )
    def test_loopback_permitted_without_dns(
        self, url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        # No raise == permitted (loopback sidecars stay reachable air-gapped).
        enforce_offline_egress(url)

    def test_noop_when_online(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Online, the helper decides nothing -- and still never resolves."""
        monkeypatch.setenv("MERCURY_OFFLINE", "0")
        _forbid_dns(monkeypatch)
        enforce_offline_egress("https://example.com/api")

    def test_schemeless_url_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A URL the parser cannot extract a host from is refused, not permitted."""
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        with pytest.raises(OfflineModeError):
            enforce_offline_egress("example.com/api")

    @pytest.mark.parametrize(
        "url",
        [
            "http://exfil.localhost/x",
            "http://a.b.localhost:9999/",
        ],
    )
    def test_localhost_subdomains_refused(self, url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """``*.localhost`` names are NOT loopback-decidable and must refuse.

        The own-transport callsites hand the URL straight to their own
        resolver, and RFC 6761 does not guarantee ``*.localhost`` resolves
        to loopback there -- a permit here would be an egress bypass.
        """
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        _forbid_sockets(monkeypatch)
        with pytest.raises(OfflineModeError):
            enforce_offline_egress(url)


class TestMITBIHLoaderOfflineGate:
    """The wfdb/PhysioNet path refuses uncached and serves the primed cache."""

    def _loader(self, tmp_path: Any) -> Any:
        from omni_mercury_engine.datasets.base import DatasetConfig
        from omni_mercury_engine.datasets.mitbih import MITBIHLoader

        config = DatasetConfig(
            name="mitbih",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            max_samples=10,
            random_seed=42,
        )
        return MITBIHLoader(config)

    def test_uncached_download_refused_pre_socket(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loader = self._loader(tmp_path)
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        _forbid_sockets(monkeypatch)
        with pytest.raises(OfflineModeError):
            loader.download()

    def test_primed_cache_served_offline_without_wfdb(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A primed segment cache serves air-gapped -- no wfdb import, no socket."""
        import numpy as np

        loader = self._loader(tmp_path)
        loader.data_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            loader.data_path / "mitbih_segments.npz",
            X=np.zeros((4, 360)),
            y=np.zeros(4, dtype=np.int32),
        )
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        _forbid_sockets(monkeypatch)
        assert loader.download() is True


class TestCognitiveHTTPXSourcesOfflineGate:
    """The ad-hoc httpx enrichment sources refuse loudly, pre-socket."""

    def test_usgs_source_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("httpx")
        from omni_mercury_engine.cognitive.anomaly_detection import (
            USGSEarthquakeSource,
        )

        source = USGSEarthquakeSource()
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        _forbid_sockets(monkeypatch)
        # The catch-all handlers swallow httpx transport errors only; the
        # offline refusal must propagate, never read as "0 earthquakes".
        with pytest.raises(OfflineModeError):
            source.fetch()

    def test_noaa_source_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("httpx")
        from omni_mercury_engine.cognitive.anomaly_detection import (
            NOAAWeatherSource,
        )

        source = NOAAWeatherSource(state="CA")
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        _forbid_sockets(monkeypatch)
        with pytest.raises(OfflineModeError):
            source.fetch()

    def test_integrator_fetch_all_surfaces_offline_explicitly(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The production path never disguises the air-gap as a fetch error.

        ``ExternalDataIntegrator.fetch_all`` is the sole in-repo consumer of
        these sources (via ``IntegratedAnomalyDetector.predict``); under
        ``MERCURY_OFFLINE`` it must emit one explicit offline log -- not a
        per-source "Error fetching..." line -- and return empty so local
        detection continues.
        """
        pytest.importorskip("httpx")
        from omni_mercury_engine.cognitive.anomaly_detection import (
            ExternalDataIntegrator,
            USGSEarthquakeSource,
        )

        integrator = ExternalDataIntegrator()
        integrator.register_source("usgs_earthquakes", USGSEarthquakeSource())
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        _forbid_sockets(monkeypatch)
        logger_name = "omni_mercury_engine.cognitive.anomaly_detection"
        with caplog.at_level("WARNING", logger=logger_name):
            assert integrator.fetch_all() == []
        assert any("MERCURY_OFFLINE" in r.message for r in caplog.records)
        assert not any(r.message.startswith("Error fetching") for r in caplog.records)

    def test_sources_refuse_construction_without_httpx(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without httpx the sources raise a clear ImportError, not a NameError.

        The module guards ``import httpx`` behind ``HTTPX_AVAILABLE`` but the
        constructors used to reference ``httpx`` unconditionally, crashing
        with ``NameError`` on any install without the ``[api]`` extra. The
        guard must stay wired to the constructors.
        """
        from omni_mercury_engine.cognitive import anomaly_detection

        monkeypatch.setattr(anomaly_detection, "HTTPX_AVAILABLE", False)
        with pytest.raises(ImportError, match=r"mercury-agent\[api\]"):
            anomaly_detection.USGSEarthquakeSource()
        with pytest.raises(ImportError, match=r"mercury-agent\[api\]"):
            anomaly_detection.NOAAWeatherSource(state="CA")


class TestNISTCSFFetcherOfflineGate:
    """The compliance reference fetch refuses uncached and serves a fresh cache."""

    def test_uncached_fetch_refused_pre_socket(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.compliance.nist_csf_integrator import (
            NISTCSFReferenceFetcher,
        )

        fetcher = NISTCSFReferenceFetcher(cache_dir=tmp_path / "nist")
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        _forbid_sockets(monkeypatch)
        with pytest.raises(OfflineModeError):
            fetcher.fetch_payload()

    def test_fresh_cache_served_offline(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.compliance.nist_csf_integrator import (
            NISTCSFReferenceFetcher,
        )

        fetcher = NISTCSFReferenceFetcher(cache_dir=tmp_path / "nist")
        cache_path = fetcher._cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"PK\x03\x04primed-reference-payload"
        cache_path.write_bytes(payload)
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        _forbid_sockets(monkeypatch)
        assert fetcher.fetch_payload() == payload


class TestBatchWebhookOfflineGate:
    """Webhook callbacks are suppressed as a logged skip, never a socket."""

    def test_callback_suppressed(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        pytest.importorskip("fastapi")
        try:
            from omni_mercury_engine.api.routes.batch import JobStatus, _send_callback
        except RuntimeError as exc:  # fastapi extras (python-multipart) missing
            pytest.skip(f"batch routes unavailable in this environment: {exc}")

        # Build the loop before any socket patching (asyncio needs its own
        # self-pipe socket); the gate fires as the first statement of the
        # coroutine, before httpx or the transport is touched.
        loop = asyncio.new_event_loop()
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        _forbid_sockets(monkeypatch)
        try:
            with caplog.at_level("WARNING", logger="omni_mercury_engine.api.routes.batch"):
                loop.run_until_complete(
                    _send_callback("https://example.com/hook", "job-1", JobStatus.COMPLETED)
                )
        finally:
            loop.close()
        assert any("suppressed" in record.message for record in caplog.records)

    def test_callback_url_rejected_at_validation_before_dns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request carrying callback_url is refused at validation, pre-DNS.

        The SSRF validator resolves the callback hostname; DNS is itself
        egress in a true air-gap, so under MERCURY_OFFLINE the field is
        rejected before any resolution -- the webhook could never fire
        anyway.
        """
        pytest.importorskip("fastapi")
        import pydantic

        try:
            from omni_mercury_engine.api.routes.batch import BatchDetectRequest
        except RuntimeError as exc:  # fastapi extras (python-multipart) missing
            pytest.skip(f"batch routes unavailable in this environment: {exc}")

        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        _forbid_sockets(monkeypatch)
        with pytest.raises(pydantic.ValidationError, match="MERCURY_OFFLINE"):
            BatchDetectRequest(
                data=[[1.0, 2.0]],
                callback_url="https://example.com/hook",
            )


class TestIntegrationsHTTPClientOfflineGate:
    """The publicly exported HTTPClient gates before breaker/retry/transport."""

    async def test_external_request_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from omni_mercury_engine.integrations.http.client import HTTPClient

        client = HTTPClient(base_url="https://api.example.com")
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        with pytest.raises(OfflineModeError):
            await client.get("/users/1")
        # The refusal fired before the resilience machinery: nothing was
        # counted as a request and no breaker recorded a failure.
        metrics = client.get_metrics()
        assert metrics["total_requests"] == 0
        assert metrics["circuit_breakers"] == {}

    async def test_absolute_external_endpoint_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.integrations.http.client import HTTPClient

        client = HTTPClient()
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        with pytest.raises(OfflineModeError):
            await client.post("https://api.example.com/events", json_data={})


class TestCrossPlatformHubOfflineGate:
    """Every hub adapter transport refuses an external platform endpoint."""

    def _config(self, endpoint: str) -> Any:
        from omni_mercury_engine.integrations.cross_platform_hub import (
            PlatformConfig,
            PlatformType,
        )

        return PlatformConfig(
            platform_type=PlatformType.CUSTOM,
            name="probe",
            endpoint=endpoint,
        )

    async def test_http_adapter_connect_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from omni_mercury_engine.integrations.cross_platform_hub import (
            HTTPPlatformAdapter,
        )

        adapter = HTTPPlatformAdapter(self._config("https://platform.example.com"))
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        with pytest.raises(OfflineModeError):
            await adapter.connect()
        assert adapter.is_connected is False

    async def test_http_adapter_send_refused_with_stale_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A session opened while online must not keep sending air-gapped."""
        from omni_mercury_engine.integrations.cross_platform_hub import (
            AnomalyEvent,
            HTTPPlatformAdapter,
        )

        adapter = HTTPPlatformAdapter(self._config("https://platform.example.com"))
        adapter._session = object()  # simulate a session established online
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        event = AnomalyEvent(
            event_id="e1",
            timestamp=datetime.now(UTC),
            source="probe",
            severity="low",
            score=1.0,
            is_anomaly=True,
        )
        with pytest.raises(OfflineModeError):
            await adapter.send_event(event)
        with pytest.raises(OfflineModeError):
            await adapter.send_batch([event])

    async def test_prometheus_flush_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from omni_mercury_engine.integrations.cross_platform_hub import (
            PrometheusAdapter,
        )

        adapter = PrometheusAdapter(self._config("https://push.example.com"))
        adapter._metrics_buffer.append("mercury_anomaly_score 1.0")
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        with pytest.raises(OfflineModeError):
            await adapter._flush_metrics()
        # The buffered metric was NOT dropped: nothing was delivered.
        assert adapter._metrics_buffer

    async def test_prometheus_connect_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A push-model adapter must not claim healthy when it can never deliver."""
        from omni_mercury_engine.integrations.cross_platform_hub import (
            PrometheusAdapter,
        )

        adapter = PrometheusAdapter(self._config("https://push.example.com"))
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        with pytest.raises(OfflineModeError):
            await adapter.connect()
        assert adapter.is_connected is False

    async def test_prometheus_disconnect_never_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cleanup completes offline: buffer retained, disconnected transition done."""
        from omni_mercury_engine.integrations.cross_platform_hub import (
            PrometheusAdapter,
        )

        adapter = PrometheusAdapter(self._config("https://push.example.com"))
        adapter._connected = True
        adapter._metrics_buffer.append("mercury_anomaly_score 1.0")
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        await adapter.disconnect()  # must not raise
        assert adapter.is_connected is False
        assert adapter._metrics_buffer  # retained, not silently dropped

    async def test_otlp_connect_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from omni_mercury_engine.integrations.cross_platform_hub import (
            OpenTelemetryAdapter,
        )

        adapter = OpenTelemetryAdapter(self._config("https://otel.example.com"))
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        with pytest.raises(OfflineModeError):
            await adapter.connect()
        assert adapter.is_connected is False

    async def test_otlp_send_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from omni_mercury_engine.integrations.cross_platform_hub import (
            AnomalyEvent,
            OpenTelemetryAdapter,
        )

        adapter = OpenTelemetryAdapter(self._config("https://otel.example.com"))
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        event = AnomalyEvent(
            event_id="e1",
            timestamp=datetime.now(UTC),
            source="probe",
            severity="low",
            score=1.0,
            is_anomaly=True,
        )
        with pytest.raises(OfflineModeError):
            await adapter.send_event(event)


class TestEmailReportSenderOfflineGate:
    """SMTP egress is refused pre-socket; the boolean contract is preserved."""

    def test_external_relay_suppressed(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from omni_mercury_engine.utils.report_generator import EmailReportSender

        sender = EmailReportSender(
            {
                "server": "smtp.example.com",
                "port": "587",
                "sender_email": "a@example.com",
                "password": "x",
            }
        )
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        _forbid_sockets(monkeypatch)
        with caplog.at_level("WARNING"):
            assert sender.send_email_report("report", "b@example.com") is False
        assert any("MERCURY_OFFLINE" in r.message for r in caplog.records)


class TestOllamaProbeOfflineGate:
    """The raw TCP availability probe honors the adapter's egress policy."""

    def test_external_host_probe_suppressed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from omni_mercury_engine.models.foundation.ollama_adapter import (
            OllamaConfig,
            OllamaLLMAdapter,
        )

        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        monkeypatch.delenv("MERCURY_MODEL_ENDPOINT", raising=False)
        monkeypatch.delenv("MERCURY_OLLAMA_HOST", raising=False)
        _forbid_dns(monkeypatch)
        _forbid_sockets(monkeypatch)
        adapter = OllamaLLMAdapter(
            ollama_config=OllamaConfig(host="ollama.example.com", model="llama3")
        )
        assert adapter._is_available is False


class TestRedisCacheOfflineGate:
    """A non-loopback REDIS_HOST is refused pre-socket; callers fall back."""

    async def test_external_redis_suppressed(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from omni_mercury_engine.integrations.stubs.cache import RedisCache

        cache = RedisCache(host="redis.example.com")
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        _forbid_dns(monkeypatch)
        with caplog.at_level("WARNING"):
            assert await cache._ensure_connected() is False
        assert any("MERCURY_OFFLINE" in r.message for r in caplog.records)
