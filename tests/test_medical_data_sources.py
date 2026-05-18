"""Tests for the medical data-source adapters.

The adapters parse real vendor responses captured as sanitized fixtures.
Network calls are not performed: ``urlopen`` is patched at the boundary so
the real parsing + auth logic runs without leaving the test process.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from omni_mercury_engine.medical.data_sources import (
    CGMDataSource,
    CGMReading,
    ConfigurationError,
    DataSourceError,
    DexcomConfig,
    DexcomV3DataSource,
    FHIRConfig,
    FHIRObservationVitalsSource,
    VitalsDataSource,
    VitalsReading,
    parse_dexcom_egvs_payload,
    parse_fhir_observation_bundle,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "medical"


def _load_fixture(name: str) -> dict[str, Any]:
    """Return the parsed JSON content of a fixture file."""
    data = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


class _FakeHTTPResponse:
    """Minimal stand-in for ``http.client.HTTPResponse`` used by ``urlopen``."""

    def __init__(self, payload: bytes, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


class _FrozenClock:
    """Replace :class:`datetime` for deterministic ``now`` values."""

    def __init__(self, fixed: datetime) -> None:
        self._fixed = fixed

    def now(self, tz: timezone | None = None) -> datetime:
        if tz is None:
            return self._fixed
        return self._fixed.astimezone(tz)


# --------------------------------------------------------------------------- #
# Dexcom payload parser
# --------------------------------------------------------------------------- #


class TestParseDexcomEgvs:
    """Pure parser tests over the sanitized fixture."""

    def test_parses_records_in_chronological_order(self) -> None:
        payload = _load_fixture("dexcom_egvs.json")
        readings = parse_dexcom_egvs_payload(payload)
        assert len(readings) == 5
        timestamps = [r.timestamp for r in readings]
        assert timestamps == sorted(timestamps)
        assert readings[0].value_mg_dl == 118
        assert readings[-1].value_mg_dl == 158
        assert all(r.source == "dexcom_v3" for r in readings)
        assert readings[0].trend == "flat"
        assert readings[1].trend == "fortyFiveUp"

    def test_parses_trend_rate_when_present(self) -> None:
        payload = _load_fixture("dexcom_egvs.json")
        readings = parse_dexcom_egvs_payload(payload)
        assert readings[0].trend_rate_mg_dl_per_min == pytest.approx(0.4)
        assert readings[-1].trend_rate_mg_dl_per_min == pytest.approx(2.0)

    def test_timestamps_are_tz_aware(self) -> None:
        payload = _load_fixture("dexcom_egvs.json")
        readings = parse_dexcom_egvs_payload(payload)
        for r in readings:
            assert r.timestamp.tzinfo is not None
            assert r.timestamp.utcoffset().total_seconds() == 0  # type: ignore[union-attr]

    def test_missing_records_key_raises(self) -> None:
        with pytest.raises(DataSourceError, match="records"):
            parse_dexcom_egvs_payload({"recordType": "egv"})

    def test_record_missing_value_raises(self) -> None:
        with pytest.raises(DataSourceError, match="systemTime"):
            parse_dexcom_egvs_payload({"records": [{"systemTime": "2024-01-15T12:00:00Z"}]})

    def test_record_with_non_numeric_value_raises(self) -> None:
        with pytest.raises(DataSourceError, match="numeric"):
            parse_dexcom_egvs_payload(
                {"records": [{"systemTime": "2024-01-15T12:00:00Z", "value": "high"}]}
            )


# --------------------------------------------------------------------------- #
# FHIR Bundle parser
# --------------------------------------------------------------------------- #


class TestParseFhirObservationBundle:
    """Parser tests over the sanitized FHIR R4 fixture."""

    def test_returns_one_snapshot_per_effective_datetime(self) -> None:
        payload = _load_fixture("fhir_observation_vitals.json")
        readings = parse_fhir_observation_bundle(payload)
        assert len(readings) == 2
        first, second = readings
        assert first.timestamp < second.timestamp
        assert first.hr_bpm == 78
        # MAP is computed from systolic + diastolic when MAP itself is not
        # reported: (118 + 2*76)/3 == 90.0.
        assert first.map_mmhg == pytest.approx(90.0)
        assert first.spo2_pct == 98
        assert first.etco2_mmhg == 38
        assert second.hr_bpm == 82
        assert second.map_mmhg is None
        assert all(r.source == "fhir_observation" for r in readings)

    def test_rejects_non_bundle_payload(self) -> None:
        with pytest.raises(DataSourceError, match="Bundle"):
            parse_fhir_observation_bundle({"resourceType": "Observation"})

    def test_empty_bundle_returns_empty_list(self) -> None:
        readings = parse_fhir_observation_bundle(
            {"resourceType": "Bundle", "type": "searchset", "total": 0}
        )
        assert readings == []

    def test_invalid_entries_raises(self) -> None:
        with pytest.raises(DataSourceError, match="entry"):
            parse_fhir_observation_bundle({"resourceType": "Bundle", "entry": "not-an-array"})

    def test_direct_map_observation_is_used_when_present(self) -> None:
        bundle = {
            "resourceType": "Bundle",
            "type": "searchset",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "status": "final",
                        "code": {"coding": [{"system": "http://loinc.org", "code": "8478-0"}]},
                        "effectiveDateTime": "2024-03-22T13:30:00Z",
                        "valueQuantity": {"value": 95, "unit": "mm[Hg]"},
                    }
                }
            ],
        }
        readings = parse_fhir_observation_bundle(bundle)
        assert len(readings) == 1
        assert readings[0].map_mmhg == 95


# --------------------------------------------------------------------------- #
# DexcomV3DataSource (auth + fetch)
# --------------------------------------------------------------------------- #


class TestDexcomV3DataSource:
    """End-to-end adapter tests with the network boundary mocked."""

    def _config(self) -> DexcomConfig:
        return DexcomConfig(
            client_id="test-client-id",
            client_secret="test-client-secret",  # noqa: S106 - sanitized test value
            refresh_token="test-refresh-token",  # noqa: S106 - sanitized test value
            redirect_uri="https://example.invalid/callback",
            base_url="https://sandbox-api.dexcom.com",
        )

    def test_missing_env_raises_configuration_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "DEXCOM_CLIENT_ID",
            "DEXCOM_CLIENT_SECRET",
            "DEXCOM_REFRESH_TOKEN",
            "DEXCOM_REDIRECT_URI",
        ):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(ConfigurationError, match="DEXCOM_CLIENT_ID"):
            DexcomV3DataSource()

    def test_explicit_config_skips_env_lookup(self) -> None:
        adapter = DexcomV3DataSource(self._config())
        assert adapter.name == "dexcom_v3"
        assert adapter.config.base_url == "https://sandbox-api.dexcom.com"

    def test_env_populated_creates_adapter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEXCOM_CLIENT_ID", "env-id")
        monkeypatch.setenv("DEXCOM_CLIENT_SECRET", "env-secret")
        monkeypatch.setenv("DEXCOM_REFRESH_TOKEN", "env-refresh")
        monkeypatch.setenv("DEXCOM_REDIRECT_URI", "https://example.invalid/cb")
        monkeypatch.delenv("DEXCOM_BASE_URL", raising=False)
        adapter = DexcomV3DataSource()
        assert adapter.config.client_id == "env-id"

    def test_fetch_recent_readings_validates_window(self) -> None:
        adapter = DexcomV3DataSource(self._config())
        with pytest.raises(ValueError, match="window_minutes"):
            adapter.fetch_recent_readings(window_minutes=0)
        with pytest.raises(ValueError, match="window_minutes"):
            adapter.fetch_recent_readings(window_minutes=1441)

    def test_fetch_recent_readings_returns_parsed_records(self) -> None:
        token_payload = json.dumps(_load_fixture("dexcom_token.json")).encode()
        egvs_payload = json.dumps(_load_fixture("dexcom_egvs.json")).encode()
        responses = iter(
            [
                _FakeHTTPResponse(token_payload),
                _FakeHTTPResponse(egvs_payload),
            ]
        )

        def _fake_urlopen(*_args: object, **_kwargs: object) -> _FakeHTTPResponse:
            return next(responses)

        clock = _FrozenClock(datetime(2024, 1, 15, 13, 0, 0, tzinfo=UTC))
        adapter = DexcomV3DataSource(self._config(), clock=clock)
        with patch("omni_mercury_engine.medical.data_sources.urlopen", _fake_urlopen):
            readings = adapter.fetch_recent_readings(window_minutes=60)
        assert len(readings) == 5
        assert readings[0].value_mg_dl == 118
        assert readings[-1].value_mg_dl == 158

    def test_token_endpoint_http_error_wrapped(self) -> None:
        def _raise_http_error(*_args: object, **_kwargs: object) -> _FakeHTTPResponse:
            raise HTTPError(
                "https://sandbox-api.dexcom.com/v2/oauth2/token",
                400,
                "Bad Request",
                {},  # type: ignore[arg-type]
                io.BytesIO(b'{"error": "invalid_grant"}'),
            )

        adapter = DexcomV3DataSource(self._config())
        with (
            patch("omni_mercury_engine.medical.data_sources.urlopen", _raise_http_error),
            pytest.raises(DataSourceError, match="token refresh failed"),
        ):
            adapter.fetch_recent_readings(window_minutes=60)

    def test_egv_endpoint_invalid_status_wrapped(self) -> None:
        token_payload = json.dumps(_load_fixture("dexcom_token.json")).encode()
        responses = iter(
            [
                _FakeHTTPResponse(token_payload, status=200),
                _FakeHTTPResponse(b"{}", status=503),
            ]
        )

        def _fake_urlopen(*_args: object, **_kwargs: object) -> _FakeHTTPResponse:
            return next(responses)

        clock = _FrozenClock(datetime(2024, 1, 15, 13, 0, 0, tzinfo=UTC))
        adapter = DexcomV3DataSource(self._config(), clock=clock)
        with (
            patch("omni_mercury_engine.medical.data_sources.urlopen", _fake_urlopen),
            pytest.raises(DataSourceError, match="503"),
        ):
            adapter.fetch_recent_readings(window_minutes=60)

    def test_token_is_cached_between_calls(self) -> None:
        token_payload = json.dumps(_load_fixture("dexcom_token.json")).encode()
        egvs_payload = json.dumps(_load_fixture("dexcom_egvs.json")).encode()
        # Single token response followed by two EGV responses; if the
        # adapter re-requests the token the iterator will be exhausted.
        responses = iter(
            [
                _FakeHTTPResponse(token_payload),
                _FakeHTTPResponse(egvs_payload),
                _FakeHTTPResponse(egvs_payload),
            ]
        )

        def _fake_urlopen(*_args: object, **_kwargs: object) -> _FakeHTTPResponse:
            return next(responses)

        clock = _FrozenClock(datetime(2024, 1, 15, 13, 0, 0, tzinfo=UTC))
        adapter = DexcomV3DataSource(self._config(), clock=clock)
        with patch("omni_mercury_engine.medical.data_sources.urlopen", _fake_urlopen):
            adapter.fetch_recent_readings(window_minutes=60)
            adapter.fetch_recent_readings(window_minutes=60)
        # If we get here the second call did not trigger another token refresh.


# --------------------------------------------------------------------------- #
# FHIRObservationVitalsSource (auth + fetch)
# --------------------------------------------------------------------------- #


class TestFHIRObservationVitalsSource:
    """End-to-end adapter tests with the network boundary mocked."""

    def _config(self, *, bearer: str | None = None) -> FHIRConfig:
        return FHIRConfig(
            base_url="https://sanitized.example/fhir",
            patient_id="sanitized-pid",
            bearer_token=bearer,
        )

    def test_missing_env_raises_configuration_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("FHIR_BASE_URL", "FHIR_PATIENT_ID", "FHIR_BEARER_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(ConfigurationError, match="FHIR_BASE_URL"):
            FHIRObservationVitalsSource()

    def test_env_populated_creates_adapter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FHIR_BASE_URL", "https://sanitized.example/fhir/")
        monkeypatch.setenv("FHIR_PATIENT_ID", "abc")
        monkeypatch.setenv("FHIR_BEARER_TOKEN", "test-token")
        adapter = FHIRObservationVitalsSource()
        # Trailing slash should be stripped.
        assert adapter.config.base_url == "https://sanitized.example/fhir"
        assert adapter.config.bearer_token == "test-token"

    def test_rejects_bad_base_url(self) -> None:
        with pytest.raises(ConfigurationError, match="http"):
            FHIRObservationVitalsSource(
                FHIRConfig(
                    base_url="not-a-url",
                    patient_id="sanitized-pid",
                )
            )

    def test_window_is_validated(self) -> None:
        adapter = FHIRObservationVitalsSource(self._config())
        with pytest.raises(ValueError, match="window_minutes"):
            adapter.fetch_recent_vitals(window_minutes=0)
        with pytest.raises(ValueError, match="window_minutes"):
            adapter.fetch_recent_vitals(window_minutes=1441)

    def test_fetch_recent_vitals_returns_parsed_snapshots(self) -> None:
        bundle_bytes = json.dumps(_load_fixture("fhir_observation_vitals.json")).encode()

        def _fake_urlopen(request: Any, *_args: object, **_kwargs: object) -> _FakeHTTPResponse:
            # Ensure the Authorization header is forwarded when configured.
            auth_header = request.get_header("Authorization")
            assert auth_header == "Bearer sanitized-token"
            return _FakeHTTPResponse(bundle_bytes)

        clock = _FrozenClock(datetime(2024, 3, 22, 13, 40, 0, tzinfo=UTC))
        adapter = FHIRObservationVitalsSource(self._config(bearer="sanitized-token"), clock=clock)
        with patch("omni_mercury_engine.medical.data_sources.urlopen", _fake_urlopen):
            readings = adapter.fetch_recent_vitals(window_minutes=15)
        assert len(readings) == 2
        assert readings[0].hr_bpm == 78
        assert readings[-1].hr_bpm == 82

    def test_omits_authorization_when_no_token(self) -> None:
        bundle_bytes = json.dumps(_load_fixture("fhir_observation_vitals.json")).encode()

        def _fake_urlopen(request: Any, *_args: object, **_kwargs: object) -> _FakeHTTPResponse:
            assert request.get_header("Authorization") is None
            return _FakeHTTPResponse(bundle_bytes)

        clock = _FrozenClock(datetime(2024, 3, 22, 13, 40, 0, tzinfo=UTC))
        adapter = FHIRObservationVitalsSource(self._config(), clock=clock)
        with patch("omni_mercury_engine.medical.data_sources.urlopen", _fake_urlopen):
            readings = adapter.fetch_recent_vitals(window_minutes=15)
        assert len(readings) == 2

    def test_server_error_wrapped(self) -> None:
        def _fake_urlopen(*_args: object, **_kwargs: object) -> _FakeHTTPResponse:
            return _FakeHTTPResponse(b"{}", status=502)

        clock = _FrozenClock(datetime(2024, 3, 22, 13, 40, 0, tzinfo=UTC))
        adapter = FHIRObservationVitalsSource(self._config(), clock=clock)
        with (
            patch("omni_mercury_engine.medical.data_sources.urlopen", _fake_urlopen),
            pytest.raises(DataSourceError, match="502"),
        ):
            adapter.fetch_recent_vitals(window_minutes=15)


# --------------------------------------------------------------------------- #
# Custom-adapter extension contract
# --------------------------------------------------------------------------- #


class TestCustomAdapters:
    """Confirm the documented extension contract is enforceable."""

    def test_cgm_data_source_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            CGMDataSource()  # type: ignore[abstract]

    def test_vitals_data_source_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            VitalsDataSource()  # type: ignore[abstract]

    def test_custom_cgm_subclass_runs(self) -> None:
        class StaticCGMSource(CGMDataSource):
            name = "static_test_source"

            def fetch_recent_readings(self, window_minutes: int = 180) -> list[CGMReading]:
                return [
                    CGMReading(
                        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                        value_mg_dl=120.0,
                        source=self.name,
                    )
                ]

        readings = StaticCGMSource().fetch_recent_readings()
        assert readings[0].source == "static_test_source"

    def test_custom_vitals_subclass_runs(self) -> None:
        class StaticVitalsSource(VitalsDataSource):
            name = "static_test_vitals"

            def fetch_recent_vitals(self, window_minutes: int = 5) -> list[VitalsReading]:
                return [
                    VitalsReading(
                        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                        map_mmhg=80.0,
                        hr_bpm=72.0,
                        spo2_pct=98.0,
                        etco2_mmhg=38.0,
                        source=self.name,
                    )
                ]

        readings = StaticVitalsSource().fetch_recent_vitals()
        assert readings[0].source == "static_test_vitals"
