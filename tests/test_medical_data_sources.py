# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for the medical data-source adapters.

The adapters parse real vendor responses captured as sanitized fixtures.
Network calls are not performed: :class:`SafeHTTPClient` is patched at
its public boundary so the real parsing + auth logic runs without
leaving the test process.  Routing through SafeHTTPClient (rather than
``urllib.request.urlopen``) puts the scheme allowlist, private-network
block, DNS-rebinding pin, and redirect refusal in front of every Dexcom
/ FHIR call; the adapter unit tests focus on the parsing, auth flow,
and exception-mapping logic - the SafeHTTPClient gates themselves are
covered by ``tests/test_safe_http.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import requests

# ``omni_mercury_engine.medical`` re-exports the anesthesiology +
# endocrinology modules from its ``__init__``, both of which import
# ``torch`` at module level.  Even though this test module only needs
# ``medical.data_sources``, the package ``__init__`` chain forces a
# torch import, so skip cleanly when torch is absent.
pytest.importorskip("torch")

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
from omni_mercury_engine.security.safe_http import SafeHTTPClient

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "medical"


def _load_fixture(name: str) -> dict[str, Any]:
    """Return the parsed JSON content of a fixture file."""
    data = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _build_http_error(url: str, status: int, reason: str = "Bad Request") -> requests.HTTPError:
    """Construct a ``requests.HTTPError`` matching ``SafeHTTPClient`` failures.

    ``SafeHTTPClient`` calls ``response.raise_for_status()`` on 4xx/5xx
    so the adapter sees exactly this exception shape (status code
    embedded in ``response.status_code``).  The reusable builder keeps
    every adapter-failure test consistent with the production gate.
    """
    response = requests.Response()
    response.status_code = status
    response.url = url
    response.reason = reason
    return requests.HTTPError(f"{status} {reason}: {url}", response=response)


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

    def test_spoofed_loinc_system_uri_is_ignored(self) -> None:
        """Look-alike LOINC system URIs must not be honoured.

        Regression test for CodeQL alert 881
        (``py/incomplete-url-substring-sanitization``).  An attacker
        cannot smuggle malicious vital-sign codes into the rule engine
        by supplying ``http://evil-loinc.org`` or
        ``https://loinc.org.example.com`` in ``Observation.code.coding[].system``
        -- only the canonical ``http://loinc.org`` URI is honoured per
        https://hl7.org/fhir/loinc.html .
        """
        spoofed_systems = [
            "http://evil-loinc.org",
            "https://loinc.org.example.com",
            "http://attacker.invalid/loinc.org",
            "https://loinc.org",  # close-but-wrong scheme -- LOINC URI is HTTP
            "",
        ]
        for spoofed in spoofed_systems:
            bundle = {
                "resourceType": "Bundle",
                "type": "searchset",
                "entry": [
                    {
                        "resource": {
                            "resourceType": "Observation",
                            "status": "final",
                            "code": {
                                "coding": [
                                    # ``8478-0`` is the LOINC code for mean BP;
                                    # rule engine must not pick it up under a
                                    # spoofed system URI.
                                    {"system": spoofed, "code": "8478-0"},
                                ]
                            },
                            "effectiveDateTime": "2024-03-22T13:30:00Z",
                            "valueQuantity": {"value": 999, "unit": "mm[Hg]"},
                        }
                    }
                ],
            }
            readings = parse_fhir_observation_bundle(bundle)
            assert readings == [], f"Spoofed LOINC system URI {spoofed!r} was honoured by parser"


# --------------------------------------------------------------------------- #
# DexcomV3DataSource (auth + fetch)
# --------------------------------------------------------------------------- #


class TestDexcomConfigAllowlist:
    """``DexcomConfig`` rejects any base URL outside the published hosts."""

    def _base_kwargs(self) -> dict[str, str]:
        return {
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "refresh_token": "test-refresh-token",
            "redirect_uri": "https://example.invalid/callback",
        }

    def test_production_base_accepted(self) -> None:
        config = DexcomConfig(**self._base_kwargs(), base_url="https://api.dexcom.com")
        assert config.base_url == "https://api.dexcom.com"

    def test_sandbox_base_accepted(self) -> None:
        config = DexcomConfig(**self._base_kwargs(), base_url="https://sandbox-api.dexcom.com")
        assert config.base_url == "https://sandbox-api.dexcom.com"

    def test_trailing_slash_is_canonicalised(self) -> None:
        config = DexcomConfig(**self._base_kwargs(), base_url="https://api.dexcom.com/")
        assert config.base_url == "https://api.dexcom.com"

    def test_unknown_host_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="DEXCOM_BASE_URL"):
            DexcomConfig(**self._base_kwargs(), base_url="https://evil.example.com")

    def test_http_scheme_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="DEXCOM_BASE_URL"):
            DexcomConfig(**self._base_kwargs(), base_url="http://api.dexcom.com")


class TestDexcomV3DataSource:
    """End-to-end adapter tests with the SafeHTTPClient boundary mocked."""

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

    def test_env_base_url_outside_allowlist_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEXCOM_CLIENT_ID", "env-id")
        monkeypatch.setenv("DEXCOM_CLIENT_SECRET", "env-secret")
        monkeypatch.setenv("DEXCOM_REFRESH_TOKEN", "env-refresh")
        monkeypatch.setenv("DEXCOM_REDIRECT_URI", "https://example.invalid/cb")
        monkeypatch.setenv("DEXCOM_BASE_URL", "https://attacker.example/v3")
        with pytest.raises(ConfigurationError, match="DEXCOM_BASE_URL"):
            DexcomV3DataSource()

    def test_fetch_recent_readings_validates_window(self) -> None:
        adapter = DexcomV3DataSource(self._config())
        with pytest.raises(ValueError, match="window_minutes"):
            adapter.fetch_recent_readings(window_minutes=0)
        with pytest.raises(ValueError, match="window_minutes"):
            adapter.fetch_recent_readings(window_minutes=1441)

    def test_fetch_recent_readings_returns_parsed_records(self) -> None:
        token_payload = _load_fixture("dexcom_token.json")
        egvs_payload = _load_fixture("dexcom_egvs.json")

        captured_token: dict[str, Any] = {}
        captured_egvs: dict[str, Any] = {}

        def _fake_post_form(url: str, **kwargs: Any) -> dict[str, Any]:
            captured_token["url"] = url
            captured_token["form_data"] = kwargs.get("form_data")
            captured_token["user_configured"] = kwargs.get("user_configured")
            return token_payload

        def _fake_get_json(url: str, **kwargs: Any) -> dict[str, Any]:
            captured_egvs["url"] = url
            captured_egvs["headers"] = kwargs.get("headers")
            captured_egvs["params"] = kwargs.get("params")
            captured_egvs["user_configured"] = kwargs.get("user_configured")
            return egvs_payload

        clock = _FrozenClock(datetime(2024, 1, 15, 13, 0, 0, tzinfo=UTC))
        adapter = DexcomV3DataSource(self._config(), clock=clock)
        with (
            patch.object(SafeHTTPClient, "post_form", staticmethod(_fake_post_form)),
            patch.object(SafeHTTPClient, "get_json", staticmethod(_fake_get_json)),
        ):
            readings = adapter.fetch_recent_readings(window_minutes=60)
        assert len(readings) == 5
        assert readings[0].value_mg_dl == 118
        assert readings[-1].value_mg_dl == 158
        # Token endpoint goes through SafeHTTPClient with user_configured=True
        # so the operator-supplied base URL passes the private-network gate
        # without needing the trusted-allowlist.
        assert captured_token["url"] == "https://sandbox-api.dexcom.com/v2/oauth2/token"
        assert captured_token["form_data"] == {
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "refresh_token": "test-refresh-token",
            "grant_type": "refresh_token",
            "redirect_uri": "https://example.invalid/callback",
        }
        assert captured_token["user_configured"] is True
        assert captured_egvs["url"] == "https://sandbox-api.dexcom.com/v3/users/self/egvs"
        assert captured_egvs["user_configured"] is True
        # The bearer token from the token response must be forwarded.
        headers = captured_egvs["headers"]
        assert headers["Authorization"].startswith("Bearer ")

    def test_token_endpoint_http_error_wrapped(self) -> None:
        def _raise_http_error(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise _build_http_error(
                "https://sandbox-api.dexcom.com/v2/oauth2/token", 400, "Bad Request"
            )

        adapter = DexcomV3DataSource(self._config())
        with (
            patch.object(SafeHTTPClient, "post_form", staticmethod(_raise_http_error)),
            pytest.raises(DataSourceError, match="token refresh failed"),
        ):
            adapter.fetch_recent_readings(window_minutes=60)

    def test_egv_endpoint_invalid_status_wrapped(self) -> None:
        token_payload = _load_fixture("dexcom_token.json")

        def _fake_post_form(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return token_payload

        def _fake_get_json(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise _build_http_error(
                "https://sandbox-api.dexcom.com/v3/users/self/egvs",
                503,
                "Service Unavailable",
            )

        clock = _FrozenClock(datetime(2024, 1, 15, 13, 0, 0, tzinfo=UTC))
        adapter = DexcomV3DataSource(self._config(), clock=clock)
        with (
            patch.object(SafeHTTPClient, "post_form", staticmethod(_fake_post_form)),
            patch.object(SafeHTTPClient, "get_json", staticmethod(_fake_get_json)),
            pytest.raises(DataSourceError, match="503"),
        ):
            adapter.fetch_recent_readings(window_minutes=60)

    def test_token_is_cached_between_calls(self) -> None:
        token_payload = _load_fixture("dexcom_token.json")
        egvs_payload = _load_fixture("dexcom_egvs.json")
        token_calls = {"n": 0}
        egv_calls = {"n": 0}

        def _fake_post_form(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            token_calls["n"] += 1
            return token_payload

        def _fake_get_json(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            egv_calls["n"] += 1
            return egvs_payload

        clock = _FrozenClock(datetime(2024, 1, 15, 13, 0, 0, tzinfo=UTC))
        adapter = DexcomV3DataSource(self._config(), clock=clock)
        with (
            patch.object(SafeHTTPClient, "post_form", staticmethod(_fake_post_form)),
            patch.object(SafeHTTPClient, "get_json", staticmethod(_fake_get_json)),
        ):
            adapter.fetch_recent_readings(window_minutes=60)
            adapter.fetch_recent_readings(window_minutes=60)
        assert token_calls["n"] == 1, "Second fetch must reuse the cached access token"
        assert egv_calls["n"] == 2


# --------------------------------------------------------------------------- #
# FHIRObservationVitalsSource (auth + fetch)
# --------------------------------------------------------------------------- #


class TestFHIRConfigHttpsPolicy:
    """``FHIRConfig`` enforces HTTPS by default with an explicit ``allow_http``."""

    def test_https_is_accepted(self) -> None:
        config = FHIRConfig(
            base_url="https://fhir.example.org/r4",
            patient_id="sanitized-pid",
        )
        assert config.allow_http is False

    def test_http_rejected_by_default(self) -> None:
        with pytest.raises(ConfigurationError, match="https://"):
            FHIRConfig(
                base_url="http://fhir.example.org/r4",
                patient_id="sanitized-pid",
            )

    def test_http_accepted_with_allow_http_optin(self) -> None:
        config = FHIRConfig(
            base_url="http://localhost:8080/fhir",
            patient_id="sanitized-pid",
            allow_http=True,
        )
        assert config.allow_http is True

    def test_unsupported_scheme_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="https://"):
            FHIRConfig(
                base_url="ftp://fhir.example.org/r4",
                patient_id="sanitized-pid",
            )

    def test_env_allow_http_flag_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FHIR_BASE_URL", "http://localhost:8080/fhir")
        monkeypatch.setenv("FHIR_PATIENT_ID", "abc")
        monkeypatch.setenv("FHIR_ALLOW_HTTP", "1")
        monkeypatch.delenv("FHIR_BEARER_TOKEN", raising=False)
        adapter = FHIRObservationVitalsSource()
        assert adapter.config.allow_http is True

    def test_env_http_without_optin_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FHIR_BASE_URL", "http://localhost:8080/fhir")
        monkeypatch.setenv("FHIR_PATIENT_ID", "abc")
        monkeypatch.delenv("FHIR_ALLOW_HTTP", raising=False)
        with pytest.raises(ConfigurationError, match="https://"):
            FHIRObservationVitalsSource()


class TestFHIRObservationVitalsSource:
    """End-to-end adapter tests with the SafeHTTPClient boundary mocked."""

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
        with pytest.raises(ConfigurationError, match="https://"):
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
        bundle = _load_fixture("fhir_observation_vitals.json")
        captured: dict[str, Any] = {}

        def _fake_get_json(url: str, **kwargs: Any) -> dict[str, Any]:
            captured["url"] = url
            captured["headers"] = kwargs.get("headers")
            captured["params"] = kwargs.get("params")
            captured["allow_http"] = kwargs.get("allow_http")
            captured["user_configured"] = kwargs.get("user_configured")
            return bundle

        clock = _FrozenClock(datetime(2024, 3, 22, 13, 40, 0, tzinfo=UTC))
        adapter = FHIRObservationVitalsSource(self._config(bearer="sanitized-token"), clock=clock)
        with patch.object(SafeHTTPClient, "get_json", staticmethod(_fake_get_json)):
            readings = adapter.fetch_recent_vitals(window_minutes=15)
        assert len(readings) == 2
        assert readings[0].hr_bpm == 78
        assert readings[-1].hr_bpm == 82
        # Ensure the Authorization header is forwarded when configured.
        headers = captured["headers"]
        assert headers["Authorization"] == "Bearer sanitized-token"
        # PHI must default to allow_http=False so the SafeHTTPClient
        # scheme gate rejects any operator that points at http:// without
        # the explicit opt-in.
        assert captured["allow_http"] is False
        assert captured["user_configured"] is True

    def test_omits_authorization_when_no_token(self) -> None:
        bundle = _load_fixture("fhir_observation_vitals.json")
        captured: dict[str, Any] = {}

        def _fake_get_json(url: str, **kwargs: Any) -> dict[str, Any]:
            captured["headers"] = kwargs.get("headers")
            return bundle

        clock = _FrozenClock(datetime(2024, 3, 22, 13, 40, 0, tzinfo=UTC))
        adapter = FHIRObservationVitalsSource(self._config(), clock=clock)
        with patch.object(SafeHTTPClient, "get_json", staticmethod(_fake_get_json)):
            readings = adapter.fetch_recent_vitals(window_minutes=15)
        assert len(readings) == 2
        assert "Authorization" not in captured["headers"]

    def test_server_error_wrapped(self) -> None:
        def _fake_get_json(url: str, **_kwargs: Any) -> dict[str, Any]:
            raise _build_http_error(url, 502, "Bad Gateway")

        clock = _FrozenClock(datetime(2024, 3, 22, 13, 40, 0, tzinfo=UTC))
        adapter = FHIRObservationVitalsSource(self._config(), clock=clock)
        with (
            patch.object(SafeHTTPClient, "get_json", staticmethod(_fake_get_json)),
            pytest.raises(DataSourceError, match="502"),
        ):
            adapter.fetch_recent_vitals(window_minutes=15)

    def test_allow_http_flag_propagates_to_safe_http_client(self) -> None:
        bundle = _load_fixture("fhir_observation_vitals.json")
        captured: dict[str, Any] = {}

        def _fake_get_json(url: str, **kwargs: Any) -> dict[str, Any]:
            captured["allow_http"] = kwargs.get("allow_http")
            return bundle

        clock = _FrozenClock(datetime(2024, 3, 22, 13, 40, 0, tzinfo=UTC))
        adapter = FHIRObservationVitalsSource(
            FHIRConfig(
                base_url="http://localhost:8080/fhir",
                patient_id="sanitized-pid",
                allow_http=True,
            ),
            clock=clock,
        )
        with patch.object(SafeHTTPClient, "get_json", staticmethod(_fake_get_json)):
            adapter.fetch_recent_vitals(window_minutes=15)
        assert captured["allow_http"] is True


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
