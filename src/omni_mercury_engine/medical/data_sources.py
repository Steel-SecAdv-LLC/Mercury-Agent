# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Medical data source adapters for Mercury Agent.

version.

This module defines the abstract data-source contracts that the medical
detectors consume and ships two reference implementations:

* :class:`DexcomV3DataSource` - Dexcom developer API v3 CGM stream
  (OAuth 2.0 refresh-token flow against ``api.dexcom.com``).
* :class:`FHIRObservationVitalsSource` - HL7 FHIR R4 ``Observation`` search
  with ``category=vital-signs``; works against any FHIR server that exposes
  the standard vital-sign LOINC codes.

Both adapters are **disabled by default**: their constructors raise
:class:`ConfigurationError` when the required credentials/endpoints are not
supplied (explicitly or via the documented environment variables).  No
synthetic fallback exists in any production code path.

in their own environments; Mercury Agent ships integration-ready, not
pre-integrated.  See ``docs/medical/SETUP.md`` for the full setup guide
and the contract for writing custom adapters against other vendors.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from omni_mercury_engine.security.safe_http import SafeHTTPClient, UnsafeURLError

logger = logging.getLogger(__name__)

# Public OAuth2 token endpoint (Dexcom Developer API v2 issues tokens for v3).
_DEXCOM_TOKEN_PATH: Final[str] = "/v2/oauth2/token"  # noqa: S105 - URL path, not a secret
_DEXCOM_EGVS_PATH: Final[str] = "/v3/users/self/egvs"
_DEXCOM_PROD_BASE: Final[str] = "https://api.dexcom.com"
_DEXCOM_SANDBOX_BASE: Final[str] = "https://sandbox-api.dexcom.com"

# Tuple of fully-qualified Dexcom base URLs accepted by the v3 adapter.
# DEXCOM_BASE_URL from the environment, or an explicit ``DexcomConfig.base_url``,
# must equal one of these.  Restricting to the published Dexcom hosts blocks an
# operator from accidentally pointing the adapter at an attacker-controlled
# domain that happens to speak the Dexcom JSON schema; combined with the
# SafeHTTPClient gates this means a typo or hostile env-var cannot redirect
# patient credentials off-vendor.
_DEXCOM_ALLOWED_BASES: Final[tuple[str, ...]] = (_DEXCOM_PROD_BASE, _DEXCOM_SANDBOX_BASE)

# LOINC vital-sign codes consumed by :class:`FHIRObservationVitalsSource`.
_LOINC_HR: Final[str] = "8867-4"
_LOINC_SBP: Final[str] = "8480-6"
_LOINC_DBP: Final[str] = "8462-4"
_LOINC_MAP: Final[str] = "8478-0"
_LOINC_SPO2: Final[str] = "2708-6"
_LOINC_SPO2_ALT: Final[str] = "59408-5"
_LOINC_ETCO2: Final[str] = "19911-5"

# Canonical FHIR ``CodeSystem`` URI for LOINC, per
# https://hl7.org/fhir/loinc.html .  The spec assigns LOINC exactly one
# system URI; any other ``Observation.code.coding[].system`` value is
# either a different terminology (SNOMED CT, UCUM, vendor-local) or a
# spoofed lookalike (e.g. ``http://evil-loinc.org``).  We match exactly
# rather than via a substring/suffix check so an attacker-controlled
# system URI cannot impersonate LOINC -- the latter is the
# ``py/incomplete-url-substring-sanitization`` weakness CodeQL flags as
# a high-severity finding.
_LOINC_SYSTEM_URI: Final[str] = "http://loinc.org"


class ConfigurationError(RuntimeError):
    """Raised when a data source is instantiated without required credentials.

    Mercury Agent's medical detectors never fabricate readings; if the operator
    has not configured a real adapter the system refuses to start so the
    misconfiguration is surfaced loudly instead of silently degraded.
    """


class DataSourceError(RuntimeError):
    """Raised when an adapter cannot fulfil a fetch request."""


@dataclass(frozen=True)
class CGMReading:
    """Single continuous-glucose-monitor sample.

    Attributes:
        timestamp: UTC timestamp of the sample (``datetime`` with ``tzinfo``).
        value_mg_dl: Glucose value in milligrams per decilitre.
        trend: Vendor-reported trend string (e.g. ``"flat"``, ``"rising"``)
            or ``None`` when the vendor does not supply one.
        trend_rate_mg_dl_per_min: Vendor-reported rate-of-change or ``None``.
        source: Free-form identifier of the originating adapter
            (e.g. ``"dexcom_v3"``).
    """

    timestamp: datetime
    value_mg_dl: float
    trend: str | None = None
    trend_rate_mg_dl_per_min: float | None = None
    source: str = "unknown"


@dataclass(frozen=True)
class VitalsReading:
    """Snapshot of operating-room vital signs at a single moment.

    Any field may be ``None`` if the source did not report that channel for
    the timestamp; consumers are expected to handle missing channels rather
    than receive synthetic placeholder values.

    Attributes:
        timestamp: UTC timestamp of the snapshot.
        map_mmhg: Mean arterial pressure (mmHg).
        hr_bpm: Heart rate (beats per minute).
        spo2_pct: Peripheral oxygen saturation (percent).
        etco2_mmhg: End-tidal carbon dioxide (mmHg).
        source: Free-form adapter identifier (e.g. ``"fhir_observation"``).
        extra: Raw additional fields preserved verbatim (for adapter-specific
            diagnostics; never used by the rule engines).
    """

    timestamp: datetime
    map_mmhg: float | None = None
    hr_bpm: float | None = None
    spo2_pct: float | None = None
    etco2_mmhg: float | None = None
    source: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)


class CGMDataSource(ABC):
    """Abstract contract every CGM adapter must implement.

    Subclass this to add support for additional CGM vendors (Abbott LibreView,
    Medtronic CareLink, etc.).  All methods must operate on **real** vendor
    data; mocking or fabricating readings violates the platform contract and
    is rejected at code review.
    """

    name: str = "abstract_cgm_source"

    @abstractmethod
    def fetch_recent_readings(self, window_minutes: int = 180) -> list[CGMReading]:
        """Return the most recent CGM samples within ``window_minutes``.

        Implementations must hit the real vendor API.  Returning an empty
        list is acceptable when the API itself reports no samples for the
        requested window; returning synthetic samples is not.

        Args:
            window_minutes: Look-back window in minutes (1 - 1440).

        Returns:
            Chronologically ordered list of :class:`CGMReading`.

        Raises:
            DataSourceError: If the adapter cannot retrieve readings.
            ValueError: If ``window_minutes`` is outside ``[1, 1440]``.
        """
        raise NotImplementedError


class VitalsDataSource(ABC):
    """Abstract contract every operating-room vitals adapter must implement.

    Reference implementations live alongside this class
    (:class:`FHIRObservationVitalsSource`).  Vendor SDKs (Philips IntelliVue,
    GE CARESCAPE, Mindray BeneVision) can subclass this to publish their
    feed under the same contract.
    """

    name: str = "abstract_vitals_source"

    @abstractmethod
    def fetch_recent_vitals(self, window_minutes: int = 5) -> list[VitalsReading]:
        """Return the most recent vitals snapshots within ``window_minutes``.

        Args:
            window_minutes: Look-back window in minutes (1 - 1440).

        Returns:
            Chronologically ordered list of :class:`VitalsReading`.

        Raises:
            DataSourceError: If the adapter cannot retrieve readings.
            ValueError: If ``window_minutes`` is outside ``[1, 1440]``.
        """
        raise NotImplementedError


def _validate_window(window_minutes: int) -> int:
    """Validate and coerce a fetch window in minutes."""
    if not isinstance(window_minutes, int) or window_minutes < 1 or window_minutes > 1440:
        raise ValueError(f"window_minutes must be an int in [1, 1440], got {window_minutes!r}")
    return window_minutes


def _parse_dexcom_timestamp(value: str) -> datetime:
    """Parse a Dexcom timestamp string into an aware UTC ``datetime``.

    Dexcom v3 emits ``systemTime`` and ``displayTime`` in ISO 8601 with a
    trailing ``Z`` (UTC).  Some sandbox payloads omit the ``Z``; we treat
    naive timestamps as UTC and stamp the timezone explicitly.
    """
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_dexcom_egvs_payload(payload: dict[str, Any]) -> list[CGMReading]:
    """Parse a Dexcom v3 ``/users/self/egvs`` JSON payload.

    Exposed as a module-level function so it can be exercised with sanitized
    fixtures in unit tests without touching the network.

    Args:
        payload: Parsed JSON dictionary returned by the Dexcom API.

    Returns:
        Chronologically ordered list of :class:`CGMReading` (oldest first).

    Raises:
        DataSourceError: If the payload schema is unexpected.
    """
    records = payload.get("records")
    if not isinstance(records, list):
        raise DataSourceError("Dexcom payload is missing 'records' array")
    readings: list[CGMReading] = []
    for record in records:
        if not isinstance(record, dict):
            raise DataSourceError("Dexcom 'records' entry is not an object")
        system_time = record.get("systemTime") or record.get("displayTime")
        value = record.get("value")
        if not isinstance(system_time, str) or value is None:
            raise DataSourceError("Dexcom record missing required 'systemTime'/'value' fields")
        try:
            value_mg_dl = float(value)
        except (TypeError, ValueError) as exc:
            raise DataSourceError(f"Dexcom value is not numeric: {value!r}") from exc
        trend_raw = record.get("trend")
        trend = trend_raw if isinstance(trend_raw, str) else None
        rate_raw = record.get("trendRate")
        try:
            trend_rate = float(rate_raw) if rate_raw is not None else None
        except (TypeError, ValueError):
            trend_rate = None
        readings.append(
            CGMReading(
                timestamp=_parse_dexcom_timestamp(system_time),
                value_mg_dl=value_mg_dl,
                trend=trend,
                trend_rate_mg_dl_per_min=trend_rate,
                source="dexcom_v3",
            )
        )
    readings.sort(key=lambda r: r.timestamp)
    return readings


def _parse_fhir_instant(value: str) -> datetime:
    """Parse a FHIR ``instant`` / ``dateTime`` string to aware UTC ``datetime``."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _loinc_codes(resource: dict[str, Any]) -> list[str]:
    """Return the LOINC codes attached to a FHIR Observation resource.

    Only ``coding`` entries whose ``system`` field equals the canonical
    LOINC ``CodeSystem`` URI (:data:`_LOINC_SYSTEM_URI`) are accepted.
    The previous implementation used ``endswith("loinc.org")``, which
    CodeQL flags as ``py/incomplete-url-substring-sanitization``: an
    attacker-controlled bundle could supply ``http://evil-loinc.org``
    and have its codes silently treated as authoritative LOINC values,
    redirecting clinical interpretation.  The FHIR specification
    assigns LOINC exactly one system URI, so exact-match is both
    sufficient and correct (https://hl7.org/fhir/loinc.html).
    """
    coding = resource.get("code", {}).get("coding", [])
    if not isinstance(coding, list):
        return []
    return [
        c.get("code", "")
        for c in coding
        if isinstance(c, dict) and c.get("system") == _LOINC_SYSTEM_URI
    ]


def _component_value(components: list[dict[str, Any]] | None, loinc_code: str) -> float | None:
    """Return the value of a sub-component matching ``loinc_code``, if any."""
    if not components:
        return None
    for comp in components:
        if not isinstance(comp, dict):
            continue
        codes = _loinc_codes(comp)
        if loinc_code in codes:
            qty = comp.get("valueQuantity", {})
            if isinstance(qty, dict) and qty.get("value") is not None:
                try:
                    return float(qty["value"])
                except (TypeError, ValueError):
                    return None
    return None


def parse_fhir_observation_bundle(payload: dict[str, Any]) -> list[VitalsReading]:
    """Parse a FHIR R4 ``Bundle`` of vital-sign ``Observation`` resources.

    Recognised LOINC codes:

    * ``8867-4`` heart rate (bpm)
    * ``8480-6`` systolic BP (mmHg) and ``8462-4`` diastolic BP (mmHg) -
      combined into MAP = ``(SBP + 2 * DBP) / 3`` when both are present
    * ``8478-0`` mean BP (mmHg) - used directly when reported
    * ``2708-6`` / ``59408-5`` SpO2 (percent)
    * ``19911-5`` end-tidal CO2 (mmHg)

    Observations sharing the same ``effectiveDateTime`` are merged into a
    single :class:`VitalsReading` snapshot.  Returns an empty list if the
    bundle contains no recognised vitals; never fabricates missing channels.

    Args:
        payload: Parsed FHIR Bundle JSON.

    Returns:
        Chronologically ordered list of :class:`VitalsReading`.

    Raises:
        DataSourceError: If the bundle schema is invalid.
    """
    if payload.get("resourceType") != "Bundle":
        raise DataSourceError("Payload is not a FHIR Bundle")
    entries = payload.get("entry")
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise DataSourceError("FHIR Bundle 'entry' is not an array")

    snapshots: dict[datetime, dict[str, float]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        resource = entry.get("resource")
        if not isinstance(resource, dict):
            continue
        if resource.get("resourceType") != "Observation":
            continue
        effective = resource.get("effectiveDateTime") or resource.get("issued")
        if not isinstance(effective, str):
            continue
        try:
            timestamp = _parse_fhir_instant(effective)
        except ValueError:
            continue

        bucket = snapshots.setdefault(timestamp, {})
        codes = _loinc_codes(resource)
        components = resource.get("component")
        if not isinstance(components, list):
            components = None

        qty = resource.get("valueQuantity")
        value: float | None = None
        if isinstance(qty, dict) and qty.get("value") is not None:
            try:
                value = float(qty["value"])
            except (TypeError, ValueError):
                value = None

        if _LOINC_HR in codes and value is not None:
            bucket["hr_bpm"] = value
        if _LOINC_MAP in codes and value is not None:
            bucket["map_mmhg"] = value
        if (_LOINC_SPO2 in codes and value is not None) or (
            _LOINC_SPO2_ALT in codes and value is not None
        ):
            bucket["spo2_pct"] = value
        if _LOINC_ETCO2 in codes and value is not None:
            bucket["etco2_mmhg"] = value

        sbp = _component_value(components, _LOINC_SBP)
        dbp = _component_value(components, _LOINC_DBP)
        if sbp is not None and dbp is not None and "map_mmhg" not in bucket:
            bucket["map_mmhg"] = (sbp + 2.0 * dbp) / 3.0

    readings = [
        VitalsReading(
            timestamp=ts,
            map_mmhg=values.get("map_mmhg"),
            hr_bpm=values.get("hr_bpm"),
            spo2_pct=values.get("spo2_pct"),
            etco2_mmhg=values.get("etco2_mmhg"),
            source="fhir_observation",
        )
        for ts, values in snapshots.items()
        if values
    ]
    readings.sort(key=lambda r: r.timestamp)
    return readings


def _validate_dexcom_base_url(base_url: str) -> str:
    """Validate an operator-supplied Dexcom base URL against the published hosts.

    The Dexcom Developer API publishes exactly two base URLs - the
    production endpoint at ``https://api.dexcom.com`` and the sandbox at
    ``https://sandbox-api.dexcom.com``.  Anything else is either an
    operator typo or a hostile redirect of patient credentials away from
    Dexcom; both are unacceptable.  Restricting the allowlist here gives
    misconfiguration a hard, loud surface (:class:`ConfigurationError`)
    instead of letting it slip through to runtime and hitting
    SafeHTTPClient's downstream gates.

    Args:
        base_url: The base URL to validate.

    Returns:
        The validated URL stripped of any trailing slash.

    Raises:
        ConfigurationError: ``base_url`` is not one of the published
            Dexcom hosts.
    """
    trimmed = base_url.rstrip("/")
    if trimmed not in _DEXCOM_ALLOWED_BASES:
        raise ConfigurationError(
            "DEXCOM_BASE_URL must be one of "
            f"{list(_DEXCOM_ALLOWED_BASES)}; got {base_url!r}. "
            "The Dexcom Developer API publishes only the production and sandbox "
            "hosts; any other value is rejected as an operator typo or a "
            "credential-redirect attempt."
        )
    return trimmed


@dataclass(frozen=True)
class DexcomConfig:
    """Resolved configuration for :class:`DexcomV3DataSource`.

    Attributes:
        client_id: OAuth2 client id (``DEXCOM_CLIENT_ID``).
        client_secret: OAuth2 client secret (``DEXCOM_CLIENT_SECRET``).
        refresh_token: Long-lived refresh token issued after the user
            authorises the application (``DEXCOM_REFRESH_TOKEN``).
        redirect_uri: Redirect URI registered for the OAuth app
            (``DEXCOM_REDIRECT_URI``); required by the token endpoint.
        base_url: API base URL.  Defaults to the production endpoint;
            point to ``https://sandbox-api.dexcom.com`` for testing.  Any
            other value is rejected by
            :func:`_validate_dexcom_base_url` at construction time.
    """

    client_id: str
    client_secret: str
    refresh_token: str
    redirect_uri: str
    base_url: str = _DEXCOM_PROD_BASE

    def __post_init__(self) -> None:
        """Enforce the Dexcom base-URL allowlist at construction time."""
        validated = _validate_dexcom_base_url(self.base_url)
        # ``frozen=True`` blocks normal attribute assignment; use
        # ``object.__setattr__`` to canonicalise the URL (strip any
        # trailing slash) without bypassing the immutability contract
        # for callers.
        object.__setattr__(self, "base_url", validated)


class DexcomV3DataSource(CGMDataSource):
    """CGM adapter for the Dexcom Developer API v3.

    Implements the OAuth 2.0 *refresh-token* flow against
    ``/v2/oauth2/token`` and fetches estimated glucose values (EGVs) from
    ``/v3/users/self/egvs``.  Access tokens are refreshed in-process when
    they expire; refresh tokens themselves rotate per Dexcom policy and
    are not persisted by this adapter - integrators should wire the
    rotation into their own secret store.

    Constructor parameters take precedence over environment variables.

    Required environment variables (when no explicit ``config`` is given):

    * ``DEXCOM_CLIENT_ID``
    * ``DEXCOM_CLIENT_SECRET``
    * ``DEXCOM_REFRESH_TOKEN``
    * ``DEXCOM_REDIRECT_URI``

    Optional:

    * ``DEXCOM_BASE_URL`` - override (e.g. for the sandbox endpoint).
    """

    name = "dexcom_v3"

    def __init__(
        self,
        config: DexcomConfig | None = None,
        *,
        timeout_seconds: float = 15.0,
        user_agent: str = "Mercury-Agent/1.7 Endocrinology",
        clock: object = datetime,
    ) -> None:
        """Initialise the Dexcom adapter.

        Args:
            config: Explicit credentials.  When ``None`` the constructor
                reads ``DEXCOM_CLIENT_ID`` / ``DEXCOM_CLIENT_SECRET`` /
                ``DEXCOM_REFRESH_TOKEN`` / ``DEXCOM_REDIRECT_URI`` from the
                environment and raises :class:`ConfigurationError` if any
                are missing.
            timeout_seconds: HTTP timeout per request.
            user_agent: HTTP ``User-Agent`` header.
            clock: Object exposing ``now(tz)``; defaults to :class:`datetime`.
                Injected for deterministic tests.

        Raises:
            ConfigurationError: If credentials are not supplied.
        """
        if config is None:
            config = self._config_from_env()
        self._config = config
        self._timeout = float(timeout_seconds)
        self._user_agent = user_agent
        self._clock = clock
        self._access_token: str | None = None
        self._access_token_expires_at: datetime | None = None

    @staticmethod
    def _config_from_env() -> DexcomConfig:
        client_id = os.environ.get("DEXCOM_CLIENT_ID")
        client_secret = os.environ.get("DEXCOM_CLIENT_SECRET")
        refresh_token = os.environ.get("DEXCOM_REFRESH_TOKEN")
        redirect_uri = os.environ.get("DEXCOM_REDIRECT_URI")
        missing = [
            name
            for name, value in (
                ("DEXCOM_CLIENT_ID", client_id),
                ("DEXCOM_CLIENT_SECRET", client_secret),
                ("DEXCOM_REFRESH_TOKEN", refresh_token),
                ("DEXCOM_REDIRECT_URI", redirect_uri),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(
                "DexcomV3DataSource is disabled by default. "
                "Set the following environment variables or pass an explicit "
                f"DexcomConfig: {', '.join(missing)}. "
                "See docs/medical/SETUP.md#dexcom-v3."
            )
        base_url = os.environ.get("DEXCOM_BASE_URL", _DEXCOM_PROD_BASE)
        # ``missing`` is empty so the four values are guaranteed non-None.
        assert client_id is not None
        assert client_secret is not None
        assert refresh_token is not None
        assert redirect_uri is not None
        return DexcomConfig(
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=refresh_token,
            redirect_uri=redirect_uri,
            base_url=base_url,
        )

    @property
    def config(self) -> DexcomConfig:
        """Return the resolved configuration (credentials redacted in logs)."""
        return self._config

    def _now_utc(self) -> datetime:
        # ``self._clock`` is the ``datetime`` type by default; tests inject a
        # frozen stub exposing the same ``now(tz)`` signature.
        return self._clock.now(UTC)  # type: ignore[attr-defined]

    def _refresh_access_token(self) -> str:
        """Exchange the refresh token for a fresh access token.

        Routes through :class:`SafeHTTPClient` so every Dexcom call goes
        through Mercury's central egress gate (scheme allowlist, IP
        validation, DNS-rebinding pin, redirect refusal).  The base URL
        is restricted by :class:`DexcomConfig` to the published Dexcom
        prod/sandbox hosts, so ``user_configured=True`` here passes the
        private-network / IMDS gate without needing the trusted-host
        allowlist.
        """
        url = self._config.base_url + _DEXCOM_TOKEN_PATH
        try:
            payload = SafeHTTPClient.post_form(
                url,
                form_data={
                    "client_id": self._config.client_id,
                    "client_secret": self._config.client_secret,
                    "refresh_token": self._config.refresh_token,
                    "grant_type": "refresh_token",
                    "redirect_uri": self._config.redirect_uri,
                },
                headers={
                    "User-Agent": self._user_agent,
                    "Accept": "application/json",
                },
                timeout=self._timeout,
                user_configured=True,
            )
        except UnsafeURLError as exc:
            raise DataSourceError(f"Dexcom token refresh failed: {exc}") from exc
        except Exception as exc:
            raise DataSourceError(f"Dexcom token refresh failed: {exc}") from exc

        if not isinstance(payload, dict):
            raise DataSourceError("Dexcom token response is not a JSON object")
        access_token = payload.get("access_token")
        expires_in = payload.get("expires_in")
        if not isinstance(access_token, str) or not isinstance(expires_in, int):
            raise DataSourceError("Dexcom token response is missing access_token/expires_in")
        self._access_token = access_token
        # Subtract a 30 s safety margin so we refresh before the server
        # rejects the token.
        self._access_token_expires_at = self._now_utc() + timedelta(seconds=max(expires_in - 30, 0))
        return access_token

    def _ensure_access_token(self) -> str:
        if self._access_token is not None and self._access_token_expires_at is not None:
            if self._now_utc() < self._access_token_expires_at:
                return self._access_token
        return self._refresh_access_token()

    def fetch_recent_readings(self, window_minutes: int = 180) -> list[CGMReading]:
        """Fetch CGM readings for the last ``window_minutes`` minutes.

        Args:
            window_minutes: Look-back window in minutes (1 - 1440).  The
                Dexcom v3 API caps queries at 24 h of history.

        Returns:
            Chronologically ordered list of :class:`CGMReading`.

        Raises:
            DataSourceError: If the API returns an error or invalid payload.
            ValueError: If ``window_minutes`` is outside ``[1, 1440]``.
        """
        _validate_window(window_minutes)
        access_token = self._ensure_access_token()
        end = self._now_utc().replace(microsecond=0)
        start = end - timedelta(minutes=window_minutes)
        url = self._config.base_url + _DEXCOM_EGVS_PATH
        params: dict[str, str] = {
            "startDate": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "endDate": end.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            payload = SafeHTTPClient.get_json(
                url,
                params=params,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                    "User-Agent": self._user_agent,
                },
                timeout=self._timeout,
                user_configured=True,
            )
        except UnsafeURLError as exc:
            raise DataSourceError(f"Dexcom EGVs request failed: {exc}") from exc
        except Exception as exc:
            raise DataSourceError(f"Dexcom EGVs request failed: {exc}") from exc
        return parse_dexcom_egvs_payload(payload)


@dataclass(frozen=True)
class FHIRConfig:
    """Resolved configuration for :class:`FHIRObservationVitalsSource`.

    Attributes:
        base_url: Root URL of the FHIR R4 server (no trailing slash).  Must
            start with ``https://`` unless ``allow_http=True`` is supplied
            for an explicit local/dev deployment.
        patient_id: Logical id of the patient resource being monitored.
        bearer_token: Optional pre-issued bearer token.  When supplied the
            adapter sends it as ``Authorization: Bearer <token>``; open
            FHIR servers may omit this.
        allow_http: If ``True``, accept a plain ``http://`` ``base_url``.
            Off by default because vital-signs payloads are PHI and
            must traverse TLS; only flip this for an explicitly
            documented local or development FHIR server.
    """

    base_url: str
    patient_id: str
    bearer_token: str | None = None
    allow_http: bool = False

    def __post_init__(self) -> None:
        """Enforce the HTTPS-by-default policy for PHI endpoints."""
        lowered = self.base_url.lower()
        if lowered.startswith("https://"):
            return
        if lowered.startswith("http://"):
            if not self.allow_http:
                raise ConfigurationError(
                    "FHIR base_url must use the https:// scheme because PHI "
                    "must traverse TLS. Set ``allow_http=True`` (or "
                    "``FHIR_ALLOW_HTTP=1``) only for an explicitly documented "
                    f"local/development FHIR server; got {self.base_url!r}."
                )
            return
        raise ConfigurationError(
            "FHIR base_url must start with https:// (or http:// behind an "
            f"explicit allow_http opt-in); got {self.base_url!r}"
        )


class FHIRObservationVitalsSource(VitalsDataSource):
    """Vitals adapter for HL7 FHIR R4 ``Observation`` resources.

    Searches ``{base_url}/Observation?category=vital-signs&patient={pid}&...``
    and parses the standard vital-sign LOINC codes (see
    :func:`parse_fhir_observation_bundle`).  Works against any spec-compliant
    FHIR server (e.g. Epic, Cerner, SMART Health IT sandboxes).

    Constructor parameters take precedence over environment variables.

    Required environment variables (when no explicit ``config`` is given):

    * ``FHIR_BASE_URL`` - must use ``https://`` unless ``FHIR_ALLOW_HTTP=1``
      is also set for an explicitly documented local/development server.
    * ``FHIR_PATIENT_ID``

    Optional:

    * ``FHIR_BEARER_TOKEN`` - pre-issued OAuth2 bearer token.
    * ``FHIR_ALLOW_HTTP`` - set to ``1`` to permit plain ``http://`` (PHI
      traverses cleartext; only safe for local/dev FHIR sandboxes).
    """

    name = "fhir_observation"

    def __init__(
        self,
        config: FHIRConfig | None = None,
        *,
        timeout_seconds: float = 15.0,
        user_agent: str = "Mercury-Agent/1.7 Anesthesiology",
        clock: object = datetime,
    ) -> None:
        """Initialise the FHIR adapter.

        Args:
            config: Explicit configuration.  When ``None`` the constructor
                reads ``FHIR_BASE_URL`` / ``FHIR_PATIENT_ID`` /
                ``FHIR_BEARER_TOKEN`` / ``FHIR_ALLOW_HTTP`` from the
                environment and raises :class:`ConfigurationError` if the
                required ones are missing or if ``FHIR_BASE_URL`` uses
                ``http://`` without ``FHIR_ALLOW_HTTP=1``.
            timeout_seconds: HTTP timeout per request.
            user_agent: HTTP ``User-Agent`` header.
            clock: Object exposing ``now(tz)``; defaults to :class:`datetime`.

        Raises:
            ConfigurationError: If required configuration is missing or
                if ``base_url`` violates the HTTPS-by-default policy.
        """
        if config is None:
            config = self._config_from_env()
        # ``FHIRConfig.__post_init__`` enforces the scheme allowlist
        # (https:// required by default; http:// only with explicit opt-in).
        self._config = config
        self._timeout = float(timeout_seconds)
        self._user_agent = user_agent
        self._clock = clock

    @staticmethod
    def _config_from_env() -> FHIRConfig:
        base_url = os.environ.get("FHIR_BASE_URL")
        patient_id = os.environ.get("FHIR_PATIENT_ID")
        missing = [
            name
            for name, value in (
                ("FHIR_BASE_URL", base_url),
                ("FHIR_PATIENT_ID", patient_id),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(
                "FHIRObservationVitalsSource is disabled by default. "
                "Set the following environment variables or pass an explicit "
                f"FHIRConfig: {', '.join(missing)}. "
                "See docs/medical/SETUP.md#fhir-vital-signs."
            )
        assert base_url is not None
        assert patient_id is not None
        return FHIRConfig(
            base_url=base_url.rstrip("/"),
            patient_id=patient_id,
            bearer_token=os.environ.get("FHIR_BEARER_TOKEN") or None,
            allow_http=os.environ.get("FHIR_ALLOW_HTTP") == "1",
        )

    @property
    def config(self) -> FHIRConfig:
        """Return the resolved configuration."""
        return self._config

    def _now_utc(self) -> datetime:
        return self._clock.now(UTC)  # type: ignore[attr-defined]

    def fetch_recent_vitals(self, window_minutes: int = 5) -> list[VitalsReading]:
        """Fetch vital-sign observations for the last ``window_minutes`` minutes.

        Args:
            window_minutes: Look-back window in minutes (1 - 1440).

        Returns:
            Chronologically ordered list of :class:`VitalsReading`.

        Raises:
            DataSourceError: If the FHIR server returns an error or the
                bundle cannot be parsed.
            ValueError: If ``window_minutes`` is outside ``[1, 1440]``.
        """
        _validate_window(window_minutes)
        end = self._now_utc().replace(microsecond=0)
        start = end - timedelta(minutes=window_minutes)
        params: dict[str, str] = {
            "patient": self._config.patient_id,
            "category": "vital-signs",
            "date": f"ge{start.strftime('%Y-%m-%dT%H:%M:%S')}Z",
            "_sort": "-date",
        }
        url = f"{self._config.base_url}/Observation"
        headers: dict[str, str] = {
            "Accept": "application/fhir+json",
            "User-Agent": self._user_agent,
        }
        if self._config.bearer_token:
            headers["Authorization"] = f"Bearer {self._config.bearer_token}"
        try:
            payload = SafeHTTPClient.get_json(
                url,
                params=params,
                headers=headers,
                timeout=self._timeout,
                allow_http=self._config.allow_http,
                user_configured=True,
            )
        except UnsafeURLError as exc:
            raise DataSourceError(f"FHIR Observation request failed: {exc}") from exc
        except Exception as exc:
            raise DataSourceError(f"FHIR Observation request failed: {exc}") from exc
        return parse_fhir_observation_bundle(payload)


__all__ = [
    "CGMDataSource",
    "CGMReading",
    "ConfigurationError",
    "DataSourceError",
    "DexcomConfig",
    "DexcomV3DataSource",
    "FHIRConfig",
    "FHIRObservationVitalsSource",
    "VitalsDataSource",
    "VitalsReading",
    "parse_dexcom_egvs_payload",
    "parse_fhir_observation_bundle",
]
