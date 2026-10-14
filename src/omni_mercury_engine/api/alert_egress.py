# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Post-detection alert egress to SIEM / observability platforms.

Every detection route records its result through
:func:`omni_mercury_engine.api.routes.export.record_detection`; that choke
point hands each record that flagged at least one anomaly to
:func:`schedule_alert`, which forwards it -- fire-and-forget, never
blocking or failing the request -- to the sinks configured in the
``MERCURY_ALERT_SINKS`` environment variable through
:class:`~omni_mercury_engine.integrations.cross_platform_hub.CrossPlatformHub`
(Splunk HEC, Datadog, Elastic / OpenSearch, Prometheus pushgateway, an
OTLP collector, or a generic HTTP receiver).

``MERCURY_ALERT_SINKS`` is a JSON list; each sink::

    {
      "name": "soc-splunk",              # required, unique
      "platform": "splunk",              # PlatformType value
      "endpoint": "https://splunk:8088", # required
      "auth_type": "api_key",            # none | api_key | basic
      "credentials_env": {"api_key": "SPLUNK_HEC_TOKEN"},
      "max_tlp": "AMBER+STRICT",         # most restrictive colour allowed out
      "min_score": 0.0,                  # skip weaker alerts
      "index": "security",               # optional (Splunk / Elastic)
      "sourcetype": "mercury:anomaly",   # optional (Splunk)
      "timeout_seconds": 10,
      "ssl_verify": true
    }

Secrets are read from the environment variables named in
``credentials_env``; they never appear in the JSON. Each alert carries the
detection's TLP 2.0 colour (the same classification the export API uses),
and a sink receives only alerts at or below its ``max_tlp``: the default
``AMBER+STRICT`` keeps ``TLP:RED`` (named recipients only) from ever being
forwarded automatically. ``MERCURY_OFFLINE`` is enforced by the hub's
adapters: a non-loopback sink is refused before any socket opens and the
refusal is counted as ``suppressed_offline``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.compliance.tlp_handler import TLPColor

if TYPE_CHECKING:
    from omni_mercury_engine.api.routes.export import DetectionRecord
    from omni_mercury_engine.integrations.cross_platform_hub import (
        AnomalyEvent,
        CrossPlatformHub,
    )

logger = logging.getLogger(__name__)

#: Environment variable holding the JSON sink list.
ALERT_SINKS_ENV = "MERCURY_ALERT_SINKS"

_TLP_BY_VALUE: dict[str, TLPColor] = {color.value: color for color in TLPColor}


@dataclass(frozen=True)
class AlertSink:
    """One configured alert destination."""

    name: str
    platform: str
    endpoint: str
    auth_type: str = "none"
    credentials_env: dict[str, str] = field(default_factory=dict)
    max_tlp: TLPColor = TLPColor.AMBER_STRICT
    min_score: float = 0.0
    index: str | None = None
    sourcetype: str | None = None
    timeout_seconds: float = 10.0
    ssl_verify: bool = True


def load_sink_configs(raw: str | None) -> list[AlertSink]:
    """Parse and validate a ``MERCURY_ALERT_SINKS`` value.

    Args:
        raw: The JSON text (``None`` or blank means no sinks).

    Returns:
        The validated sinks.

    Raises:
        ValueError: For malformed JSON, a missing ``name`` / ``endpoint``,
            a duplicate name, an unknown platform, auth type or TLP colour,
            or a ``min_score`` outside ``[0, 1]``.
    """
    from omni_mercury_engine.integrations.cross_platform_hub import PlatformType

    if raw is None or not raw.strip():
        return []
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{ALERT_SINKS_ENV} is not valid JSON: {exc}") from exc
    if not isinstance(entries, list):
        raise ValueError(f"{ALERT_SINKS_ENV} must be a JSON list of sink objects")

    platforms = {p.value for p in PlatformType}
    sinks: list[AlertSink] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"sink #{i} must be an object")
        name = str(entry.get("name", "")).strip()
        endpoint = str(entry.get("endpoint", "")).strip().rstrip("/")
        if not name or not endpoint:
            raise ValueError(f"sink #{i} needs a non-empty 'name' and 'endpoint'")
        if name in seen:
            raise ValueError(f"duplicate sink name {name!r}")
        seen.add(name)
        platform = str(entry.get("platform", "custom")).strip().lower()
        if platform not in platforms:
            raise ValueError(f"sink {name!r}: unknown platform {platform!r}")
        auth_type = str(entry.get("auth_type", "none")).strip().lower()
        if auth_type not in ("none", "api_key", "basic"):
            raise ValueError(f"sink {name!r}: auth_type must be none, api_key or basic")
        tlp_value = str(entry.get("max_tlp", TLPColor.AMBER_STRICT.value)).strip().upper()
        tlp_value = tlp_value.removeprefix("TLP:")
        if tlp_value not in _TLP_BY_VALUE:
            raise ValueError(f"sink {name!r}: unknown max_tlp {tlp_value!r}")
        min_score = float(entry.get("min_score", 0.0))
        if not 0.0 <= min_score <= 1.0:
            raise ValueError(f"sink {name!r}: min_score must be in [0, 1]")
        credentials_env = entry.get("credentials_env", {}) or {}
        if not isinstance(credentials_env, dict):
            raise ValueError(f"sink {name!r}: credentials_env must be an object")
        sinks.append(
            AlertSink(
                name=name,
                platform=platform,
                endpoint=endpoint,
                auth_type=auth_type,
                credentials_env={str(k): str(v) for k, v in credentials_env.items()},
                max_tlp=_TLP_BY_VALUE[tlp_value],
                min_score=min_score,
                index=str(entry["index"]) if entry.get("index") else None,
                sourcetype=str(entry["sourcetype"]) if entry.get("sourcetype") else None,
                timeout_seconds=float(entry.get("timeout_seconds", 10.0)),
                ssl_verify=bool(entry.get("ssl_verify", True)),
            )
        )
    return sinks


def _severity(score: float) -> str:
    if score >= 0.9:
        return "critical"
    if score >= 0.7:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


class AlertEgress:
    """Forward anomalous detections to the configured sinks, TLP-gated."""

    def __init__(self, sinks: list[AlertSink]) -> None:
        """Register one hub platform per sink."""
        from omni_mercury_engine.integrations.cross_platform_hub import (
            CrossPlatformHub,
            PlatformConfig,
            PlatformType,
        )

        self.sinks = list(sinks)
        self.hub: CrossPlatformHub = CrossPlatformHub(enable_correlation=False)
        for sink in self.sinks:
            credentials = {
                key: os.environ.get(env_name, "") for key, env_name in sink.credentials_env.items()
            }
            metadata: dict[str, Any] = {}
            if sink.index:
                metadata["index"] = sink.index
            if sink.sourcetype:
                metadata["sourcetype"] = sink.sourcetype
            self.hub.register_platform(
                sink.name,
                PlatformConfig(
                    platform_type=PlatformType(sink.platform),
                    name=sink.name,
                    endpoint=sink.endpoint,
                    auth_type=sink.auth_type,
                    credentials=credentials,
                    timeout_seconds=sink.timeout_seconds,
                    ssl_verify=sink.ssl_verify,
                    metadata=metadata,
                ),
            )
        self.stats: dict[str, int] = {
            "alerts": 0,
            "delivered": 0,
            "failed": 0,
            "withheld_tlp": 0,
            "below_min_score": 0,
            "suppressed_offline": 0,
        }

    def build_event(self, record: DetectionRecord) -> tuple[AnomalyEvent, TLPColor]:
        """The hub event for *record* and its TLP colour (no user identifiers)."""
        from omni_mercury_engine.api.routes.export import _detection_tlp
        from omni_mercury_engine.integrations.cross_platform_hub import AnomalyEvent

        classification = _detection_tlp(record)
        score = float(record.max_score)
        score = 1.0 if not math.isfinite(score) else min(max(score, 0.0), 1.0)
        labels = {"tlp": classification.color.label, "method": record.method}
        domain = (record.metadata or {}).get("domain")
        if domain:
            labels["domain"] = str(domain)
        event = AnomalyEvent(
            event_id=record.detection_id,
            timestamp=datetime.now(UTC),
            source=f"mercury-agent:{record.method}",
            severity=_severity(score),
            score=score,
            is_anomaly=True,
            metrics={"anomaly_count": float(record.anomaly_count)},
            labels=labels,
            detector_type=record.method,
            confidence=score,
            explanation=(
                f"{record.anomaly_count} anomalies flagged by {record.method} "
                f"(max score {score:.3f}); {classification.color.label}"
            ),
        )
        return event, classification.color

    async def dispatch(self, record: DetectionRecord) -> dict[str, bool]:
        """Forward one anomalous detection to every sink that may receive it.

        Returns:
            ``{sink name: delivered}`` for the sinks that were attempted.
        """
        if record.anomaly_count <= 0 or not self.sinks:
            return {}
        event, color = self.build_event(record)
        self.stats["alerts"] += 1
        targets: list[str] = []
        for sink in self.sinks:
            if color.rank > sink.max_tlp.rank:
                self.stats["withheld_tlp"] += 1
            elif event.score < sink.min_score:
                self.stats["below_min_score"] += 1
            else:
                targets.append(sink.name)
        if not targets:
            return {}
        results = await self.hub.publish_event(event, platforms=targets)
        for name, ok in results.items():
            if ok:
                self.stats["delivered"] += 1
            elif self._was_offline_refusal(name):
                self.stats["suppressed_offline"] += 1
            else:
                self.stats["failed"] += 1
        return results

    def _was_offline_refusal(self, sink_name: str) -> bool:
        """Whether *sink_name* is currently refused by the ``MERCURY_OFFLINE`` gate."""
        from omni_mercury_engine.datasets.exceptions import OfflineModeError
        from omni_mercury_engine.security.safe_http import enforce_offline_egress

        endpoint = next(s.endpoint for s in self.sinks if s.name == sink_name)
        try:
            enforce_offline_egress(endpoint)
        except OfflineModeError:
            return True
        return False

    async def shutdown(self) -> None:
        """Flush and close every sink."""
        await self.hub.disconnect_all()


_lock = threading.Lock()
_egress: AlertEgress | None = None
_configured = False
_pending: set[asyncio.Task[Any]] = set()


def get_alert_egress() -> AlertEgress | None:
    """The process-wide egress built from ``MERCURY_ALERT_SINKS`` (``None`` if unset).

    A malformed configuration is logged once and disables egress; it never
    breaks detection.
    """
    global _egress, _configured
    with _lock:
        if not _configured:
            _configured = True
            try:
                sinks = load_sink_configs(os.environ.get(ALERT_SINKS_ENV))
            except ValueError as exc:
                logger.error("Alert egress disabled: %s", exc)
                sinks = []
            _egress = AlertEgress(sinks) if sinks else None
            if _egress is not None:
                logger.info(
                    "Alert egress enabled for %d sink(s): %s",
                    len(sinks),
                    ", ".join(s.name for s in sinks),
                )
        return _egress


def reset_alert_egress() -> None:
    """Forget the cached configuration (re-read ``MERCURY_ALERT_SINKS`` next time)."""
    global _egress, _configured
    with _lock:
        _egress = None
        _configured = False


def schedule_alert(record: DetectionRecord) -> asyncio.Task[Any] | None:
    """Fire-and-forget :meth:`AlertEgress.dispatch` for *record*.

    Returns the task (``None`` when egress is off, the record flagged
    nothing, or no event loop is running). Failures are logged, never
    raised into the detection request.
    """
    if record.anomaly_count <= 0:
        return None
    egress = get_alert_egress()
    if egress is None:
        return None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None

    async def _run() -> None:
        try:
            await egress.dispatch(record)
        except Exception as exc:
            logger.error("Alert egress failed for detection %s: %s", record.detection_id, exc)

    task = loop.create_task(_run())
    _pending.add(task)
    task.add_done_callback(_pending.discard)
    return task


async def shutdown_alert_egress() -> None:
    """Wait for in-flight alerts, then close the sinks (server shutdown)."""
    if _pending:
        await asyncio.gather(*list(_pending), return_exceptions=True)
    egress = _egress
    if egress is not None:
        await egress.shutdown()


__all__ = [
    "ALERT_SINKS_ENV",
    "AlertEgress",
    "AlertSink",
    "get_alert_egress",
    "load_sink_configs",
    "reset_alert_egress",
    "schedule_alert",
    "shutdown_alert_egress",
]
