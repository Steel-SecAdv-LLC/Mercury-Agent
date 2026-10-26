# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GCN notice ingest: NASA GCN Kafka alerts normalized for Mercury's detectors.

NASA's General Coordinates Network streams multimessenger astronomy alerts
(gravitational waves, neutrinos, gamma-ray bursts, fast radio bursts, X-ray
transients, supernova watches) over Kafka with OIDC authentication. This
module turns that stream into Mercury input, in three layers that separate
what needs a broker from what does not:

1. :func:`parse_notice` — pure normalization of one raw notice into a
   :class:`GCNNotice`. Field extraction is best-effort over the shared keys
   of GCN's JSON schemas (times, sky position, significance); the raw
   payload always rides along, so nothing a schema adds later is lost.
2. :func:`build_rate_matrix` — pure feature building: per-topic notice
   counts over fixed time windows, as the ``(n_windows, n_topics)`` float
   array the engine consumes. Multimessenger coincidences (a supernova
   lighting Super-Kamiokande and IceCube in the same window) appear as
   cross-topic correlation structure, which is what the eigenvalue-
   normalized dimensional detector and the rebuilt temporal detector see.
3. :class:`GCNNoticeConsumer` — the transport. It requires
   ``confluent-kafka`` at use (no silent mock degradation) and builds the
   same librdkafka OIDC configuration as ``scripts/gcn_kafka_keepalive.py``
   builds for the credential probe. The two builders are deliberately
   separate (the keep-alive workflow installs only ``confluent-kafka``, so
   the script cannot import this package) and are pinned equal by
   ``tests/loaders/test_gcn_notices.py``.

Credentials come from the ``GCN_KAFKA_CLIENT_ID`` / ``GCN_KAFKA_CLIENT_SECRET``
environment variables — the same names the repository's Actions secrets use —
and are issued at https://gcn.nasa.gov/quickstart.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

__all__ = [
    "DEFAULT_TOPICS",
    "GCNIngestError",
    "GCNNotice",
    "GCNNoticeConsumer",
    "build_rate_matrix",
    "consumer_config",
    "parse_notice",
]

#: The topic set this deployment's credential is subscribed to, exactly as
#: selected at https://gcn.nasa.gov/quickstart (client "mercuryagent.global").
DEFAULT_TOPICS: tuple[str, ...] = (
    "gcn.notices.boom.alert",
    "gcn.notices.chime.frb",
    "gcn.notices.dsa110.frb",
    "gcn.notices.einstein_probe.wxt.alert",
    "gcn.heartbeat",
    "gcn.notices.icecube.lvk_nu_track_search",
    "gcn.notices.icecube.gold_bronze_track_alerts",
    "igwn.gwalert",
    "gcn.notices.maxi.gsc.known",
    "gcn.notices.maxi.gsc.unknown",
    "gcn.notices.superk.sn_alert",
    "gcn.notices.swift.bat.guano",
)

#: Topic prefix/name to messenger kind. Kinds group topics whose alerts carry
#: the same physical meaning, for feature grouping and downstream routing.
_TOPIC_KINDS: tuple[tuple[str, str], ...] = (
    ("gcn.heartbeat", "heartbeat"),
    ("igwn.gwalert", "gravitational_wave"),
    ("gcn.notices.icecube", "neutrino"),
    ("gcn.notices.superk", "supernova_watch"),
    ("gcn.notices.chime", "fast_radio_burst"),
    ("gcn.notices.dsa110", "fast_radio_burst"),
    ("gcn.notices.swift", "gamma_ray_burst"),
    ("gcn.notices.einstein_probe", "xray_transient"),
    ("gcn.notices.maxi", "xray_transient"),
    ("gcn.notices.boom", "xray_transient"),
)

#: Keys tried, in order, for each shared field. GCN's unified JSON schemas
#: name these differently across missions; extraction takes the first key
#: present and never invents a value.
_TIME_KEYS = ("alert_datetime", "trigger_time", "time_created", "event_time")
_RA_KEYS = ("ra", "RA", "right_ascension")
_DEC_KEYS = ("dec", "DEC", "declination")
_FAR_KEYS = ("far",)
_SIGNIFICANCE_KEYS = ("signalness", "significance", "importance", "snr")


class GCNIngestError(ValueError):
    """A notice or configuration this ingest refuses, with the reason."""


@dataclass(frozen=True)
class GCNNotice:
    """One normalized GCN notice.

    Attributes:
        topic: Kafka topic the notice arrived on.
        kind: Messenger kind derived from the topic (e.g. ``neutrino``).
        received_at: When Mercury consumed it (UTC).
        event_time: The notice's own time, when a shared time key exists.
        ra: Right ascension in degrees, when present.
        dec: Declination in degrees, when present.
        far: False-alarm rate (Hz), when present (LVK, IceCube).
        significance: Mission significance figure, when present.
        raw: The decoded JSON payload, untouched.
    """

    topic: str
    kind: str
    received_at: datetime
    event_time: datetime | None = None
    ra: float | None = None
    dec: float | None = None
    far: float | None = None
    significance: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def topic_kind(topic: str) -> str:
    """Messenger kind for a topic; ``GCNIngestError`` for an unknown one."""
    for prefix, kind in _TOPIC_KINDS:
        if topic == prefix or topic.startswith(prefix + "."):
            return kind
    raise GCNIngestError(
        f"Unknown GCN topic {topic!r}: not one this ingest maps. Known topics "
        f"start with {sorted({p for p, _ in _TOPIC_KINDS})}. Add the topic to "
        "_TOPIC_KINDS with its messenger kind rather than consuming it unmapped."
    )


def _first(payload: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    # One level of nesting covers the schemas that wrap the event ("event"
    # in igwn.gwalert); deeper structures stay available through .raw.
    event = payload.get("event")
    if isinstance(event, dict):
        for key in keys:
            if key in event and event[key] is not None:
                return event[key]
    return None


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _as_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def parse_notice(
    topic: str,
    payload: bytes | str | dict[str, Any],
    *,
    received_at: datetime | None = None,
) -> GCNNotice:
    """Normalize one raw GCN notice.

    Args:
        topic: The Kafka topic the message arrived on.
        payload: The message value: JSON bytes/text, or an already-decoded
            mapping.
        received_at: Consumption time; defaults to now (UTC).

    Returns:
        The normalized notice, with the decoded payload preserved in ``raw``.

    Raises:
        GCNIngestError: For an unmapped topic, or a payload that is not a
            JSON object — refused rather than half-parsed, so a schema
            change surfaces instead of silently emptying every field.
    """
    kind = topic_kind(topic)
    if isinstance(payload, (bytes, str)):
        try:
            decoded = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GCNIngestError(
                f"Notice on {topic} is not valid JSON ({exc}). GCN's JSON-format "
                "topics always carry a JSON object; check the subscription uses "
                "the JSON notice format, not Text/VOEvent/Binary."
            ) from exc
    else:
        decoded = payload
    if not isinstance(decoded, dict):
        raise GCNIngestError(
            f"Notice on {topic} decoded to {type(decoded).__name__}, not an object."
        )

    return GCNNotice(
        topic=topic,
        kind=kind,
        received_at=received_at or datetime.now(UTC),
        event_time=_as_time(_first(decoded, _TIME_KEYS)),
        ra=_as_float(_first(decoded, _RA_KEYS)),
        dec=_as_float(_first(decoded, _DEC_KEYS)),
        far=_as_float(_first(decoded, _FAR_KEYS)),
        significance=_as_float(_first(decoded, _SIGNIFICANCE_KEYS)),
        raw=decoded,
    )


def build_rate_matrix(
    notices: Iterable[GCNNotice],
    *,
    window_seconds: float = 3600.0,
    topics: Sequence[str] = DEFAULT_TOPICS,
) -> tuple[np.ndarray[Any, Any], list[datetime]]:
    """Per-topic notice counts over fixed, contiguous time windows.

    The result is engine-shaped input: row ``i`` counts each topic's notices
    in window ``i``, so ``fit_fusion``/``detect_batch`` consume it directly.
    Cross-topic coincidence bursts appear as correlated rows. The heartbeat
    topic is a liveness channel, not an astronomy signal, so callers who do
    not want it as a feature column leave it out of ``topics``.

    Args:
        notices: Normalized notices; each is binned by its ``event_time``
            when present, else ``received_at``.
        window_seconds: Window width; must be positive.
        topics: Feature columns, one per topic, in this order. A notice on a
            topic not listed is refused — a silently dropped column is a
            detection gap, not a convenience.

    Returns:
        ``(matrix, window_starts)``: the ``(n_windows, len(topics))`` float64
        count matrix spanning first to last notice, and each window's start.
        Zero notices yield an empty ``(0, len(topics))`` matrix.

    Raises:
        GCNIngestError: For a non-positive window, duplicate topic columns,
            or a notice whose topic has no column.
    """
    if window_seconds <= 0:
        raise GCNIngestError(f"window_seconds must be positive; got {window_seconds}.")
    if len(set(topics)) != len(topics):
        raise GCNIngestError("topics holds duplicates; each feature column must be unique.")
    column: dict[str, int] = {topic: i for i, topic in enumerate(topics)}

    stamped: list[tuple[datetime, int]] = []
    for notice in notices:
        if notice.topic not in column:
            raise GCNIngestError(
                f"Notice on {notice.topic!r} has no feature column; pass it in "
                "topics or filter it out explicitly before building the matrix."
            )
        stamped.append((notice.event_time or notice.received_at, column[notice.topic]))

    if not stamped:
        return np.zeros((0, len(topics)), dtype=np.float64), []

    stamped.sort(key=lambda pair: pair[0])
    start = stamped[0][0]
    n_windows = int((stamped[-1][0] - start).total_seconds() // window_seconds) + 1
    matrix = np.zeros((n_windows, len(topics)), dtype=np.float64)
    for when, col in stamped:
        row = int((when - start).total_seconds() // window_seconds)
        matrix[row, col] += 1.0
    window_starts = [start + timedelta(seconds=i * window_seconds) for i in range(n_windows)]
    return matrix, window_starts


_GCN_DOMAIN = "gcn.nasa.gov"
_CA_BUNDLE_CANDIDATES = (
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
)


def _ca_bundle_location() -> str | None:
    """CA bundle path for librdkafka; mirrors the keep-alive script's logic."""
    override = os.environ.get("SSL_CERT_FILE")
    candidates = (override, *_CA_BUNDLE_CANDIDATES) if override else _CA_BUNDLE_CANDIDATES
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def consumer_config(
    client_id: str,
    client_secret: str,
    *,
    group_id: str = "mercury-gcn-ingest",
    domain: str = _GCN_DOMAIN,
) -> dict[str, Any]:
    """Build the librdkafka configuration for a GCN SASL_SSL/OAUTHBEARER connection.

    The same shape ``gcn-kafka`` builds internally and the same block
    ``scripts/gcn_kafka_keepalive.py`` uses for the credential probe; the
    keep-alive workflow installs only ``confluent-kafka``, so the script
    keeps its own copy, and the parity test in
    ``tests/loaders/test_gcn_notices.py`` pins the two equal key-for-key.
    """
    config: dict[str, Any] = {
        "bootstrap.servers": f"kafka.{domain}:9092",
        "group.id": group_id,
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms": "OAUTHBEARER",
        "sasl.oauthbearer.method": "oidc",
        "sasl.oauthbearer.client.id": client_id,
        "sasl.oauthbearer.client.secret": client_secret,
        "sasl.oauthbearer.token.endpoint.url": f"https://auth.{domain}/oauth2/token",
    }
    if ca := _ca_bundle_location():
        # Both links need the CA: ssl.ca.location for the Kafka TLS
        # connection, https.ca.location for librdkafka's own HTTPS client
        # that fetches the OIDC token (its built-in CA path is the build
        # machine's, absent on Ubuntu since confluent-kafka 2.16).
        config["ssl.ca.location"] = ca
        config["https.ca.location"] = ca
    return config


class GCNNoticeConsumer:
    """Consume GCN Kafka topics as normalized :class:`GCNNotice` records.

    Requires ``confluent-kafka`` (installed by the ``gcn`` extra or
    ``pip install confluent-kafka``) and a GCN credential; construction
    fails loudly without either — this transport never degrades to a mock.

    Example:
        >>> consumer = GCNNoticeConsumer.from_env()   # doctest: +SKIP
        >>> consumer.subscribe()                      # doctest: +SKIP
        >>> for notice in consumer.consume(timeout=1.0):  # doctest: +SKIP
        ...     print(notice.topic, notice.kind)
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        group_id: str = "mercury-gcn-ingest",
        domain: str = _GCN_DOMAIN,
        consumer_factory: Any | None = None,
    ) -> None:
        """Build the underlying Kafka consumer.

        Args:
            client_id: GCN OIDC client id.
            client_secret: GCN OIDC client secret.
            group_id: Kafka consumer group.
            domain: GCN deployment (production by default).
            consumer_factory: Test seam: a callable taking the librdkafka
                config dict and returning a consumer. None means the real
                ``confluent_kafka.Consumer``.

        Raises:
            GCNIngestError: When the credential is missing, or
                ``confluent-kafka`` is not installed and no factory is given.
        """
        if not client_id or not client_secret:
            raise GCNIngestError(
                "GCN credentials are required: pass client_id/client_secret or "
                "use GCNNoticeConsumer.from_env() with GCN_KAFKA_CLIENT_ID and "
                "GCN_KAFKA_CLIENT_SECRET set. Issue credentials at "
                "https://gcn.nasa.gov/quickstart."
            )
        config = consumer_config(client_id, client_secret, group_id=group_id, domain=domain)
        factory = consumer_factory
        if factory is None:
            try:
                import confluent_kafka
            except ImportError as exc:
                raise GCNIngestError(
                    "confluent-kafka is required for the GCN ingest transport "
                    "(pip install confluent-kafka, or the gcn-kafka wrapper). "
                    "Silent mock degradation is not permitted; the parsing and "
                    "feature layers (parse_notice, build_rate_matrix) work "
                    "without it."
                ) from exc
            factory = confluent_kafka.Consumer
        self._consumer = factory(config)

    @classmethod
    def from_env(cls, **kwargs: Any) -> GCNNoticeConsumer:
        """Build from ``GCN_KAFKA_CLIENT_ID`` / ``GCN_KAFKA_CLIENT_SECRET``."""
        return cls(
            os.environ.get("GCN_KAFKA_CLIENT_ID", ""),
            os.environ.get("GCN_KAFKA_CLIENT_SECRET", ""),
            **kwargs,
        )

    def subscribe(self, topics: Sequence[str] = DEFAULT_TOPICS) -> None:
        """Subscribe to *topics*, refusing any this ingest cannot map."""
        for topic in topics:
            topic_kind(topic)  # raises GCNIngestError for an unmapped topic
        self._consumer.subscribe(list(topics))

    def consume(self, *, timeout: float = 1.0, max_messages: int = 100) -> Iterator[GCNNotice]:
        """Yield normalized notices from one consume call.

        A message with a broker error is raised as :class:`GCNIngestError`
        (never skipped silently); an unparseable payload raises from
        :func:`parse_notice` with the topic named.
        """
        for message in self._consumer.consume(max_messages, timeout):
            error = message.error()
            if error:
                raise GCNIngestError(f"GCN broker error on consume: {error}")
            topic, value = message.topic(), message.value()
            if topic is None or value is None:
                # A tombstone or topicless message carries no notice; GCN's
                # alert topics never publish these, so name it rather than
                # guessing at a payload.
                raise GCNIngestError(
                    f"GCN message without {'a topic' if topic is None else 'a payload'} "
                    f"on {topic or 'an unknown topic'}; not a notice."
                )
            yield parse_notice(topic, value)

    def close(self) -> None:
        """Close the underlying consumer."""
        self._consumer.close()
