# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GCN ingest normalizes notices, builds engine-shaped rates, and stays in lockstep with the keep-alive script's connection configuration.

The parsing and feature layers are pure and fully driven here; the transport
is driven through a factory seam (never a mock posing as a broker result the
code did not produce). The config parity test loads the keep-alive script by
path and pins its librdkafka block equal to this module's, because the
keep-alive workflow installs only confluent-kafka and cannot import the
package, so the two builders are deliberate duplicates that must not drift.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.loaders.gcn_notices import (
    DEFAULT_TOPICS,
    GCNIngestError,
    GCNNoticeConsumer,
    build_rate_matrix,
    consumer_config,
    parse_notice,
    topic_kind,
)

T0 = datetime(2026, 10, 26, 12, 0, 0, tzinfo=UTC)


class TestParseNotice:
    def test_gwalert_extracts_nested_event_fields(self) -> None:
        payload = {
            "superevent_id": "S261026a",
            "alert_type": "PRELIMINARY",
            "time_created": "2026-10-26T12:00:00Z",
            "event": {"far": 9.1e-10, "significant": True},
        }
        notice = parse_notice("igwn.gwalert", json.dumps(payload).encode())
        assert notice.kind == "gravitational_wave"
        assert notice.event_time == T0
        assert notice.far == pytest.approx(9.1e-10)
        assert notice.raw["superevent_id"] == "S261026a"

    def test_icecube_track_alert(self) -> None:
        payload = {
            "alert_datetime": "2026-10-26T12:00:00Z",
            "ra": 123.4,
            "dec": -5.6,
            "signalness": 0.82,
        }
        notice = parse_notice("gcn.notices.icecube.gold_bronze_track_alerts", json.dumps(payload))
        assert notice.kind == "neutrino"
        assert notice.ra == pytest.approx(123.4)
        assert notice.dec == pytest.approx(-5.6)
        assert notice.significance == pytest.approx(0.82)

    @pytest.mark.parametrize(
        ("topic", "kind"),
        [
            ("gcn.heartbeat", "heartbeat"),
            ("gcn.notices.superk.sn_alert", "supernova_watch"),
            ("gcn.notices.chime.frb", "fast_radio_burst"),
            ("gcn.notices.dsa110.frb", "fast_radio_burst"),
            ("gcn.notices.swift.bat.guano", "gamma_ray_burst"),
            ("gcn.notices.einstein_probe.wxt.alert", "xray_transient"),
            ("gcn.notices.maxi.gsc.known", "xray_transient"),
            ("gcn.notices.maxi.gsc.unknown", "xray_transient"),
            ("gcn.notices.boom.alert", "xray_transient"),
            ("gcn.notices.icecube.lvk_nu_track_search", "neutrino"),
        ],
    )
    def test_every_subscribed_topic_maps_to_a_kind(self, topic: str, kind: str) -> None:
        assert topic_kind(topic) == kind
        notice = parse_notice(topic, {"alert_datetime": "2026-10-26T12:00:00Z"})
        assert notice.kind == kind

    def test_missing_fields_stay_none_and_raw_is_preserved(self) -> None:
        notice = parse_notice("gcn.heartbeat", {"alive": 1})
        assert notice.event_time is None
        assert notice.ra is None and notice.dec is None
        assert notice.far is None and notice.significance is None
        assert notice.raw == {"alive": 1}

    def test_non_finite_and_non_numeric_values_are_refused_as_fields(self) -> None:
        notice = parse_notice("gcn.notices.chime.frb", {"ra": "not-a-number", "dec": float("nan")})
        assert notice.ra is None and notice.dec is None

    def test_unknown_topic_is_refused_with_the_mapping_hint(self) -> None:
        with pytest.raises(GCNIngestError, match=r"Unknown GCN topic.*_TOPIC_KINDS"):
            parse_notice("gcn.notices.mystery.alert", {})

    def test_invalid_json_is_refused_naming_the_topic_and_format(self) -> None:
        with pytest.raises(GCNIngestError, match=r"igwn\.gwalert.*JSON"):
            parse_notice("igwn.gwalert", b"\x00\x01 not json")

    def test_non_object_json_is_refused(self) -> None:
        with pytest.raises(GCNIngestError, match="not an object"):
            parse_notice("gcn.heartbeat", b"[1, 2, 3]")


class TestRateMatrix:
    @staticmethod
    def _notice(topic: str, minutes: float) -> Any:
        return parse_notice(
            topic,
            {"alert_datetime": (T0 + timedelta(minutes=minutes)).isoformat()},
        )

    def test_counts_per_window_and_engine_shape(self) -> None:
        notices = [
            self._notice("gcn.notices.superk.sn_alert", 0),
            self._notice("gcn.notices.icecube.gold_bronze_track_alerts", 1),
            self._notice("gcn.notices.superk.sn_alert", 2),
            self._notice("igwn.gwalert", 65),
        ]
        matrix, starts = build_rate_matrix(notices, window_seconds=3600.0)
        assert matrix.shape == (2, len(DEFAULT_TOPICS))
        assert matrix.dtype == np.float64
        assert starts[0] == T0 and starts[1] == T0 + timedelta(hours=1)
        superk = DEFAULT_TOPICS.index("gcn.notices.superk.sn_alert")
        icecube = DEFAULT_TOPICS.index("gcn.notices.icecube.gold_bronze_track_alerts")
        gw = DEFAULT_TOPICS.index("igwn.gwalert")
        assert matrix[0, superk] == 2.0
        assert matrix[0, icecube] == 1.0
        assert matrix[1, gw] == 1.0
        assert matrix.sum() == 4.0

    def test_coincidence_burst_is_a_correlated_row(self) -> None:
        """A multimessenger burst fills several columns of ONE window."""
        notices = []
        for minutes in (0, 120, 240, 360):  # quiet background, one topic
            notices.append(self._notice("gcn.heartbeat", minutes))
        for topic in (
            "gcn.notices.superk.sn_alert",
            "gcn.notices.icecube.gold_bronze_track_alerts",
            "igwn.gwalert",
        ):
            notices.append(self._notice(topic, 481))  # the burst window
        matrix, _ = build_rate_matrix(notices, window_seconds=3600.0)
        burst_row = matrix[8]
        assert (burst_row > 0).sum() == 3

    def test_empty_input_yields_empty_engine_shape(self) -> None:
        matrix, starts = build_rate_matrix([], window_seconds=60.0)
        assert matrix.shape == (0, len(DEFAULT_TOPICS))
        assert starts == []

    def test_unlisted_topic_is_refused_not_dropped(self) -> None:
        notice = self._notice("gcn.heartbeat", 0)
        with pytest.raises(GCNIngestError, match="no feature column"):
            build_rate_matrix([notice], topics=("igwn.gwalert",))

    def test_bad_window_and_duplicate_topics_are_refused(self) -> None:
        with pytest.raises(GCNIngestError, match="positive"):
            build_rate_matrix([], window_seconds=0.0)
        with pytest.raises(GCNIngestError, match="duplicates"):
            build_rate_matrix([], topics=("igwn.gwalert", "igwn.gwalert"))


class TestConsumerTransport:
    def test_missing_credentials_are_refused_with_the_env_names(self) -> None:
        with pytest.raises(GCNIngestError, match="GCN_KAFKA_CLIENT_ID"):
            GCNNoticeConsumer("", "")

    def test_consume_normalizes_and_raises_on_broker_error(self) -> None:
        class _Message:
            def __init__(self, topic: str, value: bytes, error: str | None = None):
                self._topic, self._value, self._error = topic, value, error

            def topic(self) -> str:
                return self._topic

            def value(self) -> bytes:
                return self._value

            def error(self) -> str | None:
                return self._error

        class _FakeConsumer:
            def __init__(self, config: dict[str, Any]) -> None:
                self.config = config
                self.subscribed: list[str] = []

            def subscribe(self, topics: list[str]) -> None:
                self.subscribed = topics

            def consume(self, max_messages: int, timeout: float) -> list[_Message]:
                return [
                    _Message("gcn.heartbeat", b'{"alert_datetime": "2026-10-26T12:00:00Z"}'),
                    _Message("igwn.gwalert", b"{}", error="broker down"),
                ]

            def close(self) -> None:
                pass

        consumer = GCNNoticeConsumer("id", "secret", consumer_factory=_FakeConsumer)
        consumer.subscribe()
        assert consumer._consumer.subscribed == list(DEFAULT_TOPICS)

        stream = consumer.consume()
        first = next(stream)
        assert first.kind == "heartbeat" and first.event_time == T0
        with pytest.raises(GCNIngestError, match="broker down"):
            next(stream)
        consumer.close()

    def test_subscribe_refuses_unmapped_topics_before_touching_the_broker(self) -> None:
        class _Untouched:
            def __init__(self, config: dict[str, Any]) -> None:
                self.touched = False

            def subscribe(self, topics: list[str]) -> None:
                self.touched = True

        consumer = GCNNoticeConsumer("id", "secret", consumer_factory=_Untouched)
        with pytest.raises(GCNIngestError, match="Unknown GCN topic"):
            consumer.subscribe(("gcn.notices.mystery.alert",))
        assert consumer._consumer.touched is False


class TestConfigParityWithKeepalive:
    """The ingest and the keep-alive build the same librdkafka block."""

    def test_consumer_config_matches_the_keepalive_script(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = Path(__file__).resolve().parents[2] / "scripts" / "gcn_kafka_keepalive.py"
        spec = importlib.util.spec_from_file_location("gcn_kafka_keepalive", script)
        assert spec is not None and spec.loader is not None
        keepalive = importlib.util.module_from_spec(spec)
        # dataclass decorators resolve sys.modules[cls.__module__]; a module
        # executed without registration breaks there.
        sys.modules["gcn_kafka_keepalive"] = keepalive
        try:
            spec.loader.exec_module(keepalive)
        finally:
            sys.modules.pop("gcn_kafka_keepalive", None)

        # Same CA resolution on both sides for the comparison.
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)

        creds = keepalive.Credentials(
            client_id="id-123",
            client_secret="secret-456",  # noqa: S106 - sanitized test value
            domain="gcn.nasa.gov",
            bootstrap_servers="kafka.gcn.nasa.gov:9092",
        )
        theirs = keepalive.consumer_config(creds, group_id="shared-group")
        ours = consumer_config("id-123", "secret-456", group_id="shared-group")
        assert ours == theirs
