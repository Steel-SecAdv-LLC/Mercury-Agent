# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Streaming Infrastructure Tests.

Tests for the SaaS streaming infrastructure including:
- In-memory stream producer/consumer
- Circuit breaker pattern
- Streaming anomaly pipeline
- Message serialization
"""

from __future__ import annotations

import importlib.util
from datetime import datetime
from unittest.mock import patch

import pytest

from omni_mercury_engine.infrastructure.streaming import (
    CircuitBreaker,
    CircuitState,
    InMemoryStreamConsumer,
    InMemoryStreamProducer,
    StreamConfig,
    StreamConsumerFactory,
    StreamingAnomalyPipeline,
    StreamingBackend,
    StreamMessage,
    StreamProducerFactory,
)

_HAS_AIOKAFKA = importlib.util.find_spec("aiokafka") is not None


class TestStreamMessage:
    """Tests for StreamMessage dataclass."""

    def test_message_creation(self) -> None:
        """Test basic message creation."""
        msg = StreamMessage(
            topic="test-topic",
            key="test-key",
            value={"data": [1, 2, 3]},
        )
        assert msg.topic == "test-topic"
        assert msg.key == "test-key"
        assert msg.value == {"data": [1, 2, 3]}
        assert isinstance(msg.timestamp, datetime)

    def test_message_to_json(self) -> None:
        """Test JSON serialization."""
        msg = StreamMessage(
            topic="test",
            key="key1",
            value={"score": 0.95},
            headers={"source": "detector"},
        )
        json_str = msg.to_json()
        assert "test" in json_str
        assert "key1" in json_str
        assert "0.95" in json_str
        assert "source" in json_str

    def test_message_from_json(self) -> None:
        """Test JSON deserialization."""
        msg = StreamMessage(
            topic="test",
            key="key1",
            value={"score": 0.95},
        )
        json_str = msg.to_json()
        restored = StreamMessage.from_json("test", json_str)
        assert restored.topic == "test"
        assert restored.value["score"] == 0.95


class TestCircuitBreaker:
    """Tests for CircuitBreaker pattern."""

    def test_initial_state_closed(self) -> None:
        """Circuit should start in closed state."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.is_allowed()

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self) -> None:
        """Circuit should open after failure threshold."""
        cb = CircuitBreaker(name="test", failure_threshold=3, timeout_seconds=10)

        # Record failures
        for _ in range(3):
            await cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert not cb.is_allowed()

    @pytest.mark.asyncio
    async def test_success_resets_count(self) -> None:
        """Success should reset failure count."""
        cb = CircuitBreaker(name="test", failure_threshold=3)

        await cb.record_failure()
        await cb.record_failure()
        await cb.record_success()

        # Should be reset, so 3 more failures needed
        await cb.record_failure()
        await cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self) -> None:
        """Circuit should be half-open after timeout."""
        cb = CircuitBreaker(name="test", failure_threshold=2, timeout_seconds=0)

        await cb.record_failure()
        await cb.record_failure()

        # With 0 second timeout, should immediately be half-open
        assert cb.state == CircuitState.HALF_OPEN


class TestInMemoryStreamProducer:
    """Tests for InMemoryStreamProducer."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self) -> None:
        """Test connection lifecycle."""
        producer = InMemoryStreamProducer()
        await producer.connect()
        assert producer._connected
        await producer.disconnect()
        assert not producer._connected

    @pytest.mark.asyncio
    async def test_send_message(self) -> None:
        """Test sending a message."""
        InMemoryStreamProducer.clear()
        producer = InMemoryStreamProducer()
        await producer.connect()

        result = await producer.send(
            topic="test-topic",
            value={"score": 0.9},
            key="key1",
        )

        assert result is True
        messages = InMemoryStreamProducer.get_messages("test-topic")
        assert len(messages) == 1
        assert messages[0].value == {"score": 0.9}

        await producer.disconnect()

    @pytest.mark.asyncio
    async def test_send_batch(self) -> None:
        """Test batch sending."""
        InMemoryStreamProducer.clear()
        producer = InMemoryStreamProducer()
        await producer.connect()

        messages = [{"id": i} for i in range(10)]
        count = await producer.send_batch("batch-topic", messages)

        assert count == 10
        stored = InMemoryStreamProducer.get_messages("batch-topic")
        assert len(stored) == 10

        await producer.disconnect()


class TestInMemoryStreamConsumer:
    """Tests for InMemoryStreamConsumer."""

    @pytest.mark.asyncio
    async def test_subscribe(self) -> None:
        """Test topic subscription."""
        consumer = InMemoryStreamConsumer()
        await consumer.connect()
        await consumer.subscribe(["topic1", "topic2"])
        assert "topic1" in consumer._subscribed_topics
        assert "topic2" in consumer._subscribed_topics
        await consumer.disconnect()

    @pytest.mark.asyncio
    async def test_consume_messages(self) -> None:
        """Test consuming messages."""
        InMemoryStreamProducer.clear()

        # Produce messages
        producer = InMemoryStreamProducer()
        await producer.connect()
        await producer.send("consume-test", {"id": 1})
        await producer.send("consume-test", {"id": 2})

        # Consume messages
        consumer = InMemoryStreamConsumer()
        await consumer.connect()
        await consumer.subscribe(["consume-test"])

        messages = []
        async for msg in consumer.consume(timeout_ms=100):
            messages.append(msg)

        assert len(messages) == 2
        assert messages[0].value["id"] == 1
        assert messages[1].value["id"] == 2

        await producer.disconnect()
        await consumer.disconnect()


class TestStreamFactories:
    """Tests for stream factory classes."""

    def test_producer_factory_memory(self) -> None:
        """Test creating in-memory producer."""
        producer = StreamProducerFactory.create("memory")
        assert isinstance(producer, InMemoryStreamProducer)

    def test_producer_factory_with_config(self) -> None:
        """Test creating producer with config."""
        config = StreamConfig(batch_size=200)
        producer = StreamProducerFactory.create("memory", config=config)
        # Factory returns abstract StreamProducer; narrow to concrete subclass.
        assert isinstance(producer, InMemoryStreamProducer)
        assert producer.config.batch_size == 200

    def test_consumer_factory_memory(self) -> None:
        """Test creating in-memory consumer."""
        consumer = StreamConsumerFactory.create("memory", group_id="test-group")
        assert isinstance(consumer, InMemoryStreamConsumer)
        assert consumer.group_id == "test-group"

    def test_factory_backend_enum(self) -> None:
        """Test using StreamingBackend enum."""
        producer = StreamProducerFactory.create(StreamingBackend.MEMORY)
        assert isinstance(producer, InMemoryStreamProducer)


class TestStreamConfig:
    """Tests for StreamConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = StreamConfig()
        assert config.backend == StreamingBackend.MEMORY
        assert config.batch_size == 100
        assert config.max_retries == 3

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = StreamConfig(
            backend=StreamingBackend.KAFKA,
            batch_size=500,
            kafka_bootstrap_servers="kafka:9092",
        )
        assert config.backend == StreamingBackend.KAFKA
        assert config.batch_size == 500
        assert config.kafka_bootstrap_servers == "kafka:9092"


class TestStreamingAnomalyPipeline:
    """Tests for StreamingAnomalyPipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_creation(self) -> None:
        """Test pipeline initialization."""
        pipeline = StreamingAnomalyPipeline(
            input_topic="input",
            output_topic="output",
            backend="memory",
        )
        assert pipeline.input_topic == "input"
        assert pipeline.output_topic == "output"

    @pytest.mark.asyncio
    async def test_pipeline_with_custom_detector(self) -> None:
        """Test pipeline with custom detector function."""

        def custom_detector(data):
            return {
                "input": data,
                "is_anomaly": data.get("value", 0) > 100,
                "score": min(data.get("value", 0) / 100, 1.0),
            }

        pipeline = StreamingAnomalyPipeline(
            input_topic="sensor-data",
            output_topic="anomalies",
            backend="memory",
            detector=custom_detector,
        )

        # Test detector function
        result = pipeline.detector({"value": 150})
        assert result["is_anomaly"] is True
        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_pipeline_stats(self) -> None:
        """Test pipeline statistics."""
        pipeline = StreamingAnomalyPipeline(
            input_topic="input",
            output_topic="output",
            backend="memory",
        )

        stats = pipeline.get_stats()
        assert "messages_processed" in stats
        assert "anomalies_detected" in stats
        assert "errors" in stats

    @pytest.mark.asyncio
    async def test_publish_success_allows_commit(self) -> None:
        """A successful anomaly publish permits the input offset to commit."""
        from unittest.mock import AsyncMock

        pipeline = StreamingAnomalyPipeline(
            input_topic="input",
            output_topic="output",
            backend="memory",
        )
        pipeline._producer = AsyncMock()
        pipeline._producer.send.return_value = True

        msg = StreamMessage(
            topic="input",
            key="k",
            value={"v": 1},
            timestamp=datetime.now(),
            partition=0,
            offset=5,
        )

        commit_ok = await pipeline._publish_anomaly({"is_anomaly": True, "score": 0.9}, msg)

        assert commit_ok is True
        pipeline._producer.send.assert_awaited_once()
        assert pipeline._stats["anomalies_published"] == 1
        assert pipeline._stats["error_breakdown"]["publish_failures"] == 0

    @pytest.mark.asyncio
    async def test_publish_failure_vetoes_commit(self) -> None:
        """A failed anomaly publish must veto the offset commit (no silent loss).

        Regression: the pipeline ignored ``send()``'s bool, counted the anomaly
        as detected, and committed the input offset anyway — so a broker blip or
        an open circuit breaker permanently dropped the detected alert.
        """
        from unittest.mock import AsyncMock

        pipeline = StreamingAnomalyPipeline(
            input_topic="input",
            output_topic="output",
            backend="memory",
        )
        pipeline._producer = AsyncMock()
        pipeline._producer.send.return_value = False  # circuit open / broker error

        msg = StreamMessage(
            topic="input",
            key="k",
            value={"v": 1},
            timestamp=datetime.now(),
            partition=0,
            offset=5,
        )

        commit_ok = await pipeline._publish_anomaly({"is_anomaly": True, "score": 0.9}, msg)

        assert commit_ok is False
        assert pipeline._stats["anomalies_published"] == 0
        assert pipeline._stats["error_breakdown"]["publish_failures"] == 1
        assert pipeline._stats["errors"] == 1


class TestKafkaProducerMocked:
    """Tests for Kafka producer with mocked aiokafka."""

    @pytest.mark.asyncio
    async def test_kafka_producer_import_error(self) -> None:
        """Test graceful handling when aiokafka not installed."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "aiokafka" or name.startswith("aiokafka."):
                raise ImportError("No module named 'aiokafka'")
            return real_import(name, *args, **kwargs)

        # Create a fresh producer instance and patch the import during connect()
        from omni_mercury_engine.infrastructure.streaming import KafkaStreamProducer

        producer = KafkaStreamProducer()

        with (
            patch.object(builtins, "__import__", side_effect=mock_import),
            pytest.raises(ImportError, match="aiokafka"),
        ):
            await producer.connect()

    @pytest.mark.asyncio
    async def test_kafka_producer_circuit_breaker(self) -> None:
        """Test circuit breaker blocks when open."""
        from omni_mercury_engine.infrastructure.streaming import KafkaStreamProducer

        producer = KafkaStreamProducer()

        # Force circuit breaker open
        for _ in range(10):
            await producer._circuit_breaker.record_failure()

        # Should return False without trying to send
        result = await producer.send("topic", {"data": 1})
        assert result is False


class TestKafkaConsumerCommit:
    """Tests for KafkaStreamConsumer.commit offset guarding."""

    @pytest.mark.asyncio
    async def test_commit_skips_non_int_offset(self) -> None:
        """A non-int (e.g. Redis-style string) offset must be skipped, not asserted.

        Regression: the commit path used ``assert isinstance(offset, int)`` to
        guard ``message.offset + 1``. Under ``python -O`` asserts are stripped,
        so a stray string offset would reach ``offset + 1`` and raise
        TypeError. The guard is now an explicit runtime check that skips with a
        debug log, matching the offset-None / partition-None contract.
        """
        from unittest.mock import AsyncMock

        from omni_mercury_engine.infrastructure.streaming import KafkaStreamConsumer

        consumer = KafkaStreamConsumer(auto_commit=False)
        consumer._consumer = AsyncMock()

        msg = StreamMessage(
            topic="t",
            key=None,
            value={"x": 1},
            timestamp=datetime.now(),
            partition=0,
            offset="1526-0",  # Redis stream id shape reaching the Kafka path
        )

        # Must not raise and must not attempt the commit.
        await consumer.commit(msg)
        consumer._consumer.commit.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not _HAS_AIOKAFKA,
        reason=(
            "commit()'s int-offset path imports aiokafka.TopicPartition at call "
            "time; without aiokafka the ImportError is swallowed by the commit "
            "error handler and the mocked commit is never awaited. Requires the "
            "[streaming] extra."
        ),
    )
    async def test_commit_int_offset_commits_next(self) -> None:
        """A normal int offset commits ``offset + 1`` (aiokafka semantics)."""
        from unittest.mock import AsyncMock

        from omni_mercury_engine.infrastructure.streaming import KafkaStreamConsumer

        consumer = KafkaStreamConsumer(auto_commit=False)
        consumer._consumer = AsyncMock()

        msg = StreamMessage(
            topic="t",
            key=None,
            value={"x": 1},
            timestamp=datetime.now(),
            partition=2,
            offset=41,
        )

        await consumer.commit(msg)
        consumer._consumer.commit.assert_awaited_once()
        (committed,), _ = consumer._consumer.commit.call_args
        assert list(committed.values()) == [42]

    @pytest.mark.asyncio
    async def test_commit_missing_aiokafka_raises_import_error(self) -> None:
        """A missing aiokafka surfaces as ImportError, not a swallowed log line.

        Regression: the ``TopicPartition`` import lived inside commit()'s
        ``try/except Exception``, so an absent dependency was logged as
        "Kafka commit failed" and the consumer group's offset cursor silently
        stopped advancing. The import now sits outside the try and must
        propagate.
        """
        import builtins
        from unittest.mock import AsyncMock

        from omni_mercury_engine.infrastructure.streaming import KafkaStreamConsumer

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "aiokafka" or name.startswith("aiokafka."):
                raise ImportError("No module named 'aiokafka'")
            return real_import(name, *args, **kwargs)

        consumer = KafkaStreamConsumer(auto_commit=False)
        consumer._consumer = AsyncMock()

        msg = StreamMessage(
            topic="t",
            key=None,
            value={"x": 1},
            timestamp=datetime.now(),
            partition=0,
            offset=7,
        )

        with (
            patch.object(builtins, "__import__", side_effect=mock_import),
            pytest.raises(ImportError, match="aiokafka"),
        ):
            await consumer.commit(msg)
        consumer._consumer.commit.assert_not_called()


class TestRedisProducerMocked:
    """Tests for Redis producer with mocked redis."""

    @pytest.mark.asyncio
    async def test_redis_producer_import_error(self) -> None:
        """Test graceful handling when redis not installed."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "redis" or name.startswith("redis."):
                raise ImportError("No module named 'redis'")
            return real_import(name, *args, **kwargs)

        # Create a fresh producer instance and patch the import during connect()
        from omni_mercury_engine.infrastructure.streaming import RedisStreamProducer

        producer = RedisStreamProducer()

        with (
            patch.object(builtins, "__import__", side_effect=mock_import),
            pytest.raises(ImportError, match="redis"),
        ):
            await producer.connect()


# Integration test that can be run with actual backends
class TestIntegrationInMemory:
    """Integration tests using in-memory backend."""

    @pytest.mark.asyncio
    async def test_full_producer_consumer_flow(self) -> None:
        """Test complete produce-consume cycle."""
        InMemoryStreamProducer.clear()

        # Setup
        producer = StreamProducerFactory.create("memory")
        consumer = StreamConsumerFactory.create("memory", group_id="integration-test")

        await producer.connect()
        await consumer.connect()
        await consumer.subscribe(["integration-topic"])

        # Produce
        for i in range(5):
            await producer.send("integration-topic", {"sequence": i})

        # Consume
        received = []
        async for msg in consumer.consume(timeout_ms=200):
            received.append(msg.value["sequence"])

        assert received == [0, 1, 2, 3, 4]

        await producer.disconnect()
        await consumer.disconnect()

    @pytest.mark.asyncio
    async def test_multiple_topics(self) -> None:
        """Test consuming from multiple topics."""
        InMemoryStreamProducer.clear()

        producer = StreamProducerFactory.create("memory")
        consumer = StreamConsumerFactory.create("memory")

        await producer.connect()
        await consumer.connect()

        # Send to different topics
        await producer.send("topic-a", {"source": "a"})
        await producer.send("topic-b", {"source": "b"})

        # Subscribe to both
        await consumer.subscribe(["topic-a", "topic-b"])

        received = []
        async for msg in consumer.consume(timeout_ms=200):
            received.append(msg.value["source"])

        assert set(received) == {"a", "b"}

        await producer.disconnect()
        await consumer.disconnect()
