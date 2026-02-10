"""
Mercury Agent - Streaming Infrastructure Tests

Tests for the SaaS streaming infrastructure including:
- In-memory stream producer/consumer
- Circuit breaker pattern
- Streaming anomaly pipeline
- Message serialization

Copyright (C) 2025 Steel Security Advisors LLC
Licensed under GPL-3.0-or-later
"""

from __future__ import annotations

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


class TestStreamMessage:
    """Tests for StreamMessage dataclass."""

    def test_message_creation(self):
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

    def test_message_to_json(self):
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

    def test_message_from_json(self):
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

    def test_initial_state_closed(self):
        """Circuit should start in closed state."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.is_allowed()

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self):
        """Circuit should open after failure threshold."""
        cb = CircuitBreaker(name="test", failure_threshold=3, timeout_seconds=10)

        # Record failures
        for _ in range(3):
            await cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert not cb.is_allowed()

    @pytest.mark.asyncio
    async def test_success_resets_count(self):
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
    async def test_half_open_after_timeout(self):
        """Circuit should be half-open after timeout."""
        cb = CircuitBreaker(name="test", failure_threshold=2, timeout_seconds=0)

        await cb.record_failure()
        await cb.record_failure()

        # With 0 second timeout, should immediately be half-open
        assert cb.state == CircuitState.HALF_OPEN


class TestInMemoryStreamProducer:
    """Tests for InMemoryStreamProducer."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test connection lifecycle."""
        producer = InMemoryStreamProducer()
        await producer.connect()
        assert producer._connected
        await producer.disconnect()
        assert not producer._connected

    @pytest.mark.asyncio
    async def test_send_message(self):
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
    async def test_send_batch(self):
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
    async def test_subscribe(self):
        """Test topic subscription."""
        consumer = InMemoryStreamConsumer()
        await consumer.connect()
        await consumer.subscribe(["topic1", "topic2"])
        assert "topic1" in consumer._subscribed_topics
        assert "topic2" in consumer._subscribed_topics
        await consumer.disconnect()

    @pytest.mark.asyncio
    async def test_consume_messages(self):
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

    def test_producer_factory_memory(self):
        """Test creating in-memory producer."""
        producer = StreamProducerFactory.create("memory")
        assert isinstance(producer, InMemoryStreamProducer)

    def test_producer_factory_with_config(self):
        """Test creating producer with config."""
        config = StreamConfig(batch_size=200)
        producer = StreamProducerFactory.create("memory", config=config)
        assert producer.config.batch_size == 200

    def test_consumer_factory_memory(self):
        """Test creating in-memory consumer."""
        consumer = StreamConsumerFactory.create("memory", group_id="test-group")
        assert isinstance(consumer, InMemoryStreamConsumer)
        assert consumer.group_id == "test-group"

    def test_factory_backend_enum(self):
        """Test using StreamingBackend enum."""
        producer = StreamProducerFactory.create(StreamingBackend.MEMORY)
        assert isinstance(producer, InMemoryStreamProducer)


class TestStreamConfig:
    """Tests for StreamConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StreamConfig()
        assert config.backend == StreamingBackend.MEMORY
        assert config.batch_size == 100
        assert config.max_retries == 3

    def test_custom_values(self):
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
    async def test_pipeline_creation(self):
        """Test pipeline initialization."""
        pipeline = StreamingAnomalyPipeline(
            input_topic="input",
            output_topic="output",
            backend="memory",
        )
        assert pipeline.input_topic == "input"
        assert pipeline.output_topic == "output"

    @pytest.mark.asyncio
    async def test_pipeline_with_custom_detector(self):
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
    async def test_pipeline_stats(self):
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


class TestKafkaProducerMocked:
    """Tests for Kafka producer with mocked aiokafka."""

    @pytest.mark.asyncio
    async def test_kafka_producer_import_error(self):
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
    async def test_kafka_producer_circuit_breaker(self):
        """Test circuit breaker blocks when open."""
        from omni_mercury_engine.infrastructure.streaming import KafkaStreamProducer

        producer = KafkaStreamProducer()

        # Force circuit breaker open
        for _ in range(10):
            await producer._circuit_breaker.record_failure()

        # Should return False without trying to send
        result = await producer.send("topic", {"data": 1})
        assert result is False


class TestRedisProducerMocked:
    """Tests for Redis producer with mocked redis."""

    @pytest.mark.asyncio
    async def test_redis_producer_import_error(self):
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
    async def test_full_producer_consumer_flow(self):
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
    async def test_multiple_topics(self):
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
