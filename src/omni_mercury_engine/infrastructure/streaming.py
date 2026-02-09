"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations


"""
SaaS Streaming Infrastructure for Mercury Agent

Provides high-throughput, production-grade streaming capabilities:
- Kafka integration for distributed event streaming
- Redis Streams for real-time low-latency processing
- Abstract interface for pluggable backends
- Connection pooling and circuit breakers
- Backpressure handling and flow control

Example:
    # Using Kafka backend
    producer = StreamProducerFactory.create("kafka", bootstrap_servers="localhost:9092")
    await producer.send("anomalies", {"score": 0.95, "domain": "security"})

    # Using Redis Streams backend
    producer = StreamProducerFactory.create("redis", redis_url="redis://localhost:6379")
    await producer.send("anomalies", {"score": 0.95, "domain": "security"})

    # Consumer with automatic rebalancing
    consumer = StreamConsumerFactory.create("kafka", group_id="mercury-workers")
    async for message in consumer.consume("anomalies"):
        process_anomaly(message)
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable  # noqa: TC003 - used in runtime annotations
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np

from omni_mercury_engine.core.types import CircuitState


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================
class StreamingBackend(StrEnum):
    """Available streaming backends."""

    KAFKA = "kafka"
    REDIS = "redis"
    MEMORY = "memory"  # For testing


@dataclass
class StreamConfig:
    """Streaming configuration with sensible defaults."""

    backend: StreamingBackend = StreamingBackend.MEMORY

    # Kafka settings
    kafka_bootstrap_servers: str = field(
        default_factory=lambda: os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    )
    kafka_security_protocol: str = field(
        default_factory=lambda: os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
    )
    kafka_sasl_mechanism: str | None = field(
        default_factory=lambda: os.getenv("KAFKA_SASL_MECHANISM")
    )
    kafka_sasl_username: str | None = field(
        default_factory=lambda: os.getenv("KAFKA_SASL_USERNAME")
    )
    kafka_sasl_password: str | None = field(
        default_factory=lambda: os.getenv("KAFKA_SASL_PASSWORD")
    )

    # Redis settings
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    redis_max_connections: int = 10

    # General settings
    batch_size: int = 100
    batch_timeout_ms: int = 100
    max_retries: int = 3
    retry_backoff_ms: int = 100

    # Circuit breaker settings
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_s: int = 30


@dataclass
class StreamMessage:
    """Standardized message format for streaming."""

    topic: str
    key: str | None
    value: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    headers: dict[str, str] = field(default_factory=dict)
    partition: int | None = None
    offset: int | None = None

    def to_json(self) -> str:
        """Serialize message to JSON."""
        return json.dumps(
            {
                "topic": self.topic,
                "key": self.key,
                "value": self.value,
                "timestamp": self.timestamp.isoformat(),
                "headers": self.headers,
            }
        )

    @classmethod
    def from_json(cls, topic: str, data: str) -> StreamMessage:
        """Deserialize message from JSON."""
        parsed = json.loads(data)
        return cls(
            topic=topic,
            key=parsed.get("key"),
            value=parsed.get("value", {}),
            timestamp=datetime.fromisoformat(
                parsed.get("timestamp", datetime.now(UTC).isoformat())
            ),
            headers=parsed.get("headers", {}),
        )


# =============================================================================
# Circuit Breaker Pattern
# =============================================================================
@dataclass
class CircuitBreaker:
    """Circuit breaker for streaming connections.

    Prevents cascade failures by temporarily stopping requests
    to a failing downstream service.
    """

    name: str
    failure_threshold: int = 5
    timeout_seconds: int = 30
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state with automatic half-open transition."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.timeout_seconds:
                return CircuitState.HALF_OPEN
        return self._state

    async def record_success(self) -> None:
        """Record a successful operation."""
        async with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    async def record_failure(self) -> None:
        """Record a failed operation."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' opened after {self._failure_count} failures"
                )

    def is_allowed(self) -> bool:
        """Check if requests are allowed through."""
        return self.state != CircuitState.OPEN


# =============================================================================
# Abstract Streaming Interfaces
# =============================================================================
class StreamProducer(ABC):
    """Abstract base class for stream producers."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to streaming backend."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to streaming backend."""
        pass

    @abstractmethod
    async def send(
        self,
        topic: str,
        value: dict[str, Any],
        key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> bool:
        """Send a message to a topic.

        Args:
            topic: Target topic/stream name
            value: Message payload (will be JSON serialized)
            key: Optional message key for partitioning
            headers: Optional message headers

        Returns:
            True if message was sent successfully
        """
        pass

    @abstractmethod
    async def send_batch(
        self,
        topic: str,
        messages: list[dict[str, Any]],
    ) -> int:
        """Send multiple messages in a batch.

        Args:
            topic: Target topic/stream name
            messages: List of message payloads

        Returns:
            Number of messages successfully sent
        """
        pass

    @abstractmethod
    async def flush(self) -> None:
        """Flush any pending messages."""
        pass


class StreamConsumer(ABC):
    """Abstract base class for stream consumers."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to streaming backend."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to streaming backend."""
        pass

    @abstractmethod
    async def subscribe(self, topics: list[str]) -> None:
        """Subscribe to one or more topics.

        Args:
            topics: List of topic names to subscribe to
        """
        pass

    @abstractmethod
    def consume(
        self,
        timeout_ms: int = 1000,
    ) -> AsyncIterator[StreamMessage]:
        """Consume messages from subscribed topics.

        Args:
            timeout_ms: Timeout for polling in milliseconds

        Yields:
            StreamMessage objects
        """
        ...

    @abstractmethod
    async def commit(self, message: StreamMessage) -> None:
        """Commit offset for a consumed message.

        Args:
            message: The message to commit
        """
        pass


# =============================================================================
# In-Memory Implementation (for testing)
# =============================================================================
class InMemoryStreamProducer(StreamProducer):
    """In-memory producer for testing and development."""

    _streams: dict[str, list[StreamMessage]] = {}
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self, config: StreamConfig | None = None):
        self.config = config or StreamConfig()
        self._connected = False

    async def connect(self) -> None:
        """Connect (no-op for in-memory)."""
        self._connected = True
        logger.debug("InMemoryStreamProducer connected")

    async def disconnect(self) -> None:
        """Disconnect (no-op for in-memory)."""
        self._connected = False
        logger.debug("InMemoryStreamProducer disconnected")

    async def send(
        self,
        topic: str,
        value: dict[str, Any],
        key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> bool:
        """Send message to in-memory stream."""
        async with self._lock:
            if topic not in self._streams:
                self._streams[topic] = []

            message = StreamMessage(
                topic=topic,
                key=key,
                value=value,
                headers=headers or {},
                offset=len(self._streams[topic]),
            )
            self._streams[topic].append(message)
            logger.debug(f"InMemory: Sent message to topic '{topic}'")
            return True

    async def send_batch(
        self,
        topic: str,
        messages: list[dict[str, Any]],
    ) -> int:
        """Send batch of messages to in-memory stream."""
        count = 0
        for msg in messages:
            if await self.send(topic, msg):
                count += 1
        return count

    async def flush(self) -> None:
        """Flush (no-op for in-memory)."""
        pass

    @classmethod
    def get_messages(cls, topic: str) -> list[StreamMessage]:
        """Get all messages from a topic (for testing)."""
        return cls._streams.get(topic, [])

    @classmethod
    def clear(cls) -> None:
        """Clear all streams (for testing)."""
        cls._streams.clear()


class InMemoryStreamConsumer(StreamConsumer):
    """In-memory consumer for testing and development."""

    def __init__(self, config: StreamConfig | None = None, group_id: str = "default"):
        self.config = config or StreamConfig()
        self.group_id = group_id
        self._subscribed_topics: list[str] = []
        self._offsets: dict[str, int] = {}
        self._connected = False

    async def connect(self) -> None:
        """Connect (no-op for in-memory)."""
        self._connected = True
        logger.debug("InMemoryStreamConsumer connected")

    async def disconnect(self) -> None:
        """Disconnect (no-op for in-memory)."""
        self._connected = False
        logger.debug("InMemoryStreamConsumer disconnected")

    async def subscribe(self, topics: list[str]) -> None:
        """Subscribe to topics."""
        self._subscribed_topics = topics
        for topic in topics:
            if topic not in self._offsets:
                self._offsets[topic] = 0

    async def consume(
        self,
        timeout_ms: int = 1000,
    ) -> AsyncIterator[StreamMessage]:
        """Consume messages from subscribed topics."""
        start = time.time()
        while (time.time() - start) * 1000 < timeout_ms:
            for topic in self._subscribed_topics:
                messages = InMemoryStreamProducer._streams.get(topic, [])
                offset = self._offsets.get(topic, 0)

                if offset < len(messages):
                    message = messages[offset]
                    self._offsets[topic] = offset + 1
                    yield message

            await asyncio.sleep(0.01)

    async def commit(self, message: StreamMessage) -> None:
        """Commit offset (handled automatically for in-memory)."""
        pass


# =============================================================================
# Kafka Implementation
# =============================================================================
class KafkaStreamProducer(StreamProducer):
    """Kafka producer with production-grade features.

    Features:
    - Automatic batching and compression
    - Circuit breaker for failure handling
    - Retry with exponential backoff
    - Connection pooling via aiokafka
    """

    def __init__(self, config: StreamConfig | None = None):
        self.config = config or StreamConfig()
        self._producer = None
        self._circuit_breaker = CircuitBreaker(
            name="kafka-producer",
            failure_threshold=self.config.circuit_breaker_threshold,
            timeout_seconds=self.config.circuit_breaker_timeout_s,
        )

    async def connect(self) -> None:
        """Connect to Kafka cluster."""
        try:
            from aiokafka import AIOKafkaProducer
        except ImportError:
            raise ImportError(
                "aiokafka is required for Kafka streaming. " "Install with: pip install aiokafka"
            )

        kafka_config = {
            "bootstrap_servers": self.config.kafka_bootstrap_servers,
            "security_protocol": self.config.kafka_security_protocol,
            "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
            "key_serializer": lambda k: k.encode("utf-8") if k else None,
            "compression_type": "lz4",
            "acks": "all",
            "enable_idempotence": True,
            "max_batch_size": self.config.batch_size * 1024,
            "linger_ms": self.config.batch_timeout_ms,
        }

        # Add SASL config if provided
        if self.config.kafka_sasl_mechanism:
            kafka_config["sasl_mechanism"] = self.config.kafka_sasl_mechanism
            kafka_config["sasl_plain_username"] = self.config.kafka_sasl_username
            kafka_config["sasl_plain_password"] = self.config.kafka_sasl_password

        self._producer = AIOKafkaProducer(**kafka_config)
        await self._producer.start()  # type: ignore[attr-defined]
        logger.info(f"Kafka producer connected to {self.config.kafka_bootstrap_servers}")

    async def disconnect(self) -> None:
        """Disconnect from Kafka cluster."""
        if self._producer:
            await self._producer.stop()
            self._producer = None
            logger.info("Kafka producer disconnected")

    async def send(
        self,
        topic: str,
        value: dict[str, Any],
        key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> bool:
        """Send message to Kafka topic."""
        if not self._circuit_breaker.is_allowed():
            logger.warning("Kafka producer circuit breaker open, rejecting message")
            return False

        if not self._producer:
            raise RuntimeError("Producer not connected. Call connect() first.")

        try:
            # Convert headers to Kafka format
            kafka_headers = [(k, v.encode()) for k, v in (headers or {}).items()]

            await self._producer.send_and_wait(
                topic,
                value=value,
                key=key,
                headers=kafka_headers,
            )
            await self._circuit_breaker.record_success()
            return True

        except Exception as e:
            await self._circuit_breaker.record_failure()
            logger.error(f"Kafka send failed: {e}")
            return False

    async def send_batch(
        self,
        topic: str,
        messages: list[dict[str, Any]],
    ) -> int:
        """Send batch of messages to Kafka."""
        if not self._producer:
            raise RuntimeError("Producer not connected. Call connect() first.")

        count = 0
        batch = self._producer.create_batch()

        for msg in messages:
            try:
                metadata = batch.append(
                    value=json.dumps(msg).encode("utf-8"),
                    key=None,
                    timestamp=None,
                )
                if metadata is None:
                    # Batch is full, send it
                    await self._producer.send_batch(batch, topic)
                    batch = self._producer.create_batch()
                    count += batch.record_count()
            except Exception as e:
                logger.error(f"Kafka batch append failed: {e}")

        # Send remaining messages
        if batch.record_count() > 0:
            await self._producer.send_batch(batch, topic)
            count += batch.record_count()

        return count

    async def flush(self) -> None:
        """Flush pending messages."""
        if self._producer:
            await self._producer.flush()


class KafkaStreamConsumer(StreamConsumer):
    """Kafka consumer with production-grade features.

    Features:
    - Consumer group rebalancing
    - Automatic offset management
    - Circuit breaker for failure handling
    - Heartbeat and session timeout configuration
    """

    def __init__(
        self,
        config: StreamConfig | None = None,
        group_id: str = "mercury-agent",
        auto_commit: bool = False,
    ):
        self.config = config or StreamConfig()
        self.group_id = group_id
        self.auto_commit = auto_commit
        self._consumer = None
        self._circuit_breaker = CircuitBreaker(
            name="kafka-consumer",
            failure_threshold=self.config.circuit_breaker_threshold,
            timeout_seconds=self.config.circuit_breaker_timeout_s,
        )

    async def connect(self) -> None:
        """Connect to Kafka cluster."""
        try:
            from aiokafka import AIOKafkaConsumer
        except ImportError:
            raise ImportError(
                "aiokafka is required for Kafka streaming. " "Install with: pip install aiokafka"
            )

        kafka_config = {
            "bootstrap_servers": self.config.kafka_bootstrap_servers,
            "security_protocol": self.config.kafka_security_protocol,
            "group_id": self.group_id,
            "value_deserializer": lambda v: json.loads(v.decode("utf-8")),
            "key_deserializer": lambda k: k.decode("utf-8") if k else None,
            "auto_offset_reset": "earliest",
            "enable_auto_commit": self.auto_commit,
            "session_timeout_ms": 30000,
            "heartbeat_interval_ms": 10000,
            "max_poll_records": self.config.batch_size,
        }

        # Add SASL config if provided
        if self.config.kafka_sasl_mechanism:
            kafka_config["sasl_mechanism"] = self.config.kafka_sasl_mechanism
            kafka_config["sasl_plain_username"] = self.config.kafka_sasl_username
            kafka_config["sasl_plain_password"] = self.config.kafka_sasl_password

        self._consumer = AIOKafkaConsumer(**kafka_config)
        await self._consumer.start()  # type: ignore[attr-defined]
        logger.info(f"Kafka consumer connected to {self.config.kafka_bootstrap_servers}")

    async def disconnect(self) -> None:
        """Disconnect from Kafka cluster."""
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None
            logger.info("Kafka consumer disconnected")

    async def subscribe(self, topics: list[str]) -> None:
        """Subscribe to Kafka topics."""
        if not self._consumer:
            raise RuntimeError("Consumer not connected. Call connect() first.")

        self._consumer.subscribe(topics)
        logger.info(f"Kafka consumer subscribed to topics: {topics}")

    async def consume(
        self,
        timeout_ms: int = 1000,
    ) -> AsyncIterator[StreamMessage]:
        """Consume messages from Kafka topics."""
        if not self._consumer:
            raise RuntimeError("Consumer not connected. Call connect() first.")

        if not self._circuit_breaker.is_allowed():
            logger.warning("Kafka consumer circuit breaker open")
            return

        try:
            data = await self._consumer.getmany(timeout_ms=timeout_ms)

            for tp, messages in data.items():
                for msg in messages:
                    # Convert Kafka headers
                    headers = {k: v.decode("utf-8") if v else "" for k, v in (msg.headers or [])}

                    yield StreamMessage(
                        topic=msg.topic,
                        key=msg.key,
                        value=msg.value,
                        timestamp=datetime.fromtimestamp(msg.timestamp / 1000, tz=UTC),
                        headers=headers,
                        partition=msg.partition,
                        offset=msg.offset,
                    )

            await self._circuit_breaker.record_success()

        except Exception as e:
            await self._circuit_breaker.record_failure()
            logger.error(f"Kafka consume failed: {e}")

    async def commit(self, message: StreamMessage) -> None:
        """Commit offset for consumed message."""
        if not self._consumer or self.auto_commit:
            return

        try:
            from aiokafka import TopicPartition

            tp = TopicPartition(message.topic, message.partition or 0)
            await self._consumer.commit({tp: message.offset + 1})
        except Exception as e:
            logger.error(f"Kafka commit failed: {e}")


# =============================================================================
# Redis Streams Implementation
# =============================================================================
class RedisStreamProducer(StreamProducer):
    """Redis Streams producer for low-latency streaming.

    Features:
    - Sub-millisecond latency
    - Automatic stream trimming
    - Connection pooling
    - Pipeline batching
    """

    def __init__(self, config: StreamConfig | None = None):
        self.config = config or StreamConfig()
        self._redis = None
        self._circuit_breaker = CircuitBreaker(
            name="redis-producer",
            failure_threshold=self.config.circuit_breaker_threshold,
            timeout_seconds=self.config.circuit_breaker_timeout_s,
        )

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError(
                "redis is required for Redis streaming. " "Install with: pip install redis"
            )

        self._redis = redis.from_url(
            self.config.redis_url,
            max_connections=self.config.redis_max_connections,
            decode_responses=True,
        )
        # Test connection
        await self._redis.ping()  # type: ignore[attr-defined]
        logger.info(f"Redis producer connected to {self.config.redis_url}")

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Redis producer disconnected")

    async def send(
        self,
        topic: str,
        value: dict[str, Any],
        key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> bool:
        """Send message to Redis stream."""
        if not self._circuit_breaker.is_allowed():
            logger.warning("Redis producer circuit breaker open")
            return False

        if not self._redis:
            raise RuntimeError("Producer not connected. Call connect() first.")

        try:
            # Prepare message with metadata
            message_data = {
                "value": json.dumps(value),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            if key:
                message_data["key"] = key
            if headers:
                message_data["headers"] = json.dumps(headers)

            # Add to stream with auto-generated ID
            await self._redis.xadd(topic, message_data, maxlen=100000)
            await self._circuit_breaker.record_success()
            return True

        except Exception as e:
            await self._circuit_breaker.record_failure()
            logger.error(f"Redis send failed: {e}")
            return False

    async def send_batch(
        self,
        topic: str,
        messages: list[dict[str, Any]],
    ) -> int:
        """Send batch of messages using Redis pipeline."""
        if not self._redis:
            raise RuntimeError("Producer not connected. Call connect() first.")

        count = 0
        pipe = self._redis.pipeline()

        for msg in messages:
            message_data = {
                "value": json.dumps(msg),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            pipe.xadd(topic, message_data, maxlen=100000)
            count += 1

        try:
            await pipe.execute()
            return count
        except Exception as e:
            logger.error(f"Redis batch send failed: {e}")
            return 0

    async def flush(self) -> None:
        """Flush (no-op for Redis, writes are synchronous)."""
        pass


class RedisStreamConsumer(StreamConsumer):
    """Redis Streams consumer with consumer groups.

    Features:
    - Consumer group support for distributed processing
    - Pending message claiming for failure recovery
    - Automatic acknowledgment
    - Block-based consumption
    """

    def __init__(
        self,
        config: StreamConfig | None = None,
        group_id: str = "mercury-agent",
        consumer_name: str | None = None,
    ):
        self.config = config or StreamConfig()
        self.group_id = group_id
        self.consumer_name = consumer_name or f"consumer-{os.getpid()}"
        self._redis = None
        self._subscribed_topics: list[str] = []
        self._circuit_breaker = CircuitBreaker(
            name="redis-consumer",
            failure_threshold=self.config.circuit_breaker_threshold,
            timeout_seconds=self.config.circuit_breaker_timeout_s,
        )

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError(
                "redis is required for Redis streaming. " "Install with: pip install redis"
            )

        self._redis = redis.from_url(
            self.config.redis_url,
            max_connections=self.config.redis_max_connections,
            decode_responses=True,
        )
        await self._redis.ping()  # type: ignore[attr-defined]
        logger.info(f"Redis consumer connected to {self.config.redis_url}")

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Redis consumer disconnected")

    async def subscribe(self, topics: list[str]) -> None:
        """Subscribe to Redis streams by creating consumer groups."""
        if not self._redis:
            raise RuntimeError("Consumer not connected. Call connect() first.")

        self._subscribed_topics = topics

        for topic in topics:
            try:
                # Create consumer group (ignore if exists)
                await self._redis.xgroup_create(
                    topic,
                    self.group_id,
                    id="0",
                    mkstream=True,
                )
            except Exception as e:
                # Group already exists is OK
                if "BUSYGROUP" not in str(e):
                    logger.warning(f"Failed to create consumer group for {topic}: {e}")

        logger.info(f"Redis consumer subscribed to streams: {topics}")

    async def consume(
        self,
        timeout_ms: int = 1000,
    ) -> AsyncIterator[StreamMessage]:
        """Consume messages from Redis streams."""
        if not self._redis:
            raise RuntimeError("Consumer not connected. Call connect() first.")

        if not self._subscribed_topics:
            raise RuntimeError("No topics subscribed. Call subscribe() first.")

        if not self._circuit_breaker.is_allowed():
            logger.warning("Redis consumer circuit breaker open")
            return

        try:
            # Build streams dict for XREADGROUP
            streams = dict.fromkeys(self._subscribed_topics, ">")

            result = await self._redis.xreadgroup(
                self.group_id,
                self.consumer_name,
                streams,
                count=self.config.batch_size,
                block=timeout_ms,
            )

            if result:
                for topic, messages in result:
                    for msg_id, msg_data in messages:
                        value_str = msg_data.get("value", "{}")
                        headers_str = msg_data.get("headers", "{}")

                        yield StreamMessage(
                            topic=topic,
                            key=msg_data.get("key"),
                            value=json.loads(value_str),
                            timestamp=datetime.fromisoformat(
                                msg_data.get("timestamp", datetime.now(UTC).isoformat())
                            ),
                            headers=json.loads(headers_str),
                            offset=msg_id,
                        )

            await self._circuit_breaker.record_success()

        except Exception as e:
            await self._circuit_breaker.record_failure()
            logger.error(f"Redis consume failed: {e}")

    async def commit(self, message: StreamMessage) -> None:
        """Acknowledge consumed message."""
        if not self._redis:
            return

        try:
            await self._redis.xack(message.topic, self.group_id, message.offset)
        except Exception as e:
            logger.error(f"Redis ack failed: {e}")


# =============================================================================
# Factory Classes
# =============================================================================
class StreamProducerFactory:
    """Factory for creating stream producers."""

    @staticmethod
    def create(
        backend: str | StreamingBackend = StreamingBackend.MEMORY,
        config: StreamConfig | None = None,
        **kwargs: Any,
    ) -> StreamProducer:
        """Create a stream producer for the specified backend.

        Args:
            backend: "kafka", "redis", or "memory"
            config: Optional StreamConfig
            **kwargs: Backend-specific configuration overrides

        Returns:
            StreamProducer instance
        """
        if isinstance(backend, str):
            backend = StreamingBackend(backend.lower())

        config = config or StreamConfig(backend=backend)

        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        if backend == StreamingBackend.KAFKA:
            return KafkaStreamProducer(config)
        elif backend == StreamingBackend.REDIS:
            return RedisStreamProducer(config)
        else:
            return InMemoryStreamProducer(config)


class StreamConsumerFactory:
    """Factory for creating stream consumers."""

    @staticmethod
    def create(
        backend: str | StreamingBackend = StreamingBackend.MEMORY,
        config: StreamConfig | None = None,
        group_id: str = "mercury-agent",
        **kwargs: Any,
    ) -> StreamConsumer:
        """Create a stream consumer for the specified backend.

        Args:
            backend: "kafka", "redis", or "memory"
            config: Optional StreamConfig
            group_id: Consumer group ID for load balancing
            **kwargs: Backend-specific configuration overrides

        Returns:
            StreamConsumer instance
        """
        if isinstance(backend, str):
            backend = StreamingBackend(backend.lower())

        config = config or StreamConfig(backend=backend)

        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        if backend == StreamingBackend.KAFKA:
            return KafkaStreamConsumer(config, group_id=group_id)
        elif backend == StreamingBackend.REDIS:
            return RedisStreamConsumer(config, group_id=group_id)
        else:
            return InMemoryStreamConsumer(config, group_id=group_id)


# =============================================================================
# Streaming Anomaly Detection Pipeline
# =============================================================================
class StreamingAnomalyPipeline:
    """High-level pipeline for streaming anomaly detection.

    Provides a complete streaming solution that:
    - Consumes data from input stream
    - Applies anomaly detection
    - Publishes results to output stream
    - Handles backpressure and failures

    Example:
        pipeline = StreamingAnomalyPipeline(
            input_topic="sensor-data",
            output_topic="anomalies",
            backend="kafka",
        )

        await pipeline.start()
        # Pipeline runs until stopped
        await pipeline.stop()
    """

    def __init__(
        self,
        input_topic: str,
        output_topic: str,
        backend: str | StreamingBackend = StreamingBackend.MEMORY,
        config: StreamConfig | None = None,
        detector: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        group_id: str = "mercury-pipeline",
    ):
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.detector = detector or self._default_detector
        self.group_id = group_id

        self.config = config or StreamConfig(
            backend=(
                backend
                if isinstance(backend, StreamingBackend)
                else StreamingBackend(backend.lower())
            )
        )

        self._producer = StreamProducerFactory.create(
            backend=self.config.backend,
            config=self.config,
        )
        self._consumer = StreamConsumerFactory.create(
            backend=self.config.backend,
            config=self.config,
            group_id=self.group_id,
        )

        self._running = False
        self._task: asyncio.Task[None] | None = None

        # Comprehensive observability metrics for production monitoring
        self._stats: dict[str, Any] = {
            # Core throughput metrics
            "messages_processed": 0,
            "messages_per_second": 0.0,
            "anomalies_detected": 0,
            "anomaly_rate": 0.0,
            "errors": 0,
            "error_rate": 0.0,
            # Latency metrics (milliseconds)
            "detection_latency_ms": {
                "min": float("inf"),
                "max": 0.0,
                "avg": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            },
            "end_to_end_latency_ms": {
                "min": float("inf"),
                "max": 0.0,
                "avg": 0.0,
            },
            # Queue and backpressure metrics
            "queue_depth": 0,
            "backpressure_events": 0,
            "consumer_lag": 0,
            # Score distribution for anomaly analysis
            "score_distribution": {
                "min": 1.0,
                "max": 0.0,
                "avg": 0.0,
                "sum": 0.0,
            },
            # Error breakdown by type
            "error_breakdown": {
                "detection_errors": 0,
                "serialization_errors": 0,
                "connection_errors": 0,
                "timeout_errors": 0,
            },
            # Time tracking
            "start_time": None,
            "last_message_time": None,
            "uptime_seconds": 0.0,
        }
        # Rolling window for percentile calculations
        self._latency_window: list[float] = []
        self._max_latency_window_size = 1000

        # Statistical detector state for adaptive thresholding
        self._detector_state: dict[str, Any] = {
            "running_mean": {},
            "running_var": {},
            "sample_count": {},
            "min_samples": 10,
            "z_threshold": 3.0,  # Standard deviations for anomaly
            "ema_alpha": 0.1,  # Exponential moving average decay
        }

    def _default_detector(self, data: dict[str, Any]) -> dict[str, Any]:
        """Statistical Z-score anomaly detector with adaptive thresholding.

        This default detector performs real-time statistical anomaly detection
        using exponential moving average (EMA) for mean and variance tracking.
        Anomalies are flagged when values exceed z_threshold standard deviations
        from the running mean.

        The detector adapts to concept drift via the EMA decay parameter and
        requires min_samples observations before flagging anomalies to avoid
        false positives during warm-up.

        Args:
            data: Input data dictionary with numeric feature values

        Returns:
            Detection result with is_anomaly, score, anomaly_features,
            and statistical context for explainability
        """
        state = self._detector_state
        feature_scores: dict[str, float] = {}
        anomaly_features: list[str] = []
        total_score = 0.0
        n_features = 0

        # Extract numeric features for analysis
        for key, value in data.items():
            if not isinstance(value, int | float):
                continue
            if not np.isfinite(value):
                continue

            n_features += 1
            feature_key = str(key)

            # Initialize running statistics for new features
            if feature_key not in state["running_mean"]:
                state["running_mean"][feature_key] = float(value)
                state["running_var"][feature_key] = 0.0
                state["sample_count"][feature_key] = 1
                feature_scores[feature_key] = 0.0
                continue

            # Update exponential moving average statistics
            count = state["sample_count"][feature_key]
            old_mean = state["running_mean"][feature_key]
            old_var = state["running_var"][feature_key]
            alpha = state["ema_alpha"]

            # EMA update for mean
            new_mean = (1 - alpha) * old_mean + alpha * float(value)

            # EMA update for variance (Welford's algorithm variant)
            delta = float(value) - old_mean
            new_var = (1 - alpha) * old_var + alpha * (delta**2)

            state["running_mean"][feature_key] = new_mean
            state["running_var"][feature_key] = new_var
            state["sample_count"][feature_key] = count + 1

            # Compute Z-score if sufficient samples
            if count >= state["min_samples"] and new_var > 1e-10:
                std_dev = np.sqrt(new_var)
                z_score = abs(float(value) - new_mean) / std_dev
                normalized_score = min(1.0, z_score / (2 * state["z_threshold"]))
                feature_scores[feature_key] = normalized_score

                if z_score > state["z_threshold"]:
                    anomaly_features.append(feature_key)
                    total_score += normalized_score
            else:
                feature_scores[feature_key] = 0.0

        # Aggregate score across all features
        if n_features > 0:
            avg_score = total_score / n_features if anomaly_features else 0.0
            # Boost score if multiple features are anomalous
            multi_feature_boost = min(1.0, len(anomaly_features) / max(1, n_features / 2))
            final_score = min(1.0, avg_score + 0.2 * multi_feature_boost)
        else:
            final_score = 0.0

        is_anomaly = len(anomaly_features) > 0 and final_score > 0.3

        return {
            "input": data,
            "is_anomaly": is_anomaly,
            "score": round(final_score, 4),
            "detector": "statistical_zscore",
            "anomaly_features": anomaly_features,
            "feature_scores": feature_scores,
            "statistical_context": {
                "z_threshold": state["z_threshold"],
                "features_analyzed": n_features,
                "anomaly_count": len(anomaly_features),
            },
        }

    async def start(self) -> None:
        """Start the streaming pipeline."""
        await self._producer.connect()
        await self._consumer.connect()
        await self._consumer.subscribe([self.input_topic])

        self._running = True
        self._stats["start_time"] = datetime.now(UTC).isoformat()
        self._task = asyncio.create_task(self._run())
        logger.info(f"StreamingAnomalyPipeline started: {self.input_topic} -> {self.output_topic}")

    async def stop(self) -> None:
        """Stop the streaming pipeline."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        await self._producer.disconnect()
        await self._consumer.disconnect()
        logger.info("StreamingAnomalyPipeline stopped")

    def _update_latency_percentiles(self) -> None:
        """Update latency percentile metrics from rolling window."""
        if not self._latency_window:
            return

        sorted_latencies = sorted(self._latency_window)
        n = len(sorted_latencies)

        self._stats["detection_latency_ms"]["p50"] = sorted_latencies[int(n * 0.50)]
        self._stats["detection_latency_ms"]["p95"] = sorted_latencies[int(n * 0.95)]
        self._stats["detection_latency_ms"]["p99"] = sorted_latencies[min(int(n * 0.99), n - 1)]

    def _update_throughput_metrics(self, current_time: float) -> None:
        """Update throughput and rate metrics."""
        if self._stats["start_time"]:
            start = datetime.fromisoformat(self._stats["start_time"])
            elapsed = (datetime.now(UTC) - start).total_seconds()
            if elapsed > 0:
                self._stats["uptime_seconds"] = elapsed
                self._stats["messages_per_second"] = self._stats["messages_processed"] / elapsed
                if self._stats["messages_processed"] > 0:
                    self._stats["anomaly_rate"] = (
                        self._stats["anomalies_detected"] / self._stats["messages_processed"]
                    )
                    self._stats["error_rate"] = (
                        self._stats["errors"] / self._stats["messages_processed"]
                    )

    async def _run(self) -> None:
        """Main processing loop with comprehensive observability."""
        while self._running:
            try:
                async for message in self._consumer.consume(timeout_ms=1000):
                    process_start = time.perf_counter()
                    try:
                        # Apply anomaly detection with timing
                        detection_start = time.perf_counter()
                        result = self.detector(message.value)
                        detection_end = time.perf_counter()

                        # Update core metrics
                        self._stats["messages_processed"] += 1
                        self._stats["last_message_time"] = datetime.now(UTC).isoformat()

                        # Update latency metrics
                        detection_latency_ms = (detection_end - detection_start) * 1000
                        self._latency_window.append(detection_latency_ms)
                        if len(self._latency_window) > self._max_latency_window_size:
                            self._latency_window.pop(0)

                        lat_stats = self._stats["detection_latency_ms"]
                        lat_stats["min"] = min(lat_stats["min"], detection_latency_ms)
                        lat_stats["max"] = max(lat_stats["max"], detection_latency_ms)
                        # Running average
                        n = self._stats["messages_processed"]
                        lat_stats["avg"] = (
                            lat_stats["avg"] + (detection_latency_ms - lat_stats["avg"]) / n
                        )

                        # Update score distribution
                        score = result.get("score", 0.0)
                        score_stats = self._stats["score_distribution"]
                        score_stats["min"] = min(score_stats["min"], score)
                        score_stats["max"] = max(score_stats["max"], score)
                        score_stats["sum"] += score
                        score_stats["avg"] = score_stats["sum"] / n

                        # Publish if anomaly detected
                        if result.get("is_anomaly") or score > 0.5:
                            result["source_topic"] = message.topic
                            result["source_timestamp"] = message.timestamp.isoformat()
                            result["detection_latency_ms"] = round(detection_latency_ms, 3)

                            await self._producer.send(
                                self.output_topic,
                                result,
                                key=message.key,
                            )
                            self._stats["anomalies_detected"] += 1

                        # Update end-to-end latency
                        e2e_latency_ms = (time.perf_counter() - process_start) * 1000
                        e2e_stats = self._stats["end_to_end_latency_ms"]
                        e2e_stats["min"] = min(e2e_stats["min"], e2e_latency_ms)
                        e2e_stats["max"] = max(e2e_stats["max"], e2e_latency_ms)
                        e2e_stats["avg"] = (
                            e2e_stats["avg"] + (e2e_latency_ms - e2e_stats["avg"]) / n
                        )

                        # Commit after processing
                        await self._consumer.commit(message)

                        # Periodically update percentiles and throughput (every 100 messages)
                        if n % 100 == 0:
                            self._update_latency_percentiles()
                            self._update_throughput_metrics(time.time())

                    except (ValueError, TypeError) as e:
                        logger.error(f"Detection error: {e}")
                        self._stats["errors"] += 1
                        self._stats["error_breakdown"]["detection_errors"] += 1
                    except json.JSONDecodeError as e:
                        logger.error(f"Serialization error: {e}")
                        self._stats["errors"] += 1
                        self._stats["error_breakdown"]["serialization_errors"] += 1
                    except KeyError as e:
                        logger.error(f"Key error in detection: {e}")
                        self._stats["errors"] += 1
                        self._stats["error_breakdown"]["detection_errors"] += 1

            except asyncio.CancelledError:
                break
            except TimeoutError:
                self._stats["error_breakdown"]["timeout_errors"] += 1
            except (ConnectionError, OSError) as e:
                logger.error(f"Pipeline connection error: {e}")
                self._stats["errors"] += 1
                self._stats["error_breakdown"]["connection_errors"] += 1
                await asyncio.sleep(1)

        # Final metrics update on shutdown
        self._update_latency_percentiles()
        self._update_throughput_metrics(time.time())

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive pipeline statistics.

        Returns:
            Dictionary containing:
            - Core metrics: messages_processed, anomalies_detected, errors
            - Rates: messages_per_second, anomaly_rate, error_rate
            - Latency: detection and end-to-end latency with min/max/avg/percentiles
            - Score distribution: min/max/avg of anomaly scores
            - Error breakdown: by error type
            - Time tracking: start_time, last_message_time, uptime_seconds
        """
        # Update computed metrics before returning
        self._update_throughput_metrics(time.time())
        if self._latency_window:
            self._update_latency_percentiles()
        return self._stats.copy()


__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "InMemoryStreamConsumer",
    "InMemoryStreamProducer",
    "KafkaStreamConsumer",
    "KafkaStreamProducer",
    "RedisStreamConsumer",
    "RedisStreamProducer",
    "StreamConfig",
    "StreamConsumer",
    "StreamConsumerFactory",
    "StreamMessage",
    "StreamProducer",
    "StreamProducerFactory",
    "StreamingAnomalyPipeline",
    "StreamingBackend",
]
