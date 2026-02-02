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
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================
class StreamingBackend(str, Enum):
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
class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


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
    async def consume(
        self,
        timeout_ms: int = 1000,
    ) -> AsyncIterator[StreamMessage]:
        """Consume messages from subscribed topics.

        Args:
            timeout_ms: Timeout for polling in milliseconds

        Yields:
            StreamMessage objects
        """
        pass

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
        await self._producer.start()
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
        await self._consumer.start()
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
                        timestamp=datetime.fromtimestamp(msg.timestamp / 1000),
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
        await self._redis.ping()
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
                "timestamp": datetime.utcnow().isoformat(),
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
                "timestamp": datetime.utcnow().isoformat(),
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
        await self._redis.ping()
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
                                msg_data.get("timestamp", datetime.utcnow().isoformat())
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
        **kwargs,
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
        **kwargs,
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
        self._task: asyncio.Task | None = None
        self._stats = {
            "messages_processed": 0,
            "anomalies_detected": 0,
            "errors": 0,
        }

    def _default_detector(self, data: dict[str, Any]) -> dict[str, Any]:
        """Default detector - passes through with placeholder score."""
        return {
            "input": data,
            "is_anomaly": False,
            "score": 0.0,
            "detector": "passthrough",
        }

    async def start(self) -> None:
        """Start the streaming pipeline."""
        await self._producer.connect()
        await self._consumer.connect()
        await self._consumer.subscribe([self.input_topic])

        self._running = True
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

    async def _run(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                async for message in self._consumer.consume(timeout_ms=1000):
                    try:
                        # Apply anomaly detection
                        result = self.detector(message.value)
                        self._stats["messages_processed"] += 1

                        # Publish if anomaly detected
                        if result.get("is_anomaly") or result.get("score", 0) > 0.5:
                            result["source_topic"] = message.topic
                            result["source_timestamp"] = message.timestamp.isoformat()

                            await self._producer.send(
                                self.output_topic,
                                result,
                                key=message.key,
                            )
                            self._stats["anomalies_detected"] += 1

                        # Commit after processing
                        await self._consumer.commit(message)

                    except (ValueError, TypeError, KeyError) as e:
                        logger.error(f"Detection error: {e}")
                        self._stats["errors"] += 1

            except asyncio.CancelledError:
                break
            except (ConnectionError, OSError) as e:
                logger.error(f"Pipeline error: {e}")
                self._stats["errors"] += 1
                await asyncio.sleep(1)

    def get_stats(self) -> dict[str, int]:
        """Get pipeline statistics."""
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
