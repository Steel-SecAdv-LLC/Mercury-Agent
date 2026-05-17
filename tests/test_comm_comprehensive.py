"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Comprehensive tests for utils/comm.py module.
Targets coverage improvement for async communication utilities.
"""

from __future__ import annotations

from typing import Any

import pytest

from omni_mercury_engine.utils.comm import (
    AsyncMessageQueue,
    Message,
    MessagePriority,
    SimplePubSub,
)


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self) -> None:
        """Test basic message creation."""
        msg = Message(
            sender="agent_a",
            recipient="agent_b",
            content={"action": "detect"},
            priority=MessagePriority.HIGH,
        )

        assert msg.sender == "agent_a"
        assert msg.recipient == "agent_b"
        assert msg.content == {"action": "detect"}
        assert msg.priority == MessagePriority.HIGH

    def test_message_defaults(self) -> None:
        """Test default message values."""
        msg = Message(sender="a", recipient="b", content="test")

        assert msg.priority == MessagePriority.NORMAL
        assert msg.timestamp > 0
        assert msg.message_id.startswith("msg_")

    def test_to_dict(self) -> None:
        """Test message serialization to dict."""
        msg = Message(
            sender="a",
            recipient="b",
            content="hello",
            priority=MessagePriority.CRITICAL,
        )

        d = msg.to_dict()

        assert d["sender"] == "a"
        assert d["recipient"] == "b"
        assert d["content"] == "hello"
        assert d["priority"] == MessagePriority.CRITICAL.value
        assert "timestamp" in d
        assert "message_id" in d

    def test_from_dict(self) -> None:
        """Test message deserialization from dict."""
        d = {
            "sender": "x",
            "recipient": "y",
            "content": {"data": 123},
            "priority": 3,
            "timestamp": 1234567890.0,
            "message_id": "msg_test",
        }

        msg = Message.from_dict(d)

        assert msg.sender == "x"
        assert msg.recipient == "y"
        assert msg.content == {"data": 123}
        assert msg.priority == MessagePriority.HIGH
        assert msg.timestamp == 1234567890.0
        assert msg.message_id == "msg_test"

    def test_from_dict_defaults(self) -> None:
        """Test from_dict with minimal data."""
        d = {
            "sender": "a",
            "recipient": "b",
            "content": "test",
        }

        msg = Message.from_dict(d)

        assert msg.sender == "a"
        assert msg.priority == MessagePriority.NORMAL


class TestAsyncMessageQueue:
    """Tests for AsyncMessageQueue."""

    def test_init(self) -> None:
        """Test queue initialization."""
        queue = AsyncMessageQueue(max_size=100)

        assert queue.stats["messages_sent"] == 0
        assert queue.stats["messages_received"] == 0
        assert queue.stats["errors"] == 0

    @pytest.mark.asyncio
    async def test_send_receive(self) -> None:
        """Test basic send and receive."""
        queue = AsyncMessageQueue()

        msg = Message(sender="a", recipient="b", content="hello")

        # Send
        result = await queue.send(msg)
        assert result is True
        assert queue.stats["messages_sent"] == 1

        # Receive
        received = await queue.receive()
        assert received is not None
        assert received.content == "hello"
        assert queue.stats["messages_received"] == 1

    @pytest.mark.asyncio
    async def test_receive_timeout(self) -> None:
        """Test receive with timeout on empty queue."""
        queue = AsyncMessageQueue()

        # Should return None after timeout
        received = await queue.receive(timeout=0.1)
        assert received is None

    @pytest.mark.asyncio
    async def test_multiple_messages(self) -> None:
        """Test sending and receiving multiple messages."""
        queue = AsyncMessageQueue()

        # Send multiple messages
        for i in range(5):
            msg = Message(sender="a", recipient="b", content=f"msg_{i}")
            await queue.send(msg)

        assert queue.stats["messages_sent"] == 5

        # Receive all
        for i in range(5):
            received = await queue.receive()
            assert received is not None
            assert received.content == f"msg_{i}"

        assert queue.stats["messages_received"] == 5

    def test_register_handler(self) -> None:
        """Test handler registration."""
        queue = AsyncMessageQueue()

        def handler(msg: Any) -> None:
            pass

        queue.register_handler("str", handler)

        assert "str" in queue.handlers
        assert handler in queue.handlers["str"]

    def test_register_multiple_handlers(self) -> None:
        """Test registering multiple handlers for same type."""
        queue = AsyncMessageQueue()

        def handler1(msg: Any) -> None:
            pass

        def handler2(msg: Any) -> None:
            pass

        queue.register_handler("str", handler1)
        queue.register_handler("str", handler2)

        assert len(queue.handlers["str"]) == 2

    def test_get_stats(self) -> None:
        """Test getting queue statistics."""
        queue = AsyncMessageQueue()

        stats = queue.get_stats()

        assert "messages_sent" in stats
        assert "messages_received" in stats
        assert "errors" in stats

        # Stats should be a copy
        stats["messages_sent"] = 999
        assert queue.stats["messages_sent"] == 0


class TestSimplePubSub:
    """Tests for SimplePubSub."""

    def test_init(self) -> None:
        """Test pubsub initialization."""
        pubsub = SimplePubSub()
        assert len(pubsub.subscribers) == 0

    def test_subscribe(self) -> None:
        """Test subscribing to topic."""
        pubsub = SimplePubSub()

        called = []

        def callback(msg: Any) -> None:
            called.append(msg)

        pubsub.subscribe("test_topic", callback)

        assert "test_topic" in pubsub.subscribers
        assert callback in pubsub.subscribers["test_topic"]

    def test_subscribe_multiple(self) -> None:
        """Test multiple subscribers to same topic."""
        pubsub = SimplePubSub()

        def callback1(msg: Any) -> None:
            pass

        def callback2(msg: Any) -> None:
            pass

        pubsub.subscribe("topic", callback1)
        pubsub.subscribe("topic", callback2)

        assert len(pubsub.subscribers["topic"]) == 2

    def test_unsubscribe(self) -> None:
        """Test unsubscribing from topic."""
        pubsub = SimplePubSub()

        def callback(msg: Any) -> None:
            pass

        pubsub.subscribe("topic", callback)
        assert callback in pubsub.subscribers["topic"]

        pubsub.unsubscribe("topic", callback)
        assert callback not in pubsub.subscribers["topic"]

    def test_publish(self) -> None:
        """Test publishing to topic."""
        pubsub = SimplePubSub()

        received = []

        def callback(msg: Any) -> None:
            received.append(msg)

        pubsub.subscribe("events", callback)
        pubsub.publish("events", {"event": "detected"})

        assert len(received) == 1
        assert received[0]["event"] == "detected"

    def test_publish_multiple_subscribers(self) -> None:
        """Test publishing to multiple subscribers."""
        pubsub = SimplePubSub()

        results1 = []
        results2 = []

        def callback1(msg: Any) -> None:
            results1.append(msg)

        def callback2(msg: Any) -> None:
            results2.append(msg)

        pubsub.subscribe("broadcast", callback1)
        pubsub.subscribe("broadcast", callback2)

        pubsub.publish("broadcast", "hello")

        assert results1 == ["hello"]
        assert results2 == ["hello"]

    def test_publish_no_subscribers(self) -> None:
        """Test publishing to topic with no subscribers."""
        pubsub = SimplePubSub()

        # Should not raise
        pubsub.publish("empty_topic", "message")

    def test_publish_callback_exception(self) -> None:
        """Test publishing when callback raises exception."""
        pubsub = SimplePubSub()

        results = []

        def failing_callback(msg: Any) -> None:
            raise ValueError("Intentional error")

        def working_callback(msg: Any) -> None:
            results.append(msg)

        pubsub.subscribe("topic", failing_callback)
        pubsub.subscribe("topic", working_callback)

        # Should not raise, exception is suppressed
        pubsub.publish("topic", "test")

        # Working callback should still be called
        assert results == ["test"]

    @pytest.mark.asyncio
    async def test_publish_async(self) -> None:
        """Test async publishing."""
        pubsub = SimplePubSub()

        received = []

        def sync_callback(msg: Any) -> None:
            received.append(f"sync:{msg}")

        async def async_callback(msg: Any) -> None:
            received.append(f"async:{msg}")

        pubsub.subscribe("async_topic", sync_callback)
        pubsub.subscribe("async_topic", async_callback)

        await pubsub.publish_async("async_topic", "hello")

        # Both callbacks should be called
        assert "sync:hello" in received
        assert "async:hello" in received

    @pytest.mark.asyncio
    async def test_publish_async_no_subscribers(self) -> None:
        """Test async publishing with no subscribers."""
        pubsub = SimplePubSub()

        # Should not raise
        await pubsub.publish_async("empty", "message")

    @pytest.mark.asyncio
    async def test_publish_async_with_exception(self) -> None:
        """Test async publishing when callback raises."""
        pubsub = SimplePubSub()

        results = []

        async def failing_async(msg: Any) -> None:
            raise RuntimeError("Async error")

        def working_sync(msg: Any) -> None:
            results.append(msg)

        pubsub.subscribe("topic", failing_async)
        pubsub.subscribe("topic", working_sync)

        # Should not raise
        await pubsub.publish_async("topic", "test")

        # Sync callback should still work
        assert results == ["test"]
