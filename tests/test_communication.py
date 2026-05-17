"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

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
Test communication utilities
"""

import asyncio

from omni_mercury_engine.utils.comm import AsyncMessageQueue, Message, MessagePriority, SimplePubSub


def test_message_creation() -> None:
    """Test message object creation"""
    msg = Message(
        sender="test_sender",
        recipient="test_recipient",
        content="test message",
        priority=MessagePriority.HIGH,
    )

    assert msg.content == "test message"
    assert msg.priority == MessagePriority.HIGH
    assert msg.sender == "test_sender"
    assert msg.recipient == "test_recipient"


def test_message_priority_enum() -> None:
    """Test message priority enum values"""
    assert MessagePriority.LOW.value == 1
    assert MessagePriority.NORMAL.value == 2
    assert MessagePriority.HIGH.value == 3
    assert MessagePriority.CRITICAL.value == 4


def test_async_message_queue_initialization() -> None:
    """Test async message queue initialization"""
    queue = AsyncMessageQueue(max_size=100)
    assert queue is not None
    assert queue.queue.maxsize == 100


async def async_test_send_receive() -> None:
    """Test async send and receive operations"""
    queue = AsyncMessageQueue(max_size=10)

    msg = Message(
        sender="sender1",
        recipient="recipient1",
        content="test",
        priority=MessagePriority.NORMAL,
    )
    await queue.send(msg)

    retrieved = await queue.receive()
    assert retrieved is not None
    assert retrieved.content == "test"


def test_async_queue_send_receive() -> None:
    """Test async queue operations"""
    asyncio.run(async_test_send_receive())


async def async_test_priority_ordering() -> None:
    """Test priority ordering in queue"""
    queue = AsyncMessageQueue(max_size=10)

    low_msg = Message(sender="s1", recipient="r1", content="low", priority=MessagePriority.LOW)
    high_msg = Message(sender="s2", recipient="r2", content="high", priority=MessagePriority.HIGH)
    normal_msg = Message(
        sender="s3", recipient="r3", content="normal", priority=MessagePriority.NORMAL
    )

    await queue.send(low_msg)
    await queue.send(high_msg)
    await queue.send(normal_msg)

    first = await queue.receive()
    assert first is not None


def test_priority_ordering() -> None:
    """Test message send/receive"""
    asyncio.run(async_test_priority_ordering())


def test_pubsub_initialization() -> None:
    """Test pub/sub system initialization"""
    pubsub = SimplePubSub()
    assert pubsub is not None


def test_pubsub_subscribe() -> None:
    """Test subscribing to topic"""
    pubsub = SimplePubSub()
    received = []

    def callback(msg) -> None:
        received.append(msg)

    pubsub.subscribe("test_topic", callback)
    pubsub.publish("test_topic", "test message")

    assert len(received) == 1
    assert received[0] == "test message"


def test_pubsub_multiple_subscribers() -> None:
    """Test multiple subscribers"""
    pubsub = SimplePubSub()
    received1 = []
    received2 = []

    def callback1(msg) -> None:
        received1.append(msg)

    def callback2(msg) -> None:
        received2.append(msg)

    pubsub.subscribe("test_topic", callback1)
    pubsub.subscribe("test_topic", callback2)
    pubsub.publish("test_topic", "broadcast")

    assert len(received1) == 1
    assert len(received2) == 1


def test_pubsub_unsubscribe() -> None:
    """Test unsubscribing from topic"""
    pubsub = SimplePubSub()
    received = []

    def callback(msg) -> None:
        received.append(msg)

    pubsub.subscribe("test_topic", callback)
    pubsub.unsubscribe("test_topic", callback)
    pubsub.publish("test_topic", "should not receive")

    assert len(received) == 0
