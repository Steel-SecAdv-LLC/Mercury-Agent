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
Optional lightweight communication utilities for distributed computing
Extracted from Communication Engine for future scalability
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable


class MessagePriority(Enum):
    """Message priority levels"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Message:
    """Lightweight message structure"""

    sender: str
    recipient: str
    content: Any
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    message_id: str = field(default_factory=lambda: f"msg_{datetime.now().timestamp()}")

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Create message from dictionary"""
        return cls(
            sender=data["sender"],
            recipient=data["recipient"],
            content=data["content"],
            priority=MessagePriority(data.get("priority", MessagePriority.NORMAL.value)),
            timestamp=data.get("timestamp", datetime.now().timestamp()),
            message_id=data.get("message_id", f"msg_{datetime.now().timestamp()}"),
        )


class AsyncMessageQueue:
    """
    Asynchronous message queue for inter-process communication
    Useful for distributed anomaly detection processing
    """

    def __init__(self, max_size: int = 1000) -> None:
        self.queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=max_size)
        self.handlers: dict[str, list[Callable[..., Any]]] = {}
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
        }

    async def send(self, message: Message) -> bool:
        """
        Send message to queue

        Args:
            message: Message to send

        Returns:
            True if successful
        """
        try:
            await self.queue.put(message)
            self.stats["messages_sent"] += 1
            return True
        except asyncio.QueueFull as e:
            logger.warning(f"Failed to send message: {e}")
            self.stats["errors"] += 1
            return False

    async def receive(self, timeout: float | None = None) -> Message | None:
        """
        Receive message from queue

        Args:
            timeout: Timeout in seconds

        Returns:
            Message or None if timeout
        """
        try:
            if timeout:
                message = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            else:
                message = await self.queue.get()

            self.stats["messages_received"] += 1
            return message
        except TimeoutError:
            return None
        except asyncio.CancelledError as e:
            logger.warning(f"Failed to receive message: {e}")
            self.stats["errors"] += 1
            return None

    def register_handler(self, message_type: str, handler: Callable[..., Any]) -> None:
        """
        Register handler for specific message type

        Args:
            message_type: Type of message to handle
            handler: Callback function
        """
        if message_type not in self.handlers:
            self.handlers[message_type] = []
        self.handlers[message_type].append(handler)

    async def process_messages(self) -> None:
        """
        Process messages using registered handlers
        Run this in a background task for automatic processing
        """
        while True:
            message = await self.receive()
            if message is None:
                await asyncio.sleep(0.1)
                continue

            message_type = type(message.content).__name__

            if message_type in self.handlers:
                for handler in self.handlers[message_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)
                    except (TypeError, ValueError, RuntimeError) as e:
                        # Handler execution errors - log for debugging
                        logger.debug(f"Message handler error: {e}")
                        self.stats["errors"] += 1

    def get_stats(self) -> dict[str, int]:
        """Get queue statistics"""
        return self.stats.copy()


class SimplePubSub:
    """
    Simple publish-subscribe pattern for event-driven communication
    Useful for broadcasting anomaly detection results
    """

    def __init__(self) -> None:
        self.subscribers: dict[str, list[Callable[..., Any]]] = {}

    def subscribe(self, topic: str, callback: Callable[..., Any]) -> None:
        """
        Subscribe to a topic

        Args:
            topic: Topic name
            callback: Function to call when message published
        """
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable[..., Any]) -> None:
        """
        Unsubscribe from a topic

        Args:
            topic: Topic name
            callback: Callback to remove
        """
        if topic in self.subscribers:
            self.subscribers[topic].remove(callback)

    def publish(self, topic: str, message: Any) -> None:
        """
        Publish message to topic

        Args:
            topic: Topic name
            message: Message to publish
        """
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    callback(message)
                except (TypeError, ValueError, RuntimeError) as e:
                    # Callback errors logged but don't break pub/sub flow
                    logger.debug(f"Publish callback error for topic '{topic}': {e}")

    async def publish_async(self, topic: str, message: Any) -> None:
        """
        Asynchronously publish message to topic

        Args:
            topic: Topic name
            message: Message to publish
        """
        if topic in self.subscribers:
            tasks = []
            for callback in self.subscribers[topic]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        tasks.append(callback(message))
                    else:
                        callback(message)
                except (TypeError, ValueError, RuntimeError):
                    # Callback errors logged but don't break pub/sub flow
                    continue

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
