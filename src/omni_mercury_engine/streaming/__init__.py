# Copyright (C) 2025 Steel Security Advisors LLC
"""Streaming anomaly detection with async data ingestion."""

from __future__ import annotations


def __getattr__(name: str) -> type:
    """Lazy import to avoid pulling in torch at import time."""
    if name == "StreamingDetector":
        from omni_mercury_engine.streaming.streaming_detector import StreamingDetector

        return StreamingDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["StreamingDetector"]
