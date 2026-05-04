"""
External service stubs for testing and development.

These stubs provide mock implementations of external services, allowing development and testing
without actual service dependencies.
"""

from __future__ import annotations

from omni_mercury_engine.integrations.stubs.cache import CacheEntry, CacheStub
from omni_mercury_engine.integrations.stubs.database import DatabaseStub, QueryResult
from omni_mercury_engine.integrations.stubs.financial import (
    FinancialServiceStub,
    MarketData,
    SecurityPrice,
    TradingSignal,
)
from omni_mercury_engine.integrations.stubs.weather import (
    WeatherCondition,
    WeatherData,
    WeatherServiceStub,
)

__all__ = [
    "CacheEntry",
    # Cache
    "CacheStub",
    # Database
    "DatabaseStub",
    # Financial
    "FinancialServiceStub",
    "MarketData",
    "QueryResult",
    "SecurityPrice",
    "TradingSignal",
    "WeatherCondition",
    "WeatherData",
    # Weather
    "WeatherServiceStub",
]
