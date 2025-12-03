"""
External service stubs for testing and development.

These stubs provide mock implementations of external services,
allowing development and testing without actual service dependencies.
"""

from omni_anomaly_engine.integrations.stubs.weather import (
    WeatherServiceStub,
    WeatherData,
    WeatherCondition,
)
from omni_anomaly_engine.integrations.stubs.financial import (
    FinancialServiceStub,
    MarketData,
    SecurityPrice,
    TradingSignal,
)
from omni_anomaly_engine.integrations.stubs.database import (
    DatabaseStub,
    QueryResult,
)
from omni_anomaly_engine.integrations.stubs.cache import (
    CacheStub,
    CacheEntry,
)

__all__ = [
    # Weather
    "WeatherServiceStub",
    "WeatherData",
    "WeatherCondition",
    # Financial
    "FinancialServiceStub",
    "MarketData",
    "SecurityPrice",
    "TradingSignal",
    # Database
    "DatabaseStub",
    "QueryResult",
    # Cache
    "CacheStub",
    "CacheEntry",
]
