# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""External service stubs for testing and development.

These stubs provide mock implementations of external services, allowing development and testing
without actual service dependencies.
"""

from __future__ import annotations

from tools.service_doubles.cache import CacheEntry, CacheStub
from tools.service_doubles.database import DatabaseStub, QueryResult
from tools.service_doubles.financial import (
    FinancialServiceStub,
    MarketData,
    SecurityPrice,
    TradingSignal,
)
from tools.service_doubles.weather import (
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
