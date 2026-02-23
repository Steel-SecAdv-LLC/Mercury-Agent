"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Backend configuration and wiring for Mercury Agent integration stubs.

Environment variable overrides:
    MERCURY_DB_BACKEND      - "sqlite" (default) | "postgresql"
    MERCURY_CACHE_BACKEND   - "memory" (default) | "redis"
    MERCURY_WEATHER_BACKEND - "openmeteo" (default) | "openweathermap"
    MERCURY_FINANCIAL_BACKEND - "yahoo" (default) | "alphavantage"

All backends default to zero-dependency local implementations suitable
for CI and first-responder field deployments without infrastructure.

Example:
    >>> from omni_mercury_engine.integrations.backends import get_backends
    >>> backends = get_backends()
    >>> print(backends.database.health_check())
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from omni_mercury_engine.integrations.stubs.cache import (
    CacheStub,
    RedisCache,
    create_cache,
)
from omni_mercury_engine.integrations.stubs.database import (
    AsyncDatabase,
    DatabaseStub,
    create_database,
)
from omni_mercury_engine.integrations.stubs.financial import (
    FinancialService,
    FinancialServiceStub,
    create_financial_service,
)
from omni_mercury_engine.integrations.stubs.weather import (
    WeatherService,
    WeatherServiceStub,
    create_weather_service,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache TTL by domain (seconds) — used for API response caching
# ---------------------------------------------------------------------------
CACHE_TTL_SECONDS: dict[str, int] = {
    "environmental": 300,    # 5 min  — sensor data refreshes frequently
    "ocean": 600,            # 10 min — buoy data update cadence
    "climate": 3600,         # 1 hour — climate data is slower-moving
    "financial": 900,        # 15 min — market data window
    "security": 60,          # 1 min  — threat intel must be fresh
    "space": 1800,           # 30 min — solar/exoplanet cadence
    "medical": 3600,         # 1 hour — clinical data batches
    "default": 600,          # 10 min — safe fallback
}


def get_cache_ttl(domain: str) -> int:
    """Return the cache TTL for a given domain.

    Args:
        domain: Domain name (e.g. "environmental", "security").

    Returns:
        TTL in seconds.
    """
    return CACHE_TTL_SECONDS.get(domain, CACHE_TTL_SECONDS["default"])


# ---------------------------------------------------------------------------
# Backend container
# ---------------------------------------------------------------------------
@dataclass
class Backends:
    """Container for all integration backends.

    Attributes:
        database: Database backend (SQLite or PostgreSQL).
        cache: Cache backend (in-memory LRU or Redis).
        weather: Weather data backend (Open-Meteo or OpenWeatherMap).
        financial: Financial data backend (Yahoo Finance or Alpha Vantage).
    """

    database: AsyncDatabase | DatabaseStub = field(default_factory=DatabaseStub)
    cache: RedisCache | CacheStub = field(default_factory=CacheStub)
    weather: WeatherService | WeatherServiceStub = field(
        default_factory=WeatherServiceStub,
    )
    financial: FinancialService | FinancialServiceStub = field(
        default_factory=FinancialServiceStub,
    )

    async def health_check(self) -> dict[str, Any]:
        """Run health checks on all backends.

        Returns:
            Dictionary with per-backend health status following the contract:
            ``{"status": "healthy"|"degraded"|"unhealthy",
               "latency_ms": float, "details": str}``
        """
        results: dict[str, Any] = {}
        overall = "healthy"

        for name, backend in [
            ("database", self.database),
            ("cache", self.cache),
        ]:
            start = time.monotonic()
            try:
                raw = await backend.health_check()
                latency = (time.monotonic() - start) * 1000
                healthy = raw.get("healthy", True)
                results[name] = {
                    "status": "healthy" if healthy else "degraded",
                    "latency_ms": round(latency, 2),
                    "details": raw.get("message", ""),
                }
                if not healthy and overall == "healthy":
                    overall = "degraded"
            except Exception as exc:
                latency = (time.monotonic() - start) * 1000
                results[name] = {
                    "status": "unhealthy",
                    "latency_ms": round(latency, 2),
                    "details": str(exc),
                }
                overall = "unhealthy"

        results["overall"] = overall
        return results


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_backends: Backends | None = None


def get_backends() -> Backends:
    """Return the global ``Backends`` instance, creating it on first call.

    Backend selection is driven by environment variables:

    * ``MERCURY_DB_BACKEND``       → ``"sqlite"`` (default) or ``"postgresql"``
    * ``MERCURY_CACHE_BACKEND``    → ``"memory"`` (default) or ``"redis"``
    * ``MERCURY_WEATHER_BACKEND``  → ``"openmeteo"`` (default) or ``"openweathermap"``
    * ``MERCURY_FINANCIAL_BACKEND``→ ``"yahoo"`` (default) or ``"alphavantage"``
    """
    global _backends
    if _backends is not None:
        return _backends

    # --- Database -----------------------------------------------------------
    db_backend = os.getenv("MERCURY_DB_BACKEND", "sqlite")
    if db_backend == "postgresql":
        db = create_database(
            backend="postgresql",
            host=os.getenv("MERCURY_DB_HOST", "localhost"),
            port=int(os.getenv("MERCURY_DB_PORT", "5432")),
            database=os.getenv("MERCURY_DB_NAME", "mercury"),
            user=os.getenv("MERCURY_DB_USER", "mercury"),
            password=os.getenv("MERCURY_DB_PASSWORD"),
        )
    else:
        db = create_database(
            backend="sqlite",
            database=os.getenv("MERCURY_DB_PATH", "mercury.db"),
        )
    logger.info("Database backend: %s", db_backend)

    # --- Cache --------------------------------------------------------------
    cache_backend = os.getenv("MERCURY_CACHE_BACKEND", "memory")
    if cache_backend == "redis":
        cache = create_cache(
            backend="redis",
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD"),
        )
    else:
        cache = create_cache(backend="memory")
    logger.info("Cache backend: %s", cache_backend)

    # --- Weather ------------------------------------------------------------
    weather_backend = os.getenv("MERCURY_WEATHER_BACKEND", "openmeteo")
    use_real_weather = weather_backend != "stub"
    provider_weather = "noaa" if weather_backend == "openmeteo" else weather_backend
    weather = create_weather_service(
        use_real_api=use_real_weather,
        provider=provider_weather,
    )
    logger.info("Weather backend: %s", weather_backend)

    # --- Financial ----------------------------------------------------------
    fin_backend = os.getenv("MERCURY_FINANCIAL_BACKEND", "yahoo")
    use_real_fin = fin_backend != "stub"
    financial = create_financial_service(
        use_real_api=use_real_fin,
        provider=fin_backend,
    )
    logger.info("Financial backend: %s", fin_backend)

    _backends = Backends(
        database=db,
        cache=cache,
        weather=weather,
        financial=financial,
    )
    return _backends
