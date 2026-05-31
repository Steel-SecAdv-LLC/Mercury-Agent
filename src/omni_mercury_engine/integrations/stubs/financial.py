"""
Mercury Agent

Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Financial market data service stub for testing and development.

Example:
    >>> service = FinancialServiceStub()
    >>> price = await service.get_price("AAPL")
    >>> print(f"AAPL: ${price.price}")
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from omni_mercury_engine.security.safe_http import SafeHTTPClient

logger = logging.getLogger(__name__)


class TradingSignal(Enum):
    """Trading signal types."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class MarketStatus(Enum):
    """Market status."""

    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    AFTER_HOURS = "after_hours"


@dataclass
class SecurityPrice:
    """
    Security price data.

    Attributes:
        symbol: Security symbol/ticker.
        price: Current price.
        change: Price change from previous close.
        change_percent: Percentage change.
        volume: Trading volume.
        high: Day high.
        low: Day low.
        open: Opening price.
        previous_close: Previous closing price.
        timestamp: Price timestamp.
    """

    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    high: float
    low: float
    open: float
    previous_close: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "change": self.change,
            "change_percent": self.change_percent,
            "volume": self.volume,
            "high": self.high,
            "low": self.low,
            "open": self.open,
            "previous_close": self.previous_close,
            "timestamp": self.timestamp.isoformat(),
        }

    @property
    def is_positive(self) -> bool:
        """Check if price change is positive."""
        return self.change >= 0


@dataclass
class MarketData:
    """
    Market-wide data.

    Attributes:
        index_name: Market index name.
        value: Index value.
        change: Change from previous close.
        change_percent: Percentage change.
        status: Market status.
        volume: Total market volume.
        advancing: Number of advancing stocks.
        declining: Number of declining stocks.
    """

    index_name: str
    value: float
    change: float
    change_percent: float
    status: MarketStatus
    volume: int
    advancing: int
    declining: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index_name": self.index_name,
            "value": self.value,
            "change": self.change,
            "change_percent": self.change_percent,
            "status": self.status.value,
            "volume": self.volume,
            "advancing": self.advancing,
            "declining": self.declining,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HistoricalBar:
    """Historical price bar (OHLCV)."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


# Common stock base prices for realistic simulation
STOCK_BASE_PRICES = {
    "AAPL": 175.0,
    "GOOGL": 140.0,
    "MSFT": 380.0,
    "AMZN": 180.0,
    "META": 500.0,
    "NVDA": 480.0,
    "TSLA": 250.0,
    "JPM": 195.0,
    "V": 280.0,
    "JNJ": 155.0,
}


class FinancialServiceStub:
    """Stub implementation of financial data service.

    Provides realistic mock market data for testing without
    requiring actual market data API access.

    Example:
        >>> stub = FinancialServiceStub(seed=42)
        >>> price = await stub.get_price("AAPL")
        >>> history = await stub.get_history("AAPL", days=30)
    """

    def __init__(
        self,
        seed: int | None = None,
        latency_ms: tuple[int, int] = (20, 100),
        failure_rate: float = 0.0,
    ):
        """
        Initialize financial stub.

        Args:
            seed: Random seed for reproducibility.
            latency_ms: Min/max simulated latency in milliseconds.
            failure_rate: Probability of simulated failure (0-1).
        """
        self._rng = random.Random(seed)
        self._latency_ms = latency_ms
        self._failure_rate = failure_rate
        self._call_count = 0
        self._price_cache: dict[str, SecurityPrice] = {}

    async def _simulate_latency(self) -> None:
        """Simulate network latency."""
        latency = self._rng.randint(*self._latency_ms) / 1000.0
        await asyncio.sleep(latency)

    def _maybe_fail(self) -> None:
        """Potentially raise exception to simulate failure."""
        if self._rng.random() < self._failure_rate:
            raise ConnectionError("Simulated financial service failure")

    def _get_base_price(self, symbol: str) -> float:
        """Get base price for symbol."""
        if symbol in STOCK_BASE_PRICES:
            return STOCK_BASE_PRICES[symbol]
        # Generate consistent price based on symbol hash
        return 50 + (hash(symbol) % 500)

    def _generate_price(self, symbol: str) -> SecurityPrice:
        """Generate realistic price data."""
        base = self._get_base_price(symbol)

        # Simulate intraday movement
        hour = datetime.now().hour
        intraday_factor = 1 + 0.01 * math.sin(hour / 24 * 2 * math.pi)

        # Random walk
        change_pct = self._rng.gauss(0, 0.02)
        price = base * intraday_factor * (1 + change_pct)

        previous_close = base * (1 + self._rng.gauss(0, 0.01))
        change = price - previous_close

        # Generate realistic OHLC
        daily_range = abs(self._rng.gauss(0, 0.02)) * base
        open_price = previous_close * (1 + self._rng.gauss(0, 0.005))
        high = max(price, open_price) + daily_range * self._rng.random()
        low = min(price, open_price) - daily_range * self._rng.random()

        return SecurityPrice(
            symbol=symbol,
            price=round(price, 2),
            change=round(change, 2),
            change_percent=round((change / previous_close) * 100, 2),
            volume=self._rng.randint(1000000, 50000000),
            high=round(high, 2),
            low=round(low, 2),
            open=round(open_price, 2),
            previous_close=round(previous_close, 2),
        )

    async def get_price(self, symbol: str) -> SecurityPrice:
        """
        Get current price for symbol.

        Args:
            symbol: Security symbol.

        Returns:
            Current price data.

        Raises:
            ConnectionError: On simulated failure.
        """
        self._call_count += 1
        await self._simulate_latency()
        self._maybe_fail()

        price = self._generate_price(symbol.upper())
        self._price_cache[symbol.upper()] = price
        return price

    async def get_prices(self, symbols: list[str]) -> dict[str, SecurityPrice]:
        """
        Get prices for multiple symbols.

        Args:
            symbols: List of security symbols.

        Returns:
            Dictionary mapping symbols to prices.
        """
        self._call_count += 1
        await self._simulate_latency()
        self._maybe_fail()

        return {symbol.upper(): self._generate_price(symbol.upper()) for symbol in symbols}

    async def get_market_data(self, index: str = "SPX") -> MarketData:
        """
        Get market index data.

        Args:
            index: Index symbol (SPX, DJI, IXIC, etc.).

        Returns:
            Market index data.
        """
        self._call_count += 1
        await self._simulate_latency()
        self._maybe_fail()

        # Base values for major indices
        index_bases = {
            "SPX": 5000.0,
            "DJI": 38000.0,
            "IXIC": 16000.0,
            "RUT": 2000.0,
        }
        base = index_bases.get(index, 1000.0)

        change_pct = self._rng.gauss(0, 0.01)
        value = base * (1 + change_pct)
        change = value - base

        # Determine market status based on time
        hour = datetime.now().hour
        if 4 <= hour < 9:
            status = MarketStatus.PRE_MARKET
        elif 9 <= hour < 16:
            status = MarketStatus.OPEN
        elif 16 <= hour < 20:
            status = MarketStatus.AFTER_HOURS
        else:
            status = MarketStatus.CLOSED

        advancing = self._rng.randint(1000, 3000)
        declining = self._rng.randint(1000, 3000)

        return MarketData(
            index_name=index,
            value=round(value, 2),
            change=round(change, 2),
            change_percent=round(change_pct * 100, 2),
            status=status,
            volume=self._rng.randint(1000000000, 5000000000),
            advancing=advancing,
            declining=declining,
        )

    async def get_history(
        self,
        symbol: str,
        days: int = 30,
        interval: str = "1d",
    ) -> list[HistoricalBar]:
        """
        Get historical price data.

        Args:
            symbol: Security symbol.
            days: Number of days of history.
            interval: Bar interval (1d, 1h, etc.).

        Returns:
            List of historical bars.
        """
        self._call_count += 1
        await self._simulate_latency()
        self._maybe_fail()

        bars = []
        base = self._get_base_price(symbol.upper())
        current_price = base

        for day in range(days, 0, -1):
            timestamp = datetime.now() - timedelta(days=day)

            # Random walk
            change = self._rng.gauss(0, 0.02)
            current_price = current_price * (1 + change)

            # Generate OHLC
            daily_range = abs(self._rng.gauss(0, 0.02)) * current_price
            open_price = current_price * (1 + self._rng.gauss(0, 0.005))
            close_price = current_price
            high = max(open_price, close_price) + daily_range * self._rng.random()
            low = min(open_price, close_price) - daily_range * self._rng.random()

            bars.append(
                HistoricalBar(
                    timestamp=timestamp,
                    open=round(open_price, 2),
                    high=round(high, 2),
                    low=round(low, 2),
                    close=round(close_price, 2),
                    volume=self._rng.randint(1000000, 50000000),
                )
            )

        return bars

    async def get_signal(self, symbol: str) -> TradingSignal:
        """
        Get trading signal for symbol.

        Args:
            symbol: Security symbol.

        Returns:
            Trading signal.
        """
        self._call_count += 1
        await self._simulate_latency()
        self._maybe_fail()

        # Random signal with weighted distribution
        signals = [
            (TradingSignal.STRONG_BUY, 0.1),
            (TradingSignal.BUY, 0.25),
            (TradingSignal.HOLD, 0.3),
            (TradingSignal.SELL, 0.25),
            (TradingSignal.STRONG_SELL, 0.1),
        ]

        r = self._rng.random()
        cumulative: float = 0.0
        for signal, weight in signals:
            cumulative += weight
            if r < cumulative:
                return signal
        return TradingSignal.HOLD

    async def detect_anomaly(
        self,
        symbol: str,
        threshold: float = 2.0,
    ) -> dict[str, Any]:
        """
        Detect price anomalies for symbol.

        Args:
            symbol: Security symbol.
            threshold: Standard deviation threshold.

        Returns:
            Anomaly detection result.
        """
        self._call_count += 1
        await self._simulate_latency()
        self._maybe_fail()

        # Simulate anomaly detection
        is_anomaly = self._rng.random() < 0.1
        score = self._rng.gauss(0, 1)
        if is_anomaly:
            score = self._rng.gauss(threshold + 0.5, 0.5)

        return {
            "symbol": symbol.upper(),
            "is_anomaly": abs(score) > threshold,
            "anomaly_score": round(score, 4),
            "threshold": threshold,
            "confidence": round(min(1.0, abs(score) / (threshold * 2)), 4),
            "timestamp": datetime.now().isoformat(),
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get service metrics."""
        return {
            "call_count": self._call_count,
            "cache_size": len(self._price_cache),
            "failure_rate": self._failure_rate,
        }


class FinancialAPIProvider(Enum):
    """Supported financial API providers."""

    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO_FINANCE = "yahoo_finance"
    STUB = "stub"


class FinancialService:
    """Production-ready financial data service with multiple API backends.

    Supports real-time and historical market data from:
    - Alpha Vantage (requires API key)
    - Yahoo Finance (free tier available)
    - Stub/Mock (for testing)

    Example:
        >>> # Using Alpha Vantage (requires API key)
        >>> service = FinancialService(
        ...     provider=FinancialAPIProvider.ALPHA_VANTAGE,
        ...     api_key=os.getenv("ALPHA_VANTAGE_API_KEY")
        ... )
        >>> price = await service.get_price("AAPL")

        >>> # Using Yahoo Finance (no API key required)
        >>> service = FinancialService(provider=FinancialAPIProvider.YAHOO_FINANCE)
        >>> price = await service.get_price("MSFT")

        >>> # Fallback to stub for testing
        >>> service = FinancialService(provider=FinancialAPIProvider.STUB)
    """

    # API endpoints
    ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
    YAHOO_FINANCE_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

    def __init__(
        self,
        provider: FinancialAPIProvider = FinancialAPIProvider.STUB,
        api_key: str | None = None,
        timeout: int = 30,
        cache_ttl: int = 60,
    ):
        """
        Initialize financial service.

        The ``fallback_to_stub`` parameter was removed in the May 2026
        Phase 2 audit cure — silent fallback to stub data is not
        permitted.  If the configured provider fails, the error
        propagates to the caller.

        Args:
            provider: API provider to use.
            api_key: API key for Alpha Vantage (not needed for Yahoo/Stub).
            timeout: Request timeout in seconds.
            cache_ttl: Cache time-to-live in seconds.
        """
        self.provider = provider
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.timeout = timeout
        self.cache_ttl = cache_ttl

        # Price cache with TTL
        self._cache: dict[str, tuple[SecurityPrice, datetime]] = {}
        self._call_count = 0
        self._api_errors = 0

        # Stub instance for explicit STUB provider only
        self._stub = FinancialServiceStub()

        # Validate Alpha Vantage configuration
        if provider == FinancialAPIProvider.ALPHA_VANTAGE and not self.api_key:
            raise NotImplementedError(
                "Alpha Vantage requires an API key. Set ALPHA_VANTAGE_API_KEY "
                "environment variable or provide api_key parameter. "
                "Silent fallback to stub mode is not permitted "
                "(Phase 2 audit cure)."
            )

    def _get_cached(self, symbol: str) -> SecurityPrice | None:
        """Get cached price if still valid."""
        if symbol in self._cache:
            price, cached_at = self._cache[symbol]
            if (datetime.now() - cached_at).total_seconds() < self.cache_ttl:
                return price
        return None

    def _set_cached(self, symbol: str, price: SecurityPrice) -> None:
        """Cache price data."""
        self._cache[symbol] = (price, datetime.now())

    async def _fetch_alpha_vantage(self, symbol: str) -> SecurityPrice:
        """
        Fetch price from Alpha Vantage API.

        API Documentation: https://www.alphavantage.co/documentation/
        """
        if not self.api_key:
            raise ValueError("Alpha Vantage API key required")

        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol.upper(),
            "apikey": self.api_key,
        }

        def fetch() -> dict[str, Any]:
            return SafeHTTPClient.get_json(
                self.ALPHA_VANTAGE_BASE,
                params=params,
                headers={"User-Agent": "Mercury-Agent/1.0"},
                timeout=self.timeout,
                user_configured=True,
            )

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, fetch)

        # Parse Alpha Vantage response
        quote = data.get("Global Quote", {})
        if not quote:
            error_msg = data.get("Note") or data.get("Information") or "No data returned"
            raise ValueError(f"Alpha Vantage API error: {error_msg}")

        price = float(quote.get("05. price", 0))
        previous_close = float(quote.get("08. previous close", price))
        change = float(quote.get("09. change", 0))
        change_percent = float(quote.get("10. change percent", "0%").rstrip("%"))
        volume = int(quote.get("06. volume", 0))
        high = float(quote.get("03. high", price))
        low = float(quote.get("04. low", price))
        open_price = float(quote.get("02. open", price))

        return SecurityPrice(
            symbol=symbol.upper(),
            price=price,
            change=change,
            change_percent=change_percent,
            volume=volume,
            high=high,
            low=low,
            open=open_price,
            previous_close=previous_close,
        )

    async def _fetch_yahoo_finance(self, symbol: str) -> SecurityPrice:
        """
        Fetch price from Yahoo Finance API.

        Uses the public Yahoo Finance chart API endpoint.
        """
        url = f"{self.YAHOO_FINANCE_BASE}/{symbol.upper()}"
        params = {
            "interval": "1d",
            "range": "1d",
        }

        def fetch() -> dict[str, Any]:
            return SafeHTTPClient.get_json(
                url,
                params=params,
                headers={
                    "User-Agent": "Mercury-Agent/1.0",
                    "Accept": "application/json",
                },
                timeout=self.timeout,
                user_configured=True,
            )

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, fetch)

        # Parse Yahoo Finance response
        result = data.get("chart", {}).get("result", [])
        if not result:
            error = data.get("chart", {}).get("error")
            raise ValueError(f"Yahoo Finance API error: {error or 'No data returned'}")

        quote_data = result[0]
        meta = quote_data.get("meta", {})
        indicators = quote_data.get("indicators", {}).get("quote", [{}])[0]

        price = meta.get("regularMarketPrice", 0)
        previous_close = meta.get("previousClose", price)
        change = price - previous_close
        change_percent = (change / previous_close * 100) if previous_close else 0

        opens = indicators.get("open", [])
        highs = indicators.get("high", [])
        lows = indicators.get("low", [])
        volumes = indicators.get("volume", [])

        open_price = opens[-1] if opens and opens[-1] is not None else price
        high = highs[-1] if highs and highs[-1] is not None else price
        low = lows[-1] if lows and lows[-1] is not None else price
        volume = volumes[-1] if volumes and volumes[-1] is not None else 0

        return SecurityPrice(
            symbol=symbol.upper(),
            price=round(price, 2),
            change=round(change, 2),
            change_percent=round(change_percent, 2),
            volume=int(volume),
            high=round(high, 2),
            low=round(low, 2),
            open=round(open_price, 2),
            previous_close=round(previous_close, 2),
        )

    async def get_price(self, symbol: str) -> SecurityPrice:
        """
        Get current price for symbol.

        Args:
            symbol: Security symbol/ticker.

        Returns:
            Current price data.

        Raises:
            ValueError: If symbol is invalid or API fails.
        """
        self._call_count += 1

        cached = self._get_cached(symbol)
        if cached:
            return cached

        # Use stub if explicitly configured
        if self.provider == FinancialAPIProvider.STUB:
            return await self._stub.get_price(symbol)

        if self.provider == FinancialAPIProvider.ALPHA_VANTAGE:
            price = await self._fetch_alpha_vantage(symbol)
        elif self.provider == FinancialAPIProvider.YAHOO_FINANCE:
            price = await self._fetch_yahoo_finance(symbol)
        else:
            raise ValueError(f"Unknown financial API provider: {self.provider}")

        # Cache successful result
        self._set_cached(symbol, price)
        return price

    async def get_prices(self, symbols: list[str]) -> dict[str, SecurityPrice]:
        """
        Get prices for multiple symbols.

        Args:
            symbols: List of security symbols.

        Returns:
            Dictionary mapping symbols to prices.
        """
        results: dict[str, SecurityPrice] = {}
        for symbol in symbols:
            try:
                results[symbol.upper()] = await self.get_price(symbol)
            except Exception as e:
                logger.warning(f"Failed to get price for {symbol}: {e}")
        return results

    async def get_history(
        self,
        symbol: str,
        days: int = 30,
        interval: str = "1d",
    ) -> list[HistoricalBar]:
        """
        Get historical price data.

        Args:
            symbol: Security symbol.
            days: Number of days of history.
            interval: Bar interval (1d, 1h, etc.).

        Returns:
            List of historical bars.
        """
        if self.provider == FinancialAPIProvider.STUB:
            return await self._stub.get_history(symbol, days, interval)

        if self.provider == FinancialAPIProvider.YAHOO_FINANCE:
            return await self._fetch_yahoo_history(symbol, days, interval)

        # Alpha Vantage historical requires different endpoint
        if self.provider == FinancialAPIProvider.ALPHA_VANTAGE:
            return await self._fetch_alpha_vantage_history(symbol, days)

        raise ValueError(f"Unknown financial API provider: {self.provider}")

    async def _fetch_yahoo_history(
        self,
        symbol: str,
        days: int,
        interval: str,
    ) -> list[HistoricalBar]:
        """Fetch historical data from Yahoo Finance."""
        # Map days to range parameter
        if days <= 7:
            range_param = "5d"
        elif days <= 30:
            range_param = "1mo"
        elif days <= 90:
            range_param = "3mo"
        elif days <= 180:
            range_param = "6mo"
        elif days <= 365:
            range_param = "1y"
        else:
            range_param = "2y"

        url = f"{self.YAHOO_FINANCE_BASE}/{symbol.upper()}"
        params = {
            "interval": interval,
            "range": range_param,
        }

        def fetch() -> dict[str, Any]:
            return SafeHTTPClient.get_json(
                url,
                params=params,
                headers={
                    "User-Agent": "Mercury-Agent/1.0",
                    "Accept": "application/json",
                },
                timeout=self.timeout,
                user_configured=True,
            )

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, fetch)

        result = data.get("chart", {}).get("result", [])
        if not result:
            return []

        quote_data = result[0]
        timestamps = quote_data.get("timestamp", [])
        indicators = quote_data.get("indicators", {}).get("quote", [{}])[0]

        opens = indicators.get("open", [])
        highs = indicators.get("high", [])
        lows = indicators.get("low", [])
        closes = indicators.get("close", [])
        volumes = indicators.get("volume", [])

        bars = []
        for i, ts in enumerate(timestamps[-days:]):
            if i >= len(opens):
                break

            bars.append(
                HistoricalBar(
                    timestamp=datetime.fromtimestamp(ts),
                    open=round(opens[i] or 0, 2),
                    high=round(highs[i] or 0, 2),
                    low=round(lows[i] or 0, 2),
                    close=round(closes[i] or 0, 2),
                    volume=int(volumes[i] or 0),
                )
            )

        return bars

    async def _fetch_alpha_vantage_history(
        self,
        symbol: str,
        days: int,
    ) -> list[HistoricalBar]:
        """Fetch historical data from Alpha Vantage."""
        if not self.api_key:
            raise NotImplementedError(
                "Alpha Vantage API key required for historical data. "
                "Silent fallback to stub is not permitted (Phase 2 audit cure)."
            )

        # Use TIME_SERIES_DAILY for daily data
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol.upper(),
            "outputsize": "full" if days > 100 else "compact",
            "apikey": self.api_key,
        }

        def fetch() -> dict[str, Any]:
            return SafeHTTPClient.get_json(
                self.ALPHA_VANTAGE_BASE,
                params=params,
                headers={"User-Agent": "Mercury-Agent/1.0"},
                timeout=self.timeout,
                user_configured=True,
            )

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, fetch)

        time_series = data.get("Time Series (Daily)", {})
        if not time_series:
            return []

        bars = []
        sorted_dates = sorted(time_series.keys(), reverse=True)[:days]

        for date_str in sorted_dates:
            day_data = time_series[date_str]
            bars.append(
                HistoricalBar(
                    timestamp=datetime.strptime(date_str, "%Y-%m-%d"),
                    open=float(day_data.get("1. open", 0)),
                    high=float(day_data.get("2. high", 0)),
                    low=float(day_data.get("3. low", 0)),
                    close=float(day_data.get("4. close", 0)),
                    volume=int(day_data.get("5. volume", 0)),
                )
            )

        return bars

    def get_metrics(self) -> dict[str, Any]:
        """Get service metrics."""
        return {
            "provider": self.provider.value,
            "call_count": self._call_count,
            "api_errors": self._api_errors,
            "cache_size": len(self._cache),
            "error_rate": self._api_errors / self._call_count if self._call_count > 0 else 0,
        }


# Factory function for easy service creation
def create_financial_service(
    use_real_api: bool = False,
    provider: str = "yahoo",
    api_key: str | None = None,
) -> FinancialService | FinancialServiceStub:
    """Create financial service with appropriate backend.

    Args:
        use_real_api: Whether to use real API or stub.
        provider: API provider ("alpha_vantage", "yahoo", "stub").
        api_key: API key for Alpha Vantage.

    Returns:
        Configured financial service.

    Example:
        >>> # For testing
        >>> service = create_financial_service(use_real_api=False)

        >>> # For production with Yahoo Finance
        >>> service = create_financial_service(use_real_api=True, provider="yahoo")

        >>> # For production with Alpha Vantage
        >>> service = create_financial_service(
        ...     use_real_api=True,
        ...     provider="alpha_vantage",
        ...     api_key="YOUR_API_KEY"
        ... )
    """
    if not use_real_api:
        return FinancialServiceStub()

    provider_map = {
        "alpha_vantage": FinancialAPIProvider.ALPHA_VANTAGE,
        "yahoo": FinancialAPIProvider.YAHOO_FINANCE,
        "stub": FinancialAPIProvider.STUB,
    }

    provider_enum = provider_map.get(provider.lower(), FinancialAPIProvider.STUB)

    return FinancialService(
        provider=provider_enum,
        api_key=api_key,
    )
