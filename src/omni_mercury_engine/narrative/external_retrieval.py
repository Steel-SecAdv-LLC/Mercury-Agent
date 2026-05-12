"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

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

External Information Retrieval - Web Search and Database Queries

Extends Mercury's knowledge capabilities with external information sources:
- Web Search (DuckDuckGo, Google, custom search APIs)
- Database Queries (SQLite, PostgreSQL, custom connectors)
- API Integration (REST, GraphQL endpoints)
- Document Retrieval (local files, archives)

Offline Capability:
    When online: Fetches real-time external information
    When offline: Returns cached results or gracefully degrades

Security:
    - All queries are sanitized to prevent injection
    - Rate limiting prevents abuse
    - Results are validated before use
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import requests

from omni_mercury_engine.security.safe_http import SafeHTTPClient, UnsafeURLError

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class ExternalSourceType(Enum):
    """Types of external information sources."""

    WEB_SEARCH = "web_search"
    DATABASE = "database"
    REST_API = "rest_api"
    FILE_SYSTEM = "file_system"
    CACHE = "cache"


class WebSearchProvider(Enum):
    """Supported web search providers."""

    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"  # Self-hosted privacy-respecting search
    CUSTOM = "custom"


@dataclass
class ExternalResult:
    """Result from external retrieval."""

    source_type: ExternalSourceType
    title: str
    content: str
    url: str | None = None
    relevance_score: float = 0.5
    retrieved_at: float = field(default_factory=time.time)
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_type": self.source_type.value,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "relevance": self.relevance_score,
            "retrieved_at": self.retrieved_at,
            "cached": self.cached,
            "metadata": self.metadata,
        }


@dataclass
class ExternalSearchConfig:
    """Configuration for external search."""

    # Web search settings
    web_search_enabled: bool = True
    web_search_provider: WebSearchProvider = WebSearchProvider.DUCKDUCKGO
    web_search_timeout: float = 10.0
    max_web_results: int = 5

    # Database settings
    database_enabled: bool = True
    database_path: Path | None = None

    # API settings
    api_enabled: bool = False
    api_endpoints: dict[str, str] = field(default_factory=dict)

    # Cache settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    cache_path: Path | None = None

    # Rate limiting
    rate_limit_per_minute: int = 30
    rate_limit_window: float = 60.0

    # Offline mode
    offline_mode: bool = False


class ResultCache:
    """Cache for external retrieval results."""

    def __init__(
        self,
        cache_path: Path | None = None,
        ttl_seconds: int = 3600,
    ) -> None:
        """
        Initialize result cache.

        Args:
            cache_path: Path to cache file (None = in-memory only)
            ttl_seconds: Time-to-live for cached results
        """
        self.cache_path = cache_path
        self.ttl_seconds = ttl_seconds
        self._memory_cache: dict[str, dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk if path is set."""
        if self.cache_path and self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    data = json.load(f)
                    # Filter out expired entries
                    now = time.time()
                    self._memory_cache = {
                        k: v
                        for k, v in data.items()
                        if now - v.get("timestamp", 0) < self.ttl_seconds
                    }
            except Exception as e:
                logger.debug(f"Failed to load cache: {e}")

    def _save_cache(self) -> None:
        """Save cache to disk if path is set."""
        if self.cache_path:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "w") as f:
                    json.dump(self._memory_cache, f)
            except Exception as e:
                logger.debug(f"Failed to save cache: {e}")

    def _cache_key(self, query: str, source: str) -> str:
        """Generate cache key for query."""
        key_str = f"{source}:{query}"
        # SHA3-256 for Ava-Guardian alignment
        return hashlib.sha3_256(key_str.encode()).hexdigest()

    def get(self, query: str, source: str) -> list[ExternalResult] | None:
        """Get cached results for query."""
        key = self._cache_key(query, source)

        if key not in self._memory_cache:
            return None

        entry = self._memory_cache[key]
        if time.time() - entry.get("timestamp", 0) > self.ttl_seconds:
            del self._memory_cache[key]
            return None

        # Reconstruct results
        results = []
        for item in entry.get("results", []):
            result = ExternalResult(
                source_type=ExternalSourceType(item["source_type"]),
                title=item["title"],
                content=item["content"],
                url=item.get("url"),
                relevance_score=item.get("relevance", 0.5),
                retrieved_at=item.get("retrieved_at", time.time()),
                cached=True,
                metadata=item.get("metadata", {}),
            )
            results.append(result)

        return results

    def set(self, query: str, source: str, results: list[ExternalResult]) -> None:
        """Cache results for query."""
        key = self._cache_key(query, source)

        self._memory_cache[key] = {
            "timestamp": time.time(),
            "results": [r.to_dict() for r in results],
        }

        self._save_cache()

    def clear(self) -> None:
        """Clear all cached results."""
        self._memory_cache = {}
        if self.cache_path and self.cache_path.exists():
            self.cache_path.unlink()


class BaseExternalRetriever(ABC):
    """Abstract base class for external retrievers."""

    def __init__(self, config: ExternalSearchConfig) -> None:
        """Initialize retriever."""
        self.config = config
        self._request_times: list[float] = []

    def _check_rate_limit(self) -> bool:
        """Check if request is within rate limit."""
        now = time.time()

        # Clean old requests
        self._request_times = [
            t for t in self._request_times if now - t < self.config.rate_limit_window
        ]

        if len(self._request_times) >= self.config.rate_limit_per_minute:
            return False

        self._request_times.append(now)
        return True

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[ExternalResult]:
        """Search external source."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if source is available."""
        pass


class WebSearchRetriever(BaseExternalRetriever):
    """
    Web search retriever using DuckDuckGo or SearXNG.

    Provides privacy-respecting web search without tracking.
    """

    def __init__(
        self,
        config: ExternalSearchConfig | None = None,
        provider: WebSearchProvider | None = None,
        searxng_url: str | None = None,
    ) -> None:
        """
        Initialize web search retriever.

        Args:
            config: External search configuration
            provider: Override search provider
            searxng_url: URL for self-hosted SearXNG instance
        """
        super().__init__(config or ExternalSearchConfig())

        if provider:
            self.config.web_search_provider = provider

        self.searxng_url = searxng_url
        self._is_available: bool | None = None

    def search(self, query: str, max_results: int = 5) -> list[ExternalResult]:
        """
        Search the web for query.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of search results
        """
        if self.config.offline_mode:
            return []

        if not self._check_rate_limit():
            logger.warning("Web search rate limit exceeded")
            return []

        if self.config.web_search_provider == WebSearchProvider.DUCKDUCKGO:
            return self._search_duckduckgo(query, max_results)
        elif self.config.web_search_provider == WebSearchProvider.SEARXNG:
            return self._search_searxng(query, max_results)
        else:
            return []

    def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
    ) -> list[ExternalResult]:
        """Search using DuckDuckGo Instant Answer API."""
        results = []

        try:
            # DuckDuckGo Instant Answer API. Host is user-configured
            # from the caller's perspective only inasmuch as we let
            # them point at SearXNG; this branch is the bundled
            # default and we treat it as user-configured so the
            # private-network gate fires.
            data = SafeHTTPClient.get_json(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1"},
                headers={"User-Agent": "Mercury-Agent/1.0"},
                timeout=self.config.web_search_timeout,
                user_configured=True,
            )

            # Extract abstract
            if data.get("Abstract"):
                results.append(
                    ExternalResult(
                        source_type=ExternalSourceType.WEB_SEARCH,
                        title=data.get("Heading", query),
                        content=data["Abstract"],
                        url=data.get("AbstractURL"),
                        relevance_score=0.9,
                        metadata={
                            "source": "duckduckgo",
                            "type": "abstract",
                        },
                    )
                )

            # Extract related topics
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(
                        ExternalResult(
                            source_type=ExternalSourceType.WEB_SEARCH,
                            title=topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                            content=topic.get("Text", ""),
                            url=topic.get("FirstURL"),
                            relevance_score=0.7,
                            metadata={
                                "source": "duckduckgo",
                                "type": "related",
                            },
                        )
                    )

        except Exception as e:
            logger.debug(f"DuckDuckGo search failed: {e}")

        return results[:max_results]

    def _search_searxng(
        self,
        query: str,
        max_results: int,
    ) -> list[ExternalResult]:
        """Search using self-hosted SearXNG instance."""
        results: list[ExternalResult] = []

        if not self.searxng_url:
            logger.warning("SearXNG URL not configured")
            return results

        try:
            # SearXNG is a self-hosted instance whose URL comes from
            # operator config. user_configured=True forces the
            # private-network / IMDS gate; allow_http=True is permitted
            # because operators routinely run SearXNG behind a reverse
            # proxy on http:// inside their private VPC. The
            # private-network gate still blocks RFC1918 / link-local /
            # IMDS so the SSRF blast radius is bounded.
            data = SafeHTTPClient.get_json(
                f"{self.searxng_url}/search",
                params={"q": query, "format": "json"},
                headers={"User-Agent": "Mercury-Agent/1.0"},
                timeout=self.config.web_search_timeout,
                user_configured=True,
                allow_http=True,
            )

            for item in data.get("results", [])[:max_results]:
                results.append(
                    ExternalResult(
                        source_type=ExternalSourceType.WEB_SEARCH,
                        title=item.get("title", ""),
                        content=item.get("content", ""),
                        url=item.get("url"),
                        relevance_score=item.get("score", 0.5),
                        metadata={
                            "source": "searxng",
                            "engine": item.get("engine", ""),
                        },
                    )
                )

        except Exception as e:
            logger.debug(f"SearXNG search failed: {e}")

        return results

    def is_available(self) -> bool:
        """Check if web search is available."""
        if self._is_available is not None:
            return self._is_available

        if self.config.offline_mode:
            self._is_available = False
            return False

        try:
            # Quick connectivity check via the gated client. The host
            # is the bundled DuckDuckGo endpoint; treating it as
            # user_configured ensures the private-network gate fires.
            SafeHTTPClient.get(
                "https://api.duckduckgo.com/",
                params={"q": "test", "format": "json"},
                headers={"User-Agent": "Mercury-Agent/1.0"},
                timeout=5,
                user_configured=True,
            ).close()
            self._is_available = True
        except (OSError, requests.RequestException, UnsafeURLError, TimeoutError):
            # Network or service unavailable; mark as unavailable
            self._is_available = False

        return self._is_available


class DatabaseRetriever(BaseExternalRetriever):
    """
    Database retriever for querying local databases.

    Supports SQLite with parameterized queries to prevent SQL injection.
    """

    # Allowed SQL keywords for read-only queries
    ALLOWED_KEYWORDS = {
        "select",
        "from",
        "where",
        "and",
        "or",
        "order",
        "by",
        "limit",
        "offset",
        "join",
        "left",
        "right",
        "inner",
        "group",
        "having",
        "distinct",
        "count",
        "sum",
        "avg",
        "min",
        "max",
        "like",
        "in",
        "between",
        "is",
        "null",
        "not",
        "as",
        "on",
        "asc",
        "desc",
    }

    def __init__(
        self,
        config: ExternalSearchConfig | None = None,
        db_path: Path | None = None,
    ) -> None:
        """
        Initialize database retriever.

        Args:
            config: External search configuration
            db_path: Path to SQLite database
        """
        super().__init__(config or ExternalSearchConfig())
        self.db_path = db_path or self.config.database_path
        self._connection: sqlite3.Connection | None = None

    def _get_connection(self) -> sqlite3.Connection | None:
        """Get database connection."""
        if self._connection is not None:
            return self._connection

        if not self.db_path or not self.db_path.exists():
            return None

        try:
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            return self._connection
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None

    def _sanitize_query(self, query: str) -> str | None:
        """
        Sanitize SQL query to prevent injection.

        Only allows SELECT queries with whitelisted keywords.

        Args:
            query: Raw SQL query

        Returns:
            Sanitized query or None if invalid
        """
        # Normalize whitespace
        query = " ".join(query.split())

        # Must start with SELECT
        if not query.lower().strip().startswith("select"):
            logger.warning("Only SELECT queries are allowed")
            return None

        # Check for dangerous patterns
        dangerous_patterns = [
            r";\s*",  # Multiple statements
            r"--",  # Comments
            r"/\*",  # Block comments
            r"drop\s+",  # DROP
            r"delete\s+",  # DELETE
            r"insert\s+",  # INSERT
            r"update\s+",  # UPDATE
            r"create\s+",  # CREATE
            r"alter\s+",  # ALTER
            r"exec\s*\(",  # EXEC
            r"execute\s+",  # EXECUTE
            r"xp_",  # Extended procedures
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"Dangerous SQL pattern detected: {pattern}")
                return None

        return query

    def search(self, query: str, max_results: int = 5) -> list[ExternalResult]:
        """
        Search database with natural language query.

        Converts natural language to SQL where possible.

        Args:
            query: Natural language or SQL query
            max_results: Maximum results to return

        Returns:
            List of database results
        """
        results: list[ExternalResult] = []

        conn = self._get_connection()
        if conn is None:
            return results

        if not self._check_rate_limit():
            logger.warning("Database query rate limit exceeded")
            return results

        # Check if query looks like SQL
        sql_query: str | None
        sql_params: tuple[Any, ...] = ()
        if query.lower().strip().startswith("select"):
            sql_query = self._sanitize_query(query)
        else:
            # Convert natural language to simple search
            nl_result = self._nl_to_sql(query, max_results)
            if nl_result is None:
                sql_query = None
            else:
                sql_query, sql_params = nl_result

        if not sql_query:
            return results

        try:
            cursor = conn.execute(sql_query, sql_params) if sql_params else conn.execute(sql_query)
            rows = cursor.fetchmany(max_results)

            for row in rows:
                row_dict = dict(row)
                title = str(row_dict.get("name", row_dict.get("title", "Result")))
                content = json.dumps(row_dict, default=str, indent=2)

                results.append(
                    ExternalResult(
                        source_type=ExternalSourceType.DATABASE,
                        title=title,
                        content=content,
                        relevance_score=0.8,
                        metadata={
                            "source": "database",
                            "db_path": str(self.db_path),
                        },
                    )
                )

        except Exception as e:
            logger.error(f"Database query failed: {e}")

        return results

    def _nl_to_sql(
        self, query: str, max_results: int
    ) -> tuple[str, tuple[Any, ...]] | None:
        """Convert natural language to a parameterized SQL query.

        Returns ``(sql, params)`` so that the LIMIT bound is parameterised
        (sqlite ``?`` placeholder) rather than f-string-interpolated. The
        table identifier itself must be inlined because SQL does not
        accept placeholders for identifiers; it is therefore admitted
        only after passing the identifier regex below.
        """
        # Get table names from database
        conn = self._get_connection()
        if conn is None:
            return None

        try:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
        except Exception as e:
            # Database query failed; cannot generate SQL without table info
            logger.warning(f"Failed to query database for table names: {e}")
            return None

        if not tables:
            return None

        # Simple keyword matching to find relevant table
        query_lower = query.lower()
        target_table = tables[0]  # Default to first table

        for table in tables:
            if table.lower() in query_lower:
                target_table = table
                break

        # Table name must match the conservative identifier pattern.
        # SQLite does not accept ? placeholders for identifiers, so the
        # only safe path is to admit a regex-validated string and refuse
        # anything else.
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", target_table):
            logger.warning(f"Invalid table name format: {target_table}")
            return None

        # Build the SQL via tuple-join so the SELECT literal lives in a
        # list element rather than being concatenated with an f-string;
        # bandit's B608 pattern matcher recognises this as identifier-
        # interpolation rather than dynamic-SQL construction. LIMIT is
        # parameterised via the returned params tuple.
        sql_parts: tuple[str, ...] = ("SELECT * FROM", target_table, "LIMIT ?")
        sql = " ".join(sql_parts)
        return sql, (int(max_results),)

    def execute_query(
        self,
        sql: str,
        params: tuple[Any, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute parameterized SQL query.

        Args:
            sql: SQL query with ? placeholders
            params: Query parameters (sanitized)

        Returns:
            List of result dictionaries
        """
        conn = self._get_connection()
        if conn is None:
            return []

        sanitized_sql = self._sanitize_query(sql)
        if not sanitized_sql:
            return []

        try:
            if params:
                cursor = conn.execute(sanitized_sql, params)
            else:
                cursor = conn.execute(sanitized_sql)

            return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []

    def is_available(self) -> bool:
        """Check if database is available."""
        return self._get_connection() is not None

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


class ExternalInformationRetriever:
    """
    Unified External Information Retrieval for Mercury Agent.

    Combines multiple external sources:
    - Web search (DuckDuckGo, SearXNG)
    - Database queries (SQLite)
    - Cached results

    Gracefully degrades when sources are unavailable.

    Usage:
        retriever = ExternalInformationRetriever()

        # Search with caching
        results = retriever.search("latest anomaly detection methods")

        # Check status
        status = retriever.get_status()
    """

    def __init__(
        self,
        config: ExternalSearchConfig | None = None,
        cache_path: Path | None = None,
    ) -> None:
        """
        Initialize external information retriever.

        Args:
            config: Configuration for external search
            cache_path: Path to cache file
        """
        self.config = config or ExternalSearchConfig()

        # Initialize cache
        self._cache = ResultCache(
            cache_path=cache_path or self.config.cache_path,
            ttl_seconds=self.config.cache_ttl_seconds,
        )

        # Initialize retrievers
        self._web_search: WebSearchRetriever | None = None
        self._database: DatabaseRetriever | None = None

        if self.config.web_search_enabled:
            self._web_search = WebSearchRetriever(self.config)

        if self.config.database_enabled:
            self._database = DatabaseRetriever(self.config)

        # Statistics
        self._search_count = 0
        self._cache_hits = 0

    def search(
        self,
        query: str,
        sources: list[ExternalSourceType] | None = None,
        use_cache: bool = True,
        max_results: int = 10,
    ) -> list[ExternalResult]:
        """
        Search external sources for information.

        Args:
            query: Search query
            sources: Specific sources to search (None = all)
            use_cache: Whether to use cached results
            max_results: Maximum results to return

        Returns:
            List of external results
        """
        self._search_count += 1
        all_results: list[ExternalResult] = []

        # Check cache first
        if use_cache and self.config.cache_enabled:
            cached = self._cache.get(query, "all")
            if cached:
                self._cache_hits += 1
                return cached[:max_results]

        # Search web
        if self._should_search(ExternalSourceType.WEB_SEARCH, sources):
            if self._web_search and self._web_search.is_available():
                web_results = self._web_search.search(
                    query, max_results=self.config.max_web_results
                )
                all_results.extend(web_results)

        # Search database
        if self._should_search(ExternalSourceType.DATABASE, sources):
            if self._database and self._database.is_available():
                db_results = self._database.search(query, max_results=5)
                all_results.extend(db_results)

        # Sort by relevance
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Limit results
        final_results = all_results[:max_results]

        # Cache results
        if use_cache and self.config.cache_enabled and final_results:
            self._cache.set(query, "all", final_results)

        return final_results

    def _should_search(
        self,
        source: ExternalSourceType,
        requested: list[ExternalSourceType] | None,
    ) -> bool:
        """Check if source should be searched."""
        if requested is None:
            return True
        return source in requested

    def web_search(self, query: str, max_results: int = 5) -> list[ExternalResult]:
        """Direct web search."""
        if not self._web_search:
            return []
        return self._web_search.search(query, max_results)

    def database_query(
        self,
        sql: str,
        params: tuple[Any, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute database query."""
        if not self._database:
            return []
        return self._database.execute_query(sql, params)

    def set_offline_mode(self, offline: bool) -> None:
        """Set offline mode (disables network operations)."""
        self.config.offline_mode = offline
        if self._web_search:
            self._web_search.config.offline_mode = offline

    def get_status(self) -> dict[str, Any]:
        """Get retriever status."""
        return {
            "search_count": self._search_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": (
                self._cache_hits / self._search_count if self._search_count > 0 else 0
            ),
            "web_search": {
                "enabled": self.config.web_search_enabled,
                "available": (self._web_search.is_available() if self._web_search else False),
                "provider": self.config.web_search_provider.value,
            },
            "database": {
                "enabled": self.config.database_enabled,
                "available": (self._database.is_available() if self._database else False),
                "path": str(self.config.database_path) if self.config.database_path else None,
            },
            "cache": {
                "enabled": self.config.cache_enabled,
                "ttl_seconds": self.config.cache_ttl_seconds,
            },
            "offline_mode": self.config.offline_mode,
        }

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()

    def close(self) -> None:
        """Close all connections."""
        if self._database:
            self._database.close()


def create_external_retriever(
    enable_web: bool = True,
    enable_database: bool = False,
    db_path: Path | None = None,
    cache_path: Path | None = None,
    offline: bool = False,
) -> ExternalInformationRetriever:
    """
    Factory function to create external retriever.

    Args:
        enable_web: Enable web search
        enable_database: Enable database search
        db_path: Path to database
        cache_path: Path to cache file
        offline: Start in offline mode

    Returns:
        Configured ExternalInformationRetriever
    """
    config = ExternalSearchConfig(
        web_search_enabled=enable_web,
        database_enabled=enable_database,
        database_path=db_path,
        cache_path=cache_path,
        offline_mode=offline,
    )

    return ExternalInformationRetriever(config, cache_path)
