# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Database service stub for testing and development.

Example:
    >>> db = DatabaseStub()
    >>> result = await db.query("SELECT * FROM anomalies WHERE score > 0.8")
    >>> print(f"Found {len(result.rows)} anomalies")
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Database query result.

    Attributes:
        rows: Result rows.
        columns: Column names.
        row_count: Number of rows returned.
        affected_rows: Number of rows affected (for INSERT/UPDATE/DELETE).
        execution_time_ms: Query execution time.
        query_id: Unique query identifier.
    """

    rows: list[dict[str, Any]]
    columns: list[str]
    row_count: int
    affected_rows: int
    execution_time_ms: float
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rows": self.rows,
            "columns": self.columns,
            "row_count": self.row_count,
            "affected_rows": self.affected_rows,
            "execution_time_ms": self.execution_time_ms,
            "query_id": self.query_id,
        }


class DatabaseError(Exception):
    """Database error."""

    def __init__(self, message: str, query: str | None = None) -> None:
        """Initialize the instance."""
        super().__init__(message)
        self.query = query


class DatabaseStub:
    """Stub implementation of database connection.

    Provides mock database functionality for testing.
    Supports basic SQL parsing and in-memory data storage.

    Example:
        >>> db = DatabaseStub()
        >>> await db.execute("INSERT INTO users (name) VALUES ('Alice')")
        >>> result = await db.query("SELECT * FROM users")
    """

    def __init__(
        self,
        seed: int | None = None,
        latency_ms: tuple[int, int] = (5, 50),
        failure_rate: float = 0.0,
    ):
        """Initialize database stub.

        Args:
            seed: Random seed for reproducibility.
            latency_ms: Min/max simulated latency.
            failure_rate: Probability of simulated failure.
        """
        self._rng = random.Random(seed)
        self._latency_ms = latency_ms
        self._failure_rate = failure_rate

        # In-memory tables
        self._tables: dict[str, list[dict[str, Any]]] = {}
        self._schemas: dict[str, list[str]] = {}

        # Connection state
        self._connected = False
        self._in_transaction = False

        # Metrics
        self._query_count = 0
        self._error_count = 0
        self._total_latency = 0.0

        # Initialize default tables
        self._init_default_tables()

    def _init_default_tables(self) -> None:
        """Initialize default test tables."""
        self._tables["anomalies"] = [
            {
                "id": i,
                "score": self._rng.random(),
                "type": self._rng.choice(["outlier", "pattern", "drift"]),
                "timestamp": datetime.now().isoformat(),
                "source": f"sensor_{self._rng.randint(1, 10)}",
            }
            for i in range(100)
        ]
        self._schemas["anomalies"] = ["id", "score", "type", "timestamp", "source"]

        self._tables["detections"] = [
            {
                "id": i,
                "anomaly_id": self._rng.randint(0, 99),
                "confidence": self._rng.random(),
                "algorithm": self._rng.choice(["isolation_forest", "autoencoder", "lstm"]),
                "processed": self._rng.choice([True, False]),
            }
            for i in range(50)
        ]
        self._schemas["detections"] = ["id", "anomaly_id", "confidence", "algorithm", "processed"]

    async def _simulate_latency(self) -> float:
        """Simulate query latency."""
        latency = self._rng.randint(*self._latency_ms) / 1000.0
        await asyncio.sleep(latency)
        self._total_latency += latency * 1000
        return latency * 1000

    def _maybe_fail(self, query: str) -> None:
        """Potentially raise exception to simulate failure."""
        if self._rng.random() < self._failure_rate:
            self._error_count += 1
            raise DatabaseError("Simulated database failure", query=query)

    def _parse_select(self, query: str) -> tuple[str, list[str], str | None]:
        """Parse simple SELECT query."""
        # Very basic SQL parsing
        query_upper = query.upper()

        # Extract table name
        from_match = re.search(r"FROM\s+(\w+)", query_upper)
        table = from_match.group(1).lower() if from_match else ""

        # Extract columns
        select_match = re.search(r"SELECT\s+(.*?)\s+FROM", query_upper)
        if select_match:
            cols_str = select_match.group(1)
            if cols_str.strip() == "*":
                columns = self._schemas.get(table, [])
            else:
                columns = [c.strip().lower() for c in cols_str.split(",")]
        else:
            columns = []

        # Extract WHERE clause
        where_match = re.search(r"WHERE\s+(.+?)(?:ORDER|LIMIT|$)", query_upper)
        where_clause = where_match.group(1).strip() if where_match else None

        return table, columns, where_clause

    def _filter_rows(
        self,
        rows: list[dict[str, Any]],
        where_clause: str | None,
    ) -> list[dict[str, Any]]:
        """Filter rows based on WHERE clause."""
        if not where_clause:
            return rows

        # Very basic WHERE parsing (supports = and > operators)
        filtered = []
        for row in rows:
            match = True

            # Parse conditions (simplified)
            conditions = re.split(r"\s+AND\s+", where_clause, flags=re.IGNORECASE)
            for condition in conditions:
                # Handle > comparison
                gt_match = re.match(r"(\w+)\s*>\s*(\d+\.?\d*)", condition, re.IGNORECASE)
                if gt_match:
                    col, val = gt_match.groups()
                    if row.get(col.lower(), 0) <= float(val):
                        match = False
                        break
                    continue

                # Handle = comparison
                eq_match = re.match(r"(\w+)\s*=\s*'?([^']+)'?", condition, re.IGNORECASE)
                if eq_match:
                    col, val = eq_match.groups()
                    if str(row.get(col.lower(), "")) != val:
                        match = False
                        break

            if match:
                filtered.append(row)

        return filtered

    async def query(self, sql: str) -> QueryResult:
        """Execute a SELECT query.

        Args:
            sql: SQL query string.

        Returns:
            Query result.

        Raises:
            DatabaseError: On query error.
        """
        self._query_count += 1
        latency = await self._simulate_latency()
        self._maybe_fail(sql)

        try:
            table, columns, where = self._parse_select(sql)

            if table not in self._tables:
                raise DatabaseError(f"Table not found: {table}", query=sql)

            rows = self._tables[table]
            filtered = self._filter_rows(rows, where)

            # Apply column selection
            if columns and columns != self._schemas.get(table, []):
                result_rows = [{c: r.get(c) for c in columns} for r in filtered]
            else:
                result_rows = filtered
                columns = self._schemas.get(table, [])

            return QueryResult(
                rows=result_rows,
                columns=columns,
                row_count=len(result_rows),
                affected_rows=0,
                execution_time_ms=latency,
            )

        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"Query error: {e}", query=sql) from e

    async def execute(self, sql: str) -> QueryResult:
        """Execute an INSERT/UPDATE/DELETE/CREATE/DROP query.

        Args:
            sql: SQL statement.

        Returns:
            Execution result.
        """
        self._query_count += 1
        latency = await self._simulate_latency()
        self._maybe_fail(sql)

        sql_upper = sql.upper().strip()
        affected = 0

        # Handle CREATE TABLE
        if sql_upper.startswith("CREATE TABLE"):
            # Parse table name and columns
            match = re.search(r"CREATE\s+TABLE\s+(\w+)\s*\((.+)\)", sql, re.IGNORECASE)
            if match:
                table_name = match.group(1).lower()
                columns_str = match.group(2)
                # Parse column names (ignore types)
                columns = []
                for col_def in columns_str.split(","):
                    col_name = col_def.strip().split()[0].lower()
                    columns.append(col_name)
                self._tables[table_name] = []
                self._schemas[table_name] = columns
            affected = 0

        # Handle DROP TABLE
        elif sql_upper.startswith("DROP TABLE"):
            match = re.search(r"DROP\s+TABLE\s+(\w+)", sql, re.IGNORECASE)
            if match:
                table_name = match.group(1).lower()
                if table_name in self._tables:
                    del self._tables[table_name]
                if table_name in self._schemas:
                    del self._schemas[table_name]
            affected = 0

        # Handle INSERT
        elif sql_upper.startswith("INSERT"):
            match = re.search(
                r"INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)",
                sql,
                re.IGNORECASE,
            )
            if match:
                table_name = match.group(1).lower()
                columns = [c.strip().lower() for c in match.group(2).split(",")]
                values_str = match.group(3)
                # Parse values (handle strings and numbers)
                values: list[str | int | float | None] = []
                for v in values_str.split(","):
                    v = v.strip()
                    if v.upper() == "NULL":
                        values.append(None)
                    elif v.startswith("'") and v.endswith("'"):
                        values.append(v[1:-1])
                    else:
                        try:
                            if "." in v:
                                values.append(float(v))
                            else:
                                values.append(int(v))
                        except ValueError:
                            values.append(v)
                # Create row
                if table_name in self._tables:
                    row = dict(zip(columns, values))
                    self._tables[table_name].append(row)
            affected = 1

        # Handle UPDATE
        elif sql_upper.startswith("UPDATE"):
            affected = self._rng.randint(0, 10)

        # Handle DELETE
        elif sql_upper.startswith("DELETE"):
            affected = self._rng.randint(0, 5)

        # Check for invalid SQL syntax (basic validation)
        else:
            # If it doesn't match any known pattern, it might be invalid
            valid_starts = ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER")
            if not any(sql_upper.startswith(s) for s in valid_starts):
                raise DatabaseError(f"Syntax error in SQL: {sql}", query=sql)

        return QueryResult(
            rows=[],
            columns=[],
            row_count=0,
            affected_rows=affected,
            execution_time_ms=latency,
        )

    async def transaction(self) -> TransactionContext:
        """Start a transaction.

        Returns:
            Transaction context manager.
        """
        return TransactionContext(self)

    def create_table(
        self,
        name: str,
        columns: list[str],
        initial_data: list[dict[str, Any]] | None = None,
    ) -> None:
        """Create a new table.

        Args:
            name: Table name.
            columns: Column names.
            initial_data: Initial rows.
        """
        self._tables[name] = initial_data or []
        self._schemas[name] = columns

    def get_metrics(self) -> dict[str, Any]:
        """Get database metrics."""
        return {
            "query_count": self._query_count,
            "error_count": self._error_count,
            "avg_latency_ms": (
                self._total_latency / self._query_count if self._query_count > 0 else 0
            ),
            "table_count": len(self._tables),
            "total_rows": sum(len(t) for t in self._tables.values()),
        }

    async def health_check(self) -> dict[str, Any]:
        """Check database health.

        Returns:
            Health status.
        """
        try:
            await self._simulate_latency()
            return {
                "healthy": True,
                "latency_ms": self._rng.randint(*self._latency_ms),
                "connections_available": self._rng.randint(5, 10),
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
            }

    async def connect(self) -> bool:
        """Connect to the database.

        Returns:
            True if connection successful.
        """
        await self._simulate_latency()
        self._maybe_fail("")
        self._connected = True
        return True

    async def disconnect(self) -> None:
        """Disconnect from the database."""
        await self._simulate_latency()
        self._connected = False

    async def is_connected(self) -> bool:
        """Check if connected to database.

        Returns:
            True if connected.
        """
        return self._connected

    async def begin(self) -> None:
        """Begin a transaction."""
        await self._simulate_latency()
        self._maybe_fail("")
        self._in_transaction = True

    async def commit(self) -> None:
        """Commit the current transaction."""
        await self._simulate_latency()
        self._in_transaction = False

    async def rollback(self) -> None:
        """Rollback the current transaction."""
        await self._simulate_latency()
        self._in_transaction = False

    async def list_tables(self) -> list[str]:
        """List all tables in the database.

        Returns:
            List of table names.
        """
        await self._simulate_latency()
        self._maybe_fail("")
        return list(self._tables.keys())

    async def describe(self, table: str) -> list[str]:
        """Get schema for a table.

        Args:
            table: Table name.

        Returns:
            List of column names.

        Raises:
            DatabaseError: If table not found.
        """
        await self._simulate_latency()
        self._maybe_fail("")

        if table not in self._schemas:
            raise DatabaseError(f"Table not found: {table}")

        return self._schemas[table]

    async def get_stats(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database statistics.
        """
        await self._simulate_latency()
        return {
            "total_queries": self._query_count,
            "tables_count": len(self._tables),
            "error_count": self._error_count,
            "avg_latency_ms": (
                self._total_latency / self._query_count if self._query_count > 0 else 0
            ),
            "total_rows": sum(len(t) for t in self._tables.values()),
        }


class TransactionContext:
    """Database transaction context manager."""

    def __init__(self, db: DatabaseStub) -> None:
        """Initialize the instance."""
        self._db = db
        self._committed = False
        self._rolled_back = False

    async def __aenter__(self) -> TransactionContext:
        """Enter transaction."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit transaction."""
        if exc_type is not None and not self._rolled_back:
            await self.rollback()
        elif not self._committed and not self._rolled_back:
            await self.commit()
        return False

    async def commit(self) -> None:
        """Commit transaction."""
        self._committed = True

    async def rollback(self) -> None:
        """Rollback transaction."""
        self._rolled_back = True

    async def query(self, sql: str) -> QueryResult:
        """Execute query within transaction."""
        return await self._db.query(sql)

    async def execute(self, sql: str) -> QueryResult:
        """Execute statement within transaction."""
        return await self._db.execute(sql)


class DatabaseBackend(Enum):
    """Supported database backends."""

    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    STUB = "stub"


class AsyncDatabase:
    """Production-ready async database client.

    Supports PostgreSQL via asyncpg and SQLite via aiosqlite,
    with automatic fallback to in-memory stub.

    Example:
        >>> # Using PostgreSQL
        >>> db = AsyncDatabase(
        ...     backend=DatabaseBackend.POSTGRESQL,
        ...     host="localhost",
        ...     database="mercury",
        ...     user="postgres",
        ...     password=os.getenv("POSTGRES_PASSWORD")
        ... )
        >>> await db.connect()
        >>> result = await db.query("SELECT * FROM anomalies")

        >>> # Using SQLite
        >>> db = AsyncDatabase(
        ...     backend=DatabaseBackend.SQLITE,
        ...     database="mercury.db"
        ... )

        >>> # From environment variables
        >>> db = AsyncDatabase.from_env()
    """

    def __init__(
        self,
        backend: DatabaseBackend = DatabaseBackend.STUB,
        host: str | None = None,
        port: int = 5432,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        min_connections: int = 1,
        max_connections: int = 10,
        fallback_to_stub: bool = True,
    ):
        """Initialize async database.

        Args:
            backend: Database backend to use.
            host: Database host (PostgreSQL only).
            port: Database port (PostgreSQL only).
            database: Database name or SQLite file path.
            user: Database username (PostgreSQL only).
            password: Database password (PostgreSQL only).
            min_connections: Minimum pool connections.
            max_connections: Maximum pool connections.
            fallback_to_stub: Fall back to stub on connection failure.
        """
        self.backend = backend
        self.host = host or os.getenv("DATABASE_HOST", "localhost")
        self.port = port
        self.database = database or os.getenv("DATABASE_NAME", "mercury")
        self.user = user or os.getenv("DATABASE_USER", "postgres")
        self.password = password or os.getenv("DATABASE_PASSWORD")
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.fallback_to_stub = fallback_to_stub

        self._pool: Any = None
        self._connection: Any = None
        self._stub = DatabaseStub()
        self._connected = False
        self._connection_error: str | None = None

        # Metrics
        self._query_count = 0
        self._errors = 0
        self._fallback_count = 0

    @classmethod
    def from_env(cls) -> AsyncDatabase:
        """Create database from environment variables.

        Environment variables:
            DATABASE_BACKEND: Backend type (postgresql, sqlite, stub)
            DATABASE_HOST: Database host
            DATABASE_PORT: Database port
            DATABASE_NAME: Database name or SQLite path
            DATABASE_USER: Database username
            DATABASE_PASSWORD: Database password
            DATABASE_MIN_CONNECTIONS: Min pool size
            DATABASE_MAX_CONNECTIONS: Max pool size
        """
        backend_str = os.getenv("DATABASE_BACKEND", "stub").lower()
        backend_map = {
            "postgresql": DatabaseBackend.POSTGRESQL,
            "postgres": DatabaseBackend.POSTGRESQL,
            "pg": DatabaseBackend.POSTGRESQL,
            "sqlite": DatabaseBackend.SQLITE,
            "sqlite3": DatabaseBackend.SQLITE,
            "stub": DatabaseBackend.STUB,
            "memory": DatabaseBackend.STUB,
        }
        backend = backend_map.get(backend_str, DatabaseBackend.STUB)

        return cls(
            backend=backend,
            host=os.getenv("DATABASE_HOST", "localhost"),
            port=int(os.getenv("DATABASE_PORT", "5432")),
            database=os.getenv("DATABASE_NAME", "mercury"),
            user=os.getenv("DATABASE_USER", "postgres"),
            password=os.getenv("DATABASE_PASSWORD"),
            min_connections=int(os.getenv("DATABASE_MIN_CONNECTIONS", "1")),
            max_connections=int(os.getenv("DATABASE_MAX_CONNECTIONS", "10")),
        )

    async def connect(self) -> bool:
        """Connect to the database.

        Returns:
            True if connection successful.
        """
        if self._connected:
            return True

        if self.backend == DatabaseBackend.STUB:
            self._connected = True
            return True

        try:
            if self.backend == DatabaseBackend.POSTGRESQL:
                return await self._connect_postgresql()
            elif self.backend == DatabaseBackend.SQLITE:
                return await self._connect_sqlite()
            else:
                self._connected = True
                return True

        except Exception as e:
            self._connection_error = str(e)
            logger.warning(f"Database connection failed: {e}")
            if self.fallback_to_stub:
                logger.info("Falling back to stub database")
                self._connected = True
                return True
            return False

    async def _connect_postgresql(self) -> bool:
        """Connect to PostgreSQL using asyncpg."""
        try:
            import asyncpg
        except ImportError:
            self._connection_error = "asyncpg not installed (pip install asyncpg)"
            logger.warning(self._connection_error)
            return False

        self._pool = await asyncpg.create_pool(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
            min_size=self.min_connections,
            max_size=self.max_connections,
        )

        # Test connection
        async with self._pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

        self._connected = True
        logger.info(f"Connected to PostgreSQL at {self.host}:{self.port}/{self.database}")
        return True

    async def _connect_sqlite(self) -> bool:
        """Connect to SQLite using aiosqlite."""
        try:
            import aiosqlite
        except ImportError:
            self._connection_error = "aiosqlite not installed (pip install aiosqlite)"
            logger.warning(self._connection_error)
            return False

        db_path = self.database or ":memory:"
        self._connection = await aiosqlite.connect(db_path)
        self._connection.row_factory = aiosqlite.Row

        # Enable foreign keys
        await self._connection.execute("PRAGMA foreign_keys = ON")

        self._connected = True
        logger.info(f"Connected to SQLite database: {db_path}")
        return True

    async def disconnect(self) -> None:
        """Disconnect from the database."""
        if self._pool:
            await self._pool.close()
            self._pool = None
        if self._connection:
            await self._connection.close()
            self._connection = None
        self._connected = False
        logger.info("Database connection closed")

    async def query(self, sql: str, *args: Any) -> QueryResult:
        """Execute a SELECT query.

        Args:
            sql: SQL query string.
            *args: Query parameters.

        Returns:
            Query result.
        """
        self._query_count += 1
        start_time = datetime.now()

        if self.backend == DatabaseBackend.STUB or not self._connected:
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.query(sql)
            raise DatabaseError("Not connected", query=sql)

        try:
            if self.backend == DatabaseBackend.POSTGRESQL:
                return await self._query_postgresql(sql, args, start_time)
            elif self.backend == DatabaseBackend.SQLITE:
                return await self._query_sqlite(sql, args, start_time)
            else:
                return await self._stub.query(sql)

        except Exception as e:
            self._errors += 1
            logger.warning(f"Query error: {e}")
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.query(sql)
            raise DatabaseError(str(e), query=sql) from e

    async def _query_postgresql(
        self, sql: str, args: tuple[Any, ...], start_time: datetime
    ) -> QueryResult:
        """Execute PostgreSQL query."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)

            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            result_rows = [dict(row) for row in rows]
            columns = list(rows[0].keys()) if rows else []

            return QueryResult(
                rows=result_rows,
                columns=columns,
                row_count=len(result_rows),
                affected_rows=0,
                execution_time_ms=elapsed,
            )

    async def _query_sqlite(
        self, sql: str, args: tuple[Any, ...], start_time: datetime
    ) -> QueryResult:
        """Execute SQLite query."""
        cursor = await self._connection.execute(sql, args)
        rows = await cursor.fetchall()

        elapsed = (datetime.now() - start_time).total_seconds() * 1000

        # Convert Row objects to dicts
        if rows:
            columns = list(rows[0].keys())
            result_rows = [dict(row) for row in rows]
        else:
            columns = []
            result_rows = []

        return QueryResult(
            rows=result_rows,
            columns=columns,
            row_count=len(result_rows),
            affected_rows=0,
            execution_time_ms=elapsed,
        )

    async def execute(self, sql: str, *args: Any) -> QueryResult:
        """Execute an INSERT/UPDATE/DELETE/CREATE/DROP query.

        Args:
            sql: SQL statement.
            *args: Query parameters.

        Returns:
            Execution result.
        """
        self._query_count += 1
        start_time = datetime.now()

        if self.backend == DatabaseBackend.STUB or not self._connected:
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.execute(sql)
            raise DatabaseError("Not connected", query=sql)

        try:
            if self.backend == DatabaseBackend.POSTGRESQL:
                return await self._execute_postgresql(sql, args, start_time)
            elif self.backend == DatabaseBackend.SQLITE:
                return await self._execute_sqlite(sql, args, start_time)
            else:
                return await self._stub.execute(sql)

        except Exception as e:
            self._errors += 1
            logger.warning(f"Execute error: {e}")
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.execute(sql)
            raise DatabaseError(str(e), query=sql) from e

    async def _execute_postgresql(
        self, sql: str, args: tuple[Any, ...], start_time: datetime
    ) -> QueryResult:
        """Execute PostgreSQL statement."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, *args)

            elapsed = (datetime.now() - start_time).total_seconds() * 1000

            # Parse affected rows from result string (e.g., "INSERT 0 1")
            affected = 0
            if result:
                parts = result.split()
                if len(parts) >= 2:
                    try:
                        affected = int(parts[-1])
                    except ValueError:
                        pass

            return QueryResult(
                rows=[],
                columns=[],
                row_count=0,
                affected_rows=affected,
                execution_time_ms=elapsed,
            )

    async def _execute_sqlite(
        self, sql: str, args: tuple[Any, ...], start_time: datetime
    ) -> QueryResult:
        """Execute SQLite statement."""
        cursor = await self._connection.execute(sql, args)
        await self._connection.commit()

        elapsed = (datetime.now() - start_time).total_seconds() * 1000

        return QueryResult(
            rows=[],
            columns=[],
            row_count=0,
            affected_rows=cursor.rowcount,
            execution_time_ms=elapsed,
        )

    async def executemany(self, sql: str, args_list: list[tuple[Any, ...]]) -> QueryResult:
        """Execute a statement with multiple parameter sets.

        Args:
            sql: SQL statement with placeholders.
            args_list: List of parameter tuples.

        Returns:
            Execution result.
        """
        self._query_count += 1
        start_time = datetime.now()

        if self.backend == DatabaseBackend.STUB or not self._connected:
            if self.fallback_to_stub:
                self._fallback_count += 1
                return QueryResult(
                    rows=[],
                    columns=[],
                    row_count=0,
                    affected_rows=len(args_list),
                    execution_time_ms=0,
                )
            raise DatabaseError("Not connected", query=sql)

        try:
            if self.backend == DatabaseBackend.POSTGRESQL:
                async with self._pool.acquire() as conn:
                    await conn.executemany(sql, args_list)
            elif self.backend == DatabaseBackend.SQLITE:
                await self._connection.executemany(sql, args_list)
                await self._connection.commit()

            elapsed = (datetime.now() - start_time).total_seconds() * 1000

            return QueryResult(
                rows=[],
                columns=[],
                row_count=0,
                affected_rows=len(args_list),
                execution_time_ms=elapsed,
            )

        except Exception as e:
            self._errors += 1
            logger.warning(f"Executemany error: {e}")
            if self.fallback_to_stub:
                self._fallback_count += 1
                return QueryResult(
                    rows=[],
                    columns=[],
                    row_count=0,
                    affected_rows=0,
                    execution_time_ms=0,
                )
            raise DatabaseError(str(e), query=sql) from e

    async def transaction(self) -> AsyncTransactionContext:
        """Start a transaction.

        Returns:
            Transaction context manager.
        """
        return AsyncTransactionContext(self)

    async def health_check(self) -> dict[str, Any]:
        """Check database health.

        Returns:
            Health status.
        """
        if not self._connected:
            return {
                "healthy": False,
                "backend": self.backend.value,
                "error": self._connection_error or "Not connected",
            }

        try:
            if self.backend == DatabaseBackend.POSTGRESQL:
                async with self._pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
            elif self.backend == DatabaseBackend.SQLITE:
                await self._connection.execute("SELECT 1")
            else:
                return await self._stub.health_check()

            return {
                "healthy": True,
                "backend": self.backend.value,
                "host": self.host if self.backend == DatabaseBackend.POSTGRESQL else None,
                "database": self.database,
            }

        except Exception as e:
            return {
                "healthy": False,
                "backend": self.backend.value,
                "error": str(e),
            }

    def get_metrics(self) -> dict[str, Any]:
        """Get database metrics."""
        return {
            "backend": self.backend.value,
            "connected": self._connected,
            "query_count": self._query_count,
            "errors": self._errors,
            "fallback_count": self._fallback_count,
            "error_rate": self._errors / self._query_count if self._query_count > 0 else 0,
        }

    async def __aenter__(self) -> AsyncDatabase:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Async context manager exit."""
        await self.disconnect()
        return False


class AsyncTransactionContext:
    """Async database transaction context manager."""

    def __init__(self, db: AsyncDatabase) -> None:
        """Initialize the instance."""
        self._db = db
        self._conn: Any = None
        self._transaction: Any = None
        self._committed = False
        self._rolled_back = False

    async def __aenter__(self) -> AsyncTransactionContext:
        """Enter transaction."""
        if self._db.backend == DatabaseBackend.POSTGRESQL and self._db._pool:
            self._conn = await self._db._pool.acquire()
            self._transaction = self._conn.transaction()
            await self._transaction.start()
        elif self._db.backend == DatabaseBackend.SQLITE and self._db._connection:
            # SQLite transactions are implicit
            pass
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit transaction."""
        if exc_type is not None and not self._rolled_back:
            await self.rollback()
        elif not self._committed and not self._rolled_back:
            await self.commit()

        # Release PostgreSQL connection
        if self._conn and self._db._pool:
            await self._db._pool.release(self._conn)
            self._conn = None

        return False

    async def commit(self) -> None:
        """Commit transaction."""
        if self._db.backend == DatabaseBackend.POSTGRESQL and self._transaction:
            await self._transaction.commit()
        elif self._db.backend == DatabaseBackend.SQLITE and self._db._connection:
            await self._db._connection.commit()
        self._committed = True

    async def rollback(self) -> None:
        """Rollback transaction."""
        if self._db.backend == DatabaseBackend.POSTGRESQL and self._transaction:
            await self._transaction.rollback()
        elif self._db.backend == DatabaseBackend.SQLITE and self._db._connection:
            await self._db._connection.rollback()
        self._rolled_back = True

    async def query(self, sql: str, *args: Any) -> QueryResult:
        """Execute query within transaction."""
        if self._db.backend == DatabaseBackend.POSTGRESQL and self._conn:
            start_time = datetime.now()
            rows = await self._conn.fetch(sql, *args)
            elapsed = (datetime.now() - start_time).total_seconds() * 1000

            result_rows = [dict(row) for row in rows]
            columns = list(rows[0].keys()) if rows else []

            return QueryResult(
                rows=result_rows,
                columns=columns,
                row_count=len(result_rows),
                affected_rows=0,
                execution_time_ms=elapsed,
            )
        else:
            return await self._db.query(sql, *args)

    async def execute(self, sql: str, *args: Any) -> QueryResult:
        """Execute statement within transaction."""
        if self._db.backend == DatabaseBackend.POSTGRESQL and self._conn:
            start_time = datetime.now()
            result = await self._conn.execute(sql, *args)
            elapsed = (datetime.now() - start_time).total_seconds() * 1000

            affected = 0
            if result:
                parts = result.split()
                if len(parts) >= 2:
                    try:
                        affected = int(parts[-1])
                    except ValueError:
                        pass

            return QueryResult(
                rows=[],
                columns=[],
                row_count=0,
                affected_rows=affected,
                execution_time_ms=elapsed,
            )
        else:
            return await self._db.execute(sql, *args)


# Factory function for database creation
def create_database(
    backend: str = "stub",
    host: str | None = None,
    port: int = 5432,
    database: str | None = None,
    user: str | None = None,
    password: str | None = None,
    **kwargs: Any,
) -> AsyncDatabase | DatabaseStub:
    """Create database with appropriate backend.

    Args:
        backend: Database backend ("postgresql", "sqlite", "stub").
        host: Database host.
        port: Database port.
        database: Database name or SQLite path.
        user: Database username.
        password: Database password.
        **kwargs: Additional backend-specific options.

    Returns:
        Configured database instance.

    Example:
        >>> # For testing
        >>> db = create_database(backend="stub")

        >>> # For production with PostgreSQL
        >>> db = create_database(
        ...     backend="postgresql",
        ...     host="localhost",
        ...     database="mercury",
        ...     user="postgres",
        ...     password="secret"
        ... )

        >>> # For SQLite
        >>> db = create_database(
        ...     backend="sqlite",
        ...     database="mercury.db"
        ... )
    """
    backend_map = {
        "postgresql": DatabaseBackend.POSTGRESQL,
        "postgres": DatabaseBackend.POSTGRESQL,
        "pg": DatabaseBackend.POSTGRESQL,
        "sqlite": DatabaseBackend.SQLITE,
        "sqlite3": DatabaseBackend.SQLITE,
        "stub": DatabaseBackend.STUB,
        "memory": DatabaseBackend.STUB,
    }

    backend_enum = backend_map.get(backend.lower(), DatabaseBackend.STUB)

    if backend_enum == DatabaseBackend.STUB:
        return DatabaseStub()

    return AsyncDatabase(
        backend=backend_enum,
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        **kwargs,
    )
