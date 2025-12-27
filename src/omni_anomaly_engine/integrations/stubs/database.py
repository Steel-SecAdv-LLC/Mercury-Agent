"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Database service stub for testing and development.

Example:
    >>> db = DatabaseStub()
    >>> result = await db.query("SELECT * FROM anomalies WHERE score > 0.8")
    >>> print(f"Found {len(result.rows)} anomalies")
"""

from __future__ import annotations

import asyncio
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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
        """Execute an INSERT/UPDATE/DELETE query.

        Args:
            sql: SQL statement.

        Returns:
            Execution result.
        """
        self._query_count += 1
        latency = await self._simulate_latency()
        self._maybe_fail(sql)

        sql_upper = sql.upper().strip()

        # Simulate affected rows
        if sql_upper.startswith("INSERT"):
            affected = 1
        elif sql_upper.startswith("UPDATE"):
            affected = self._rng.randint(0, 10)
        elif sql_upper.startswith("DELETE"):
            affected = self._rng.randint(0, 5)
        else:
            affected = 0

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


class TransactionContext:
    """Database transaction context manager."""

    def __init__(self, db: DatabaseStub) -> None:
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
