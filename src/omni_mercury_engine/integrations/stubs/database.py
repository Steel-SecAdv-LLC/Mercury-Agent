"""
Mercury Agent ♱
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
