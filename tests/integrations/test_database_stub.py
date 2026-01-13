"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Tests for integrations/stubs/database.py module.
Comprehensive test coverage for database stub functionality.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.integrations.stubs.database import (
    DatabaseError,
    DatabaseStub,
    QueryResult,
)


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_basic_result(self):
        """Test basic query result creation."""
        result = QueryResult(
            rows=[{"id": 1, "name": "Alice"}],
            columns=["id", "name"],
            row_count=1,
            affected_rows=0,
            execution_time_ms=5.5,
        )
        assert result.row_count == 1
        assert len(result.rows) == 1
        assert result.rows[0]["name"] == "Alice"

    def test_empty_result(self):
        """Test empty query result."""
        result = QueryResult(
            rows=[],
            columns=["id", "name"],
            row_count=0,
            affected_rows=0,
            execution_time_ms=1.0,
        )
        assert result.row_count == 0
        assert len(result.rows) == 0

    def test_affected_rows(self):
        """Test affected rows for DML operations."""
        result = QueryResult(
            rows=[],
            columns=[],
            row_count=0,
            affected_rows=5,
            execution_time_ms=10.0,
        )
        assert result.affected_rows == 5

    def test_query_id_generated(self):
        """Test query ID is auto-generated."""
        result = QueryResult(
            rows=[],
            columns=[],
            row_count=0,
            affected_rows=0,
            execution_time_ms=1.0,
        )
        assert result.query_id is not None
        assert len(result.query_id) > 0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = QueryResult(
            rows=[{"id": 1}],
            columns=["id"],
            row_count=1,
            affected_rows=0,
            execution_time_ms=2.5,
        )
        d = result.to_dict()
        assert "rows" in d
        assert "columns" in d
        assert "row_count" in d
        assert "affected_rows" in d
        assert "execution_time_ms" in d
        assert "query_id" in d


class TestDatabaseError:
    """Tests for DatabaseError exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = DatabaseError("Connection failed")
        assert str(error) == "Connection failed"
        assert error.query is None

    def test_error_with_query(self):
        """Test error with query."""
        error = DatabaseError("Syntax error", query="SELECT * FORM users")
        assert error.query == "SELECT * FORM users"


class TestDatabaseStubInitialization:
    """Tests for DatabaseStub initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        db = DatabaseStub()
        assert db._failure_rate == 0.0
        assert db._latency_ms == (5, 50)

    def test_custom_seed(self):
        """Test initialization with custom seed."""
        db = DatabaseStub(seed=42)
        assert db._rng is not None

    def test_custom_latency(self):
        """Test initialization with custom latency."""
        db = DatabaseStub(latency_ms=(1, 5))
        assert db._latency_ms == (1, 5)

    def test_custom_failure_rate(self):
        """Test initialization with custom failure rate."""
        db = DatabaseStub(failure_rate=0.1)
        assert db._failure_rate == 0.1


class TestDatabaseStubQuery:
    """Tests for database query operations."""

    @pytest.fixture
    def db(self):
        """Create database fixture."""
        return DatabaseStub(seed=42, latency_ms=(0, 1))

    @pytest.mark.asyncio
    async def test_create_table(self, db):
        """Test creating a table."""
        result = await db.execute("CREATE TABLE users (id INTEGER, name TEXT, email TEXT)")
        assert result is not None

    @pytest.mark.asyncio
    async def test_insert(self, db):
        """Test inserting a row."""
        await db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        result = await db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        assert result.affected_rows >= 0

    @pytest.mark.asyncio
    async def test_select_all(self, db):
        """Test selecting all rows."""
        await db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        await db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        await db.execute("INSERT INTO users (id, name) VALUES (2, 'Bob')")

        result = await db.query("SELECT * FROM users")
        assert result.row_count >= 0

    @pytest.mark.asyncio
    async def test_select_with_where(self, db):
        """Test selecting with WHERE clause."""
        await db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        await db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        await db.execute("INSERT INTO users (id, name) VALUES (2, 'Bob')")

        result = await db.query("SELECT * FROM users WHERE id = 1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_update(self, db):
        """Test updating rows."""
        await db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        await db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")

        result = await db.execute("UPDATE users SET name = 'Alicia' WHERE id = 1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_delete(self, db):
        """Test deleting rows."""
        await db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        await db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")

        result = await db.execute("DELETE FROM users WHERE id = 1")
        assert result is not None


class TestDatabaseStubTransactions:
    """Tests for transaction support."""

    @pytest.fixture
    def db(self):
        """Create database fixture."""
        return DatabaseStub(seed=42, latency_ms=(0, 1))

    @pytest.mark.asyncio
    async def test_begin_transaction(self, db):
        """Test beginning a transaction."""
        await db.begin()
        assert db._in_transaction is True

    @pytest.mark.asyncio
    async def test_commit_transaction(self, db):
        """Test committing a transaction."""
        await db.begin()
        await db.execute("CREATE TABLE test (id INTEGER)")
        await db.commit()
        assert db._in_transaction is False

    @pytest.mark.asyncio
    async def test_rollback_transaction(self, db):
        """Test rolling back a transaction."""
        await db.begin()
        await db.rollback()
        assert db._in_transaction is False


class TestDatabaseStubSchemaOperations:
    """Tests for schema operations."""

    @pytest.fixture
    def db(self):
        """Create database fixture."""
        return DatabaseStub(seed=42, latency_ms=(0, 1))

    @pytest.mark.asyncio
    async def test_list_tables(self, db):
        """Test listing tables."""
        await db.execute("CREATE TABLE users (id INTEGER)")
        await db.execute("CREATE TABLE products (id INTEGER)")

        tables = await db.list_tables()
        assert "users" in tables
        assert "products" in tables

    @pytest.mark.asyncio
    async def test_describe_table(self, db):
        """Test describing table schema."""
        await db.execute("CREATE TABLE users (id INTEGER, name TEXT, email TEXT)")

        schema = await db.describe("users")
        assert "id" in schema or len(schema) > 0

    @pytest.mark.asyncio
    async def test_drop_table(self, db):
        """Test dropping a table."""
        await db.execute("CREATE TABLE temp (id INTEGER)")
        result = await db.execute("DROP TABLE temp")
        assert result is not None

        tables = await db.list_tables()
        assert "temp" not in tables


class TestDatabaseStubDataTypes:
    """Tests for different data types."""

    @pytest.fixture
    def db(self):
        """Create database fixture."""
        return DatabaseStub(seed=42, latency_ms=(0, 1))

    @pytest.mark.asyncio
    async def test_integer_values(self, db):
        """Test integer values."""
        await db.execute("CREATE TABLE nums (value INTEGER)")
        await db.execute("INSERT INTO nums (value) VALUES (42)")

        result = await db.query("SELECT * FROM nums")
        assert result is not None

    @pytest.mark.asyncio
    async def test_float_values(self, db):
        """Test float values."""
        await db.execute("CREATE TABLE floats (value REAL)")
        await db.execute("INSERT INTO floats (value) VALUES (3.14159)")

        result = await db.query("SELECT * FROM floats")
        assert result is not None

    @pytest.mark.asyncio
    async def test_text_values(self, db):
        """Test text values."""
        await db.execute("CREATE TABLE texts (value TEXT)")
        await db.execute("INSERT INTO texts (value) VALUES ('Hello, World!')")

        result = await db.query("SELECT * FROM texts")
        assert result is not None

    @pytest.mark.asyncio
    async def test_null_values(self, db):
        """Test NULL values."""
        await db.execute("CREATE TABLE nulls (value TEXT)")
        await db.execute("INSERT INTO nulls (value) VALUES (NULL)")

        result = await db.query("SELECT * FROM nulls")
        assert result is not None


class TestDatabaseStubAggregations:
    """Tests for aggregation functions."""

    @pytest.fixture
    def db(self):
        """Create database fixture."""
        return DatabaseStub(seed=42, latency_ms=(0, 1))

    @pytest.mark.asyncio
    async def test_count(self, db):
        """Test COUNT aggregation."""
        await db.execute("CREATE TABLE items (id INTEGER)")
        await db.execute("INSERT INTO items (id) VALUES (1)")
        await db.execute("INSERT INTO items (id) VALUES (2)")
        await db.execute("INSERT INTO items (id) VALUES (3)")

        result = await db.query("SELECT COUNT(*) FROM items")
        assert result is not None


class TestDatabaseStubStatistics:
    """Tests for database statistics."""

    @pytest.fixture
    def db(self):
        """Create database fixture."""
        return DatabaseStub(seed=42, latency_ms=(0, 1))

    @pytest.mark.asyncio
    async def test_get_stats(self, db):
        """Test getting database statistics."""
        await db.execute("CREATE TABLE test (id INTEGER)")
        await db.query("SELECT * FROM test")

        stats = await db.get_stats()
        assert "total_queries" in stats
        assert "tables_count" in stats

    @pytest.mark.asyncio
    async def test_query_count_tracking(self, db):
        """Test query count tracking."""
        await db.execute("CREATE TABLE test (id INTEGER)")
        await db.query("SELECT * FROM test")
        await db.query("SELECT * FROM test")

        stats = await db.get_stats()
        assert stats["total_queries"] >= 2


class TestDatabaseStubErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def db(self):
        """Create database fixture."""
        return DatabaseStub(seed=42, latency_ms=(0, 1))

    @pytest.mark.asyncio
    async def test_query_nonexistent_table(self, db):
        """Test querying nonexistent table."""
        with pytest.raises(DatabaseError):
            await db.query("SELECT * FROM nonexistent")

    @pytest.mark.asyncio
    async def test_syntax_error(self, db):
        """Test SQL syntax error handling."""
        # Invalid SQL should raise error
        with pytest.raises(DatabaseError):
            await db.execute("SELEC * FORM users")


class TestDatabaseStubConnectionManagement:
    """Tests for connection management."""

    @pytest.fixture
    def db(self):
        """Create database fixture."""
        return DatabaseStub(seed=42, latency_ms=(0, 1))

    @pytest.mark.asyncio
    async def test_connect(self, db):
        """Test connecting to database."""
        await db.connect()
        assert db._connected is True

    @pytest.mark.asyncio
    async def test_disconnect(self, db):
        """Test disconnecting from database."""
        await db.connect()
        await db.disconnect()
        assert db._connected is False

    @pytest.mark.asyncio
    async def test_is_connected(self, db):
        """Test connection status check."""
        assert await db.is_connected() is False
        await db.connect()
        assert await db.is_connected() is True
