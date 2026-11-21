# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""No query in the library is assembled from strings.

``WatchStore.findings`` built its ``WHERE`` clause by interpolation and marked
the line ``# noqa: S608``. That quietened Ruff; Bandit does not read ``noqa``,
so the Security Scan and Dependency Security Scan jobs failed on B608 at the
pushed head while every local gate passed. The clause held only fixed text, so
nothing was injectable that day, but a statement put together at run time is
one edit away from carrying a value, and the scan cannot tell the difference.

This walks every ``execute``/``executemany``/``executescript`` call under
``src/`` and requires the statement that reads or writes rows to be a literal.
Schema statements that take an identifier (``ALTER TABLE``, ``PRAGMA``) cannot
bind one as a parameter and are not row queries; they are out of scope here.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "omni_mercury_engine"
_ROW_STATEMENT = re.compile(r"\b(select|insert|update|delete|replace)\b", re.IGNORECASE)
_EXECUTORS = {"execute", "executemany", "executescript"}


def _literal_text(node: ast.AST) -> str:
    """The constant text inside an expression that builds a string."""
    return " ".join(
        part.value
        for part in ast.walk(node)
        if isinstance(part, ast.Constant) and isinstance(part.value, str)
    )


def _assembled_row_statements(tree: ast.AST) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _EXECUTORS
            and node.args
        ):
            continue
        statement = node.args[0]
        assembled = isinstance(statement, ast.JoinedStr | ast.BinOp) or (
            isinstance(statement, ast.Call)
            and isinstance(statement.func, ast.Attribute)
            and statement.func.attr in {"format", "join"}
        )
        if assembled and _ROW_STATEMENT.search(_literal_text(statement)):
            found.append((node.lineno, ast.unparse(statement)))
    return found


def test_the_scan_sees_an_assembled_query() -> None:
    """The check itself: the statement this replaced is one it reports."""
    was = (
        "connection.execute("
        'f"SELECT * FROM findings {where} ORDER BY last_seen DESC, key LIMIT ?", args)'
    )
    assert [line for line, _ in _assembled_row_statements(ast.parse(was))] == [1]
    for built in (
        'c.execute("SELECT * FROM t WHERE " + clause)',
        'c.execute("DELETE FROM {}".format(table))',
        'c.execute(" AND ".join(["UPDATE t SET a=1", clause]))',
    ):
        assert _assembled_row_statements(ast.parse(built)), built
    for fixed in (
        'c.execute("SELECT * FROM t WHERE (:f IS NULL OR feed = :f)", {"f": feed})',
        'c.execute(f"ALTER TABLE findings ADD COLUMN {column} {kind}")',
        "c.execute(_SCHEMA)",
    ):
        assert not _assembled_row_statements(ast.parse(fixed)), fixed


def test_no_row_query_under_src_is_assembled_from_strings() -> None:
    offenders = [
        f"{path.relative_to(SRC.parent.parent)}:{line}: {statement}"
        for path in sorted(SRC.rglob("*.py"))
        for line, statement in _assembled_row_statements(ast.parse(path.read_text("utf-8")))
    ]
    assert offenders == [], "queries assembled from strings:\n" + "\n".join(offenders)
