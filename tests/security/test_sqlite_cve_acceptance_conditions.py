# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The conditions behind the two accepted libsqlite3-0 CVEs stay true.

``.trivyignore`` accepts CVE-2026-119373 (heap overflow in SQLite's FTS5
extension) and CVE-2026-119586 (command injection in the ``sqlite3``
command-line shell's ``.recover``) because neither code path exists in the
shipped image: Mercury creates no FTS virtual table, and the Dockerfile
installs no ``sqlite3`` CLI package. A ledger entry states those facts once;
this test re-checks them on every run, so the acceptance fails loudly the day
a module adds ``CREATE VIRTUAL TABLE ... USING fts5`` or the runtime stage
installs the shell, instead of the entry quietly covering a path that has
become reachable.

Offline, dependency-free, and import-free of the engine so it runs in the
torch-free security lane.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "omni_mercury_engine"
_DOCKERFILE = _REPO / "Dockerfile"
_LEDGER = _REPO / ".trivyignore"

_ACCEPTED = ("CVE-2026-119373", "CVE-2026-119586")

#: Any FTS virtual-table creation, FTS3/4/5, in any SQL string.
_FTS_TABLE = re.compile(r"CREATE\s+VIRTUAL\s+TABLE[^;]*?USING\s+fts\d?", re.IGNORECASE | re.DOTALL)

#: An ``apt-get install`` (or ``apt install``) argument list in the Dockerfile.
_APT_INSTALL = re.compile(
    r"apt(?:-get)?\s+install[^&]*?(?=&&|\n\n|RUN |\Z)", re.IGNORECASE | re.DOTALL
)


def _ledger_entries() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in _LEDGER.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*(CVE-\d{4}-\d+)\s+exp:(\d{4}-\d{2}-\d{2})", line)
        if m:
            entries[m.group(1)] = m.group(2)
    return entries


def test_ledger_carries_both_sqlite_acceptances_with_an_expiry() -> None:
    entries = _ledger_entries()
    for cve in _ACCEPTED:
        assert cve in entries, f"{cve} is not an active ledger entry"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", entries[cve])


def test_engine_creates_no_fts_virtual_table() -> None:
    """CVE-2026-119373 is in FTS5; the acceptance rests on Mercury never building one."""
    offenders = [
        str(path.relative_to(_REPO))
        for path in _SRC.rglob("*.py")
        if _FTS_TABLE.search(path.read_text(encoding="utf-8", errors="replace"))
    ]
    assert offenders == [], (
        "an FTS virtual table is created in the engine; CVE-2026-119373 is no longer "
        f"unreachable and its ledger entry must be re-reviewed: {offenders}"
    )


def _apt_package_names(text: str) -> set[str]:
    """Package names an ``apt-get install`` in ``text`` would install, normalised.

    APT accepts a qualified name: ``sqlite3:amd64`` (architecture),
    ``sqlite3=3.46.1-1`` (version), ``sqlite3/trixie`` (release). Each would
    install the shell while a bare-token comparison saw a different string, so
    the qualifiers are stripped before the name is compared.
    """
    installed: set[str] = set()
    for block in _APT_INSTALL.findall(text):
        # Strip comments and line continuations, then take the bare package tokens.
        body = "\n".join(line.split("#", 1)[0] for line in block.splitlines())
        for token in body.replace("\\", " ").split():
            if token.startswith("-") or token in {"apt-get", "apt", "install", "&&"}:
                continue
            name = re.split(r"[:=/]", token, maxsplit=1)[0]
            installed.add(name.lower())
    return installed


def test_qualified_apt_package_names_are_normalised() -> None:
    """A qualified ``sqlite3`` spelling must be seen as the shell, not pass as another token."""
    for qualified in (
        "sqlite3:amd64",
        "sqlite3=3.46.1-1",
        "sqlite3/trixie",
        "SQLite3:arm64=3.46.1",
    ):
        text = f"RUN apt-get update && apt-get install -y --no-install-recommends curl \\\n    {qualified} && rm -rf /var/lib/apt/lists/*\n"
        assert "sqlite3" in _apt_package_names(text), qualified
    assert _apt_package_names("RUN apt-get install -y libsqlite3-0\n") == {"libsqlite3-0"}


def test_dockerfile_installs_no_sqlite3_cli() -> None:
    """CVE-2026-119586 is in the sqlite3 shell; the acceptance rests on it not being installed."""
    text = _DOCKERFILE.read_text(encoding="utf-8")
    installed = _apt_package_names(text)
    assert "sqlite3" not in installed, (
        "the Dockerfile installs the sqlite3 command-line shell; CVE-2026-119586 is no "
        "longer unreachable and its ledger entry must be re-reviewed"
    )
    assert (
        "libsqlite3-0" in text or "libsqlite3" not in installed
    )  # the library is base-image provided
