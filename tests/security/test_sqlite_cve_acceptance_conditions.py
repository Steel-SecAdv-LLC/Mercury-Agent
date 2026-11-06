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
import shlex
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "omni_mercury_engine"
_DOCKERFILE = _REPO / "Dockerfile"
_LEDGER = _REPO / ".trivyignore"

_ACCEPTED = (
    "CVE-2026-119373",
    "CVE-2026-119586",
    # Published 2026-11-06, after #399 merged, against an unchanged image.
    "CVE-2026-119374",  # arbitrary code execution via a corrupted FTS5 index
    "CVE-2026-119379",  # arbitrary code execution via crafted session changesets
)

#: Anything that would reach SQLite's session extension, the layer
#: CVE-2026-119379 lives in.
_SESSION_API = re.compile(
    r"sqlite3session|sqlite3changeset|sqlite3changegroup|sqlite3rebaser"
    r"|\bchangeset\b|\bpatchset\b",
    re.IGNORECASE,
)

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


def test_ledger_carries_every_sqlite_acceptance_with_an_expiry() -> None:
    entries = _ledger_entries()
    for cve in _ACCEPTED:
        assert cve in entries, f"{cve} is not an active ledger entry"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", entries[cve])


def test_engine_creates_no_fts_virtual_table() -> None:
    """CVE-2026-119373 and CVE-2026-119374 are both in FTS5.

    Both acceptances rest on the same fact: Mercury builds no FTS table, so
    there is no FTS5 index for a prefix MATCH to walk or for a corruption to
    be planted in. FTS5 is compiled into the library (``ENABLE_FTS5`` is in
    ``pragma compile_options``), so the extension is present and the
    unreachability is a property of this codebase, not of the build.
    """
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
        # Strip comments and line continuations, then tokenise as the shell
        # would: a quoted name ("sqlite3", 'sqlite3') installs the same package
        # as a bare one, so the quotes must not survive into the comparison.
        body = "\n".join(line.split("#", 1)[0] for line in block.splitlines())
        try:
            tokens = shlex.split(body.replace("\\\n", " "), comments=False, posix=True)
        except ValueError:
            tokens = body.replace("\\", " ").split()
        for token in tokens:
            if token.startswith("-") or token in {"apt-get", "apt", "install", "&&"}:
                continue
            name = re.split(r"[:=/]", token, maxsplit=1)[0]
            if name:
                installed.add(name.lower())
    return installed


def test_qualified_apt_package_names_are_normalised() -> None:
    """A qualified ``sqlite3`` spelling must be seen as the shell, not pass as another token."""
    for qualified in (
        "sqlite3:amd64",
        "sqlite3=3.46.1-1",
        "sqlite3/trixie",
        "SQLite3:arm64=3.46.1",
        '"sqlite3"',
        "'sqlite3'",
        '"sqlite3:amd64"',
        "'sqlite3=3.46.1-1'",
    ):
        text = f"RUN apt-get update && apt-get install -y --no-install-recommends curl \\\n    {qualified} && rm -rf /var/lib/apt/lists/*\n"
        assert "sqlite3" in _apt_package_names(text), qualified
    assert _apt_package_names("RUN apt-get install -y libsqlite3-0\n") == {"libsqlite3-0"}
    assert _apt_package_names("RUN apt-get install -y \"libsqlite3-0\" 'curl'\n") == {
        "libsqlite3-0",
        "curl",
    }


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


def test_python_does_not_surface_sqlites_session_extension() -> None:
    """CVE-2026-119379 is in the session extension, which CPython never wraps.

    The acceptance rests on the vulnerable entry point having no Python caller
    at all. CPython's ``sqlite3`` exports no session, changeset, patchset or
    rebaser API and a ``Connection`` carries no session attribute, so no Python
    code -- first-party or dependency -- can construct, apply or rebase a
    changeset.

    This deliberately does **not** assert that ``SQLITE_ENABLE_SESSION`` is
    absent from ``pragma compile_options``. That is a property of whichever
    libsqlite3 the interpreter happens to link, so it differs between the
    shipped image and a developer's host, and an acceptance resting on it
    would be checking the wrong machine. The missing Python binding holds
    everywhere.
    """
    import sqlite3

    exported = [
        name for name in dir(sqlite3) if not name.startswith("_") and _SESSION_API.search(name)
    ]
    assert not exported, (
        f"sqlite3 now exports a session-extension API ({exported}), so "
        "CVE-2026-119379's entry point is reachable from Python and its "
        "acceptance no longer holds"
    )

    connection = sqlite3.connect(":memory:")
    try:
        attrs = [name for name in dir(connection) if _SESSION_API.search(name)]
    finally:
        connection.close()
    assert not attrs, (
        f"sqlite3.Connection now carries a session API ({attrs}); "
        "CVE-2026-119379 is reachable and the ledger entry must be re-reviewed"
    )


def test_no_first_party_source_reaches_the_session_extension() -> None:
    """No Mercury source references a changeset, so nothing would apply one."""
    offenders = []
    for path in sorted(_SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in _SESSION_API.finditer(text):
            line = text[: match.start()].count("\n") + 1
            offenders.append(f"{path.relative_to(_REPO)}:{line}: {match.group(0)}")
    assert not offenders, (
        "first-party source reaches SQLite's session extension, so "
        "CVE-2026-119379 is no longer unreachable:\n  " + "\n  ".join(offenders)
    )


def test_the_session_detector_would_catch_a_real_caller() -> None:
    """Vacuity guard: the pattern matches the APIs it claims to look for."""
    for probe in (
        "sqlite3session_create(db, 'main', &session)",
        "sqlite3changeset_apply(db, n, blob, NULL, conflict, NULL)",
        "applied = conn.changeset()",
        "sqlite3rebaser_create(&rebaser)",
    ):
        assert _SESSION_API.search(probe), f"detector missed {probe!r}"
    # And does not fire on ordinary SQLite vocabulary that merely looks close,
    # so a passing run above means the absence is real and not an over-narrow
    # pattern that would miss a caller.
    for benign in ("conn.total_changes", "cursor.rowcount", "set_change_tracker"):
        assert not _SESSION_API.search(benign), f"detector over-matched {benign!r}"
