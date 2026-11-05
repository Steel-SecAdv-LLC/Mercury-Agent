# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The conditions behind the accepted zlib1g CVE stay true.

``.trivyignore`` accepts CVE-2026-76844 (heap buffer overflow in zlib's
``gz_write()``) because the vulnerable code path has no caller in the shipped
image, and because no fixed version exists for the suite the image is built
from. A ledger entry states those facts once; this test re-checks the ones that
are checkable offline on every run, so the acceptance fails loudly the day a
dependency starts calling zlib's ``gz*`` layer instead of the entry quietly
covering a path that has become reachable.

The vulnerability is in zlib's ``gz*`` FILE-based streaming API (``gzwrite.c``):
after an underlying ``write()`` fails, ``gz_write()`` returns without resetting
``strm.next_in``, so a later ``gz*`` write derives a position from the stale
pointer and writes past a heap allocation. Mercury is a Python application, and
CPython's ``zlib`` extension does not surface that API at all -- it exposes the
deflate-stream and checksum entry points only. Python's ``gzip`` module builds
the gzip container itself over ``zlib.compressobj``, so the ``gz*`` layer is
never entered.

Two conditions are therefore asserted:

* the ``zlib`` extension exposes no ``gz*`` (and no ``crc32_combine*``) symbol,
  which is what makes the path unreachable from Python; and
* no first-party source in ``src/``, ``rust_crypto/`` or ``scripts/`` calls a
  ``gz*`` entry point or opens ``libz`` through ``ctypes``/``cffi``, which is
  the only way first-party code could reach it despite the above.

What this test deliberately does **not** assert is the absence of an upstream
fix: that is a fact about Debian's archive at a point in time, it is cited in
the ledger with its source, and the 90-day expiry is the control that forces it
to be re-checked. Pinning it here would mean either a network call in the
security lane or a hardcoded claim that rots.

Offline, dependency-free, and import-free of the engine so it runs in the
torch-free security lane.
"""

from __future__ import annotations

import re
import zlib
from datetime import date
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_LEDGER = _REPO / ".trivyignore"

#: The accepted finding this module guards.
_ACCEPTED = "CVE-2026-76844"

#: zlib's FILE-based streaming API, whose ``gz_write()`` carries the overflow.
#: ``gz_vacate`` is CVE-2026-85091's function, adjacent and listed in the
#: ledger's note; a caller appearing for it is the same class of event.
#: Both spellings matter: ``gzwrite``/``gzopen`` are the public entry points,
#: while ``gz_write``/``gz_vacate`` are the internal functions the two CVEs
#: actually name. Matching only the public form would miss a vendored copy.
_GZ_CALL = re.compile(
    r"\bgz_?(?:open|dopen|write|printf|puts|close|close_w|flush|setparams"
    r"|buffer|vacate|read|error|eof|rewind|seek|tell)\s*\(",
)

#: A ctypes/cffi handle on the shared library, which would bypass the Python
#: API surface entirely and reach ``gz_write`` directly.
#: Two shapes, because the library is named differently in each. ``CDLL`` and
#: ``dlopen`` take a filename (``libz.so.1``); ``find_library`` takes the bare
#: linker name (``z``), so a pattern looking only for "libz" would miss it.
_LIBZ_FFI = re.compile(
    r"(?:CDLL|cdll\.LoadLibrary|dlopen)\s*\(\s*['\"][^'\"]*\blibz\b"
    r"|find_library\s*\(\s*['\"]z['\"]",
    re.IGNORECASE,
)

#: First-party trees that ship into or build the image.
_SOURCE_ROOTS = ("src", "rust_crypto", "scripts")
_SOURCE_SUFFIXES = {".py", ".rs", ".c", ".h", ".pyx", ".sh"}


def _ledger_entries() -> dict[str, str]:
    """``{CVE: expiry}`` for every active ledger line."""
    entries: dict[str, str] = {}
    for line in _LEDGER.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*(CVE-\d{4}-\d+)\s+exp:(\d{4}-\d{2}-\d{2})", line)
        if match:
            entries[match.group(1)] = match.group(2)
    return entries


def test_ledger_carries_the_zlib_acceptance_with_an_expiry() -> None:
    """The acceptance exists and is dated, so the gate re-opens on expiry."""
    entries = _ledger_entries()
    assert _ACCEPTED in entries, (
        f"{_ACCEPTED} is not an active ledger entry; if the finding was "
        "remediated rather than accepted, delete this module with the entry"
    )
    raw = entries[_ACCEPTED]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw), (
        f"{_ACCEPTED} carries no exp: date ({raw!r}); an undated acceptance "
        "is a permanent waiver, which this ledger does not allow"
    )
    # A real calendar date, not merely a well-shaped one: "2027-02-30" would
    # satisfy the pattern above and silently never expire.
    try:
        expiry = date.fromisoformat(raw)
    except ValueError as exc:  # pragma: no cover - guards a typo, not a path
        raise AssertionError(
            f"{_ACCEPTED} carries {raw!r}, which is not a real calendar date, "
            f"so the acceptance would never expire: {exc}"
        ) from exc
    # Trivy fails the container gate once an entry expires. Asserting it here
    # too surfaces the re-review in the fast test lane rather than only in the
    # image-scan lane, which runs later and costs a container build.
    assert expiry >= date.today(), (
        f"the {_ACCEPTED} acceptance expired on {expiry}. Re-review it: "
        "confirm the Debian trixie fix status, re-measure whether the "
        "package is still irreducible, and either renew the entry with "
        "fresh evidence or remove it together with this module"
    )


def test_python_does_not_surface_zlibs_gz_file_api() -> None:
    """The reachability premise: CPython exposes no ``gz*`` entry point.

    This is what makes ``gz_write()`` unreachable from Python code, and it is
    the whole basis of the acceptance. A future CPython that surfaced the
    ``gz*`` API (or a vendored zlib binding that did) would make the path
    reachable and must re-open the review.
    """
    surfaced = sorted(name for name in dir(zlib) if name.lower().startswith("gz"))
    assert surfaced == [], (
        f"the zlib extension now exposes {surfaced}, so zlib's gz* FILE API is "
        f"reachable from Python and {_ACCEPTED}'s acceptance no longer holds"
    )
    combine = sorted(name for name in dir(zlib) if "combine" in name.lower())
    assert combine == [], (
        f"the zlib extension now exposes {combine}; crc32_combine64 is "
        "CVE-2026-27171's function, noted as adjacent in the ledger, and its "
        "reachability must be re-reviewed"
    )


def test_gzip_builds_its_container_over_compressobj_not_the_gz_api() -> None:
    """The second half of the premise: the stdlib route avoids ``gz*`` too.

    ``gzip`` is the obvious way a Python application touches gzip framing. It
    implements the container itself on top of the deflate-stream API, which is
    why using ``gzip`` does not enter the vulnerable layer.
    """
    import inspect

    source = inspect.getsource(__import__("gzip"))
    assert "compressobj" in source, "gzip no longer uses zlib.compressobj; re-derive the premise"
    assert not _GZ_CALL.search(source), (
        "the stdlib gzip module now calls zlib's gz* API directly, so the "
        f"vulnerable layer is reachable through ordinary use and {_ACCEPTED} "
        "must be re-reviewed"
    )


def test_no_first_party_source_reaches_zlibs_gz_layer() -> None:
    """No shipped or build-time source calls ``gz*`` or opens ``libz`` via FFI."""
    offenders: list[str] = []
    for root in _SOURCE_ROOTS:
        base = _REPO / root
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix not in _SOURCE_SUFFIXES or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if _GZ_CALL.search(text) or _LIBZ_FFI.search(text):
                offenders.append(str(path.relative_to(_REPO)))
    assert offenders == [], (
        "first-party source now reaches zlib's gz* layer or loads libz through "
        f"FFI, so {_ACCEPTED} is no longer unreachable and its ledger entry "
        f"must be re-reviewed: {sorted(offenders)}"
    )


def test_the_detectors_would_catch_a_real_caller() -> None:
    """The guards above must be able to fail, not merely pass vacuously.

    A regex that matched nothing would make both checks above green forever.
    These are the shapes a real caller takes.
    """
    assert _GZ_CALL.search("gzwrite(f, buf, len);")
    assert _GZ_CALL.search("out = gzopen(path, 'wb')")
    assert _GZ_CALL.search("gz_vacate (state)")
    assert _LIBZ_FFI.search('lib = ctypes.CDLL("libz.so.1")')
    assert _LIBZ_FFI.search("ctypes.util.find_library('z')")
    assert _LIBZ_FFI.search('dlopen("/lib/x86_64-linux-gnu/libz.so.1")')
    # And must not fire on ordinary, safe zlib use.
    assert not _GZ_CALL.search("zlib.compressobj(9)")
    assert not _GZ_CALL.search("import gzip; gzip.open(p)")
    assert not _LIBZ_FFI.search('ctypes.CDLL("libama_cryptography.so")')
