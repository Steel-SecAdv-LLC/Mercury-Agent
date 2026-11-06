# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The conditions behind the accepted zlib1g CVEs stay true.

``.trivyignore`` accepts CVE-2026-76844 (heap buffer overflow in zlib's
``gz_write()``) and CVE-2026-85091 (heap buffer overflow in ``gz_vacate()``),
both in zlib's ``gz*`` stream API, because no fixed version exists for the
suite the image is built from and because the one in-process binder of that
API is never invoked. A ledger entry states those facts once; this module
re-checks the ones that are checkable offline on every run, so the acceptances
fail loudly the day the ``gz*`` layer becomes reachable instead of the entries
quietly covering a path that has.

**The binder that exists, stated because an earlier version of this module
implied there was none.** A symbol audit of every native library in
site-packages found exactly one importer of ``gz*`` symbols:
opencv-python-headless's ``cv2`` extension, which links the unmangled system
``libz.so.1`` (the wheel vendors no libz) and imports
``gzopen``/``gzclose``/``gzeof``/``gzgets``/``gzputs``/``gzrewind`` for its
FileStorage persistence layer, entered only when a caller hands it a
gz-suffixed filename. Mercury contains no FileStorage use and no gz-suffixed
persistence path, so the accurate claim is binder-present, caller-absent --
and both halves are pinned below: the audit fails on any *new* gz importer,
and a source scan fails on any first-party FileStorage/gz use.

The vulnerability is in zlib's ``gz*`` FILE-based streaming API (``gzwrite.c``):
after an underlying ``write()`` fails, ``gz_write()`` returns without resetting
``strm.next_in``, so a later ``gz*`` write derives a position from the stale
pointer and writes past a heap allocation. Mercury is a Python application, and
CPython's ``zlib`` extension does not surface that API at all -- it exposes the
deflate-stream and checksum entry points only. Python's ``gzip`` module builds
the gzip container itself over ``zlib.compressobj``, so the ``gz*`` layer is
never entered.

Four conditions are therefore asserted:

* the ``zlib`` extension exposes no ``gz*`` (and no ``crc32_combine*``) symbol,
  which is what makes the path unreachable from pure Python;
* no first-party source in ``src/``, ``rust_crypto/`` or ``scripts/`` calls a
  ``gz*`` entry point or opens ``libz`` through ``ctypes``/``cffi``;
* no native library in site-packages imports a ``gz*`` symbol except cv2, the
  audited and pinned binder -- a new dependency that binds the layer fails
  here rather than extending the acceptance silently; and
* no first-party source invokes cv2's FileStorage or names a gz-suffixed
  persistence file, which is the only route into cv2's ``gz*`` use.

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
#: Both accepted zlib findings live in the same ``gz*`` stream API, so one
#: set of reachability conditions covers both. CVE-2026-76844 is the
#: ``gz_write()`` stale-``next_in`` overflow; CVE-2026-85091 is a distinct
#: overflow in ``gz_vacate()``, published 2026-11-06.
_ACCEPTED = ("CVE-2026-76844", "CVE-2026-85091")
#: Rendered into failure messages, where a bare tuple reads badly.
_ACCEPTED_TEXT = " and ".join(_ACCEPTED)

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


def test_ledger_carries_both_zlib_acceptances_with_an_expiry() -> None:
    """The acceptance exists and is dated, so the gate re-opens on expiry."""
    entries = _ledger_entries()
    for cve in _ACCEPTED:
        assert cve in entries, (
            f"{cve} is not an active ledger entry, so this module is "
            "asserting conditions for an acceptance that is not in force. "
            "Either the entry was removed because the finding was remediated, "
            "or the ledger drifted; the two call for opposite responses and "
            "the record does not say which happened"
        )
        raw = entries[cve]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw), (
            f"{cve} carries no exp: date ({raw!r}); an undated acceptance "
            "is a permanent waiver, which this ledger does not allow"
        )
        # A real calendar date, not merely a well-shaped one: "2027-02-30" would
        # satisfy the pattern above and silently never expire.
        try:
            expiry = date.fromisoformat(raw)
        except ValueError as exc:  # pragma: no cover - guards a typo
            raise AssertionError(
                f"{cve} carries {raw!r}, which is not a real calendar date, "
                f"so the acceptance would never expire: {exc}"
            ) from exc
        # Trivy fails the container gate once an entry expires. Asserting it here
        # too surfaces the re-review in the fast test lane rather than only in the
        # image-scan lane, which runs later and costs a container build.
        assert expiry >= date.today(), (
            f"the {cve} acceptance expired on {expiry}. The ledger's "
            "90-day cadence has lapsed, so two facts the acceptance rests on "
            "are now unverified: Debian's trixie fix status, and whether "
            "zlib1g is still irreducible. Both were last confirmed on the "
            "date the entry was written"
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
        f"reachable from Python and the acceptance of {_ACCEPTED_TEXT} no longer holds"
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
        f"vulnerable layer is reachable through ordinary use and {_ACCEPTED_TEXT} "
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
        f"FFI, so {_ACCEPTED_TEXT} is no longer unreachable and the ledger entries "
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


#: The zlib ``gz*`` FILE API, as upstream exports it. Matching against this
#: exact list keeps the native audit precise: an unrelated symbol that merely
#: starts with "gz" in some other library is not evidence about zlib.
_ZLIB_GZ_API = frozenset(
    {
        "gzopen",
        "gzopen64",
        "gzdopen",
        "gzbuffer",
        "gzsetparams",
        "gzread",
        "gzfread",
        "gzwrite",
        "gzfwrite",
        "gzprintf",
        "gzvprintf",
        "gzputs",
        "gzgets",
        "gzputc",
        "gzgetc",
        "gzgetc_",
        "gzungetc",
        "gzflush",
        "gzseek",
        "gzseek64",
        "gzrewind",
        "gztell",
        "gztell64",
        "gzoffset",
        "gzoffset64",
        "gzeof",
        "gzdirect",
        "gzclose",
        "gzclose_r",
        "gzclose_w",
        "gzerror",
        "gzclearerr",
        "gzvacate",
    }
)

#: The audited, accepted binder of the gz layer. Anything else that imports a
#: gz symbol is a change to the acceptance's premise and must fail the suite.
_PERMITTED_GZ_IMPORTERS = re.compile(r"(^|/)cv2[^/]*\.so(\.|$)|(^|/)cv2/")


def _site_native_libraries() -> list[Path]:
    import site
    import sysconfig

    roots: set[Path] = set()
    for raw in [
        *site.getsitepackages(),
        sysconfig.get_paths().get("purelib", ""),
        sysconfig.get_paths().get("platlib", ""),
    ]:
        if raw:
            root = Path(raw)
            if root.is_dir():
                roots.add(root)
    found: set[Path] = set()
    for root in roots:
        found.update(x for x in root.rglob("*.so*") if x.is_file())
    return sorted(found)


def _undefined_gz_imports(libraries: list[Path]) -> dict[str, list[str]]:
    """``{library: [gz symbols it imports]}`` via ``nm``, batched for speed."""
    import subprocess

    importers: dict[str, list[str]] = {}
    batch = 150
    for start in range(0, len(libraries), batch):
        chunk = [str(x) for x in libraries[start : start + batch]]
        # nm exits nonzero for any file without a dynamic symbol table
        # (static archives, stripped stubs); those contribute no imports and
        # must not abort the batch, so the exit code is deliberately ignored.
        result = subprocess.run(
            ["nm", "-D", "--undefined-only", *chunk],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        current = chunk[0] if len(chunk) == 1 else ""
        for line in result.stdout.splitlines():
            if line.endswith(":") and not line.startswith(" "):
                current = line[:-1]
                continue
            parts = line.split()
            if parts and parts[-1] in _ZLIB_GZ_API and current:
                importers.setdefault(current, []).append(parts[-1])
    return {k: sorted(set(v)) for k, v in importers.items()}


def test_no_native_library_imports_the_gz_layer_except_the_audited_binder() -> None:
    """A new dependency binding zlib's gz API must fail here, not pass silently.

    cv2 is the one audited importer (FileStorage persistence; see the module
    docstring and the ledger entry). Its presence also serves as the vacuity
    guard for the scanner itself: when cv2 is installed, the audit must FIND
    its gz imports, or the scan is reading nothing and a pass would be
    meaningless.
    """
    import shutil

    if shutil.which("nm") is None:  # pragma: no cover - CI always has binutils
        import pytest

        pytest.skip("binutils' nm is unavailable, so dynamic symbols cannot be read")

    libraries = _site_native_libraries()
    assert libraries, "no native libraries found; the scan is looking nowhere"
    importers = _undefined_gz_imports(libraries)

    unexpected = {
        lib: syms for lib, syms in importers.items() if not _PERMITTED_GZ_IMPORTERS.search(lib)
    }
    assert not unexpected, (
        "native libraries outside the audited cv2 binder now import zlib's "
        f"gz* API, so the {_ACCEPTED_TEXT} acceptances' premise no longer "
        f"holds: {unexpected}"
    )

    cv2_installed = any(
        "/cv2/" in str(lib) or str(lib).endswith("cv2.abi3.so") for lib in libraries
    )
    if cv2_installed:
        assert any(_PERMITTED_GZ_IMPORTERS.search(lib) for lib in importers), (
            "cv2 is installed but the audit found none of its known gz "
            "imports -- the scanner is not reading symbols, so its pass "
            "above is vacuous"
        )


def test_no_first_party_source_reaches_cv2s_gz_route() -> None:
    """The only route into cv2's gz use is FileStorage on a gz-suffixed path.

    Mercury uses cv2 for image operations only. If FileStorage or a
    gz-suffixed persistence filename ever appears in first-party source, the
    binder acquires a caller and the ledger entries must be re-reviewed
    rather than this scan widened.
    """
    pattern = re.compile(r"FileStorage|\.(?:xml|ya?ml|json)\.gz\b", re.IGNORECASE)
    offenders: list[str] = []
    for root in _SOURCE_ROOTS:
        base = _REPO / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in _SOURCE_SUFFIXES or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for match in pattern.finditer(text):
                line = text[: match.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(_REPO)}:{line}: {match.group(0)}")
    assert not offenders, (
        "first-party source now reaches cv2's FileStorage/gz persistence "
        "route, the one in-process path into zlib's gz layer:\n  " + "\n  ".join(offenders)
    )
