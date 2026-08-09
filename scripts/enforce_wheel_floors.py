#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Enforce supply-chain version floors against what a scanner actually reads.

Why this exists
---------------

The Dockerfile already upgrades ``setuptools`` and ``msgpack`` past their CVE
floors and asserts the result by importing them. That assert passed while the
blocking container scan kept reporting ``setuptools 70.3.0``
(CVE-2025-47273) and ``msgpack 1.1.2`` (GHSA-6v7p-g79w-8964) in the shipped
image.

Both statements were true at once, because they measure different things:

* ``import setuptools; setuptools.__version__`` reports the version Python
  *resolves* -- the winner of the ``sys.path`` search.
* Trivy's ``python-pkg`` analyzer reads ``*.dist-info/METADATA`` **files on
  disk**. It does not import anything, and it does not care which copy would
  win.

So a stale ``setuptools-70.3.0.dist-info`` left behind anywhere on the image --
in a second site-packages, a vendored tree, or an interrupted uninstall -- is
invisible to the import assert and fully visible to the gate. Upgrading harder
cannot fix that; the leftover metadata has to be found and removed, and its
absence has to be the thing asserted.

This script does exactly that: it walks every ``*.dist-info`` under the given
roots, and for each floored package removes directories whose version is below
the floor **only when a compliant copy of the same package also exists**. A
below-floor version with no compliant sibling is a genuine failure -- the
package really is too old -- and is reported rather than deleted, because
deleting it would hide the problem from the scanner instead of fixing it.

Usage::

    python scripts/enforce_wheel_floors.py /opt/venv /usr/local
    python scripts/enforce_wheel_floors.py --check /opt/venv   # report only

Exit code is non-zero when any below-floor distribution remains.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

#: package name (normalised, lower-case) -> minimum acceptable version.
#: Each entry names the advisory it closes so the floor is never mistaken for
#: an arbitrary preference.
FLOORS: dict[str, tuple[tuple[int, ...], str]] = {
    "setuptools": ((83, 0, 0), "CVE-2025-47273 (fixed 78.1.1), CVE-2026-59890 (fixed 83.0.0)"),
    "msgpack": ((1, 2, 1), "GHSA-6v7p-g79w-8964"),
}

_DIST_INFO = re.compile(r"^(?P<name>.+?)-(?P<version>\d[^-]*)\.dist-info$")


def _parse_version(raw: str) -> tuple[int, ...]:
    """Parse a PEP 440-ish release tuple, tolerating suffixes.

    Args:
        raw: Version string taken from a ``.dist-info`` directory name.

    Returns:
        The numeric release components, e.g. ``(83, 0, 0)``. Non-numeric
        trailing segments (``rc1``, ``post2``) stop the parse, which is the
        conservative reading for a floor comparison.
    """
    parts: list[int] = []
    for chunk in raw.split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _normalise(name: str) -> str:
    """Return the PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def scan(roots: list[Path]) -> dict[str, list[tuple[Path, tuple[int, ...]]]]:
    """Collect every ``.dist-info`` for a floored package under ``roots``.

    Args:
        roots: Filesystem roots to walk.

    Returns:
        Mapping of normalised package name to ``(path, version)`` pairs.
    """
    found: dict[str, list[tuple[Path, tuple[int, ...]]]] = {name: [] for name in FLOORS}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.dist-info"):
            match = _DIST_INFO.match(path.name)
            if not match:
                continue
            name = _normalise(match.group("name"))
            if name in found:
                found[name].append((path, _parse_version(match.group("version"))))
    return found


def main() -> int:
    """Remove stale below-floor metadata and fail if any remains.

    Returns:
        Process exit code: 0 when every floored package is compliant on disk.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="filesystem roots to sweep")
    parser.add_argument("--check", action="store_true", help="report only; never delete anything")
    args = parser.parse_args()

    found = scan(args.roots)
    violations: list[str] = []

    for name, (floor, advisory) in FLOORS.items():
        entries = found.get(name, [])
        if not entries:
            print(f"  {name}: not present under {', '.join(str(r) for r in args.roots)}")
            continue

        compliant = [(p, v) for p, v in entries if v >= floor]
        stale = [(p, v) for p, v in entries if v < floor]

        for path, version in sorted(compliant):
            print(f"  {name} {'.'.join(map(str, version))}: OK  {path}")

        for path, version in sorted(stale):
            shown = ".".join(map(str, version))
            if not compliant:
                # Nothing compliant to fall back on: the package genuinely is
                # too old. Deleting the metadata would only blind the scanner.
                violations.append(
                    f"{name} {shown} is below the {'.'.join(map(str, floor))} floor "
                    f"({advisory}) and no compliant copy exists: {path}"
                )
                continue
            if args.check:
                violations.append(f"stale metadata would be removed: {path}")
                continue
            shutil.rmtree(path)
            print(f"  {name} {shown}: REMOVED stale metadata (compliant copy present)  {path}")

    if violations:
        print("\nFLOOR VIOLATIONS:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1

    print("\nAll floored packages are compliant on disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
