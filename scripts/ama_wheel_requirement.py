#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write a hash-locked requirement for the AMA wheels in a verified directory.

The prebuilt AMA archive is checked against its release ``SHA256SUMS`` before
it is unpacked (``scripts/fetch_ama_prebuilt.sh``), or was built moments ago in
the same job. Installing it by name with ``--no-index --find-links`` leaves pip
to pick the wheel whose tag matches the interpreter, which is right, but pins
nothing pip can check. This writes one requirement --
``ama-cryptography==<version>`` with a ``--hash`` for every wheel in the
directory -- so ``pip install --require-hashes --no-index --find-links DIR -r
FILE`` keeps pip's tag selection and refuses any file that is not one of those
exact, verified wheels (OpenSSF Scorecard Pinned-Dependencies).

Usage::

    python scripts/ama_wheel_requirement.py VERSION WHEEL_DIR OUT_FILE

Exit status 1 when the directory holds no wheel for that version.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def requirement(version: str, wheel_dir: Path) -> str | None:
    """The requirement line, or ``None`` when no wheel of ``version`` is present."""
    wheels = sorted(wheel_dir.glob(f"ama_cryptography-{version}-*.whl"))
    if not wheels:
        return None
    hashes = " ".join(
        f"--hash=sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}" for wheel in wheels
    )
    return f"ama-cryptography=={version} {hashes}\n"


def main(argv: list[str]) -> int:
    """Write the requirement file; non-zero when there is nothing to install."""
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    version, wheel_dir, out = argv[0], Path(argv[1]), Path(argv[2])
    line = requirement(version, wheel_dir)
    if line is None:
        print(f"no ama_cryptography-{version} wheel in {wheel_dir}", file=sys.stderr)
        return 1
    out.write_text(line, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
