"""Friendly help message for ``python -m tools``.

The :mod:`tools` package intentionally does NOT implement an aggregate
``python -m tools <subcommand>`` dispatcher.  Every operator tool lives
in its own submodule and exposes its own ``if __name__ == "__main__"``
block, so the canonical invocation pattern is::

    python -m tools.lyapunov_validator <config>

which Python's import machinery routes directly to
``tools/lyapunov_validator.py`` -- it does *not* pass through this
``__main__`` module.  This file only runs when a user types
``python -m tools`` with no submodule, in which case we print the
usage above and exit with code 2 so the wrong invocation is loud.

If a real subcommand router is ever wanted (``tools list``,
``tools verify``, ...), this module is the right place to add it; the
docstring above will need to be updated in lock-step.
"""

from __future__ import annotations

import sys


def _main() -> int:
    print(
        "usage: python -m tools.<submodule> [args...]\n"
        "       e.g. python -m tools.lyapunov_validator configs/lyapunov_canonical.yaml\n"
        "\n"
        "       (`python -m tools` has no aggregate dispatcher; this entry "
        "       point only prints usage and exits 2.)",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":  # pragma: no cover - package entry-point
    raise SystemExit(_main())
