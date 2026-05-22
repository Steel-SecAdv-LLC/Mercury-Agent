"""Dispatch ``python -m tools <subcommand>`` to the appropriate tool.

Currently only ``lyapunov_validator`` is wired; the dispatcher exists so
``configs/lyapunov_canonical.yaml`` can document the canonical CLI as
``python -m tools.lyapunov_validator <config>`` (which Python's import
machinery resolves through this package).
"""

from __future__ import annotations

import sys


def _main() -> int:
    print(
        "usage: python -m tools.lyapunov_validator <config>\n"
        "       (no aggregate `python -m tools` entry-point; invoke a "
        "specific tool module.)",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":  # pragma: no cover - package entry-point
    raise SystemExit(_main())
