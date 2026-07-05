#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""One-command rollback of a staged retrain model.

Swaps the staged registry's ``active`` model back to the ``previous`` one in a
single audited, atomic operation -- the closed loop's break-glass control when a
staged candidate misbehaves. Never touches production directly; it restores the
prior *staged* pointer, which an operator then re-promotes (or not).

Usage::

    python scripts/mercury_retrain_rollback.py --staging-dir artifacts/closed_loop/staging
    python scripts/mercury_retrain_rollback.py --staging-dir <dir> --status   # show pointers only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from omni_mercury_engine.intel.feedback_loop.rollback import ModelRegistry


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-dir",
        required=True,
        help="the staged model registry directory (contains registry.json)",
    )
    parser.add_argument(
        "--status", action="store_true", help="print active/previous pointers and exit"
    )
    args = parser.parse_args(argv)

    registry = ModelRegistry(args.staging_dir)
    active = registry.active()
    previous = registry.previous()

    if args.status:
        print(
            json.dumps(
                {
                    "active": active.as_dict() if active else None,
                    "previous": previous.as_dict() if previous else None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    result = registry.rollback()
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.rolled_back else 1


if __name__ == "__main__":
    raise SystemExit(main())
