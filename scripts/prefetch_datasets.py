# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Prefetch real datasets into the local cache for offline (air-gapped) operation.

Run this while online to prime ``MERCURY_DATA_DIR`` (default ``./data``);
afterwards every cached dataset loads with ``MERCURY_OFFLINE=1`` set and no
network access. Fail-closed: any dataset that cannot be fetched is reported
and the script exits non-zero — it never records a partial fetch as success.

Usage::

    python scripts/prefetch_datasets.py --adbench cardio thyroid breastw WBC Pima
    python scripts/prefetch_datasets.py --adbench-all
    MERCURY_DATA_DIR=/srv/mercury/data python scripts/prefetch_datasets.py --adbench cardio
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Mirror scripts/run_api.py: runnable from a fresh checkout without an
# editable install.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adbench",
        nargs="*",
        default=[],
        metavar="NAME",
        help="ADBench dataset names to prefetch (e.g. cardio thyroid breastw WBC Pima)",
    )
    parser.add_argument(
        "--adbench-all",
        action="store_true",
        help="Prefetch the full 47-dataset ADBench Classical catalog",
    )
    args = parser.parse_args()

    from omni_mercury_engine.datasets.adbench import ADBenchLoader
    from omni_mercury_engine.datasets.base import DatasetConfig
    from omni_mercury_engine.datasets.exceptions import offline_mode_active

    if offline_mode_active():
        print(
            "MERCURY_OFFLINE is set — prefetching requires network access. "
            "Unset it, prime the cache, then re-enable offline mode."
        )
        return 2

    names = list(args.adbench)
    if args.adbench_all:
        names = ADBenchLoader.list_datasets()
    if not names:
        parser.print_help()
        return 2

    failures: list[str] = []
    cache_root = None
    for name in names:
        loader = ADBenchLoader(DatasetConfig(name="adbench", preprocessing={"dataset": name}))
        cache_root = loader.data_path
        try:
            loader.download()
            print(f"  cached  {name:<18} -> {loader.data_path / loader.npz_filename}")
        except Exception as exc:
            failures.append(name)
            print(f"  FAILED  {name:<18} ({type(exc).__name__}: {exc})")

    print("-" * 72)
    print(f"cache root: {cache_root}")
    print(f"{len(names) - len(failures)}/{len(names)} datasets cached")
    if failures:
        print(f"FAILED: {failures} — offline runs needing these will fail closed.")
        return 1
    print("Cache primed. Offline runs: export MERCURY_OFFLINE=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
