"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""Print measured codebase-scale numbers for the Mercury Agent README.

Reports:
  * source files and SLOC under ``src/omni_mercury_engine``,
  * top-level subpackages,
  * test files and test SLOC,
  * counts of ``import torch`` / ``from torch`` modules and
    ``class …(nn.Module)`` definitions.

These numbers back the "Codebase Scale" callout in ``README.md``.
The script is read-only and has no third-party dependencies, so it can
run in any CI environment without installing the Mercury Agent package.

Usage:
    python scripts/measure_codebase_scale.py
    python scripts/measure_codebase_scale.py --json
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "omni_mercury_engine"
TESTS = ROOT / "tests"

TORCH_IMPORT_RE = re.compile(r"^\s*(?:from\s+torch\b|import\s+torch\b)", re.MULTILINE)
NN_MODULE_RE = re.compile(r"^\s*class\s+\w+\([^)]*\bnn\.Module\b[^)]*\):", re.MULTILINE)


def _count_lines(paths: list[Path]) -> int:
    total = 0
    for p in paths:
        try:
            with p.open("r", encoding="utf-8", errors="replace") as fh:
                for _ in fh:
                    total += 1
        except OSError:
            continue
    return total


def measure() -> dict[str, int | list[str]]:
    src_files = sorted(SRC.rglob("*.py")) if SRC.is_dir() else []
    test_files = sorted(TESTS.rglob("test_*.py")) if TESTS.is_dir() else []
    subpackages = sorted(p.name for p in SRC.iterdir() if p.is_dir()) if SRC.is_dir() else []

    torch_files = 0
    nn_module_classes = 0
    for p in src_files:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if TORCH_IMPORT_RE.search(text):
            torch_files += 1
        nn_module_classes += len(NN_MODULE_RE.findall(text))

    return {
        "src_files": len(src_files),
        "src_loc": _count_lines(src_files),
        "subpackages": subpackages,
        "subpackage_count": len(subpackages),
        "test_files": len(test_files),
        "test_loc": _count_lines(test_files),
        "torch_importing_files": torch_files,
        "nn_module_subclasses": nn_module_classes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = parser.parse_args(argv)

    stats = measure()
    if args.json:
        json.dump(stats, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    print("Mercury Agent — measured codebase scale")
    print("---------------------------------------")
    print(f"src files:                {stats['src_files']}")
    print(f"src LOC:                  {stats['src_loc']:,}")
    print(f"subpackages:              {stats['subpackage_count']}")
    print(f"test files:               {stats['test_files']}")
    print(f"test LOC:                 {stats['test_loc']:,}")
    print(f"torch-importing files:    {stats['torch_importing_files']}")
    print(f"nn.Module subclasses:     {stats['nn_module_subclasses']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
